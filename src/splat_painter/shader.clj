(ns splat-painter.shader
  "The splat rasterizer as a GLSL fragment shader — the GPU twin of
  splat-painter.gaussian/rasterize. Same math: the symmetric 2×2 covariance's
  closed-form precision, the peak-normalized exp at the mean (pdf=0 there, so
  exp(-pdf)=1.0 without needing the reference's per-grid subtraction), and the
  additive color sum. Evaluated per fragment over every splat stored in an
  RGBA32F texture, two texels each.

  The body needs a dynamic for-loop + texelFetch, which glimmer-gl.shader's data
  IR can't express (its compile-stmt only handles :let/:set/:if), so the GLSL is
  written directly and compiled with glimmer-gl.gl/make-program. Uniform and
  attribute locations are cached once at build time.

  Splat texture layout (RGBA32F). Splats are TILED into a 2D texture so the count can exceed
  GL_MAX_TEXTURE_SIZE (16384 on Apple Silicon). Each splat uses THREE adjacent texels;
  TILE_W splats per texture row, so the texture is (3·TILE_W)×ceil(N/TILE_W):
    col = i % TILE_W, row = i / TILE_W
    texelFetch(u_splats, ivec2(3·col,   row),0) = (mean_x, mean_y, c00, c01)
    texelFetch(u_splats, ivec2(3·col+1, row),0) = (c11,     color_r, color_g, color_b)
    texelFetch(u_splats, ivec2(3·col+2, row),0) = (alpha,   0, 0, 0)
  where the covariance is symmetric [c00 c01; c01 c11] in (row, col) space and alpha is the
  per-splat paint alpha (brush strokes taper it toward the stroke tail)."
  (:require [glimmer-gl.gl :as gl]))

(def max-splats "shader splat ceiling + transform-feedback buffer capacity" 131072)
(def tile-w "splats per splat-texture row (must match the shader's TILE_W)" 4096)

(def ^:private vs-src
  "#version 330 core
in vec2 a_pos;                 // fullscreen quad, -1..1
void main(){
  gl_Position = vec4(a_pos, 0.0, 1.0);
}")

(def ^:private fs-src
  (str "#version 330 core
out vec4 frag;
uniform sampler2D u_splats;    // RGBA32F, tiled (2*TILE_W) x ceil(N/TILE_W)
uniform int  u_count;          // splats actually present (<= MAX_SPLATS)
uniform vec2 u_viewport;       // pane pixels (pw, ph)
uniform vec2 u_image;          // image pixels (iw, ih)
uniform vec3 u_bg;             // background color (weighted by remaining T)
uniform float u_opacity;        // per-splat alpha gain
uniform float u_hard_sharp;     // edge sharpness for the SMALLEST (detail) splats: >1 = flat core + hard edge
uniform float u_hard_soft;      // edge sharpness for the LARGEST splats: ~1 = soft round gaussian
uniform float u_sig_min;        // smallest splat stdev in the field (det^0.25)
uniform float u_sig_max;        // largest splat stdev in the field
const int MAX_SPLATS = " max-splats ";
const int TILE_W = " tile-w ";

void main(){
  float pw = u_viewport.x, ph = u_viewport.y;
  float iw = u_image.x,    ih = u_image.y;
  // fit the image into the pane (contain), centered
  float scale = min(pw / iw, ph / ih);
  float dw = iw * scale, dh = ih * scale;
  vec2 fc = gl_FragCoord.xy - vec2((pw - dw) * 0.5, (ph - dh) * 0.5);
  if (fc.x < 0.0 || fc.x > dw || fc.y < 0.0 || fc.y > dh) {
    frag = vec4(u_bg, 1.0);
    return;
  }
  vec2 imgpx = fc / scale;                 // [0,iw]x[0,ih], y bottom-up
  float x = ih - imgpx.y;                  // row (top-down) — splat x-axis
  float y = imgpx.x;                       // col (left-right) — splat y-axis
  // front-to-back over-compositing (the 2DGS render equation):
  //   C = Σ c·α·T,  T *= (1-α),  final = C + T·u_bg
  float T = 1.0;
  vec3 acc = vec3(0.0);
  for (int i = 0; i < MAX_SPLATS; i++) {
    if (i >= u_count) break;
    int col = i % TILE_W, row = i / TILE_W;
    vec4 t0 = texelFetch(u_splats, ivec2(3 * col,     row), 0);
    vec4 t1 = texelFetch(u_splats, ivec2(3 * col + 1, row), 0);
    vec4 t2 = texelFetch(u_splats, ivec2(3 * col + 2, row), 0);
    float dx = x - t0.x, dy = y - t0.y;
    float c00 = t0.z, c01 = t0.w, c11 = t1.x;
    float det = max(c00 * c11 - c01 * c01, 1e-8);
    float p00 = c11 / det, p11 = c00 / det, cross = -2.0 * c01 / det;
    float pdf = 0.5 * (p00 * dx * dx + cross * dx * dy + p11 * dy * dy);
    // per-splat edge hardness scales PROGRESSIVELY with size: sig = det^0.25 = the base
    // stroke stdev (elongation cancels). Small detail strokes stay crisp (u_hard_sharp);
    // large strokes soften toward a round gaussian (u_hard_soft). smoothstep = easing.
    float sig = sqrt(sqrt(det));
    float ts  = clamp((sig - u_sig_min) / max(u_sig_max - u_sig_min, 1e-4), 0.0, 1.0);
    ts = ts * ts * (3.0 - 2.0 * ts);
    float hardness = mix(u_hard_sharp, u_hard_soft, ts);
    float a = t2.x * u_opacity * exp(-pow(pdf, hardness));  // t2.x = per-splat paint alpha (stroke taper)
    float wa = T * a;
    acc += wa * t1.yzw;
    T *= (1.0 - a);
  }
  frag = vec4(acc + T * u_bg, 1.0);
}
"))

;; --- texture-buffer render variant -------------------------------------------
;; Same math as fs-src, but the splats come from a samplerBuffer (a 1D texture view
;; over a buffer object) instead of a tiled 2D texture. A texture buffer has no
;; GL_MAX_TEXTURE_SIZE ceiling, so the stream is flat (2 texels per splat, no tiling)
;; — and it's exactly what the GPU generation pass writes via transform feedback, so
;; the generated buffer feeds straight in with no readback/repack.
(def ^:private fs-src-buf
  (str "#version 330 core
out vec4 frag;
uniform samplerBuffer u_splats;  // RGBA32F texture buffer, 2 texels per splat
uniform int  u_count;
uniform vec2 u_viewport;
uniform vec2 u_image;
uniform vec3 u_bg;
uniform float u_opacity;
uniform float u_hard_sharp;
uniform float u_hard_soft;
uniform float u_sig_min;
uniform float u_sig_max;
const int MAX_SPLATS = " max-splats ";

void main(){
  float pw = u_viewport.x, ph = u_viewport.y;
  float iw = u_image.x,    ih = u_image.y;
  float scale = min(pw / iw, ph / ih);
  float dw = iw * scale, dh = ih * scale;
  vec2 fc = gl_FragCoord.xy - vec2((pw - dw) * 0.5, (ph - dh) * 0.5);
  if (fc.x < 0.0 || fc.x > dw || fc.y < 0.0 || fc.y > dh) {
    frag = vec4(u_bg, 1.0);
    return;
  }
  vec2 imgpx = fc / scale;
  float x = ih - imgpx.y;
  float y = imgpx.x;
  float T = 1.0;
  vec3 acc = vec3(0.0);
  for (int i = 0; i < MAX_SPLATS; i++) {
    if (i >= u_count) break;
    vec4 t0 = texelFetch(u_splats, 3 * i);
    vec4 t1 = texelFetch(u_splats, 3 * i + 1);
    vec4 t2 = texelFetch(u_splats, 3 * i + 2);
    float dx = x - t0.x, dy = y - t0.y;
    float c00 = t0.z, c01 = t0.w, c11 = t1.x;
    float det = max(c00 * c11 - c01 * c01, 1e-8);
    float p00 = c11 / det, p11 = c00 / det, cross = -2.0 * c01 / det;
    float pdf = 0.5 * (p00 * dx * dx + cross * dx * dy + p11 * dy * dy);
    float sig = sqrt(sqrt(det));
    float ts  = clamp((sig - u_sig_min) / max(u_sig_max - u_sig_min, 1e-4), 0.0, 1.0);
    ts = ts * ts * (3.0 - 2.0 * ts);
    float hardness = mix(u_hard_sharp, u_hard_soft, ts);
    float a = t2.x * u_opacity * exp(-pow(pdf, hardness));
    float wa = T * a;
    acc += wa * t1.yzw;
    T *= (1.0 - a);
  }
  frag = vec4(acc + T * u_bg, 1.0);
}
"))

;; --- per-splat quad render (fixes the O(pixels × splats) hang) ----------------
;; The loop shaders above evaluate EVERY splat at EVERY pixel — 3.4e10 iterations at
;; the 48k slider max, which trips macOS's GPU watchdog. This variant is the standard
;; gaussian-splatting renderer instead: one quad per splat covering its ~3.5σ extent,
;; the fragment shader evaluates only THAT splat's gaussian, and hardware blending
;; does the over-compositing. Work = Σ quad areas (~2e8 fragments) regardless of count.
;;
;;   order: the buffer is finest-first (index 0 = topmost), so drawing BACK-TO-FRONT
;;   means reverse index order — splat = (u_count-1) - id. Premultiplied over-blend
;;   (ONE, ONE_MINUS_SRC_ALPHA) onto a bg-cleared target is then exactly the loop
;;   shader's front-to-back C = Σ c·α·T, final = C + T·bg.
;;
;;   geometry: attribute-less — 6 vertices per splat from gl_VertexID alone (the TF
;;   generation pass already proves attribute-less draws on this driver). Axis-aligned
;;   half-extents 3.5·(√c00, √c11) are the exact marginal stdevs of the ellipse, so
;;   the quad bounds the 3.5σ contour; at the softest hardness (1.0) the truncated
;;   tail is exp(-6.1)·opacity ≈ 0.2% ≈ half an 8-bit step — invisible.
(def ^:private vs-src-quad
  "#version 330 core
uniform samplerBuffer u_splats;  // RGBA32F, 2 texels per splat (finest-first)
uniform int   u_count;
uniform vec2  u_viewport;        // pane pixels (pw, ph)
uniform vec2  u_image;           // image pixels (iw, ih)
uniform float u_hard_sharp;
uniform float u_hard_soft;
uniform float u_sig_min;
uniform float u_sig_max;
flat out vec3  v_color;
flat out vec3  v_prec;           // p00, p11, cross
flat out float v_hard;
flat out float v_alpha;          // per-splat paint alpha (stroke taper)
out vec2 v_d;                    // image-space offset from the mean (rows, cols)

void main(){
  int splat  = (u_count - 1) - (gl_VertexID / 6);   // back-to-front paint order
  int corner = gl_VertexID - 6 * (gl_VertexID / 6);
  vec4 t0 = texelFetch(u_splats, 3 * splat);
  vec4 t1 = texelFetch(u_splats, 3 * splat + 1);
  v_alpha = texelFetch(u_splats, 3 * splat + 2).x;
  float c00 = t0.z, c01 = t0.w, c11 = t1.x;
  float det = max(c00 * c11 - c01 * c01, 1e-8);
  v_prec  = vec3(c11 / det, c00 / det, -2.0 * c01 / det);
  v_color = t1.yzw;
  float sig = sqrt(sqrt(det));
  float ts  = clamp((sig - u_sig_min) / max(u_sig_max - u_sig_min, 1e-4), 0.0, 1.0);
  ts = ts * ts * (3.0 - 2.0 * ts);
  v_hard = mix(u_hard_sharp, u_hard_soft, ts);
  // two triangles (0,1,2)(2,1,3) over corner ids 0..3 = (∓,∓)(±,∓)(∓,±)(±,±)
  int cid = corner == 0 ? 0 : (corner == 1 || corner == 4) ? 1
          : (corner == 2 || corner == 3) ? 2 : 3;
  vec2 s  = vec2((cid & 1) == 0 ? -1.0 : 1.0, (cid & 2) == 0 ? -1.0 : 1.0);
  vec2 he = 3.5 * sqrt(vec2(c00, c11));   // marginal stdevs = exact AABB of the ellipse
  v_d = s * he;
  vec2 ip = t0.xy + v_d;                  // image position (x=row top-down, y=col)
  // image px -> pane px (contain fit, centered; inverse of the loop shader's mapping)
  float scale = min(u_viewport.x / u_image.x, u_viewport.y / u_image.y);
  vec2 org = 0.5 * (u_viewport - u_image * scale);
  vec2 pane = vec2(ip.y * scale + org.x, (u_image.y - ip.x) * scale + org.y);
  gl_Position = vec4(pane / u_viewport * 2.0 - 1.0, 0.0, 1.0);
}")

(def ^:private fs-src-quad
  "#version 330 core
flat in vec3  v_color;
flat in vec3  v_prec;
flat in float v_hard;
flat in float v_alpha;
in vec2 v_d;
uniform float u_opacity;
out vec4 frag;
void main(){
  float pdf = 0.5 * (v_prec.x * v_d.x * v_d.x + v_prec.z * v_d.x * v_d.y + v_prec.y * v_d.y * v_d.y);
  float a = v_alpha * u_opacity * exp(-pow(pdf, v_hard));
  frag = vec4(v_color * a, a);   // premultiplied; blend (ONE, ONE_MINUS_SRC_ALPHA)
}")

(defn build-program-quad
  "Compile + link the per-splat quad renderer (needs a current GL context).
  Attribute-less: bind any VAO with no enabled attribs and draw 6·count GL_TRIANGLES
  with blending (ONE, ONE_MINUS_SRC_ALPHA) onto a target cleared to the background.
  Returns {:program :locs} or nil."
  []
  (when-let [prog (gl/make-program vs-src-quad fs-src-quad)]
    {:program prog
     :locs {:u_splats   (gl/gl-get-uniform-location prog "u_splats")
            :u_count    (gl/gl-get-uniform-location prog "u_count")
            :u_viewport (gl/gl-get-uniform-location prog "u_viewport")
            :u_image    (gl/gl-get-uniform-location prog "u_image")
            :u_opacity  (gl/gl-get-uniform-location prog "u_opacity")
            :u_hard_sharp (gl/gl-get-uniform-location prog "u_hard_sharp")
            :u_hard_soft  (gl/gl-get-uniform-location prog "u_hard_soft")
            :u_sig_min    (gl/gl-get-uniform-location prog "u_sig_min")
            :u_sig_max    (gl/gl-get-uniform-location prog "u_sig_max")}}))

(defn- render-uniform-locs [prog]
  {:u_splats   (gl/gl-get-uniform-location prog "u_splats")
   :u_count    (gl/gl-get-uniform-location prog "u_count")
   :u_viewport (gl/gl-get-uniform-location prog "u_viewport")
   :u_image    (gl/gl-get-uniform-location prog "u_image")
   :u_bg       (gl/gl-get-uniform-location prog "u_bg")
   :u_opacity  (gl/gl-get-uniform-location prog "u_opacity")
   :u_hard_sharp (gl/gl-get-uniform-location prog "u_hard_sharp")
   :u_hard_soft  (gl/gl-get-uniform-location prog "u_hard_soft")
   :u_sig_min    (gl/gl-get-uniform-location prog "u_sig_min")
   :u_sig_max    (gl/gl-get-uniform-location prog "u_sig_max")})

(defn build-program-buf
  "Compile + link the samplerBuffer render variant (needs a current GL context).
  Returns {:program :locs} or nil. Uses a fullscreen quad, so no a_pos here —
  it shares vs-src's attribute; callers reuse the same VAO/VBO."
  []
  (when-let [prog (gl/make-program vs-src fs-src-buf)]
    {:program prog
     :locs    (assoc (render-uniform-locs prog)
                     :a_pos (gl/gl-get-attrib-location prog "a_pos"))}))

(defn sources
  "Return {:vs-src :fs-src :fs-src-buf :vs-src-quad :fs-src-quad} — pure, no GL context
  (for headless inspection/tests)."
  []
  {:vs-src vs-src :fs-src fs-src :fs-src-buf fs-src-buf
   :vs-src-quad vs-src-quad :fs-src-quad fs-src-quad})

(defn pack-splats
  "Flatten a seq of splats into the RGBA32F texture payload (length 3*N*4): splat i
  is [mean_x mean_y c00 c01  c11 r g b  alpha 0 0 0]. Pure, no GL."
  [splats]
  (loop [out (transient [])
         s   splats]
    (if-not s
      (persistent! out)
      (let [{[mx my] :mean [c00 c01 _ c11] :cov [r g b] :color a :alpha} (first s)]
        (recur (-> out
                   (conj! mx) (conj! my) (conj! c00) (conj! c01)
                   (conj! c11) (conj! r) (conj! g) (conj! b)
                   (conj! (or a 1.0)) (conj! 0.0) (conj! 0.0) (conj! 0.0))
               (next s))))))

(defn build-program
  "Compile + link the splat shader (needs a current GL context). Returns a map
  {:program id :locs {uniform-or-attrib location}} or nil on failure (the GL
  info log is printed by gl/make-program)."
  []
  (let [prog (gl/make-program vs-src fs-src)]
    (when prog
      {:program prog
       :locs    {:a_pos      (gl/gl-get-attrib-location  prog "a_pos")
                 :u_splats   (gl/gl-get-uniform-location prog "u_splats")
                 :u_count    (gl/gl-get-uniform-location prog "u_count")
                 :u_viewport (gl/gl-get-uniform-location prog "u_viewport")
                 :u_image    (gl/gl-get-uniform-location prog "u_image")
                  :u_bg       (gl/gl-get-uniform-location prog "u_bg")
                  :u_opacity  (gl/gl-get-uniform-location prog "u_opacity")
                 :u_hard_sharp (gl/gl-get-uniform-location prog "u_hard_sharp")
                 :u_hard_soft  (gl/gl-get-uniform-location prog "u_hard_soft")
                 :u_sig_min    (gl/gl-get-uniform-location prog "u_sig_min")
                 :u_sig_max    (gl/gl-get-uniform-location prog "u_sig_max")}})))
