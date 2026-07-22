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
  GL_MAX_TEXTURE_SIZE (16384 on Apple Silicon — a flat 2×N would cap N there). Each splat uses
  two adjacent texels; TILE_W splats per texture row, so the texture is (2·TILE_W)×ceil(N/TILE_W):
    col = i % TILE_W, row = i / TILE_W
    texelFetch(u_splats, ivec2(2·col,   row),0) = (mean_x, mean_y, c00, c01)
    texelFetch(u_splats, ivec2(2·col+1, row),0) = (c11,     color_r, color_g, color_b)
  where the covariance is symmetric [c00 c01; c01 c11] in (row, col) space."
  (:require [glimmer-gl.gl :as gl]))

(def ^:private max-splats 49152)
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
    vec4 t0 = texelFetch(u_splats, ivec2(2 * col,     row), 0);
    vec4 t1 = texelFetch(u_splats, ivec2(2 * col + 1, row), 0);
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
    float a = u_opacity * exp(-pow(pdf, hardness));  // peak alpha*opacity; hardness>1 = flat core, hard edge (discrete brush marks)
    float wa = T * a;
    acc += wa * t1.yzw;
    T *= (1.0 - a);
  }
  frag = vec4(acc + T * u_bg, 1.0);
}
"))

(defn sources
  "Return {:vs-src :fs-src} — pure, no GL context (for headless inspection/tests)."
  []
  {:vs-src vs-src :fs-src fs-src})

(defn pack-splats
  "Flatten a seq of splats into the RGBA32F texture payload (length 2*N*4): row i
  is [mean_x mean_y c00 c01  c11 r g b]. Pure, no GL."
  [splats]
  (loop [out (transient [])
         s   splats]
    (if-not s
      (persistent! out)
      (let [{[mx my] :mean [c00 c01 _ c11] :cov [r g b] :color} (first s)]
        (recur (-> out
                   (conj! mx) (conj! my) (conj! c00) (conj! c01)
                   (conj! c11) (conj! r) (conj! g) (conj! b))
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
