(ns splat-painter.shader
  "The splat rasterizer as a GLSL fragment shader — the GPU twin of
  splat-painter.gaussian/rasterize. Same math: the symmetric 2×2 covariance's
  closed-form precision, the peak-normalized exp at the mean (pdf=0 there, so
  exp(-pdf)=1.0 without needing the reference's per-grid subtraction), and the
  additive color sum.

  The GLSL is written directly and compiled with glimmer-gl.gl/make-program —
  glimmer-gl.shader's data IR can't express what these need (its compile-stmt only
  handles :let/:set/:if). Uniform and attribute locations are cached at build time.

  Splat stream layout (RGBA32F, THREE texels each — the transform-feedback capture
  format, so a generated buffer feeds the renderers with no readback or repack):
    texel 3·i   = (mean_x, mean_y, c00, c01)
    texel 3·i+1 = (c11,    color_r, color_g, color_b)
    texel 3·i+2 = (alpha,  0, 0, 0)
  where the covariance is symmetric [c00 c01; c01 c11] in (row, col) space and alpha is the
  per-splat paint alpha (brush strokes taper it toward the stroke tail).

  build-program-quad is the production renderer (one blended quad per splat);
  build-program-buf is the same math as a pixels×splats loop, kept for A/B compares
  behind GA_PAINTER_LOOP_RENDER — it hangs the GPU watchdog at high splat counts."
  (:require [glimmer-gl.gl :as gl]))


(def perlin-glsl
  "Ken Perlin's improved noise against a 512x1 permutation texture (u_permTex),
   mirroring splat-painter.noise. Shared verbatim by the generation geometry
   shader and the GPU field passes: the noise has to agree between the two, and
   the surest way is one copy of the source rather than two that look alike."
  "// --- helpers (mirror splat-painter.noise / seed) ---------------------------
int perm(int i){ return int(texelFetch(u_permTex, ivec2(i, 0), 0).r + 0.5); }
float fade(float t){ return t*t*t*(t*(t*6.0-15.0)+10.0); }
float lerpf(float t, float a, float b){ return a + t*(b-a); }
float grad(int h, float x, float y, float z){
  int hh = h & 15;
  float u = hh < 8 ? x : y;
  float v = hh < 4 ? y : (hh == 12 || hh == 14 ? x : z);
  return ((hh & 1) == 0 ? u : -u) + ((hh & 2) == 0 ? v : -v);
}
float noise3(float x, float y, float z){
  float fx = floor(x), fy = floor(y), fz = floor(z);
  int X = int(fx) & 255, Y = int(fy) & 255, Z = int(fz) & 255;
  float xf = x - fx, yf = y - fy, zf = z - fz;
  float u = fade(xf), v = fade(yf), w = fade(zf);
  int A  = perm(X) + Y,   AA = perm(A) + Z,     AB = perm(A + 1) + Z;
  int B  = perm(X+1) + Y, BA = perm(B) + Z,     BB = perm(B + 1) + Z;
  float n = lerpf(w,
    lerpf(v, lerpf(u, grad(perm(AA),   xf,       yf,       zf),
                      grad(perm(BA),   xf - 1.0, yf,       zf)),
             lerpf(u, grad(perm(AB),   xf,       yf - 1.0, zf),
                      grad(perm(BB),   xf - 1.0, yf - 1.0, zf))),
    lerpf(v, lerpf(u, grad(perm(AA+1), xf,       yf,       zf - 1.0),
                      grad(perm(BA+1), xf - 1.0, yf,       zf - 1.0)),
             lerpf(u, grad(perm(AB+1), xf,       yf - 1.0, zf - 1.0),
                      grad(perm(BB+1), xf - 1.0, yf - 1.0, zf - 1.0))));
  return (1.0 + n) / 2.0;
}
float noise2(float x, float y){ return noise3(x, y, 0.0); }")

(def max-splats "shader splat ceiling + transform-feedback buffer capacity" 786432)

(def ^:private vs-src
  "#version 330 core
in vec2 a_pos;                 // fullscreen quad, -1..1
void main(){
  gl_Position = vec4(a_pos, 0.0, 1.0);
}")

;; --- texture-buffer LOOP render variant (GA_PAINTER_LOOP_RENDER only) --------
;; The splats come from a samplerBuffer (a 1D texture view over a buffer object), so
;; there is no GL_MAX_TEXTURE_SIZE ceiling and the transform-feedback buffer feeds
;; straight in. Loops every splat per fragment: correct, but Σ(pixels×splats) work,
;; which trips the GPU watchdog at realistic counts. build-program-quad superseded it.
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
    // ANTIALIAS: below ~2.5px stdev a hard-edged profile spans less than a pixel and
    // shimmers as jaggies — tiny marks ease back to a pure gaussian (soft dab).
    hardness = 1.0 + (hardness - 1.0) * clamp(sig / 2.5, 0.0, 1.0);
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
uniform float u_tex_edge;        // paint-texture: edge-raggedness amount (0 = clean ellipse)
flat out vec3  v_color;
flat out vec3  v_prec;           // p00, p11, cross
flat out float v_hard;
flat out float v_alpha;          // per-splat paint alpha (stroke taper)
flat out vec2  v_major;          // stroke long-axis unit dir (rows, cols) — bristle frame
flat out vec2  v_mean;           // splat mean — per-stroke noise seed
flat out float v_edge;           // per-stroke edge-raggedness amount (0 on base strokes)
flat out float v_sig;            // stroke stdev — sets the PROPORTIONAL texture frequency
flat out float v_texg;           // size mute: full on fine strokes, faint on large ones
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
  v_mean  = t0.xy;
  // stroke long axis = eigenvector of the LARGER eigenvalue of [[c00,c01],[c01,c11]].
  // Bristle streaks run along this; the ragged edge and tonal grooves are keyed to it.
  float disc = sqrt(max(0.25 * (c00 - c11) * (c00 - c11) + c01 * c01, 0.0));
  float l1   = 0.5 * (c00 + c11) + disc;             // major eigenvalue
  // ROBUST eigenvector: for an exactly axis-aligned stroke the float covariance
  // has c01 == 0.0, and with c11 > c00 the first eigenvector form (l1-c11, c01)
  // is exactly (0,0) - normalize((0,0)) is NaN, and one NaN fragment poisons its
  // whole pixel black forever under blending (the black rectangle artifacts on
  // flat image borders). Fall back to the second form, then to the x axis.
  vec2 ev = vec2(l1 - c11, c01);
  if (dot(ev, ev) < 1e-12) ev = vec2(c01, l1 - c00);
  v_major = (disc < 1e-6 || dot(ev, ev) < 1e-12) ? vec2(1.0, 0.0) : normalize(ev);
  float sig = sqrt(sqrt(det));
  float ts  = clamp((sig - u_sig_min) / max(u_sig_max - u_sig_min, 1e-4), 0.0, 1.0);
  ts = ts * ts * (3.0 - 2.0 * ts);
  v_hard = mix(u_hard_sharp, u_hard_soft, ts);
  // ANTIALIAS: tiny marks ease back to a pure gaussian (see the loop shaders)
  v_hard = 1.0 + (v_hard - 1.0) * clamp(sig / 2.5, 0.0, 1.0);
  // edge raggedness rides the SMALLER strokes only: the base/large coverage layer
  // (ts→1) stays solid so thinning coverage can't open gaps to the black clear;
  // fine marks (ts→0) get the full break-up, over the underpainting where it reads.
  float edgeAmt = u_tex_edge * (1.0 - ts);
  v_edge = edgeAmt;
  // paint texture scales with the LAYER: a large low-detail stroke gets it VERY muted
  // (a big smooth daub must not read as sandpaper); fine strokes keep full strength.
  // v_sig drives the proportional frequency in the fragment shader.
  v_sig  = sig;
  v_texg = mix(1.0, 0.12, ts);
  // two triangles (0,1,2)(2,1,3) over corner ids 0..3 = (∓,∓)(±,∓)(∓,±)(±,±)
  int cid = corner == 0 ? 0 : (corner == 1 || corner == 4) ? 1
          : (corner == 2 || corner == 3) ? 2 : 3;
  vec2 s  = vec2((cid & 1) == 0 ? -1.0 : 1.0, (cid & 2) == 0 ? -1.0 : 1.0);
  // marginal stdevs = exact AABB of the ellipse; grow the quad enough to CONTAIN the
  // loosened gaussian when a bristle erodes pdf (effective σ scales ~1/√(1-edgeAmt)),
  // so the tail reaches ~0 inside the quad instead of clipping to a hard rectangle.
  vec2 he = (3.5 + 5.0 * edgeAmt) * sqrt(vec2(c00, c11));
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
flat in vec2  v_major;
flat in vec2  v_mean;
flat in float v_edge;
flat in float v_sig;
flat in float v_texg;
in vec2 v_d;
uniform float u_opacity;
uniform float u_tex_streak;      // bristle tonal-streak amount (0 = off)
uniform float u_tex_grain;       // canvas-grain brightness+chroma amount (0 = off)
uniform float u_tex_edge;        // edge-raggedness amount (0 = clean ellipse)
out vec4 frag;

// hash-without-sine (Dave Hoskins) + bilinear value noise — no trig, no loops, no
// uniform-array indexing, so it steers clear of the Apple GL 4.1 driver quirks.
float hash21(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash21(i),               b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0,1.0)), d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main(){
  float pdf0 = 0.5 * (v_prec.x * v_d.x * v_d.x + v_prec.z * v_d.x * v_d.y + v_prec.y * v_d.y * v_d.y);

  // stroke-local frame: `across` = perpendicular to the drag, `along` = down it.
  float across = dot(v_d, vec2(-v_major.y, v_major.x));
  float along  = dot(v_d, v_major);
  float seed   = hash21(v_mean) * 137.0;                  // per-stroke phase

  // PROPORTIONAL FREQUENCY: scale the noise by stroke size so a big stroke gets
  // proportionally COARSER cells (∝ its scale) instead of the fine sandpaper an
  // absolute frequency paints over a large smooth daub. refSig≈3px keeps the finest
  // strokes at their current look; larger strokes ease to lower frequency.
  float fscale = min(1.0, 3.0 / max(v_sig, 0.5));

  // BRISTLE STREAKS: fine bands ACROSS the stroke, slowly varying ALONG it — the
  // grooves a loaded brush drags. One field drives both the tonal streak and the
  // ragged edge (bristles fall short / overshoot the clean ellipse).
  float streak = vnoise(vec2(across * 0.7, along * 0.06) * fscale + seed) - 0.5;

  // GRAIN: a PER-STROKE isotropic mottle in the stroke's OWN frame with its own
  // seed, so every dab carries its own tooth rather than one shared canvas texture.
  float gn = vnoise(vec2(across, along) * 0.5 * fscale + seed * 1.7)        - 0.5;
  float gs = vnoise(vec2(across, along) * 0.5 * fscale + seed * 1.7 + 19.0) - 0.5;

  // ragged edge: only ever scales pdf, so it's invisible at the core (pdf≈0) and
  // grows toward the shoulder where the contour actually reads. v_edge is 0 on the
  // base coverage strokes (see the vertex shader) so they stay solid. It bites
  // OUTWARD freely (streak<0 → pdf down → the contour feathers past the ellipse) but
  // only LIGHTLY inward — a full inward carve on a ~2px liner beads it into dashes, so
  // damp the shrink side.
  float esf = (streak < 0.0) ? streak : streak * 0.35;
  float pdf = max(pdf0 * (1.0 + v_edge * 2.0 * esf), 0.0);
  float a = v_alpha * u_opacity * exp(-pow(pdf, v_hard));

  // texture catches the LIGHT: bristle marks and tooth show in lit passages, not the
  // dark underlayers — gate by the stroke's own luminance so shadows stay smooth.
  float lum  = dot(v_color, vec3(0.299, 0.587, 0.114));
  float gate = smoothstep(0.02, 0.32, lum);
  // v_texg mutes streak+grain on the large low-detail strokes (see the vertex shader).
  float sAmt = u_tex_streak * gate * v_texg, gAmt = u_tex_grain * gate * v_texg;

  // grain INHERITS the stroke's colour + lightness: it mottles brightness and the
  // stroke's OWN saturation (thicker vs thinner pigment = a richer/greyer version of
  // the same hue), never a foreign colour cast.
  float bright = 1.0 + sAmt * streak + gAmt * gn;
  float sat = 1.0 + gAmt * 0.6 * gs;
  vec3  base = mix(vec3(lum), v_color, sat);   // same hue, varied richness
   vec3  col = clamp(base * bright, 0.0, 1.0);
   frag = vec4(col * a, a);        // premultiplied; blend (ONE, ONE_MINUS_SRC_ALPHA)
}")

;; --- layer blit (layered repainting) ------------------------------------------
;; Attribute-less, like vs-src-quad: 6 GL_TRIANGLES from gl_VertexID cover the
;; letterboxed image rect, and the FS samples a committed pass captured by
;; glReadPixels (a solo render over a transparent clear = premultiplied RGBA).
;; Drawn with blending ENABLED (src-over, ONE/ONE_MINUS_SRC_ALPHA): each layer's
;; premultiplied content is scaled by u_alpha (its glaze opacity) and composited
;; over whatever is below it. The layer texture is uploaded verbatim from a
;; bottom-up readback, so texcoord V rises with image-up (v=0 = image bottom =
;; framebuffer row 0) and the blit overlays the capture exactly upright
;; (see gpu-draw!: v_uv = (pane - org) / (u_image * scale)).
(def ^:private vs-src-blit
  "#version 330 core
uniform vec2 u_viewport;        // framebuffer pixels (vw, vh)
uniform vec2 u_image;           // image pixels (iw, ih)
out vec2 v_uv;                  // 0..1 texcoord into u_layer (v up = image up)
void main(){
  // two triangles (0,1,2)(2,1,3) over corner ids 0..3, same winding as vs-src-quad
  int corner = gl_VertexID;
  int cid = corner == 0 ? 0 : (corner == 1 || corner == 4) ? 1
          : (corner == 2 || corner == 3) ? 2 : 3;
  vec2 s = vec2((cid & 1) == 0 ? -1.0 : 1.0, (cid & 2) == 0 ? -1.0 : 1.0);
  // letterbox rect (contain fit, centred) — same mapping as the splat quad VS
  float scale = min(u_viewport.x / u_image.x, u_viewport.y / u_image.y);
  vec2 org = 0.5 * (u_viewport - u_image * scale);
  vec2 pane = org + 0.5 * u_image * scale * (vec2(1.0) + s);
  v_uv = (pane - org) / (u_image * scale);
  gl_Position = vec4(pane / u_viewport * 2.0 - 1.0, 0.0, 1.0);
}")

(def ^:private fs-src-blit
  "#version 330 core
uniform sampler2D u_layer;      // a committed pass, captured bottom-up via glReadPixels
uniform float u_alpha;          // per-blit opacity gain (the layer glaze strength)
in vec2 v_uv;
out vec4 frag;
void main(){
  vec4 c = texture(u_layer, v_uv);
  frag = vec4(c.rgb * u_alpha, c.a * u_alpha);   // premultiplied content scaled by layer opacity
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
            :u_sig_max    (gl/gl-get-uniform-location prog "u_sig_max")
            :u_tex_streak (gl/gl-get-uniform-location prog "u_tex_streak")
            :u_tex_grain  (gl/gl-get-uniform-location prog "u_tex_grain")
            :u_tex_edge   (gl/gl-get-uniform-location prog "u_tex_edge")}}))

(defn build-program-blit
  "Compile + link the attribute-less layer blit (needs a current GL context).
  Draws 6 GL_TRIANGLES covering the letterboxed image rect and samples u_layer,
  scaling its premultiplied RGBA by u_alpha (the layer's glaze opacity). Draw with
  blending ENABLED (src-over, ONE/ONE_MINUS_SRC_ALPHA) so committed layers composite
  over the base. Reuse any VAO with no enabled attribs (gen-vao).
  Returns {:program :locs} or nil."
  []
  (when-let [prog (gl/make-program vs-src-blit fs-src-blit)]
    {:program prog
     :locs {:u_viewport (gl/gl-get-uniform-location prog "u_viewport")
            :u_image    (gl/gl-get-uniform-location prog "u_image")
            :u_layer    (gl/gl-get-uniform-location prog "u_layer")
            :u_alpha    (gl/gl-get-uniform-location prog "u_alpha")}}))

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
  "Return {:vs-src :fs-src-buf :vs-src-quad :fs-src-quad :vs-src-blit :fs-src-blit}
  — pure, no GL context (for headless inspection/tests)."
  []
  {:vs-src vs-src :fs-src-buf fs-src-buf
   :vs-src-quad vs-src-quad :fs-src-quad fs-src-quad
   :vs-src-blit vs-src-blit :fs-src-blit fs-src-blit})

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
