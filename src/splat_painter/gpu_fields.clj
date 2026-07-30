(ns splat-painter.gpu-fields
  "Per-image field construction as GPU render-to-texture passes.

   The CPU twin is splat-painter.fields/prepare, which stays the headless
   reference: structure/analyze and wavelet/placement-map are called directly by
   ~40 test and diagnostic sites, and glimmer-gl can only make a context current
   on a realized GtkGLArea — so there is no context outside the running app.

   WHY THIS IS A TEXTURE PIPELINE AND NOT A PORT OF THE THREE HOT FUNCTIONS:
   FFI reads run at ~8.6M floats/s, so pulling one 768×512 RGBA32F result back
   costs ~184ms. Moving a single function to the GPU therefore LOSES — bilateral
   blur alone calls box-blur-2d 36 times over full-res arrays, and paying an
   upload+readback per call is ~6s against 1.28s on the CPU. The win only exists
   if the whole chain stays in VRAM and feeds the generation shader's textures
   directly, which is exactly what gen/upload-fields! would otherwise have
   uploaded. So: compute here, never read back — except the detail map, which
   seed/layer-params reduces on the CPU for the splat budget (one ~184ms
   readback per image load, not per render).

   Every pass is a fullscreen triangle over an attributeless VAO (gl_VertexID →
   clip space), rendering into an RGBA32F colour attachment."
  (:require [glimmer-gl.gl :as gl]
            [splat-painter.shader :as shader]
            [jolt.ffi :as ffi]))

;; --- pass plumbing -----------------------------------------------------------

(def ^:private vs-src
  "#version 330 core
// attributeless fullscreen triangle: 3 verts covering clip space, no VBO
out vec2 v_uv;
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}")

(defn- new-target
  "An RGBA32F texture of `w`×`h` with NEAREST/CLAMP sampling — both a pass target
   and an input to the next pass. Returns the texture id."
  [w h]
  (let [t (gl/gen-one gl/gl-gen-textures)]
    (gl/gl-bind-texture gl/GL-TEXTURE-2D t)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-S gl/GL-CLAMP-TO-EDGE)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-T gl/GL-CLAMP-TO-EDGE)
    (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 gl/GL-RGBA32F (int w) (int h) 0
                        gl/GL-RGBA gl/GL-FLOAT ffi/null)
    t))

(defn- read-fbo-binding []
  (let [p (ffi/alloc (ffi/sizeof :int))]
    (gl/gl-get-integerv gl/GL-FRAMEBUFFER-BINDING p)
    (let [v (ffi/read p :int 0)] (ffi/free p) v)))

(def ^:private GL-VIEWPORT 0x0BA2)

(defn- read-viewport []
  (let [p (ffi/alloc (* 4 (ffi/sizeof :int)))]
    (gl/gl-get-integerv GL-VIEWPORT p)
    (let [v (mapv #(ffi/read p :int (* 4 %)) (range 4))] (ffi/free p) v)))

(declare read-viewport)

(defn restore-viewport!
  "Put back the viewport that was current when `ctx` was made. A pass chain leaves
   the viewport sized to its last target, and the caller is usually about to draw
   to a differently-sized framebuffer — on screen that shows up as the image
   painted at the last pass's size in the corner."
  [ctx]
  (when-let [[x y w h] (:vp ctx)]
    (gl/gl-viewport (int x) (int y) (int w) (int h)))
  nil)

(defn make-ctx
  "GL objects shared by every pass: one FBO to attach targets to, one empty VAO
   (core profile still requires a bound VAO for an attributeless draw), and the
   caller's framebuffer so passes can restore it. Call with a current context."
  []
  {:fbo  (gl/gen-one gl/gl-gen-framebuffers)
   :vao  (gl/gen-one gl/gl-gen-vertex-arrays)
   :prev (read-fbo-binding)
   ;; Every pass resizes the viewport to its own target. Whoever called us was
   ;; about to draw something else at a different size, so remember theirs and
   ;; hand it back — see restore-viewport!.
   :vp   (read-viewport)})

(defn free-ctx!
  "Release a ctx's FBO and VAO. One of each per image load is a slow leak, but a
   leak — the ctx has to be remade per load because it captures the viewport."
  [ctx]
  (gl/delete-one gl/gl-delete-framebuffers (:fbo ctx))
  (gl/delete-one gl/gl-delete-vertex-arrays (:vao ctx))
  nil)

(defn run-pass!
  "Render one fullscreen pass of `prog` into `dst` (w×h RGBA32F).
   `inputs` is a seq of [uniform-name texture-id]; `uniforms` a seq of
   [uniform-name kind & vals] with kind :1f / :2f / :1i. Leaves `dst` bound to no
   framebuffer — the caller's binding is restored."
  [ctx prog dst w h inputs uniforms]
  (let [{:keys [fbo vao prev]} ctx
        pid (:program prog)]
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
    (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0
                                  gl/GL-TEXTURE-2D dst 0)
    (when (not= (gl/gl-check-framebuffer-status gl/GL-FRAMEBUFFER)
                gl/GL-FRAMEBUFFER-COMPLETE)
      (throw (ex-info "gpu-fields: framebuffer incomplete" {:w w :h h})))
    (gl/gl-viewport 0 0 (int w) (int h))
    (gl/gl-disable gl/GL-DEPTH-TEST)
    (gl/gl-use-program pid)
    (doseq [[i [uname tex]] (map-indexed vector inputs)]
      (gl/gl-active-texture (+ gl/GL-TEXTURE0 i))
      (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
      (gl/gl-uniform-1i (gl/gl-get-uniform-location pid uname) (int i)))
    (doseq [[uname kind & vals] uniforms]
      (let [loc (gl/gl-get-uniform-location pid uname)]
        (case kind
          :1f (gl/gl-uniform-1f loc (double (first vals)))
          :2f (gl/gl-uniform-2f loc (double (first vals)) (double (second vals)))
          :1i (gl/gl-uniform-1i loc (int (first vals))))))
    (gl/gl-bind-vertex-array vao)
    (gl/gl-draw-arrays gl/GL-TRIANGLES 0 3)
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev)
    dst))

;; --- separable box blur ------------------------------------------------------
;; The CPU keeps an O(1) sliding sum; a fragment shader cannot carry state across
;; pixels, so this is the brute-force 2r+1 taps per pixel. That is the right trade
;; here: the radii in play are 2 (tensor), 3 (light blur) and ~8-16 (flow/heavy),
;; and the GPU runs every pixel at once. Edge-replicate matches box-blur-2d.

(def ^:private box-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2  u_dim;      // (w, h) in texels
uniform vec2  u_step;     // (1,0) horizontal pass, (0,1) vertical
uniform float u_radius;
void main(){
  vec2 dim = u_dim;
  ivec2 c = ivec2(floor(v_uv * dim));
  int r = int(u_radius);
  ivec2 st = ivec2(u_step);
  vec4 sum = vec4(0.0);
  // Dynamic bound, not a constant one with `continue`: the placement map blurs at
  // radius min(sh,sw)/8, which is ~96 at 768 — a fixed +/-64 cap would silently
  // truncate the window, and a cap wide enough for it would burn 1000+ iterations
  // per pixel on the radius-2 tensor blur.
  for (int d = -r; d <= r; ++d) {
    ivec2 p = clamp(c + st * d, ivec2(0), ivec2(dim) - ivec2(1));
    sum += texelFetch(u_src, p, 0);
  }
  frag = sum / (2.0 * u_radius + 1.0);
}")

(defn box-blur!
  "Separable box blur of `src` (w×h RGBA32F) with `radius`, into `dst`. `tmp` is a
   scratch target of the same size. Returns `dst`."
  [ctx progs src dst tmp w h radius]
  (run-pass! ctx (:box progs) tmp w h
             [["u_src" src]]
             [["u_dim" :2f w h] ["u_step" :2f 1 0] ["u_radius" :1f radius]])
  (run-pass! ctx (:box progs) dst w h
             [["u_src" tmp]]
             [["u_dim" :2f w h] ["u_step" :2f 0 1] ["u_radius" :1f radius]])
  dst)

;; --- Di Zenzo colour structure tensor ----------------------------------------
;; structure/analyze in three passes: nearest downscale, gamma + per-channel
;; Sobel into tensor products, then the eigen decomposition. The radius-2 smooth
;; between products and eigen is the box blur above — and because that shader
;; works on vec4, ONE pass smooths jxx/jyy/jxy together where the CPU makes three
;; separate box-blur-2d calls.
;;
;; Orientation convention follows structure.clj, where x is the ROW and y the
;; COLUMN: gx is the derivative down rows, gy across columns. Textures are
;; uploaded row-major, so texel.x is the column and texel.y the row — hence the
;; Sobel below differences texel.y for gx.

(def ^:private down-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_src_dim;
uniform vec2 u_dst_dim;
void main(){
  ivec2 d = ivec2(floor(v_uv * u_dst_dim));
  // structure/downsample is nearest-neighbour at floor(i * src/dst). Integer
  // arithmetic here rather than a float ratio: it is exact, so a destination row
  // can never land one off from a rounding wobble in the divide.
  ivec2 s = (d * ivec2(u_src_dim)) / ivec2(u_dst_dim);
  frag = texelFetch(u_src, min(s, ivec2(u_src_dim) - ivec2(1)), 0);
}")

(def ^:private sobel-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_dim;
vec3 tap(ivec2 p){
  ivec2 q = clamp(p, ivec2(0), ivec2(u_dim) - ivec2(1));
  // gamma 1/2.2 per channel, so an edge in shadow counts like one in light
  return pow(texelFetch(u_src, q, 0).rgb, vec3(0.4545));
}
void main(){
  ivec2 c = ivec2(floor(v_uv * u_dim));
  vec3 L00 = tap(c + ivec2(-1,-1)), L01 = tap(c + ivec2(0,-1)), L02 = tap(c + ivec2(1,-1));
  vec3 L10 = tap(c + ivec2(-1, 0)),                             L12 = tap(c + ivec2(1, 0));
  vec3 L20 = tap(c + ivec2(-1, 1)), L21 = tap(c + ivec2(0, 1)), L22 = tap(c + ivec2(1, 1));
  vec3 gx = (L20 + 2.0*L21 + L22) - (L00 + 2.0*L01 + L02);
  vec3 gy = (-L00 + L02) + (-2.0*L10 + 2.0*L12) + (-L20 + L22);
  // Di Zenzo: sum the per-channel outer products, so a chroma edge with no luma
  // step still registers.
  frag = vec4(dot(gx,gx), dot(gy,gy), dot(gx,gy), 1.0);
}")

(def ^:private eigen-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_dim;
void main(){
  vec4 t = texelFetch(u_src, ivec2(floor(v_uv * u_dim)), 0);
  float a = t.r, b = t.g, f = t.b;
  // minor-eigenvector angle: the direction the edge RUNS, not its normal.
  // In a perfectly uniform region every Sobel tap is exactly 0, so both
  // arguments are 0 — GLSL leaves atan(0,0) undefined and this driver returns
  // NaN where Math/atan2 gives 0. Any solid-colour area (a blown background, a
  // letterbox bar) hits it, and one NaN theta poisons a splat's covariance.
  float num = 2.0*f, den = a - b;
  float phi = (num == 0.0 && den == 0.0) ? 0.0 : atan(num, den);
  float theta = 0.5 * phi + 1.5707963267948966;
  float d = sqrt(((a-b)*0.5)*((a-b)*0.5) + f*f);
  float coh = clamp(2.0*d / (a + b + 1e-9), 0.0, 1.0);
  frag = vec4(theta, coh, a + b, 1.0);
}")

(defn tensor-dims
  "The reduced tensor resolution structure/analyze would use for an H×W image —
   computed here so both paths downscale to exactly the same grid."
  [H W max-side]
  (let [scale (min 1.0 (/ (double max-side) (double (max (long H) (long W)))))]
    (if (>= scale 1.0)
      [(long H) (long W)]
      [(max 1 (long (Math/round (* (double H) scale))))
       (max 1 (long (Math/round (* (double W) scale))))])))

(defn analyze!
  "GPU twin of structure/analyze. `src` is the full-res RGBA32F image texture.
   Returns {:h Ht :w Wt :eigen tex :flow tex :tensor tex}, where :eigen and :flow
   hold (theta, coherence, grad2) and (flow-theta, flow-str, _) per texel.

   Allocates its own targets; the caller owns them for the life of the image."
  [ctx progs src H W max-side]
  (let [[Ht Wt] (tensor-dims H W max-side)
        full?   (and (= Ht (long H)) (= Wt (long W)))
        raw     (new-target Wt Ht)
        tmp     (new-target Wt Ht)
        tensor  (new-target Wt Ht)
        flow    (new-target Wt Ht)
        eigen-t (new-target Wt Ht)
        eigen-f (new-target Wt Ht)
        ;; the flow radius scales with the tensor grid, not the source image —
        ;; same expression as analyze, so the diffusion covers the same fraction
        fr      (max 3 (min 16 (quot (min Ht Wt) 3)))
        dims    [["u_dim" :2f Wt Ht]]]
    ;; At or below max-side, analyze skips the downscale — so sample the source
    ;; directly rather than paying an identity copy through a scratch target.
    (let [small (if full?
                  src
                  (let [t (new-target Wt Ht)]
                    (run-pass! ctx (:down progs) t Wt Ht
                               [["u_src" src]]
                               [["u_src_dim" :2f W H] ["u_dst_dim" :2f Wt Ht]])
                    t))]
      (run-pass! ctx (:sobel progs) raw Wt Ht [["u_src" small]] dims))
    ;; one vec4 blur where the CPU runs three: jxx, jyy and jxy ride the channels
    (box-blur! ctx progs raw tensor tmp Wt Ht 2)
    (box-blur! ctx progs tensor flow tmp Wt Ht fr)
    (run-pass! ctx (:eigen progs) eigen-t Wt Ht [["u_src" tensor]] dims)
    (run-pass! ctx (:eigen progs) eigen-f Wt Ht [["u_src" flow]] dims)
    {:h Ht :w Wt :eigen eigen-t :flow eigen-f :tensor tensor}))

;; --- whole-grid reductions ---------------------------------------------------
;; placement-map needs six scalars over the whole grid (gmax, and a dmax + mean
;; per fused band). A fragment shader can only gather locally, so these come from
;; a log-step halving: seed each texel to (v, v), then repeatedly fold 2x2 blocks
;; taking max of maxes and sum of sums until one texel is left.
;;
;; That 1x1 target is then SAMPLED by the shaders that need it. Reading it back to
;; feed in as a uniform would cost a pipeline stall per statistic, which is the
;; whole thing this design exists to avoid. Tree summation is also more accurate
;; in float32 than the CPU's sequential loop is in float64 at this size.

(def ^:private redseed-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_dim;
uniform float u_channel;
void main(){
  vec4 t = texelFetch(u_src, ivec2(floor(v_uv * u_dim)), 0);
  int c = int(u_channel);
  float v = c == 0 ? t.r : (c == 1 ? t.g : (c == 2 ? t.b : t.a));
  frag = vec4(v, v, 0.0, 1.0);
}")

(def ^:private reduce-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_src_dim;
uniform vec2 u_dst_dim;
void main(){
  ivec2 d = ivec2(floor(v_uv * u_dst_dim));
  ivec2 s = d * 2;
  ivec2 lim = ivec2(u_src_dim);
  // Odd dimensions leave a ragged edge; the identity for max is the first tap and
  // for sum is 0, so out-of-range taps are skipped rather than clamped — clamping
  // would double-count a texel into the sum.
  vec4 acc = texelFetch(u_src, min(s, lim - ivec2(1)), 0);
  float mx = acc.r, sm = acc.g;
  if (s.x + 1 < lim.x) { vec4 t = texelFetch(u_src, ivec2(s.x+1, s.y), 0); mx = max(mx, t.r); sm += t.g; }
  if (s.y + 1 < lim.y) { vec4 t = texelFetch(u_src, ivec2(s.x, s.y+1), 0); mx = max(mx, t.r); sm += t.g; }
  if (s.x + 1 < lim.x && s.y + 1 < lim.y) { vec4 t = texelFetch(u_src, ivec2(s.x+1, s.y+1), 0); mx = max(mx, t.r); sm += t.g; }
  frag = vec4(mx, sm, 0.0, 1.0);
}")

(defn- free-textures! [ts]
  (doseq [t ts] (gl/delete-one gl/gl-delete-textures t)))

(defn stats!
  "Reduce channel `c` of a w×h texture to a 1×1 RGBA32F holding (max, sum).
   Returns [tex scratch] — `tex` is the 1×1 result, `scratch` the intermediates
   the caller must free."
  [ctx progs src w h c]
  (let [seed (new-target w h)]
    (run-pass! ctx (:redseed progs) seed w h
               [["u_src" src]] [["u_dim" :2f w h] ["u_channel" :1f c]])
    (loop [cur seed cw (long w) ch (long h) scratch []]
      (if (and (= cw 1) (= ch 1))
        [cur scratch]
        (let [nw (max 1 (quot (+ cw 1) 2))
              nh (max 1 (quot (+ ch 1) 2))
              dst (new-target nw nh)]
          (run-pass! ctx (:reduce progs) dst nw nh
                     [["u_src" cur]]
                     [["u_src_dim" :2f cw ch] ["u_dst_dim" :2f nw nh]])
          (recur dst nw nh (conj scratch cur)))))))

;; --- Haar placement map ------------------------------------------------------
;; wavelet/placement-map as passes. The CPU SCATTERS: each half-cell at level L
;; adds its detail energy to every pixel of its 2^L block. A fragment shader can
;; only gather, so this inverts it — output pixel (gr,gc) at level L reads the one
;; half-cell that covers it, at (gr >> L, gc >> L). Same sum, opposite direction.
;;
;; The LL pyramid the levels walk down is just the 2x2 means the CPU already
;; writes into `ll`, so it is built once and each level's energy pass reads the
;; rung above it.

(def ^:private luma-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_src_dim;
uniform vec2 u_dst_dim;
void main(){
  ivec2 d = ivec2(floor(v_uv * u_dst_dim));
  ivec2 s = min((d * ivec2(u_src_dim)) / ivec2(u_dst_dim), ivec2(u_src_dim) - ivec2(1));
  vec3 c = texelFetch(u_src, s, 0).rgb;
  // BT.601 luma, then gamma 1/2.2 — placement-map gamma-corrects AFTER the
  // luma reduction, not per channel before it like gradient-field does.
  float l = 0.299*c.r + 0.587*c.g + 0.114*c.b;
  frag = vec4(pow(l, 0.4545), 0.0, 0.0, 1.0);
}")

(def ^:private ll-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_src_dim;
uniform vec2 u_dst_dim;
void main(){
  ivec2 d = ivec2(floor(v_uv * u_dst_dim));
  ivec2 s = d * 2;
  float a = texelFetch(u_src, s, 0).r;
  float b = texelFetch(u_src, ivec2(s.x+1, s.y), 0).r;
  float dd = texelFetch(u_src, ivec2(s.x, s.y+1), 0).r;
  float e = texelFetch(u_src, ivec2(s.x+1, s.y+1), 0).r;
  frag = vec4(0.25*(a+b+dd+e), 0.0, 0.0, 1.0);
}")

(def ^:private haar-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_acc;    // energy accumulated by the coarser levels so far
uniform sampler2D u_ll;     // the rung above this level
uniform vec2 u_dim;         // placement-map grid (sw, sh)
uniform vec2 u_ll_dim;      // dims of u_ll
uniform vec2 u_cell_dim;    // dims of the half-cell grid = floor(u_ll_dim / 2)
uniform float u_block;      // 2^level
void main(){
  ivec2 g = ivec2(floor(v_uv * u_dim));
  float acc = texelFetch(u_acc, g, 0).r;
  int block = int(u_block);
  ivec2 rc = g / block;
  // Cells past the half-grid never got scattered into on the CPU (its inner loop
  // only walks r < hc2), so they contribute nothing here either.
  if (rc.x < int(u_cell_dim.x) && rc.y < int(u_cell_dim.y)) {
    ivec2 s = rc * 2;
    float a = texelFetch(u_ll, s, 0).r;
    float b = texelFetch(u_ll, ivec2(s.x+1, s.y), 0).r;
    float d = texelFetch(u_ll, ivec2(s.x, s.y+1), 0).r;
    float e = texelFetch(u_ll, ivec2(s.x+1, s.y+1), 0).r;
    // 2x2 Haar: vertical, horizontal and diagonal detail. LL is handled by ll-fs.
    float lh = (a + b) - (d + e);
    float hl = (a + d) - (b + e);
    float hh = (a + e) - (b + d);
    acc += abs(lh) + abs(hl) + abs(hh);
  }
  frag = vec4(acc, 0.0, 0.0, 1.0);
}")

(def ^:private sub-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform vec2 u_dim;
void main(){
  ivec2 p = ivec2(floor(v_uv * u_dim));
  frag = vec4(texelFetch(u_a, p, 0).r - texelFetch(u_b, p, 0).r, 0.0, 0.0, 1.0);
}")

(def ^:private edge-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_eigen;   // (theta, coherence, grad2) on the tensor grid
uniform sampler2D u_stats;   // 1x1 (max, sum) of grad2 -> gmax
uniform vec2 u_dim;          // placement grid (sw, sh)
uniform vec2 u_sf_dim;       // tensor grid (sf_w, sf_h)
uniform vec2 u_src_dim;      // source image (W, H)
void main(){
  ivec2 g = ivec2(floor(v_uv * u_dim));
  // Two-step exactly as the CPU does it (placement cell -> source px -> tensor
  // cell) rather than the algebraically equal single ratio, so the rounding
  // lands the same way. floor(x+0.5), not round(): GLSL leaves round()'s
  // behaviour at .5 up to the implementation, Math/round does not.
  float x = float(g.y) * (u_src_dim.y / u_dim.y);
  float y = float(g.x) * (u_src_dim.x / u_dim.x);
  float sfi = floor(x * (u_sf_dim.y / u_src_dim.y) + 0.5);
  float sfj = floor(y * (u_sf_dim.x / u_src_dim.x) + 0.5);
  ivec2 sp = ivec2(clamp(sfj, 0.0, u_sf_dim.x - 1.0), clamp(sfi, 0.0, u_sf_dim.y - 1.0));
  float gmax = max(texelFetch(u_stats, ivec2(0), 0).r, 1e-12);
  frag = vec4(sqrt(max(0.0, texelFetch(u_eigen, sp, 0).b) / gmax), 0.0, 0.0, 1.0);
}")

(def ^:private efuse-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_e;
uniform sampler2D u_me;      // wide blur of E
uniform sampler2D u_stats;   // 1x1 (max, sum) of E
uniform vec2 u_dim;
uniform float u_n;
void main(){
  ivec2 p = ivec2(floor(v_uv * u_dim));
  float e = texelFetch(u_e, p, 0).r;
  float me = texelFetch(u_me, p, 0).r;
  float meanE = texelFetch(u_stats, ivec2(0), 0).g / u_n;
  // local prominence: an eyelid crease is faint beside a hard background edge,
  // and it is the local ratio that says it matters
  float ln = min(1.0, (0.85 * e) / (2.0 * me + 0.30 * meanE + 1e-12));
  frag = vec4(max(e, ln), 0.0, 0.0, 1.0);
}")

(def ^:private fuse-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_a;       // the band, blurred
uniform sampler2D u_ma;      // wide blur of u_a
uniform sampler2D u_stats;   // 1x1 (max, sum) of u_a
uniform sampler2D u_efuse;
uniform vec2 u_dim;
uniform float u_n;
uniform float u_esq;
void main(){
  ivec2 p = ivec2(floor(v_uv * u_dim));
  float a = texelFetch(u_a, p, 0).r;
  vec2 st = texelFetch(u_stats, ivec2(0), 0).rg;
  float dmax = st.r, meanA = st.g / u_n;
  float g = dmax > 0.0 ? a / dmax : 0.0;                       // global floor
  float l = min(1.0, a / (2.0 * texelFetch(u_ma, p, 0).r + 0.30 * meanA + 1e-12));
  float E0 = texelFetch(u_efuse, p, 0).r;
  // E carries the tensor blur's width, so fusing it raw seeds a band of echo
  // strokes parallel to each edge; squaring concentrates it onto the core.
  float E = u_esq > 0.5 ? E0 * E0 : E0;
  frag = vec4(min(1.0, max(max(g, l), 0.85 * E)), 0.0, 0.0, 1.0);
}")

(def ^:private subjraw-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_fine;
uniform sampler2D u_e;
uniform vec2 u_dim;
void main(){
  ivec2 p = ivec2(floor(v_uv * u_dim));
  // ABSOLUTE, globally-scaled: the fused maps normalize locally, which lights
  // smooth bokeh up to full 'detail'. Bokeh has to score ~0 here.
  frag = vec4(min(1.0, max(texelFetch(u_fine, p, 0).r / 0.35,
                           texelFetch(u_e, p, 0).r / 0.30)), 0.0, 0.0, 1.0);
}")

(def ^:private dilate-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_dim;
uniform vec2 u_step;
uniform float u_radius;
void main(){
  ivec2 c = ivec2(floor(v_uv * u_dim));
  int r = int(u_radius);
  ivec2 st = ivec2(u_step);
  float m = 0.0;
  for (int d = -r; d <= r; ++d) {
    ivec2 p = clamp(c + st * d, ivec2(0), ivec2(u_dim) - ivec2(1));
    m = max(m, texelFetch(u_src, p, 0).r);
  }
  frag = vec4(m, 0.0, 0.0, 1.0);
}")

;; --- binned colour fields (bilateral + dominant) -----------------------------
;; structure/bilateral-blur and dominant-blur-full share one algorithm: K=9 luma
;; bins, each pixel given a hat weight around its bin centre, the weighted colour
;; AND the weight box-blurred over the window, then the bins recombined. They
;; differ only in the recombination — bilateral weights a bin by the centre
;; pixel's own hat value, dominant by the blurred weight raised to p, which turns
;; `average tone` into `modal tone`.
;;
;; The CPU carries wk/wr/wg/wb as four arrays and blurs each separately. Here they
;; are the four channels of one vec4, so a single box blur does all four — and the
;; accumulator packs (rgb, weight) the same way.

(def ^:private binw-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_dim;
uniform float u_lk;      // this bin's luma centre
uniform float u_dl;      // bin spacing; the hat reaches exactly one bin either way
void main(){
  vec3 c = texelFetch(u_src, ivec2(floor(v_uv * u_dim)), 0).rgb;
  float L = 0.299*c.r + 0.587*c.g + 0.114*c.b;   // raw luma, no gamma here
  float w = max(0.0, 1.0 - abs(L - u_lk) / u_dl);
  frag = vec4(w * c, w);
}")

(def ^:private bilacc-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_acc;
uniform sampler2D u_bin;   // box-blurred (w*rgb, w) for this bin
uniform sampler2D u_src;
uniform vec2 u_dim;
uniform float u_lk;
uniform float u_dl;
void main(){
  ivec2 p = ivec2(floor(v_uv * u_dim));
  vec4 acc = texelFetch(u_acc, p, 0);
  vec3 c = texelFetch(u_src, p, 0).rgb;
  float L = 0.299*c.r + 0.587*c.g + 0.114*c.b;
  float h = max(0.0, 1.0 - abs(L - u_lk) / u_dl);
  if (h > 0.0) {
    vec4 b = texelFetch(u_bin, p, 0);
    // bin colour is b.rgb/b.a; this pixel's share of it is h
    acc.rgb += (h / max(b.a, 1e-12)) * b.rgb;
    acc.a += h;
  }
  frag = acc;
}")

(def ^:private domacc-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_acc;
uniform sampler2D u_bin;
uniform vec2 u_dim;
uniform float u_p;
void main(){
  ivec2 q = ivec2(floor(v_uv * u_dim));
  vec4 acc = texelFetch(u_acc, q, 0);
  vec4 b = texelFetch(u_bin, q, 0);
  if (b.a > 1e-12) {
    // vote is w^p for colour b.rgb/w, so accumulate w^(p-1)*b.rgb and never
    // divide twice
    float wp = pow(b.a, u_p);
    acc.rgb += (wp / b.a) * b.rgb;
    acc.a += wp;
  }
  frag = acc;
}")

(def ^:private binfin-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_acc;
uniform vec2 u_dim;
uniform float u_unity;   // 1.0 = hat kernels partition unity, only divide if off
void main(){
  vec4 a = texelFetch(u_acc, ivec2(floor(v_uv * u_dim)), 0);
  if (u_unity > 0.5 && abs(a.a - 1.0) <= 1e-9) { frag = vec4(a.rgb, 1.0); return; }
  frag = vec4(a.rgb / max(a.a, 1e-12), 1.0);
}")

(def ^:private upsample-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_src_dim;
uniform vec2 u_dst_dim;
void main(){
  ivec2 d = ivec2(floor(v_uv * u_dst_dim));
  // matches upsample-rgb: (n-1)/(N-1) spacing, clamped at the last sample
  vec2 r = (max(vec2(2.0), u_src_dim) - 1.0) / max(vec2(1.0), u_dst_dim - 1.0);
  vec2 f = min(u_src_dim - 1.0, vec2(d) * r);
  vec2 i0 = floor(f);
  vec2 t = f - i0;
  ivec2 a = ivec2(i0);
  ivec2 b = min(a + ivec2(1), ivec2(u_src_dim) - ivec2(1));
  vec3 c00 = texelFetch(u_src, ivec2(a.x, a.y), 0).rgb;
  vec3 c01 = texelFetch(u_src, ivec2(b.x, a.y), 0).rgb;
  vec3 c10 = texelFetch(u_src, ivec2(a.x, b.y), 0).rgb;
  vec3 c11 = texelFetch(u_src, ivec2(b.x, b.y), 0).rgb;
  vec3 top = mix(c00, c01, t.x);
  vec3 bot = mix(c10, c11, t.x);
  frag = vec4(mix(top, bot, t.y), 1.0);
}")

;; --- baked orientation field (seed/prep-noise) -------------------------------
;; Both ends of the Swirl dial, at tensor resolution: .rg carry cos2θ/sin2θ and
;; .b the coherence, which is exactly the layout gen/upload-fields! hands the
;; geometry shader — so these two passes ARE the noise textures, not a step
;; toward them.
;;
;; Double-angle components rather than the angle itself because orientation is
;; undirected (0 ≡ π); the GS's bilinear fetch has to interpolate something that
;; wraps correctly, and an angle grid stair-steps into a visible zipper.

(def ^:private noise-fs
  (str
   "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_eigen;    // (theta, coherence, grad2) sharp tensor
uniform sampler2D u_flow;     // (flow-theta, flow-str, _) diffused tensor
uniform sampler2D u_permTex;  // 512x1 Perlin permutation
uniform vec2 u_dim;           // tensor grid (w, h)
uniform vec2 u_src;           // source image (W, H)
uniform float u_swirl0;       // 1.0 = bake the Swirl 0 end instead
"
   shader/perlin-glsl
   "
// undirected blend: mix in the double-angle plane so 0 and pi are the same axis
float blendAngle(float t1, float t2, float w){
  float bx = (1.0 - w) * cos(2.0*t1) + w * cos(2.0*t2);
  float by = (1.0 - w) * sin(2.0*t1) + w * sin(2.0*t2);
  // two exactly-opposite axes at w=0.5 cancel; atan(0,0) is NaN here, 0 on the CPU
  return (bx == 0.0 && by == 0.0) ? 0.0 : 0.5 * atan(by, bx);
}
void main(){
  ivec2 g = ivec2(floor(v_uv * u_dim));
  vec4 e = texelFetch(u_eigen, g, 0);
  vec4 f = texelFetch(u_flow, g, 0);
  float sTheta = e.r, coherence = e.g;
  float fTheta = f.r, fStr = f.g;
  float theta;
  if (u_swirl0 > 0.5) {
    // Swirl 0: flat regions take the diffused tensor's own direction -- the
    // nearby edges voted inward -- with no image-independent field at all.
    theta = blendAngle(fTheta, sTheta, coherence);
  } else {
    // seed/prep-noise indexes x by ROW and y by COLUMN, so g.y is x here
    float x = float(g.y) * (u_src.y / u_dim.y);
    float y = float(g.x) * (u_src.x / u_dim.x);
    float fs = 0.004;
    float fvx = noise2(x * fs, y * fs) - 0.5;
    float fvy = noise2(x * fs + 137.0, y * fs + 91.0) - 0.5;
    // At an integer lattice point improved Perlin is exactly 0.5, so both
    // components are exactly 0 -- which happens at texel (0,0). GLSL leaves
    // atan(0,0) undefined and this driver returns NaN, where Math/atan2 gives 0;
    // one NaN texel would then bleed through the GS's bilinear fetch.
    float ang = (fvx == 0.0 && fvy == 0.0) ? 0.0 : atan(fvy, fvx);
    float flowBase = blendAngle(ang, fTheta, min(1.0, 2.5 * fStr));
    theta = blendAngle(flowBase, sTheta, coherence);
  }
  frag = vec4(cos(2.0*theta), sin(2.0*theta), coherence, 0.0);
}"))

(def ^:private pack-fs
  "#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_detail;
uniform sampler2D u_sharp;
uniform sampler2D u_edge;
uniform sampler2D u_mid;
uniform vec2 u_dim;
void main(){
  ivec2 p = ivec2(floor(v_uv * u_dim));
  // the layout gen's shader expects: aggregate, sharp fine-band, raw edge, mid band
  frag = vec4(texelFetch(u_detail, p, 0).r, texelFetch(u_sharp, p, 0).r,
              texelFetch(u_edge, p, 0).r,   texelFetch(u_mid, p, 0).r);
}")

(defn build-programs
  "Compile the field-construction programs. Needs a current GL context. Returns
   nil if any shader fails, so a caller can report rather than draw garbage."
  []
  (let [ps {:box     (gl/make-program vs-src box-fs)
            :down    (gl/make-program vs-src down-fs)
            :sobel   (gl/make-program vs-src sobel-fs)
            :eigen   (gl/make-program vs-src eigen-fs)
            :redseed (gl/make-program vs-src redseed-fs)
            :reduce  (gl/make-program vs-src reduce-fs)
            :luma    (gl/make-program vs-src luma-fs)
            :ll      (gl/make-program vs-src ll-fs)
            :haar    (gl/make-program vs-src haar-fs)
            :sub     (gl/make-program vs-src sub-fs)
            :edge    (gl/make-program vs-src edge-fs)
            :efuse   (gl/make-program vs-src efuse-fs)
            :fuse    (gl/make-program vs-src fuse-fs)
            :subjraw (gl/make-program vs-src subjraw-fs)
            :dilate  (gl/make-program vs-src dilate-fs)
            :binw    (gl/make-program vs-src binw-fs)
            :bilacc  (gl/make-program vs-src bilacc-fs)
            :domacc  (gl/make-program vs-src domacc-fs)
            :binfin  (gl/make-program vs-src binfin-fs)
            :upsamp  (gl/make-program vs-src upsample-fs)
            :noise   (gl/make-program vs-src noise-fs)
            :pack    (gl/make-program vs-src pack-fs)}]
    (when (every? some? (vals ps))
      (into {} (map (fn [[k v]] [k {:program v}])) ps))))

(defn- clear-target!
  "Zero a target — the Haar accumulator starts empty and each level adds to it."
  [ctx tex w h]
  (let [{:keys [fbo prev]} ctx]
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
    (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0
                                  gl/GL-TEXTURE-2D tex 0)
    (gl/gl-viewport 0 0 (int w) (int h))
    (gl/gl-clear-color 0.0 0.0 0.0 0.0)
    (gl/gl-clear gl/GL-COLOR-BUFFER-BIT)
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev)
    tex))

(defn placement-map!
  "GPU twin of wavelet/placement-map. `src` is the full-res RGBA32F image and
   `eigen` the tensor grid's (theta, coherence, grad2) from `analyze!`.

   Returns {:h :w :detail :sharp :mid :edge :subject}, each a w×h RGBA32F texture
   carrying its value in R. Frees every intermediate before returning; the caller
   owns the five outputs."
  ([ctx progs src eigen H W sf-h sf-w] (placement-map! ctx progs src eigen H W sf-h sf-w 768 4))
  ([ctx progs src eigen H W sf-h sf-w max-side levels]
   (let [[sh sw] (tensor-dims H W max-side)
         n       (* (long sh) (long sw))
         scratch (atom [])
         tmp!    (fn [] (let [t (new-target sw sh)] (swap! scratch conj t) t))
         dims    [["u_dim" :2f sw sh]]
         tmpbuf  (tmp!)                       ; shared blur scratch
         blur!   (fn [in out r] (box-blur! ctx progs in out tmpbuf sw sh r) out)
         stat!   (fn [tex w h c]
                   (let [[t sc] (stats! ctx progs tex w h c)]
                     (swap! scratch into sc) (swap! scratch conj t) t))
         ;; gamma-corrected luma at placement resolution
         lum     (tmp!)
         _       (run-pass! ctx (:luma progs) lum sw sh
                            [["u_src" src]]
                            [["u_src_dim" :2f W H] ["u_dst_dim" :2f sw sh]])
         ;; Walk the LL pyramid, accumulating each level's detail energy. Keeping
         ;; every rung's accumulator is what gives the band snapshots for free:
         ;; the CPU aclones `acc` at levels 1-3 for exactly this.
         accs    (loop [lev 1, cur lum, ch (long sh), cw (long sw)
                        acc (clear-target! ctx (tmp!) sw sh), out [] ]
                   (if (and (<= lev (long levels)) (>= ch 2) (>= cw 2))
                     (let [hc2 (quot ch 2) wc2 (quot cw 2)
                           ll  (let [t (new-target wc2 hc2)] (swap! scratch conj t) t)
                           nxt (tmp!)]
                       (run-pass! ctx (:ll progs) ll wc2 hc2
                                  [["u_src" cur]]
                                  [["u_src_dim" :2f cw ch] ["u_dst_dim" :2f wc2 hc2]])
                       (run-pass! ctx (:haar progs) nxt sw sh
                                  [["u_acc" acc] ["u_ll" cur]]
                                  [["u_dim" :2f sw sh] ["u_ll_dim" :2f cw ch]
                                   ["u_cell_dim" :2f wc2 hc2]
                                   ["u_block" :1f (bit-shift-left 1 lev)]])
                       (recur (inc lev) ll hc2 wc2 nxt (conj out nxt)))
                     (if (seq out) out [acc])))
         final   (peek accs)
         at      (fn [i] (if (> (count accs) i) (nth accs i) final))
         acc1    (at 0)  ; finest band alone
         accfine (at 1)  ; bands 1-2 — text/eye scale
         acc3    (at 2)
         accmid  (let [t (tmp!)]
                   (run-pass! ctx (:sub progs) t sw sh
                              [["u_a" acc3] ["u_b" acc1]] dims)
                   t)
         ;; E: tensor edge strength, globally scaled by gmax and resampled onto
         ;; the placement grid. gmax is the reduction that ema asked for — it
         ;; stays a texture, so nothing stalls on a readback.
         gstat   (stat! eigen sf-w sf-h 2)
         E       (tmp!)
         _       (run-pass! ctx (:edge progs) E sw sh
                            [["u_eigen" eigen] ["u_stats" gstat]]
                            [["u_dim" :2f sw sh] ["u_sf_dim" :2f sf-w sf-h]
                             ["u_src_dim" :2f W H]])
         wide-r  (max 2 (quot (min (long sh) (long sw)) 8))
         mE      (blur! E (tmp!) wide-r)
         eStat   (stat! E sw sh 0)
         efuse   (tmp!)
         _       (run-pass! ctx (:efuse progs) efuse sw sh
                            [["u_e" E] ["u_me" mE] ["u_stats" eStat]]
                            [["u_dim" :2f sw sh] ["u_n" :1f n]])
         fuse!   (fn [raw blur-r smooth-r e-sq?]
                   (let [a  (blur! raw (tmp!) blur-r)
                         st (stat! a sw sh 0)
                         ma (blur! a (tmp!) wide-r)
                         P  (tmp!)]
                     (run-pass! ctx (:fuse progs) P sw sh
                                [["u_a" a] ["u_ma" ma] ["u_stats" st] ["u_efuse" efuse]]
                                [["u_dim" :2f sw sh] ["u_n" :1f n]
                                 ["u_esq" :1f (if e-sq? 1.0 0.0)]])
                     (blur! P (new-target sw sh) smooth-r)))
         mn      (min (long sh) (long sw))
         ;; aggregate: smoothed hard, so stroke density changes gradually
         detail  (fuse! final (max 1 (quot mn 40)) (max 1 (quot mn 50)) false)
         ;; sharp: fine bands, almost no smoothing — text-scale structure survives
         sharp   (fuse! accfine 1 1 true)
         mid     (fuse! accmid (max 1 (quot mn 60)) 1 true)
         ;; subject: absolute structure, then DILATED — smooth skin between the
         ;; eyes has as little raw fine energy as bokeh, and without the dilation
         ;; the whole face reads as background.
         sraw    (tmp!)
         _       (run-pass! ctx (:subjraw progs) sraw sw sh
                            [["u_fine" accfine] ["u_e" E]] dims)
         sb      (blur! sraw (tmp!) (max 2 (quot mn 24)))
         dr      (max 4 (quot mn 12))
         dx      (tmp!)
         dy      (tmp!)
         _       (run-pass! ctx (:dilate progs) dx sw sh [["u_src" sb]]
                            [["u_dim" :2f sw sh] ["u_step" :2f 1 0] ["u_radius" :1f dr]])
         _       (run-pass! ctx (:dilate progs) dy sw sh [["u_src" dx]]
                            [["u_dim" :2f sw sh] ["u_step" :2f 0 1] ["u_radius" :1f dr]])
         subject (blur! dy (new-target sw sh) (max 2 (quot mn 30)))]
     ;; `edge` is an output, so hand back a copy that outlives the scratch sweep
     (let [edge-out (blur! E (new-target sw sh) 0)]
       (free-textures! @scratch)
       {:h sh :w sw :detail detail :sharp sharp :mid mid
        :edge edge-out :subject subject}))))

(def ^:private bins 9)

(defn- binned!
  "Shared body of bilateral-blur and dominant-blur-full: walk the K luma bins,
   box-blurring each bin's (w*rgb, w) and folding it in with `acc-prog`.
   Writes into `dst`. `extra` are uniforms the accumulator needs beyond the bin."
  [ctx progs src dst w h radius acc-prog extra unity?]
  (let [scratch (atom [])
        tmp!    (fn [] (let [t (new-target w h)] (swap! scratch conj t) t))
        dl      (/ 1.0 (double (dec bins)))
        binbuf  (tmp!) blurred (tmp!) blurtmp (tmp!)]
    (loop [k 0, acc (clear-target! ctx (tmp!) w h)]
      (if (= k bins)
        (run-pass! ctx (:binfin progs) dst w h
                   [["u_acc" acc]]
                   [["u_dim" :2f w h] ["u_unity" :1f (if unity? 1.0 0.0)]])
        (let [lk  (* (double k) dl)
              nxt (tmp!)]
          (run-pass! ctx (:binw progs) binbuf w h
                     [["u_src" src]]
                     [["u_dim" :2f w h] ["u_lk" :1f lk] ["u_dl" :1f dl]])
          (box-blur! ctx progs binbuf blurred blurtmp w h radius)
          (run-pass! ctx acc-prog nxt w h
                     (into [["u_acc" acc] ["u_bin" blurred]]
                           (when (= acc-prog (:bilacc progs)) [["u_src" src]]))
                     (into [["u_dim" :2f w h] ["u_lk" :1f lk] ["u_dl" :1f dl]] extra))
          (recur (inc k) nxt))))
    (free-textures! @scratch)
    dst))

(defn bilateral-blur!
  "GPU twin of structure/bilateral-blur — the edge-aware colour field strokes load
   from. Writes an RGB result into `dst`."
  [ctx progs src dst w h radius]
  (binned! ctx progs src dst w h radius (:bilacc progs) [] true))

(defn dominant-blur!
  "GPU twin of structure/dominant-blur: the DOMINANT tone in a brush-sized window.
   Like the CPU it runs on a nearest-downsampled copy when the radius allows and
   bilinearly expands the result — the output is a window statistic, so nothing
   near pixel frequency is lost, and full-res binning is ~16x the work."
  ([ctx progs src dst w h radius] (dominant-blur! ctx progs src dst w h radius 4.0))
  ([ctx progs src dst w h radius p]
   (let [step (max 1 (quot (long radius) 3))]
     (if (<= step 1)
       (binned! ctx progs src dst w h radius (:domacc progs) [["u_p" :1f p]] false)
       (let [sw (max 1 (quot (long w) step))
             sh (max 1 (quot (long h) step))
             small (new-target sw sh)
             dom   (new-target sw sh)]
         (run-pass! ctx (:down progs) small sw sh
                    [["u_src" src]]
                    [["u_src_dim" :2f w h] ["u_dst_dim" :2f sw sh]])
         (binned! ctx progs small dom sw sh (quot (long radius) step)
                  (:domacc progs) [["u_p" :1f p]] false)
         (run-pass! ctx (:upsamp progs) dst w h
                    [["u_src" dom]]
                    [["u_src_dim" :2f sw sh] ["u_dst_dim" :2f w h]])
         (free-textures! [small dom])
         dst)))))

(defn noise-fields!
  "GPU twin of seed/prep-noise: the baked orientation field at tensor resolution,
   both ends of the Swirl dial. `perm` is gen/upload-perm!'s 512x1 texture.
   Returns [swirl1 swirl0] — the two textures gen's u_noiseTex / u_noiseSTex want."
  [ctx progs eigen flow perm w h W H]
  (let [mk (fn [swirl0?]
             (let [t (new-target w h)]
               (run-pass! ctx (:noise progs) t w h
                          [["u_eigen" eigen] ["u_flow" flow] ["u_permTex" perm]]
                          [["u_dim" :2f w h] ["u_src" :2f W H]
                           ["u_swirl0" :1f (if swirl0? 1.0 0.0)]])
               t))]
    [(mk false) (mk true)]))

;; --- source upload / readback (dev verification) -----------------------------

(defn upload-rgb!
  "Upload an image map's H*W*3 :pixels as a w×h RGBA32F texture (alpha 1)."
  [img]
  (let [H (long (:height img)) W (long (:width img))
        ^doubles px (:pixels img)
        n (* H W)
        ptr (ffi/alloc (* n 4 (ffi/sizeof :float)))]
    (dotimes [i n]
      (let [b (* 3 i) o (* i 4 4)]
        (ffi/write ptr :float o            (aget px b))
        (ffi/write ptr :float (+ o 4)      (aget px (+ b 1)))
        (ffi/write ptr :float (+ o 8)      (aget px (+ b 2)))
        (ffi/write ptr :float (+ o 12)     1.0)))
    (let [t (new-target W H)]
      (gl/gl-bind-texture gl/GL-TEXTURE-2D t)
      (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 gl/GL-RGBA32F (int W) (int H) 0
                          gl/GL-RGBA gl/GL-FLOAT ptr)
      (ffi/free ptr)
      t)))

(defn read-rgba
  "Read all four channels of a w×h RGBA32F texture in ONE glReadPixels, as
   [r g b a] ^doubles. Four read-channel calls would pull the same pixels four
   times over the FFI, which at ~8.6M floats/s is the expensive part."
  [ctx tex w h]
  (let [{:keys [fbo prev]} ctx
        n   (* (long w) (long h))
        ptr (ffi/alloc (* n 4 (ffi/sizeof :float)))
        out (mapv (fn [_] (double-array n)) (range 4))]
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
    (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0
                                  gl/GL-TEXTURE-2D tex 0)
    (gl/gl-read-pixels 0 0 (int w) (int h) gl/GL-RGBA gl/GL-FLOAT ptr)
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev)
    (dotimes [i n]
      (let [o (* i 16)]
        (dotimes [c 4]
          (aset ^doubles (nth out c) i (double (ffi/read ptr :float (+ o (* c 4))))))))
    (ffi/free ptr)
    out))

(defn read-channel
  "Read channel `c` (0..3) of a w×h RGBA32F texture into a fresh ^doubles of w*h.
   DEV ONLY — ~8.6M floats/s, so this is for verification, never the render path."
  [ctx tex w h c]
  (let [{:keys [fbo prev]} ctx
        n   (* (long w) (long h))
        ptr (ffi/alloc (* n 4 (ffi/sizeof :float)))
        out (double-array n)]
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
    (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0
                                  gl/GL-TEXTURE-2D tex 0)
    (gl/gl-read-pixels 0 0 (int w) (int h) gl/GL-RGBA gl/GL-FLOAT ptr)
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev)
    (dotimes [i n]
      (aset out i (double (ffi/read ptr :float (+ (* i 16) (* c 4))))))
    (ffi/free ptr)
    out))

(defn new-scratch [w h] (new-target w h))

;; --- the whole chain ---------------------------------------------------------

(defn heavy-radius
  "Window for the coverage tiers' colour source. Mirrors fields/heavy-radius —
   duplicated rather than required to keep this namespace off the CPU field
   builders, which pull in the whole analysis stack."
  [H] (max 6 (quot (long H) 80)))

(defn build-fields!
  "Every per-image field, built by passes and left in VRAM. Returns the same map
   gen/upload-fields! does, so generate! cannot tell the difference.

   The CPU twin is fields/prepare + gen/upload-fields!, which stays the reference
   the goldens pin and GA_PAINTER_GPU_VERIFY compares against.

   TWO readbacks survive, both once per image load: seed/layer-params reduces the
   placement maps to size the splat budget and re-runs on every slider drag, so
   those arrays have to live on the CPU. Everything else stays on the card."
  [ctx progs img perm]
  (let [H (long (:height img)) W (long (:width img))
        src    (upload-rgb! img)
        tensor (analyze! ctx progs src H W 768)
        Ht (:h tensor) Wt (:w tensor)
        pm     (placement-map! ctx progs src (:eigen tensor) H W Ht Wt 768 4)
        Hd (:h pm) Wd (:w pm)
        packed (new-target Wd Hd)
        _      (run-pass! ctx (:pack progs) packed Wd Hd
                          [["u_detail" (:detail pm)] ["u_sharp" (:sharp pm)]
                           ["u_edge" (:edge pm)] ["u_mid" (:mid pm)]]
                          [["u_dim" :2f Wd Hd]])
        [noise noise0] (noise-fields! ctx progs (:eigen tensor) (:flow tensor)
                                      perm Wt Ht W H)
        blur   (bilateral-blur! ctx progs src (new-target W H) W H 3)
        drift  (box-blur! ctx progs src (new-target W H) (new-scratch W H) W H 2)
        heavy  (dominant-blur! ctx progs src (new-target W H) W H (heavy-radius H))
        ;; the budget reduction's inputs — the only thing that comes back
        [dd ds de dm] (read-rgba ctx packed Wd Hd)
        [su _ _ _]    (read-rgba ctx (:subject pm) Wd Hd)]
    ;; NOT (:subject pm) — it is returned as :subject and outlives this call
    (free-textures! [(:eigen tensor) (:flow tensor) (:tensor tensor)
                     (:detail pm) (:sharp pm) (:mid pm) (:edge pm)])
    ;; the caller is mid-draw; give back the viewport the passes trampled
    (restore-viewport! ctx)
    {:detail packed :subject (:subject pm) :noise noise :noise-swirl0 noise0
     :blur blur :blur-drift drift :blur-heavy heavy :raw src :perm perm
     ;; placement-map is pre-normalized, so dmax is 1.0 by construction
     :dmap {:h Hd :w Wd :detail dd :sharp ds :edge de :mid dm :subject su
            :dmax 1.0 :src-h H :src-w W}
     :dmax 1.0
     :detail-dim [(double Hd) (double Wd)] :detail-src [(double H) (double W)]
     :noise-dim  [(double Ht) (double Wt)] :noise-src  [(double H) (double W)]
     :H H :W W}))
