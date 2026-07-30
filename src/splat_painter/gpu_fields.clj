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

(defn make-ctx
  "GL objects shared by every pass: one FBO to attach targets to, one empty VAO
   (core profile still requires a bound VAO for an attributeless draw), and the
   caller's framebuffer so passes can restore it. Call with a current context."
  []
  {:fbo  (gl/gen-one gl/gl-gen-framebuffers)
   :vao  (gl/gen-one gl/gl-gen-vertex-arrays)
   :prev (read-fbo-binding)})

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
  for (int d = -64; d <= 64; d++) {
    if (d < -r || d > r) continue;
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
  // minor-eigenvector angle: the direction the edge RUNS, not its normal
  float theta = 0.5 * atan(2.0*f, a - b) + 1.5707963267948966;
  float d = sqrt(((a-b)*0.5)*((a-b)*0.5) + f*f);
  float coh = clamp(2.0*d / (a + b + 1e-9), 0.0, 1.0);
  frag = vec4(theta, coh, a + b, 1.0);
}")

(defn build-programs
  "Compile the field-construction programs. Needs a current GL context. Returns
   nil if any shader fails, so a caller can report rather than draw garbage."
  []
  (let [ps {:box   (gl/make-program vs-src box-fs)
            :down  (gl/make-program vs-src down-fs)
            :sobel (gl/make-program vs-src sobel-fs)
            :eigen (gl/make-program vs-src eigen-fs)}]
    (when (every? some? (vals ps))
      (into {} (map (fn [[k v]] [k {:program v}])) ps))))

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
