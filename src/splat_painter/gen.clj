(ns splat-painter.gen
  "GPU splat GENERATION — the whole splat-field built on the GPU, replacing the CPU
   splat-painter.seed/splat-field per render. A vertex+geometry transform-feedback
   program draws one GL_POINT per candidate cell (splat-painter.seed/layer-params gives
   the per-level grid), the geometry shader threshold-tests each cell and, for the
   survivors, jitters + Perlin-warps the centre, samples the precomputed fields, runs
   the splat-record math, and EmitVertex()s the 8-float splat record. Transform
   feedback CAPTURES only the survivors into a buffer (variable-count compaction with
   no compute/SSBO — see splat-painter.tf-smoke), and a query reports the count. That
   buffer is then bound as a samplerBuffer and consumed by the render shader with no
   readback. The GLSL mirrors splat-painter.seed/splat-record + layered-means and
   splat-painter.noise (hash01 + Perlin), verified against the CPU golden (visually +
   count) — the CPU path stays the tested reference.

   Field textures (uploaded once per image load, all RGBA32F, texelFetch nearest):
     u_detail  W_d×H_d  .r = wavelet detail; normalized by u_dmax
     u_noise   W_n×H_n  .rgba = (theta, coherence, snoise, tnoise) from prep-noise
     u_blur    W×H      .rgb = smooth base colour
     u_raw     W×H      .rgb = crisp pixel colour
     u_perm    512×1    .r = Perlin permutation (0..255)
   Reduced-res maps (detail/noise) carry their src dims so the shader maps a full-image
   (x,y) into them exactly like the CPU (round(x·dim/src))."
  (:require [glimmer-gl.gl :as gl]
            [splat-painter.seed :as seed]
            [splat-painter.noise :as noise]
            [jolt.ffi :as ffi]))

(def ^:private max-levels 4)

;; --- generation program: vertex + geometry (transform feedback) --------------
(def ^:private vs-src
  "#version 330 core
flat out int v_id;
void main(){ v_id = gl_VertexID; }")

(def ^:private gs-src
  "#version 330 core
layout(points) in;
layout(points, max_vertices = 1) out;
flat in int v_id[];

// captured by transform feedback: 8 floats = 2 RGBA texels per splat, matching the
// render shader's t0=(mean_x,mean_y,c00,c01) t1=(c11,r,g,b).
out vec4 o_a;
out vec4 o_b;

const int   ML      = 4;
const float MIN_COH = 0.28;

// per-level placement (finest-first slots), from seed/layer-params
uniform int   u_nlev;
uniform float u_ssz[ML];
uniform float u_sp[ML];
uniform float u_th[ML];
uniform int   u_nx[ML];
uniform int   u_ny[ML];
uniform int   u_off[ML];
uniform int   u_lvl[ML];
uniform float u_warp;
uniform int   u_H;
uniform int   u_W;

// controls
uniform float u_stroke;
uniform float u_variation;
uniform float u_contrast;
uniform float u_detail;   // the Detail slider (deff scale)

// fields
uniform sampler2D u_detailTex;
uniform float u_dmax;
uniform vec2  u_detailDim;   // (H_d, W_d)
uniform vec2  u_detailSrc;   // (src_h, src_w) = image (H, W)
uniform sampler2D u_noiseTex;
uniform vec2  u_noiseDim;
uniform vec2  u_noiseSrc;
uniform sampler2D u_blurTex;  // W×H
uniform sampler2D u_rawTex;   // W×H
uniform sampler2D u_permTex;  // 512×1

// --- helpers (mirror splat-painter.noise / seed) ---------------------------
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
float noise2(float x, float y){ return noise3(x, y, 0.0); }

float hash01(int a, int b, int salt){
  uint h = uint(a)*73856093u + uint(b)*19349663u + uint(salt)*83492791u; // wraps mod 2^32
  return float(h) / 4294967296.0;
}

// nearest-map a full-image (x,y) into a reduced field grid (dim=(rows,cols),
// src=(src_h,src_w)), matching CPU round()=floor(v+0.5), then texelFetch(col,row).
ivec2 fieldTexel(float x, float y, vec2 dim, vec2 src){
  int xi = int(floor(x * dim.x / src.x + 0.5));
  int yi = int(floor(y * dim.y / src.y + 0.5));
  xi = clamp(xi, 0, int(dim.x) - 1);
  yi = clamp(yi, 0, int(dim.y) - 1);
  return ivec2(yi, xi);
}
float detailAt(float x, float y){
  vec4 t = texelFetch(u_detailTex, fieldTexel(x, y, u_detailDim, u_detailSrc), 0);
  return u_dmax > 0.0 ? min(1.0, t.r / u_dmax) : 0.0;
}
vec3 sampleRGB(sampler2D tex, float x, float y){   // W×H, sample-arr nearest (int trunc)
  int xi = clamp(int(x), 0, u_H - 1);
  int yi = clamp(int(y), 0, u_W - 1);
  return texelFetch(tex, ivec2(yi, xi), 0).rgb;
}

void main(){
  int v = v_id[0];
  // decode candidate index -> finest-first level slot k
  int k = 0;
  for (int m = 0; m < u_nlev; m++) {
    if (v >= u_off[m] && v < u_off[m] + u_nx[m] * u_ny[m]) { k = m; break; }
  }
  int local = v - u_off[k];
  int i = local / u_ny[k];
  int j = local - i * u_ny[k];
  int lvl = u_lvl[k];
  float sp = u_sp[k], ssz = u_ssz[k], th = u_th[k];
  float cx = (float(i) + 0.5) * sp;
  float cy = (float(j) + 0.5) * sp;
  float dv = detailAt(cx, cy);
  if (lvl > 0 && dv < th) return;                 // not detailed enough -> discard

  float jx = sp * 0.45 * (hash01(i*137 + lvl, j, 3) - 0.5);
  float jy = sp * 0.45 * (hash01(i*149 + lvl, j, 7) - 0.5);
  float x = cx + jx, y = cy + jy;
  float D = min(1.0, u_detail * dv * 2.2);
  float aw = u_warp * (1.0 - D) * ssz;
  float x2 = (aw < 0.2) ? x : x + aw * noise2(0.06*x, 0.06*y);
  float y2 = (aw < 0.2) ? y : y + aw * noise2(41.3 + 0.06*x, 17.9 + 0.06*y);
  x2 = clamp(x2, 0.0, float(u_H - 1));
  y2 = clamp(y2, 0.0, float(u_W - 1));

  // sample precomputed fields at the final centre
  vec4  nf   = texelFetch(u_noiseTex, fieldTexel(x2, y2, u_noiseDim, u_noiseSrc), 0);
  float theta = nf.x, coh0 = nf.y, snoise = nf.z, tnoise = nf.w;
  vec3  blur = sampleRGB(u_blurTex, x2, y2);
  vec3  raw  = sampleRGB(u_rawTex,  x2, y2);

  // splat-record (mirror seed/splat-record) ---------------------------------
  float coh = MIN_COH + (1.0 - MIN_COH) * coh0;
  float e   = 1.0 + u_stroke * coh * (0.25 + 0.75 * D);
  float se  = sqrt(e);
  float s0  = ssz * (1.0 + u_variation * 0.5 * (2.0 * snoise));
  float sx  = s0 * se;
  float sy  = s0 / se;
  float sx2 = sx*sx, sy2 = sy*sy;
  float c = cos(theta), s = sin(theta);
  float c00 = sx2*c*c + sy2*s*s;
  float c01 = (sx2 - sy2)*c*s;
  float c11 = sx2*s*s + sy2*c*c;
  float t = clamp(0.55 + 0.45 * max(coh0, D), 0.0, 1.0);
  vec3 color0 = mix(blur, raw, t);
  vec3 colorAc = (u_contrast == 1.0) ? color0 : clamp((color0 - 0.5)*u_contrast + 0.5, 0.0, 1.0);
  float tone = 1.0 + u_variation * 0.15 * (2.0 * tnoise);
  vec3 color = clamp(colorAc * tone, 0.0, 1.0);

  o_a = vec4(x2, y2, c00, c01);
  o_b = vec4(c11, color.r, color.g, color.b);
  EmitVertex();
  EndPrimitive();
}")

(def ^:private gen-uniform-names
  ["u_nlev" "u_warp" "u_H" "u_W" "u_stroke" "u_variation" "u_contrast" "u_detail"
   "u_ssz" "u_sp" "u_th" "u_nx" "u_ny" "u_off" "u_lvl"
   "u_detailTex" "u_dmax" "u_detailDim" "u_detailSrc"
   "u_noiseTex" "u_noiseDim" "u_noiseSrc" "u_blurTex" "u_rawTex" "u_permTex"])

(defn sources
  "Return {:vs-src :gs-src} — pure, no GL context (for headless inspection/tests)."
  []
  {:vs-src vs-src :gs-src gs-src})

(defn build-gen-program
  "Compile + link the VS+GS transform-feedback generation program, capturing
   o_a,o_b interleaved (8 floats/splat). Returns {:program :locs} or nil."
  []
  (when-let [prog (gl/make-tf-program vs-src gs-src ["o_a" "o_b"])]
    {:program prog
     :locs (into {} (map (fn [n] [(keyword n) (gl/gl-get-uniform-location prog n)]))
                 gen-uniform-names)}))

;; --- field texture upload (once per image) -----------------------------------
(defn- new-tex []
  (let [t (gl/gen-one gl/gl-gen-textures)]
    (gl/gl-bind-texture gl/GL-TEXTURE-2D t)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-S gl/GL-CLAMP-TO-EDGE)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-T gl/GL-CLAMP-TO-EDGE)
    t))

(defn- upload-rgba! [tex tw th ^"[D" ptr]
  (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
  (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 gl/GL-RGBA32F (int tw) (int th) 0
                      gl/GL-RGBA gl/GL-FLOAT ptr)
  (ffi/free ptr))

;; Build an RGBA32F buffer (W wide, H tall; texel (row=xi,col=yi)) directly from
;; source double-arrays with a getter that returns [r g b a] for flat index xi*W+yi.
(defn- rgba-ptr [H W getter]
  (let [n (* (long H) (long W))
        ptr (ffi/alloc (* n 4 (ffi/sizeof :float)))]
    (dotimes [idx n]
      (let [o (* idx 16)                     ; 4 floats × 4 bytes
            [r g b a] (getter idx)]
        (ffi/write ptr :float o        (double r))
        (ffi/write ptr :float (+ o 4)  (double g))
        (ffi/write ptr :float (+ o 8)  (double b))
        (ffi/write ptr :float (+ o 12) (double a))))
    ptr))

(defn upload-fields!
  "Upload the per-image field textures. `img` must carry :structure/:detail/:blur/
   :noise-fields (the same caches the CPU path uses). Returns a map of texture ids +
   dims for the generation uniforms. `perm-tex` is uploaded once and reused."
  [img perm-tex]
  (let [H (long (:height img)) W (long (:width img))
        ^doubles raw (:pixels img)
        ^doubles blur (or (:blur img) (:pixels img))
        dmap (:detail img)
        Hd (long (:h dmap)) Wd (long (:w dmap))
        ^doubles dd (:detail dmap)
        nf (:noise-fields img)
        Hn (long (:h nf)) Wn (long (:w nf))
        ^doubles th (:theta nf) ^doubles co (:coherence nf)
        ^doubles sn (:snoise nf) ^doubles tn (:tnoise nf)
        detail-t (new-tex) noise-t (new-tex) blur-t (new-tex) raw-t (new-tex)]
    (upload-rgba! detail-t Wd Hd (rgba-ptr Hd Wd (fn [i] [(aget dd i) 0.0 0.0 1.0])))
    (upload-rgba! noise-t  Wn Hn (rgba-ptr Hn Wn (fn [i] [(aget th i) (aget co i) (aget sn i) (aget tn i)])))
    (upload-rgba! blur-t   W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget blur b) (aget blur (+ b 1)) (aget blur (+ b 2)) 1.0]))))
    (upload-rgba! raw-t    W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget raw b) (aget raw (+ b 1)) (aget raw (+ b 2)) 1.0]))))
    {:detail detail-t :noise noise-t :blur blur-t :raw raw-t :perm perm-tex
     :dmap dmap                              ; the CPU detail map, for layer-params' budget
     :dmax (double (:dmax dmap))
     :detail-dim [(double Hd) (double Wd)] :detail-src [(double H) (double W)]
     :noise-dim  [(double Hn) (double Wn)] :noise-src  [(double (or (:src-h nf) H)) (double (or (:src-w nf) W))]
     :H H :W W}))

(defn upload-perm!
  "Upload Ken Perlin's 512 permutation entries as a 512×1 RGBA32F texture (.r = value)."
  []
  (let [^ints p noise/perm512
        t (new-tex)]
    (upload-rgba! t 512 1 (rgba-ptr 1 512 (fn [i] [(double (aget p i)) 0.0 0.0 1.0])))
    t))

;; --- run generation ----------------------------------------------------------
(defn- set-1fv! [loc xs]
  (let [ptr (gl/write-floats xs)] (gl/gl-uniform-1fv loc (count xs) ptr) (ffi/free ptr)))
(defn- set-1iv! [loc xs]
  (let [ptr (gl/write-ints xs)] (gl/gl-uniform-1iv loc (count xs) ptr) (ffi/free ptr)))

(defn sig-range
  "Approximate the field's stdev range (det^0.25 = s0 = ssz·(1±0.5·variation)) from the
   level ssz values — the GPU path has no CPU-side splats to reduce over, and this only
   feeds the size→hardness smoothstep easing."
  [levels variation]
  (let [sszs (map :ssz levels)
        v (double variation)]
    [(* (reduce min sszs) (max 0.05 (- 1.0 (* 0.5 v))))
     (* (reduce max sszs) (+ 1.0 (* 0.5 v)))]))

(defn read-splats
  "Read `n` generated splats back from the transform-feedback buffer as a vector of
   {:mean [x y] :cov [c00 c01 c01 c11] :color [r g b]} — the same shape as the CPU
   splat-field, for numerical verification against the golden reference. `tf-buf` must
   be bound to GL_TRANSFORM_FEEDBACK_BUFFER."
  [tf-buf n]
  (let [nf  (* n 8)
        ptr (ffi/alloc (* nf (ffi/sizeof :float)))]
    (gl/gl-bind-buffer gl/GL-TRANSFORM-FEEDBACK-BUFFER tf-buf)
    (gl/gl-get-buffer-sub-data gl/GL-TRANSFORM-FEEDBACK-BUFFER 0
                               (* nf (ffi/sizeof :float)) ptr)
    (let [fl (gl/read-floats ptr nf)]
      (ffi/free ptr)
      (mapv (fn [i]
              (let [b (* i 8)
                    mx (nth fl b) my (nth fl (+ b 1))
                    c00 (nth fl (+ b 2)) c01 (nth fl (+ b 3)) c11 (nth fl (+ b 4))]
                {:mean [mx my] :cov [c00 c01 c01 c11]
                 :color [(nth fl (+ b 5)) (nth fl (+ b 6)) (nth fl (+ b 7))]}))
            (range n)))))

(defn generate!
  "Run the generation pass into `tf-buf` (a GL buffer bound to binding point 0). Sets
   all uniforms from `controls` + `fields`, draws `total` candidate points through the
   VS+GS with rasterizer discard, and returns {:count n :sig-min :sig-max}. Needs a
   current context, a bound (complete) framebuffer, the gen program, a bound VAO, and
   a query object `q`."
  [gen fields controls tf-buf q vao {:keys [height width]}]
  (let [{:keys [program locs]} gen
        {:keys [count size stroke detail variation curvature contrast]} controls
        H (long height) W (long width)
        params (seed/layer-params (:dmap fields) detail size variation curvature count H W)
        {:keys [nlev warp levels total]} params
        ;; finest-first level arrays, padded to max-levels
        pad (fn [xs d] (vec (take max-levels (concat xs (repeat d)))))
        ssz (pad (map :ssz levels) 0.0)
        sp  (pad (map :sp levels) 1.0)
        th  (pad (map :th levels) -1.0)
        nx  (pad (map :nx levels) 0)
        ny  (pad (map :ny levels) 1)
        off (pad (map :offset levels) 0)
        lvl (pad (map :lvl levels) 0)
        [sig-min sig-max] (sig-range levels variation)]
    (gl/gl-use-program program)
    ;; per-level + controls
    (gl/gl-uniform-1i (:u_nlev locs) (int nlev))
    (gl/gl-uniform-1f (:u_warp locs) (double warp))
    (gl/gl-uniform-1i (:u_H locs) (int H))
    (gl/gl-uniform-1i (:u_W locs) (int W))
    (gl/gl-uniform-1f (:u_stroke locs) (double stroke))
    (gl/gl-uniform-1f (:u_variation locs) (double variation))
    (gl/gl-uniform-1f (:u_contrast locs) (double contrast))
    (gl/gl-uniform-1f (:u_detail locs) (double detail))
    (set-1fv! (:u_ssz locs) ssz)
    (set-1fv! (:u_sp locs) sp)
    (set-1fv! (:u_th locs) th)
    (set-1iv! (:u_nx locs) nx)
    (set-1iv! (:u_ny locs) ny)
    (set-1iv! (:u_off locs) off)
    (set-1iv! (:u_lvl locs) lvl)
    ;; field textures on units 1..5 (unit 0 is the render splat buffer)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 1)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:detail fields)) (gl/gl-uniform-1i (:u_detailTex locs) 1)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 2)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:noise fields))  (gl/gl-uniform-1i (:u_noiseTex locs) 2)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 3)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:blur fields))   (gl/gl-uniform-1i (:u_blurTex locs) 3)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 4)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:raw fields))    (gl/gl-uniform-1i (:u_rawTex locs) 4)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 5)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:perm fields))   (gl/gl-uniform-1i (:u_permTex locs) 5)
    (gl/gl-uniform-1f (:u_dmax locs) (double (:dmax fields)))
    (gl/gl-uniform-2f (:u_detailDim locs) (double (nth (:detail-dim fields) 0)) (double (nth (:detail-dim fields) 1)))
    (gl/gl-uniform-2f (:u_detailSrc locs) (double (nth (:detail-src fields) 0)) (double (nth (:detail-src fields) 1)))
    (gl/gl-uniform-2f (:u_noiseDim locs) (double (nth (:noise-dim fields) 0)) (double (nth (:noise-dim fields) 1)))
    (gl/gl-uniform-2f (:u_noiseSrc locs) (double (nth (:noise-src fields) 0)) (double (nth (:noise-src fields) 1)))
    ;; capture survivors via transform feedback
    (gl/gl-bind-vertex-array vao)
    (gl/gl-bind-buffer-base gl/GL-TRANSFORM-FEEDBACK-BUFFER 0 tf-buf)
    (gl/gl-enable gl/GL-RASTERIZER-DISCARD)
    (gl/gl-begin-query gl/GL-TRANSFORM-FEEDBACK-PRIMITIVES-WRITTEN q)
    (gl/gl-begin-transform-feedback gl/GL-POINTS)
    (gl/gl-draw-arrays gl/GL-POINTS 0 (int total))
    (gl/gl-end-transform-feedback)
    (gl/gl-end-query gl/GL-TRANSFORM-FEEDBACK-PRIMITIVES-WRITTEN)
    (gl/gl-disable gl/GL-RASTERIZER-DISCARD)
    (let [n (gl/get-query-object-uiv q gl/GL-QUERY-RESULT)]
      {:count n :sig-min sig-min :sig-max sig-max :total total})))
