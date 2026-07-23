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

(def ^:private max-levels 5)

;; --- generation program: vertex + geometry (transform feedback) --------------
(def ^:private vs-src
  "#version 330 core
flat out int v_id;
void main(){ v_id = gl_VertexID; }")

(def ^:private gs-src
  "#version 330 core
layout(points) in;
layout(points, max_vertices = 6) out;
flat in int v_id[];

// captured by transform feedback: 12 floats = 3 RGBA texels per splat, matching the
// render shaders' t0=(mean_x,mean_y,c00,c01) t1=(c11,r,g,b) t2=(alpha,0,0,0).
out vec4 o_a;
out vec4 o_b;
out vec4 o_c;

const int   ML      = 5;
const float MIN_COH = 0.28;
const int   SEGS    = 6;      // segments per fine-level brush stroke (seed/stroke-segs)

// per-level placement (finest-first slots), from seed/layer-params
uniform int   u_nlev;
uniform float u_ssz[ML];
uniform float u_sp[ML];
uniform float u_th[ML];
uniform int   u_nx[ML];
uniform int   u_ny[ML];
uniform int   u_off[ML];
uniform int   u_lvl[ML];
// scale-relative stroke behaviour per level slot (seed/layer-params: seg-count,
// step-frac, bend-frac, sharp-level?)
uniform int   u_segs[ML];
uniform float u_stepf[ML];
uniform float u_bendf[ML];
uniform int   u_sharp[ML];
uniform float u_warp;
uniform int   u_H;
uniform int   u_W;

// controls
uniform float u_stroke;
uniform float u_variation;
uniform float u_contrast;
uniform float u_detail;   // the Detail slider (deff scale)
uniform float u_curv;     // Curvature raw 0..1 — Perlin bend of the stroke trace

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
// the SHARP fine-band map (texel .g) — what the finest levels threshold against
float sharpAt(float x, float y){
  vec4 t = texelFetch(u_detailTex, fieldTexel(x, y, u_detailDim, u_detailSrc), 0);
  return u_dmax > 0.0 ? min(1.0, t.g / u_dmax) : 0.0;
}

// BILINEAR orientation-field sample (mirrors seed/sample-fields exactly: same
// continuous coord fx = x·dim/src, same floor/clamp). The texture stores cos2θ /
// sin2θ / coherence — double-angle components interpolate undirected orientations
// (0 ≡ π) smoothly, where nearest-sampled raw angles stair-step along contours.
// Returns (theta, coherence).
vec2 fieldsAt(float x, float y){
  float fx = clamp(x * u_noiseDim.x / u_noiseSrc.x, 0.0, u_noiseDim.x - 1.0);
  float fy = clamp(y * u_noiseDim.y / u_noiseSrc.y, 0.0, u_noiseDim.y - 1.0);
  int i0 = int(fx); int i1 = min(i0 + 1, int(u_noiseDim.x) - 1); float wx = fx - float(i0);
  int j0 = int(fy); int j1 = min(j0 + 1, int(u_noiseDim.y) - 1); float wy = fy - float(j0);
  vec3 v00 = texelFetch(u_noiseTex, ivec2(j0, i0), 0).xyz;
  vec3 v01 = texelFetch(u_noiseTex, ivec2(j1, i0), 0).xyz;
  vec3 v10 = texelFetch(u_noiseTex, ivec2(j0, i1), 0).xyz;
  vec3 v11 = texelFetch(u_noiseTex, ivec2(j1, i1), 0).xyz;
  vec3 b = mix(mix(v00, v01, wy), mix(v10, v11, wy), wx);
  return vec2(0.5 * atan(b.y, b.x), clamp(b.z, 0.0, 1.0));
}
vec3 sampleRGB(sampler2D tex, float x, float y){   // W×H, sample-arr nearest (int trunc)
  int xi = clamp(int(x), 0, u_H - 1);
  int yi = clamp(int(y), 0, u_W - 1);
  return texelFetch(tex, ivec2(yi, xi), 0).rgb;
}

// splat-record (mirror seed/splat-record) + emit one captured record. `alpha` is the
// stroke taper (1.0 for base fills, fading toward a fine stroke's tail).
void emitSplat(float px, float py, float csz, float D, float sn, float tn, float alpha){
  vec2  tc    = fieldsAt(px, py);
  float theta = tc.x, coh0 = tc.y;
  vec3  blur = sampleRGB(u_blurTex, px, py);
  vec3  raw  = sampleRGB(u_rawTex,  px, py);
  float coh = MIN_COH + (1.0 - MIN_COH) * coh0;
  float e   = 1.0 + u_stroke * coh * (0.25 + 0.75 * D);
  float se  = sqrt(e);
  float s0  = csz * (1.0 + u_variation * 0.5 * (2.0 * sn));
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
  float tone = 1.0 + u_variation * 0.15 * (2.0 * tn);
  vec3 color = clamp(colorAc * tone, 0.0, 1.0);
  o_a = vec4(px, py, c00, c01);
  o_b = vec4(c11, color.r, color.g, color.b);
  o_c = vec4(alpha, 0.0, 0.0, 0.0);
  EmitVertex();
  EndPrimitive();
}

void main(){
  int v = v_id[0];
  // decode candidate index -> finest-first level slot k. CONSTANT loop bound (ML) so
  // the compiler can fully unroll: with a uniform bound (u_nlev) + break, Apple's GLSL
  // mis-executed the last slot in this geometry shader — the base level (slot 4 at five
  // levels) emitted NOTHING, which truncated the underpainting and left gaps.
  int k = 0;
  for (int m = 0; m < ML; m++) {
    if (m < u_nlev && v >= u_off[m] && v < u_off[m] + u_nx[m] * u_ny[m]) { k = m; break; }
  }
  int local = v - u_off[k];
  int i = local / u_ny[k];
  int j = local - i * u_ny[k];
  int lvl = u_lvl[k];
  float sp = u_sp[k], ssz = u_ssz[k], th = u_th[k];
  float cx = (float(i) + 0.5) * sp;
  float cy = (float(j) + 0.5) * sp;
  // each level reads the map matched to ITS scale (mirror of seed/layered-means)
  float dv = (u_sharp[k] == 1) ? sharpAt(cx, cy) : detailAt(cx, cy);
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

  // per-seed size/tone jitter is hashed (independent per stroke), mirroring
  // seed/layered-means.
  float snoise = hash01(i*31 + lvl, j, 11) - 0.5;
  float tnoise = hash01(i*37 + lvl, j, 13) - 0.5;

  if (lvl == 0) {                                 // base fill: one full-alpha splat
    emitSplat(x2, y2, ssz, D, snoise, tnoise, 1.0);
    return;
  }

  // fine level: TRACE A BRUSH STROKE (mirror seed/stroke-segments) — SEGS segments
  // stepped along the orientation field (the edge tangent), sign-continuous, bent by
  // smooth Perlin noise scaled by Curvature, size+alpha tapering toward the tail.
  float dirsign = hash01(i*41 + lvl, j, 17) < 0.5 ? 1.0 : -1.0;
  int   segs  = u_segs[k];                       // scale-relative stroke behaviour:
  float stepf = u_stepf[k];                      // broad levels stroke long and curl,
  float bendf = u_bendf[k];                      // fine levels make short precise marks
  float px = x2, py = y2, dxp = 0.0, dyp = 0.0;
  for (int q = 0; q < SEGS; q++) {
    if (q >= segs) break;
    float tt = float(q) / float(segs - 1);
    float sz = ssz * (1.0 - 0.45 * tt * sqrt(tt));   // width tapers to the tip
    float al = 1.0 - 0.65 * tt * tt;                 // …and the paint thins out
    emitSplat(px, py, sz, D, snoise, tnoise, al);
    vec2  tc  = fieldsAt(px, py);
    float bend = u_curv * 0.9 * bendf * (noise2(0.05*px, 0.05*py) - 0.5);
    float cb = cos(bend), sb = sin(bend);
    float dx0 = cos(tc.x), dy0 = sin(tc.x);
    float sgn = (q == 0) ? dirsign : ((dx0*dxp + dy0*dyp) < 0.0 ? -1.0 : 1.0);
    float dx1 = sgn*dx0, dy1 = sgn*dy0;
    float dx = cb*dx1 - sb*dy1, dy = sb*dx1 + cb*dy1;
    float L = ssz * stepf;
    px = clamp(px + L*dx, 0.0, float(u_H - 1));
    py = clamp(py + L*dy, 0.0, float(u_W - 1));
    dxp = dx; dyp = dy;
  }
}")

(def ^:private gen-uniform-names
  ["u_nlev" "u_warp" "u_H" "u_W" "u_stroke" "u_variation" "u_contrast" "u_detail" "u_curv"
   "u_ssz" "u_sp" "u_th" "u_nx" "u_ny" "u_off" "u_lvl"
   "u_segs" "u_stepf" "u_bendf" "u_sharp"
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
  (when-let [prog (gl/make-tf-program vs-src gs-src ["o_a" "o_b" "o_c"])]
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
        ^doubles ds (or (:sharp dmap) dd)
        nf (:noise-fields img)
        Hn (long (:h nf)) Wn (long (:w nf))
        ^doubles c2 (:c2 nf) ^doubles s2 (:s2 nf) ^doubles co (:coherence nf)
        detail-t (new-tex) noise-t (new-tex) blur-t (new-tex) raw-t (new-tex)]
    ;; .r = aggregate placement map, .g = sharp fine-band map (finest levels read it)
    (upload-rgba! detail-t Wd Hd (rgba-ptr Hd Wd (fn [i] [(aget dd i) (aget ds i) 0.0 1.0])))
    ;; orientation as double-angle components (cos2θ, sin2θ) + coherence — the GS
    ;; bilinearly blends the components (fieldsAt), never the raw angle.
    (upload-rgba! noise-t  Wn Hn (rgba-ptr Hn Wn (fn [i] [(aget c2 i) (aget s2 i) (aget co i) 0.0])))
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
   {:mean [x y] :cov [c00 c01 c01 c11] :color [r g b] :alpha a} — the same shape as the CPU
   splat-field, for numerical verification against the golden reference. `tf-buf` must
   be bound to GL_TRANSFORM_FEEDBACK_BUFFER."
  [tf-buf n]
  (let [nf  (* n 12)
        ptr (ffi/alloc (* nf (ffi/sizeof :float)))]
    (gl/gl-bind-buffer gl/GL-TRANSFORM-FEEDBACK-BUFFER tf-buf)
    (gl/gl-get-buffer-sub-data gl/GL-TRANSFORM-FEEDBACK-BUFFER 0
                               (* nf (ffi/sizeof :float)) ptr)
    (let [fl (gl/read-floats ptr nf)]
      (ffi/free ptr)
      (mapv (fn [i]
              (let [b (* i 12)
                    mx (nth fl b) my (nth fl (+ b 1))
                    c00 (nth fl (+ b 2)) c01 (nth fl (+ b 3)) c11 (nth fl (+ b 4))]
                {:mean [mx my] :cov [c00 c01 c01 c11]
                 :color [(nth fl (+ b 5)) (nth fl (+ b 6)) (nth fl (+ b 7))]
                 :alpha (nth fl (+ b 8))}))
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
        sgs (pad (map :segs levels) 1)
        stf (pad (map :stepf levels) 1.0)
        bnf (pad (map :bendf levels) 1.0)
        shp (pad (map (fn [l] (if (:sharp? l) 1 0)) levels) 0)
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
    (gl/gl-uniform-1f (:u_curv locs) (double curvature))
    (set-1fv! (:u_ssz locs) ssz)
    (set-1fv! (:u_sp locs) sp)
    (set-1fv! (:u_th locs) th)
    (set-1iv! (:u_nx locs) nx)
    (set-1iv! (:u_ny locs) ny)
    (set-1iv! (:u_off locs) off)
    (set-1iv! (:u_lvl locs) lvl)
    (set-1iv! (:u_segs locs) sgs)
    (set-1fv! (:u_stepf locs) stf)
    (set-1fv! (:u_bendf locs) bnf)
    (set-1iv! (:u_sharp locs) shp)
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
