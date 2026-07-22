(ns splat-painter.structure
  "Per-pixel 2D structure-tensor field for an image.

  Sobel gradients → J = [[jxx jxy][jxy jyy]] per pixel, box-blurred over a
  local neighbourhood. The minor-eigenvector direction of J is the local edge
  (isophote / brushstroke) direction — perpendicular to the gradient.

  Orientation follows the splat-painter.gaussian convention:
    x = row index (0..H-1, increasing downward)
    y = col index (0..W-1, increasing rightward)
    θ = 0 means the major axis is along the x/row direction.

  All per-pixel buffers are Java double-arrays for speed on ~1e6-pixel images.")

;; ---------------------------------------------------------------------------
;; helpers
;; ---------------------------------------------------------------------------

(defn- clamp [v lo hi]
  (max lo (min hi v)))

(defn- round-int [v]
  (int (Math/round (double v))))

;; ---------------------------------------------------------------------------
;; luma
;; ---------------------------------------------------------------------------

(defn luma
  "Grayscale luminance (0..1) from an RGB image. Returns a ^doubles array
   of length H*W using standard BT.601 weights."
  [image]
  (let [H (:height image)
        W (:width image)
        ^doubles pix (:pixels image)
        n (* H W)
        L (double-array n)]
    (dotimes [x H]
      (let [row-off (* x W)]
        (dotimes [y W]
          (let [base (* 3 (+ row-off y))
                r (aget pix base)
                g (aget pix (inc base))
                b (aget pix (+ 2 base))]
            (aset L (+ row-off y)
                  (+ (* 0.299 r) (* 0.587 g) (* 0.114 b)))))))
    L))

;; ---------------------------------------------------------------------------
;; gradient-field
;; ---------------------------------------------------------------------------

(defn gradient-field
  "3×3 Sobel gradients (edge-replicate) from an image.
   Returns {:h H :w W :gx ^doubles :gy ^doubles} each length H*W.
   gx = d/dx (derivative across rows), gy = d/dy (derivative across cols)."
  [image]
  (let [^doubles L  (luma image)
        H  (:height image)
        W  (:width image)
        ^doubles gx (double-array (* H W))
        ^doubles gy (double-array (* H W))]
    (dotimes [x H]
      (dotimes [y W]
        (let [idx (+ (* x W) y)
              ;; clamped neighbour indices
              xm1 (clamp (dec x) 0 (dec H))
              xp1 (clamp (inc x) 0 (dec H))
              ym1 (clamp (dec y) 0 (dec W))
              yp1 (clamp (inc y) 0 (dec W))
              ;; luma values at 9 neighbours
              L00 (aget L (+ (* xm1 W) ym1))
              L01 (aget L (+ (* xm1 W) y))
              L02 (aget L (+ (* xm1 W) yp1))
              L10 (aget L (+ (* x   W) ym1))
              L11 (aget L (+ (* x   W) y))
              L12 (aget L (+ (* x   W) yp1))
              L20 (aget L (+ (* xp1 W) ym1))
              L21 (aget L (+ (* xp1 W) y))
              L22 (aget L (+ (* xp1 W) yp1))]
          ;; Sobel x: d/dx (row derivative)
          ;; kx: row-1=[-1 -2 -1], row=[0 0 0], row+1=[1 2 1]
          (aset gx idx (- (+ L20 (* 2.0 L21) L22)
                          (+ L00 (* 2.0 L01) L02)))
          ;; Sobel y: d/dy (col derivative)
          ;; ky: row-1=[-1 0 1], row=[-2 0 2], row+1=[-1 0 1]
          (aset gy idx (+ (- L00) L02
                          (* -2.0 L10) (* 2.0 L12)
                          (- L20) L22)))))
    {:h H :w W :gx gx :gy gy}))

;; ---------------------------------------------------------------------------
;; structure-tensor (separable box-blur)
;; ---------------------------------------------------------------------------

(defn- box-blur-2d
  "Separable 2D box-blur of `src` (H×W) with `radius`, edge-replicate.
   Returns a new ^doubles array."
  [^doubles src H W radius]
  ;; ^doubles src/tmp/dst + a double divisor `wd` let the sliding sums read via the
  ;; unboxed flvector path (jolt-flaget) and the +/-// lower to fl+/fl-/fl/.
  (let [wd (double (inc (* 2 radius)))
        ^doubles tmp (double-array (* H W))]
    ;; horizontal pass: blur each row
    (dotimes [x H]
      (let [row-off (* x W)
            sum-init (loop [dy (- radius) s 0.0]
                       (if (> dy radius)
                         s
                         (let [cy (clamp dy 0 (dec W))]
                           (recur (inc dy) (+ s (aget src (+ row-off cy)))))))]
        (aset tmp (+ row-off 0) (/ sum-init wd))
        (loop [y 1 y-sum sum-init]
          (when (< y W)
            (let [drop-y (clamp (- y radius 1) 0 (dec W))
                  add-y  (clamp (+ y radius) 0 (dec W))
                  new-sum (+ y-sum
                             (aget src (+ row-off add-y))
                             (- (aget src (+ row-off drop-y))))]
              (aset tmp (+ row-off y) (/ new-sum wd))
              (recur (inc y) new-sum))))))
    ;; vertical pass: blur each column of tmp into dst
    (let [^doubles dst (double-array (* H W))]
      (dotimes [y W]
        (let [sum-init (loop [dx (- radius) s 0.0]
                         (if (> dx radius)
                           s
                           (let [cx (clamp dx 0 (dec H))]
                             (recur (inc dx) (+ s (aget tmp (+ (* cx W) y)))))))]
          (aset dst (+ (* 0 W) y) (/ sum-init wd))
          (loop [x 1 v-sum sum-init]
            (when (< x H)
              (let [drop-x (clamp (- x radius 1) 0 (dec H))
                    add-x  (clamp (+ x radius) 0 (dec H))
                    new-sum (+ v-sum
                               (aget tmp (+ (* add-x W) y))
                               (- (aget tmp (+ (* drop-x W) y))))]
                (aset dst (+ (* x W) y) (/ new-sum wd))
                (recur (inc x) new-sum))))))
      dst)))

(defn structure-tensor
  "From gradient field, compute the 2×2 structure tensor J = [[jxx jxy][jxy jyy]]
   per pixel, box-blurred (separable) over a (2*radius+1)×(2*radius+1) window.
   Returns {:h H :w W :jxx ^doubles :jyy ^doubles :jxy ^doubles}."
  [gfield radius]
  (let [H (:h gfield) W (:w gfield)
        ^doubles gx (:gx gfield) ^doubles gy (:gy gfield)
        n (* H W)
        ^doubles jxx (double-array n)
        ^doubles jyy (double-array n)
        ^doubles jxy (double-array n)]
    (dotimes [i n]
      (let [a (aget gx i) b (aget gy i)]
        (aset jxx i (* a a))
        (aset jyy i (* b b))
        (aset jxy i (* a b))))
    {:h H :w W
     :jxx (box-blur-2d jxx H W radius)
     :jyy (box-blur-2d jyy H W radius)
     :jxy (box-blur-2d jxy H W radius)}))

;; ---------------------------------------------------------------------------
;; blur-image — one-time average-colour precompute
;; ---------------------------------------------------------------------------

(defn box-blur
  "Public separable box-blur of a single-channel H×W ^doubles array (edge-replicate)."
  [arr H W radius]
  (box-blur-2d arr H W radius))

(defn blur-image
  "Box-blur an RGB image's pixels (flat H*W*3 doubles) with `radius`, returning a
   new flat H*W*3 double-array. Computed once per image load so the seed can read
   a smooth average colour with a single lookup instead of a per-splat window
   scan + sort (the per-render cost that made the sliders lag)."
  [image radius]
  (let [H (:height image) W (:width image)
        ^doubles px (:pixels image)
        n (* H W)
        chan (fn [off]
               (let [a (double-array n)]
                 (dotimes [i n] (aset a i (aget px (+ (* 3 i) off))))
                 (box-blur-2d a H W radius)))
        r (chan 0) g (chan 1) b (chan 2)
        out (double-array (* n 3))]
    (dotimes [i n]
      (aset out (* 3 i)       (aget r i))
      (aset out (+ (* 3 i) 1) (aget g i))
      (aset out (+ (* 3 i) 2) (aget b i)))
    out))

;; ---------------------------------------------------------------------------
;; analyze
;; ---------------------------------------------------------------------------

(defn- downsample
  "Nearest-neighbour downscale of `image` to `sh` rows × `sw` cols. Returns a
   pseudo-image {:height :width :channels :pixels} the gradient pass can consume.
   Orientation varies slowly, so a coarse tensor is plenty — and it's ~(H/sh)²
   cheaper to build than a full-resolution one (jolt has no primitive-array
   fast path, so the win is purely doing fewer ops)."
  [image sh sw]
  (let [H (:height image) W (:width image)
        ^doubles px (:pixels image)
        ^doubles out (double-array (* sh sw 3))
        rx (/ (double H) sh) ry (/ (double W) sw)]
    (dotimes [r sh]
      (let [sr (min (dec H) (long (* r rx)))]
        (dotimes [c sw]
          (let [sc (min (dec W) (long (* c ry)))
                sbase (* 3 (+ (* sr W) sc))
                dbase (* 3 (+ (* r sw) c))]
            (aset out dbase       (aget px sbase))
            (aset out (+ dbase 1) (aget px (+ sbase 1)))
            (aset out (+ dbase 2) (aget px (+ sbase 2)))))))
    {:height sh :width sw :channels 3 :pixels out}))

(defn luma-of
  "Reduced-resolution luminance: downscale `image` to `sh`×`sw` (nearest) and
   return its luma ^doubles (length sh*sw). Used by splat-painter.wavelet."
  [image sh sw]
  (luma (if (and (= sh (:height image)) (= sw (:width image)))
          image
          (downsample image sh sw))))

(defn- tensor-eigen
  "Minor-eigenvector angle (stroke direction) + coherence + energy of a symmetric
   2×2 tensor [[a f][f b]]. Returns [theta coherence (a+b)]."
  [a b f]
  (let [phi   (* 0.5 (Math/atan2 (* 2.0 f) (- a b)))
        theta (+ phi (/ Math/PI 2.0))
        tt    (/ (+ a b) 2.0)
        d     (Math/sqrt (+ (* (/ (- a b) 2.0) (/ (- a b) 2.0)) (* f f)))
        coh   (max 0.0 (min 1.0 (/ (- (+ tt d) (- tt d)) (+ (+ tt d) (- tt d) 1e-9))))]
    [theta coh (+ a b)]))

(defn analyze
  "Full pipeline: (downsample →) gradient-field → structure-tensor (radius 2).
   The tensor is computed at ≤`max-side` (default 384) — a coarse orientation
   field is enough to steer splats and cuts the one-time cost sharply. Returns
   {:h Ht :w Wt :jxx :jyy :jxy :gmax :src-h H :src-w W}; orient-at maps
   full-image coords into the tensor grid via :src-h/:src-w."
  ([image] (analyze image 384))
  ([image max-side]
   (let [H (:height image) W (:width image)
         scale (min 1.0 (/ (double max-side) (double (max H W))))
         sh (max 1 (long (Math/round (* H scale))))
         sw (max 1 (long (Math/round (* W scale))))
         small (if (>= scale 1.0) image (downsample image sh sw))
         gfield (gradient-field small)
         sfield (structure-tensor gfield 2)
         Ht (:h sfield) Wt (:w sfield)
         ^doubles jxx (:jxx sfield) ^doubles jyy (:jyy sfield) ^doubles jxy (:jxy sfield)
         n (* Ht Wt)
         gmax (loop [i 0 m 0.0]
                (if (< i n)
                  (recur (inc i) (max m (+ (aget jxx i) (aget jyy i))))
                  m))
         ;; A heavily-blurred copy of the tensor: the strong edges' orientation
         ;; DIFFUSES into the surrounding flat areas (tensor smoothing / voting), so
         ;; flow-at gives, at a flat point, the direction of the nearby edges. This is
         ;; the "edges seed the flow" field — strokes in low-detail regions follow the
         ;; main features instead of an image-independent noise swirl.
         fr (max 3 (min 16 (quot (min Ht Wt) 3)))
         ^doubles fjxx (box-blur-2d jxx Ht Wt fr)
         ^doubles fjyy (box-blur-2d jyy Ht Wt fr)
         ^doubles fjxy (box-blur-2d jxy Ht Wt fr)
         ;; PRECOMPUTE the per-cell eigen fields once (theta/coherence/energy for the
         ;; sharp tensor, theta/strength for the diffused flow) so orient-at / flow-at
         ;; are cheap array samples in the hot per-splat loop, not per-splat atan2+sqrt.
         theta-a (double-array n) coh-a (double-array n) grad-a (double-array n)
         fth-a (double-array n) fstr-a (double-array n)]
     (dotimes [i n]
       (let [[th coh gr] (tensor-eigen (aget jxx i) (aget jyy i) (aget jxy i))]
         (aset theta-a i th) (aset coh-a i coh) (aset grad-a i gr))
       (let [[fth fstr _] (tensor-eigen (aget fjxx i) (aget fjyy i) (aget fjxy i))]
         (aset fth-a i fth) (aset fstr-a i fstr)))
     (assoc sfield :gmax gmax :src-h H :src-w W
            :theta theta-a :coherence coh-a :grad2 grad-a
            :flow-theta fth-a :flow-str fstr-a))))

(defn- grid-idx [sfield x y]
  (let [H (:h sfield) W (:w sfield)
        src-h (long (or (:src-h sfield) H)) src-w (long (or (:src-w sfield) W))
        xi (clamp (round-int (* (double x) (/ (double H) src-h))) 0 (dec H))
        yi (clamp (round-int (* (double y) (/ (double W) src-w))) 0 (dec W))]
    (+ (* xi W) yi)))

;; ---------------------------------------------------------------------------
;; orient-at
;; ---------------------------------------------------------------------------

(defn orient-at
  "Sample the structure tensor at (x,y) — given in full-image (:src-h × :src-w)
   coords, mapped into the coarse tensor grid — and return
   {:theta θ :coherence c :grad2 g}.
   - θ = stroke (minor-eigenvector) angle in radians
   - c = coherence ∈ [0,1] (how directional the edge is)
   - g = jxx+jyy at the pixel (gradient energy)"
  [sfield x y]
  (let [idx (grid-idx sfield x y)
        ^doubles theta (:theta sfield) ^doubles coh (:coherence sfield) ^doubles grad (:grad2 sfield)]
    {:theta (aget theta idx) :coherence (aget coh idx) :grad2 (aget grad idx)}))

(defn edge-strength-fn
  "Returns a cheap (fn [x y] -> [0,1]) reading a PRECOMPUTED normalized-gradient
   edge-strength array at full-image coords — feeds splat-painter.grid/optimize, which
   calls it ~1e6 times, so the per-call sqrt is done once here, not per call."
  [sfield]
  (let [H (:h sfield) W (:w sfield)
        src-h (double (or (:src-h sfield) H)) src-w (double (or (:src-w sfield) W))
        rx (/ (double H) src-h) ry (/ (double W) src-w)
        ^doubles jxx (:jxx sfield) ^doubles jyy (:jyy sfield)
        gmax (max (double (:gmax sfield)) 1e-12)
        n (* H W)
        es (double-array n)]
    (dotimes [i n] (aset es i (min 1.0 (Math/sqrt (/ (+ (aget jxx i) (aget jyy i)) gmax)))))
    (fn [x y]
      (let [xi (clamp (long (Math/round (* (double x) rx))) 0 (dec H))
            yi (clamp (long (Math/round (* (double y) ry))) 0 (dec W))]
        (aget es (+ (* xi W) yi))))))

(defn flow-at
  "Sample the DIFFUSED tensor (heavily-blurred, so edge orientations propagate into
   flat areas) at full-image coords (x,y). Returns {:theta θ :strength s}: θ is the
   local flow direction seeded by the nearby edges, s ∈ [0,1] how strongly those
   edges define it (near an edge → high; a truly featureless region → ~0, where the
   caller falls back to a noise flow)."
  [sfield x y]
  (let [idx (grid-idx sfield x y)
        ^doubles fth (:flow-theta sfield) ^doubles fstr (:flow-str sfield)]
    (if (and fth fstr)
      {:theta (aget fth idx) :strength (aget fstr idx)}
      {:theta 0.0 :strength 0.0})))
