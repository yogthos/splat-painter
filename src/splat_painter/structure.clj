(ns splat-painter.structure
  "Per-pixel 2D structure-tensor field for an image.

  Sobel gradients → J = [[jxx jxy][jxy jyy]] per pixel, box-blurred over a
  local neighbourhood. The minor-eigenvector direction of J is the local edge
  (isophote / brushstroke) direction — perpendicular to the gradient.

  Orientation follows the splat-painter.gaussian convention:
    x = row index (0..H-1, increasing downward)
    y = col index (0..W-1, increasing rightward)
    θ = 0 means the major axis is along the x/row direction.

  All per-pixel buffers are Java double-arrays for speed on ~1e6-pixel images, and
  the per-pixel loops band across threads via splat-painter.par — each band writes
  a disjoint index range, so the output matches the serial loop exactly."
  (:require [splat-painter.par :as par]))

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
  "Di Zenzo COLOUR structure-tensor products from 3×3 Sobel gradients per RGB
   channel (edge-replicate): jxx = Σ_c gx_c², jyy = Σ_c gy_c², jxy = Σ_c gx_c·gy_c.
   Luma-only gradients are blind to iso-luminant chroma edges — lips against skin,
   red on green — which then never receive edge-guided strokes and blend into
   blobs. Each channel is gamma-corrected (1/2.2) so dark regions keep their edge
   signal. Returns {:h H :w W :jxx ^doubles :jyy ^doubles :jxy ^doubles} (pre-blur)."
  [image]
  (let [H  (:height image)
        W  (:width image)
        ^doubles px (:pixels image)
        n  (* H W)
        ^doubles C (double-array (* n 3))
        _  (dotimes [i (* n 3)] (aset C i (Math/pow (aget px i) 0.4545)))
        ^doubles jxx (double-array n)
        ^doubles jyy (double-array n)
        ^doubles jxy (double-array n)]
    (dotimes [x H]
      (dotimes [y W]
        (let [idx (+ (* x W) y)
              xm1 (clamp (dec x) 0 (dec H))
              xp1 (clamp (inc x) 0 (dec H))
              ym1 (clamp (dec y) 0 (dec W))
              yp1 (clamp (inc y) 0 (dec W))]
          (loop [c 0 sxx 0.0 syy 0.0 sxy 0.0]
            (if (== c 3)
              (do (aset jxx idx sxx) (aset jyy idx syy) (aset jxy idx sxy))
              (let [at (fn [xi yi] (aget C (+ (* 3 (+ (* xi W) yi)) c)))
                    L00 (at xm1 ym1) L01 (at xm1 y) L02 (at xm1 yp1)
                    L10 (at x   ym1)                 L12 (at x   yp1)
                    L20 (at xp1 ym1) L21 (at xp1 y) L22 (at xp1 yp1)
                    gx (- (+ L20 (* 2.0 L21) L22)
                          (+ L00 (* 2.0 L01) L02))
                    gy (+ (- L00) L02
                          (* -2.0 L10) (* 2.0 L12)
                          (- L20) L22)]
                (recur (inc c) (+ sxx (* gx gx)) (+ syy (* gy gy)) (+ sxy (* gx gy)))))))))
    {:h H :w W :jxx jxx :jyy jyy :jxy jxy}))

;; ---------------------------------------------------------------------------
;; structure-tensor (separable box-blur)
;; ---------------------------------------------------------------------------

(defn- box-blur-2d
  "Separable 2D box-blur of `src` (H×W) with `radius`, edge-replicate.
   Returns a new ^doubles array."
  [^doubles src H W radius]
  ;; ^doubles src/tmp/dst + a double divisor `wd` let the sliding sums read via the
  ;; unboxed flvector path (jolt-flaget) and the +/-// lower to fl+/fl-/fl/.
  ;; each pass is a disjoint write per row (then per column), so the loops band
  ;; across threads unchanged — same sliding sums, same order within a band.
  (let [wd (double (inc (* 2 radius)))
        ^doubles tmp (double-array (* H W))]
    ;; horizontal pass: blur each row
    (par/dobands! H (* H W)
      (fn [xlo xhi]
        (loop [x (long xlo)]
          (when (< x (long xhi))
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
                    (recur (inc y) new-sum)))))
            (recur (inc x))))))
    ;; vertical pass: blur each column of tmp into dst
    (let [^doubles dst (double-array (* H W))]
      (par/dobands! W (* H W)
        (fn [ylo yhi]
          (loop [y (long ylo)]
            (when (< y (long yhi))
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
                      (recur (inc x) new-sum)))))
              (recur (inc y))))))
      dst)))

(defn structure-tensor
  "From gradient field, compute the 2×2 structure tensor J = [[jxx jxy][jxy jyy]]
   per pixel, box-blurred (separable) over a (2*radius+1)×(2*radius+1) window.
   Returns {:h H :w W :jxx ^doubles :jyy ^doubles :jxy ^doubles}."
  [gfield radius]
  (let [H (:h gfield) W (:w gfield)]
    ;; gradient-field already returns the colour-tensor products; just smooth them.
    {:h H :w W
     :jxx (box-blur-2d (:jxx gfield) H W radius)
     :jyy (box-blur-2d (:jyy gfield) H W radius)
     :jxy (box-blur-2d (:jxy gfield) H W radius)}))

;; ---------------------------------------------------------------------------
;; blur-image — one-time average-colour precompute
;; ---------------------------------------------------------------------------

(defn box-blur
  "Public separable box-blur of a single-channel H×W ^doubles array (edge-replicate)."
  [arr H W radius]
  (box-blur-2d arr H W radius))

(defn bilateral-blur
  "EDGE-AWARE smooth colour field: a box window averaged with RANGE weights —
   each neighbour weighted by how close its raw luma is to the centre pixel's,
   so shading within a region smooths while nothing mixes across a boundary.
   The plain box light-blur fed every stroke a colour contaminated by whatever
   sat across the nearest edge (a pixel 1-2px inside a dark strap sampled a
   lightened mix), and strokes near edges paint at HIGH specificity — the pale
   splotches in dark areas and dark smudges on smooth skin all traced back to
   those mixed samples. Returns a flat H*W*3 double-array."
  [image radius]
  ;; O(n) binned approximation (bilateral grid): K luma bins; per bin, weight
  ;; every pixel by a hat kernel around the bin centre, box-blur the weighted
  ;; colour + weight (separable, O(n)), and interpolate each pixel's result from
  ;; its two bracketing bins. The naive windowed form was O(n·(2r+1)²) — 145s
  ;; per image load; this runs in a few seconds at identical visual quality.
  (let [H (long (:height image)) W (long (:width image))
        ^doubles px (:pixels image)
        n (* H W)
        ^doubles L (luma image)
        K 9
        dl (/ 1.0 (double (dec K)))
        out (double-array (* n 3))
        osum (double-array n)]
    (dotimes [k K]
      (let [lk (* (double k) dl)
            wk (double-array n)
            wr (double-array n)
            wg (double-array n)
            wb (double-array n)]
        (par/dobands! n
          (fn [lo hi]
            (loop [i (long lo)]
              (when (< i (long hi))
                (let [w (max 0.0 (- 1.0 (/ (Math/abs (- (aget L i) lk)) dl)))
                      b (* 3 i)]
                  (aset wk i w)
                  (aset wr i (* w (aget px b)))
                  (aset wg i (* w (aget px (+ b 1))))
                  (aset wb i (* w (aget px (+ b 2)))))
                (recur (inc i))))))
        (let [^doubles bw (box-blur-2d wk H W radius)
              ^doubles br (box-blur-2d wr H W radius)
              ^doubles bg (box-blur-2d wg H W radius)
              ^doubles bb (box-blur-2d wb H W radius)]
          (par/dobands! n
            (fn [lo hi]
              (loop [i (long lo)]
                (when (< i (long hi))
                  (let [h (max 0.0 (- 1.0 (/ (Math/abs (- (aget L i) lk)) dl)))]
                    (when (pos? h)
                      (let [b (* 3 i)
                            inv (/ h (max (aget bw i) 1e-12))]
                        (aset out b       (+ (aget out b)       (* inv (aget br i))))
                        (aset out (+ b 1) (+ (aget out (+ b 1)) (* inv (aget bg i))))
                        (aset out (+ b 2) (+ (aget out (+ b 2)) (* inv (aget bb i))))
                        (aset osum i (+ (aget osum i) h)))))
                  (recur (inc i)))))))))
    ;; hat kernels form a partition of unity, but guard the division anyway
    (dotimes [i n]
      (let [s (aget osum i) b (* 3 i)]
        (when (> (Math/abs (- s 1.0)) 1e-9)
          (let [inv (/ 1.0 (max s 1e-12))]
            (aset out b       (* (aget out b) inv))
            (aset out (+ b 1) (* (aget out (+ b 1)) inv))
            (aset out (+ b 2) (* (aget out (+ b 2)) inv))))))
    out))

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

(defn- upsample-rgb
  "Bilinearly resample a flat sh*sw*3 ^doubles buffer up to H×W. The fields built at
   reduced resolution (see dominant-blur) are window statistics — smooth by
   construction — so bilinear reconstruction loses nothing a nearest-neighbour
   expansion would only replace with blocking."
  [^doubles src sh sw H W]
  (let [sh (long sh) sw (long sw) H (long H) W (long W)
        out (double-array (* H W 3))
        rx (/ (double (dec (max 2 sh))) (double (max 1 (dec H))))
        ry (/ (double (dec (max 2 sw))) (double (max 1 (dec W))))]
    (dotimes [x H]
      (let [fx (min (double (dec sh)) (* x rx))
            x0 (long fx) x1 (min (dec sh) (inc x0)) tx (- fx x0)]
        (dotimes [y W]
          (let [fy (min (double (dec sw)) (* y ry))
                y0 (long fy) y1 (min (dec sw) (inc y0)) ty (- fy y0)
                b00 (* 3 (+ (* x0 sw) y0)) b01 (* 3 (+ (* x0 sw) y1))
                b10 (* 3 (+ (* x1 sw) y0)) b11 (* 3 (+ (* x1 sw) y1))
                d   (* 3 (+ (* x W) y))]
            (dotimes [c 3]
              (let [a (+ (* (- 1.0 ty) (aget src (+ b00 c))) (* ty (aget src (+ b01 c))))
                    b (+ (* (- 1.0 ty) (aget src (+ b10 c))) (* ty (aget src (+ b11 c))))]
                (aset out (+ d c) (+ (* (- 1.0 tx) a) (* tx b)))))))))
    out))

(defn- dominant-blur-full
  "dominant-blur at the resolution it is handed. See dominant-blur."
  [image radius p]
  (let [H (long (:height image)) W (long (:width image))
        ^doubles px (:pixels image)
        n (* H W)
        ^doubles L (luma image)
        K 9
        dl (/ 1.0 (double (dec K)))
        num (double-array (* n 3))
        den (double-array n)]
    (dotimes [k K]
      (let [lk (* (double k) dl)
            wk (double-array n)
            wr (double-array n)
            wg (double-array n)
            wb (double-array n)]
        (par/dobands! n
          (fn [lo hi]
            (loop [i (long lo)]
              (when (< i (long hi))
                (let [w (max 0.0 (- 1.0 (/ (Math/abs (- (aget L i) lk)) dl)))
                      b (* 3 i)]
                  (aset wk i w)
                  (aset wr i (* w (aget px b)))
                  (aset wg i (* w (aget px (+ b 1))))
                  (aset wb i (* w (aget px (+ b 2)))))
                (recur (inc i))))))
        (let [^doubles bw (box-blur-2d wk H W radius)
              ^doubles br (box-blur-2d wr H W radius)
              ^doubles bg (box-blur-2d wg H W radius)
              ^doubles bb (box-blur-2d wb H W radius)]
          ;; bin colour is br/bw; its vote is bw^p. Accumulate bw^p·(br/bw) = bw^(p-1)·br
          ;; directly so the division never has to be done twice.
          (par/dobands! n
            (fn [lo hi]
              (loop [i (long lo)]
                (when (< i (long hi))
                  (let [w (aget bw i)]
                    (when (> w 1e-12)
                      (let [wp (Math/pow w (double p))
                            f  (/ wp w)
                            b  (* 3 i)]
                        (aset num b       (+ (aget num b)       (* f (aget br i))))
                        (aset num (+ b 1) (+ (aget num (+ b 1)) (* f (aget bg i))))
                        (aset num (+ b 2) (+ (aget num (+ b 2)) (* f (aget bb i))))
                        (aset den i (+ (aget den i) wp)))))
                  (recur (inc i)))))))))
    (let [out (double-array (* n 3))]
      (dotimes [i n]
        (let [d (max (aget den i) 1e-12) b (* 3 i)]
          (aset out b       (/ (aget num b) d))
          (aset out (+ b 1) (/ (aget num (+ b 1)) d))
          (aset out (+ b 2) (/ (aget num (+ b 2)) d))))
      out)))

(defn dominant-blur
  "The DOMINANT tone in a brush-sized window, not the average of it — the colour a
   coverage-tier stroke should be loaded with.

   The coverage tiers sample an EDGE-PRESERVING blur, which by construction keeps
   structure FINER than the brush that paints with it. On a drawing that means a
   σ6 stroke centred on a 5px letter paints solid black over a 12px radius and no
   later tier repaints the paper around it: letters fatten, counters fill, and the
   gaps between them close. Preserving edges is right at detail scale and wrong at
   coverage scale — a fat brush cannot state a feature thinner than itself, so
   sub-brush structure must drop OUT of its colour rather than be smeared across
   it (the plain box blur, which halos) or stamped by it (the edge-preserving one).

   Same binned-window machinery as `bilateral-blur`: K luma bins with hat kernels,
   each bin's weight and weighted colour box-blurred over the window. The one
   difference is how the bins are recombined — by w^p instead of w. p=1 is exactly
   the box blur (the window mean); as p rises, a minority population's bin loses
   against the majority's and drops out. So p is a dial from `average tone` to
   `modal tone`, and the structure that survives is whatever occupies most of a
   brush-sized window: big masses stay, thin lines go.

   Large regions keep the halo fix that motivated edge-preserving-blur: a window
   just inside a silhouette is majority-subject, so the mode is the subject's own
   colour and nothing bleeds in from across the boundary — bin separation IS the
   edge preservation, and unlike a blend against the raw pixel it does not carry
   sub-brush structure back in.

   p=4 is the default: swept 2/4/8/16 against the render. On line art 4 already
   clears the paper (the gray band around the title is gone and the counters open);
   8 and 16 clear it no further but start to posterize a photograph, because the
   winning bin flips abruptly where two modes are close — visible as flat patches
   in bokeh and a stepped edge on an eyelash by p=8.

   Computed on a DOWNSAMPLED copy and bilinearly upsampled: the result is a window
   statistic over a ~25px neighbourhood, so it carries nothing near pixel frequency
   and full-resolution binning is 16x the work for no visible difference (5.5s vs
   0.4s on a 0.9MP frame — this runs per image load, alongside the bilateral). The
   downsample is NEAREST-NEIGHBOUR on purpose: area-averaging first would blend a
   black line into the white paper around it and hand the filter a gray population
   to find the mode of, which is the halo this exists to remove. Nearest keeps every
   sample a true pixel, and a thin line simply loses the vote."
  ([image radius] (dominant-blur image radius 4.0))
  ([image radius p]
   (let [step (max 1 (quot (long radius) 3))]
     (if (> step 1)
       (let [H (long (:height image)) W (long (:width image))
             sh (max 1 (quot H step)) sw (max 1 (quot W step))]
         (upsample-rgb (dominant-blur (downsample image sh sw) (quot (long radius) step) p)
                       sh sw H W))
       (dominant-blur-full image radius p)))))

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
   {:h Ht :w Wt :jxx :jyy :jxy :gmax :src-h H :src-w W}; callers index
   :theta/:coherence directly, mapping via :src-h/:src-w."
  ([image] (analyze image 768))
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
         ;; the diffused tensor carries, at a flat point, the direction of the nearby
         ;; edges. This is the "edges seed the flow" field — strokes in low-detail
         ;; regions follow the main features instead of an image-independent noise swirl.
         fr (max 3 (min 16 (quot (min Ht Wt) 3)))
         ^doubles fjxx (box-blur-2d jxx Ht Wt fr)
         ^doubles fjyy (box-blur-2d jyy Ht Wt fr)
         ^doubles fjxy (box-blur-2d jxy Ht Wt fr)
         ;; PRECOMPUTE the per-cell eigen fields once (theta/coherence/energy for the
         ;; sharp tensor, theta/strength for the diffused flow) as flat arrays, so
         ;; seed.clj's direct :theta / :coherence / :flow-* reads are cheap
         ;; samples in the hot per-splat loop, not per-splat atan2+sqrt.
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
