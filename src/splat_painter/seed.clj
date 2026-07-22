(ns splat-painter.seed
  "Turn a target image into a field of 2D gaussian splats, applying the style
  controls. The covariance/precision math is splat-painter.gaussian (a port of
  rendering2d.py); here we only choose each splat's mean, covariance, and color
  from the image + controls — the same parameterization the reference optimizer
  learns (means, scale→covariance, color), but set directly from image pixels
  instead of by gradient descent, so the field resembles the picture instantly.

  Each splat is an oriented brushstroke derived from the local image structure
  tensor (splat-painter.structure). Placement is COARSE-TO-FINE (splat-painter.seed
  layered-means): a base layer of large splats that fully COVERS the image (so the
  background can never show through — no gaps), then progressively finer layers added
  only where the wavelet detail map (splat-painter.wavelet) is high, so detail sits on top
  of an unbroken underpainting. There is no cell grid, so no grid facets; Perlin noise
  (splat-painter.noise) warps flat-region strokes and varies their flow so the field never
  reads as a lattice.

  Controls:
    :count      density / overlap of the layers (Splats slider). Higher = tighter
                overlap = smoother; lower = looser. Floored so coverage holds. default 6000.
    :size       base (coarsest) splat stdev in px; each finer level halves it. default 3.0.
    :stroke     elongation strength, >=0. Larger = longer strokes. default 2.0.
    :detail     0..1 how many fine levels are added and how far they reach. 0 = base
                layer only (flat); 1 = up to 4 levels of accumulating detail. default 0.6.
    :variation  0..1 Perlin flat-region position warp + per-stroke size/tone jitter.
                0 = none. default 0.5.
    :opacity    per-splat alpha 0..1, passed through into the returned field.
                default 0.9.
    :sharpness  deprecated/ignored — stroke colour now always leans toward the raw
                (edge-preserving) pixel, fully at edges/detail. Kept for API compat.
                was: 0..1, colour blend toward the raw (edge-preserving) pixel at
                edges. 0 = always the smooth blur. default 0.7.
    :contrast   0.5..2.0 per-channel contrast about 0.5. 1.0 = no change.
    :palette    max colors for diversity quantization; 0 (or <2) = off. default 0.
    :background additive base; a number (gray) or [r g b]; defaults to black

  An image is {:height :width :pixels (flat H*W*3 double-array 0..1) :channels 3}.
  If it carries precomputed :structure, :detail, and :blur they are reused so
  live slider drags stay fast."
  (:require [splat-painter.gaussian :as gauss]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.noise :as noise]
            [splat-painter.palette :as p]))

;; Baseline elongation floor: even a flat region (coherence 0) elongates a little so
;; the field reads as brushwork, but keep it modest — too much makes flat areas a
;; thin directional hatch instead of painterly dabs.
(def ^:private min-coh 0.28)

;; --- deterministic per-stroke pseudo-random helpers --------------------------

(defn- hash01
  "Cheap deterministic per-stroke random in [0,1) from integerish coords + salt."
  [a b salt]
  (let [h (mod (+ (* (long a) 73856093) (* (long b) 19349663) (* (long salt) 83492791))
                4294967296)]
    (/ (double h) 4294967296.0)))

(defn- blend-angle
  "Undirected-orientation blend between t1 and t2 weighted by w.
   0 = all t1, 1 = all t2. Works in the double-angle representation (2θ)
   so π-edge (0≡π) is handled smoothly."
  [t1 t2 w]
  (let [bx (+ (* (- 1.0 w) (Math/cos (* 2.0 t1))) (* w (Math/cos (* 2.0 t2))))
        by (+ (* (- 1.0 w) (Math/sin (* 2.0 t1))) (* w (Math/sin (* 2.0 t2))))]
    (* 0.5 (Math/atan2 by bx))))

;; --- coarse-to-fine layered placement ----------------------------------------

;; the fragment shader brute-force loops over every splat per pixel, so the field must
;; stay under its MAX_SPLATS (16384). When small strokes would exceed that, scale ALL
;; stroke sizes UP so the whole image stays covered — NEVER truncate (that would drop
;; whole rows / cut off the bottom of the image).
;; hard ceiling = the shader's MAX_SPLATS. The Splats control sets the working budget up to
;; this; more splats = smaller strokes = more preserved detail (a detailed oil painting) at a
;; higher render cost, fewer = larger strokes = looser/abstract and faster.
(def ^:private splat-budget 48000)

(defn- detail-fraction
  "Fraction of the wavelet detail map at/above the normalized threshold t∈[0,1]. Used to
   estimate how many splats each fine level will place (so the budget can scale them)."
  [dmap t]
  (let [^doubles d (:detail dmap)
        dmax (double (max 1e-9 (:dmax dmap)))
        n    (alength d)
        thr  (* (double t) dmax)]
    (if (zero? n)
      0.0
      (loop [i 0 c 0]
        (if (>= i n)
          (/ (double c) (double n))
          (recur (inc i) (if (>= (aget d i) thr) (inc c) c)))))))

(defn- layered-means
  "COARSE-TO-FINE placement: a base layer of large splats that FULLY COVERS the image —
   spacing < stdev ⇒ heavy overlap, so the (black) background can never show through — then
   progressively finer layers, each placed only where the wavelet detail is high enough, so
   detail accumulates ON TOP of an unbroken underpainting. There is no cell grid, so no cell
   facets; each splat's orientation/colour come from the flow + detail fields.

   `size` = base (coarsest) stdev; each finer level halves it. `detail` sets how many fine
   levels (1 = base only … up to 4) and how far they reach. `count` is the SPLAT BUDGET (the
   Splats control): the placement fills up to that many strokes — more ⇒ smaller strokes ⇒
   more preserved detail; fewer ⇒ larger ⇒ looser. `variation` scales a flat-region Perlin
   warp that breaks any residual level lattice — detail strokes stay put (faithful edges).
   If the natural field would exceed the budget, every level's stdev scales up uniformly, so
   the base always keeps its heavy overlap ⇒ full coverage, no gaps. `curvature` (0..1) scales
   how far the flat-region Perlin warp bends/curves the strokes off the level grid.
   Returns [[x y stdev D]…] (D = effective detail 0..1)."
  [dmap detail size variation curvature count H W]
  (let [smax     (double size)
        hd       (double (dec (long H))) wd (double (dec (long W)))
        budget   (min (double splat-budget) (max 500.0 (double count)))
        warp     (* 0.95 (double curvature))
        area     (double (* (long H) (long W)))
        nlev     (long (max 1 (min 4 (inc (Math/round (* (double detail) 3.0))))))
        deff     (fn [D] (min 1.0 (* (double detail) (double D) 2.2)))
        thresh   (fn [lvl] (if (zero? (long lvl)) -1.0 (min 0.9 (* 0.26 (double lvl)))))
        ;; base layer overlaps heavily (spacing 0.72×stdev ⇒ full coverage); finer layers are
        ;; sparser accents (the base fills behind them, so gaps between fine strokes don't
        ;; matter). Overlap is FIXED, so coverage never depends on the budget.
        overlap  (fn [lvl] (if (zero? (long lvl)) 0.72 1.25))
        sp-of    (fn [lvl scale] (* (overlap lvl) scale (/ smax (Math/pow 2.0 (double lvl)))))
        ;; budget: total(scale)=K/scale² ⇒ smallest scale≥1 that fits under the working budget.
        K (loop [lvl 0 acc 0.0]
            (if (>= lvl nlev)
              acc
              (let [f  (if (zero? lvl) 1.0 (detail-fraction dmap (thresh lvl)))
                    sp (sp-of lvl 1.0)]
                (recur (inc lvl) (+ acc (/ (* f area) (* sp sp)))))))
        scale (max 1.0 (Math/sqrt (/ K budget)))]
    (persistent!
      (loop [lvl 0 acc (transient [])]
        (if (>= lvl nlev)
          acc
          (let [ssz (* scale (/ smax (Math/pow 2.0 (double lvl))))
                sp  (sp-of lvl scale)
                th  (thresh lvl)
                nx  (long (Math/ceil (/ (double H) sp)))
                ny  (long (Math/ceil (/ (double W) sp)))
                acc2 (loop [i 0 acc acc]
                       (if (>= i nx)
                         acc
                         (recur (inc i)
                           (loop [j 0 acc acc]
                             (if (>= j ny)
                               acc
                               (let [cx (* (+ (double i) 0.5) sp)
                                     cy (* (+ (double j) 0.5) sp)
                                     dv (wavelet/detail-at dmap cx cy)]
                                 (if (and (pos? (long lvl)) (< dv th))
                                   (recur (inc j) acc)        ; not detailed enough for this fine level
                                   (let [jx (* sp 0.45 (- (hash01 (+ (* i 137) lvl) j 3) 0.5))
                                         jy (* sp 0.45 (- (hash01 (+ (* i 149) lvl) j 7) 0.5))
                                         x  (+ cx jx) y (+ cy jy)
                                         D  (deff dv)
                                         ;; flat-region Perlin warp breaks any residual level
                                         ;; lattice; detail strokes (D≈1) stay put → faithful edges.
                                         aw (* warp (- 1.0 D) ssz)
                                         x2 (if (< aw 0.2) x
                                              (+ x (* aw (noise/noise2 (* 0.06 x) (* 0.06 y)))))
                                         y2 (if (< aw 0.2) y
                                              (+ y (* aw (noise/noise2 (+ 41.3 (* 0.06 x)) (+ 17.9 (* 0.06 y))))))]
                                     ;; keep centres in-bounds so no budget is wasted off-screen
                                     ;; (edges stay covered by the splats' tails).
                                     (recur (inc j)
                                       (conj! acc [(max 0.0 (min hd x2)) (max 0.0 (min wd y2)) ssz D]))))))))))]
            (recur (inc lvl) acc2)))))))

;; --- precomputed smooth Perlin fields (flow angle, size, tone) ---------------
;; noise2 is ~30 ops; calling it 4× per splat over ~14k splats dominated the render.
;; The fields are smooth (low frequency), so precomputing them at the tensor resolution
;; once and sampling (a cheap aget) per splat is visually identical and far faster.

(defn prep-noise
  "Precompute, at `sfield`'s tensor resolution, everything the per-splat loop needs
   that depends only on POSITION: the final blended stroke orientation (edge-seeded
   flow + Perlin fill + sharp edge), the coherence, and the size/tone Perlin fields.
   Folding the orientation blends (2 atan2/cos/sin each) into this one-time pass makes
   the hot loop a handful of array samples. Returns
   {:h :w :src-h :src-w :theta :coherence :snoise :tnoise}."
  [sfield]
  (let [H (:h sfield) W (:w sfield)
        srch (double (or (:src-h sfield) H)) srcw (double (or (:src-w sfield) W))
        n (* H W) fs 0.004
        ^doubles s-theta (:theta sfield) ^doubles s-coh (:coherence sfield)
        ^doubles f-theta (:flow-theta sfield) ^doubles f-str (:flow-str sfield)
        ftheta (double-array n) cohr (double-array n) sn (double-array n) tn (double-array n)]
    (dotimes [xi H]
      (dotimes [yi W]
        (let [idx (+ (* xi W) yi)
              x (* xi (/ srch H)) y (* yi (/ srcw W))
              fvx (- (noise/noise2 (* x fs) (* y fs)) 0.5)
              fvy (- (noise/noise2 (+ (* x fs) 137.0) (+ (* y fs) 91.0)) 0.5)
              flow-t (Math/atan2 fvy fvx)
              coherence (aget s-coh idx)
              flow-base (blend-angle flow-t (aget f-theta idx) (min 1.0 (* 2.5 (aget f-str idx))))
              theta (blend-angle flow-base (aget s-theta idx) coherence)]
          (aset ftheta idx theta)
          (aset cohr idx coherence)
          (aset sn idx (- (noise/noise2 (+ (* x fs) 613.0) (+ (* y fs) 227.0)) 0.5))
          (aset tn idx (- (noise/noise2 (+ (* x fs) 941.0) (+ (* y fs) 373.0)) 0.5)))))
    {:h H :w W :src-h (:src-h sfield) :src-w (:src-w sfield)
     :theta ftheta :coherence cohr :snoise sn :tnoise tn}))

(defn- sample-fields
  "[theta coherence snoise tnoise] from a prep-noise field at full-image (x,y)."
  [nf x y]
  (let [H (:h nf) W (:w nf)
        srch (long (or (:src-h nf) H)) srcw (long (or (:src-w nf) W))
        xi (min (dec H) (max 0 (long (Math/round (* (double x) (/ (double H) srch))))))
        yi (min (dec W) (max 0 (long (Math/round (* (double y) (/ (double W) srcw))))))
        idx (+ (* xi W) yi)]
    [(aget ^doubles (:theta nf) idx) (aget ^doubles (:coherence nf) idx)
     (aget ^doubles (:snoise nf) idx) (aget ^doubles (:tnoise nf) idx)]))

;; --- helpers (unchanged) ----------------------------------------------------

(defn- sample-arr
  "Nearest-pixel [r g b] from a flat H*W*3 double-array at grid (x,y), clamped."
  [^doubles arr width height x y]
  (let [xi (min (dec height) (max 0 (int x)))
        yi (min (dec width)  (max 0 (int y)))
        base (* 3 (+ (* xi width) yi))]
    [(aget arr base) (aget arr (+ base 1)) (aget arr (+ base 2))]))

(defn- apply-contrast
  "Per-channel linear contrast about 0.5, clamped to [0,1]."
  [contrast [r g b]]
  (let [f (fn [c] (max 0.0 (min 1.0 (+ (* (- c 0.5) contrast) 0.5))))]
    [(f r) (f g) (f b)]))

(defn- resolve-background [bg]
  (cond
    (nil? bg) [0.0 0.0 0.0]
    (number? bg) [(double bg) (double bg) (double bg)]
    (sequential? bg) [(double (nth bg 0)) (double (nth bg 1)) (double (nth bg 2))]
    :else [0.0 0.0 0.0]))

;; --- main -------------------------------------------------------------------

(defn splat-field
  "Build a splat field from `image` (see ns doc) and `controls` (see ns doc).
   Returns {:splats […] :background [r g b] :height :width :opacity}."
  [{:keys [height width pixels] :as image} controls]
  (let [{:keys [count size stroke detail variation curvature opacity contrast palette background]
         :or   {count 6000 size 3.0 stroke 2.0 detail 0.6 variation 0.5 curvature 0.5
                opacity 0.9 contrast 1.0 palette 0 background 0.0}} controls
        n          (long (or count 6000))
        size       (double (or size 3.0))
        stroke     (double stroke)
        detail     (double detail)
        variation  (double variation)
        curvature  (double curvature)
        contrast   (double contrast)
        palette-n  (int (or palette 0))
        sfield     (or (:structure image) (structure/analyze image))
        dmap       (or (:detail image)    (wavelet/detail-map image))
        ^doubles raw-px  pixels
        ^doubles blur-px (or (:blur image) pixels)
        gmax-sqrt  (Math/sqrt (max (:gmax sfield) 0.0))
        nf         (or (:noise-fields image) (prep-noise sfield))
        means      (layered-means dmap detail size variation curvature n height width)
        splats     (vec
                     (for [[x y csz dlev] means
                           ;; theta (final edge-seeded flow orientation) and coherence are
                           ;; precomputed per cell (prep-noise) — the hot loop just samples.
                           :let [[theta coherence snoise tnoise] (sample-fields nf x y)
                                 ;; broad/round dabs in flat regions, thin strokes elongated
                                 ;; along strong edges — shape varies with local structure.
                                 ;; Elongation ALSO tapers with detail level: fine detail strokes
                                 ;; (high dlev) elongate fully along contours, while large flat
                                 ;; strokes (low dlev) stay round — long flat strokes read as
                                 ;; angular facet planes, round ones blend into a smooth wash.
                                 coh (+ min-coh (* (- 1.0 min-coh) coherence))
                                 e  (+ 1.0 (* stroke coh (+ 0.25 (* 0.75 (double dlev)))))
                                 se (Math/sqrt e)
                                 ;; csz is already the detail-varied stroke size (grid-means);
                                 ;; modulate it by the smooth Perlin size field for organic variation
                                 s0 (* csz (+ 1.0 (* variation 0.5 (* 2.0 snoise))))
                                 sx (* s0 se)          ; long axis along θ
                                 sy (/ s0 se)          ; short axis across the stroke
                                 ;; colour: each stroke pulls toward the raw pixel (a
                                 ;; distinct colour, so overlapping strokes read as separate
                                 ;; marks) — fully at edges AND in high-detail cells, so
                                 ;; text/lines keep the crisp pixel colour even where the
                                 ;; structure tensor is incoherent; flat low-detail regions keep
                                 ;; a little of the smooth blur base so they don't read as noise.
                                 t (min 1.0 (max 0.0 (+ 0.55 (* 0.45 (max coherence (double dlev))))))
                                 color0 (if (< t 0.01)
                                          (sample-arr blur-px width height x y)
                                          (let [[br bg bb] (sample-arr blur-px width height x y)
                                                [rr rg rb] (sample-arr raw-px width height x y)]
                                            [(+ (* br (- 1.0 t)) (* rr t))
                                             (+ (* bg (- 1.0 t)) (* rg t))
                                             (+ (* bb (- 1.0 t)) (* rb t))]))
                                 color-ac (if (== contrast 1.0) color0 (apply-contrast contrast color0))
                                 ;; smooth tone modulation (precomputed Perlin channel)
                                 tone (+ 1.0 (* variation 0.15 (* 2.0 tnoise)))
                                 color (mapv (fn [c] (max 0.0 (min 1.0 (* c tone)))) color-ac)]]
                       {:mean  [x y]
                        :cov   (gauss/covariance sx sy theta)
                        :color color}))
        splats (if (>= palette-n 2)
                 (let [cols (map :color splats)
                       q    (p/quantize cols palette-n)]
                   (mapv (fn [s c] (assoc s :color c)) splats q))
                 splats)
        ;; PAINT ORDER: broad strokes go down first (the underpainting), fine detail strokes
        ;; last so they sit ON TOP. The shader composites front-to-back (index 0 = topmost
        ;; layer), so ordering by INCREASING size puts small crisp detail strokes at the front
        ;; and big soft strokes at the back — otherwise a big soft stroke drawn over a detail
        ;; stroke washes it out (blurry), worse the more detail there is.
        ;; Decorate with det (∝ size²) ONCE per splat then sort by that — sort-by would
        ;; otherwise recompute det on every comparison (2× per compare = the hot cost).
        keyed  (sort-by first
                        (map (fn [s] (let [[c00 c01 _ c11] (:cov s)]
                                       [(max (- (* c00 c11) (* c01 c01)) 1e-8) s]))
                             splats))
        splats (mapv second keyed)
        ;; effective stdev = det^¼ = geometric-mean stdev = base stroke size (elongation
        ;; cancels). keyed is sorted ascending by det, so first/last give the size range the
        ;; shader needs to scale edge hardness by size.
        sig-min (if (seq keyed) (Math/sqrt (Math/sqrt (first (first keyed)))) 1.0)
        sig-max (if (seq keyed) (Math/sqrt (Math/sqrt (first (last keyed)))) 1.0)]
    {:splats     splats
     :background (resolve-background background)
     :height     height
     :width      width
     :opacity    (double opacity)
     :sig-min    (double sig-min)
     :sig-max    (double sig-max)}))
