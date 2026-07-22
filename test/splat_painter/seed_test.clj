(ns splat-painter.seed-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.seed :as seed]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.gaussian :as g]))

(defn- approx= [tol a b] (< (Math/abs (- (double a) (double b))) tol))
(defn- solid [H W [r g b :as c]]
  ;; flat H*W*3 double-array buffer, matching seed/pixel-at's layout
  {:height H :width W :channels 3
   :pixels (double-array (apply concat (for [_ (range (* H W))] c)))})
(defn- within? [lo hi x] (and (>= x lo) (<= x hi)))
(defn- gray-img [H W f]
  {:height H :width W :channels 3
   :pixels (double-array (mapcat (fn [x]
                                   (mapcat (fn [y]
                                             (let [g (double (f x y))] [g g g]))
                                           (range W)))
                                 (range H)))})

(deftest solid-image-seeds-to-uniform-color
  (let [img (solid 8 8 [0.5 0.5 0.5])
        {:keys [splats background]} (seed/splat-field img {:count 16 :variation 0.0 :detail 0.0})]
    (is (pos? (count splats)))
    (is (every? #(= [0.5 0.5 0.5] (:color %)) splats))
    (is (= [0.0 0.0 0.0] background))))

(deftest composite-reconstructs-solid-image
  ;; over-compositing (unlike additive) reconstructs a solid 0.5 image to ~0.5
  ;; in the interior instead of blowing out to white — the bug the fix targets.
  (let [img (solid 16 16 [0.5 0.5 0.5])
        fld (seed/splat-field img {:count 64 :size 2.0 :background 0.0})
        out (g/composite (:splats fld)
                         (repeat (* 16 16 3) 0.0) 16 16 1.0)
        center (nth out (* 3 (+ (* 8 16) 8)))]
    (is (approx= 0.25 center 0.5))))        ; ≈ the source gray, not 1.0/white

(deftest means-cover-the-grid
  (let [img (solid 32 24 [0.2 0.2 0.2])
        {:keys [splats]} (seed/splat-field img {:count 6000})
        xs (map (comp first :mean) splats)
        ys (map (comp second :mean) splats)]
    (is (every? #(within? 0 32 %) xs))
    (is (every? #(within? 0 24 %) ys))
    (is (< (apply min xs) 16))                 ; covers past the top half
    (is (> (apply max xs) 16))))                ; and past the middle

(deftest means-are-not-a-perfect-lattice
  ;; the jitter must actually perturb means off the cell centers, or a visible
  ;; grid shows through under over-compositing. With count>1 there must be at
  ;; least two splats whose x (or y) differ by less than one cell spacing, which
  ;; is impossible on an exact lattice.
  (let [img (solid 32 32 [0.2 0.2 0.2])
        {:keys [splats]} (seed/splat-field img {:count 100})
        xs (sort (map (comp first :mean) splats))
        gaps (map (fn [[a b]] (- b a)) (partition 2 1 xs))
        min-gap (apply min gaps)
        spacing (/ 32.0 (long (Math/sqrt 100)))] ; one cell spacing
    (is (< min-gap spacing))))                   ; some adjacent pair is closer than 1 cell

(deftest field-is-deterministic
  (let [img (solid 10 10 [0.7 0.1 0.2])]
    (is (= (seed/splat-field img {:count 25})
           (seed/splat-field img {:count 25})))))

(deftest size-sets-density
  ;; stroke SIZE sets density: smaller strokes fill the same cells with more splats.
  ;; (grid resolution is image-size based; Splats/count scales density separately.)
  (let [img (solid 128 128 [0.5 0.5 0.5])]
    (is (< (count (:splats (seed/splat-field img {:size 10.0})))
           (count (:splats (seed/splat-field img {:size 3.0})))))))

(deftest splats-count-is-the-budget
  ;; the Splats/count knob is the splat BUDGET: with a small stroke size the natural field
  ;; wants more strokes than either budget, so more budget ⇒ more (smaller) splats.
  (let [img (solid 200 200 [0.5 0.5 0.5])]
    (is (< (count (:splats (seed/splat-field img {:count 2000 :size 2.0})))
           (count (:splats (seed/splat-field img {:count 12000 :size 2.0})))))))

(deftest rendered-field-resembles-a-solid-image
  ;; additive rasterizer smoke: a seeded field rasterizes to a non-trivial image
  ;; (composite-reconstructs-solid-image above covers the over-compositing path).
  (let [img (solid 16 16 [0.5 0.5 0.5])
        fld (seed/splat-field img {:count 64 :size 2.0 :background 0.0})
        out (g/rasterize (:splats fld)
                         (repeat (* 16 16 3) 0.0) 16 16)
        center (nth out (* 3 (+ (* 8 16) 8)))]
    (is (pos? center))))

(deftest flat-image-splats-are-uniform-size-elongated-strokes
  ;; a flat image gives strokes of uniform SIZE (coherence 0 → same elongation, same
  ;; scale), though orientation still follows the always-on Perlin flow. Trace
  ;; c00+c11 = sx²+sy² is the rotation-invariant size measure.
  (let [img (solid 16 16 [0.5 0.5 0.5])
        {:keys [splats]} (seed/splat-field img {:count 64 :size 3.0 :stroke 4.0 :variation 0.0})
        covs   (map :cov splats)
        traces (map (fn [[c00 _ _ c11]] (+ c00 c11)) covs)
        [c00 _ _ c11] (first covs)]
    (is (every? #(approx= 1e-6 (first traces) %) traces) "flat image → uniform stroke size")
    (is (not (approx= 1e-6 c00 c11)) "strokes are elongated (min-coh floor), not round")))

(deftest edges-elongate-splats-along-the-contour
  ;; horizontal edge (step across rows). With stroke>0, at least one splat near the edge
  ;; is strongly anisotropic; for a horizontal stroke theta≈π/2 so c11 (col axis) is the
  ;; LONG axis => c11/c00 >> 1.
  (let [img (gray-img 32 32 (fn [x _] (if (< x 16) 0.0 1.0)))
        {:keys [splats]} (seed/splat-field img {:count 256 :size 3.0 :stroke 3.0 :detail 0.0 :variation 0.0})
        ratios (map (fn [{[c00 _ _ c11] :cov}] (/ (max c00 c11) (max 1e-9 (min c00 c11)))) splats)]
    (is (> (apply max ratios) 2.0))))   ; some splat elongated at the edge

(deftest field-carries-opacity
  (is (= 0.42 (:opacity (seed/splat-field (solid 8 8 [1 1 1]) {:count 4 :opacity 0.42})))))

(deftest splat-record-spec
  ;; the pure per-splat math the GPU generation shader must reproduce; pin it directly so the
  ;; CPU/GPU spec is guarded independently of placement + field sampling.
  ;; coh=0.64, e=2 → √e; s0=5 (snoise 0); t=0.775; contrast 1, tone 1 (tnoise 0).
  (let [{:keys [mean cov color]}
        (seed/splat-record 10.0 20.0 5.0 0.5 0.0 0.5 0.0 0.0 [0.4 0.4 0.4] [0.8 0.2 0.1] 2.5 0.5 1.0)
        [c00 c01 _ c11] cov
        [cr cg cb] color]
    (is (= [10.0 20.0] mean))
    (is (approx= 1e-6 50.0    c00))   ; sx² = (s0·√e)², e=2 ⇒ 50
    (is (approx= 1e-6 0.0     c01))   ; θ=0 ⇒ axis-aligned
    (is (approx= 1e-6 12.5    c11))   ; sy² = (s0/√e)² = 12.5
    (is (approx= 1e-6 0.71    cr))    ; 0.4·0.225 + 0.8·0.775
    (is (approx= 1e-6 0.245   cg))
    (is (approx= 1e-6 0.1675  cb))))

(deftest splat-field-golden
  ;; whole-generation regression guard (placement + covariance + colour). Pins the splat count
  ;; and a checksum of every splat's mean / det(cov) / colour for a fixed image + controls. Any
  ;; change that alters the produced field — including refactoring the per-splat math into a
  ;; shared fn for the GPU path, or the GPU output drifting from this CPU reference — trips it.
  (let [img (gray-img 48 64 (fn [x y] (if (and (> x 16) (< x 32) (> y 20) (< y 44))
                                        0.9 (* 0.5 (/ (double (+ x y)) 112.0)))))
        {:keys [splats]} (seed/splat-field img {:count 4000 :size 6.0 :stroke 2.5 :detail 0.6
                                                :variation 0.5 :curvature 0.5 :opacity 0.9 :contrast 1.0})
        [sx sy sd sc] (reduce (fn [[sx sy sd sc] {[mx my] :mean [c00 c01 _ c11] :cov [cr cg cb] :color}]
                                [(+ sx mx) (+ sy my) (+ sd (- (* c00 c11) (* c01 c01))) (+ sc cr cg cb)])
                              [0.0 0.0 0.0 0.0] splats)]
    (is (= 254 (count splats)))
    (is (approx= 0.05 6438.922   sx) "Σ mean-x")
    (is (approx= 0.05 8095.514   sy) "Σ mean-y")
    (is (approx= 0.5  210455.049 sd) "Σ det(cov)")
    (is (approx= 0.05 308.503    sc) "Σ colour")))

(deftest layer-params-shared-spec
  ;; layer-params is the per-level placement spec BOTH the CPU loop and the GPU generation
  ;; shader consume, so they enumerate the same cells. Guard its contract: finest-first
  ;; ordering (levels[0] = smallest stdev), cumulative candidate offsets, total = Σ nx·ny,
  ;; and ssz halving per finer level. If any drifts, the GPU field diverges from the CPU golden.
  (let [img  (gray-img 48 64 (fn [x y] (if (and (> x 16) (< x 32) (> y 20) (< y 44))
                                         0.9 (* 0.5 (/ (double (+ x y)) 112.0)))))
        dmap (wavelet/detail-map img)
        {:keys [nlev levels total warp]} (seed/layer-params dmap 0.6 6.0 0.5 0.5 4000 48 64)
        cells (map (fn [l] (* (:nx l) (:ny l))) levels)]
    (is (= 3 nlev) "detail 0.6 -> 1+round(1.8) = 3 levels")
    (is (= 3 (count levels)))
    (is (= 2 (:lvl (first levels))) "finest level first")
    (is (= 0 (:lvl (last levels)))  "base level last")
    (is (= -1.0 (:th (last levels))) "base keeps all cells")
    (is (= 0 (:offset (first levels))))
    (is (= (map :offset levels) (reductions + 0 (butlast cells))) "cumulative finest-first offsets")
    (is (= total (reduce + cells)) "total = Σ candidate cells")
    (is (approx= 1e-9 0.475 warp) "warp = 0.95 * curvature")
    (is (approx= 1e-6 (:ssz (last levels)) (* 2.0 (:ssz (nth levels 1)))) "base stdev = 2× next-finer")))

(deftest contrast-brightens-highlights
  ;; (0.7-0.5)*2.0+0.5 = 0.9
  (let [img (gray-img 16 16 (fn [_ _] 0.7))
        {:keys [splats]} (seed/splat-field img {:count 16 :contrast 2.0 :sharpness 0.0 :detail 0.0 :variation 0.0})
        [r _ _] (:color (first splats))]
    (is (approx= 1e-6 0.9 r))))

(deftest palette-limits-colors
  ;; two-region image, palette 2 => at most 2 distinct splat colors
  (let [img (gray-img 24 24 (fn [x _] (if (< x 12) 0.2 0.8)))
        {:keys [splats]} (seed/splat-field img {:count 200 :palette 2 :sharpness 0.0 :detail 0.0 :variation 0.0})
        distinct-colors (set (map :color splats))]
    (is (<= (count distinct-colors) 2))))

(deftest region-color-blends-are-in-range
  ;; sanity: all channels stay within 0..1 on a real gradient image
  (let [img (gray-img 32 32 (fn [x y] (/ (double (+ x y)) 64.0)))
        {:keys [splats]} (seed/splat-field img {:count 256 :sharpness 0.8})]
    (is (every? (fn [{[r g b] :color}] (and (<= 0.0 r 1.0) (<= 0.0 g 1.0) (<= 0.0 b 1.0))) splats))))

(deftest detail-makes-more-splats-in-texture
  ;; half-flat, half-checkerboard. detail>0 subdivides cells in the textured half,
  ;; producing more splats than detail=0 (uniform grid, 1 splat per cell).
  (let [img (gray-img 32 32 (fn [x y]
                              (if (< x 16)
                                0.5                                       ; flat top half
                                (if (odd? (+ (int x) (int y))) 0.0 1.0)))) ; checkerboard bottom half
        ;; small min so the coarse base has room to subdivide in the textured half
        cnt-detail-0 (count (:splats (seed/splat-field img {:count 100 :size 1.0 :detail 0.0 :variation 0.0})))
        cnt-detail-1 (count (:splats (seed/splat-field img {:count 100 :size 1.0 :detail 1.0 :variation 0.0})))]
    (is (> cnt-detail-1 cnt-detail-0)
        (str "detail=1 should produce more splats than detail=0: " cnt-detail-1 " vs " cnt-detail-0))))
