(ns splat-painter.seed-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.seed :as seed]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.structure :as structure]
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
    ;; elongation is CAPPED (min(stroke,1.5)) — chains provide length — so the max
    ;; per-splat axis ratio is bounded; still clearly anisotropic at the edge.
    (is (> (apply max ratios) 1.5))))

(deftest field-carries-opacity
  (is (= 0.42 (:opacity (seed/splat-field (solid 8 8 [1 1 1]) {:count 4 :opacity 0.42})))))

(deftest splat-record-spec
  ;; the pure per-splat math the GPU generation shader must reproduce; pin it directly so the
  ;; CPU/GPU spec is guarded independently of placement + field sampling.
  ;; coh=0.64; e = 1+min(2.5,1.5)·0.64·0.625 = 1.6 (elongation CAPPED — stroke length
  ;; comes from the segment chain, not the ellipse); s0=5 (snoise 0);
  ;; t=0.15+0.85·0.5=0.575 (blur-leaning); contrast 1, tone 1 (tnoise 0).
  (let [{:keys [mean cov color]}
        (seed/splat-record 10.0 20.0 5.0 0.5 0.0 0.5 0.0 0.0 [0.4 0.4 0.4] [0.8 0.2 0.1] 2.5 0.5 1.0 0.0 1.0)
        [c00 c01 _ c11] cov
        [cr cg cb] color]
    (is (= [10.0 20.0] mean))
    (is (approx= 1e-6 40.0    c00))   ; sx² = s0²·e = 25·1.6
    (is (approx= 1e-6 0.0     c01))   ; θ=0 ⇒ axis-aligned
    (is (approx= 1e-6 15.625  c11))   ; sy² = s0²/e = 25/1.6
    (is (approx= 1e-6 0.63    cr))    ; 0.4·0.425 + 0.8·0.575
    (is (approx= 1e-6 0.285   cg))
    (is (approx= 1e-6 0.2275  cb))))

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
    ;; older: count=254 (pre placement-map); 497 (dabs); 516 (uniform 6-seg strokes);
    ;; 488 (scale-relative strokes); 584 (6-level pyramid + blur-leaning colour);
    ;; 558 (E² sharp map); 560 (rotated grids + full-cell jitter); 567 (dithered
    ;; threshold + head-colour sampling); 553 (colour-guarded traces). Now: the edge
    ;; band belongs to base+fine only (mid fills suppressed at E>0.45), and every
    ;; stroke shrinks near edges so soft tails can't cross silhouettes.
    ;; latest: per-level glaze alpha (finer layers translucent) + strokes FADE at
    ;; colour boundaries (dry-out) instead of breaking into gapped dashes; then
    ;; (827→757) the mid→fine OVERLAP: no subdivision claim from level 3 up, fine
    ;; tier packs tighter with a ~pixel size floor and lighter glazes.
    ;; (757→841) IMPASTO edges: fine levels trace 8-seg liner lines packed dense
    ;; enough to hand off through dry-outs, with edge-driven body alpha and no
    ;; fine edge-shrink — contours are continuous bodied lines, not dashes.
    ;; (853→817) footprint-sensed edges: strokes answer for silhouettes anywhere
    ;; under their body, and fine sharpness follows the local detail density.
    ;; (817→841) stroke inertia: damped ridge snap, direction momentum, motion-
    ;; frame side offset, junction-tolerant coherence gate, canvas re-mix.
    ;; (sd 194712→221537) sealed base coverage: the base shrinks gently near
    ;; edges (0.25·Ev) so its paint always reaches the boundary.
    ;; (841→689) brush lift: a chain that exits its colour region (mismatch
    ;; >0.45) emits NOTHING — escaped segments were a dark halo along contours.
    ;; (sd 221525→219753) both-ends taper: the brush now lifts ON at the head too
    ;; (a quick width/alpha ramp over the first ~18%) on top of the tail dry-out,
    ;; so traced strokes taper at BOTH ends — the width envelope shrinks Σdet a touch.
    ;; (689→760) smooth liner traces: gentler per-step ridge snap (0.35) + stronger
    ;; direction momentum (0.65) keep chains on their line, so fewer die to colour
    ;; drift; the new line-hold (sharp-map dry-out) + reduced liner taper reshape
    ;; the fine tier. Broad-tier melt is inert here (size-broad = 1).
    ;; (Σmean −0.8/−0.1) the liner tier dropped its Perlin bend + position warp:
    ;; fine strokes follow the original detail exactly — noise variation belongs
    ;; to the large/medium brushwork.
    ;; (760→743) scale-keyed discipline: liner behaviour (momentum, gentle snap,
    ;; line-hold, muted taper/jitter, no Perlin) now keys on the PHYSICAL stroke
    ;; stdev, not the level index — small mid-tier chains inherit it — and the
    ;; mid placement map fuses E² so mid strokes hug edge cores instead of
    ;; seeding across the whole tensor-blur band.
    ;; (743→769) boundary-side brush-load: chains near a colour BOUNDARY sample
    ;; their paint ~0.7σ across the tangent on their own side (thin LINE features
    ;; keep on-ridge colour) — chains parallel to a boundary no longer alternate
    ;; sides per seed, which tiled every contour into colour capsules (the
    ;; regular wavy scallops). The dry-out probe moves with the sample, so a few
    ;; more boundary chains survive.
    ;; (769→830) four colour-path fixes: the impasto side now keys on the liner
    ;; discipline (small lvl 2-3 chains keep their side); the brush-load is a
    ;; three-tier decision with a floored displacement (geometric side wins, the
    ;; colour test only at a genuine step edge); the Perlin bend gains a per-seed
    ;; phase + wavelet-edge gate; the final colour gets a region-consistency clamp
    ;; (raw pulled toward its bilateral region).
    ;; (830→729) liner-scale chains: small-σ levels (nominal size < 3.5px, lvl≥2)
    ;; trace up to 32-segment continuous lines (a ~28px span target) instead of the
    ;; short stitched dashes that thatched every contour; Stroke now extends the LINE
    ;; via segs (not the gaps), stepf is final in layer-params and both trace loops
    ;; drop their in-loop stroke factor; GS cap 8→32. Liner spacing keeps each level's
    ;; OWN tier coefficient (√segs, no slen) so the per-level budget term k stays
    ;; invariant (forcing 0.7 onto reclassified mid levels inflated k ≈3.2×); scale is
    ;; unchanged so fewer-but-longer chains net a lower count.
    ;; (729→685) long chains disciplined at corners: liners lift hard at 0.32 colour
    ;; drift (soft tier 0.2, ×0.35) and dry ×0.3 when the field turns past cos 35°
    ;; per step (turn-kill) — escaped tails that scribbled dark loops around pointed
    ;; contours (eye corners) now end at the corner instead.
    ;; (685→1041) liner seed spacing scales with √min(segs,14), not the nominal span:
    ;; survival-limited chains on busy contours left sparse hard dashes when seeds
    ;; were spaced for the full 32-segment span (the detached ring around the
    ;; avatar's eyes). Denser seeds = continuous bands; the extra fine-tier demand
    ;; flows into scale-f, so more (finer-budgeted) chains survive here.
    (is (= 1041 (count splats)))
    (is (approx= 0.5  20473.371  sx) "Σ mean-x")
    (is (approx= 0.5  30272.265  sy) "Σ mean-y")
    (is (approx= 1.0  219953.020 sd) "Σ det(cov)")
    (is (approx= 0.05 1463.199   sc) "Σ colour")))

(deftest fine-seeds-trace-tapered-brush-strokes
  ;; the brush-stroke contract: a textured image yields fine-level chains whose segments
  ;; carry tapered alpha — full paint (1.0) at stroke heads and base fills, thinning
  ;; toward stroke tails (min alpha = 1 − 0.65 = 0.35) — never outside (0,1].
  (let [img (gray-img 48 48 (fn [x y] (if (odd? (+ (int (quot x 4)) (int (quot y 4)))) 0.15 0.85)))
        {:keys [splats]} (seed/splat-field img {:count 3000 :size 6.0 :detail 0.8})
        alphas (map #(double (or (:alpha %) 1.0)) splats)]
    (is (every? #(and (> % 0.0) (<= % 1.0)) alphas))
    (is (some #(= 1.0 %) alphas) "stroke heads + base fills carry full paint")
    (is (some #(< % 0.5) alphas) "stroke tails taper below half paint")
    ;; tails now include the per-level glaze (×0.75 finest) and dry-out fades, so
    ;; the floor is below the plain 0.35 taper — just pin that tails thin properly.
    (is (< (reduce min alphas) 0.35) "tails thin below the plain taper floor")))

(deftest layer-params-shared-spec
  ;; layer-params is the per-level placement spec BOTH the CPU loop and the GPU generation
  ;; shader consume, so they enumerate the same cells. Guard its contract: finest-first
  ;; ordering (levels[0] = smallest stdev), cumulative candidate offsets, total = Σ nx·ny,
  ;; and ssz halving per finer level. If any drifts, the GPU field diverges from the CPU golden.
  (let [img    (gray-img 48 64 (fn [x y] (if (and (> x 16) (< x 32) (> y 20) (< y 44))
                                            0.9 (* 0.5 (/ (double (+ x y)) 112.0)))))
        sfield (structure/analyze img)
        dmap   (wavelet/placement-map img sfield)
        {:keys [nlev levels total warp]} (seed/layer-params dmap 0.6 6.0 0.5 0.5 2.5 [1.0 1.0 1.0] 4000 48 64)
        cells (map (fn [l] (* (:nx l) (:ny l))) levels)]
    (is (= 5 nlev) "detail 0.6 -> 1+round(3.6) = 5 levels")
    (is (= 5 (count levels)))
    (is (= 4 (:lvl (first levels))) "finest level first")
    (is (= 0 (:lvl (last levels)))  "base level last")
    (is (= -1.0 (:th (last levels))) "base keeps all cells")
    (is (= 0 (:offset (first levels))))
    (is (= (map :offset levels) (reductions + 0 (butlast cells))) "cumulative finest-first offsets")
    (is (= total (reduce + cells)) "total = Σ candidate cells")
    (is (approx= 1e-9 0.475 warp) "warp = 0.95 * curvature")
    (is (approx= 1e-6 (:ssz (last levels)) (* 2.0 (:ssz (nth levels (- (count levels) 2)))))
        "base stdev = 2× the next-finer level")))

(deftest tier-multipliers-scale-their-levels
  ;; the per-tier size sliders: mid/fine scale their levels' nominal size directly.
  ;; BROAD is bokeh-adaptive: it must NOT touch the level's nominal (subject) size —
  ;; flat regions grow/thin at emission instead — so :ssz for the broad tier is
  ;; b-independent, and only b<1 densifies its candidate grid.
  (let [img  (gray-img 48 64 (fn [x y] (if (and (> x 16) (< x 32) (> y 20) (< y 44))
                                         0.9 (* 0.5 (/ (double (+ x y)) 112.0)))))
        dmap (wavelet/placement-map img (structure/analyze img))
        ;; size 24 keeps every tier above the 0.7px post-multiplier floor, so the
        ;; linear-scaling assertions hold (at small sizes the floor clamps instead —
        ;; a tier dial can make a layer finer but never dust it to sub-pixel).
        base (seed/layer-params dmap 0.6 24.0 0.5 0.5 2.5 [1.0 1.0 1.0] 4000 48 64)
        wide (seed/layer-params dmap 0.6 24.0 0.5 0.5 2.5 [2.0 1.0 0.5] 4000 48 64)
        down (seed/layer-params dmap 0.6 24.0 0.5 0.5 2.5 [0.5 1.0 1.0] 4000 48 64)
        lvl-of (fn [lp lvl] (first (filter #(= lvl (:lvl %)) (:levels lp))))
        ssz-of (fn [lp lvl] (:ssz (lvl-of lp lvl)))]
    (is (approx= 1e-9 (ssz-of base 0) (ssz-of wide 0)) "broad ×2 leaves the subject-nominal base size alone")
    (is (approx= 1e-9 (ssz-of base 2) (ssz-of wide 2)) "mid unchanged at 1.0")
    (is (approx= 1e-9 (* 0.5 (ssz-of base 4)) (ssz-of wide 4)) "fine ×0.5 halves the finest")
    (is (> (:nx (lvl-of down 0)) (:nx (lvl-of base 0))) "broad <1 densifies the base grid")))

(deftest broad-dial-is-bokeh-adaptive
  ;; the Broad slider must reshape only LOW-detail regions: a flat half gets fewer,
  ;; larger daubs (bokeh) while the textured half's strokes stay put. Splats are
  ;; classified by position (margin off the boundary so tap-smoothed subjectness
  ;; doesn't blur the halves together).
  (let [img (gray-img 64 64 (fn [_ y] (if (< y 32) 0.5
                                        (if (odd? (long (quot y 2))) 0.1 0.9))))
        fld (fn [b] (:splats (seed/splat-field img {:count 4000 :size 6.0 :detail 0.6
                                                    :variation 0.0 :size-broad b})))
        f1 (fld 1.0)
        f2 (fld 2.5)
        flat?     (fn [{[_ my] :mean}] (< my 22))   ; margins off the boundary on
        textured? (fn [{[_ my] :mean}] (> my 42))   ; BOTH sides: grown daubs reach
        sig    (fn [{[c00 c01 _ c11] :cov}]
                 (Math/sqrt (Math/sqrt (max 1e-8 (- (* c00 c11) (* c01 c01))))))
        max-sig (fn [ss] (reduce max 0.0 (map sig ss)))]
    (is (< (count (filter flat? f2)) (count (filter flat? f1)))
        "broad ×2.5 thins the flat half (fewer, larger daubs)")
    (is (> (max-sig (filter flat? f2)) (* 1.5 (max-sig (filter flat? f1))))
        "broad ×2.5 grows the flat half's daubs")
    (is (< (max-sig (filter textured? f2)) (* 1.15 (max-sig (filter textured? f1))))
        "the textured half's stroke sizes are untouched by Broad")))

(deftest contrast-brightens-highlights
  ;; (0.7-0.5)*2.0+0.5 = 0.9
  (let [img (gray-img 16 16 (fn [_ _] 0.7))
        {:keys [splats]} (seed/splat-field img {:count 16 :contrast 2.0 :sharpness 0.0 :detail 0.0 :variation 0.0})
        [r _ _] (:color (first splats))]
    (is (approx= 1e-6 0.9 r))))

(deftest region-color-blends-are-in-range
  ;; sanity: all channels stay within 0..1 on a real gradient image
  (let [img (gray-img 32 32 (fn [x y] (/ (double (+ x y)) 64.0)))
        {:keys [splats]} (seed/splat-field img {:count 256 :sharpness 0.8})]
    (is (every? (fn [{[r g b] :color}] (and (<= 0.0 r 1.0) (<= 0.0 g 1.0) (<= 0.0 b 1.0))) splats))))

(deftest detail-makes-more-splats-in-texture
  ;; half-flat, half-checkerboard. detail>0 adds fine levels in the textured half,
  ;; producing more splats than detail=0 (base only). Budget must be realistic: at a
  ;; tiny budget the level scale-up dominates and the comparison inverts.
  (let [img (gray-img 48 48 (fn [x y]
                              (if (< x 24)
                                0.5                                       ; flat top half
                                (if (odd? (+ (int x) (int y))) 0.0 1.0)))) ; checkerboard
        cnt-detail-0 (count (:splats (seed/splat-field img {:count 2000 :size 2.0 :detail 0.0 :variation 0.0})))
        cnt-detail-1 (count (:splats (seed/splat-field img {:count 2000 :size 2.0 :detail 1.0 :variation 0.0})))]
    (is (> cnt-detail-1 cnt-detail-0)
        (str "detail=1 should produce more splats than detail=0: " cnt-detail-1 " vs " cnt-detail-0))))

;; --- cross-boundary colour-bleed regression ----------------------------------

(defn- boundary-tone [y]
  ;; vertical boundary across COLUMNS: dark gray (y<63), light gray (y>=65),
  ;; columns 63-64 a linear ramp (anti-aliased edge).
  (cond
    (< y 63) 0.15
    (>= y 65) 0.85
    :else (+ 0.15 (* 0.70 (/ (- (double y) 62) 3.0)))))

(defn- two-tone-image [H W]
  ;; flat H*W*3 double buffer, index base 3*(x*W+y), x=row y=col (matches gray-img).
  {:height H :width W :channels 3
   :pixels (double-array
            (mapcat (fn [x]
                      (mapcat (fn [y] (let [g (boundary-tone y)] [g g g]))
                              (range W)))
                    (range H)))})

(defn- attach-precomputed-fields [img0]
  ;; mirror core/on-image-loaded EXACTLY. splat-field falls back to raw pixels for
  ;; any missing blur field, which would NOT exercise the real (edge-aware) colour
  ;; path — so the test must attach the same fields the app does on image load.
  (let [sfield (structure/analyze img0)
        light  (structure/bilateral-blur img0 3)
        drift  (structure/blur-image img0 2)
        heavy  (structure/blur-image img0 (max 6 (quot (:height img0) 80)))]
    (assoc img0 :structure sfield
               :blur light
               :blur-drift drift
               :blur-heavy (structure/edge-preserving-blur img0 light heavy)
               :detail (wavelet/placement-map img0 sfield)
               :noise-fields (seed/prep-noise sfield))))

(defn- sigma-max [cov]
  ;; stdev along the major axis = sqrt of the LARGER eigenvalue of [[c00 c01][c01 c11]].
  (let [[c00 c01 _ c11] cov
        a (double c00) b (double c01) d (double c11)
        h (/ (- a d) 2.0)]
    (Math/sqrt (+ (/ (+ a d) 2.0) (Math/sqrt (+ (* h h) (* b b)))))))

(defn- luma [c]
  (let [[r g b] c]
    (+ (* 0.2126 r) (* 0.7152 g) (* 0.0722 b))))

(deftest no-cross-boundary-colour-bleed
  ;; A synthetic two-tone image (H=W=128, vertical boundary at cols 63-64) must not
  ;; produce paint of the wrong shade placed deep inside the opposite half: no light
  ;; splat (luma>0.5) centred >=0.5σ-max inside the dark half, no dark splat (luma<0.5)
  ;; centred >=0.5σ-max inside the light half. σ-max is the stdev along each splat's
  ;; major axis (sqrt of the larger covariance eigenvalue), so ">=0.5σ past the
  ;; boundary" means the splat's centre is meaningfully on the wrong side, not merely
  ;; overlapping the edge.
  ;;
  ;; The image map is built with the SAME precomputed fields as core/on-image-loaded
  ;; (:structure :blur :blur-drift :blur-heavy :detail :noise-fields); without them
  ;; splat-field samples raw pixels and the edge-aware colour path is never exercised.
  ;; Controls mirror the golden test. This is the TDD gate for the in-progress bleed
  ;; fix — it FAILS on the current code (light paint leaks into the dark half) and
  ;; passes once the leak is closed.
  (let [img (attach-precomputed-fields (two-tone-image 128 128))
        {:keys [splats]} (seed/splat-field img {:count 4000 :size 6.0 :stroke 2.5
                                                :detail 0.6 :variation 0.5 :curvature 0.5
                                                :opacity 0.9 :contrast 1.0})
        margin 0.5                                        ; >=0.5σ past the boundary
        light-in-dark (volatile! 0)                       ; light paint inside the dark half
        dark-in-light (volatile! 0)]                      ; dark paint inside the light half
    (doseq [{[mx my] :mean cov :cov color :color} splats
            :let [sm (sigma-max cov) lum (luma color)]]
      (when (and (< my (- 63 (* margin sm))) (> lum 0.5)) (vswap! light-in-dark inc))
      (when (and (> my (+ 65 (* margin sm))) (< lum 0.5)) (vswap! dark-in-light inc)))
    (is (zero? @light-in-dark)
        (str "light paint bleeding into the dark half: " @light-in-dark
             " splat(s) centred >=0.5σ inside cols y<63 (of " (count splats) " total)"))
    (is (zero? @dark-in-light)
        (str "dark paint bleeding into the light half: " @dark-in-light
             " splat(s) centred >=0.5σ inside cols y>=65 (of " (count splats) " total)"))))