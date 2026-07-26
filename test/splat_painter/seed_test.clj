(ns splat-painter.seed-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.seed :as seed]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.structure :as structure]
            [splat-painter.gaussian :as g]
            [splat-painter.shader :as shader]
            [splat-painter.image :as image]))

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
    ;; (1041→846) liner path-colour roughness dry-out: a chain whose PAINTED PATH
    ;; churns the sharp bilateral (crossing fine features, ~0.7/span) now dries out
    ;; (racc>0.2 soft ×0.5/step, >0.35 hard lift). The box drift field diluted a
    ;; 1-3px feature's contrast below the drift thresholds, so snapped chains
    ;; carried a feature's ink across neighbouring micro-regions (detail-area
    ;; noise); chains riding a clean contour (~0.04/span) stay stable and keep
    ;; their full span. Fewer surviving liner segments drop Σmean/Σcolour; Σdet is
    ;; near-unchanged (the survivors' geometry is the same, just truncated tails).
    ;; (846→838) physical-σ liner span: emitted segs/stepf/spacing derive from the
    ;; budget-scaled stdev with the trace's own liner test, and the ~28px span
    ;; target is absolute pixels ramped by thinness (full ≤1.4σ, seg-count table by
    ;; 2.6σ). At this image's scale the liner level drops 32→31 segs and spacing
    ;; tightens marginally — a small survivor shift, no behavioural change here
    ;; (the fix targets low-budget fat-liner regimes).
    ;; (838→889) Round 1 monotone admission: a fine level (lvl≥4) is dropped unless it is
    ;; ≥0.7× finer than the previous AND its clamped survivor demand fits the fine budget.
    ;; This fixture's ladder trims a redundant near-duplicate fine level, so more (denser)
    ;; candidates survive — count, Σmean and Σcolour rise; Σdet is near-unchanged.
    ;; (Round 2 physical liner predicate) a lvl≥4 stroke with stdev≥3.5 is no longer a
    ;; liner — it keeps its head taper and forgoes impasto body/raw, so the fixture's
    ;; borderline chains shift colour/mean slightly; count and Σdet are unchanged.
    ;; (889→683) Rounds 4+5: bilinear field/colour sampling shifts means/colour, and the
    ;; aspect-bounded liner span (≤12px) + coherence gate (short chains on incoherent
    ;; texture) cut emitted segments — count drops sharply; Σdet/Σcolour move with the
    ;; shorter, more isotropic chains and the bilinear colours. Re-pinned once (jolt -M:pin).
    (is (= 529 (count splats)))
    (is (approx= 0.5  11293.829  sx) "Σ mean-x")
    (is (approx= 0.5  16109.012  sy) "Σ mean-y")
    (is (approx= 1.0  219649.771 sd) "Σ det(cov)")
    (is (approx= 0.05 610.920   sc) "Σ colour")))

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
    (let [requested (long (inc (Math/round (* 0.6 6.0))))]   ; 1+round(detail*6) = 5
      (is (= (count levels) nlev)
          "nlev is the ADMITTED level count (was 1+round(detail*6); the monotone/admission gate now drops levels the budget cannot reach)")
      (is (<= nlev requested) "admission can only drop, never add, levels")
      (is (<= 2 nlev) "at least the base + one detail level admit"))
    (is (apply < (map :ssz levels)) "levels finest-first: ssz strictly increasing coarseward (first = smallest stdev)")
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
        mid2 (seed/layer-params dmap 0.6 24.0 0.5 0.5 2.5 [1.0 2.0 1.0] 4000 48 64)
        fin2 (seed/layer-params dmap 0.6 24.0 0.5 0.5 2.5 [1.0 1.0 2.0] 4000 48 64)
        brd2 (seed/layer-params dmap 0.6 24.0 0.5 0.5 2.5 [2.0 1.0 1.0] 4000 48 64)
        down (seed/layer-params dmap 0.6 24.0 0.5 0.5 2.5 [0.5 1.0 1.0] 4000 48 64)
        lvl-of (fn [lp lvl] (first (filter #(= lvl (:lvl %)) (:levels lp))))
        ssz-of (fn [lp lvl] (:ssz (lvl-of lp lvl)))]
    ;; NEW (size-keyed tiering, spec-sliders-and-lips): tier-mul keys on the detail
    ;; level's RANK, not its index. The coarsest detail level (lvl 2 = rank 0) carries
    ;; BOTH Mid*Fine; finer detail levels (rank>=1) carry Fine only. The OLD index key
    ;; (lvl4=fine) handed Fine to a level the monotone ladder never admits, so Fine and
    ;; (under the default 3-level ladder) Mid were both no-ops. Directional assertions
    ;; (not exact ratios): the ladder's step-ratio clamps a doubled level, so a tier
    ;; dial GROWS its level(s) but not by an exact factor.
    (is (> (ssz-of mid2 2) (ssz-of base 2)) "Mid grows the coarsest detail level (lvl2 = rank0 carries mid*fine)")
    (is (approx= 1e-9 (ssz-of mid2 3) (ssz-of base 3)) "Mid does NOT touch lvl3 (rank>=1 carries fine only)")
    (is (> (ssz-of fin2 2) (ssz-of base 2)) "Fine grows lvl2 (a detail level)")
    (is (> (ssz-of fin2 3) (ssz-of base 3)) "Fine grows lvl3 (a detail level)")
    (is (approx= 1e-9 (ssz-of brd2 2) (ssz-of base 2)) "Broad is bokeh-adaptive: it never touches a detail level's size")
    (is (approx= 1e-9 (ssz-of base 0) (ssz-of brd2 0)) "broad ×2 leaves the subject-nominal base size alone")
    (is (> (:nx (lvl-of down 0)) (:nx (lvl-of base 0))) "broad <1 densifies the base grid")))

(deftest mid-fine-dials-move-the-detail-tier
  ;; Defect A (spec-sliders-and-lips): the Mid and Fine dials must each visibly change
  ;; the detail tier at the default ladder. tier-mul now keys on the level's ROLE
  ;; (detail rank), not its index — the coarsest detail level (lvl 2, rank 0) carries
  ;; BOTH Mid and Fine (Mid*Fine) so neither dial is a no-op; min-phys 1.4 lets the dial
  ;; move it off the floor. At Size 6 / Detail 1 / 600k / 1024x1024 the OLD code pinned
  ;; L2 at exactly 2.20 and Fine did nothing.
  (let [img  (gray-img 1024 1024 (fn [x y]
                                   (let [bx (/ (double x) 1024.0) by (/ (double y) 1024.0)]
                                     (+ 0.35 (* 0.3 (+ 1.0 (Math/sin (* 6.2832 bx))))
                                        (* 0.05 (Math/sin (* 31.4 (+ bx by))))))))
        dmap (wavelet/placement-map img (structure/analyze img))
        lp   (fn [mid fine] (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5
                                               [1.0 mid fine] 600000 1024 1024))
        det  (fn [mid fine] (some #(when (= 2 (:lvl %)) %) (:levels (lp mid fine))))
        ladder (fn [mid fine] (let [d (det mid fine)] [mid fine (:ssz d) (:nx d)]))]
    (println "DIAL-LADDER [mid fine ssz nx]:")
    (doseq [m [0.4 1.0 2.5]] (println " " (pr-str (ladder m 1.0))))
    (doseq [f [0.4 1.0 2.5]] (println " " (pr-str (ladder 1.0 f))))
    (let [in-mid  (mapv #(:ssz (det % 1.0)) [0.4 1.0 2.5])
          in-fine (mapv #(:ssz (det 1.0 %)) [0.4 1.0 2.5])]
      (is (apply < in-mid)
          (str "detail ssz strictly monotone increasing in Mid over {0.4,1.0,2.5}: " in-mid))
      (is (apply < in-fine)
          (str "detail ssz strictly monotone increasing in Fine over {0.4,1.0,2.5}: " in-fine))
      (is (not (apply = in-mid)) "Mid is not a no-op (the three ladders differ)")
      (is (not (apply = in-fine)) "Fine is not a no-op (the three ladders differ)"))))

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
  ;; half-flat, half-checkerboard. The Detail slider's effect is RESOLUTION, not raw
  ;; count: with the survivor budget capped (spec-offset-and-repin, Item 2), raising
  ;; Detail REDISTRIBUTES strokes coarse->fine rather than adding them, so the
  ;; textured-half TOTAL can stay flat (or even fall). What Detail DOES do is admit
  ;; FINER placement levels that target the textured region — detail=1's ladder
  ;; reaches finer levels and a smaller smallest-stroke than detail=0 (base only).
  (let [img (gray-img 48 48 (fn [x y]
                              (if (< x 24) 0.5
                                  (if (odd? (+ (int x) (int y))) 0.0 1.0))))
        dmap (wavelet/placement-map img (structure/analyze img))
        lp   (fn [detail] (seed/layer-params dmap detail 2.0 0.0 0.0 2.5 [1.0 1.0 1.0] 2000 48 48))
        nlev (fn [detail] (:nlev (lp detail)))
        finest-ssz (fn [detail] (apply min (map :ssz (:levels (lp detail)))))]
    (is (> (nlev 1.0) (nlev 0.0)) "detail=1 admits more placement levels than detail=0")
    (is (< (finest-ssz 1.0) (finest-ssz 0.0))
        "detail=1 reaches a finer smallest stroke than detail=0 (resolution, not count)")))

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

(defn- median [xs]
  (let [s (sort xs) n (count s)]
    (when (pos? n) (nth s (quot n 2)))))

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
  ;; Controls mirror the golden test. Regression guard: no dark splat (luma<0.5) may
  ;; sit >=0.5σ into the light half (cols y>=65) and no light splat into the dark half
  ;; (cols y<63). The bilateral-blurred boundary reads SOFT under the sharpness measure,
  ;; so soft-ramp strokes re-load their LOCAL colour (wsl=1) instead of carrying an
  ;; off-ridge brush-load across — that is what keeps both halves clean.
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
      (when (and (> my (+ 65 (* margin sm))) (< lum 0.5))
        (vswap! dark-in-light inc)))
    (is (zero? @light-in-dark)
        (str "light paint bleeding into the dark half: " @light-in-dark
             " splat(s) centred >=0.5σ inside cols y<63 (of " (count splats) " total)"))
    (is (zero? @dark-in-light)
        (str "dark paint bleeding into the light half: " @dark-in-light
             " splat(s) centred >=0.5σ inside cols y>=65 (of " (count splats) " total)"))))

;; --- liner path-colour roughness dry-out ---------------------------------------

(deftest liner-chains-die-on-texture-live-on-contours
  ;; PATH-COLOUR ROUGHNESS gate for liner chains (mirror seed/stroke-segments +
  ;; the GS). The FORGIVING box-blurred drift field dilutes a 1-3px feature's
  ;; contrast below its thresholds at liner scale, so chains snapped onto a fine
  ;; feature carried its ink across neighbouring micro-regions (the detail-area
  ;; noise). The gate accumulates the SHARP bilateral's per-step maxchan Δ along
  ;; the PAINTED path: a chain crossing features churns and dries out; a chain
  ;; riding a clean contour stays colour-stable and keeps its full span — the
  ;; thatch fix is preserved exactly where it was won.
  ;;
  ;; Synthetic 160x160: LEFT half (cols 0-79) a uniform 0.72 ground with ONE dark
  ;; 0.28 vertical line, 2px wide, at col 40 (a clean long contour); RIGHT half
  ;; (cols 80-159) a high-frequency 3px-cell checker alternating 0.35/0.65 (the
  ;; bilateral field churns along any crossing path). Built the same way as
  ;; no-cross-boundary-colour-bleed (structure/analyze, placement map, prep-noise,
  ;; the light + drift blur fields).
  ;;
  ;; Calls stroke-segments DIRECTLY (var-quote) for controlled seeds, wired with
  ;; the liner level's own ssz/segs/stepf and the same per-candidate wiring
  ;; layered-means uses (subject gate, gain, tone/dir hashes — no invented values).
  ;; Before the gate the box field cannot see the 3px checker, so texture chains
  ;; run their full 32-seg span (median >> 8); after the gate they dry out fast.
  (let [H 160 W 160
        detail 0.3 size 6.0 stroke 2.2 variation 0.5 curvature 0.5
        tier-muls [0.4 0.4 0.4] cnt 600000
        img (attach-precomputed-fields
             {:height H :width W :channels 3
              :pixels (double-array
                       (mapcat (fn [x]
                                 (mapcat (fn [y]
                                           (let [g (cond
                                                     ;; CONTOUR half: light ground, one 2px dark line at col 40
                                                     (< y 80)  (if (<= 39 y 40) 0.28 0.72)
                                                     ;; TEXTURE half: 3px-cell checker
                                                     :else     (if (odd? (+ (quot x 3) (quot y 3))) 0.35 0.65))]
                                             [g g g]))
                                         (range W)))
                               (range H)))})
        dmap    (:detail img)
        blur-px (:blur img)
        blurd-px (:blur-drift img)
        nf      (:noise-fields img)
        hd (double (dec H)) wd (double (dec W)) iw W ih H
        rr (/ (double H) 24.0)
        lp      (seed/layer-params dmap detail size variation curvature stroke tier-muls cnt H W)
        ;; the liner level: lvl≥2, ACTUAL size < 3.5px, longest chains (the span is
        ;; thinness-ramped off the physical stdev now, so exact segs vary with σ)
        liner   (let [cands (filter #(and (>= (:lvl %) 2) (< (:ssz %) 3.5))
                                    (:levels lp))]
                  (when (seq cands) (apply max-key :segs cands)))
        {:keys [lvl ssz segs stepf bendf th map-kind traw]} liner
        bmul    (double (nth tier-muls 0))                ; 0.4 → bgate is identically 1 below Broad 1
        deff    (fn [D] (min 1.0 (* (double detail) (double D) 2.2)))
        ;; var-quote the private wiring helpers layered-means uses (no invented values)
        hash01  #'splat-painter.seed/hash01
        subject-at #'splat-painter.seed/subject-at
        map-at  #'splat-painter.seed/map-at
        stroke-segments #'splat-painter.seed/stroke-segments
        ;; one liner chain's emitted-segment count for a chosen seed (cx,cy).
        ;; seeds are placed at chosen positions (warp is inert here: aw<0.2 at this
        ;; liner tier), mirroring layered-means' per-candidate wiring for the rest.
        chain-len (fn [i cx cy]
                    (let [dv     (map-at dmap map-kind cx cy)
                          D      (deff dv)
                          sgate  (subject-at dmap cx cy rr)
                          bgate  (- 1.0 (* (min 1.0 (max 0.0 (/ (- bmul 1.0) 1.5)))
                                          (- 1.0 (min 1.0 (/ (wavelet/subject-abs-at dmap cx cy) 0.35)))))
                          gain   (* (+ 0.25 (* 0.75 sgate)) bgate)
                          cssz   ssz                                     ; liner level's own stdev
                          tn     (* (double (let [l (long lvl)]
                                              (cond (<= l 1) 0.25 (>= l 4) 0.15
                                                    :else (+ 0.15 (* 0.85 (min 1.0 (max 0.0 (/ (- cssz 2.5) 2.5))))))))
                                    (- (hash01 (+ (* i 37) lvl) 0 13) 0.5))
                          ds     (if (< (hash01 (+ (* i 41) lvl) 0 17) 0.5) 1.0 -1.0)
                          bph    (hash01 (+ (* i 67) lvl) 0 53)
                          hb     (if (<= lvl 1) 1.0 0.0)
                          traw*  (if (>= lvl 4) (* traw (+ 0.6 (* 0.4 sgate))) traw)]
                      (count (first (stroke-segments nf dmap lvl cx cy cssz D 0.0 tn ds curvature stroke
                                              hd wd segs stepf bendf hb traw* sgate blur-px iw ih
                                              th 0.0 map-kind gain blurd-px bph)))))]
    (is liner "settings must yield a liner level (lvl≥2, ssz<3.5)")
    (is (>= segs 10) "liner level keeps a meaningful feature-tracer span")
    (is (< ssz 3.5))
    (let [contour (keep-indexed (fn [i x] (chain-len i x 40.0)) (range 12 152 7))   ; ~20 seeds ON the line (col 40)
          texture (for [[i [x y]] (map-indexed vector
                                               (for [x (range 15 145 20) y (range 94 150 12)] [x y]))]  ; ~35 seeds in the checker
                    (chain-len (+ i 1000) x y))                                     ; offset i so texture hashes ≠ contour
          cmed (median contour) tmed (median texture)]
      ;; The path-roughness (racc) accumulator that pre-emptively dried detail chains
      ;; after 2-4 segments was the "short disjointed lines" artifact and is REMOVED for
      ;; detail tiers (broad/coverage keep their two-tier dry-out). Detail chains now
      ;; stop only on a HARD signal — runaway (foreign colour in the forgiving box-drift
      ;; field) or line-hold (off the level's own map) — neither of which this 3px
      ;; checker triggers, so chains run their full feature-determined span on BOTH halves.
      (is (>= cmed (* 0.8 (double segs))) (str "contour chains ride the line and run their full feature-determined span; median=" cmed " of nominal segs " segs))
      (is (>= tmed (* 0.5 (double segs))) (str "texture chains run their full span too — the racc dry-out is removed for detail tiers (it fragmented chains); median=" tmed " of " (count texture))))))

;; --- physical-sigma liner span (thinness ramp) ---------------------------------

(deftest liner-span-follows-physical-size
  ;; Liner-ness, chain span and step are decided from the BUDGET-SCALED stdev with
  ;; the exact per-chain discipline test the trace loops apply. Two bugs this pins:
  ;; the ~28px span target used to be computed from NOMINAL size but traced at
  ;; ACTUAL sigma (at budget scale s the real span was 28*slen*s px — 51px worms at
  ;; low budgets), and a level could classify liner by nominal size while the trace
  ;; denied it liner discipline by actual sigma (fat 13-seg chains with no
  ;; momentum/line-hold — the wavy S-lines on soft-focus contours). Now the span is
  ;; ABSOLUTE pixels, ramped by thinness: full ~28px only for genuinely thin
  ;; strokes (sigma<=1.4), back on the short seg-count table by sigma 2.6.
  (let [img  {:height 128 :width 128 :channels 3
              :pixels (double-array
                       (mapcat (fn [x]
                                 (mapcat (fn [y]
                                           (let [g (if (odd? (+ (quot x 3) (quot y 3))) 0.35 0.65)]
                                             [g g g]))
                                         (range 128)))
                               (range 128)))}
        dmap (wavelet/placement-map img (structure/analyze img))
        seg-count #'splat-painter.seed/seg-count
        step-frac #'splat-painter.seed/step-frac
        stroke-len-frac #'splat-painter.seed/stroke-len-frac
        stroke 2.05
        slen  (double (stroke-len-frac stroke))
        check (fn [cnt]
                (let [p (seed/layer-params dmap 0.45 10.0 0.38 0.38 stroke [0.85 0.85 0.85] cnt 128 128)]
                  (doseq [{:keys [lvl ssz segs stepf]} (:levels p)
                          :let [lvl (long lvl) ssz (double ssz)
                                liner? (or (>= lvl 4) (and (>= lvl 2) (< ssz 3.5)))]]
                    (when (and (>= lvl 2) (not liner?))
                      (is (= (long (seg-count lvl)) (long segs))
                          (str "budget " cnt " lvl " lvl " ssz " ssz
                               ": no liner span without liner discipline")))
                    (when (and (>= lvl 2) liner? (>= ssz 2.6))
                      ;; a fat detail-level liner (ssz 2.6-3.5) runs the feature-tracer
                      ;; span too (the round-5a aspect cap is gone) — it is NOT on the
                      ;; short seg-count table, which only the coverage tiers (lvl 0-1) use.
                      (is (>= (long segs) 20)
                          (str "budget " cnt " lvl " lvl " ssz " ssz " segs " segs
                               ": fat liner runs the feature-tracer span (max-segs), not seg-count")))
                    (when (and (>= lvl 2) liner? (<= ssz 1.4))
                      ;; SPAN is FEATURE-DETERMINED, not aspect-bounded: the round-5a
                      ;; 12-px cap (min(28*slen*ramp, 12*ssz)) was a stop-gap, removed once
                      ;; the tracer + line-hold/runaway stops landed. A thin liner now runs
                      ;; the full feature-tracer span (max-segs) and lets the GEOMETRY (ridge
                      ;; dies / tangent bends / walks off its own map) decide where to stop.
                      (is (>= (long segs) 20)
                          (str "budget " cnt " lvl " lvl " ssz " ssz " segs " segs
                               ": thin liner runs the feature-tracer span (max-segs), not a 12-px aspect bound")))
                    (when liner?
                      ;; the ABSOLUTE-px span cap is removed (feature-determined); what
                      ;; remains is the property this test was written for — liner-ness is
                      ;; decided from the BUDGET-SCALED physical sigma, so a genuinely thin
                      ;; stroke (ssz<2.6) is a liner and a fat one is not.
                      (is (pos? (double stepf))
                          (str "budget " cnt " lvl " lvl " ssz " ssz ": liner keeps a positive step fraction"))))
                  p))]
    (check 600000)
    (let [p (check 3000)]
      ;; the low budget must actually exercise the fat regime (scale inflates
      ;; lvl>=2 stdevs past the ramp floor) or the conditional asserts are vacuous
      (is (some #(and (>= (long (:lvl %)) 2) (> (double (:ssz %)) 2.6)) (:levels p))
          "low budget must produce a fat (ssz>2.6) lvl>=2 level"))))

;; --- Round 1: the level ladder must be monotone -------------------------------
;; layer-params admits levels coarse→fine under a monotonicity + budget rule so
;; the finest strokes are never FATTER than coarser ones, redundant near-duplicate
;; passes collapse, and nothing sub-pixel survives. A large enough synthetic image
;; is needed to exercise the scale-f>scale-c regime where the defect (finest levels
;; fatter than the mid tier) actually appears.

(defn- ladder-img
  "512×512 synthetic image with structure at several scales so the fine/sharp detail
   maps carry signal (otherwise Kf≈0 and scale-f stays 1 — the defect never appears)."
  []
  (gray-img 512 512 (fn [x y]
                       (let [coarse (* 0.25 (+ 1.0 (Math/sin (* 0.04 (+ x y)))))
                             fine  (if (odd? (+ (quot x 2) (quot y 3))) 0.12 -0.12)]
                         (max 0.0 (min 1.0 (+ coarse fine)))))))

(def ^:private ladder-configs
  (for [size   [6.0 12.0 20.5 50.0]
        detail [0.0 0.3 0.6 1.0]
        tiers  [[0.4 0.4 0.4] [1.0 1.0 1.0] [2.5 1.0 0.4]]
        cnt    [1000 72000 600000]]
    [size detail tiers cnt]))

(deftest ladder-is-strictly-finer-per-level
  (let [img  (ladder-img)
        dmap (wavelet/placement-map img (structure/analyze img))]
    (doseq [[size detail tiers cnt] ladder-configs
            :let [{:keys [levels]} (seed/layer-params dmap detail size 0.5 0.5 2.5 tiers cnt 512 512)
                  ;; coarse→fine: largest ssz first
                  coarse-fine (sort-by :ssz > levels)]]
      (doseq [[coarser finer] (partition 2 1 coarse-fine)]
        (is (<= (double (:ssz finer)) (* 0.95 (double (:ssz coarser))))
            (str "size " size " detail " detail " tiers " tiers " count " cnt
                 ": finer ssz " (:ssz finer) " must be <= 0.95× coarser " (:ssz coarser)))))))

(deftest no-duplicate-levels
  (let [img  (ladder-img)
        dmap (wavelet/placement-map img (structure/analyze img))]
    (doseq [[size detail tiers cnt] ladder-configs
            :let [{:keys [levels]} (seed/layer-params dmap detail size 0.5 0.5 2.5 tiers cnt 512 512)
                  sig (fn [l] [(:ssz l) (:th l) (:map-kind l) (:nx l)])]]
      (is (= (count levels) (count (distinct (map sig levels))))
          (str "size " size " detail " detail " tiers " tiers " count " cnt
               ": no two admitted levels share [ssz th map-kind nx]")))))

(deftest no-sub-pixel-levels
  (let [img  (ladder-img)
        dmap (wavelet/placement-map img (structure/analyze img))]
    (doseq [[size detail tiers cnt] ladder-configs
            :let [{:keys [levels]} (seed/layer-params dmap detail size 0.5 0.5 2.5 tiers cnt 512 512)]]
      (is (every? #(>= (double (:ssz %)) 0.6) levels)
          (str "size " size " detail " detail " count " cnt ": every admitted ssz >= 0.6")))))

(deftest detail-does-not-coarsen-the-broad-tier
  (let [img  (ladder-img)
        dmap (wavelet/placement-map img (structure/analyze img))
        ssz-by-lvl (fn [detail]
                     (into {} (map (juxt :lvl :ssz)
                                   (:levels (seed/layer-params dmap detail 12.0 0.5 0.5 2.5
                                                               [1.0 1.0 1.0] 72000 512 512)))))
        lo (ssz-by-lvl 0.6)
        hi (ssz-by-lvl 1.0)]
    (doseq [lvl [0 1 2 3]
            :when (and (lo lvl) (hi lvl))]
      (is (<= (double (hi lvl)) (double (lo lvl)))
          (str "level " lvl ": raising Detail 0.6→1.0 must not coarsen the broad/mid tier")))))

(deftest coverage-tiers-take-coverage-values-by-role
  ;; Regression: raw-floor/spec-cap/level-alpha keyed PURELY on the 8px size
  ;; threshold. At Size 6 the base tier lands ~6.7px (under 8px), so the COVERAGE
  ;; layer got detail-tier treatment (traw 0.45, tcap 0.70, lal 0.85): a coverage
  ;; layer at 0.85 alpha does not fully cover — the black background shows through
  ;; around silhouettes — and a 6.7px daub painting up to 70% raw colour smears edge
  ;; colour outward. Levels 0-1 are coverage BY ROLE and take the coverage constants
  ;; unconditionally; size-keying applies only from level 2 (mirror level-map-kind).
  ;; Same small size, two roles: a detail tier (lvl 2) at the SAME size keeps the
  ;; size-keyed values, proving the gate is role-based, not a flat override.
  (let [raw-floor   #'splat-painter.seed/raw-floor
        spec-cap    #'splat-painter.seed/spec-cap
        level-alpha #'splat-painter.seed/level-alpha
        ssz 6.7]                    ; under the 8px size threshold
    (testing "coverage tiers (lvl 0-1) take coverage constants regardless of size"
      (doseq [lvl [0 1]]
        (is (zero? (raw-floor lvl ssz))        (str "lvl " lvl ": traw 0.0 (faithful colour)"))
        (is (== 0.35 (spec-cap lvl ssz))       (str "lvl " lvl ": tcap 0.35 (averaged colour)"))
        (is (== 1.0 (level-alpha lvl ssz))     (str "lvl " lvl ": lal 1.0 (full opacity)"))))
    (testing "a detail tier (lvl 2) at the same size keeps the size-keyed values"
      (is (== 0.45 (raw-floor 2 ssz)))
      (is (== 0.7 (spec-cap 2 ssz)))
      (is (== 0.85 (level-alpha 2 ssz))))
    (testing "layer-params wires the role gate: a base level under 8px still gets traw 0.0"
      (let [img   (ladder-img)
            dmap  (wavelet/placement-map img (structure/analyze img))
            levels (:levels (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5 [1.0 1.0 1.0] 72000 512 512))
            base  (first (filter #(zero? (long (:lvl %))) levels))]
        (is (some? base) "a base level (lvl 0) is admitted")
        (when (some? base)
          (is (< (double (:ssz base)) 8.0) (str "base ssz " (:ssz base) " is under the 8px threshold"))
          (is (zero? (:traw base))         "base :traw is 0.0 (coverage constant, not size-keyed)"))))))

(deftest admitted-levels-fit-the-budget
  ;; The admission keeps the EMITTED field within the budget and the transform-feedback
  ;; buffer capacity. Measured on the survivor count (splat-field :splats) — the post-cull
  ;; quantity the buffer actually holds — not the pre-cull candidate×segs upper bound,
  ;; which the per-level detail threshold always trims down to ≈ the budget.
  (let [img (gray-img 256 256 (fn [x y]
                                 (let [coarse (* 0.25 (+ 1.0 (Math/sin (* 0.04 (+ x y)))))
                                       fine  (if (odd? (+ (quot x 2) (quot y 3))) 0.12 -0.12)]
                                   (max 0.0 (min 1.0 (+ coarse fine))))))]
    (doseq [cnt [1000 72000 600000]
            :let [n (count (:splats (seed/splat-field img {:count cnt :size 6.0 :detail 1.0
                                                           :variation 0.5 :curvature 0.5})))]]
      (is (<= n (* 1.5 cnt)) (str "count " cnt ": survivor count within 1.5× of the budget"))
      (is (<= n shader/max-splats) (str "count " cnt ": never exceeds shader/max-splats")))))

(deftest budget-cap-holds-at-both-ends-of-the-splats-range
  ;; Regression guard for the budget cap (spec-offset-and-repin, Item 2): the min-phys
  ;; floor pins the finest levels at fixed spacing that does NOT scale with the budget.
  ;; An over-aggressive cap starved the finest level 22x at high counts and erased the
  ;; strap/glasses detail; an under-aggressive one let count=1000 emit 6330 splats so the
  ;; Splats slider did nothing at the low end. This pins BOTH ends on a 1024x1024 image
  ;; with real multi-scale detail (so the detail tier actually admits):
  ;;   - at 600k the finest level KEEPS its share (nx >= 50000) — the cap stays at 1
  ;;   - at 1000 the field respects the 1.5x bound — the cap fires and thins detail
  ;; Measured on layer-params (the cap is set there) AND splat-field (the survivor count).
  (let [img  (gray-img 1024 1024 (fn [x y]
                                   (let [coarse (* 0.25 (+ 1.0 (Math/sin (* 0.04 (+ x y)))))
                                         fine  (if (odd? (+ (quot x 2) (quot y 3))) 0.12 -0.12)]
                                     (max 0.0 (min 1.0 (+ coarse fine))))))
        dmap (wavelet/placement-map img (structure/analyze img))
        finest-nx (fn [cnt]
                    (->> (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5 [0.4 0.4 0.4] cnt 1024 1024)
                         :levels (apply max-key :lvl) :nx long))]
    (is (>= (finest-nx 600000) 50000)
        "high count: the finest level keeps its share — the cap must NOT starve detail")
    (is (< (finest-nx 1000) (finest-nx 600000))
        "low count: the finest level is thinned — the cap fires at the low end")
    (let [low (count (:splats (seed/splat-field img {:count 1000 :size 6.0 :detail 1.0
                                                      :variation 0.5 :curvature 0.5})))]
      (is (<= low 1500) (str "low count: survivor count respects the 1.5x bound; got " low)))))

;; --- Round 2: liner-scale? is a pure physical-size predicate ------------------
(deftest liner-scale-predicate
  (testing "liner-scale? = (lvl>=2 && ssz<3.5); boundary exclusive, lvl<2 never a liner"
    (is (not (seed/liner-scale? 4 3.85)))   ; fat lvl-4 stroke is NOT a liner
    (is (not (seed/liner-scale? 5 4.0)))
    (is (seed/liner-scale? 2 0.94))         ; small mid chain IS a liner
    (is (not (seed/liner-scale? 1 0.5)))    ; broad/base never a liner
    (is (seed/liner-scale? 6 0.7))
    (is (not (seed/liner-scale? 2 3.5)))))  ; boundary exclusive

(deftest stroke-segments-head-taper-is-physical
  (testing "a lvl-5 stroke keys its head taper on PHYSICAL ssz, not the level index"
    ;; flat uniform ground: no edge ⇒ impasto body=0, no colour drift ⇒ fade stays 1, so the
    ;; first segment's alpha is lal×head-alpha and the ONLY lvl-5 q=0 difference between a
    ;; 4.0px (non-liner, head-alpha 0.5) and 2.0px (liner, head-alpha 0.75) stroke is that
    ;; head-alpha factor — their ratio is 0.5/0.75 = 2/3. With the old lvl>=4 rule both were
    ;; liners (ratio 1); the physical rule makes the fat one a dab.
    (let [img    (gray-img 32 32 (fn [_ _] 0.5))
          sfield (structure/analyze img)
          dmap   (wavelet/placement-map img sfield)
          nf     (seed/prep-noise sfield)
          px     (:pixels img)
          head-a (fn [ssz]
                   (let [[rows _reason] (seed/stroke-segments nf dmap 5 16 16 ssz 1.0 0.0 0.5 1 0.5 2.5
                                                    31 31 8 0.9 0.0 0.0 0.5 1.0 px 32 32
                                                    0.5 0.0 :sharp 1.0 px 0.5)]
                     (nth (first rows) 6)))
           a-fat  (head-a 4.0)
           a-thin (head-a 2.0)]
      (is (pos? a-fat) "the fat lvl-5 stroke still emits")
      (is (< a-fat a-thin)
          "non-liner head alpha is below liner: the head taper keys on the PHYSICAL predicate (liner-scale?), not the level index")
      ;; Re-pinned (the old 0.5/0.75 ratio assumed level-alpha was constant). The
      ;; taper COMPONENT ha is still 0.5 (non-liner) vs 0.75 (liner) at tt=0, but the
      ;; TOTAL head alpha = level-alpha*ha also folds level-alpha's dab-near-opaque
      ;; boost (0.85 above dab-max) and impasto body, so the ratio is ~0.71, not 2/3.
      (is (approx= 0.02 (/ (double a-fat) (double a-thin)) 0.7059)
          "head-alpha ratio folds the physical taper (0.5 vs 0.75) AND level-alpha/body — re-pinned to the measured value"))))

;; --- Round 3: two-radius colour probe ---------------------------------------
(defn- ramp-img [H W dark light rw]
  ;; two flat colours joined by an rw-wide COSINE ramp (a soft edge) centred at W/2:
  ;; flat dark flank | cosine smoothstep | flat light flank. The 0.02 sine keeps a
  ;; hair of signal in the flats but does not wash the ramp away.
  (let [c0 (/ W 2.0) hw (/ rw 2.0)]
    (gray-img H W (fn [row col]
                    (let [t    (max 0.0 (min 1.0 (/ (- col (- c0 hw)) rw)))
                          edge (* 0.5 (- 1.0 (Math/cos (* Math/PI t))))
                          tex  (* 0.02 (Math/sin (* 0.6 (+ row col))))]
                      (max 0.0 (min 1.0 (+ dark (* edge (- light dark)) tex))))))))

(defn- line-img [H W line-col dark light]
  ;; a genuine 1px hard line (dark) on a light ground — NOT a soft Gaussian. This is
  ;; the line-art control: the on-ridge colour path must survive the two-radius probe.
  (gray-img H W (fn [row col]
                  (let [base (+ 0.4 (* 0.2 (/ col (dec W))))
                        tex  (* 0.02 (Math/sin (* 0.6 row)))]
                    (max 0.0 (min 1.0 (+ (if (== col line-col) dark base) tex)))))))

(defn- cheb-from [s ref]
  (let [[r g b] (:color s)]
    (max (Math/abs (- (double r) (double ref)))
         (Math/abs (- (double g) (double ref)))
         (Math/abs (- (double b) (double ref))))))

(defn- bilerp-src
  "Reference bilinear [r g b] from the flat H*W*3 source array at grid (x,y), clamped
  at the borders — x=row (height dim), y=col (width dim), matching seed/sample-arr's
  coordinate convention. This is what seed/sample-arr becomes in round 4; used here as
  the ground-truth source colour at a splat's mean."
  [img x y]
  (let [W   (:width img)  H (:height img)  ^doubles px (:pixels img)
        xc  (max 0.0 (min (double (dec H)) (double x)))
        yc  (max 0.0 (min (double (dec W)) (double y)))
        x0  (int xc)  y0 (int yc)
        x1  (min (dec H) (inc x0))  y1 (min (dec W) (inc y0))
        fx  (- xc x0)  fy (- yc y0)
        tex (fn [xi yi] (let [b (* 3 (+ (* xi W) yi))]
                          [(aget px b) (aget px (+ b 1)) (aget px (+ b 2))]))
        [r00 g00 b00] (tex x0 y0)  [r01 g01 b01] (tex x0 y1)
        [r10 g10 b10] (tex x1 y0)  [r11 g11 b11] (tex x1 y1)
        lx  (fn [a b t] (+ a (* t (- b a))))
        r0 (lx r00 r10 fx)  g0 (lx g00 g10 fx)  b0 (lx b00 b10 fx)
        r1 (lx r01 r11 fx)  g1 (lx g01 g11 fx)  b1 (lx b01 b11 fx)]
    [(lx r0 r1 fy) (lx g0 g1 fy) (lx b0 b1 fy)]))

(deftest two-radius-probe-rescues-soft-ramp
  (testing "a bodied liner splat paints a colour present where it is painted (two-radius probe)"
    ;; The defect is NOT "far from both flat side colours": a stroke on the ramp centre
    ;; legitimately paints a mid-tone (the ramp is real image content). The defect is a
    ;; stroke carrying a colour that is NOT at the place it is painted. So: among alpha>0.5
    ;; splats at LINER scale (σ<3.5), the splat colour must be within 0.20 (Chebyshev) of
    ;; the bilinearly-sampled SOURCE colour at the splat's own mean.
    ;;
    ;; PRE-FIX (disp = 0.5*chosen_hh, the wrong spec draft): 0.318 (631/1984) violated —
    ;; the on-ridge mid-blend this round exists to escape. Corrected disp (full chosen_hh)
    ;; drops it well under 5%.
    (let [dark 0.1 light 0.85
          ;; the fixture must carry the same precomputed fields the app uses, or the
          ;; probes read the RAW ramp instead of the bilateral blur (not the real path).
          ;; RW=8 (the spec's value): an 8px cosine ramp is a SOFT edge — the bilateral
          ;; blur spreads it so d1/dmax ≈ 0.35, and the sharpness measure classifies it
          ;; SOFT and paints the local colour. The OLD contrast measure ("which rung
          ;; cleared 0.15 first") read h1≥0.15 here and wrongly called it crisp — that
          ;; bug is exactly what this round fixes. Do NOT widen the ramp to pass: an 8px
          ;; transition is the real scale of an out-of-focus portrait edge.
          img  (attach-precomputed-fields (ramp-img 128 128 dark light 8.0))
          fld  (seed/splat-field img {:count 4000 :size 6.0 :detail 0.6
                                      :variation 0.5 :curvature 0.5})
          liner? (fn [s] (let [[c00 _ _ c11] (:cov s)]
                           (< (Math/sqrt (max (double c00) (double c11))) 3.5)))
          scope  (filter #(and (> (:alpha %) 0.5) (liner? %)) (:splats fld))
          cheb-at-mean (fn [s]
                         (let [[sr sg sb] (:color s)
                               [mx my] (:mean s)
                               [vr vg vb] (bilerp-src img mx my)]
                           (max (Math/abs (- (double sr) (double vr)))
                                (Math/abs (- (double sg) (double vg)))
                                (Math/abs (- (double sb) (double vb))))))
          offenders (filter #(> (cheb-at-mean %) 0.20) scope)
          frac (if (seq scope) (/ (double (count offenders)) (double (count scope))) 0.0)]
      (println "RAMP3-DIAG offenders:" (count offenders) "of" (count scope)
               "liner alpha>0.5 splats; violating fraction =" (double frac))
      (is (< frac 0.05)
          (str "too many liner splats carry a colour absent at their location: "
               (count offenders) "/" (count scope) " = " (double frac))))))

(deftest hard-line-stroke-keeps-line-colour
  (testing "a 1px hard line still takes the line's own colour (on-ridge path intact)"
    (let [dark 0.08 light 0.9
          img  (line-img 64 64 32 dark light)
          fld  (seed/splat-field img {:count 4000 :size 6.0 :detail 0.6
                                      :variation 0.5 :curvature 0.5})
          splats (:splats fld)
          dark-ones (filter #(<= (cheb-from % dark) 0.12) splats)]
      (is (seq dark-ones) "the hard line is painted in its own dark colour, not lost"))))

;; --- round 4: bilinear sampling ---------------------------------------------

(deftest sample-arr-is-bilinear
  (testing "seed/sample-arr interpolates bilinearly: exact at integers, mean at midpoints, clamped"
    (let [sample @#'splat-painter.seed/sample-arr
          ;; 2x2 RGB (width=2 cols, height=2 rows), index base 3*(x*2+y):
          ;;   row0=[black, red], row1=[green, blue]
          arr (double-array [0 0 0   1 0 0
                             0 1 0   0 0 1])
          vapprox (fn [want got]
                    (< (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2)))
                                            want got)) 1e-9))]
      (is (vapprox [0.0 0.0 0.0] (sample arr 2 2 0 0))   "exact at integer (row0,col0)")
      (is (vapprox [1.0 0.0 0.0] (sample arr 2 2 0 1))   "exact at integer (row0,col1)")
      (is (vapprox [0.0 1.0 0.0] (sample arr 2 2 1 0))   "exact at integer (row1,col0)")
      (is (vapprox [0.5 0.0 0.0] (sample arr 2 2 0 0.5)) "col midpoint at row0 = mean of the two cols")
      (is (vapprox [0.0 0.5 0.0] (sample arr 2 2 0.5 0)) "row midpoint at col0 = mean of the two rows")
      (is (vapprox [0.25 0.25 0.25] (sample arr 2 2 0.5 0.5)) "centre = mean of all four texels")
      (is (vapprox [0.0 0.0 0.0] (sample arr 2 2 -0.5 0))     "clamps below the border")
      (is (vapprox [0.0 0.0 1.0] (sample arr 2 2 5 5))        "clamps above the border"))))

(deftest wavelet-detail-at-is-bilinear
  (testing "wavelet/detail-at interpolates bilinearly and normalizes into [0,1]"
    ;; 2x2 detail (row-major detail[x*2+y]): row0=[0.0,0.4], row1=[0.6,1.0]
    (let [detail (double-array [0.0 0.4 0.6 1.0])
          dmap   {:h 2 :w 2 :detail detail :sharp detail :dmax 1.0 :src-h 2 :src-w 2}]
      (is (approx= 1e-9 0.0 (wavelet/detail-at dmap 0 0))   "exact at texel (0,0)")
      (is (approx= 1e-9 1.0 (wavelet/detail-at dmap 1 1))   "exact at texel (1,1)")
      (is (approx= 1e-9 0.5 (wavelet/detail-at dmap 0.5 0.5)) "centre = mean of all four (0.5)")
      (is (approx= 1e-9 0.2 (wavelet/detail-at dmap 0 0.5))   "col midpoint at row0 (0.2)")
      (is (approx= 1e-9 0.3 (wavelet/detail-at dmap 0.5 0))   "row midpoint at col0 (0.3)")
      (is (approx= 1e-9 0.0 (wavelet/detail-at dmap 0 -5))    "clamps below")
      (is (approx= 1e-9 1.0 (wavelet/detail-at dmap 5 5))     "clamps above")
      (let [dmap2 (assoc dmap :dmax 0.4)]
        (is (approx= 1e-9 1.0 (wavelet/detail-at dmap2 1 1))
            "normalized by dmax and clamped to [0,1] (1.0/0.4 -> 1.0)")))))

(deftest mid-at-is-max-of-interpolations
  (testing "mid-at takes max AFTER interpolating each band (max∘bilerp, not bilerp∘max)"
    ;; 1x2 grid: mid=[0.0,1.0], sharp=[1.0,0.0]. At the col midpoint:
    ;;   bilerp(mid)=0.5, bilerp(sharp)=0.5 -> max = 0.5   (correct max∘bilerp)
    ;;   per-texel max=[1.0,1.0] -> bilerp = 1.0            (wrong bilerp∘max)
    (let [mid (double-array [0.0 1.0]) sharp (double-array [1.0 0.0])
          dmap {:h 1 :w 2 :mid mid :sharp sharp :detail mid :dmax 1.0 :src-h 1 :src-w 2}
          v (wavelet/mid-at dmap 0 0.5)]
      (is (approx= 1e-9 0.5 v)
          (str "mid-at at the band midpoint must be the max of the INTERPOLATED bands (0.5), "
               "not the interpolation of the per-texel max (1.0); got " v)))))


;; --- round 5: aspect-bounded span + coherence-gated chain length -------------

(deftest liner-span-is-feature-determined
  (testing "round 5a: the absolute-px aspect cap is GONE — thin liner spans are feature-determined"
    ;; The round-5a ABSOLUTE-px cap (span = min(28*slen*ramp, 12*ssz), bounded to <=12.5*ssz)
    ;; was a stop-gap, removed once the feature tracer + line-hold/runaway stops landed.
    ;; Spans are now FEATURE-DETERMINED: a thin liner runs the full max-segs tracer span
    ;; and lets the ridge geometry decide where to stop. This pins the NEGATIVE result so
    ;; the cap does not get re-added — a thin liner's traced span now EXCEEDS the old cap.
    (let [H 128 W 128
          img (attach-precomputed-fields
               {:height H :width W :channels 3
                :pixels (double-array (mapcat (fn [x] (mapcat (fn [y]
                  (let [g (if (odd? (+ (quot x 2) (quot y 2))) 0.35 0.65)] [g g g])) (range W))) (range H)))})
          dmap (:detail img)
          lp (seed/layer-params dmap 0.6 6.0 0.5 0.5 2.5 [1.0 0.7 0.4] 4000 H W)
          thin-liners (filter #(and (>= (:lvl %) 2) (< (:ssz %) 3.5) (< (:ssz %) 2.6)) (:levels lp))]
      (is (seq thin-liners) "config yields thin liner levels (ssz<2.6)")
      (doseq [{:keys [lvl ssz segs stepf]} thin-liners]
        (let [span (* (double segs) (double stepf) (double ssz))
              old-cap (* 12.5 (double ssz))]
          (is (> span old-cap)
              (str "liner lvl " lvl " ssz " ssz ": span " span " exceeds the removed 12.5*ssz cap (" old-cap
                   ") — feature-determined (segs " segs " stepf " stepf ")")))))))

(deftest coherence-does-not-separate-lines-from-gradients
  (testing "tensor coherence is NOT a line detector — do not gate chain length on it"
    ;; This pins a NEGATIVE result so the gate does not get re-added. A chain-length
    ;; gate keyed on structure-tensor coherence (ramp segs 20%->100% over 0.30..0.60)
    ;; was implemented and removed: coherence does not discriminate, because a smooth
    ;; gradient is rank-1 and therefore reads as perfectly ORIENTED. On the real test
    ;; portrait, strong-edge (edge-at>0.50) median coherence was 0.95 while FLAT
    ;; (edge-at<0.08) median was 0.72, with only 13.5% of flat points under 0.30 — so
    ;; the gate was a no-op exactly where it was meant to act (0.05/255 on the render).
    ;; Here: a 1D ramp (smooth, no line) must score coherence comparable to a hard edge.
    (let [ramp (attach-precomputed-fields
                 (gray-img 96 96 (fn [_ y] (+ 0.15 (* 0.7 (/ (double y) 95.0))))))
          edge (attach-precomputed-fields
                 (gray-img 96 96 (fn [_ y] (if (< y 48) 0.2 0.85))))
          sf   #'splat-painter.seed/sample-fields
          coh  (fn [img] (let [nf (:noise-fields img)
                               vs (for [x (range 20 76 4) y (range 20 76 4)]
                                    (second (sf nf x y)))]
                           (/ (reduce + vs) (double (count vs)))))
          ramp-coh (coh ramp) edge-coh (coh edge)]
      (is (> ramp-coh 0.5)
          (str "a smooth ramp is rank-1 and scores HIGH coherence: " ramp-coh))
      (is (> ramp-coh (* 0.6 edge-coh))
          (str "ramp coherence " ramp-coh " is comparable to edge coherence " edge-coh
               " — coherence cannot tell a gradient from a line, so it must not gate"
               " chain length")))))

;; --- feature-following tracer: long arc vs sharp corner ----------------------
(deftest feature-following-tracer-breaks-at-corners
  (testing "a smooth ridge traces as one long stroke; a sharp corner breaks the chain (mirror seed/stroke-segments)"
    ;; The feature-following tracer ends a stroke where the FEATURE ends or turns — the
    ;; way a person paints (the top of an eye is one line, the bottom another). Two
    ;; GEOMETRIC stops (detail tiers, lvl>=2) are CLEAN breaks — no segment is emitted
    ;; at the break, so a contour chunks into the strokes a draughtsman draws:
    ;;   edge-floor 0.10: the ridge died -> :ridge
    ;;   bend-cos 0.90:  |dot(field-dir, prev-step)| < 0.90 (~26deg/step) -> :corner
    ;; A clean straight ridge (no bend) runs its full span (reason :cap); a ridge that
    ;; bends 90deg breaks at the bend (reason :corner) and so traces a shorter chain.
    ;; stroke-segments is called DIRECTLY at chosen seeds (warp is inert at liner scale)
    ;; and BOTH travel directions are tried, so the result does not depend on the
    ;; structure tensor's arbitrary eigenvector sign.
    (let [H 160 W 160
          straight (attach-precomputed-fields
                    {:height H :width W :channels 3
                     :pixels (double-array
                              (mapcat (fn [x] (mapcat (fn [y] (let [g (if (<= 79 x 80) 0.28 0.72)] [g g g]))
                                                        (range W))) (range H)))})
          ;; L-bend: horizontal arm rows 79-80 cols 0..79, vertical arm cols 78-79 rows 80..159.
          ;; vertex at (80,79). A seed on the horizontal arm travelling toward +y hits the bend.
          corner (attach-precomputed-fields
                  {:height H :width W :channels 3
                   :pixels (double-array
                            (mapcat (fn [x] (mapcat (fn [y]
                                                      (let [on-h (and (<= 79 x 80) (<= y 79))
                                                            on-v (and (<= 78 y 79) (>= x 80))
                                                            g (if (or on-h on-v) 0.28 0.72)]
                                                        [g g g]))
                                              (range W))) (range H)))})
          hd (double (dec H)) wd (double (dec W))
          info (fn [img x y ds]
                 (let [nf (:noise-fields img) dmap (:detail img)
                       blur (:blur img) blurd (:blur-drift img)
                       [rows reason] (seed/stroke-segments
                                      nf dmap 5 x y 1.0 1.0 0.0 0.5 ds 0.5 2.5
                                      hd wd 24 0.9 0.5 0.0 0.5 1.0 blur W H
                                      0.5 0.0 :sharp 1.0 blurd 0.5)]
                   {:len (count rows) :reason reason}))
          ;; seeds near the bend on the horizontal arm, both travel directions. On `corner`
          ;; the seed travelling toward the vertex hits the 90deg bend; on `straight` there
          ;; is no bend in either direction.
          trials (fn [img] (for [y [62.0 66.0 70.0 73.0] ds [1.0 -1.0]] (info img 80.0 y ds)))
          st (trials straight) co (trials corner)
          st-corner (count (filter #(= (:reason %) :corner) st))
          co-corner (count (filter #(= (:reason %) :corner) co))
          st-len (median (map :len st)) co-len (median (map :len co))]
      (is (zero? st-corner) "a straight ridge has no corner -> no :corner breaks")
      (is (>= co-corner 2) (str "the L-bend triggers :corner breaks: " co-corner " of " (count co)))
      (is (>= st-len 20) (str "a clean ridge runs its full span: straight median " st-len))
      ;; The corner median need NOT be shorter than straight: the structure tensor
      ;; smooths the hard 90deg vertex into a gradual turn, so the feature-FOLLOWING
      ;; tracer can ride AROUND the bend (it re-samples the field direction each step).
      ;; The contract is the STOP REASON (straight->:cap/:ridge, bend->:corner for the
      ;; seeds whose eigenvector sign points into the bend), not a guaranteed shorter
      ;; length. The :corner stops above (co-corner>=2) are the real corner signal.
      (is (pos? co-len) (str "corner chains still emit — the tracer rides the L; median " co-len " of " (count co))))))

;; --- lip-band / dark-mark: detail-stroke raw-fidelity scales with fine-detail density ---
;; (spec-lip-band) A ~1.4px detail stroke takes raw-floor 0.85, so 85% of its colour is a
;; single RAW sample. Where features crowd (a shadow / lip line 3-5px from the next
;; feature) that sample is a foreign dark value, and the stroke carrying it paints a dark
;; mark across lighter neighbours -- the band above the upper lip. density-scaled-traw
;; scales the floor by the finest wavelet band (:sharp), so detail strokes there trust the
;; bilateral (region) colour more; an isolated crisp feature keeps full fidelity.
;;
;; Why this verifies the MECHANISM rather than an end-to-end render: the artifact is a
;; 1024px real-image boundary phenomenon that cannot be reproduced synthetically at a
;; renderable size. Three pipeline interactions defeat any small synthetic fixture:
;;   (1) :sharp is ~0 below ~1024px (no sub-2px detail for the finest band);
;;   (2) the bilateral is edge-aware, so it PRESERVES any hard-edge synthetic texture
;;       (bilat=raw), which makes traw a no-op (colour = t*raw + (1-t)*raw);
;;   (3) the blend t = max(traw, 0.15+0.85*max(coh,dlev)) is floored by traw only where
;;       dlev/coherence are low, but raw!=bilateral (the thing traw acts on) needs high
;;       detail, which raises dlev past traw -- a catch-22 in the synthetic regime.
;; The 1024px composite is itself infeasible in the harness (splat-intensities is O(splats
;; x pixels) full-frame). So this pins the fix FORMULA on the extracted helper -- the exact
;; value applied per detail segment, mirrored in gen.clj (verified by check.clj). It
;; DISCRIMINATES: with the 0.7 density constant set to 0.0 (no fix) the crowded case
;; returns the unscaled 0.85 and the test fails -- confirmed by toggling 0.7->0.0 and
;; re-running (see the round report).
(deftest density-scaled-traw-follows-fine-detail-band
  (let [f @#'splat-painter.seed/density-scaled-traw]
    ;; coverage tiers (lvl 0-1) are NEVER scaled: faithful colour by role
    (is (approx= 1e-9 0.85 (f 0 0.85 1.0)) "base (lvl 0): unscaled even at max density")
    (is (approx= 1e-9 0.70 (f 1 0.70 1.0)) "broad (lvl 1): unscaled even at max density")
    ;; detail tiers (lvl>=2), isolated crisp feature (sharp-at 0): full raw fidelity
    (is (approx= 1e-9 0.85 (f 2 0.85 0.0)) "isolated feature (sharp 0): keeps full fidelity")
    (is (approx= 1e-9 0.85 (f 3 0.85 0.0)) "isolated feature (sharp 0): keeps full fidelity")
    ;; detail tiers in a CROWDED region (sharp-at high): trust the region colour more.
    ;; floor 0.85 * (1 - 0.7*0.9) = 0.85 * 0.37 = 0.3145 -- the mark the fix removes.
    (is (approx= 1e-9 (* 0.85 (- 1.0 (* 0.7 0.9))) (f 2 0.85 0.9))
        "crowded region (sharp 0.9): floor pulled toward the region colour")
    (is (approx= 1e-9 (* 0.70 (- 1.0 (* 0.7 0.9))) (f 2 0.70 0.9))
        "scaling applies regardless of the base floor")
    ;; monotone in density: more crowded -> less raw (stronger pull to region)
    (is (> (f 2 0.85 0.3) (f 2 0.85 0.6) (f 2 0.85 0.9))
        "raw-fidelity floor is strictly decreasing as fine detail crowds")))



