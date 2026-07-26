(ns splat-painter.seed-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.seed :as seed]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.structure :as structure]
            [splat-painter.gaussian :as g]
            [splat-painter.shader :as shader]))

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
    (is (= 683 (count splats)))
    (is (approx= 0.5  13842.947  sx) "Σ mean-x")
    (is (approx= 0.5  19912.245  sy) "Σ mean-y")
    (is (approx= 1.0  219759.323 sd) "Σ det(cov)")
    (is (approx= 0.05 794.590   sc) "Σ colour")))

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
  ;; half-flat, half-checkerboard. detail>0 admits the fine levels and lowers their
  ;; placement threshold in the textured half. After rounds 5a/5b the fine chains ON
  ;; high-frequency texture are deliberately SHORT (aspect-bounded + coherence-gated),
  ;; so the grand TOTAL can stay flat — the detail slider's effect now shows up as more
  ;; FINE-LEVEL (lvl≥4) splats in the textured region, not a higher total count.
  (let [img (gray-img 48 48 (fn [x y]
                              (if (< x 24)
                                0.5                                       ; flat top half
                                (if (odd? (+ (int x) (int y))) 0.0 1.0)))) ; checkerboard
        fine (fn [detail] (count (filter #(>= (double (first (:mean %))) 24.0)
                                         (:splats (seed/splat-field img {:count 2000 :size 2.0
                                                                          :detail detail :variation 0.0})))))
        cnt-0 (fine 0.0) cnt-1 (fine 1.0)]
    (is (> cnt-1 cnt-0)
        (str "detail=1 should place more splats in the textured half (mean-x≥24) than detail=0: " cnt-1 " vs " cnt-0))))

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
                      (count (stroke-segments nf dmap lvl cx cy cssz D 0.0 tn ds curvature stroke
                                              hd wd segs stepf bendf hb traw* sgate blur-px iw ih
                                              th 0.0 map-kind gain blurd-px bph))))]
    (is liner "settings must yield a liner level (lvl≥2, ssz<3.5)")
    (is (>= segs 10) "liner level keeps a meaningful (round-5a-bounded) span")
    (is (< ssz 3.5))
    (let [contour (keep-indexed (fn [i x] (chain-len i x 40.0)) (range 12 152 7))   ; ~20 seeds ON the line (col 40)
          texture (for [[i [x y]] (map-indexed vector
                                               (for [x (range 15 145 20) y (range 94 150 12)] [x y]))]  ; ~35 seeds in the checker
                    (chain-len (+ i 1000) x y))                                     ; offset i so texture hashes ≠ contour
          cmed (median contour) tmed (median texture)]
      (is (>= cmed (* 0.8 (double segs)))   (str "contour chains run their full (coherence-ungated) span; median=" cmed " of nominal segs " segs))
      (is (<= tmed 8)    (str "texture chains should dry out; median=" tmed " of " (count texture)))
      (is (>= cmed (* 2.0 tmed))
          (str "contour median (" cmed ") must be ≥ 2× texture median (" tmed ")")))))

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
                      (is (= (long (seg-count lvl)) (long segs))
                          (str "budget " cnt " lvl " lvl " ssz " ssz
                               ": ramp floor — fat liners stay on the table")))
                    (when (and (>= lvl 2) liner? (<= ssz 1.4))
                      ;; ROUND 5a bounds the span to ≤12px (min(28·slen·ramp, 12·ssz)):
                      ;; thin liners no longer draw 24-seg hairlines; they keep a
                      ;; meaningful span that is ABSOLUTELY bounded (≤12.5·ssz, with the
                      ;; 0.5-seg rounding slack included). The physical-sigma ramp that
                      ;; this test pins is unchanged — only its ceiling lowered.
                      (is (and (>= (long segs) 10)
                               (<= (* (double segs) (double stepf) ssz) (* 12.5 ssz)))
                          (str "budget " cnt " lvl " lvl " ssz " ssz " segs " segs
                               ": thin liner span is meaningful and round-5a-bounded to ≤12px")))
                    (when liner?
                      (is (<= (* (double segs) (double stepf) ssz)
                              (* 1.15 28.0 slen))
                          (str "budget " cnt " lvl " lvl " ssz " ssz " segs " segs
                               ": liner span must be capped in ABSOLUTE px"))))
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
                   (let [segs (seed/stroke-segments nf dmap 5 16 16 ssz 1.0 0.0 0.5 1 0.5 2.5
                                                    31 31 8 0.9 0.0 0.0 0.5 1.0 px 32 32
                                                    0.5 0.0 :sharp 1.0 px 0.5)]
                     (nth (first segs) 6)))
          a-fat  (head-a 4.0)
          a-thin (head-a 2.0)]
      (is (pos? a-fat) "the fat lvl-5 stroke still emits")
      (is (< a-fat a-thin) "non-liner head alpha (0.5×lal) is below liner (0.75×lal)")
      (is (approx= 0.02 (/ (double a-fat) (double a-thin)) (/ 0.5 0.75))
          "head-alpha ratio 0.5/0.75 proves the taper keys on the physical predicate"))))

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

(deftest liner-span-is-aspect-bounded
  (testing "round 5a: a thin liner chain's absolute span (segs·stepf·ssz) is bounded to ≤12.5·ssz"
    ;; span = min(28·slen·ramp, 12·ssz) caps the 22:1 hairlines that outran soft
    ;; features. Asserted on THIN liners (ssz<2.6, ramp>0) where the span term — not
    ;; the seg-count floor — sets the length; fat liners (ramp=0) keep the short table.
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
              limit (* 12.5 (double ssz))]
          (is (<= span limit)
              (str "liner lvl " lvl " ssz " ssz ": traced span " span " > 12.5·ssz " limit
                   " (segs " segs " stepf " stepf ")")))))))

(deftest low-coherence-seeds-make-short-marks
  (testing "round 5b: a liner seed traces few segments where the structure tensor is incoherent"
    ;; A "line" only exists where the structure tensor is coherent. On an isotropic
    ;; texture (a fine checker — gradients in BOTH axes → coherence ~0) a liner seed
    ;; draws a short dab; on a clean hard edge (one strong orientation → coherence ~1)
    ;; it runs long. NOTE: the spec's "smooth gradient" is a misnomer — a 1D ramp is
    ;; itself oriented (structure tensor rank-1 → coherence HIGH); an isotropic texture
    ;; is the correct low-coherence fixture.
    (let [mean-chain (fn [img seeds]
                       (let [dmap (:detail img) blur-px (:blur img) blurd-px (:blur-drift img)
                             nf (:noise-fields img)
                             H (:height img) W (:width img)
                             hd (double (dec H)) wd (double (dec W)) iw W ih H
                             rr (/ (double H) 24.0)
                             lp (seed/layer-params dmap 0.6 6.0 0.5 0.5 2.5 [1.0 0.7 0.4] 4000 H W)
                             liner (->> (:levels lp) (filter #(and (>= (:lvl %) 2) (< (:ssz %) 3.5)))
                                        (apply max-key :segs))
                             {:keys [lvl ssz segs stepf bendf map-kind traw]} liner
                             hash01 #'splat-painter.seed/hash01
                             subject-at #'splat-painter.seed/subject-at
                             map-at #'splat-painter.seed/map-at
                             stroke-segments #'splat-painter.seed/stroke-segments
                             deff (fn [D] (min 1.0 (* 0.6 (double D) 2.2)))
                             len (fn [i cx cy]
                                   (let [D (deff (map-at dmap map-kind cx cy))
                                         sgate (subject-at dmap cx cy rr)
                                         tn (* 0.2 (- (hash01 (+ (* i 37) lvl) 0 13) 0.5))
                                         ds (if (< (hash01 (+ (* i 41) lvl) 0 17) 0.5) 1.0 -1.0)
                                         bph (hash01 (+ (* i 67) lvl) 0 53)]
                                     (count (stroke-segments nf dmap lvl cx cy ssz D 0.0 tn ds 0.5 2.5
                                                             hd wd segs stepf bendf 0.0 traw sgate blur-px iw ih
                                                             0.0 0.0 map-kind 1.0 blurd-px bph))))]
                         (when liner
                           (/ (double (transduce (map-indexed (fn [i [x y]] (len i x y))) + 0 seeds))
                              (max 1 (count seeds))))))
          edge-img (attach-precomputed-fields
                    {:height 128 :width 128 :channels 3
                     :pixels (double-array (mapcat (fn [x] (mapcat (fn [y]
                       (let [g (if (<= 39 y 40) 0.28 0.72)] [g g g])) (range 128))) (range 128)))})
          checker-img (attach-precomputed-fields
                       {:height 128 :width 128 :channels 3
                        :pixels (double-array (mapcat (fn [x] (mapcat (fn [y]
                          (let [g (if (odd? (+ (quot x 2) (quot y 2))) 0.35 0.65)] [g g g])) (range 128))) (range 128)))})
          edge-seeds (for [x (range 20 120 8)] [x 40.0])      ; ~12 seeds ON the hard edge
          tex-seeds  (for [x (range 20 120 12) y (range 20 120 12)] [x y])  ; ~80 seeds in the checker
          edge-mean (mean-chain edge-img edge-seeds)
          tex-mean  (mean-chain checker-img tex-seeds)]
      (is edge-mean  "config yields a liner level on the edge image")
      (is tex-mean   "config yields a liner level on the checker image")
      (when (and edge-mean tex-mean)
        (is (>= edge-mean 8.0) (str "coherent seeds run long; edge mean " edge-mean))
        (is (<= tex-mean 4.0) (str "incoherent seeds make short marks; checker mean " tex-mean))
        (is (> edge-mean (* 2.0 tex-mean))
            (str "coherent mean (" edge-mean ") must be ≥ 2× incoherent mean (" tex-mean ")"))))))
