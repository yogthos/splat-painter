(ns splat-painter.seed-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.fields :as fields]
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
        (seed/splat-record 10.0 20.0 5.0 0.5 0.0 0.5 0.0 0.0 [0.4 0.4 0.4] [0.8 0.2 0.1] 2.5 0.5 1.0 0.0 1.0 0.0)
        [c00 c01 _ c11] cov
        [cr cg cb] color]
    (is (= [10.0 20.0] mean))
    (is (approx= 1e-6 40.0    c00))   ; sx² = s0²·e = 25·1.6
    (is (approx= 1e-6 0.0     c01))   ; θ=0 ⇒ axis-aligned
    (is (approx= 1e-6 15.625  c11))   ; sy² = s0²/e = 25/1.6
    (is (approx= 1e-6 0.63    cr))    ; 0.4·0.425 + 0.8·0.575
    (is (approx= 1e-6 0.285   cg))
    (is (approx= 1e-6 0.2275  cb)))
  ;; selong > 0 REPLACES the coherence-derived elongation (the edge-band tier is born
  ;; long-and-thin rather than inheriting the local tensor's anisotropy): se = selong
  ;; exactly, so sx² = s0²·selong² and sy² = s0²/selong². Same inputs as above, whose
  ;; coherence-derived se would be √1.6 ≈ 1.265 — so this discriminates: at selong 2.6
  ;; the across-axis is 25/6.76 = 3.70, less than a quarter of the 15.625 above.
  (let [{:keys [cov]}
        (seed/splat-record 10.0 20.0 5.0 0.5 0.0 0.5 0.0 0.0 [0.4 0.4 0.4] [0.8 0.2 0.1] 2.5 0.5 1.0 0.0 1.0 2.6)
        [c00 _ _ c11] cov]
    (is (approx= 1e-6 169.0 c00))     ; sx² = 25·2.6²
    (is (approx= 1e-6 (/ 25.0 6.76) c11))
    (is (< c11 15.625) "forced elongation is THINNER across than the coherence-derived one")))

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
    ;; (529→953) the EDGE-BAND tier reaches this fixture. It shipped gated at Detail 0.75
    ;; and this golden runs at 0.6, so it was previously absent here; the Cut-in dial
    ;; replaced that gate, and at the default dial (1.0) the tier now paints. The jump is
    ;; all thin band strokes: count and Σcolour nearly double while Σdet moves 0.16%
    ;; (219649.771→220003.611), which is the signature of many LOW-det splats — a band
    ;; stroke is ssz/selong ≈ 0.54px across. Anything that moved Σdet materially here
    ;; would NOT be the band tier and should be investigated rather than re-pinned.
    ;; (953→850) the fine tier is ADMITTED AND THINNED instead of dropped whole, and the
    ;; two detail slices now PARTITION the detail budget instead of each thinning against
    ;; all of it. This fixture gains a level-4 rung and the mid tier pays for it, so the
    ;; count falls while the field reaches finer. Σdet moves 0.02% (220003.611→219958.654):
    ;; the strokes traded are the same size class, which is what a slice reallocation
    ;; between two adjacent rungs looks like. A materially different Σdet here would mean
    ;; something other than the reallocation moved and should be investigated.
    ;; (933.485→933.066) `wsl` in stroke-segments read a `detail?` that was bound
    ;; inside the tracer loop but referenced in the FINALIZE block, so it resolved
    ;; to an unbound var — a TRUTHY object — and the cond took the detail branch
    ;; unconditionally: every non-soft-ramp tier re-loaded colour per segment
    ;; (wsl=1) instead of carrying one brush-load, the opposite of what the
    ;; comment above wsl describes. jolt 0.5.11 reports the unresolved symbol
    ;; instead of late-binding it, which is how this surfaced. Only Σcolour moves,
    ;; by 0.045%; count, both means and Σdet are unchanged — a colour-sampling
    ;; change with no geometry change, which is the signature of exactly this fix.
    (is (= 850 (count splats)))
    (is (approx= 0.5  19111.677  sx) "Σ mean-x")
    (is (approx= 0.5  25937.161  sy) "Σ mean-y")
    (is (approx= 1.0  219958.654 sd) "Σ det(cov)")
    (is (approx= 0.05 933.066    sc) "Σ colour")))

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
    ;; the ladder is strictly finer-first; the EDGE-BAND overlay is excluded because it
    ;; is sized AT the ladder's finest rung by design (it draws a line over the finest
    ;; marks), so it ties rather than strictly decreasing. Its own placement at index 0
    ;; is pinned by edge-band-tier-is-drawn-topmost.
    (is (apply < (map :ssz (remove :band levels))) "ladder levels finest-first: ssz strictly increasing coarseward")
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
  ;; the same fields the app attaches on image load — splat-field falls back to raw
  ;; pixels for any missing blur field, which would NOT exercise the real colour path.
  ;; This used to be a hand-copy of core/prepare-image, kept in sync by a comment.
  (fields/prepare img0))

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
                                              th 0.0 map-kind gain blurd-px bph 0.55 0.0)))))]
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
  ;; The EDGE-BAND tier is excluded: it is an OVERLAY, not a rung. It is sized AT the
  ;; ladder's finest stdev on purpose (it draws a line over the finest marks), so it
  ;; is legitimately not 0.95× finer than the level it sits above — the monotonicity
  ;; rule is about the coarse→fine ladder, which is what `remove :band` leaves.
  ;; The band tier's own invariants are pinned by edge-band-tier-* below.
  (let [img  (ladder-img)
        dmap (wavelet/placement-map img (structure/analyze img))]
    (doseq [[size detail tiers cnt] ladder-configs
            :let [{:keys [levels]} (seed/layer-params dmap detail size 0.5 0.5 2.5 tiers cnt 512 512)
                  ;; coarse→fine: largest ssz first
                  coarse-fine (sort-by :ssz > (remove :band levels))]]
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

(deftest glaze-alpha-has-one-threshold-at-every-paintable-size
  ;; level-alpha carried a third arm below dab-max (1.2) for a DAB tier that min-phys
  ;; (1.4) makes unreachable, and the GLSL twin had drifted to keying that arm at 2.5 —
  ;; where it DID fire. Every fine stroke between 1.4 and 2.5 therefore painted 0.95 on
  ;; the GPU against 0.85 on the CPU (splat-painter-b1d), on every render, since that is
  ;; exactly where the fine tier sits. Pins the CPU half across the whole paintable range;
  ;; check.clj pins the GLSL literal that has to match it.
  (let [level-alpha #'splat-painter.seed/level-alpha]
    (doseq [ssz [1.4 1.6 2.0 2.4 2.5 3.5 7.9]]
      (is (== 0.85 (level-alpha 2 ssz))
          (str "detail stroke at " ssz "px glazes at 0.85 — no second threshold below 8px")))
    (doseq [ssz [8.0 12.0 20.5]]
      (is (== 1.0 (level-alpha 2 ssz))
          (str "a stroke at " ssz "px is coverage-scale and opaque")))
    ;; min-phys is what made the removed arm unreachable; if it ever drops below 1.2 the
    ;; question reopens on both paths at once.
    (is (>= 1.4 1.2) "min-phys (1.4) floors every emitted stroke above the old dab threshold")))

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

(defn- ladder-img []
  ;; multi-scale texture: a coarse ramp the base tier answers plus a 2-3px checker the
  ;; detail tiers reach for. Its detail runs out around level 4, so the ladder hits the
  ;; min-phys floor there — the DSC_8428-at-Size-13.66 case.
  (gray-img 256 256 (fn [x y]
                      (let [coarse (* 0.25 (+ 1.0 (Math/sin (* 0.04 (+ x y)))))
                            fine  (if (odd? (+ (quot x 2) (quot y 3))) 0.12 -0.12)]
                        (max 0.0 (min 1.0 (+ coarse fine)))))))

(defn- dense-img []
  ;; FULL-CONTRAST 2px checker: every rung of the ladder has detail to answer, so the fine
  ;; tier's natural density (17060 candidates at level 6) runs ~1700x over the slice the
  ;; budget affords it. This is the fixture where the old drop-whole gate bit: it cut the
  ;; ladder at 4 rungs with a 3.59px floor, where admit-and-thin reaches 7 rungs at 1.40.
  (gray-img 256 256 (fn [x y] (if (odd? (+ (quot x 2) (quot y 2))) 0.9 0.1))))

(deftest fine-tier-is-admitted-and-thinned-not-dropped
  ;; The fine tier (lvl >= broad-end) was dropped OUTRIGHT when its cost exceeded the
  ;; remaining budget, which is what made Detail inert above 0.6: the level the slider
  ;; wanted to add was tens of times over its slice, so it never appeared at any Detail. It
  ;; is now admitted and thinned against that slice the way levels 2-3 are thinned against
  ;; theirs, so it contributes in proportion to what the budget affords.
  ;;
  ;; Discriminating: restore the old `(and fine? (> cost rem))` drop and this fixture comes
  ;; back 4 rungs with a 3.59px floor, so both the seq and the floor assertion fail.
  (let [H 256 W 256 area (* H W)
        img (dense-img)
        dmap (wavelet/placement-map img (structure/analyze img))
        levels (:levels (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5 [1.0 1.0 1.0 1.0] 4000 H W))
        ladder (remove :band levels)
        ;; the density the level's own spacing asks for, before any thinning
        natural (fn [l] (Math/ceil (/ (double area) (* (double (:sp l)) (double (:sp l))))))
        thin    (fn [l] (/ (double (:nx l)) (natural l)))
        fine (filter #(>= (long (:lvl %)) 4) ladder)
        mid  (filter #(and (>= (long (:lvl %)) 2) (< (long (:lvl %)) 4)) ladder)]
    (is (seq fine) "the fine tier is admitted, not dropped whole")
    (is (every? #(< (thin %) 0.1) fine)
        (str "...and thinned hard to fit its slice: " (mapv thin fine)))
    (is (every? pos? (map :nx fine)) "thinned to a real candidate pool, not to nothing")
    (is (seq mid) "the mid tier is still there")
    ;; INDEPENDENT slices. One shared cand-thin would drag the mid tier down with the fine
    ;; tier's demand — the coarsening the two-tier budget exists to prevent — while letting
    ;; each thin against the WHOLE detail budget overspends it (Splats 1000 hit 1662 against
    ;; the 1500 bound above). Here the mid tier fits its slice untouched while the fine tier
    ;; thins ~100x, which only separate slices can produce.
    (is (every? #(> (thin %) (* 10.0 (thin (first fine)))) mid)
        (str "mid and fine thin by their own factors: mid " (mapv thin mid)
             " vs fine " (mapv thin fine)))
    ;; the ladder now descends to the min-paintable floor instead of stopping short of it
    (is (approx= 1e-9 1.4 (reduce min (map :ssz ladder)))
        (str "the finest rung sits on the min-phys floor (1.4), got "
             (reduce min (map :ssz ladder))))))

(deftest ladder-depth-is-bounded-by-the-floor-not-by-detail
  ;; Redundancy is the only drop left, and it is also what TERMINATES the ladder: the level
  ;; after the one landing on min-phys is always within keep-ratio of it, so the ladder ends
  ;; one rung past the floor however high Detail goes. Where an image's detail runs out at
  ;; the floor (ladder-img, the DSC_8428 case) Detail above 0.6 therefore adds no rungs —
  ;; the slider's advertised seven levels are bounded by Size and min-phys, not by Detail.
  ;; Pins the bound so a future change to the drop rules cannot reopen the ladder into
  ;; sub-paintable dust. (On dense-img, where detail survives past the floor, Detail DOES
  ;; still add rungs — see fine-tier-is-admitted-and-thinned-not-dropped, 7 of them.)
  (let [dmap (wavelet/placement-map (ladder-img) (structure/analyze (ladder-img)))
        ladder-at (fn [det] (->> (seed/layer-params dmap det 6.0 0.5 0.5 2.5 [1.0 1.0 1.0 1.0] 4000 256 256)
                                 :levels (remove :band) (mapv :ssz)))]
    (is (= (ladder-at 0.6) (ladder-at 1.0))
        (str "Detail 0.6 and 1.0 admit the same ladder: " (ladder-at 0.6) " vs " (ladder-at 1.0)))
    (is (= (ladder-at 0.8) (ladder-at 1.0)) "and every Detail in between")
    ;; strictly finer per rung, so no two rungs are the duplicate pass the redundancy rule
    ;; is there to drop (:levels is finest-first, so the sigmas run increasing)
    (is (apply < (ladder-at 1.0)) "rungs stay strictly finer, finest-first")
    (is (every? (fn [[fine coarse]] (< fine (* 0.95 coarse)))
                (partition 2 1 (ladder-at 1.0)))
        "each rung clears keep-ratio against the one above it")))

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
        ;; `remove :band`: the edge-band overlay carries the highest :lvl of all (it sits
        ;; outside the ladder's index range so its hash stream cannot collide), so a bare
        ;; max-key :lvl would measure the OVERLAY here instead of the ladder's finest rung.
        finest-nx (fn [cnt]
                    (->> (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5 [0.4 0.4 0.4] cnt 1024 1024)
                         :levels (remove :band) (apply max-key :lvl) :nx long))]
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
                                                    0.5 0.0 :sharp 1.0 px 0.5 0.55 0.0)]
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
                                      0.5 0.0 :sharp 1.0 blurd 0.5 0.55 0.0)]
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




;; --- the EDGE-BAND tier -------------------------------------------------------
;; An OVERLAY, not a rung of the coarse→fine ladder: placed off the raw edge channel,
;; born long-and-thin, pushed clear of the ridge, drawn topmost. See seed/band-level.
;; Every assertion below is paired with the case that makes it FAIL if the mechanism
;; is disabled — a flat image, a low Detail setting, or the coherence-derived elongation.

(defn- band-img
  "512×512 with ONE straight SOFT silhouette — a bright subject against a dark ground,
   the transition spread over ~10px the way a defocused background boundary is. The
   fine texture is confined WELL INSIDE the bright side (x < 230), away from the
   transition, so the silhouette carries edge-gradient energy but no fine-band energy:
   that separation is the whole reason a tier keyed on :edge reaches a place a tier
   keyed on :sharp never does. The texture still gives the ladder a detail rung to
   admit, so the band has a finest stdev to size itself against."
  []
  (gray-img 512 512 (fn [x _y]
                      (let [ramp (min 1.0 (max 0.0 (/ (- 250.0 (double x)) 10.0)))
                            tex  (if (and (< x 230.0) (odd? (quot (long x) 3))) 0.06 0.0)]
                        (max 0.0 (min 1.0 (+ (* 0.7 ramp) tex)))))))

(defn- band-of
  "The edge-band level at Cut-in `dial` (tier-muls[3]), or nil when the tier is absent."
  [dmap dial H W]
  (first (filter :band (:levels (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5
                                                   [0.4 0.4 0.4 dial] 600000 H W)))))

(defn- disk-field
  "A grid of small bright disks on dark ground: each disk's silhouette is a tight closed
   curve, so a band stroke turn-kills (cos 35°/step) after a couple of segments — the
   short, photo-like trace length that band-img's single straight ramp does NOT exercise.
   Used to calibrate/check the band's demand against its real per-candidate yield."
  [H W cell rad]
  (gray-img H W (fn [x y]
                  (let [centres (for [i (range 0 H cell) j (range 0 W cell)]
                                  [(double (+ i (/ cell 2.0))) (double (+ j (/ cell 2.0)))])
                        inside? (some (fn [[cx cy]]
                                        (<= (+ (* (- x cx) (- x cx)) (* (- y cy) (- y cy))) (* rad rad)))
                                      centres)]
                    (if inside? 0.9 0.25)))))

(deftest edge-band-tier-places-off-the-raw-edge-channel
  ;; map-at :edge must be wavelet/edge-at EXACTLY — unnormalized, unlike the three
  ;; wavelet bands. The GLSL twin is pinned separately in check.clj ("if (sel == 3)").
  (let [img   (band-img)
        dmap  (wavelet/placement-map img (structure/analyze img))
        map-at #'splat-painter.seed/map-at]
    (doseq [[x y] [[245.0 100.0] [247.5 300.0] [40.0 40.0] [500.0 480.0]]]
      (is (== (double (map-at dmap :edge x y)) (double (wavelet/edge-at dmap x y)))
          (str "map-at :edge at " [x y] " is the raw edge channel")))
    ;; NEGATIVE RESULT, pinned so it is not re-assumed: the tier is NOT keyed on :edge
    ;; because the sharp band is weak at a soft silhouette. It is not — on this fixture
    ;; sharp-at SATURATES at 1.0 right on the ramp (edge reads 0.46 there), because the
    ;; locally-normalized bands light up on any local variation. So "a denser :sharp
    ;; tier never reached the band" is a wrong explanation of the measured +7.28 ->
    ;; +24.85; whatever made densifying worse, it was not absence.
    (is (>= (double (map-at dmap :sharp 245.0 100.0)) 0.9)
        "the sharp band is NOT weak at a soft silhouette — do not key the tier on that premise")
    ;; What the edge channel actually buys is LOCALIZATION: it peaks on the silhouette
    ;; and dies in the flat ground, so a threshold on it selects the boundary itself —
    ;; which is where a tier that restates a boundary has to be seeded.
    (let [e-ridge (double (map-at dmap :edge 245.0 100.0))
          e-flat  (double (map-at dmap :edge 400.0 100.0))]
      (is (>= e-ridge 0.30) (str "the soft silhouette clears the band threshold: " e-ridge))
      (is (< e-flat 0.30)   (str "the flat ground does not: " e-flat))
      (is (> e-ridge (* 3.0 (max e-flat 1e-6)))
          (str "the edge channel localizes the silhouette: ridge " e-ridge " vs ground " e-flat)))
    (is (= (:map-kind (band-of dmap 1.0 512 512)) :edge)
        "the band tier is wired to that map")))

(deftest edge-band-tier-is-born-thin-across-the-edge
  ;; the point of the tier: its across-edge sigma is a property of the TIER
  ;; (ssz/selong), not of the local tensor, and the push off the ridge exceeds
  ;; 2 sigma across so the stroke cannot straddle the boundary it restates.
  (let [img  (band-img)
        dmap (wavelet/placement-map img (structure/analyze img))
        {:keys [ssz selong sideo]} (band-of dmap 1.0 512 512)
        across (/ (double ssz) (double selong))
        along  (* (double ssz) (double selong))]
    (is (< across 1.0) (str "across-edge sigma " across " is sub-pixel"))
    (is (> (/ along across) 6.0) (str "aspect ratio " (/ along across) " is a line, not a daub"))
    ;; the SMALLEST push (jitter 0.6×, see soff) must still clear 2 sigma across
    (is (> (* 0.6 (double sideo) (double ssz)) (* 2.0 across))
        "even the least-pushed band stroke sits more than 2 sigma clear of the ridge")
    ;; discriminating: the coherence-derived elongation this replaces is far fatter.
    ;; se = sqrt(1 + min(stroke,1.5)·coh·(0.25+0.75·D)) maxes at sqrt(2.5) ≈ 1.58.
    (is (> (double selong) (Math/sqrt 2.5))
        "forced elongation exceeds anything the coherence-derived formula can reach")))

(deftest edge-band-tier-needs-edges-and-the-cutin-dial
  (let [flat  (gray-img 512 512 (fn [_ _] 0.5))
        edged (band-img)
        dmap-flat  (wavelet/placement-map flat  (structure/analyze flat))
        dmap-edged (wavelet/placement-map edged (structure/analyze edged))]
    (is (nil? (band-of dmap-flat 1.0 512 512))
        "a flat image has no edges, so the band tier does not exist")
    (is (some? (band-of dmap-edged 1.0 512 512))
        "an image with a silhouette DOES get the tier — the flat case above is not vacuous")
    ;; the Cut-in dial is the control: 0 is off, and it scales DENSITY above that. There
    ;; is deliberately no Detail gate — a dial that silently did nothing below another
    ;; slider's threshold would be worse than no dial.
    (is (nil? (band-of dmap-edged 0.0 512 512))
        "Cut-in 0 turns the tier off entirely")
    (let [half (band-of dmap-edged 0.5 512 512)
          full (band-of dmap-edged 1.0 512 512)]
      (is (some? half) "a partial Cut-in still places the tier")
      (is (< (long (:nx half)) (long (:nx full)))
          (str "Cut-in scales density: " (:nx half) " candidates at 0.5 vs " (:nx full) " at 1.0"))
      (is (= (:ssz half) (:ssz full))
          "Cut-in scales DENSITY, not stroke geometry — the measured lever is coverage"))))

(deftest edge-band-tier-is-drawn-topmost
  ;; levels are finest-first and composited front-to-back, so index 0 is the topmost
  ;; paint. The band exists to COVER the coverage tiers' outward bleed, so it has to
  ;; sit there — and every other level's candidate offset must shift past its block.
  (let [img    (band-img)
        dmap   (wavelet/placement-map img (structure/analyze img))
        levels (:levels (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5 [0.4 0.4 0.4] 600000 512 512))
        band   (first levels)]
    (is (:band band) "the band tier is level index 0 (topmost)")
    (is (zero? (long (:offset band))) "and owns the first candidate block")
    ;; offsets stay a valid cumulative partition after the prepend
    (is (= (map :offset levels)
           (reductions + 0 (map #(* (long (:nx %)) (long (:ny %))) (butlast levels))))
        "candidate offsets remain cumulative across the prepended tier")
    (is (every? false? (map :band (rest levels)))
        "exactly one band tier")))

(deftest edge-band-tier-comes-out-of-the-budget
  ;; The tier must be charged against the detail slice, not added on top of it. This
  ;; only BITES where the budget cap is already binding — at a slack budget nothing
  ;; needs to thin and the charge is correctly a no-op — so measure at 72000 on a
  ;; 512x512, where the detail tier's demand exceeds its slice.
  ;;
  ;; Discriminating: with the charge removed (det-budget not subtracting :demand) the
  ;; suite fails on admitted-levels-fit-the-budget with 118330 splats against a 108000
  ;; bound — the band's candidates went straight on top of a field that was already at
  ;; 1.44x. That test is the end-to-end guard; this one pins the mechanism.
  (let [img  (band-img)
        dmap (wavelet/placement-map img (structure/analyze img))
        lp   (fn [cutin] (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5 [0.4 0.4 0.4 cutin] 72000 512 512))
        det-cand (fn [p] (->> p :levels (remove :band) (filter #(>= (long (:lvl %)) 2))
                              (map #(long (:nx %))) (reduce + 0)))
        off (lp 0.0) on (lp 1.0)]
    (is (nil? (first (filter :band (:levels off)))) "Cut-in 0: no band tier")
    (is (some? (first (filter :band (:levels on)))) "Cut-in 1: the tier is admitted")
    (is (< (det-cand on) (det-cand off))
        (str "admitting the band thins the ladder's detail tier: "
             (det-cand off) " -> " (det-cand on) " candidates"))))

(deftest edge-band-demand-tracks-what-the-tier-paints
  ;; The band's :demand is subtracted from the detail slice (see ...comes-out-of-the-budget),
  ;; so it must predict what the tier ACTUALLY emits. The old nx·frac·band-segs term is
  ;; cap-bound at exactly band-share·budget (18000 at Splats 72000) regardless of the realised
  ;; yield, which over-charged ~2.2× — the band truly paints ~8200 splats at 1024px
  ;; (DSC_8428), not the 18000 it was charged — and the over-charge came straight out of the
  ;; detail tier's candidate allowance (cand-thin). Calibrated band-trace against the band's
  ;; MEASURED mean traced length on two photos (DSC_8428 ~5.3, coyote ~8.6 segs/survivor);
  ;; band-segs (12) still sets spacing and the nx cap, so the tier paints identically — only
  ;; its charge drops. This pins the RELATIONSHIP: demand within 1.5× of the splats the tier
  ;; actually contributes on a fixture where it is live. The bound is deliberately that tight
  ;; — measured demand/pure runs 0.92 here (0.67 coyote 512, 1.12 DSC 1024) while the old
  ;; band-segs charge measures 1.83 here (1.34 / 2.25), so a looser 3× would admit the very
  ;; over-charge this test exists to catch.
  ;;
  ;; "What it contributes" is the band's PURE segment count — segments whose level carries
  ;; selong>0 (band = band-se 2.6; ladder = 0) — counted straight out of layered-means. NOT
  ;; the Cut-in on/off field delta: that is confounded, because the demand it tests changes
  ;; cand-thin between the on/off runs (a too-high charge starves the detail tier in the ON
  ;; run, so the delta under-counts the band). band-img's single straight ramp traces far too
  ;; long (rt≈28) to exercise this, so a disk field supplies the short photo-like traces.
  (let [img  (disk-field 256 256 14 3)
        sfield (structure/analyze img)
        dmap (wavelet/placement-map img sfield)
        nf   (seed/prep-noise sfield)
        px   (:pixels img)
        cnt  72000  H 256 W 256  area (* H W)
        layered-means #'splat-painter.seed/layered-means
        band-level    #'splat-painter.seed/band-level
        segs (layered-means dmap nf 0.6 6.0 0.5 0.5 2.5 1.0 [1.0 1.0 1.0 1.0] cnt H W px px)
        pure (count (filter (fn [s] (pos? (double (nth s 14)))) segs))
        lp   (seed/layer-params dmap 0.6 6.0 0.5 0.5 2.5 [1.0 1.0 1.0 1.0] cnt H W)
        finest (->> lp :levels (remove :band) (map :ssz) (reduce min))
        demand (double (:demand (band-level dmap 1.0 area cnt finest)))]
    (is (pos? pure) "the band tier is live on this fixture")
    (is (<= demand (* 1.5 pure))
        (str "demand must not over-charge: demand " demand " vs pure " pure
             " (x" (/ demand (double pure)) ")"))
    (is (>= demand (/ pure 1.5))
        (str "demand must not under-charge: demand " demand " vs pure " pure
             " (x" (/ demand (double pure)) ")"))))

(deftest detail-demand-tracks-what-the-tier-paints
  ;; DETAIL-tier twin of edge-band-demand-tracks-what-the-tier-paints. det-demand charges
  ;; the FULL candidate pool (surv 1.0 — lvl-frac under-predicted dithered survival ~20x,
  ;; see layer-params) x a per-candidate YIELD constant (det-yield). It must predict what
  ;; the tier ACTUALLY emits, because cand-thin = det-budget/det-demand: an inflated demand
  ;; throttles the placed candidate pool by the same factor and the field lands far under the
  ;; requested count. The old charge was expected-segs (4), which sets SEED SPACING and the
  ;; k-of term and must stay — but measured detail emission is only ~1.1-2.0 splats per
  ;; candidate (DSC_8428 ~1.3, coyote ~1.9 at 512/1024px, budget-invariant across 36k-600k),
  ;; so 4 over-charged ~2-3x and cand-thin sat near 0.24, spending ~52% of budget. det-yield
  ;; carries the MEASURED yield for the demand term only, the way band-trace carries the
  ;; band's measured length; expected-segs still drives spacing.
  ;;
  ;; Emission counted straight out of layered-means with stroke-segments wrapped to tag each
  ;; segment row with its level (index 15): detail = selong==0 (excludes the band, which is
  ;; charged by band-trace) and lvl>=2 (excludes coverage 0-1). NOT a Cut-in on/off delta —
  ;; that is confounded, because the demand it tests moves cand-thin between the two runs.
  ;; cand-thin is 1 here (small fixture, high count) so emission = raw_nx x yield and the
  ;; demand/emission ratio is exactly charge/yield — which is why flipping det-yield back to
  ;; expected-segs (4) trips the upper bound (ratio ~2.5) while the calibrated value sits
  ;; inside [0.67, 1.5].
  (let [img (gray-img 128 128 (fn [x y] (if (odd? (+ (int (quot x 8)) (int (quot y 8)))) 0.15 0.85)))
        sfield (structure/analyze img)
        dmap (wavelet/placement-map img sfield)
        nf (seed/prep-noise sfield)
        px (:pixels img)
        cnt 72000  H 128  W 128
        layered-means #'splat-painter.seed/layered-means
        orig @#'splat-painter.seed/stroke-segments
        segs (with-redefs [splat-painter.seed/stroke-segments
                           (fn [& args]
                             (let [lvl (long (nth args 2))
                                   [rows reason] (apply orig args)]
                               [(mapv (fn [r] (conj r lvl)) rows) reason]))]
              (layered-means dmap nf 0.6 6.0 0.5 0.5 2.5 1.0 [1.0 1.0 1.0 1.0] cnt H W px px))
        emitted (count (filter (fn [s] (and (zero? (double (nth s 14)))
                                            (>= (long (nth s 15)) 2))) segs))
        lp (seed/layer-params dmap 0.6 6.0 0.5 0.5 2.5 [1.0 1.0 1.0 1.0] cnt H W)
        demand (double (:det-demand lp))]
    (is (pos? emitted) "the detail tier is live on this fixture")
    (is (<= demand (* 1.5 emitted))
        (str "detail demand must not over-charge: demand " demand " vs emitted " emitted
             " (x" (/ demand (double emitted)) ")"))
    (is (>= demand (/ emitted 1.5))
        (str "detail demand must not under-charge: demand " demand " vs emitted " emitted
             " (x" (/ demand (double emitted)) ")"))))

(deftest fine-tier-is-charged-its-own-lower-yield
  ;; The FINE tier (lvl >= broad-end) sits on the min-phys floor and paints measurably LESS
  ;; per candidate than the mid rungs above it — it is where survival is thinnest and chains
  ;; die shortest. Measured per level by tagging every emitted segment with the level that
  ;; made it (`jolt -M:yield <image> 1024 '{:count 550000 :size 20.48 :detail 1.0}'`), 1024px:
  ;;               DSC_8428  coyote  crow  street  portrait
  ;;   fine σ1.40      0.83    0.85  0.53    1.43      1.07   → mean 0.94
  ;;   mid  σ2.56      1.05    1.43  0.83    2.20      1.66
  ;;   mid  σ5.12      0.67    2.00  0.45    1.68      0.83   → mid mean ~1.28
  ;; Fine runs ~78% of mid on every one. Charging both 1.8 therefore thinned the fine tier to
  ;; about half the density its own slice had already paid for, because fine-thin =
  ;; fine-slice/fine-demand: the over-charge comes straight back out of the pool the tier
  ;; places from. The tier being ADMITTED at all is pinned separately (see
  ;; fine-tier-is-admitted-and-thinned-not-dropped); this pins what it is charged.
  ;;
  ;; What is NOT pinned here is the absolute yield, and deliberately: it is image texture, not
  ;; a constant. The same -M:yield run reads 0.06 on the low-contrast ladder fixture and 11 on
  ;; the full-contrast dense one, against 0.53-1.43 on photos, so no synthetic fixture can
  ;; judge the absolute value and an assertion against one would be pinning noise. A per-image
  ;; measured calibration is what closes that gap (splat-painter-zig). The systematic part —
  ;; fine costs less per candidate than mid — is what one constant can carry, so that is what
  ;; this asserts. Discriminating: put both tiers back on one constant and the ratio is 1.0.
  (let [H 256 W 256 area (* H W)
        img (dense-img)
        dmap (wavelet/placement-map img (structure/analyze img))
        lp (seed/layer-params dmap 1.0 6.0 0.5 0.5 2.5 [1.0 1.0 1.0 1.0] 4000 H W)
        ladder (remove :band (:levels lp))
        ;; the density the level's own spacing asks for, before any thinning — the pool the
        ;; demand terms are computed over
        natural (fn [l] (Math/ceil (/ (double area) (* (double (:sp l)) (double (:sp l))))))
        pool (fn [pred] (reduce + 0.0 (map natural (filter pred ladder))))
        ;; Detail 1.0 ⇒ 7 requested levels ⇒ broad-end 4
        mid-pool  (pool (fn [l] (and (>= (long (:lvl l)) 2) (< (long (:lvl l)) 4))))
        fine-pool (pool (fn [l] (>= (long (:lvl l)) 4)))
        mid-charge  (/ (double (:mid-demand lp)) mid-pool)
        fine-charge (/ (double (:fine-demand lp)) fine-pool)]
    (is (pos? fine-pool) "the fine tier is live on this fixture")
    (is (pos? mid-pool)  "and so is the mid tier")
    (is (< fine-charge (* 0.8 mid-charge))
        (str "the fine tier must be charged its own lower yield: fine " fine-charge
             " vs mid " mid-charge " (x" (/ fine-charge mid-charge) ")"))
    ;; the thinning must still fire on this fixture, or the charge above is moot
    (is (< (double (:fine-thin lp)) 1.0) "the fine tier is thinned against its slice here")))

(deftest detail-budget-spend-not-throttled-by-overcharge
  ;; cand-thin = det-budget/det-demand, so a det-demand inflated ~2-3x over real emission
  ;; cuts the placed detail candidate pool by the same factor and the WHOLE field lands far
  ;; under the requested count (the measured ~52% at Splats 72000). On a textured image
  ;; where the detail tier dominates the field, spend must reach a healthy fraction of the
  ;; budget — not the ~50% the old expected-segs(4) charge left it at. 70% is the honest
  ;; floor for a single per-candidate constant across images whose yield spans ~1.1-2.0
  ;; (see detail-demand-tracks-what-the-tier-paints): raising the charge to chase 100% on a
  ;; low-yield image under-charges a high-yield one past the budget, which is just as wrong.
  (let [img (gray-img 256 256 (fn [x y] (if (odd? (+ (int (quot x 8)) (int (quot y 8)))) 0.15 0.85)))
        cnt 20000
        {:keys [splats]} (seed/splat-field img {:count cnt :size 6.0 :detail 0.6 :stroke 2.5})]
    (is (>= (count splats) (* 0.70 cnt))
        (str "field must spend most of the budget: " (count splats) " of " cnt
             " (" (* 100.0 (/ (double (count splats)) cnt)) "%)"))))



;; --- Swirl: the dial on the image-INDEPENDENT Perlin noise ------------------
;; Two placement terms are steered by a Perlin field rather than by the photo: the
;; flat-region flow that orients strokes where the tensor has no opinion, and the
;; position warp that pushes flat-region seeds off the level lattice. Both are
;; spatially COHERENT — neighbours drift and turn together — which is what makes
;; them read as brushwork, and also what lets them carry structure away from where
;; the photo put it. Swirl is the dial: 1.0 is the shipped look, 0.0 keeps the
;; lattice-breaking but sources it from the image (orientation) or from each seed's
;; own hash (position) instead of a shared field.

(defn- swirl-img []
  ;; CROSSED bars: each bar is high-coherence (the tensor decides orientation on it,
  ;; whatever Swirl says), and because their directions cancel in the diffused tensor
  ;; the flat ground between them has weak flow strength — which is exactly where the
  ;; Perlin swirl takes over. A single bar will NOT do: its orientation diffuses over
  ;; the whole frame at full strength (flow-str ≥ 0.99 everywhere), the swirl gets zero
  ;; weight, and every assertion below passes vacuously.
  (gray-img 128 128 (fn [x y] (if (or (and (> x 54) (< x 60)) (and (> y 30) (< y 36)))
                                0.9 0.35))))

(deftest prep-noise-bakes-both-ends-of-the-swirl-dial
  ;; The flow field is baked once per image (2.3s on a 1MP photo — far too slow to
  ;; re-bake on a slider drag), so prep-noise bakes BOTH extremes and the dial mixes
  ;; them. The mix is linear in the double-angle components, which is what lets the
  ;; GPU mix per texel fetch instead: bilinear and mix are both linear, so
  ;; mix(bilerp(A),bilerp(B)) = bilerp(mix(A,B)) and the paths cannot diverge.
  (let [sfield (structure/analyze (swirl-img))
        nf     (seed/prep-noise sfield)
        n      (count (:c2 nf))
        coh    (vec (:coherence nf))
        c2 (vec (:c2 nf)) s2 (vec (:s2 nf))
        c2s (vec (:c2s nf)) s2s (vec (:s2s nf))]
    (is (= n (count c2s)) "the swirl-free pair is baked at the flow field's resolution")
    (is (= n (count s2s)))
    ;; not vacuous: over the weak-flow ground the two ends of the dial point elsewhere
    (let [dev (mapv (fn [a b c d] (Math/sqrt (+ (* (- a b) (- a b)) (* (- c d) (- c d)))))
                    c2 c2s s2 s2s)
          moved (count (filter #(> (double %) 0.02) dev))]
      (is (> (/ (double moved) n) 0.1)
          (str "the Perlin swirl really does steer the flat regions away from the "
               "edge-seeded flow (" moved "/" n " texels move)")))
    ;; ...but never on a bar, where the tensor is coherent and structure wins in both
    (let [on-edge (keep-indexed (fn [i c] (when (> (double c) 0.95) i)) coh)]
      (is (seq on-edge) "the bars give us coherent texels to check")
      (is (every? (fn [i] (and (approx= 0.12 (nth c2 i) (nth c2s i))
                               (approx= 0.12 (nth s2 i) (nth s2s i))))
                  on-edge)
          "on a coherent edge both ends of the dial agree — Swirl cannot rotate a silhouette"))))

(deftest with-swirl-mixes-the-orientation-fields
  (let [sfield (structure/analyze (swirl-img))
        nf     (seed/prep-noise sfield)
        c2 (vec (:c2 nf)) c2s (vec (:c2s nf))]
    (is (identical? nf (seed/with-swirl nf 1.0))
        "Swirl 1.0 is the shipped field verbatim — no mix, no allocation, bit-identical")
    (is (= c2s (vec (:c2 (seed/with-swirl nf 0.0))))
        "Swirl 0 hands the sampler the edge-seeded field")
    (is (every? true? (map (fn [a b m] (approx= 1e-12 m (* 0.5 (+ a b))))
                           c2 c2s (vec (:c2 (seed/with-swirl nf 0.5)))))
        "and mixes linearly between them")
    (is (= (vec (:coherence nf)) (vec (:coherence (seed/with-swirl nf 0.3))))
        "coherence is a property of the photo — the dial does not touch it")))

(deftest swirl-decorrelates-the-position-warp
  ;; The warp is what breaks the residual level lattice, so it has to keep its
  ;; amplitude at Swirl 0 — only its spatial coherence changes. At 1.0 it is the
  ;; Perlin field: over its ~17px wavelength neighbouring seeds drift together, and
  ;; since the amplitude scales with stroke size (aw = warp·(1−D)·ssz), a large Size
  ;; makes the displacement map fold — that is the "melted" look at scale. At 0.0
  ;; each seed reads its own avalanche hash: same envelope, no shared direction.
  (let [aw 8.0
        row (fn [swirl salt] (mapv (fn [i] (* aw (seed/warp-noise swirl i 0 salt
                                                                  (* 0.06 (double i)) 0.0)))
                                   (range 400)))
        pearson (fn [v]
                  (let [a (butlast v) b (rest v) n (count a)
                        ma (/ (reduce + a) n) mb (/ (reduce + b) n)
                        num (reduce + (map (fn [x y] (* (- x ma) (- y mb))) a b))
                        da (Math/sqrt (reduce + (map #(let [d (- % ma)] (* d d)) a)))
                        db (Math/sqrt (reduce + (map #(let [d (- % mb)] (* d d)) b)))]
                    (/ num (max 1e-9 (* da db)))))
        coherent (row 1.0 61)
        hashed   (row 0.0 61)]
    (is (> (pearson coherent) 0.9)
        (str "Perlin warp: adjacent seeds drift together (r=" (pearson coherent) ")"))
    (is (< (Math/abs (pearson hashed)) 0.2)
        (str "hashed warp: adjacent seeds are independent (r=" (pearson hashed) ")"))
    (is (every? #(within? 0.0 aw %) hashed) "the displacement envelope is unchanged")
    (is (every? #(within? 0.0 aw %) coherent))
    ;; both ends still cover the cell — a dial that killed the warp would re-expose
    ;; the lattice the warp exists to hide
    (is (> (apply max hashed) (* 0.8 aw)) "the hashed warp still spans its amplitude")
    (is (not= (row 0.0 61) (row 0.0 67)) "the two axes draw independent streams")))

(deftest swirl-is-a-live-control-on-the-field
  ;; end-to-end: the dial reaches splat-field, defaults to the shipped look, and
  ;; actually moves paint at 0.
  (let [img (gray-img 64 64 (fn [x y] (if (and (> x 20) (< x 44) (> y 20) (< y 44))
                                        0.85 (* 0.4 (/ (double (+ x y)) 128.0)))))
        ctl {:count 3000 :size 6.0 :stroke 2.5 :detail 0.6 :variation 0.5
             :curvature 0.5 :opacity 0.9 :contrast 1.0}
        means (fn [c] (mapv :mean (:splats (seed/splat-field img c))))
        default (means ctl)
        one     (means (assoc ctl :swirl 1.0))
        zero    (means (assoc ctl :swirl 0.0))]
    (is (= default one) "Swirl defaults to 1.0 — the shipped field is untouched")
    (is (not= one zero) "and 0 places paint somewhere else")
    ;; the dial re-aims strokes rather than re-budgeting them: a warped seed lands on a
    ;; different detail value, so the dithered thresholds admit a slightly different
    ;; set, but the ladder and its budget are untouched.
    (is (< (Math/abs (- (count one) (count zero))) (* 0.1 (count one)))
        (str "the dial moves strokes, it does not re-budget them: "
             (count one) " -> " (count zero) " splats"))))
