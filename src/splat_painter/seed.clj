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
    :contrast   0.5..2.0 per-channel contrast about 0.5. 1.0 = no change.
    :background additive base; a number (gray) or [r g b]; defaults to black

  An image is {:height :width :pixels (flat H*W*3 double-array 0..1) :channels 3}.
  If it carries precomputed :structure, :detail, and :blur they are reused so
  live slider drags stay fast."
  (:require [splat-painter.gaussian :as gauss]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.noise :as noise]))

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

(defn- wang32
  "Wang avalanche hash over 32-bit values (exact mod-2^32 arithmetic — the GPU
   mirrors it in uint32). hash01's LINEAR mix is fine for jitter but as a POSITION
   generator frac(i·A) is a rank-1 lattice: the points fall on straight lines
   (Marsaglia hyperplanes) — worse stripes than any grid."
  [v]
  (let [v (mod (bit-xor (bit-xor (long v) 61) (bit-shift-right (long v) 16)) 4294967296)
        v (mod (* v 9) 4294967296)
        v (bit-xor v (bit-shift-right v 4))
        v (mod (* v 668265261) 4294967296)
        v (bit-xor v (bit-shift-right v 15))]
    (mod v 4294967296)))

(defn- poshash
  "Avalanche-hashed position component in [0,1) from candidate index + level + salt.
   Only the TOP 23 hash bits are used so the value is exactly representable in both
   float32 (GPU) and double (CPU) — full 32-bit fractions rounded differently in the
   GPU's float conversion and broke exact CPU/GPU count parity."
  [n lvl salt]
  (/ (double (bit-shift-right
               (wang32 (bit-xor (wang32 (+ (* (long n) 2) (long lvl)))
                                (mod (* (long salt) 2654435769) 4294967296)))
               9))
     8388608.0))

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
(def ^:private splat-budget 600000)

(defn- detail-fraction
  "Fraction of the map array under `key` (:detail aggregate or :sharp fine-band)
   at/above the normalized threshold t∈[0,1]. Used to estimate how many splats each
   fine level will place (so the budget can scale them)."
  [dmap key t]
  (let [^doubles d (or (get dmap key) (:detail dmap))
        dmax (double (max 1e-9 (:dmax dmap)))
        n    (alength d)
        thr  (* (double t) dmax)]
    (if (zero? n)
      0.0
      (loop [i 0 c 0]
        (if (>= i n)
          (/ (double c) (double n))
          (recur (inc i) (if (>= (aget d i) thr) (inc c) c)))))))

(defn- mean-inv-m2
  "E[1/m²] over the ABSOLUTE subject map, where m(x) = 1+(b−1)·(1−s(x)) is the
   bokeh-adaptive Broad multiplier (mirrors layered-means' mloc). The base
   level's budget term: candidates thin by (bmin/m)² as their strokes grow ×m,
   so the cost integrates the thinning."
  [dmap b]
  (let [^doubles d (or (:subject dmap) (:detail dmap))
        n    (alength d)]
    (if (zero? n)
      1.0
      (loop [i 0 acc 0.0]
        (if (>= i n)
          (/ acc (double n))
          (let [s (min 1.0 (aget d i))
                m (+ 1.0 (* (- (double b) 1.0) (- 1.0 s)))]
            (recur (inc i) (+ acc (/ 1.0 (* m m))))))))))

;; Fine-level seeds don't place one dab — they trace a BRUSH STROKE: a chain of
;; gaussian segments stepped along the orientation field. Stroke behaviour is
;; SCALE-RELATIVE (the coarse-to-fine pass adjusts its parameters to the scale it
;; is painting at): the broadest detail level lays long, freely-curving strokes;
;; each finer level makes shorter, straighter, more precise marks — and the two
;; finest levels read the SHARP fine-band detail map (wavelet/sharp-at), so they
;; land on (and preserve) text/eye-scale structure the smoothed aggregate blurs.
(defn- seg-count
  "segments per stroke at placement level — the FALLBACK table for NON-liner
   levels (broad/mid at large Size, where the chains are coverage strokes, not
   thin contour liners). LINER levels (nominal size < 3.5px, lvl≥2) override this
   via layer-params' segs-of with a span-targeted count (up to 32), so a small-σ
   level traces one long continuous line instead of the short stitched dashes this
   3-8 table lays down (the contour thatch). Budget-invariant: seed spacing scales
   with √segs, so the same segment count is arranged into fewer, longer strokes."
  [lvl]
  (let [l (long lvl)] (cond (zero? l) 1 (== l 1) 6 (== l 2) 4 (== l 3) 3 :else 8)))
(defn- step-frac
  "step length as a fraction of the level stdev. Fine liner strokes step ~0.9σ —
   close enough that the segment gaussians fuse into a smooth continuous rod, far
   enough that 8 segments span a real line along the edge."
  [lvl]
  (let [l (long lvl)] (cond (== l 1) 1.1 (== l 2) 0.9 (== l 3) 0.75 (== l 4) 0.9 (== l 5) 0.85 :else 0.8)))
(defn- bend-frac
  "How much of the Curvature Perlin bend this level keeps. The LINER tier (lvl≥4)
   keeps NONE: Perlin gives large/medium strokes their natural brush-vector
   variation, but fine strokes exist to follow the original detail exactly —
   any noise wobble at that scale reads as jaggedness, not character."
  [lvl]
  (let [l (long lvl)] (cond (== l 1) 1.0 (== l 2) 0.55 (== l 3) 0.3 :else 0.0)))
(defn- tier-mul
  "Per-tier size multiplier from [broad mid fine], keyed on the level's ROLE not its
   index. Coverage levels (0-1) take Broad (applied via bmul at emission, so smul
   returns 1.0 for them). Detail levels (lvl>=2) split by their RANK among the detail
   levels: the coarsest detail level (rank 0) carries BOTH Mid and Fine (Mid*Fine) so
   neither dial is a no-op at the default 3-level ladder (where it is the only detail
   level that admits); finer detail levels (rank>=1) take Fine. The OLD index key
   handed muls[2] to lvl>=4, which the monotone ladder never reaches — Fine was dead.
   `rank` is the 0-based position among detail levels (lvl 2 = rank 0)."
  [muls rank]
  (double (if (zero? (long rank))
            (* (nth muls 1) (nth muls 2))
            (nth muls 2))))

(def ^:private dab-max
  "Physical stdev (px) below which a level paints DABS instead of tracing chains.
   A dab has no path, so it cannot wander or dry out — the only sane mark below the
   scale at which the orientation field (a ~7px tensor average) carries usable
   direction. Set to 1.2 so the DETAIL tier TRACES feature-following strokes rather
   than dabbing: the feature-tracer (below) terminates on geometry, not on the
   colour-drift guard that used to lift sub-dab-max chains into short dashes. The
   min-paintable floor (min-phys 2.2) already keeps every emitted stroke above this,
   so in practice nothing dabs today — the machinery stays for genuinely
   sub-paintable levels if that floor is ever lowered."
  1.2)

;; --- feature-following tracer tunables (mirror the GLSL literals in gen.clj) ---
;; Stroke length is guided by the FEATURE, not a count: a trace stops where the ridge
;; dies or the tangent bends (a corner). These are the geometric/colour thresholds.
(def ^:private max-segs 32)        ; the geometry-shader vertex cap; detail levels run to it and let geometry decide
(def ^:private expected-segs 4)
;; ^ the MEAN traced length of a feature-following stroke, and the single constant
;; that now sets BOTH the fine tier's seed spacing and its budget term — they have to
;; agree or the budget scale and the real splat count diverge. It is a measurement,
;; not a preference: once the tracer stops at the feature boundary, chains are short,
;; and spacing derived from the old nominal count (32, via the segs cap) left the fine
;; tier at ~9.9k seeds when it needs ~35k. Swept 10/6/4 against the render: 10 (sp
;; 8.70, nx 13866) stays soft, 6 (sp 6.74, nx 23110) is intermediate, 4 (sp 5.50, nx
;; 34664) is where the camera strap, the phone edge and the lips come back. Re-measure
;; this if the stop rules change — it is the mean stroke length they produce.
(def ^:private edge-floor 0.10)    ; ridge-alive stop: the edge is dead below this, the feature has ended
(def ^:private bend-cos 0.90)      ; bend-break: |dot(field-dir, prev-step)| below this (~26°/step) is a corner
(def ^:private runaway 0.60)       ; chroma BACKSTOP only: the stroke has wandered into a foreign colour region

(def ^:private dab-overlap
  "Dab spacing as a multiple of the stroke stdev. A gaussian reads out to ~2σ, so at
   2σ spacing about three dabs meet at any point — sparse enough that each mark stands
   on its own, dense enough to cover. The chain spacing formula (1.25·√segs) collapses
   to ~1.25σ for a 1-segment mark and packs them ~12 deep, which is mush."
  2.0)

(defn- dab-level?
  "Does this level paint dabs rather than tracing chains? Only the DETAIL tiers
   (lvl>=2) ever dab. Levels 0-1 are the COVERAGE tiers by role, not by size: their
   job is an unbroken underpainting, and dab spacing (2σ) would thin level 1's grid
   ~3x and open gaps in it."
  [lvl ssz]
  (and (>= (long lvl) 2) (< (double ssz) dab-max)))

(defn- level-alpha
  "Paint translucency by PHYSICAL stroke size — progressive refinement: broad layers
   are opaque coverage, mid layers glaze so detail builds on the underpainting rather
   than scratching over it. DABS are near-opaque on purpose: at 1-2px they are placed
   sparsely (see dab-overlap) so each mark must STAND ON ITS OWN. Glaze alpha there
   just averages neighbours into mush — the exact failure that made fine detail soft."
  [lvl ssz]
  (if (<= (long lvl) 1)
    1.0                          ; coverage tiers always fully opaque, by role
    (let [v (double ssz)]
      (cond (>= v 8.0) 1.0
            (>= v dab-max) 0.85
            :else 0.95))))

(defn- level-map-kind
  "Which placement map a level reads, matched to the scale it paints — keyed on
   PHYSICAL size, not the level index. Index-keyed, a ladder that ends at level 2-3
   never reached the :else branch, so the :sharp fine-band map — the one built to find
   eye/text-scale structure — was NEVER CONSULTED and the finest tier placed off the
   dimmer :mid band. Measured: the fine tier then put ~0 strokes on the face, so all
   face detail came from the 4px broad tier and no amount of fine-tier tuning changed
   anything."
  [lvl ssz]
  (if (<= (long lvl) 1)
    :detail                     ; coverage tiers always place off the aggregate map
    (let [v (double ssz)] (if (>= v 3.5) :mid :sharp))))
(defn- raw-floor
  "Colour-rawness floor by PHYSICAL stroke size: a small stroke must paint faithful
   colour — a half-blur blend at feature scale just softens the feature it exists to
   keep. Keyed on size, NOT the level index: once the monotone ladder ends at level
   2-3 the index-keyed form handed the FINEST strokes mid-tier averaged colour
   (t capped at 0.7 — 30% blur at exactly the scale that wants fidelity), which is
   why small details read as unclear."
  [lvl ssz]
  (if (<= (long lvl) 1)
    0.0                          ; coverage tiers: faithful colour, by role
     (let [v (double ssz)]
       (cond (>= v 8.0) 0.0 (>= v 3.5) 0.45 (>= v 1.5) 0.7 :else 0.85))))
(defn- density-scaled-traw
  "Scale a detail stroke's raw-fidelity floor by the FINEST wavelet band (`sharp-at`,
   0..1). At feature scale (a ~1.4px stroke, traw 0.85) 85% of the colour is a single
   RAW sample; where features crowd (sharp-at high) that one sample lands on a shadow /
   lip line 3-5px from the next feature and paints a foreign dark mark (the band above
   the upper lip). Trust the bilateral (region) colour more there. An isolated crisp
   feature (sharp-at low) keeps full raw fidelity. Coverage tiers (lvl 0-1) are never
   scaled — they paint faithful colour by role. GLSL mirror: gen.clj emitSplat block
   (`if (lvl >= 2) traw *= 1.0 - 0.7 * sharpAt(cx, cy);`)."
  [lvl traw sharp-at]
  (if (<= (long lvl) 1)
    traw
    (let [dens (double sharp-at)]
      (* (double traw) (- 1.0 (* 0.7 dens))))))
(defn- spec-cap
  "Ceiling on colour SPECIFICITY (the blur→raw blend t) by PHYSICAL stroke size —
   progressive colour refinement: a fat brush cannot place a pixel-specific highlight,
   so broad layers stay AVERAGED and full specificity arrives only at feature scale.
   Size-keyed for the same reason as raw-floor: index-keyed, the finest surviving
   level was capped at 0.7 and painted 30% blur."
  [lvl ssz]
  (if (<= (long lvl) 1)
    0.35                         ; coverage tiers: averaged colour, by role
    (let [v (double ssz)]
      (cond (>= v 8.0) 0.35 (>= v 3.5) 0.7 :else 1.0))))
;; The PHYSICAL stroke stdev below which a level reads as a drawn LINE and earns liner
;; discipline (gentle ridge snap, direction momentum, line-hold, impasto body). A named
;; constant so the Clojure and the GLSL twin stay pinned to one value (the GS uses the
;; literal 3.5 in `bool liner`, `aw` and `body`).
(def ^:private liner-threshold 3.5)
(defn liner-scale?
  "lvl<2 (base/broad coverage) is never a liner; otherwise a level is liner-scale iff
   its PHYSICAL stdev is below `liner-threshold` (boundary exclusive). Keys on physical
   size, not the level index — which level paints fine detail depends on the sliders."
  [lvl ssz]
  (and (>= (long lvl) 2) (< (double ssz) liner-threshold)))

(defn- stroke-len-frac
  "The Stroke slider as stroke LENGTH: scales the chain step. 2.5 (default) = 1.0."
  [stroke]
  (+ 0.4 (* 0.24 (double stroke))))

;; --- the EDGE-BAND tier -------------------------------------------------------
;; A tier that OWNS the silhouette band, laid OVER the coverage tiers. It is NOT part
;; of the coarse→fine ladder: it is placed by the raw EDGE channel, born strongly
;; elongated ALONG the edge (thin ACROSS it), and pushed CLEAR of the ridge onto one
;; side, so it restates a boundary with that side's OWN colour instead of straddling it.
;;
;; Why a separate tier at all (spec-coat-edge-fuzz — twelve measured rejections):
;; a stroke straddling a soft silhouette carries ONE colour across a body that spans
;; the whole ramp, so it paints the ramp's MID value on the dark side. Level 1 alone
;; lifts the outward band +49.6 luma at coat-likeness 1.11. Shrinking those strokes,
;; suppressing them, clipping their reach and densifying the fine tier were all
;; measured WORSE — suppression just hands coverage to level 0, which reaches further.
;; This tier differs in KIND on four counts, each one answering a rejected attempt:
;;   • placed off :edge, not :sharp — the edge channel LOCALIZES the silhouette (it
;;     peaks on the ridge and dies in flat ground), so a threshold on it seeds strokes
;;     on the boundary itself rather than across whatever is textured nearby. NOT
;;     because the sharp band is weak there: it is not — sharp-at saturates on a soft
;;     ramp (pinned in edge-band-tier-places-off-the-raw-edge-channel), so absence is
;;     not what made a denser :sharp tier measure worse;
;;   • it always takes a SIDE, including on a soft ramp where the liner tier gives up;
;;   • the side push is ≥ its own 2σ across, so it cannot straddle what it restates;
;;   • born at a fixed elongation, not the coherence-derived one, so the thinness is
;;     a property of the tier rather than of the local tensor.
;;
;; MEASURED (1024×1024 portrait, Size 6 / tiers 0.4 / Detail 1 / 600k / Stroke 1.5 /
;; Contrast 1 / Hardness 2.5), band tier off vs on, everything else identical:
;;   outward band bleed   +7.35 -> +5.13 luma   (-30%)
;;   whole-image delta    -0.56 -> -0.31        (inside the ±0.3 tolerance)
;;   coat-likeness         0.13 ->  0.10
;;   emitted splats     418,997 -> 534,594      (cap 786,432; budget 600,000)
;; By ring, outward from the silhouette: 1px -3%, 2px 28%, 3px 52%, 4px 49%, 5px 35%,
;; 6px 24%, then ~0 past 7px. The tier cuts the bleed roughly in half exactly where it
;; reaches, and the residual is dominated by two things it does NOT address: the first
;; pixel out (its push starts at ~1.2px, clear of its own 2σ) and the far field past
;; 7px, which is the BASE tier's own reach (σ 7.2 ⇒ a 25px quad), not a band artifact.
;; This is the first of thirteen attempts on this artifact to move the metric the RIGHT
;; way; the twelve rejected ones are tabulated in .dirge/spec-coat-edge-fuzz.md.
(def ^:private band-th 0.30)          ; edge strength a band seed needs (raw :edge, unnormalized)
(def ^:private band-se 2.6)           ; forced elongation: sx = s0·se, sy = s0/se (NOT coherence-derived)
(def ^:private band-sideo 1.4)        ; side push in units of ssz, jittered ×0.6–3.15 per seed (see soff)
(def ^:private band-ssz-max 1.6)      ; the band paints a LINE; above this it would read as a daub
(def ^:private band-lvl 7)            ; outside the ladder's 0..6, so poshash streams can never collide
(def ^:private band-share 0.25)       ; hard ceiling on the band tier's slice of the working budget
;; The Cut-in dial scales this tier's density. Measured sweep on the test portrait
;; (outward band delta): 0.0 +7.28, 0.5 +5.56, 1.0 +5.39, 2.0 +5.39. It SATURATES at
;; about 1.0 — doubling to 2.0 spends another ~220k candidates for nothing, because the
;; residual sits where band strokes do not reach at all (the first pixel out, and the
;; far field past 7px, which is the base tier's own 25px reach) rather than where they
;; reach too thinly. Do not chase this artifact with more band density; the slider is
;; capped at 1.5 for the same reason.
(def ^:private band-ovl 0.45)
;; ^ seed spacing coefficient, sp = band-ovl·√band-segs·ssz. Deliberately far tighter
;; than the ladder's 0.7–1.25: a band seed only survives on the ridge, which is a
;; ~2px-wide sliver of the image, so image-wide spacing that looks dense is sparse
;; ALONG a contour. Measured at 1.1 (sp 5.33px): the tier covered 18.2% of the outward
;; band and 33% of its worst 1–3px zone, and cut the bleed there by only 16–19% —
;; the mechanism worked wherever it landed and mostly did not land. At 0.45 the
;; nx-cap below binds instead, so band-share is what actually sets the density.
(def ^:private band-segs 12)
;; ^ the band tier's NOMINAL traced length, governing both its seed spacing and its
;; budget term (they have to agree — see expected-segs). It is NOT expected-segs: that
;; 4 was measured for :sharp-placed strokes, which stop as soon as the fine-band ridge
;; dies. A band stroke follows a SILHOUETTE, which is long and smooth by construction,
;; so it traces much further before the ridge dies or the tangent breaks. 12 is an
;; estimate, not a measurement — re-measure the mean traced length of this tier once
;; the render is visually right, the way expected-segs was swept.

(defn- band-level
  "The edge-band level map, or nil when the tier should not exist. `finest` is the
   finest admitted LADDER stdev — the band sits at or below it so it reads as a line
   drawn over the ladder's finest marks, never as a competing daub.

   `strength` is the Cut-in dial (tier-muls[3]): 0 turns the tier off entirely and
   otherwise it scales the tier's share of the budget, i.e. its DENSITY — which the
   ring measurements show is what governs how much of the band it actually covers,
   not its geometry. There is deliberately no Detail gate: Detail decides how many
   LADDER rungs get painted, and a control that silently did nothing below some other
   slider's threshold would be a worse dial than no dial. The one structural
   precondition stays — the ladder must itself be painting lines (finest below
   liner-threshold), because a band drawn at daub scale is not a band.

   Budget: candidates are gated on the EDGE-map fraction, so a tier that only ever
   paints a thin band cannot charge the budget as though it covered the image; the
   estimate is then capped at `band-share` of the working budget outright, and
   `:demand` is charged against the detail tier's slice by the caller so the band
   comes OUT of the budget rather than on top of it."
  [dmap strength area budget finest]
  (when (and (pos? (double strength))
             (< (double finest) liner-threshold))
    (let [frac (detail-fraction dmap :edge band-th)]
      (when (pos? frac)
        (let [ssz    (max 1.0 (min band-ssz-max (double finest)))
              ;; the dial scales DENSITY, so it divides the spacing by √strength
              ;; (candidates ∝ 1/sp²  ⇒  nx ∝ strength) and scales the budget cap by the
              ;; same factor. Scaling only the cap did nothing wherever the natural
              ;; spacing was the binding constraint, which at a roomy budget is always —
              ;; the dial measured identical at 0.5 and 1.0. At strength 1.0 both terms
              ;; are exactly the shipped values, so the default look is unchanged.
              sp     (/ (* band-ovl (Math/sqrt (double band-segs)) ssz)
                        (Math/sqrt (double strength)))
              nx-nat (Math/ceil (/ (double area) (* sp sp)))
              ;; splats ≈ nx · frac · band-segs, so invert that for the cap
              nx-cap (/ (* band-share (double strength) (double budget))
                        (* frac (double band-segs)))
              nx     (long (max 0.0 (min nx-nat nx-cap)))]
          (when (pos? nx)
            {:lvl band-lvl :ssz ssz :sp sp :th band-th
             :nx nx :ny 1 :offset 0
             :segs max-segs :stepf (step-frac band-lvl) :bendf 0.0
             :map-kind :edge :traw (raw-floor band-lvl ssz)
             :sideo band-sideo :selong band-se :band true
             :demand (* (double nx) frac (double band-segs))}))))))

(defn- band-prepend
  "Put the edge-band level at the FRONT of a finest-first level vector (index 0 =
   topmost, so it covers the coverage tiers' bleed), shifting every existing
   candidate offset past its own block. A standalone top-level fn rather than a
   let-wrap inside layer-params' level loop — demoting that loop out of tail
   position is what sends the jolt compiler pathological (see stub-glaze)."
  [levels band]
  (if (nil? band)
    levels
    (let [n (long (:nx band))]
      (into [band] (mapv (fn [l] (update l :offset + n)) levels)))))

(defn layer-params
  "Pure per-level placement parameters — THE SHARED SPEC for the CPU loop
   (layered-means) and the GPU generation pass, so both enumerate the same cells.
   `tier-muls` is [broad mid fine cut-in]; the 4th (the edge-band tier's strength)
   defaults to 1.0 when absent, so three-element callers predating that tier still work.
   `levels` is ordered FINEST-FIRST (index 0 = finest); a consumer that walks
   levels[0]→levels[n-1] emitting each cell gets paint order for free (small strokes
   at the front, over the large base — no sort). Each level is
   {:lvl :ssz :sp :th :nx :ny :offset}: ssz = stdev, sp = spacing, th = detail
   threshold (−1 keeps all, base), nx·ny = candidate grid, offset = cumulative
   candidate-cell start (finest-first). :total = Σ nx·ny (candidate count the GPU
   draws as GL_POINTS). :warp = flat-region Perlin warp gain, :scale = the uniform
   size-up that keeps the field under budget."
  [dmap detail size variation curvature stroke tier-muls count H W]
  (let [smax    (double size)
        slen    (stroke-len-frac stroke)
        budget  (min (double splat-budget) (max 500.0 (double count)))
        warp    (* 0.95 (double curvature))
        area    (double (* (long H) (long W)))
        nlev    (long (max 1 (min 7 (inc (Math/round (* (double detail) 6.0))))))
        thresh  (fn [lvl] (if (zero? (long lvl)) -1.0 (min 0.9 (* 0.26 (double lvl)))))
        ;; level size ladder: halves per level down to level 4, then decays gently
        ;; (×0.7 per level) with a ~pixel floor — the finest detail lands at a
        ;; couple-of-pixels footprint, never sub-pixel dust the AA clamp fades away.
        lsize   (fn [lvl] (let [l (double (long lvl))]
                            (max 0.7
                                 (if (<= l 4.0)
                                   (/ smax (Math/pow 2.0 l))
                                   (* (/ smax 16.0) (Math/pow 0.7 (- l 4.0)))))))
        ;; the BROAD tier (lvl≤1) is BOKEH-ADAPTIVE: the Broad slider must never
        ;; touch the detailed subjects, only loosen the flat regions. Its :ssz stays
        ;; the SUBJECT-nominal size (no broad multiplier); at emission each seed
        ;; grows ×m(x) = 1+(b−1)·(1−subjectness) and flat regions THIN candidates by
        ;; (bmin/m)² to keep overlap constant — few LARGE smooth daubs in bokeh.
        ;; The grid uses bmin = min(1,b): the densest spacing any region needs.
        bmul    (double (nth tier-muls 0))
        bmin    (min 1.0 bmul)
        smul    (fn [lvl] (if (<= (long lvl) 1) 1.0 (tier-mul tier-muls (max 0 (- (long lvl) 2)))))
        ;; final nominal size floored at ~a pixel AFTER the tier multiplier: a tier
        ;; dial at 0.4 must make its layer finer, never reduce it to sub-pixel dust —
        ;; dusted mid/fine layers punch the gradation ladder out of the painting and
        ;; leave raw base-to-line transitions (Mid/Fine turned left looked worse).
        nsize   (fn [lvl] (max 0.7 (* (smul lvl) (lsize lvl))))
        ;; LINER-SCALE classification (mirrors the per-chain liner? in stroke-segments:
        ;; the same 3.5px NOMINAL-size test with the lvl≥2 floor — lvl 0 is the base
        ;; fill, lvl 1 the broad coverage tier, neither is ever a liner). At small
        ;; Stroke/Size the mid levels (2-3) drop below the threshold and join the fine
        ;; liners (4+). Nominal (pre-budget-scale) so it is deterministic and identical
        ;; CPU/GPU (both receive these maps as uniforms).
        liner?  (fn [lvl] (and (>= (long lvl) 2) (< (nsize lvl) liner-threshold)))
        ;; LINER chains trace one long continuous line instead of the short stitched
        ;; dashes a small-σ grid of 3-8-segment chains lays down (the contour thatch):
        ;; ~28px span target at the default Stroke, scaled BY Stroke so the slider
        ;; extends the LINE not the gaps; clamped 8..32 (32 = the GS vertex cap).
        ;; NOMINAL-keyed, so this (and liner?/stepf-of/overlap below) now serves
        ;; only the BUDGET ESTIMATE (k-of) — the EMITTED per-level segs/stepf/sp
        ;; are derived from the PHYSICAL (budget-scaled) stdev in the levels loop.
        segs-of (fn [lvl] (if (liner? lvl)
                            (let [span (* 28.0 slen)
                                  step (* (step-frac lvl) (nsize lvl))]
                              (long (max 8 (min 32 (Math/round (/ span step))))))
                            (seg-count lvl)))
        ;; stepf carries Stroke folded in: liner levels get no Stroke factor (the
        ;; slider acts through segs/span instead), broad/mid carry slen. Both trace
        ;; loops consume the emitted stepf as-is — no in-loop stroke-length
        ;; multiplication. (Budget-estimate variant; the emitted stepf uses the
        ;; physical liner test in the levels loop.)
        stepf-of (fn [lvl] (* (step-frac lvl) (if (liner? lvl) 1.0 slen)))
        ;; base layer overlaps heavily (~0.65×stdev ⇒ full coverage); finer layers are
        ;; sparser accents (the base fills behind them). Overlap is FIXED, so coverage
        ;; never depends on the budget. Fine seeds are √segs sparser than dabs — each
        ;; seed traces a segs-segment stroke that COVERS the along-edge span, so the
        ;; total segment count (and thus the budget scale — the stroke WIDTH) stays what
        ;; single dabs cost: the ×segs term and the √segs spacing cancel
        ;; (k = segs·f·area/sp², sp² ∝ segs ⇒ k ∝ f/size², segs-invariant).
        ;; LINER levels trace LONG continuous chains (segs-of, up to 32) instead of the
        ;; short stitched dashes that thatch small-σ contours; their spacing keeps the
        ;; level's OWN tier coefficient (1.25 mid / 0.7 fine) scaled by √segs with NO
        ;; slen factor (Stroke is folded into segs/span). Keeping the tier coefficient —
        ;; not forcing 0.7 onto reclassified mid levels — is what holds the invariant:
        ;; forcing 0.7 raised a mid level's k ≈(1.25/0.7)²≈3.2× and inflated the shared
        ;; scale, which damped the Broad slider's bokeh growth below its test ratio.
        ;; Non-liner (broad/mid coverage) stays √(seg-count·slen): short strokes (low
        ;; Stroke) pack denser or they pearl-string; at default Stroke (slen 1) = 1.
        ;; base overlap 0.65 (was 0.72): hash-random placement has gap variance a
        ;; lattice doesn't; slightly tighter spacing keeps coverage airtight.
         ;; liner spacing keys off the NOMINAL chain length (expected-segs) — the SAME
         ;; value k-of counts segments with — so the budget estimate and the real seed
         ;; density stay consistent (k = expected-segs·f·area/sp², sp² ∝ expected-segs).
         ;; Seeding for the full max-segs cap left the fine tier ~2.6σ apart and
         ;; under-populated; ~10-seg spacing packs survivors to ~2.2σ and the render
         ;; fills in. Chains that trace longer simply overlap more (the handoff
         ;; continuity we want); the extra k flows into scale-f like any other demand.
        overlap (fn [lvl] (let [l (long lvl)
                                 cnt (if (liner? lvl) (min expected-segs 14) (seg-count lvl))
                                 stf (if (liner? lvl) 1.0 slen)]
                            (cond (zero? l) 0.65
                                  (<= l 3)  (* 1.25 (Math/sqrt (* (double cnt) stf)))
                                  :else     (* 0.7  (Math/sqrt (* (double cnt) stf))))))
        ;; tier multipliers scale size AND spacing together (constant overlap), so
        ;; each tier's density rebalances through the budget automatically.
        sp-of   (fn [lvl scale] (* (overlap lvl) scale
                                   (if (<= (long lvl) 1) (* bmin (lsize lvl)) (nsize lvl))))
        ;; budget: total(scale)=K/scale² ⇒ smallest scale≥1 that fits under the working
        ;; budget. Fine-level seeds emit segs(lvl) SEGMENTS each (a traced brush stroke),
        ;; so their term is multiplied accordingly — the budget counts splats, not seeds.
        ;; Each fine level estimates its survivor fraction on ITS OWN map (aggregate vs
        ;; sharp fine-band — the same map it thresholds against when placing).
        lvl-frac (fn [lvl] (detail-fraction dmap (level-map-kind lvl (nsize lvl)) (thresh lvl)))  ; nominal size here — the whole budget pass is nominal-keyed
        ;; SUBDIVISION within the broad/mid tiers only: levels 1-2 hand cells off to
        ;; the next-finer level (exclusive fractions). From level 3 up the finer
        ;; levels OVERLAP instead of claiming — mid keeps painting under the fine
        ;; glazes, so the mid→fine transition is a mixed gradient, not a seam — and
        ;; each overlapping level pays its FULL fraction in the budget.
        ;; TWO-TIER budget: the levels that exist at moderate Detail (0-3) get the
        ;; scale THEY need; the added ultra-fine levels (4-5) fit their own scale into
        ;; the REMAINING budget. A single uniform scale let the finest levels' huge
        ;; demand fatten the base ×2.6+ as Detail rose — maxing Detail made everything
        ;; coarser. Now raising Detail leaves the broad/mid painting untouched and
        ;; adds fine accents on top: monotone by construction.
        ;; the broad-tier grid runs at bmin spacing, so its budget terms carry bmin²
        ;; (and the base integrates the bokeh thinning via E[1/m²]).
        einv (if (== bmul 1.0) 1.0 (mean-inv-m2 dmap bmul))
        k-of (fn [lvl]
               (let [f (cond
                         (zero? (long lvl))        (* einv bmin bmin)
                         (== (long lvl) 1)         (* bmin bmin
                                                      (if (< 1 (dec nlev))
                                                        (max 0.0 (- (lvl-frac 1) (lvl-frac 2)))
                                                        (lvl-frac 1)))
                         (and (<= (long lvl) 2) (< (long lvl) (dec nlev)))
                                                   (max 0.0 (- (lvl-frac lvl) (lvl-frac (inc (long lvl)))))
                         :else                     (lvl-frac lvl))
                    sp (sp-of lvl 1.0)]
                ;; budget on the NOMINAL expected traced length (expected-segs), NOT the
                ;; max-segs cap — actual length is data-dependent (geometry decides), so
                ;; the estimate is approximate by construction. segs=32 here would blow it up.
                (/ (* (double expected-segs) f area) (* sp sp))))
        Kc (reduce + 0.0 (map k-of (range 0 (min nlev 4))))
        Kf (if (> nlev 4) (reduce + 0.0 (map k-of (range 4 nlev))) 0.0)
        scale-c (max 1.0 (Math/sqrt (/ Kc budget)))
        ;; fine-tier remaining budget (the slice left after the broad/mid tier 0–3 has
        ;; taken its scale-c share). Reused by scale-f and the admission gate.
        fine-rem (if (> nlev 4)
                   (max (* 0.15 (double budget)) (- (double budget) (/ Kc (* scale-c scale-c))))
                   0.0)
        scale-f (if (<= nlev 4)
                  scale-c
                  (max scale-c (Math/sqrt (/ Kf fine-rem))))
        scale-of (fn [lvl] (if (<= (long lvl) 3) scale-c scale-f))
        scale scale-c
        ;; build FINEST level first (lvl nlev-1 → 0), assigning cumulative candidate offsets
        ;; in that same order, so GPU gl_VertexID order == CPU emission order == paint order.
        ;; Each level carries its SCALE-RELATIVE stroke behaviour: segment count, step
        ;; length, curvature share, and which detail map it reads.
        ;; HASH-RANDOM placement: candidates are hashed positions, not a grid. Any
        ;; lattice — axis-aligned OR rotated — keeps a spectral peak at its row
        ;; frequency that reads as parallel stripes across smooth regions (worst
        ;; with Variation at 0, when no size/tone diversity masks it). White-noise
        ;; positions have no periodicity to show. nx = candidate count, ny = 1.
        ;; FINAL per-level stroke values are PHYSICAL: liner-ness, chain span and
        ;; spacing are decided from the BUDGET-SCALED stdev (ssz = lsc·nsize) with
        ;; the exact per-chain discipline test the trace loops apply, so a level
        ;; can never get liner-length chains without liner discipline. Nominal-keyed
        ;; classification did exactly that at low budgets (scale>1): a level with
        ;; nominal 2.1 / actual 4.4 traced 13-segment chains with no momentum or
        ;; line-hold, and the ~28px span target — computed in nominal px but traced
        ;; at actual σ — doubled to 51px: the fat wavy S-worms along every soft
        ;; contour. The span is now ABSOLUTE pixels, ramped by thinness: only
        ;; genuinely thin strokes (σ≤1.4) read as drawn lines at the full ~28px
        ;; span; by σ2.6 the level is back on the short seg-count table (fat
        ;; accents are dabs, not liners). The budget pass above keeps the NOMINAL
        ;; helpers (segs-of/stepf-of/overlap/sp-of): k = segs·f·area/sp² ∝ f is
        ;; segs-invariant below the 14-seg spacing cap, so the estimate is robust
        ;; to the final segs differing — shorter final chains spend less than
        ;; estimated, which errs conservative.
        ;; PHYSICAL per-level stroke spec for a given FINAL (post-clamp) stdev. Factored
        ;; out so the coarse→fine admission pass and the finest-first output pass derive
        ;; identical segs/stepf/ovl/sp from the same ssz. ldisc? is the PHYSICAL liner
        ;; predicate (liner-scale?): chain length keys on stdev, not the level index.
        phys-spec (fn [lvl ssz]
                    (if (dab-level? lvl ssz)
                      ;; DAB TIER. Below dab-max the orientation field has no
                      ;; information at the stroke's own scale, so a traced chain
                      ;; wanders (waviness) and its colour-drift guard lifts it after
                      ;; 2-4 segments (short disjointed dashes). A single oriented dab
                      ;; has no path: it cannot wander and cannot dry out.
                      ;; Spacing is dab-overlap·σ, NOT the chain form. The chain
                      ;; spacing carries a √segs factor that, applied to a 1-segment
                      ;; mark, packs dabs ~1σ apart — ~12 gaussians overlapping every
                      ;; point, which averages them into mush (measured: that is why
                      ;; the first dab experiment went soft). At 2σ spacing roughly
                      ;; three dabs meet at a point, so each mark still reads as its
                      ;; own stroke while coverage stays continuous.
                      {:segs 1 :stepf 0.0 :sp (* dab-overlap ssz)}
                      (let [ldisc? (liner-scale? lvl ssz)
                          detail? (>= (long lvl) 2)
                          ;; FEATURE TRACER (supersedes Round 5a): detail tiers run to
                          ;; max-segs and let the GEOMETRY (ridge dies / tangent bends)
                          ;; decide where to stop — never a fixed span, never a count.
                          ;; Broad/coverage tiers (0-1) keep the short coverage-stroke
                          ;; table. Actual traced length is data-dependent; the budget
                          ;; estimate (k-of) uses expected-segs, not this cap. SEED
                          ;; SPACING keys off expected-segs too (not max-segs): seeding
                          ;; for the max-segs cap left the fine tier ~2.6sigma apart and
                          ;; under-populated; ~10-seg spacing packs survivors to ~2.2sigma.
                          segs (if detail? max-segs (seg-count lvl))
                          stepf (* (step-frac lvl) (if ldisc? 1.0 slen))
                          ovl (let [cnt (if ldisc? (min (if detail? expected-segs segs) 14)
                                              (seg-count lvl))
                                    stf (if ldisc? 1.0 slen)]
                                (cond (zero? (long lvl)) 0.65
                                      (<= (long lvl) 3) (* 1.25 (Math/sqrt (* (double cnt) stf)))
                                      :else             (* 0.7  (Math/sqrt (* (double cnt) stf)))))
                          sp (if (<= (long lvl) 1)
                               (* ovl (scale-of lvl) bmin (lsize lvl))
                               (* ovl ssz))]
                        {:segs segs :stepf stepf :sp sp})))
        ;; ADMISSION coarse→fine: keep a level only if it is meaningfully finer than the
        ;; previous (keep-ratio) AND the budget affords its clamped survivor demand. The
        ;; physical ssz is clamped to step-ratio×prev so the ladder is strictly monotone,
        ;; floored at min-phys (never sub-pixel dust). Dropping a level drops every finer
        ;; one too — a monotone ladder the budget cannot reach simply ends earlier.
        ;; MIN PAINTABLE STROKE. A stroke below ~1.4px stdev does not read as a mark,
        ;; it reads as a scratch: the render shader already eases hardness back to a
        ;; pure gaussian below 2.5px because anything thinner aliases. At Size 6 with
        ;; the tier dials at 0.4 the nominal ladder asks for 0.6px and 0.3px strokes;
        ;; floored at 0.6 those became two hairline levels covering just 6% and 14% of
        ;; the image — sparse, aligned, alpha ~0.6 marks that scratch the underpainting
        ;; instead of building a surface (the contour thatch). Clamping UP to the
        ;; minimum paintable size instead keeps the detail and loses the thatch: sp is
        ;; derived from the clamped ssz (phys-spec below), so seed count falls as
        ;; (ssz/min-phys)² and the ink is preserved at a paintable scale. This is a
        ;; no-op wherever the ladder already lands above 1.4px (any default-ish Size).
        ;; Defect A: lowered 2.2 -> 1.4 so the Mid/Fine dials can MOVE the detail level
        ;; (at 2.2 it pinned at exactly 2.20 and both dials were dead). The redundancy
        ;; drop above still prevents two levels piling here (the original thatch), so a
        ;; single 1.4px detail level reads as a soft glaze, not a scratch-field.
        min-phys   1.4
        step-ratio 0.7
        keep-ratio 0.95
        ssz0  (* (scale-of 0) (nsize 0))
        ;; broad/mid tier 0..broad-end-1 is ALWAYS admitted — scale-c sized it to fit and
        ;; the two-tier budget guarantees raising Detail never coarsens those levels. Only
        ;; the fine tier (4+) is gated: keep it iff it is meaningfully finer than the
        ;; previous (keep-ratio) AND its clamped survivor demand fits the fine remaining
        ;; budget. Dropping a fine level drops every finer one too.
        broad-end (long (min nlev 4))
        ;; admitted: coarse→fine vector of [lvl ssz] (ssz already clamped to the monotone
        ;; ladder + min-phys floor during admission).
        admitted (loop [lvl 1 prev (double ssz0) rem (double fine-rem)
                        acc [[(long 0) ssz0]]]
                   (if (>= lvl nlev)
                     acc
                     (let [raw (* (scale-of lvl) (nsize lvl))
                           ssz (max min-phys (min raw (* step-ratio prev)))
                           {:keys [segs sp]} (phys-spec lvl ssz)
                           cost (/ (* (double segs) (lvl-frac lvl) area) (* sp sp))
                           fine? (>= lvl broad-end)]
                       ;; REDUNDANCY drop applies at every level: once min-phys clamps
                       ;; two levels to the same physical size (Size 6 + tiers 0.4 puts
                       ;; both 2 and 3 on the floor) the finer one is a duplicate pass,
                       ;; not extra detail. Dropping it cannot coarsen anything — it is
                       ;; the same size as its parent. The BUDGET drop stays fine-tier
                       ;; only, so raising Detail still never coarsens levels 0-3.
                       (if (or (>= ssz (* keep-ratio prev))
                               (and fine? (> cost rem)))
                         acc                                  ; drop this + every finer
                         (recur (inc lvl) ssz (if fine? (- rem cost) rem)
                                (conj acc [(long lvl) ssz]))))))
        ;; the EDGE-BAND overlay, sized off the finished ladder's finest rung. Computed
        ;; HERE (not at the end) so its demand can be charged against the detail tier's
        ;; slice below — the band comes out of the budget, not on top of it.
        ;; tier-muls[3] is the Cut-in dial. Read with a default so the three-element
        ;; [broad mid fine] callers that predate the tier keep working unchanged.
        band (band-level dmap (nth tier-muls 3 1.0) area budget
                         (reduce min (map (fn [[_ ssz]] (double ssz)) admitted)))
        ;; build FINEST-FIRST with cumulative candidate offsets, so GPU gl_VertexID
        ;; order == CPU emission order == paint order. :lvl stays the ORIGINAL index so
        ;; both paths' tiering (liner?, map-kind, raw-floor, …) still keys on it; :nlev
        ;; becomes the ADMITTED count (the GS decodes exactly that many slots).
        ;; BUDGET CAP on candidate density: the min-phys floor pins the finest levels
        ;; at a FIXED px spacing that does NOT scale with the budget, so at low counts
        ;; their candidate pool (area/sp^2) swamps the target and the Splats slider
        ;; stops biting (count 1000 emitted ~6330). Cap the DETAIL tiers' (lvl>=2)
        ;; candidate count so their candidate pool fits the budget left after the
        ;; coverage tiers (0-1). Coverage levels keep full density so the base never gaps;
        ;; hashed positions mean a thinned nx is still white noise (a prefix of hashed
        ;; indices), so no lattice returns. expected-segs still drives BOTH spacing (ovl)
        ;; and the k-of budget term -- this cap is layered on top, it does not change either.
        ;;
        ;; DEMAND counts survivors x emitted-segments for COVERAGE but the FULL candidate
        ;; pool x emitted-segments for DETAIL. Two earlier attempts got this wrong:
        ;;   (1) charging the base for CANDIDATES over-stated it ~6x at Broad<1 (bmin<1
        ;;       densifies the grid but thins survivors by bmin^2), which starved the
        ;;       detail tier's share 22x at 600k and erased the strap/glasses detail; so
        ;;       coverage demand uses the bmin^2-thinned surv-frac.
        ;;   (2) charging detail for lvl-frac SURVIVORS under-stated it ~20x at low budgets
        ;;       (lvl-frac is a point estimate at the FULL threshold; the dithered
        ;;       threshold drops to 0.75x and bilinear sampling lets far more candidates
        ;;       through), so the cap never fired and count=1000 still emitted 6330; so
        ;;       detail demand uses the full candidate pool -- the honest upper bound on a
        ;;       textured region where survival is high. This over-predicts detail at low
        ;;       budgets (fires the cap, which is the point) and is accurate at high ones.
        surv-frac (fn [lvl]
                    (cond (zero? (long lvl)) (* einv bmin bmin)
                          (== (long lvl) 1)  (* bmin bmin
                                               (if (< 1 (dec nlev))
                                                 (max 0.0 (- (lvl-frac 1) (lvl-frac 2)))
                                                 (lvl-frac 1)))
                          (and (<= (long lvl) 2) (< (long lvl) (dec nlev)))
                                             (max 0.0 (- (lvl-frac lvl) (lvl-frac (inc (long lvl)))))
                          :else              (lvl-frac lvl)))
        demand    (fn [lvl ssz]
                    (let [{:keys [sp]} (phys-spec lvl ssz)
                          nx (Math/ceil (/ area (* sp sp)))
                          detail? (>= (long lvl) 2)
                          ;; COVERAGE: bokeh-thinned survivors (bmin^2). DETAIL: full
                          ;; candidate pool (lvl-frac under-predicts dithered survival).
                          surv (if detail? 1.0 (surv-frac lvl))
                          segs-per-seed (if detail? (double expected-segs)
                                                 (double (seg-count lvl)))]
                      (* (double nx) (double surv) segs-per-seed)))
        cov-demand (reduce + 0.0 (map (fn [[lvl ssz]] (if (< (long lvl) 2) (demand lvl ssz) 0.0)) admitted))
        det-demand (reduce + 0.0 (map (fn [[lvl ssz]] (if (>= (long lvl) 2) (demand lvl ssz) 0.0)) admitted))
        ;; the band overlay is charged here alongside coverage, so admitting it thins
        ;; the detail tier instead of pushing the field past the budget.
        det-budget (max (* 0.1 (double budget))
                        (- (double budget) cov-demand (double (or (:demand band) 0.0))))
        cand-thin  (if (> det-demand det-budget) (/ det-budget det-demand) 1.0)
        levels (loop [rem (rseq admitted) off 0 out []]
                 (if (empty? rem)
                   out
                   (let [[lvl ssz] (first rem)
                         lvl (long lvl)
                         {:keys [segs stepf sp]} (phys-spec lvl ssz)
                         raw-nx (long (Math/ceil (/ area (* sp sp))))
                         nx (long (if (>= lvl 2) (Math/ceil (* (double raw-nx) (double cand-thin))) raw-nx))]
                     (recur (rest rem) (+ off nx)
                            (conj out {:lvl lvl :ssz ssz :sp sp :th (thresh lvl)
                                       :nx nx :ny 1 :offset off
                                       :segs segs :stepf stepf
                                       :bendf (bend-frac lvl) :map-kind (level-map-kind lvl ssz)
                                       :traw (raw-floor lvl ssz)
                                       ;; ladder levels keep the liner side push and take
                                       ;; their elongation from local coherence (selong 0)
                                       :sideo 0.55 :selong 0.0 :band false})))))
        ;; the EDGE-BAND tier rides at index 0 (topmost) over the finished ladder.
        levels (band-prepend levels band)]
    {:nlev (clojure.core/count levels) :warp warp :scale scale :levels levels
     :total (reduce + 0 (map (fn [{:keys [nx ny]}] (* nx ny)) levels))}))

(declare sample-fields)

(defn- edge-snap
  "Move a fine-stroke position onto the local EDGE RIDGE: sample edge strength at
   p and at ±h across the local tangent, fit a parabola, step to its peak (clamped
   to ±h). Seeds scatter across a thin line by the placement noise; without the
   snap each stroke traces PARALLEL to the line at its own offset and a crisp
   1px line renders as a wobbly multi-strand braid.
   `gain` damps the corrector: the edge map is texel-quantized, so the full
   parabola step jitters. The SEED snap uses 0.65 (one-shot converge onto the
   ridge); liner-chain STEPS use 0.85 — a strong per-step corrector keeps a
   traced contour glued to its ridge instead of drifting parallel into a braid."
  [dmap nf x y h hd wd gain]
  (let [[th _] (sample-fields nf x y)
        nx (- (Math/sin th)) ny (Math/cos th)
        e0 (wavelet/edge-at dmap x y)
        ep (wavelet/edge-at dmap (+ x (* nx h)) (+ y (* ny h)))
        em (wavelet/edge-at dmap (- x (* nx h)) (- y (* ny h)))]
    (if (< (max e0 ep em) 0.12)
      [x y]
      (let [den (- (+ em ep) (* 2.0 e0))
            d   (if (< (Math/abs den) 1e-9)
                  0.0
                  (max -1.0 (min 1.0 (/ (- em ep) (* 2.0 den)))))]
        [(max 0.0 (min hd (+ x (* nx h d gain))))
         (max 0.0 (min wd (+ y (* ny h d gain))))]))))

(defn- map-at
  "Sample the placement map matched to a level's scale."
  [dmap kind x y]
  (cond
    (= kind :sharp) (wavelet/sharp-at dmap x y)
    (= kind :mid)   (wavelet/mid-at dmap x y)
    ;; the EDGE-BAND tier places off raw edge strength — UNNORMALIZED, unlike the
    ;; three wavelet bands (edge-at does not divide by dmax), so the GLSL twin must
    ;; route sel 3 through edgeAt rather than through mapAt's /u_dmax tail.
    (= kind :edge)  (wavelet/edge-at dmap x y)
    :else           (wavelet/detail-at dmap x y)))

(defn- edge-near
  "MAX edge strength over the centre + 4 diagonal taps at radius d. A stroke of
   size ~d must answer for edges anywhere under its BODY, not just at its centre:
   centre-sampled Ev let daubs and chains seeded just off a silhouette escape the
   edge-band shrink/suppression and ribbon their soft mixed-colour bodies along
   it — the ghost veil around heads and shoulders."
  [dmap x y d]
  (max (wavelet/edge-at dmap x y)
       (wavelet/edge-at dmap (+ x d) (+ y d))
       (wavelet/edge-at dmap (- x d) (- y d))
       (wavelet/edge-at dmap (+ x d) (- y d))
       (wavelet/edge-at dmap (- x d) (+ y d))))

(defn- subject-at
  "Wavelet SUBJECTNESS at (x,y): how much detail density surrounds this spot.
   The smoothed aggregate detail (centre + 4 diagonal taps at radius r) finds the
   detailed subjects of the image; the raw centre term keeps thin isolated
   features (a wire against bokeh) alive. 0 = flat bokeh, 1 = detailed subject.
   Drives the bokeh-adaptive broad tier AND the mid/fine placement gate: splat
   size follows the wavelet's detail density — low detail = few big smooth
   daubs, high detail = many small precise strokes."
  [dmap x y r]
  (let [p0 (wavelet/detail-at dmap x y)
        ps (* 0.2 (+ p0
                     (wavelet/detail-at dmap (+ x r) (+ y r))
                     (wavelet/detail-at dmap (- x r) (- y r))
                     (wavelet/detail-at dmap (+ x r) (- y r))
                     (wavelet/detail-at dmap (- x r) (+ y r))))
        s  (min 1.0 (max 0.0 (/ (- ps 0.05) 0.30)))]
    (max s (min 1.0 (max 0.0 (/ (- p0 0.10) 0.35))))))


(defn- stroke-segments
  "Emit one seed's splat segments — THE SHARED BRUSH-STROKE SPEC the GPU generation
   shader mirrors. A base seed (lvl 0) is a single full-alpha fill splat. A fine seed
   TRACES A BRUSH STROKE: `segs` segments stepped along the orientation field
   (the edge tangent — structure/tensor-eigen's minor eigenvector), each step
   direction kept sign-continuous (the field is undirected) and bent by smooth Perlin
   noise scaled by `curvature`, with size AND alpha tapering toward the tail — a brush
   line that fades out at the end, layered over the underpainting. `dirsign` picks
   which way along the tangent the stroke pulls (per-seed hash, so strokes alternate).
   `hb` (1.0 for base + the broadest detail level) selects the HEAVY blur as the
   stroke's smooth colour source — broad strokes carry colour smoothed at their own
   scale, so smooth gradients (bokeh, sky) reproduce without stroke banding.
   Every segment carries the chain HEAD's position (hx,hy) as its colour-sample
   point: ONE STROKE = ONE BRUSH-LOAD OF PAINT (per-segment colour sampling made
   edge strokes alternate the two sides' colours as centres jittered across the
   contour — a bright/dark bead necklace along every silhouette).
   Returns [[x y size D sn tn alpha theta coherence hb hx hy]…]."
  [nf dmap lvl x y ssz D sn tn dirsign curvature stroke hd wd segs stepf bendf hb traw sgate blur-px iw ih lth melt mkind gainv blurd-px bph sideo selong]
  (if (zero? (long lvl))
    (let [[th coh] (sample-fields nf x y)]
      ;; melted bokeh daubs ROUND OFF (coherence → 0 kills the elongation and pulls
      ;; the colour toward the smooth blur): an elongated needle on a soft gradient
      ;; always reads as a directional streak, however faithful its colour.
      ;; base tier returns the SAME [rows reason] shape as the traced branch — a bare
      ;; row vector here made the call site read one row as the whole row list.
      [[[x y ssz D sn tn 1.0 th (* coh (- 1.0 (double melt))) hb x y traw (spec-cap lvl ssz) 0.0]] :base])
    (let [          ;; the BAND tier is opaque by ROLE, like the coverage tiers: it exists
                    ;; to COVER the outward bleed, and a glaze alpha there just averages
                    ;; the bleed back in at the exact spot it is meant to replace.
                    lal  (if (pos? (double selong)) 1.0 (level-alpha lvl ssz))
          ;; fine strokes snap onto the edge ridge at the seed and after every step
          ;; (predictor: tangent step; corrector: ridge snap) — the stroke GLUES to
          ;; the line it is painting instead of braiding beside it.
          snap? (>= (long lvl) 2)
          ;; LINER discipline keys on the PHYSICAL stroke size, not the level index:
          ;; which level paints "fine detail" depends on the sliders (Mid at 0.6
          ;; makes levels 2-3 smaller than the lvl-4 liners at Fine 1.0). Any small
          ;; accent chain must follow the original detail exactly — momentum,
          ;; gentle ridge snap, line-hold, no Perlin — or it reads as waviness.
          liner? (liner-scale? lvl ssz)
          ;; the EDGE-BAND tier (selong > 0: born elongated rather than taking its
          ;; elongation from the local tensor). It restates ONE side of a boundary,
          ;; so unlike an ordinary liner it must take a side even on a soft ramp.
          band? (pos? (double selong))
          ;; NO tensor-coherence gate on chain length. It was tried (ramp segs 20%→100%
          ;; over coherence 0.30→0.60, on the theory that a "line" only exists where the
          ;; tensor is coherent) and it does not work, because coherence does not
          ;; discriminate: a smooth gradient is rank-1, i.e. perfectly ORIENTED, so bokeh
          ;; scores nearly as coherent as a hard contour. Measured over this repo's test
          ;; portrait — strong edge (edge-at>0.50) median coherence 0.95, FLAT (<0.08)
          ;; median 0.72, only 13.5% of flat points below 0.30. The gate was a no-op
          ;; where it was meant to act: removing it moved the render by 0.05/255.
          ;; A chain-length gate needs a signal that separates the two populations;
          ;; LINE-HOLD (the level's own placement map) already does that job per step.
          kmax (dec (long segs))
          ;; liner chains correct AGGRESSIVELY mid-stroke: the in-trace ridge corrector
          ;; runs at 0.85 (was 0.35) to keep a traced contour glued to its ridge instead
          ;; of drifting parallel into a braid. The SEED snap stays 0.65 (one-shot).
          sgain (if liner? 0.85 0.65)
          ;; the GEOMETRY snaps to the ridge, but the COLOUR samples the pre-snap
          ;; position: on-ridge colour is the two sides' mix — darker than either —
          ;; and painted along a silhouette it reads as a drawn OUTLINE. Pre-snap
          ;; seeds land on one side or the other, so contour strokes interleave the
          ;; two sides' actual colours and the edge blends like meeting paint.
          cx0 x cy0 y
          ;; CRISPNESS = SHARPNESS, not contrast (mirror gen). Probe ALL THREE rungs of
          ;; the ridge (no early-out) and keep the fraction of the total transition that
          ;; completes within h1: crisp? = d1 >= 0.75*dmax. A hard step / 1px-AA edge
          ;; reads d1~=d3 (ratio ~1 -> CRISP, two opaque paints meet, impasto kept); a
          ;; wide ramp reads d1<<d3 (ratio ~0.35 -> SOFT RAMP, no meeting line -> paint
          ;; the LOCAL colour, damp the body). Contrast-invariant, so an 8px and a 24px
          ;; ramp both classify soft — the fixture stays at the spec's 8px. dmax<0.15 ->
          ;; a thin LINE feature / flat, keep the on-ridge colour. Hoisted ahead of
          ;; edge-snap + the geometric side offset: it keys both.
          [nx0 ny0] (let [[t0 _] (sample-fields nf cx0 cy0)]
                      [(- (Math/sin t0)) (Math/cos t0)])
          h1 (max 1.75 (* 0.8 ssz))
          h2 (max 3.0 (* 1.5 ssz))
          h3 (max 5.0 (* 2.5 ssz))
          rung (fn [hh]
                 (let [[rp gp bp] (sample-arr blur-px iw ih (+ cx0 (* nx0 hh)) (+ cy0 (* ny0 hh)))
                       [rm gm bm] (sample-arr blur-px iw ih (- cx0 (* nx0 hh)) (- cy0 (* ny0 hh)))
                       d (max (Math/abs (- rp rm)) (Math/abs (- gp gm)) (Math/abs (- bp bm)))]
                   {:hh hh :rp rp :gp gp :bp bp :rm rm :gm gm :bm bm :d d}))
          r1 (rung h1)
          r2 (rung h2)
          r3 (rung h3)
          d1 (:d r1) d2 (:d r2) d3 (:d r3)
          dmax (max d1 d2 d3)
          crisp? (>= d1 (* 0.75 dmax))
          soft-ramp? (and (not crisp?) (>= dmax 0.15))
          dsides dmax
          disp h1
          rp (:rp r1) gp (:gp r1) bp (:bp r1)
          rm (:rm r1) gm (:gm r1) bm (:bm r1)
          [x y] (if snap? (edge-snap dmap nf x y 1.75 hd wd 0.65) [x y])
          ;; IMPASTO meeting line: bodied liner strokes keep to THEIR side of the ridge.
          ;; Suppressed on a SOFT RAMP (side=0, no geometric backoff) — a gradient has no
          ;; meeting line to draw; thin-line/crisp edges keep today's displacement sign.
          ;; the BAND tier always takes a side — a soft silhouette is exactly where it
          ;; exists to work, and a band stroke with no side sits on the ridge and paints
          ;; the two sides' mix outward, which is the artifact itself. A seed that lands
          ;; exactly on the ridge (d≈0) falls back to its own direction hash, so the two
          ;; sides get restated in roughly equal numbers with no side detection at all.
          side (if (and snap? liner? (or band? (not soft-ramp?)))
                 (let [[th0 _] (sample-fields nf x y)
                       snx (- (Math/sin th0)) sny (Math/cos th0)
                       d   (+ (* (- cx0 x) snx) (* (- cy0 y) sny))]
                   (cond (> d 1e-9) 1.0
                         (< d -1e-9) -1.0
                         :else (if band? (double dirsign) 0.0)))
                 0.0)
          ;; how far off the ridge the stroke sits, in units of its own stdev. A liner
          ;; nudges 0.55σ (enough to keep to its side of a meeting line); the BAND is
          ;; pushed clear — at selong 2.6 its across-body σ is ssz/2.6, so even the
          ;; smallest push here (0.6·1.4 = 0.84σ_ssz ≈ 2.2σ across) keeps the stroke off
          ;; the ridge it restates. The push is jittered by the seed's bend hash (bendf
          ;; is 0 for the band, so bph is otherwise unused) so band strokes TILE the band
          ;; instead of stacking into one hard line — SQUARED, so the distribution
          ;; crowds the near zone: the measured outward excess falls off steeply with
          ;; distance (+19.8 luma at 1px, +2.8 at 10px), so uniform spreading spends most
          ;; of the tier's paint where there is least to cover.
          soff (if band?
                 (* (double sideo) (+ 0.6 (* 2.55 (double bph) (double bph))))
                 (double sideo))
          offset (fn [ox oy]
                   (if (zero? (double side))
                     [ox oy]
                     (let [[th0 _] (sample-fields nf ox oy)
                           snx (- (Math/sin th0)) sny (Math/cos th0)]
                       [(max 0.0 (min hd (+ ox (* (double side) soff ssz snx))))
                        (max 0.0 (min wd (+ oy (* (double side) soff ssz sny))))])))
          [x y] (offset x y)
          ;; the side sign relative to the stroke's MOTION frame: at the head the
          ;; motion perpendicular is dirsign·(field normal), so side·dirsign along
          ;; (−dy,dx) reproduces the head offset — and stays consistent through
          ;; field sign flips that would wobble a per-step θ resample.
          sidem (* (double side) (double dirsign))
          ;; BOUNDARY-SIDE BRUSH-LOAD: a chain running PARALLEL to a colour boundary
          ;; carries one brush-load for its whole span. Keyed on the crispness ladder:
          ;;  • no rung clears (dsides<0.15) → thin LINE feature, keep the on-ridge colour;
          ;;  • SOFT RAMP (h2/h3 only) → paint the LOCAL colour (bax=bay=0); a gradient
          ;;    has no uniform side colour near the ridge, so any off-ridge brush-load
          ;;    samples a colour absent where the stroke is painted (the dark rim on
          ;;    out-of-focus fingers). The body is damped separately.
          ;;  • a crisp GEOMETRIC side wins — the body was already offset toward that
          ;;    side, so the brush-load samples the SAME side and the meeting line is
          ;;    drawn by stroke geometry, not per-seed colour luck.
          ;;  • a crisp edge with no geometric side → the colour test, only at a genuine
          ;;    STEP edge (min dp dm < 0.3·dsides); else the on-ridge colour.
          [bax bay] (cond
                      (< dsides 0.15) [0.0 0.0]
                      soft-ramp? [0.0 0.0]
                      (not (zero? (double side)))
                      [(* (double side) disp nx0) (* (double side) disp ny0)]
                      :else (let [[r0 g0 b0] (sample-arr blur-px iw ih cx0 cy0)
                                  dp (max (Math/abs (- rp r0)) (Math/abs (- gp g0)) (Math/abs (- bp b0)))
                                  dm (max (Math/abs (- rm r0)) (Math/abs (- gm g0)) (Math/abs (- bm b0)))]
                              (if (< (min dp dm) (* 0.3 dsides))
                                (let [sidec (if (< dp dm) 1.0 -1.0)]
                                  [(* sidec disp nx0) (* sidec disp ny0)])
                                [0.0 0.0])))
          ;; the drift reference + probes read the FORGIVING box field (blurd-px):
          ;; on the razor-sharp bilateral paint field any probe wobble across a
          ;; boundary trips the lift instantly and dashes contour chains into beads
          ;; the drift reference belongs where the stroke actually STARTS. For a band
          ;; stroke that is the pushed-off-ridge head (x,y): referencing the pre-snap
          ;; seed, which can sit on the FAR side of the boundary, makes the chroma
          ;; backstop fire on step 1 and the band never traces at all.
          [hr hg hb0] (sample-arr blurd-px iw ih
                                  (if band? x (+ cx0 bax))
                                  (if band? y (+ cy0 bay)))]
      (let [traced (loop [k 0 px (double x) py (double y) dxp 0.0 dyp 0.0 fade 1.0 acc []]
                     (if (> k kmax)
                       [acc :cap]
                       (let [[th coh] (sample-fields nf px py)
                             dx0 (Math/cos th) dy0 (Math/sin th)
                             ev  (wavelet/edge-at dmap px py)
                             detail? (>= (long lvl) 2)
                             ;; FEATURE-FOLLOWING TRACER: a stroke ends where the FEATURE
                             ;; ends or turns — the way a person paints (the top of an eye is
                             ;; one line, the bottom another; a finger contour one long wavy
                             ;; line). These GEOMETRIC stops are a CLEAN break: no segment is
                             ;; emitted at the break, so a contour chunks into the strokes a
                             ;; draughtsman draws. They replace the old fixed-span cap and the
                             ;; TURN-KILL fade — length now follows the feature — and apply to
                             ;; ALL detail tiers (lvl>=2), not just liners. (mirror gen.)
                             geo (cond
                                   (and detail? (pos? k) (< ev edge-floor)) :ridge
                                   (and detail? (pos? k) (< (Math/abs (+ (* dx0 dxp) (* dy0 dyp))) bend-cos)) :corner
                                   :else nil)]
                         (if geo
                           [acc geo]
                           (let [;; COLOUR is a BACKSTOP, not a stop signal. Along a real
                                 ;; feature the colour SHOULD change, so the two-tier dry-out
                                 ;; and the path-roughness (racc) accumulator — which fired
                                 ;; after 2-4 segments in detailed areas — WERE the "short
                                 ;; disjointed lines" artifact. Detail tiers keep only a HARD
                                 ;; stop at `runaway` (the stroke wandered into a foreign
                                 ;; colour region); the broad/coverage tiers keep their
                                 ;; two-tier dry-out unchanged. (mirror gen.)
                                 dmx (if (pos? k)
                                       (let [[br bg bb] (sample-arr blurd-px iw ih (+ px bax) (+ py bay))]
                                         (max (Math/abs (- br hr)) (Math/abs (- bg hg)) (Math/abs (- bb hb0))))
                                       0.0)
                                 chroma? (and detail? (pos? k) (> dmx runaway))
                                 fade (cond (not (pos? k)) fade
                                            chroma? 0.0
                                            detail? fade
                                            (> dmx 0.18) 0.0
                                            (> dmx 0.22) (* fade 0.4)
                                            :else fade)
                                 ;; LINE-HOLD stays — the one non-geometric stop. A liner that
                                 ;; walked off ITS OWN placement map has left the feature, so
                                 ;; lift. It reads the level's map (the right signal), unlike
                                 ;; tensor coherence, which does NOT discriminate a line from a
                                 ;; gradient (a ramp is rank-1 = perfectly oriented), so the old
                                 ;; coherence-follow fade guard is removed.
                                 mv  (if (and liner? (pos? k)) (* (map-at dmap mkind px py) (double gainv)) 1.0)
                                 lh? (and liner? (pos? k) (< mv (* 0.35 (double lth))))
                                 fade (cond lh? 0.0
                                            (and liner? (pos? k) (< mv (* 0.7 (double lth)))) (* fade 0.5)
                                            :else fade)]
                             (if (< fade 0.15)
                               [acc (cond chroma? :chroma lh? :line-hold :else :drift)]
                               (let [body (* (if liner?
                                               (min 1.0 (max 0.0 (/ (- ev 0.25) 0.45)))
                                               (+ 0.4 (* 0.6 (double sgate))))
                                             (if soft-ramp? 0.35 1.0))
                                     lal2 (+ lal (* (max 0.0 (- 0.9 lal)) body))
                                     ;; store the tt-INDEPENDENT per-step state; the BOTH-ENDS
                                     ;; taper is finalized after the loop against the ACTUAL
                                     ;; traced length, so a corner-broken stroke still gets a
                                     ;; full head->tail profile instead of a truncated one.
                                     pre [k px py D sn tn th (* coh (- 1.0 (double melt))) hb body lal2 fade]
                                     bend (* (double curvature) 0.9 (double bendf)
                                             (min 1.0 (max 0.0 (/ (- (double ssz) 2.5) 2.5)))
                                             (- 1.0 (* 0.7 coh))
                                             (- 1.0 (min 1.0 (max 0.0 (/ (- ev 0.3) 0.3))))
                                             (- (noise/noise2 (+ (* 0.05 px) (* 89.0 (double bph)))
                                                              (+ (* 0.05 py) (* 57.0 (double bph)))) 0.5))
                                     cb (Math/cos bend) sb (Math/sin bend)
                                     sgn (if (zero? k) (double dirsign)
                                             (if (neg? (+ (* dx0 dxp) (* dy0 dyp))) -1.0 1.0))
                                     dx1 (* sgn dx0) dy1 (* sgn dy0)
                                     dx (- (* cb dx1) (* sb dy1)) dy (+ (* sb dx1) (* cb dy1))
                                     [dx dy] (if (and liner? (pos? k))
                                               (let [mx (+ (* 0.35 dx) (* 0.65 dxp))
                                                     my (+ (* 0.35 dy) (* 0.65 dyp))
                                                     ml (Math/sqrt (+ (* mx mx) (* my my)))]
                                                 (if (> ml 1e-6) [(/ mx ml) (/ my ml)] [dx dy]))
                                               [dx dy])
                                     L  (* ssz (double stepf))
                                     nx0 (max 0.0 (min hd (+ px (* L dx))))
                                     ny0 (max 0.0 (min wd (+ py (* L dy))))
                                     [nx1 ny1] (if snap? (edge-snap dmap nf nx0 ny0 1.75 hd wd sgain) [nx0 ny0])
                                     [nx2 ny2] (if (zero? (double side))
                                                 [nx1 ny1]
                                                 [(max 0.0 (min hd (+ nx1 (* sidem soff ssz (- dy)))))
                                                  (max 0.0 (min wd (+ ny1 (* sidem soff ssz dx))))])]
                                 (recur (inc k) nx2 ny2 dx dy fade (conj acc pre)))))))))]
        ;; FINALIZE: taper follows the ACTUAL traced length. A stroke that stopped at a corner
        ;; still gets a full head->tail profile (tt = k/(finalLen-1)), not the truncated profile
        ;; the old tt=k/kmax gave when geometry ended the chain early. (mirror gen two-phase.)
        (let [[pre-records reason] traced
              n (count pre-records)
              denom (double (max 1 (dec n)))
              rows (mapv (fn [pr]
                           (let [[k px py D0 sn0 tn0 th chb hb body lal2 fadep] pr
                                 tt (/ (double k) denom)
                                 hw (let [u (min 1.0 (/ tt 0.18)) s (* u u (- 3.0 (* 2.0 u)))]
                                      (if liner? (+ 0.8 (* 0.2 s)) (+ 0.55 (* 0.45 s))))
                                 ha (let [u (min 1.0 (/ tt 0.15)) s (* u u (- 3.0 (* 2.0 u)))]
                                      (if liner? (+ 0.75 (* 0.25 s)) (+ 0.5 (* 0.5 s))))
                                 sz (* ssz (- 1.0 (* 0.45 tt (Math/sqrt tt))) hw)
                                 al (* lal2 fadep (- 1.0 (* 0.65 tt tt)) ha)
                                 ;; DETAIL tiers RE-LOAD colour each segment (wsl=1): one
                                 ;; brush-load per stroke is right for a gestural mark, but
                                 ;; at detail scale (ssz~2.2, ~4 segs = 8px) it drags the
                                 ;; head's colour onto the next feature (nostril, philtrum
                                 ;; and lip line are 3-5px apart on a face).
                                 wsl (cond soft-ramp? 1.0 detail? 1.0
                                           (pos? (double melt)) (* 0.85 (double melt) tt) :else 0.0)
                                 ;; the perpendicular offset escapes the AA ramp for a stroke
                                 ;; CARRYING one brush-load across its length (wsl<1). A tier
                                 ;; that RE-LOADS colour every segment (wsl=1) samples its OWN
                                 ;; position; the fixed offset there reads the dark ground along
                                 ;; a lit edge (the dark line through a finger), so it drops for
                                 ;; carried loads only.
                                 obx (* (double bax) (double (if (< wsl 1.0) 1.0 0.0)))
                                 oby (* (double bay) (double (if (< wsl 1.0) 1.0 0.0)))
                                 cxs (+ cx0 obx (* wsl (- (double px) (double cx0))))
                                 cys (+ cy0 oby (* wsl (- (double py) (double cy0))))]
                             [px py sz D0 sn0 tn0 al th chb hb cxs cys traw (spec-cap lvl ssz) selong]))
                         pre-records)]
          [rows reason])))))

(defn- stub-glaze
  "Judge a traced chain by WHY it stopped, not how long it got. A feature-following
   stroke that ended at a corner or where its ridge died is a COMPLETE short stroke
   (the top of an eye is one line, the bottom another) and paints at full alpha. Only
   a chain that died to the chroma BACKSTOP — it wandered into a foreign colour
   region, i.e. it FAILED to follow a feature — is demoted to glaze (×0.5): the old
   length test halved exactly the short corner-strokes that are now the wanted effect.
   A standalone helper (NOT a let-wrap around the trace loop — demoting that loop out
   of tail position sent the jolt compiler pathological)."
  [lvl reason rows]
  (if (and (>= (long lvl) 2) (= reason :chroma))
    (mapv (fn [r] (assoc r 6 (* 0.5 (double (nth r 6))))) rows)
    rows))

(defn- layered-means
  "COARSE-TO-FINE placement: a base layer of large splats that FULLY COVERS the image —
   spacing < stdev ⇒ heavy overlap, so the (black) background can never show through — then
   progressively finer layers, each placed only where the wavelet detail is high enough, so
   detail accumulates ON TOP of an unbroken underpainting. There is no cell grid, so no cell
   facets; each splat's orientation/colour come from the flow + detail fields.

   Per-level geometry (ssz/sp/th/nx/ny, budget scale, finest-first order) comes from
   `layer-params` — the same spec the GPU generation pass consumes, so the two paths place
   identical cells. Here we walk it on the CPU: threshold-test each cell, jitter + Perlin-warp
   the surviving seed, then hand it to `stroke-segments` (base fill vs traced brush stroke).
   Emits [x y size D sn tn alpha theta coherence] per SEGMENT (D = effective detail 0..1;
   sn/tn = per-seed size/tone jitter hashes in [-0.5,0.5])."
  [dmap nf detail size variation curvature stroke tier-muls count H W blur-px blurd-px]
  (let [hd   (double (dec (long H))) wd (double (dec (long W)))
        iw   (long W) ih (long H)
        deff (fn [D] (min 1.0 (* (double detail) (double D) 2.2)))
        rr   (/ (double H) 24.0)                    ; subjectness tap radius
        bmul (double (nth tier-muls 0))
        bmin (min 1.0 bmul)
        {:keys [warp levels]} (layer-params dmap detail size variation curvature stroke tier-muls count H W)]
    (persistent!
      (reduce
        (fn [acc [idx {:keys [lvl ssz sp th nx ny segs stepf bendf map-kind traw sideo selong]}]]
          (loop [i 0 acc acc]
            (if (>= i nx)
              acc
              (recur (inc i)
                (loop [j 0 acc acc]
                  (if (>= j ny)
                    acc
                    (let [;; white-noise candidate position — AVALANCHE-hashed (a linear
                          ;; hash here lays the points on Marsaglia lines), in-bounds
                          cx (* (double H) (poshash i lvl 29))
                          cy (* (double W) (poshash i lvl 31))]
                      (if false
                        (recur (inc j) acc)
                        (let [;; wavelet subjectness (LOCAL-relative): gates mid/fine
                              ;; placement — splat size follows the wavelet's local
                              ;; detail density, so dark low-contrast texture still
                              ;; receives strokes.
                              sgate (subject-at dmap cx cy rr)
                              ;; ABSOLUTE subjectness: drives the bokeh-adaptive broad
                              ;; tier. The local-relative gate saturates to 1 on smooth
                              ;; bokeh (its normalization amplifies sensor noise), which
                              ;; made Broad growth/thinning/melt inert exactly where
                              ;; they exist to act.
                              sabs  (wavelet/subject-abs-at dmap cx cy)
                              ;; the Broad growth is gated by the whole GROWN FOOTPRINT,
                              ;; not just the centre: a daub centred 25px out in the bokeh
                              ;; reads flat there, grows ×2.5, and its body reaches back
                              ;; across the silhouette — a wash veil over the subject's rim
                              ;; that gets worse the higher Broad goes. The subject map is
                              ;; wide (box-blurred), so 8 sparse taps at the grown radius
                              ;; can't miss a nearby subject the way thin edge-band taps do.
                              sfoot (if (and (<= (long lvl) 1) (> bmul 1.0))
                                      (let [m0 (+ 1.0 (* (- bmul 1.0) (- 1.0 sabs)))
                                            d  (* 1.2 ssz m0)
                                            dd (* 0.7071 d)
                                            sa (fn [x y] (wavelet/subject-abs-at dmap x y))]
                                        (max sabs
                                             (sa (+ cx d) cy) (sa (- cx d) cy)
                                             (sa cx (+ cy d)) (sa cx (- cy d))
                                             (sa (+ cx dd) (+ cy dd)) (sa (- cx dd) (- cy dd))
                                             (sa (+ cx dd) (- cy dd)) (sa (- cx dd) (+ cy dd))))
                                      sabs)
                              mloc  (+ 1.0 (* (- bmul 1.0) (- 1.0 sfoot)))
                              ;; broad tier: flat regions thin candidates by (bmin/m)² as
                              ;; the kept seeds grow ×m — few LARGE daubs = smooth bokeh;
                              ;; at full subjectness m=1 and the Broad dial has no effect.
                              thin? (and (<= (long lvl) 1)
                                         (let [pr (/ bmin mloc)]
                                           (>= (hash01 (+ (* i 61) lvl) j 43) (* pr pr))))
                              ;; mid/fine strokes belong where the wavelets see detail:
                              ;; their map value is gated by subjectness so flat bokeh
                              ;; keeps only the big smooth daubs. The ABSOLUTE gate
                              ;; rides the Broad slider: past 1.0 it thins mid/fine
                              ;; marks out of truly flat regions (isolated dark flecks
                              ;; on a melted wash), leaving them at Broad ≤ 1 where
                              ;; visible strokes are the wanted effect.
                              ;; SQUARED: the linear form bottomed out around 0.55 — never
                              ;; enough to actually stop accents whose locally-normalized
                              ;; maps light up on bokeh noise. Squaring lets the gate reach
                              ;; blocking strength as Broad rises while staying exactly 1 at
                              ;; Broad ≤ 1. It also gates LEVEL 1 now: the level-1 chains
                              ;; were the fibrous filament texture covering empty regions.
                              bgate (let [g (- 1.0 (* (min 1.0 (max 0.0 (/ (- bmul 1.0) 1.5)))
                                                      (- 1.0 (min 1.0 (/ sabs 0.35)))))]
                                      (* g g))
                              gain  (cond (>= (long lvl) 2) (* (+ 0.25 (* 0.75 sgate)) bgate)
                                          (== (long lvl) 1) bgate
                                          :else 1.0)
                              ;; each level reads the map matched to ITS scale: the finest
                              ;; levels use the sharp fine-band map so they land on (and
                              ;; preserve) small structure the smoothed aggregate blurs away.
                              ;; The cutoff is DITHERED ±25% per seed — a hard threshold on
                              ;; a map oscillating around it dashes contours into beads.
                              dv (map-at dmap map-kind cx cy)
                              thd (* th (+ 0.75 (* 0.5 (hash01 (+ (* i 43) lvl) j 19))))
                              ;; SUBDIVISION (broad/mid tiers only): skip if the next-finer
                              ;; level (previous entry — levels are finest-first) claims
                              ;; this cell, dithered like the threshold so the handoff
                              ;; interleaves. From level 3 up there is NO claim — the fine
                              ;; glazes overlap the mid strokes and mix instead of
                              ;; replacing them.
                              claimed? (and (pos? (long lvl)) (<= (long lvl) 2) (pos? (long idx))
                                            (let [fl (nth levels (dec (long idx)))]
                                              ;; the EDGE-BAND tier is NOT a rung of the
                                              ;; ladder — it overlays it — so it never
                                              ;; claims a cell away from the level below.
                                              (and (not (:band fl))
                                                   ;; the finer level is always ≥2, so its
                                                   ;; claim carries the same subject gate
                                                   (let [fdv (* (map-at dmap (:map-kind fl) cx cy)
                                                                (+ 0.25 (* 0.75 sgate)) bgate)]
                                                     (>= fdv (* (:th fl)
                                                                (+ 0.75 (* 0.5 (hash01 (+ (* i 47) lvl) j 23)))))))))]
                          (if (or thin?
                                  (and (pos? (long lvl)) (or claimed? (< (* dv gain) thd))))
                            (recur (inc j) acc)      ; thinned bokeh / not detailed enough
                            (let [;; bokeh-adaptive broad size: kept flat-region daubs grow ×m
                              ssz (if (<= (long lvl) 1) (* ssz mloc) ssz)
                              ;; hashed positions need no jitter — they ARE the noise
                              x  cx y cy
                              D  (deff dv)
                              ;; flat-region Perlin warp breaks any residual level lattice;
                              ;; detail strokes (D≈1) stay put → faithful edges. The liner
                              ;; tier gets NO warp at all — fine seeds must land exactly on
                              ;; the detail they trace (see bend-frac).
                              aw (if (liner-scale? lvl ssz) 0.0 (* warp (- 1.0 D) ssz))
                              x2 (if (< aw 0.2) x
                                   (+ x (* aw (noise/noise2 (* 0.06 x) (* 0.06 y)))))
                              y2 (if (< aw 0.2) y
                                   (+ y (* aw (noise/noise2 (+ 41.3 (* 0.06 x)) (+ 17.9 (* 0.06 y))))))
                              sn0 (- (hash01 (+ (* i 31) lvl) j 11) 0.5)
                              ;; MELT: how much a flat-region broad stroke should sink into
                              ;; the wash. Grows only past Broad 1.0 (below that, strokes
                              ;; are the wanted effect) and only where FOOTPRINT subjectness
                              ;; is low: strokes touching a subject keep their identity.
                              melt (if (<= (long lvl) 1)
                                     (* (min 1.0 (max 0.0 (/ (- bmul 1.0) 1.5)))
                                        (- 1.0 sfoot))
                                     0.0)
                              ;; size jitter applies at SEED level to the whole chain —
                              ;; segment size AND step together — so chains stay
                              ;; self-overlapping at any Variation (per-segment size
                              ;; jitter with a fixed step beaded strokes into dotted
                              ;; pearls). Broad levels keep 40% (base coverage), muted
                              ;; further by melt — a wash has no per-stroke identity.
                              ;; the shrink side is CLAMPED at 0.75: strokes jittered far
                              ;; below their level's size land at the bottom of the
                              ;; hardness ramp and render as isolated hard pearls along
                              ;; edges — variety comes from growing, not vanishing.
                              szf (max 0.75 (+ 1.0 (* variation sn0
                                                      (if (<= (long lvl) 1) (* 0.4 (- 1.0 melt)) 1.0))))
                              ;; near a strong edge the mid fill levels don't paint (their
                              ;; boundary-band chains ribbon mixed colour along silhouettes
                              ;; as a ghost veil) and base daubs SHRINK so their soft tails
                              ;; can't reach across the silhouette. Sensed over the stroke's
                              ;; FOOTPRINT (taps at 0.75·size), not just its centre.
                              Ev (if (<= (long lvl) 3)
                                   (edge-near dmap cx cy (* 0.75 ssz))
                                   (wavelet/edge-at dmap cx cy))
                              ;; the chain's FINAL stdev: seed jitter × the σ-aware
                              ;; near-edge shrink (small strokes keep the gentle old
                              ;; coefficients — base coverage still reaches the boundary —
                              ;; but past ~8px a daub centred on a thin bright feature
                              ;; shrinks toward the feature scale instead of ghosting
                              ;; its colour across the silhouette, capped 0.7).
                              cssz (* ssz szf (- 1.0 (* (min 0.7
                                                             (* (cond (zero? (long lvl)) 0.25
                                                                      (<= (long lvl) 3) 0.45
                                                                      :else 0.1)
                                                                (max 1.0 (/ (* ssz szf) 8.0))))
                                                        Ev)))
                              ;; tone jitter follows the PHYSICAL stroke scale, not the
                              ;; level index: broad fills keep 25% (melt-muted), and any
                              ;; small mark keeps 15% regardless of which level painted it
                              ;; — which level is "fine" depends on the sliders, and full
                              ;; jitter on small strokes beads contours into speckle.
                              tn (* (let [l (long lvl)]
                                      (cond (<= l 1) (* 0.25 (- 1.0 melt))
                                            (>= l 4) 0.15
                                            :else (+ 0.15 (* 0.85 (min 1.0 (max 0.0 (/ (- cssz 2.5) 2.5)))))))
                                    (- (hash01 (+ (* i 37) lvl) j 13) 0.5))
                              ds (if (< (hash01 (+ (* i 41) lvl) j 17) 0.5) 1.0 -1.0)
                              ;; keep centres in-bounds so no budget is wasted off-screen
                              ;; (edges stay covered by the splats' tails).
                              emitted (if (and (or (== (long lvl) 1) (== (long lvl) 2))
                                               (> Ev 0.45)
                                               ;; dithered — a few mid strokes still fill
                                               ;; the edge band so fine contour strokes sit
                                               ;; IN paint, but level 1's chains are OPAQUE
                                               ;; heavy-blur ribbons, so they suppress at
                                               ;; 90% vs level 2's 75%. Level 3 stays: it
                                               ;; carries text/eye-scale features; near
                                               ;; edges it shrinks hard (below) instead.
                                               (< (hash01 (+ (* i 53) lvl) j 37)
                                                  (if (== (long lvl) 1) 0.9 0.75)))
                                        []
                                        (let [[rows reason] (stroke-segments nf dmap lvl
                                                         (max 0.0 (min hd x2)) (max 0.0 (min wd y2))
                                                         cssz
                                                         D 0.0 tn ds curvature stroke hd wd
                                                         segs stepf bendf
                                                         (if (<= (long lvl) 1) 1.0 0.0)
                                                           ;; fine colour rawness follows the LOCAL FINE-DETAIL
                                                           ;; DENSITY (:sharp) — in a crowded region a single
                                                           ;; raw sample is unreliable, so trust the region more
                                                           (density-scaled-traw lvl traw (wavelet/sharp-at dmap cx cy))
                                                         sgate blur-px iw ih th melt
                                                         map-kind gain blurd-px (hash01 (+ (* i 67) lvl) j 53)
                                                         (double sideo) (double selong))]
                                          (stub-glaze lvl reason rows)))]
                          (recur (inc j) (reduce conj! acc emitted)))))))))))))
        (transient [])
        (map-indexed vector levels)))))

;; --- precomputed smooth Perlin fields (flow angle, size, tone) ---------------
;; noise2 is ~30 ops; calling it 4× per splat over ~14k splats dominated the render.
;; The fields are smooth (low frequency), so precomputing them at the tensor resolution
;; once and sampling (a cheap aget) per splat is visually identical and far faster.

(defn prep-noise
  "Precompute, at `sfield`'s tensor resolution, the per-POSITION stroke orientation
   field: the final blended orientation (edge-seeded flow + Perlin fill + sharp edge)
   stored as its DOUBLE-ANGLE components cos(2θ)/sin(2θ) — the representation that
   interpolates correctly for undirected orientations (0 ≡ π) — plus the coherence.
   Storing components instead of the raw angle lets sample-fields (and the GPU's
   texture fetch) blend BILINEARLY between texels: nearest-neighbour sampling of a
   coarse angle grid stair-steps stroke orientation along every contour, which reads
   as a regular sawtooth/zipper in the render. Per-stroke size/tone jitter is NOT a
   field any more — it's per-seed hash01 in layered-means (jitter should be
   independent per stroke, not spatially smooth). Returns
   {:h :w :src-h :src-w :c2 :s2 :coherence}."
  [sfield]
  (let [H (:h sfield) W (:w sfield)
        srch (double (or (:src-h sfield) H)) srcw (double (or (:src-w sfield) W))
        n (* H W) fs 0.004
        ^doubles s-theta (:theta sfield) ^doubles s-coh (:coherence sfield)
        ^doubles f-theta (:flow-theta sfield) ^doubles f-str (:flow-str sfield)
        c2 (double-array n) s2 (double-array n) cohr (double-array n)]
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
          (aset c2 idx (Math/cos (* 2.0 theta)))
          (aset s2 idx (Math/sin (* 2.0 theta)))
          (aset cohr idx coherence))))
    {:h H :w W :src-h (:src-h sfield) :src-w (:src-w sfield)
     :c2 c2 :s2 s2 :coherence cohr}))

(defn- sample-fields
  "[theta coherence] at full-image (x,y), BILINEARLY interpolated from the prep-noise
   grid. The orientation blends in double-angle space (c2/s2 components) so 0 ≡ π is
   seamless; θ = ½·atan2(s2,c2). The GPU generation shader implements this exact
   formula (same continuous coord fx = x·H/srch, same floor/clamp), so both paths
   compute identical fields."
  [nf x y]
  (let [H (long (:h nf)) W (long (:w nf))
        srch (double (or (:src-h nf) H)) srcw (double (or (:src-w nf) W))
        ^doubles c2 (:c2 nf) ^doubles s2 (:s2 nf) ^doubles coh (:coherence nf)
        fx (min (double (dec H)) (max 0.0 (* (double x) (/ (double H) srch))))
        fy (min (double (dec W)) (max 0.0 (* (double y) (/ (double W) srcw))))
        i0 (long fx) i1 (min (dec H) (inc i0)) wx (- fx (double i0))
        j0 (long fy) j1 (min (dec W) (inc j0)) wy (- fy (double j0))
        bl (fn [^doubles a]
             (let [v00 (aget a (+ (* i0 W) j0)) v01 (aget a (+ (* i0 W) j1))
                   v10 (aget a (+ (* i1 W) j0)) v11 (aget a (+ (* i1 W) j1))]
               (+ (* (- 1.0 wx) (+ (* (- 1.0 wy) v00) (* wy v01)))
                  (* wx         (+ (* (- 1.0 wy) v10) (* wy v11))))))]
    [(* 0.5 (Math/atan2 (bl s2) (bl c2)))
     (min 1.0 (max 0.0 (bl coh)))]))

;; --- helpers (unchanged) ----------------------------------------------------

(defn- sample-arr
  "Bilinear [r g b] from a flat H*W*3 double-array at grid (x,y) (x=row, y=col),
   CLAMPED at the borders. At integer coords returns the stored texel; at a texel
   midpoint returns the neighbours' mean. Matches the GLSL sampleRGB exactly
   (CPU/GPU parity) and the test reference bilerp-src. width=cols, height=rows."
  [^doubles arr width height x y]
  (let [W (long width) H (long height)
        gx (double (max 0.0 (min (double (dec H)) (double x))))
        gy (double (max 0.0 (min (double (dec W)) (double y))))
        x0 (int gx) y0 (int gy)
        x1 (min (long (dec H)) (inc x0)) y1 (min (long (dec W)) (inc y0))
        fx (- gx (double x0)) fy (- gy (double y0))
        tex (fn [xi yi] (let [b (* 3 (+ (* xi W) yi))]
                          [(aget arr b) (aget arr (+ b 1)) (aget arr (+ b 2))]))
        [r00 g00 b00] (tex x0 y0) [r10 g10 b10] (tex x1 y0)
        [r01 g01 b01] (tex x0 y1) [r11 g11 b11] (tex x1 y1)
        lx (fn [a b t] (+ (double a) (* (double t) (- (double b) (double a)))))
        r0 (lx r00 r10 fx) g0 (lx g00 g10 fx) b0 (lx b00 b10 fx)
        r1 (lx r01 r11 fx) g1 (lx g01 g11 fx) b1 (lx b01 b11 fx)]
    [(lx r0 r1 fy) (lx g0 g1 fy) (lx b0 b1 fy)]))

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

(defn splat-record
  "The pure per-splat math — THE SPEC the GPU generation shader mirrors. Given a stroke's mean
   (x,y), its size `csz` and detail level `dlev`, the sampled orientation fields (θ, coherence,
   size-noise, tone-noise) and the two sampled source colours (`blur-rgb` smooth base + `raw-rgb`
   crisp pixel), returns {:mean :cov :color}. All field/colour SAMPLING is done by the caller
   (CPU: array lookups; GPU: texture fetches) — this fn is only the arithmetic, so both paths
   compute identical splats.

     covariance: elongation e = 1 + min(stroke,1.5)·coh·(0.25+0.75·dlev) — capped: stroke
                 LENGTH comes from the segment chain (which follows the curve); unbounded
                 segment elongation made rigid needles that ink dark contours across edges;
                 s0 = csz·(1 + variation·snoise) jitters size; Σ = R(θ)·diag((s0·√e)²,(s0/√e)²)·Rᵀ.
     colour:     t = 0.15 + 0.85·max(coherence,dlev) blends blur→raw — mostly the smooth
                 blur in flat regions (seamless gradients, no stroke banding), raw at
                 edges/detail;
                 contrast about 0.5; tone = 1 + variation·0.3·tnoise."
  [x y csz dlev theta coherence snoise tnoise blur-rgb raw-rgb stroke variation contrast traw tcap selong]
  (let [coh (+ min-coh (* (- 1.0 min-coh) coherence))
        e   (+ 1.0 (* (min (double stroke) 1.5) coh (+ 0.25 (* 0.75 (double dlev)))))
        ;; `selong` > 0 forces the elongation instead of deriving it from the local
        ;; tensor — the EDGE-BAND tier is BORN long-and-thin so that its across-edge
        ;; sigma is a property of the tier (ssz/selong) and not of whatever coherence
        ;; happens to be at that spot. 0 keeps the coherence-derived elongation.
        se  (if (pos? (double selong)) (double selong) (Math/sqrt e))
        s0  (* csz (+ 1.0 (* variation 0.5 (* 2.0 snoise))))
        sx  (* s0 se)                 ; long axis along θ
        sy  (/ s0 se)                 ; short axis across the stroke
        ;; t is FLOORED by the level's rawness (traw) and CEILINGED by its
        ;; specificity cap (tcap) — the progressive colour ladder: broad layers
        ;; averaged, fine layers specific, whatever coherence says.
        ;; the cap also follows the BRUSH SIZE: a fat brush cannot place a
        ;; pixel-specific highlight — as the stroke stdev grows past ~4px the
        ;; ceiling eases to fully-averaged (0.35). At high budgets (small mid
        ;; strokes) this is a no-op; at low budgets it stops fat mid dabs from
        ;; stamping raw bright speckles over the surface they sit on.
        tcap2 (min (double tcap)
                   (+ 0.3 (* 0.7 (min 1.0 (/ 3.0 (max csz 1e-6))))))
        t   (min tcap2
                 (max (double traw)
                      (min 1.0 (max 0.0 (+ 0.15 (* 0.85 (max coherence (double dlev))))))))
        [br bg bb] blur-rgb [rr rg rb] raw-rgb
        color0 [(+ (* br (- 1.0 t)) (* rr t))
                (+ (* bg (- 1.0 t)) (* rg t))
                (+ (* bb (- 1.0 t)) (* rb t))]
        color-ac (if (== contrast 1.0) color0 (apply-contrast contrast color0))
        tone (+ 1.0 (* variation 0.15 (* 2.0 tnoise)))
        ;; per-stroke TEMPERATURE: each brush-load leans a touch warm (R up, B down)
        ;; or cool — reloaded paint is never mixed identically. snoise is the seed's
        ;; per-stroke noise (constant along the whole stroke), sampled identically on
        ;; the GPU, so this stays exact CPU/GPU parity without a new hash.
        temp (* variation 0.10 (* 2.0 snoise))
        [r g b] color-ac
        color [(max 0.0 (min 1.0 (* r tone (+ 1.0 temp))))
               (max 0.0 (min 1.0 (* g tone)))
               (max 0.0 (min 1.0 (* b tone (- 1.0 temp))))]]
    {:mean  [x y]
     :cov   (gauss/covariance sx sy theta)
     :color color}))

;; --- main -------------------------------------------------------------------

(defn splat-field
  "Build a splat field from `image` (see ns doc) and `controls` (see ns doc).
   Returns {:splats […] :background [r g b] :height :width :opacity}."
  [{:keys [height width pixels] :as image} controls]
  (let [{:keys [count size stroke detail variation curvature opacity contrast background
                size-broad size-mid size-fine edge-band]
         :or   {count 6000 size 3.0 stroke 2.0 detail 0.6 variation 0.5 curvature 0.5
                opacity 0.9 contrast 1.0 background 0.0
                size-broad 1.0 size-mid 1.0 size-fine 1.0 edge-band 1.0}} controls
        n          (long (or count 6000))
        size       (double (or size 3.0))
        stroke     (double stroke)
        detail     (double detail)
        variation  (double variation)
        curvature  (double curvature)
        contrast   (double contrast)
        sfield     (or (:structure image) (structure/analyze image))
        dmap       (or (:detail image)    (wavelet/placement-map image sfield))
        ^doubles raw-px  pixels
        ^doubles blur-px (or (:blur image) pixels)
        ^doubles blurh-px (or (:blur-heavy image) blur-px)
        ^doubles blurd-px (or (:blur-drift image) blur-px)
        nf         (or (:noise-fields image) (prep-noise sfield))
        segments   (layered-means dmap nf detail size variation curvature stroke
                                  [(double size-broad) (double size-mid) (double size-fine)
                                   (double edge-band)]
                                  n height width blur-px blurd-px)
        ;; each segment carries its sampled fields + taper alpha (stroke-segments did the
        ;; tracing); hand off to the pure `splat-record` math shared with the GPU.
        splats     (vec
                     (for [[x y csz dlev sn tn alpha theta coherence hb hx hy traw tcap selong] segments
                           :let [bilat-rgb (sample-arr blur-px width height hx hy)
                                 blur-rgb  (if (and hb (pos? (double hb)))
                                             (sample-arr blurh-px width height hx hy)
                                             bilat-rgb)
                                 ;; REGION-CONSISTENCY clamp: the bilateral DEFINES the
                                 ;; region, and raw specificity is only trusted when it
                                 ;; agrees with it. In-region micro-contrast (raw within
                                 ;; 0.12 of the bilateral — highlights, freckles) keeps
                                 ;; full raw pop; a raw pixel that disagrees with its own
                                 ;; region's colour by more sits on the AA ramp or the
                                 ;; wrong side of the boundary and is pulled to the region
                                 ;; colour — a single edge-straddling raw sample can no
                                 ;; longer bleed a light region across into a dark one at
                                 ;; the high raw weight the per-level floor (0.45–0.85)
                                 ;; otherwise gives it.
                                 [bcr bcg bcb] bilat-rgb
                                 [rr rg rb]   (sample-arr raw-px width height hx hy)
                                 d (max (Math/abs (- rr bcr)) (Math/abs (- rg bcg)) (Math/abs (- rb bcb)))
                                 w (max 0.0 (min 1.0 (/ (- d 0.12) 0.15)))
                                 raw-rgb [(+ (* (- 1.0 w) rr) (* w bcr))
                                          (+ (* (- 1.0 w) rg) (* w bcg))
                                          (+ (* (- 1.0 w) rb) (* w bcb))]]]
                       (assoc (splat-record x y csz dlev theta coherence sn tn
                                            blur-rgb raw-rgb stroke variation contrast
                                            (or traw 0.0) (or tcap 1.0) (or selong 0.0))
                              :alpha (double alpha))))
        ;; PAINT ORDER needs NO sort: `layered-means` emits finest level first, so the field is
        ;; already small→large. The shader composites front-to-back (index 0 = topmost), so the
        ;; small crisp detail strokes sit at the front over the big soft underpainting. Dropping
        ;; the O(n log n) sort matters at high splat counts and mirrors the GPU path (which
        ;; likewise gets paint order for free from level order). One pass for the size range.
        sigs   (map (fn [{[c00 c01 _ c11] :cov}]
                      (Math/sqrt (Math/sqrt (max (- (* c00 c11) (* c01 c01)) 1e-8))))
                    splats)
        sig-min (if (seq sigs) (reduce min sigs) 1.0)
        sig-max (if (seq sigs) (reduce max sigs) 1.0)]
    {:splats     splats
     :background (resolve-background background)
     :height     height
     :width      width
     :opacity    (double opacity)
     :sig-min    (double sig-min)
     :sig-max    (double sig-max)}))
