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
  "segments per stroke at placement level. The fine tier (lvl≥4) traces LONG
   8-segment liner lines: at a couple-of-pixels stdev a 2-segment chain is a dot,
   and a contour drawn as dots reads jagged — a clean thin line needs one
   continuous stroke. Budget-invariant: seed spacing scales with √segs, so the
   same segment count is arranged into fewer, longer strokes."
  [lvl]
  (let [l (long lvl)] (cond (zero? l) 1 (== l 1) 6 (== l 2) 4 (== l 3) 3 :else 8)))
(defn- step-frac
  "step length as a fraction of the level stdev. Fine liner strokes step ~0.9σ —
   close enough that the segment gaussians fuse into a smooth continuous rod, far
   enough that 8 segments span a real line along the edge."
  [lvl]
  (let [l (long lvl)] (cond (== l 1) 1.1 (== l 2) 0.9 (== l 3) 0.75 (== l 4) 0.9 (== l 5) 0.85 :else 0.8)))
(defn- bend-frac "how much of the Curvature bend this level keeps" [lvl]
  (let [l (long lvl)] (cond (== l 1) 1.0 (== l 2) 0.55 (== l 3) 0.3 (== l 4) 0.15 (== l 5) 0.1 :else 0.05)))
(defn- tier-mul
  "Per-tier size multiplier from [broad mid fine] — the user's independent control
   of each resolution band: loosen/blur the background (broad up) while keeping
   small details focused (fine at or below 1)."
  [muls lvl]
  (let [l (long lvl)]
    (double (cond (<= l 1) (nth muls 0) (<= l 3) (nth muls 1) :else (nth muls 2)))))

(defn- level-alpha
  "Paint translucency per level — PROGRESSIVE REFINEMENT: broad layers are opaque
   (coverage), each finer layer glazes more, letting the accumulated layers show
   through so detail builds on the underpainting instead of scratching over it.
   From level 3 up the layers OVERLAP (no subdivision handoff), so the fine tier
   glazes light and the many stacked strokes MIX into a smooth mid→fine gradient
   instead of one stroke owning each spot."
  [lvl]
  (let [l (long lvl)] (cond (<= l 1) 1.0 (<= l 3) 0.85 (<= l 5) 0.65 :else 0.55)))

(defn- level-map-kind
  "Which placement map a level reads — matched to the scale it paints: broad levels
   the smoothed aggregate, mid levels the MID band map (face-feature frequencies),
   the finest levels the sharp fine-band map."
  [lvl]
  (let [l (long lvl)] (cond (<= l 1) :detail (<= l 3) :mid :else :sharp)))
(defn- raw-floor
  "Colour-rawness floor per level: small strokes must paint faithful colour — a
   half-blur blend at feature scale just softens the feature it exists to keep."
  [lvl]
  (let [l (long lvl)] (cond (<= l 1) 0.0 (<= l 3) 0.45 (<= l 5) 0.7 :else 0.85)))
(defn- spec-cap
  "Ceiling on colour SPECIFICITY (the blur→raw blend t) per level — PROGRESSIVE
   COLOUR refinement: the coherence-driven t formula would let a base daub on a
   strong edge paint one raw pixel's colour across its whole σ (a blotch). Broad
   layers stay AVERAGED, mid layers go halfway, and full specificity arrives only
   with the fine detail levels — turning detail down degrades to a soft averaged
   underpainting, never to misplaced specific colour."
  [lvl]
  (let [l (long lvl)] (cond (<= l 1) 0.35 (<= l 3) 0.7 :else 1.0)))
(defn- stroke-len-frac
  "The Stroke slider as stroke LENGTH: scales the chain step. 2.5 (default) = 1.0."
  [stroke]
  (+ 0.4 (* 0.24 (double stroke))))


(defn layer-params
  "Pure per-level placement parameters — THE SHARED SPEC for the CPU loop
   (layered-means) and the GPU generation pass, so both enumerate the same cells.
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
        ;; base layer overlaps heavily (spacing 0.72×stdev ⇒ full coverage); finer layers are
        ;; sparser accents (the base fills behind them, so gaps between fine strokes don't
        ;; matter). Overlap is FIXED, so coverage never depends on the budget. Fine seeds are
        ;; √segs(lvl) sparser than dabs: each seed traces a segs(lvl)-segment stroke that
        ;; COVERS the along-edge span, so the total segment count (and thus the budget scale —
        ;; the stroke WIDTH) stays what single dabs cost. Without this the ×segs budget term
        ;; would fatten every stroke and smear the detail the chains exist for.
        ;; fine spacing also scales with √(stroke length): the seed grid assumes each
        ;; chain spans ~segs·step of edge — short strokes (low Stroke) must pack
        ;; denser or they render as sparse isolated pearls; long strokes space out.
        ;; At the default Stroke (2.5) the factor is exactly 1.
        ;; base overlap 0.65 (was 0.72): hash-random placement has gap variance a
        ;; lattice doesn't; slightly tighter spacing keeps coverage airtight.
        ;; the fine tier (lvl≥4) packs TIGHTER (0.7 vs 1.25): its levels overlap each
        ;; other AND the mid tier, so detailed areas get many strokes mixing into one
        ;; surface — and its liner strokes die early at colour boundaries (dry-out),
        ;; so seeds must overlap enough that successive lines hand off without the
        ;; contour breaking into stitched dashes.
        overlap (fn [lvl] (let [l (long lvl)]
                            (cond (zero? l) 0.65
                                  (<= l 3)  (* 1.25 (Math/sqrt (* (double (seg-count lvl)) slen)))
                                  :else     (* 0.7 (Math/sqrt (* (double (seg-count lvl)) slen))))))
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
        smul    (fn [lvl] (if (<= (long lvl) 1) 1.0 (tier-mul tier-muls lvl)))
        ;; final nominal size floored at ~a pixel AFTER the tier multiplier: a tier
        ;; dial at 0.4 must make its layer finer, never reduce it to sub-pixel dust —
        ;; dusted mid/fine layers punch the gradation ladder out of the painting and
        ;; leave raw base-to-line transitions (Mid/Fine turned left looked worse).
        nsize   (fn [lvl] (max 0.7 (* (smul lvl) (lsize lvl))))
        ;; tier multipliers scale size AND spacing together (constant overlap), so
        ;; each tier's density rebalances through the budget automatically.
        sp-of   (fn [lvl scale] (* (overlap lvl) scale
                                   (if (<= (long lvl) 1) (* bmin (lsize lvl)) (nsize lvl))))
        ;; budget: total(scale)=K/scale² ⇒ smallest scale≥1 that fits under the working
        ;; budget. Fine-level seeds emit segs(lvl) SEGMENTS each (a traced brush stroke),
        ;; so their term is multiplied accordingly — the budget counts splats, not seeds.
        ;; Each fine level estimates its survivor fraction on ITS OWN map (aggregate vs
        ;; sharp fine-band — the same map it thresholds against when placing).
        lvl-frac (fn [lvl] (detail-fraction dmap (level-map-kind lvl) (thresh lvl)))
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
                 (/ (* (double (seg-count lvl)) f area) (* sp sp))))
        Kc (reduce + 0.0 (map k-of (range 0 (min nlev 4))))
        Kf (if (> nlev 4) (reduce + 0.0 (map k-of (range 4 nlev))) 0.0)
        scale-c (max 1.0 (Math/sqrt (/ Kc budget)))
        scale-f (if (<= nlev 4)
                  scale-c
                  (let [rem (max (* 0.15 budget) (- budget (/ Kc (* scale-c scale-c))))]
                    (max scale-c (Math/sqrt (/ Kf rem)))))
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
        levels (loop [lvl (dec nlev) off 0 out []]
                 (if (< lvl 0)
                   out
                   (let [lsc (scale-of lvl)
                         ssz (* lsc (nsize lvl))
                         sp  (sp-of lvl lsc)
                         nx  (long (Math/ceil (/ area (* sp sp))))
                         ny  1]
                     (recur (dec lvl) (+ off (* nx ny))
                            (conj out {:lvl lvl :ssz ssz :sp sp :th (thresh lvl)
                                       :nx nx :ny ny :offset off
                                       :segs (seg-count lvl) :stepf (step-frac lvl)
                                       :bendf (bend-frac lvl) :map-kind (level-map-kind lvl)
                                       :traw (raw-floor lvl)})))))]
    {:nlev nlev :warp warp :scale scale :levels levels
     :total (reduce + 0 (map (fn [{:keys [nx ny]}] (* nx ny)) levels))}))

(declare sample-fields)

(defn- edge-snap
  "Move a fine-stroke position onto the local EDGE RIDGE: sample edge strength at
   p and at ±h across the local tangent, fit a parabola, step to its peak (clamped
   to ±h). Seeds scatter across a thin line by the placement noise; without the
   snap each stroke traces PARALLEL to the line at its own offset and a crisp
   1px line renders as a wobbly multi-strand braid.
   `gain` damps the corrector: the edge map is texel-quantized, so the full
   parabola step jitters. The SEED snap uses 0.65 (converge onto the ridge);
   liner-chain STEPS use a gentler gain — a strong per-step lateral corrector
   fought the direction momentum and scalloped thin traced lines into wobble."
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
  [nf dmap lvl x y ssz D sn tn dirsign curvature stroke hd wd segs stepf bendf hb traw sgate blur-px iw ih lth melt]
  (if (zero? (long lvl))
    (let [[th coh] (sample-fields nf x y)]
      ;; melted bokeh daubs ROUND OFF (coherence → 0 kills the elongation and pulls
      ;; the colour toward the smooth blur): an elongated needle on a soft gradient
      ;; always reads as a directional streak, however faithful its colour.
      [[x y ssz D sn tn 1.0 th (* coh (- 1.0 (double melt))) hb x y traw (spec-cap lvl)]])
    (let [kmax (dec (long segs))
          lal  (level-alpha lvl)
          ;; fine strokes snap onto the edge ridge at the seed and after every step
          ;; (predictor: tangent step; corrector: ridge snap) — the stroke GLUES to
          ;; the line it is painting instead of braiding beside it.
          snap? (>= (long lvl) 2)
          ;; liner chains correct gently mid-stroke (see edge-snap's gain doc)
          sgain (if (>= (long lvl) 4) 0.35 0.65)
          ;; the GEOMETRY snaps to the ridge, but the COLOUR samples the pre-snap
          ;; position: on-ridge colour is the two sides' mix — darker than either —
          ;; and painted along a silhouette it reads as a drawn OUTLINE. Pre-snap
          ;; seeds land on one side or the other, so contour strokes interleave the
          ;; two sides' actual colours and the edge blends like meeting paint.
          cx0 x cy0 y
          [x y] (if snap? (edge-snap dmap nf x y 1.75 hd wd 0.65) [x y])
          ;; IMPASTO meeting line: bodied liner strokes keep to THEIR side of the
          ;; ridge — the centre backs off ~half a width toward the colour-sample
          ;; side, so the two sides' opaque paints MEET at the edge instead of
          ;; alternating across it (on-ridge bodied strokes scalloped every
          ;; silhouette into light/dark beads). side=0 (no ridge) leaves the
          ;; stroke untouched.
          side (if (and snap? (>= (long lvl) 4))
                 (let [[th0 _] (sample-fields nf x y)
                       snx (- (Math/sin th0)) sny (Math/cos th0)
                       d   (+ (* (- cx0 x) snx) (* (- cy0 y) sny))]
                   (cond (> d 1e-9) 1.0 (< d -1e-9) -1.0 :else 0.0))
                 0.0)
          offset (fn [ox oy]
                   (if (zero? (double side))
                     [ox oy]
                     (let [[th0 _] (sample-fields nf ox oy)
                           snx (- (Math/sin th0)) sny (Math/cos th0)]
                       [(max 0.0 (min hd (+ ox (* (double side) 0.55 ssz snx))))
                        (max 0.0 (min wd (+ oy (* (double side) 0.55 ssz sny))))])))
          [x y] (offset x y)
          ;; the side sign relative to the stroke's MOTION frame: at the head the
          ;; motion perpendicular is dirsign·(field normal), so side·dirsign along
          ;; (−dy,dx) reproduces the head offset — and stays consistent through
          ;; field sign flips that would wobble a per-step θ resample.
          sidem (* (double side) (double dirsign))
          [hr hg hb0] (sample-arr blur-px iw ih x y)]
      (loop [k 0 px (double x) py (double y) dxp 0.0 dyp 0.0 fade 1.0 acc []]
        (if (or (> k kmax) (< fade 0.15))
          acc
          (let [;; TWO-TIER dry-out. Gradual drift DRIES the brush (×0.4) — abrupt
                ;; ends left broken dashes, and liner strokes (lvl≥4) tolerate a
                ;; little more drift (0.3) since on-ridge blur shifts along a lit
                ;; contour. But a LARGE mismatch (>0.45) means the stroke has
                ;; EXITED its colour region — a chain tangentially escaping a
                ;; curved silhouette would paint its dark brush-load into the
                ;; background (the halo of black lines hovering over hair) — so
                ;; the painter LIFTS the brush: fade 0, and the guard below stops
                ;; the chain BEFORE this segment is emitted.
                fade (if (pos? k)
                       (let [[br bg bb] (sample-arr blur-px iw ih px py)
                             dmx (max (Math/abs (- br hr)) (Math/abs (- bg hg)) (Math/abs (- bb hb0)))]
                         (cond (> dmx 0.45) 0.0
                               (> dmx (if (>= (long lvl) 4) 0.3 0.22)) (* fade 0.4)
                               :else fade))
                       fade)
                [th coh] (sample-fields nf px py)
                ;; follow the line only while there IS a line: when local coherence
                ;; collapses (busy texture, letter junctions) the liner stroke runs
                ;; dry fast — long chains wandering through dense detail smear it.
                ;; …but a strong edge under the brush keeps the line alive: real
                ;; ink lines push THROUGH junctions (glasses frame crossing a brow),
                ;; where coherence dips while edge energy stays high.
                fade (if (and (pos? k) (>= (long lvl) 4) (< coh 0.35)
                              (< (wavelet/edge-at dmap px py) 0.5))
                       (* fade 0.5)
                       fade)
                ;; LINE-HOLD: a liner stroke exists to trace the fine structure it
                ;; was seeded on. When the sharp fine-band map under the brush falls
                ;; below the level's own placement threshold, the stroke has WALKED
                ;; OFF its line — a chain escaping a silhouette tangentially would
                ;; drag its bodied paint into the featureless background (the ghost
                ;; tendrils around every contour) — so the painter lifts the brush.
                fade (if (and (pos? k) (>= (long lvl) 4))
                       (let [mv (* (wavelet/sharp-at dmap px py)
                                   (+ 0.25 (* 0.75 (double sgate))))]
                         (cond (< mv (* 0.35 (double lth))) 0.0
                               (< mv (* 0.7  (double lth))) (* fade 0.5)
                               :else fade))
                       fade)]
           (if (< fade 0.15)
            acc                                     ; brush lifted — emit nothing
            (let [t   (/ (double k) (double kmax))
                ;; IMPASTO body: ON a strong edge the fine liner strokes carry nearly
                ;; full paint — the contour is defined by opaque thin lines whose soft
                ;; shoulders blend, not by translucent glazes that let the mixed-colour
                ;; underpainting bleed through as a halo. Off-edge texture strokes keep
                ;; the light glaze and mix with the layers beneath.
                ;; the body follows the SURROUNDING detail density too: a bodied
                ;; line at full opacity on soft ground reads as pen-on-watercolour;
                ;; sparse-detail areas get gentler, more gradual contour marks.
                body (* (if (>= (long lvl) 4)
                          (min 1.0 (max 0.0 (/ (- (wavelet/edge-at dmap px py) 0.25) 0.45)))
                          0.0)
                        (+ 0.4 (* 0.6 (double sgate))))
                lal2 (+ lal (* (- 0.9 lal) body))
                ;; BOTH-ENDS taper: a quick lift-on at the head (the brush lands thin
                ;; and light, swells to full over the first ~18%) on top of the existing
                ;; longer dry-out at the tail — so the mark tapers at BOTH ends like a
                ;; real brushstroke, not just the tail. smoothstep = the same cubic the
                ;; GPU's smoothstep() mirrors. The LINER tier (lvl≥4) keeps only a hint
                ;; of it: a thin line is drawn by several overlapping chains handing
                ;; off, and strong per-chain taper turns the handoffs into a lumpy
                ;; string of tadpoles instead of one continuous rod.
                hw  (let [u (min 1.0 (/ t 0.18)) s (* u u (- 3.0 (* 2.0 u)))]
                      (if (>= (long lvl) 4) (+ 0.8 (* 0.2 s)) (+ 0.55 (* 0.45 s))))
                ha  (let [u (min 1.0 (/ t 0.15)) s (* u u (- 3.0 (* 2.0 u)))]
                      (if (>= (long lvl) 4) (+ 0.75 (* 0.25 s)) (+ 0.5 (* 0.5 s))))
                sz  (* ssz (- 1.0 (* 0.45 t (Math/sqrt t))) hw)  ; width tapers at both ends
                al  (* lal2 fade (- 1.0 (* 0.65 t t)) ha)        ; alpha: lift-on × glaze × dry-out
                ;; the brush-load RE-MIXES with the canvas as the stroke travels:
                ;; the colour-sample point slides up to 35% from the head toward
                ;; the current position, so long strokes grade into their
                ;; surroundings instead of carrying one colour to a hard break.
                ;; MELT (broad tier, bokeh): at high Broad the flat-region chains
                ;; re-mix much harder — a long chain carrying one brush-load across
                ;; a smooth gradient reads as a feathery streak on the wash; melted
                ;; strokes keep re-loading the local colour and disappear into it.
                wsl (cond (>= (long lvl) 4)       (* 0.35 t)
                          (pos? (double melt))    (* 0.85 (double melt) t)
                          :else                   0.0)
                acc (conj acc [px py sz D sn tn al th (* coh (- 1.0 (double melt))) hb
                               (+ cx0 (* wsl (- px cx0))) (+ cy0 (* wsl (- py cy0)))
                               traw (spec-cap lvl)])
                ;; step: along the local tangent, sign-continuous with the previous step,
                ;; bent by low-frequency Perlin scaled by this LEVEL's curvature share —
                ;; broad strokes curl freely, fine marks stay faithful to the edge.
                ;; the Perlin bend is GATED by coherence: a straight, strongly
                ;; oriented edge (coh→1) is traced straight — wobble belongs to
                ;; flow regions, not to lines.
                bend (* (double curvature) 0.9 (double bendf) (- 1.0 (* 0.7 coh))
                        (- (noise/noise2 (* 0.05 px) (* 0.05 py)) 0.5))
                cb (Math/cos bend) sb (Math/sin bend)
                dx0 (Math/cos th) dy0 (Math/sin th)
                sgn (if (zero? k)
                      (double dirsign)
                      (if (neg? (+ (* dx0 dxp) (* dy0 dyp))) -1.0 1.0))
                dx1 (* sgn dx0) dy1 (* sgn dy0)
                dx (- (* cb dx1) (* sb dy1)) dy (+ (* sb dx1) (* cb dy1))
                ;; DIRECTION MOMENTUM: a hand-pulled stroke has inertia — re-deciding
                ;; direction from the noisy field every step turns thin traced lines
                ;; wavy. Liner strokes carry 65% of the previous step's direction and
                ;; plow straight through junctions where the field goes incoherent.
                [dx dy] (if (and (>= (long lvl) 4) (pos? k))
                          (let [mx (+ (* 0.35 dx) (* 0.65 dxp))
                                my (+ (* 0.35 dy) (* 0.65 dyp))
                                ml (Math/sqrt (+ (* mx mx) (* my my)))]
                            (if (> ml 1e-6) [(/ mx ml) (/ my ml)] [dx dy]))
                          [dx dy])
                ;; the Stroke slider is stroke LENGTH: it scales the chain step (the
                ;; curve-following extent), not the segment ellipse. 2.5 (default) = 1.
                L  (* ssz (double stepf) (stroke-len-frac stroke))]
            (let [nx0 (max 0.0 (min hd (+ px (* L dx))))
                  ny0 (max 0.0 (min wd (+ py (* L dy))))
                  [nx1 ny1] (if snap? (edge-snap dmap nf nx0 ny0 1.75 hd wd sgain) [nx0 ny0])
                  ;; side offset along the stroke's OWN motion perpendicular — the
                  ;; path is a stable frame; re-sampling θ at every step let field
                  ;; noise wobble the offset into a wavy line.
                  [nx2 ny2] (if (zero? (double side))
                              [nx1 ny1]
                              [(max 0.0 (min hd (+ nx1 (* sidem 0.55 ssz (- dy)))))
                               (max 0.0 (min wd (+ ny1 (* sidem 0.55 ssz dx))))])]
              (recur (inc k) nx2 ny2 dx dy fade acc))))))))))

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
  [dmap nf detail size variation curvature stroke tier-muls count H W blur-px]
  (let [hd   (double (dec (long H))) wd (double (dec (long W)))
        iw   (long W) ih (long H)
        deff (fn [D] (min 1.0 (* (double detail) (double D) 2.2)))
        rr   (/ (double H) 24.0)                    ; subjectness tap radius
        bmul (double (nth tier-muls 0))
        bmin (min 1.0 bmul)
        {:keys [warp levels]} (layer-params dmap detail size variation curvature stroke tier-muls count H W)]
    (persistent!
      (reduce
        (fn [acc [idx {:keys [lvl ssz sp th nx ny segs stepf bendf map-kind traw]}]]
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
                              mloc  (+ 1.0 (* (- bmul 1.0) (- 1.0 sabs)))
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
                              bgate (- 1.0 (* (min 1.0 (max 0.0 (/ (- bmul 1.0) 1.5)))
                                              (- 1.0 (min 1.0 (/ sabs 0.35)))))
                              gain  (if (>= (long lvl) 2) (* (+ 0.25 (* 0.75 sgate)) bgate) 1.0)
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
                                            (let [fl (nth levels (dec (long idx)))
                                                  ;; the finer level is always ≥2, so its
                                                  ;; claim carries the same subject gate
                                                  fdv (* (map-at dmap (:map-kind fl) cx cy)
                                                         (+ 0.25 (* 0.75 sgate)) bgate)]
                                              (>= fdv (* (:th fl)
                                                         (+ 0.75 (* 0.5 (hash01 (+ (* i 47) lvl) j 23)))))))]
                          (if (or thin?
                                  (and (pos? (long lvl)) (or claimed? (< (* dv gain) thd))))
                            (recur (inc j) acc)      ; thinned bokeh / not detailed enough
                            (let [;; bokeh-adaptive broad size: kept flat-region daubs grow ×m
                              ssz (if (<= (long lvl) 1) (* ssz mloc) ssz)
                              ;; hashed positions need no jitter — they ARE the noise
                              x  cx y cy
                              D  (deff dv)
                              ;; flat-region Perlin warp breaks any residual level lattice;
                              ;; detail strokes (D≈1) stay put → faithful edges.
                              aw (* warp (- 1.0 D) ssz)
                              x2 (if (< aw 0.2) x
                                   (+ x (* aw (noise/noise2 (* 0.06 x) (* 0.06 y)))))
                              y2 (if (< aw 0.2) y
                                   (+ y (* aw (noise/noise2 (+ 41.3 (* 0.06 x)) (+ 17.9 (* 0.06 y))))))
                              sn0 (- (hash01 (+ (* i 31) lvl) j 11) 0.5)
                              ;; size jitter applies at SEED level to the whole chain —
                              ;; segment size AND step together — so chains stay
                              ;; self-overlapping at any Variation (per-segment size
                              ;; jitter with a fixed step beaded strokes into dotted
                              ;; pearls). Broad levels keep 40% (base coverage).
                              ;; the shrink side is CLAMPED at 0.75: strokes jittered far
                              ;; below their level's size land at the bottom of the
                              ;; hardness ramp and render as isolated hard pearls along
                              ;; edges — variety comes from growing, not vanishing.
                              szf (max 0.75 (+ 1.0 (* variation sn0 (if (<= (long lvl) 1) 0.4 1.0))))
                              ;; near a strong edge the mid fill levels don't paint (their
                              ;; boundary-band chains ribbon mixed colour along silhouettes
                              ;; as a ghost veil) and base daubs SHRINK so their soft tails
                              ;; can't reach across the silhouette. Sensed over the stroke's
                              ;; FOOTPRINT (taps at 0.75·size), not just its centre.
                              Ev (if (<= (long lvl) 3)
                                   (edge-near dmap cx cy (* 0.75 ssz))
                                   (wavelet/edge-at dmap cx cy))
                              ;; MELT: how much a flat-region broad stroke should sink into
                              ;; the wash. Grows only past Broad 1.0 (below that, strokes
                              ;; are the wanted effect) and only where subjectness is low —
                              ;; the Broad slider at max makes bokeh strokes invisible while
                              ;; detailed regions keep their brushwork untouched.
                              melt (if (<= (long lvl) 1)
                                     (* (min 1.0 (max 0.0 (/ (- bmul 1.0) 1.5)))
                                        (- 1.0 sabs))
                                     0.0)
                              ;; tone jitter is scale-relative: broad fills keep 25% (full
                              ;; jitter banded smooth walls) and the FINEST marks keep 15%
                              ;; (alternating-tone bodied lines bead contours) — the
                              ;; visible mid-scale brushwork carries the painterly variety.
                              ;; Melted bokeh strokes mute it further: a wash has no
                              ;; per-stroke tone identity to show.
                              tn (* (let [l (long lvl)]
                                      (cond (<= l 1) (* 0.25 (- 1.0 melt)) (>= l 4) 0.15 :else 1.0))
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
                                        (stroke-segments nf dmap lvl
                                                         (max 0.0 (min hd x2)) (max 0.0 (min wd y2))
                                                         ;; fat strokes shrink near edges so soft
                                                         ;; tails can't cross the silhouette; the
                                                         ;; fine liner strokes ARE the edge's own
                                                         ;; paint and keep their size. The BASE is
                                                         ;; the COVERAGE layer: it shrinks gently
                                                         ;; (≥0.75×, spacing still seals) so paint
                                                         ;; always reaches the boundary — turning
                                                         ;; detail down falls back to soft averaged
                                                         ;; edges, never to an unpainted moat.
                                                         (* ssz szf (- 1.0 (* (cond (zero? (long lvl)) 0.25
                                                                                    (<= (long lvl) 3) 0.45
                                                                                    :else 0.1)
                                                                              Ev)))
                                                         D 0.0 tn ds curvature stroke hd wd
                                                         segs stepf bendf
                                                         (if (<= (long lvl) 1) 1.0 0.0)
                                                         ;; fine colour rawness follows the local
                                                         ;; detail density — a crisp raw mark never
                                                         ;; pops at full contrast on soft ground
                                                         (if (>= (long lvl) 4)
                                                           (* traw (+ 0.6 (* 0.4 sgate)))
                                                           traw)
                                                         sgate blur-px iw ih th melt))]
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
  [x y csz dlev theta coherence snoise tnoise blur-rgb raw-rgb stroke variation contrast traw tcap]
  (let [coh (+ min-coh (* (- 1.0 min-coh) coherence))
        e   (+ 1.0 (* (min (double stroke) 1.5) coh (+ 0.25 (* 0.75 (double dlev)))))
        se  (Math/sqrt e)
        s0  (* csz (+ 1.0 (* variation 0.5 (* 2.0 snoise))))
        sx  (* s0 se)                 ; long axis along θ
        sy  (/ s0 se)                 ; short axis across the stroke
        ;; t is FLOORED by the level's rawness (traw) and CEILINGED by its
        ;; specificity cap (tcap) — the progressive colour ladder: broad layers
        ;; averaged, fine layers specific, whatever coherence says.
        t   (min (double tcap)
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
                size-broad size-mid size-fine]
         :or   {count 6000 size 3.0 stroke 2.0 detail 0.6 variation 0.5 curvature 0.5
                opacity 0.9 contrast 1.0 background 0.0
                size-broad 1.0 size-mid 1.0 size-fine 1.0}} controls
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
        nf         (or (:noise-fields image) (prep-noise sfield))
        segments   (layered-means dmap nf detail size variation curvature stroke
                                  [(double size-broad) (double size-mid) (double size-fine)]
                                  n height width blur-px)
        ;; each segment carries its sampled fields + taper alpha (stroke-segments did the
        ;; tracing); hand off to the pure `splat-record` math shared with the GPU.
        splats     (vec
                     (for [[x y csz dlev sn tn alpha theta coherence hb hx hy traw tcap] segments
                           :let [blur-rgb (sample-arr (if (and hb (pos? (double hb))) blurh-px blur-px)
                                                      width height hx hy)
                                 raw-rgb  (sample-arr raw-px width height hx hy)]]
                       (assoc (splat-record x y csz dlev theta coherence sn tn
                                            blur-rgb raw-rgb stroke variation contrast
                                            (or traw 0.0) (or tcap 1.0))
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
