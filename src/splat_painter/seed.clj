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
(def ^:private splat-budget 200000)

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

;; Fine-level seeds don't place one dab — they trace a BRUSH STROKE: a chain of
;; gaussian segments stepped along the orientation field. Stroke behaviour is
;; SCALE-RELATIVE (the coarse-to-fine pass adjusts its parameters to the scale it
;; is painting at): the broadest detail level lays long, freely-curving strokes;
;; each finer level makes shorter, straighter, more precise marks — and the two
;; finest levels read the SHARP fine-band detail map (wavelet/sharp-at), so they
;; land on (and preserve) text/eye-scale structure the smoothed aggregate blurs.
(defn- seg-count "segments per stroke at placement level" [lvl]
  (let [l (long lvl)] (cond (zero? l) 1 (== l 1) 6 (== l 2) 4 (== l 3) 3 :else 2)))
(defn- step-frac "step length as a fraction of the level stdev" [lvl]
  (let [l (long lvl)] (cond (== l 1) 1.1 (== l 2) 0.9 (== l 3) 0.75 (== l 4) 0.6 (== l 5) 0.5 :else 0.4)))
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
   through so detail builds on the underpainting instead of scratching over it."
  [lvl]
  (let [l (long lvl)] (cond (<= l 1) 1.0 (<= l 3) 0.9 :else 0.75)))

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
        overlap (fn [lvl] (if (zero? (long lvl))
                            0.65
                            (* 1.25 (Math/sqrt (* (double (seg-count lvl)) slen)))))
        ;; tier multipliers scale size AND spacing together (constant overlap), so
        ;; each tier's density rebalances through the budget automatically.
        sp-of   (fn [lvl scale] (* (overlap lvl) scale (tier-mul tier-muls lvl)
                                   (/ smax (Math/pow 2.0 (double lvl)))))
        ;; budget: total(scale)=K/scale² ⇒ smallest scale≥1 that fits under the working
        ;; budget. Fine-level seeds emit segs(lvl) SEGMENTS each (a traced brush stroke),
        ;; so their term is multiplied accordingly — the budget counts splats, not seeds.
        ;; Each fine level estimates its survivor fraction on ITS OWN map (aggregate vs
        ;; sharp fine-band — the same map it thresholds against when placing).
        lvl-frac (fn [lvl] (detail-fraction dmap (level-map-kind lvl) (thresh lvl)))
        ;; SUBDIVISION: a cell claimed by a finer level is NOT painted by coarser
        ;; detail levels (base always paints — coverage); the per-level costs below
        ;; use the exclusive fractions.
        ;; TWO-TIER budget: the levels that exist at moderate Detail (0-3) get the
        ;; scale THEY need; the added ultra-fine levels (4-5) fit their own scale into
        ;; the REMAINING budget. A single uniform scale let the finest levels' huge
        ;; demand fatten the base ×2.6+ as Detail rose — maxing Detail made everything
        ;; coarser. Now raising Detail leaves the broad/mid painting untouched and
        ;; adds fine accents on top: monotone by construction.
        k-of (fn [lvl]
               (let [f (cond
                         (zero? (long lvl))        1.0
                         (< (long lvl) (dec nlev)) (max 0.0 (- (lvl-frac lvl) (lvl-frac (inc (long lvl)))))
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
                         ssz (* lsc (tier-mul tier-muls lvl) (/ smax (Math/pow 2.0 (double lvl))))
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
   1px line renders as a wobbly multi-strand braid."
  [dmap nf x y h hd wd]
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
        [(max 0.0 (min hd (+ x (* nx h d))))
         (max 0.0 (min wd (+ y (* ny h d))))]))))

(defn- map-at
  "Sample the placement map matched to a level's scale."
  [dmap kind x y]
  (cond
    (= kind :sharp) (wavelet/sharp-at dmap x y)
    (= kind :mid)   (wavelet/mid-at dmap x y)
    :else           (wavelet/detail-at dmap x y)))

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
  [nf dmap lvl x y ssz D sn tn dirsign curvature stroke hd wd segs stepf bendf hb traw blur-px iw ih]
  (if (zero? (long lvl))
    (let [[th coh] (sample-fields nf x y)]
      [[x y ssz D sn tn 1.0 th coh hb x y traw]])
    (let [kmax (dec (long segs))
          lal  (level-alpha lvl)
          ;; fine strokes snap onto the edge ridge at the seed and after every step
          ;; (predictor: tangent step; corrector: ridge snap) — the stroke GLUES to
          ;; the line it is painting instead of braiding beside it.
          snap? (>= (long lvl) 2)
          ;; the GEOMETRY snaps to the ridge, but the COLOUR samples the pre-snap
          ;; position: on-ridge colour is the two sides' mix — darker than either —
          ;; and painted along a silhouette it reads as a drawn OUTLINE. Pre-snap
          ;; seeds land on one side or the other, so contour strokes interleave the
          ;; two sides' actual colours and the edge blends like meeting paint.
          cx0 x cy0 y
          [x y] (if snap? (edge-snap dmap nf x y 1.75 hd wd) [x y])
          [hr hg hb0] (sample-arr blur-px iw ih x y)]
      (loop [k 0 px (double x) py (double y) dxp 0.0 dyp 0.0 fade 1.0 acc []]
        (if (or (> k kmax) (< fade 0.15))
          acc
          (let [;; the stroke FADES when the canvas stops matching its brush-load —
                ;; a brush running dry — instead of breaking dead: abrupt ends left
                ;; broken dashes with gaps around busy detail (eyes, fur ticking).
                fade (if (and (pos? k)
                              (let [[br bg bb] (sample-arr blur-px iw ih px py)]
                                (> (max (Math/abs (- br hr)) (Math/abs (- bg hg)) (Math/abs (- bb hb0)))
                                   0.22)))
                       (* fade 0.4)
                       fade)
                [th coh] (sample-fields nf px py)
                t   (/ (double k) (double kmax))
                sz  (* ssz (- 1.0 (* 0.45 t (Math/sqrt t))))     ; width tapers to the tip
                al  (* lal fade (- 1.0 (* 0.65 t t)))            ; taper × glaze × dry-out
                acc (conj acc [px py sz D sn tn al th coh hb cx0 cy0 traw])
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
                ;; the Stroke slider is stroke LENGTH: it scales the chain step (the
                ;; curve-following extent), not the segment ellipse. 2.5 (default) = 1.
                L  (* ssz (double stepf) (stroke-len-frac stroke))]
            (let [nx0 (max 0.0 (min hd (+ px (* L dx))))
                  ny0 (max 0.0 (min wd (+ py (* L dy))))
                  [nx1 ny1] (if snap? (edge-snap dmap nf nx0 ny0 1.75 hd wd) [nx0 ny0])]
              (recur (inc k) nx1 ny1 dx dy fade acc))))))))

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
                        (let [;; each level reads the map matched to ITS scale: the finest
                              ;; levels use the sharp fine-band map so they land on (and
                              ;; preserve) small structure the smoothed aggregate blurs away.
                              ;; The cutoff is DITHERED ±25% per seed — a hard threshold on
                              ;; a map oscillating around it dashes contours into beads.
                              dv (map-at dmap map-kind cx cy)
                              thd (* th (+ 0.75 (* 0.5 (hash01 (+ (* i 43) lvl) j 19))))
                              ;; SUBDIVISION: skip if the next-finer level (previous
                              ;; entry — levels are finest-first) claims this cell. The
                              ;; claim is DITHERED like the threshold so the handoff
                              ;; between scales is a gradual interleave, not a seam.
                              claimed? (and (pos? (long lvl)) (pos? (long idx))
                                            (let [fl (nth levels (dec (long idx)))
                                                  fdv (map-at dmap (:map-kind fl) cx cy)]
                                              (>= fdv (* (:th fl)
                                                         (+ 0.75 (* 0.5 (hash01 (+ (* i 47) lvl) j 23)))))))]
                          (if (and (pos? (long lvl)) (or claimed? (< dv thd)))
                            (recur (inc j) acc)      ; not detailed enough for this fine level
                            (let [;; hashed positions need no jitter — they ARE the noise
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
                              ;; can't reach across the silhouette.
                              Ev (wavelet/edge-at dmap cx cy)
                              ;; tone jitter is scale-relative: broad fills keep 40% (full
                              ;; jitter banded smooth walls) and the FINEST marks keep 30%
                              ;; (alternating-tone hard dabs bead edges into pearls) — the
                              ;; visible mid-scale brushwork carries the painterly variety.
                              tn (* (let [l (long lvl)]
                                      (cond (<= l 1) 0.25 (>= l 4) 0.3 :else 1.0))
                                    (- (hash01 (+ (* i 37) lvl) j 13) 0.5))
                              ds (if (< (hash01 (+ (* i 41) lvl) j 17) 0.5) 1.0 -1.0)
                              ;; keep centres in-bounds so no budget is wasted off-screen
                              ;; (edges stay covered by the splats' tails).
                              emitted (if (and (or (== (long lvl) 1) (== (long lvl) 2))
                                               (> Ev 0.45)
                                               ;; dithered: ~75% suppressed — a few mid
                                               ;; strokes still fill the edge band, so
                                               ;; fine contour strokes sit IN paint
                                               ;; instead of standing alone as outlines
                                               (< (hash01 (+ (* i 53) lvl) j 37) 0.75))
                                        []
                                        (stroke-segments nf dmap lvl
                                                         (max 0.0 (min hd x2)) (max 0.0 (min wd y2))
                                                         (* ssz szf (- 1.0 (* 0.45 Ev)))
                                                         D 0.0 tn ds curvature stroke hd wd
                                                         segs stepf bendf
                                                         (if (<= (long lvl) 1) 1.0 0.0)
                                                         traw blur-px iw ih))]
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
  [x y csz dlev theta coherence snoise tnoise blur-rgb raw-rgb stroke variation contrast traw]
  (let [coh (+ min-coh (* (- 1.0 min-coh) coherence))
        e   (+ 1.0 (* (min (double stroke) 1.5) coh (+ 0.25 (* 0.75 (double dlev)))))
        se  (Math/sqrt e)
        s0  (* csz (+ 1.0 (* variation 0.5 (* 2.0 snoise))))
        sx  (* s0 se)                 ; long axis along θ
        sy  (/ s0 se)                 ; short axis across the stroke
        t   (max (double traw)
                 (min 1.0 (max 0.0 (+ 0.15 (* 0.85 (max coherence (double dlev)))))))
        [br bg bb] blur-rgb [rr rg rb] raw-rgb
        color0 [(+ (* br (- 1.0 t)) (* rr t))
                (+ (* bg (- 1.0 t)) (* rg t))
                (+ (* bb (- 1.0 t)) (* rb t))]
        color-ac (if (== contrast 1.0) color0 (apply-contrast contrast color0))
        tone (+ 1.0 (* variation 0.15 (* 2.0 tnoise)))
        color (mapv (fn [c] (max 0.0 (min 1.0 (* c tone)))) color-ac)]
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
                     (for [[x y csz dlev sn tn alpha theta coherence hb hx hy traw] segments
                           :let [blur-rgb (sample-arr (if (and hb (pos? (double hb))) blurh-px blur-px)
                                                      width height hx hy)
                                 raw-rgb  (sample-arr raw-px width height hx hy)]]
                       (assoc (splat-record x y csz dlev theta coherence sn tn
                                            blur-rgb raw-rgb stroke variation contrast
                                            (or traw 0.0))
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
