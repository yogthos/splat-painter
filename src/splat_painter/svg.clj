(ns splat-painter.svg
  "Splat field → SVG. Pure: no GL, no I/O. Takes the same field map
   `splat-painter.seed/splat-field` returns (and that `splat-painter.gen/read-splats`
   reads back off the GPU) and returns an SVG document string.

   A splat is an anisotropic 2D gaussian: Σ = R·diag(sx²,sy²)·Rᵀ. Eigen-decomposing
   Σ gives an ellipse — centre, two radii, one rotation — which SVG can express
   exactly. What SVG has no primitive for is the gaussian FALLOFF across that
   ellipse, and that is the whole design problem here.

   THE FALLOFF. Three ways to get a per-splat colour with a soft profile, measured
   (rsvg-convert, 20k splats, 1000×1000) before picking:

     shared <mask> in objectBoundingBox units   exact colour, 1 def   ~100 s
     <radialGradient> per quantized colour      palette colour, N defs ~1.1 s
     flat ellipse at an iso-contour             exact colour, 0 defs   ~0.5 s

   Masks are out: every masked element forces the renderer into an offscreen
   buffer, and at splat counts that is 5 ms per element. A gradient CANNOT take its
   colour from the element referencing it — `currentColor` and `var()` in a <stop>
   both resolve against the gradient's own ancestors (verified in Chromium and
   librsvg), so one shared gradient can only ever paint one colour. Hence the
   default: quantize the field's colours to a median-cut palette and emit one
   gradient per (palette entry, hardness bucket) actually used. Paintings quantize
   well — the per-stroke tone/temperature jitter dithers the palette for free.

   THE OTHER 15×. Do NOT put `opacity` on a splat: the attribute is defined to
   composite the element as an isolated group, so renderers allocate a scratch
   surface per ellipse. `fill-opacity` is the same picture here (one shape, one
   fill, nothing to isolate) and it is 15× faster — 16 s → 1.1 s on the 20k
   benchmark. Per-splat alpha is exact; only the colour is quantized.

   NOT REPRODUCED. The app's look also carries a bristle-texture term in the
   fragment shader (streak/grain/ragged edge) and a final sharpen pass over the
   composite. Neither has a per-element SVG equivalent that survives the element
   count, so an export is the painting's GEOMETRY and COLOUR, not a pixel match of
   the pane. See `splat-painter.svgout` for the side-by-side."
  (:require [clojure.string :as str]
            [splat-painter.shader :as shader]))

;; ---------------------------------------------------------------- number output

;; table-driven rather than Math/pow — `fixed` runs a few times per splat.
(def ^:private pow10 [1 10 100 1000 10000 100000 1000000])
(def ^:private zeros ["" "0" "00" "000" "0000" "00000" "000000"])

(defn- fixed
  "`v` rounded to `dp` decimals (0–6), as the shortest string that means it: trailing
   zeros and a bare \".0\" dropped, and the leading zero of a fraction dropped too
   (\".85\", not \"0.85\" — a valid SVG number, and there is one on most elements in
   the file). `format` would be called several times per splat, so this exists to keep
   the emit loop off it."
  [^double v ^long dp]
  (let [m (nth pow10 dp)
        r (Math/round (* v (double m)))
        neg (neg? r)
        r (Math/abs r)
        i (quot r m)
        f (rem r m)
        sign (if neg "-" "")]
    (if (zero? f)
      (str sign i)
      (let [fs (str f)
            fs (str (nth zeros (- dp (count fs))) fs)
            fs (str/replace fs #"0+$" "")]
        (str sign (when-not (zero? i) i) "." fs)))))

(defn- hex2 [^double c]
  (let [v (long (max 0 (min 255 (Math/round (* 255.0 c)))))]
    (str (when (< v 16) "0") (Integer/toString v 16))))

(defn- hex-rgb [[r g b]] (str "#" (hex2 r) (hex2 g) (hex2 b)))

;; ---------------------------------------------------------------- splat geometry

(defn ellipse
  "Σ (as [c00 c01 c10 c11] in (row,col) space) → the SVG ellipse that carries it:
   {:rx :ry :deg :sig}. rx/ry are ONE stdev along the major/minor axis (the caller
   scales by the gaussian extent it wants to draw out to); `deg` is the rotation
   about the centre, in SVG's frame where +x is the image's COLUMN axis and +y its
   ROW axis. `sig` = det(Σ)^¼, the scalar stdev the render shaders key hardness on."
  [[c00 c01 _ c11]]
  (let [tr   (* 0.5 (+ c00 c11))
        disc (Math/sqrt (max (+ (* 0.25 (- c00 c11) (- c00 c11)) (* c01 c01)) 0.0))
        l1   (+ tr disc)
        l2   (max (- tr disc) 0.0)
        ;; same robust eigenvector as the vertex shader: an exactly axis-aligned
        ;; stroke has c01 == 0.0, and then (l1-c11, c01) is the zero vector.
        [er ec] (let [a [(- l1 c11) c01]]
                  (if (< (+ (* (a 0) (a 0)) (* (a 1) (a 1))) 1e-12)
                    (let [b [c01 (- l1 c00)]]
                      (if (< (+ (* (b 0) (b 0)) (* (b 1) (b 1))) 1e-12) [1.0 0.0] b))
                    a))]
    {:rx  (Math/sqrt (max l1 0.0))
     :ry  (Math/sqrt l2)
     ;; the eigenvector is (row, col); SVG's x is col, y is row.
     :deg (Math/toDegrees (Math/atan2 er ec))
     :sig (Math/sqrt (Math/sqrt (max (- (* c00 c11) (* c01 c01)) 1e-8)))}))

(defn hardness
  "The exponent in α = alpha·opacity·exp(−pdf^hard) for a splat of stdev `sig` and
   subjectness `detail` — the same three steps the render vertex shader runs
   (size→hardness smoothstep, the detail floor, the sub-pixel antialias ease)."
  [^double sig ^double sig-min ^double sig-max ^double detail
   ^double hard-sharp ^double hard-soft]
  (let [t  (max 0.0 (min 1.0 (/ (- sig sig-min) (max (- sig-max sig-min) 1e-4))))
        ts (* t t (- 3.0 (* 2.0 t)))
        h  (+ (* hard-sharp (- 1.0 ts)) (* hard-soft ts))
        h  (+ 1.0 (* (- h 1.0) (shader/detail-hardness-scale detail)))]
    (+ 1.0 (* (- h 1.0) (max 0.0 (min 1.0 (/ sig 2.5)))))))

(defn profile-alpha
  "Peak-normalized gaussian alpha at Mahalanobis distance `r` (in stdevs) for
   exponent `hard` — exp(−(r²/2)^hard), the render shaders' profile."
  [^double r ^double hard]
  (Math/exp (- (Math/pow (* 0.5 r r) hard))))

;; ---------------------------------------------------------------- palette

(defn- chan-range [colors ch]
  (loop [cs (seq colors) lo 255 hi 0]
    (if cs
      (let [v (long (nth (first cs) ch))]
        (recur (next cs) (min lo v) (max hi v)))
      [lo hi (- hi lo)])))

(defn- box-of
  "Wrap colour rows in a box, caching its widest channel and its split PRIORITY.
   Priority is population × spread, not population alone. Population-first is the
   textbook median cut and it loses RARE SATURATED paint: on a grey-and-blue cat the
   few green eye strokes never win a split against the fur, so they land in a grey box
   and the eyes come out grey. Multiplying in the longest channel range promotes a
   small box that sits far from everything else — exactly the accent colour a painting
   cannot afford to lose. A single-colour box gets priority 0 and is never picked."
  [rows]
  (let [[_ _ dr] (chan-range rows 0)
        [_ _ dg] (chan-range rows 1)
        [_ _ db] (chan-range rows 2)
        d (max dr dg db)]
    {:rows rows
     :ch   (cond (and (>= dr dg) (>= dr db)) 0 (>= dg db) 1 :else 2)
     :prio (* (count rows) d)}))

(defn- split-box [{:keys [rows ch]}]
  (let [sorted (vec (sort-by #(nth % ch) rows))
        half   (quot (count sorted) 2)]
    [(box-of (subvec sorted 0 half)) (box-of (subvec sorted half))]))

(defn palette
  "Median-cut `colors` (8-bit [r g b idx] rows) into at most `n` boxes. Returns
   {:entries [[r g b]…] :index {splat-idx → entry}} — every input colour is assigned
   by CONSTRUCTION (it is in exactly one box), so there is no nearest-neighbour
   search over the field afterwards."
  [colors ^long n]
  (loop [boxes [(box-of (vec colors))]]
    (let [i (when (< (count boxes) n)
              (first (sort-by #(- (:prio (nth boxes %))) (range (count boxes)))))]
      (if (and i (pos? (:prio (nth boxes i))))
        (recur (into (into (subvec boxes 0 i) (split-box (nth boxes i)))
                     (subvec boxes (inc i))))
        (let [entries (mapv (fn [{:keys [rows]}]
                              (let [c (count rows)
                                    [r g b] (reduce (fn [[r g b] row]
                                                      [(+ r (long (row 0)))
                                                       (+ g (long (row 1)))
                                                       (+ b (long (row 2)))])
                                                    [0 0 0] rows)]
                                [(/ (double r) (max c 1) 255.0)
                                 (/ (double g) (max c 1) 255.0)
                                 (/ (double b) (max c 1) 255.0)]))
                            boxes)
              index (persistent!
                      (reduce (fn [m [bi {:keys [rows]}]]
                                (reduce (fn [m row] (assoc! m (long (row 3)) bi)) m rows))
                              (transient {})
                              (map-indexed vector boxes)))]
          {:entries entries :index index})))))

;; ---------------------------------------------------------------- document

(def ^:private default-opts
  {:mode        :gradient  ; :gradient (soft, palette colour) | :flat (hard, exact colour)
   :fidelity    1.0        ; 1.0 = keep every splat; lower prunes, quantizes, rounds
   :hard-levels 3          ; distinct falloff profiles
   :stops       8          ; gradient stops per profile — 6 and 16 measure the same
   :extent      3.5        ; how many stdevs the ellipse spans, :gradient mode
   :flat-extent 1.2        ; iso-contour radius in stdevs, :flat mode
   :hard-sharp  1.7        ; the app's Hardness slider
   :hard-soft   1.0
   :cull-alpha  0.004      ; below half an 8-bit step, the splat is invisible
   :cull-stride 2          ; transmittance grid coarsening in the visibility pass
   :scale       1.0        ; width/height multiplier — the viewBox stays in image px
   :lift        1.0        ; the app's Lift present pass; 1.0 = no filter emitted
   :brightness  1.0})      ; the app's Brightness present pass; 1.0 = none
   ;; :colors, :cull-peak, :dp and :round default from :fidelity — see fidelity->opts

(defn fidelity->opts
  "What the Fidelity dial actually moves. One number, four knobs, all of them
   monotone, so the dial reads as a single quality/size trade:

     :cull-peak  the visibility threshold (see `keep-mask`) — the big one. This is
                 what deletes elements, and elements ARE the file size.
     :colors     the palette; fewer gradients in <defs> and coarser colour.
     :round      the rx/ry ratio at which a stroke is written as a <circle>.
     :dp         decimals on coordinates; 0 is half-pixel placement.

   1.0 is defined to be a no-op on every one of them (threshold 0, full palette,
   circles only when already round, full precision), so Fidelity 1 is exactly what
   the exporter produced before it had a dial."
  [fidelity]
  (let [f (max 0.0 (min 1.0 (double fidelity)))
        d (- 1.0 f)]
    {:cull-peak (* 0.35 d d)               ; 0 at 1.0, 0.35 at 0 — quadratic, gentle at the top
     :colors    (long (+ 24 (* 488 f f)))  ; 24..512
     :round     (+ 1.0 (* 0.45 d))         ; 1.0..1.45
     :dp        (if (< f 0.35) 0 1)}))

(defn- splat->draw
  "One splat → the drawing primitive, or nil if it cannot leave a mark."
  [{:keys [mean cov color alpha detail]} opts]
  (let [{:keys [sig-min sig-max hard-sharp hard-soft opacity cull-alpha]} opts
        {:keys [rx ry deg sig]} (ellipse cov)
        a (* (double (or alpha 1.0)) (double opacity))]
    (when (and (> a (double cull-alpha)) (> rx 1e-3))
      {:cx (double (nth mean 1))          ; SVG x is the image's column axis
       :cy (double (nth mean 0))
       :rx rx :ry ry :deg deg
       ;; the rotation as a unit vector too — the visibility pass evaluates the
       ;; profile per pixel and must not call cos/sin inside that loop
       :cos (Math/cos (Math/toRadians deg))
       :sin (Math/sin (Math/toRadians deg))
       :hard (hardness sig sig-min sig-max (double (or detail 1.0)) hard-sharp hard-soft)
       :alpha a
       :color color})))

;; ---------------------------------------------------------------- visibility

(defn- prof
  "Peak-normalized gaussian of the splat whose half-axes are `rx`,`ry` and whose long
   axis is the unit vector (`co`,`si`), at offset (`dx`,`dy`). Takes primitives, not the
   draw map: this runs once per covered pixel per splat, and reading `(:cx d)` out of a
   hash map in that loop cost more than everything else in the exporter put together."
  ^double [^double dx ^double dy ^double rx ^double ry ^double co ^double si ^double hard]
  (let [u (/ (+ (* dx co) (* dy si)) rx)
        v (/ (- (* dy co) (* dx si)) ry)
        pdf (* 0.5 (+ (* u u) (* v v)))]
    (Math/exp (- (if (< (Math/abs (- hard 1.0)) 0.01) pdf (Math/pow pdf hard))))))

(defn keep-mask
  "Which splats survive at peak-visibility threshold `thresh`. Returns
   {:keep <int-array, 1 = keep, indexed like `draws`> :residual <max leftover T>}.

   THE CRITERION is each splat's PEAK contribution to the finished image,
   max_p α(p)·T(p) — how strongly it shows through everything painted in front of it,
   at the one pixel where it shows most. That is the 3DGS pruning score (accumulated
   opacity × transmittance) with RadSplat's max in place of LightGaussian's sum, and
   the max matters enormously here: a sum is an AREA measure, so it ranks a 1px liner
   stroke below a barely-visible fat glaze and prunes exactly the detail the painting
   is made of. The max asks 'is this mark ever visible anywhere', which is the
   question. It is a real max over the footprint, not the value at the centre — a fat
   glaze can be hidden at its centre and still be the only paint on the patch its rim
   covers, and reading only the centre culled it and opened a hole there.

   COVERAGE IS SAFE BY CONSTRUCTION: a dropped splat does NOT consume transmittance.
   If pruning the strokes in front of a base daub leaves it as the only thing covering
   a patch, T there is still 1.0, its peak is its full alpha, and it cannot then be
   pruned itself. So this is one greedy front-to-back pass, not a global sort — a sort
   has no way to know that dropping A is what makes B load-bearing. `:residual` is the
   audit: the largest transmittance left anywhere when the pass ends, which is how much
   background shows through the worst pixel. It should stay at essentially zero.

   `stride` samples the transmittance on a coarser grid. T is an accumulation over many
   overlapping strokes and is smooth, so the peak survives the coarsening and the pass
   gets stride² cheaper. `extent` must match what the emitter draws, or the pass reads
   coverage the file will actually have as a hole.

   THEN A REPAIR PASS. The greedy rule bounds how much background can show but does not
   drive it to zero: a splat whose peak is just under the threshold gets dropped even
   where it was the last cover, and enough of those in one place is a hole. Left alone
   that reads as a systematic DARKENING — every hole shows the black clear, so the whole
   picture loses a couple of levels. So after pruning, walk the dropped splats back to
   front and re-instate any that still overlap uncovered ground. Transmittance is a
   product over the kept set, so re-instating in any order is exact."
  [draws ^long height ^long width ^double thresh ^long stride ^double extent]
  (let [n (count draws)
        keep (int-array n)]
    (dotimes [i n] (aset keep i 1))
    (if (<= thresh 0.0)
      {:keep keep :residual 0.0 :repaired 0}
      (let [s  (double (max 1 stride))
            gw (max 1 (long (Math/ceil (/ (double width) s))))
            gh (max 1 (long (Math/ceil (/ (double height) s))))
            ^doubles T (double-array (* gw gh))
            hole 0.03                     ; background this visible is a hole to repair
            ;; bbox of splat i on the transmittance grid
            box (fn [d]
                  (let [cx (double (:cx d)) cy (double (:cy d))
                        rad (* extent (double (:rx d)))]
                    [(max 0 (long (Math/floor (/ (- cx rad) s))))
                     (min (dec gw) (long (Math/ceil (/ (+ cx rad) s))))
                     (max 0 (long (Math/floor (/ (- cy rad) s))))
                     (min (dec gh) (long (Math/ceil (/ (+ cy rad) s))))]))
            deposit! (fn [d ^long lo-x ^long hi-x ^long lo-y ^long hi-y]
                       (let [cx (double (:cx d)) cy (double (:cy d))
                             rx (double (:rx d)) ry (double (:ry d))
                             co (double (:cos d)) si (double (:sin d))
                             hd (double (:hard d)) al (double (:alpha d))]
                         (loop [gy lo-y]
                           (when (<= gy hi-y)
                             (let [dy (- (* (+ gy 0.5) s) cy) row (* gy gw)]
                               (loop [gx lo-x]
                                 (when (<= gx hi-x)
                                   (let [a (* al (prof (- (* (+ gx 0.5) s) cx) dy rx ry co si hd))
                                         k (+ row gx)]
                                     (aset T k (* (aget T k) (- 1.0 a))))
                                   (recur (inc gx)))))
                             (recur (inc gy))))))]
        (dotimes [i (* gw gh)] (aset T i 1.0))
        (dotimes [i n]
          (let [d  (nth draws i)
                [lo-x hi-x lo-y hi-y] (box d)
                cx (double (:cx d)) cy (double (:cy d))
                rx (double (:rx d)) ry (double (:ry d))
                co (double (:cos d)) si (double (:sin d))
                hd (double (:hard d)) al (double (:alpha d))
                ;; O(1) ACCEPT: the profile is 1.0 at the centre, so alpha·T(centre) is
                ;; already a lower bound on the peak. Most strokes in a painting are
                ;; visible, and this clears them without touching their footprint twice.
                gcx (min (dec gw) (max 0 (long (/ cx s))))
                gcy (min (dec gh) (max 0 (long (/ cy s))))
                centre (* al (aget T (+ (* gcy gw) gcx)))
                peak (if (>= centre thresh)
                       centre
                       (loop [gy lo-y mx centre]
                         (if (> gy hi-y)
                           mx
                           (let [dy  (- (* (+ gy 0.5) s) cy)
                                 row (* gy gw)
                                 rmx (loop [gx lo-x m mx]
                                       (if (> gx hi-x)
                                         m
                                         (let [t (aget T (+ row gx))]
                                           ;; α ≤ al everywhere, so a cell whose T alone
                                           ;; cannot beat the running max needs no exp()
                                           (recur (inc gx)
                                                  (if (<= (* al t) m)
                                                    m
                                                    (max m (* al t (prof (- (* (+ gx 0.5) s) cx)
                                                                         dy rx ry co si hd))))))))]
                             (recur (inc gy) rmx)))))]
            (if (< peak thresh)
              (aset keep i 0)
              (deposit! d lo-x hi-x lo-y hi-y))))
        ;; repair: back to front, re-instate anything still sitting over bare ground
        (let [repaired
              (loop [i (dec n) fixed 0]
                (if (neg? i)
                  fixed
                  (if (== 1 (aget keep i))
                    (recur (dec i) fixed)
                    (let [d (nth draws i)
                          [lo-x hi-x lo-y hi-y] (box d)
                          bare? (loop [gy lo-y]
                                  (cond
                                    (> gy hi-y) false
                                    (loop [gx lo-x]
                                      (cond (> gx hi-x) false
                                            (> (aget T (+ (* gy gw) gx)) hole) true
                                            :else (recur (inc gx)))) true
                                    :else (recur (inc gy))))]
                      (if bare?
                        (do (aset keep i 1)
                            (deposit! d lo-x hi-x lo-y hi-y)
                            (recur (dec i) (inc fixed)))
                        (recur (dec i) fixed))))))]
          {:keep keep
           :repaired repaired
           :residual (loop [i 0 mx 0.0]
                       (if (>= i (* gw gh)) mx (recur (inc i) (max mx (aget T i)))))})))))

(defn- gradient-def
  "One falloff profile in one colour. Stops below `cull` are written as a flat 0: the
   ellipse is drawn out to `extent` stdevs and there the gaussian is worth 0.2% — but
   a rim that stops at 0.2% instead of 0 is a STEP, and on a base stroke that step is
   a visible disc edge across a smooth sky. Below half an 8-bit step it is zero."
  [id [r g b] ^double hard ^long stops ^double extent ^double cull]
  (let [hexc (hex-rgb [r g b])]
    (str "<radialGradient id=\"" id "\">"
         (apply str
                (for [i (range (inc stops))
                      :let [t (/ (double i) stops)
                            a (profile-alpha (* t extent) hard)]]
                  (str "<stop offset=\"" (fixed t 3) "\" stop-color=\"" hexc
                       "\" stop-opacity=\"" (fixed (if (< a cull) 0.0 a) 3) "\"/>")))
         "</radialGradient>")))

(defn- transfer [type* attr* ^double v]
  (apply str (for [ch ["R" "G" "B"]]
               (str "<feFunc" ch " type=\"" type* "\" " attr* "=\"" (fixed v 4) "\"/>"))))

(defn tone-filter
  "The Lift and Brightness present passes as ONE filter over the finished picture.

   Both are per-channel point operations on the composite — Lift is c^(1/amount)
   (shader/fs-src-lift), Brightness is c·amount (fs-src-brightness) — so they are
   exactly `feComponentTransfer` gamma and linear, in that order. Applying them to
   each splat's colour instead would be a different picture: the passes run AFTER
   compositing, and a tone curve does not commute with over-alpha blending.

   One filter on one group costs one offscreen surface for the whole canvas, not one
   per element — this is the only place a filter is affordable here. Sharpen and
   Antialias get no equivalent: both are gated on a local gradient, which
   feConvolveMatrix cannot express.

   `color-interpolation-filters=\"sRGB\"` is not optional. SVG filters default to
   linearRGB, and the shaders operate on the stored sRGB values."
  [id ^double lift ^double brightness ^long w ^long h]
  (str "<filter id=\"" id "\" filterUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\""
       " width=\"" w "\" height=\"" h "\" color-interpolation-filters=\"sRGB\">"
       (when (not= lift 1.0)
         (str "<feComponentTransfer>"
              (transfer "gamma" "exponent" (/ 1.0 (max lift 0.01)))
              "</feComponentTransfer>"))
       (when (not= brightness 1.0)
         (str "<feComponentTransfer>"
              (transfer "linear" "slope" brightness)
              "</feComponentTransfer>"))
       "</filter>"))

(defn- el
  "One splat as markup. A near-round splat is written as a <circle>, which drops both
   the second radius and the whole rotate() — about 40% of the bytes, and for a
   perfectly round splat it is exactly the same shape, since rotating a circle does
   nothing. `round` is the rx/ry ratio at which that substitution is allowed: 1.0 keeps
   it lossless, and above that it is the cheapest lossy trade in the file (a stroke
   fattens by a few percent across its short axis) — the gradient still maps to the
   bounding box, so the falloff comes out right either way."
  [{:keys [cx cy rx ry deg alpha]} fill ^double k ^long dp ^double round]
  (let [cxs (fixed cx dp) cys (fixed cy dp)
        opa (when (< alpha 0.999) (str " fill-opacity=\"" (fixed alpha 2) "\""))]
    (cond
      (<= (/ (double rx) (max (double ry) 1e-6)) round)
      (str "<circle cx=\"" cxs "\" cy=\"" cys
           "\" r=\"" (fixed (* k 0.5 (+ rx ry)) dp) "\" fill=\"" fill "\"" opa "/>")

      ;; A rotated ellipse costs less as translate+rotate than as cx/cy + a rotate that
      ;; REPEATS the centre: cx/cy default to 0, so the centre is written once instead
      ;; of three times. transform is the single largest attribute in the file.
      (> (Math/abs (double deg)) 0.5)
      (str "<ellipse rx=\"" (fixed (* k rx) dp) "\" ry=\"" (fixed (* k ry) dp)
           "\" transform=\"translate(" cxs " " cys ")rotate(" (fixed deg 0) ")\""
           " fill=\"" fill "\"" opa "/>")

      :else
      (str "<ellipse cx=\"" cxs "\" cy=\"" cys
           "\" rx=\"" (fixed (* k rx) dp) "\" ry=\"" (fixed (* k ry) dp) "\""
           " fill=\"" fill "\"" opa "/>"))))

(defn field->svg*
  "Splat field (seed/splat-field's return shape) → {:doc :total :kept} — the SVG
   document plus how many splats went in and how many survived pruning.

   Paint order is REVERSED on the way out: the field is finest-first (index 0 is
   the topmost stroke, which is how both compositors read it), and SVG paints in
   document order, so the last splat is written first.

   Options (see `default-opts`): `:mode` :gradient | :flat, `:fidelity` the one
   quality/size dial (see `fidelity->opts`, and any knob it drives can be overridden
   on its own), `:hard-levels` falloff buckets, `:extent` stdevs drawn, `:hard-sharp`
   the app's Hardness, `:lift` / `:brightness` its two tone passes, `:scale` a
   width/height multiplier for upscaled output."
  [{:keys [splats background height width opacity sig-min sig-max]} opts]
  (let [o (merge default-opts
                 (fidelity->opts (:fidelity (merge default-opts opts)))
                 {:opacity (double (or opacity 1.0))
                  :sig-min (double (or sig-min 1.0))
                  :sig-max (double (or sig-max 1.0))}
                 opts)                     ; explicit knobs still win over :fidelity
        {:keys [mode colors hard-levels stops extent flat-extent dp scale cull-alpha
                cull-peak cull-stride round lift brightness]} o
        all   (vec (keep #(splat->draw % o) splats))
        ;; THE LOSSY STEP: drop the splats that never show. Everything below this
        ;; point only decides how compactly the survivors are written.
        {:keys [^ints keep residual repaired]}
        (keep-mask all (long height) (long width)
                   (double cull-peak) (long cull-stride)
                   ;; the visibility pass must integrate the SAME footprint the file
                   ;; draws, or it scores coverage the render will actually have
                   (double (if (= mode :flat) flat-extent extent)))
        draws (vec (keep-indexed (fn [i d] (when (== 1 (aget keep i)) d)) all))
        ;; hardness buckets: uniform over the range the field actually spans, so a
        ;; field of all-soft base strokes does not spend three profiles on nothing.
        hs   (map :hard draws)
        hlo  (if (seq hs) (reduce min hs) 1.0)
        hhi  (if (seq hs) (reduce max hs) 1.0)
        nh   (max 1 (long hard-levels))
        hbin (fn [^double h]
               (if (< (- hhi hlo) 1e-6)
                 0
                 (min (dec nh) (long (* nh (/ (- h hlo) (- hhi hlo)))))))
        hmid (fn [^long b] (+ hlo (* (/ (+ b 0.5) nh) (- hhi hlo))))
        [bg-r bg-g bg-b] (or background [0.0 0.0 0.0])
        w (long width) h (long height)
        ;; the tone passes run on the COMPOSITE, so they wrap the background rect too
        tone? (or (not= (double lift) 1.0) (not= (double brightness) 1.0))
        head (str "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                  "<svg xmlns=\"http://www.w3.org/2000/svg\" "
                  "width=\"" (fixed (* scale w) 0) "\" height=\"" (fixed (* scale h) 0) "\" "
                  "viewBox=\"0 0 " w " " h "\">"
                  (when tone?
                    (str "<defs>" (tone-filter "tone" (double lift) (double brightness) w h)
                         "</defs><g filter=\"url(#tone)\">"))
                  "<rect width=\"" w "\" height=\"" h "\" fill=\""
                  (hex-rgb [bg-r bg-g bg-b]) "\"/>")
        body
        (if (= mode :flat)
          {:defs "" :els (map #(el % (hex-rgb (:color %)) (double flat-extent) dp round)
                              (rseq draws))}
          (let [q8   (fn [^double c] (long (max 0 (min 255 (Math/round (* 255.0 c))))))
                rows (vec (map-indexed (fn [i d]
                                         (let [[r g b] (:color d)]
                                           [(q8 r) (q8 g) (q8 b) i]))
                                       draws))
                {:keys [entries index]} (palette rows (max 1 (long colors)))
                key-of (fn [i d] [(index i) (hbin (:hard d))])
                used   (into #{} (map-indexed key-of draws))
                gid    (into {} (map-indexed (fn [i k] [k (str "g" i)]) (sort-by first (vec used))))]
            {:defs (str "<defs>"
                        (apply str (for [[[pi hb] id] (sort-by val gid)]
                                     (gradient-def id (nth entries pi) (hmid hb)
                                                   (long stops) (double extent)
                                                   (double cull-alpha))))
                        "</defs>")
             :els (let [n (count draws)]
                    (for [i (range (dec n) -1 -1)
                          :let [d (nth draws i)]]
                      (el d (str "url(#" (gid (key-of i d)) ")") (double extent) dp round)))}))]
    {:doc      (str head (:defs body) (apply str (:els body)) (when tone? "</g>") "</svg>")
     :total    (count all)
     :kept     (count draws)
     :repaired repaired
     :residual residual}))

(defn field->svg
  "`field->svg*`'s document, for callers that do not want the pruning stats."
  [field opts]
  (:doc (field->svg* field opts)))
