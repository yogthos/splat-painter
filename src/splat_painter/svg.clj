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

(defn- fixed
  "`v` rounded to `dp` decimals, as a compact string — trailing zeros and a bare
   \".0\" dropped. `format` is called once per coordinate per splat, so this exists
   to keep the emit loop off it."
  [^double v ^long dp]
  (let [m (nth [1 10 100 1000] dp)
        r (Math/round (* v (double m)))
        neg (neg? r)
        r (Math/abs r)
        i (quot r m)
        f (rem r m)
        sign (if neg "-" "")]
    (if (zero? f)
      (str sign i)
      (let [fs (str f)
            fs (str (subs "000" 0 (- dp (count fs))) fs)
            fs (str/replace fs #"0+$" "")]
        (str sign i "." fs)))))

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
   :colors      512        ; palette size, :gradient mode
   :hard-levels 3          ; distinct falloff profiles
   :stops       8          ; gradient stops per profile — 6 and 16 measure the same
   :extent      3.5        ; how many stdevs the ellipse spans, :gradient mode
   :flat-extent 1.2        ; iso-contour radius in stdevs, :flat mode
   :hard-sharp  1.7        ; the app's Hardness slider
   :hard-soft   1.0
   :cull-alpha  0.004      ; below half an 8-bit step, the splat is invisible
   :dp          1          ; decimals on coordinates
   :scale       1.0})      ; width/height multiplier — the viewBox stays in image px

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
       :hard (hardness sig sig-min sig-max (double (or detail 1.0)) hard-sharp hard-soft)
       :alpha a
       :color color})))

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

(defn- el [{:keys [cx cy rx ry deg alpha]} fill ^double k ^long dp]
  (let [cxs (fixed cx dp) cys (fixed cy dp)]
    (str "<ellipse cx=\"" cxs "\" cy=\"" cys
         "\" rx=\"" (fixed (* k rx) dp) "\" ry=\"" (fixed (* k ry) dp) "\""
         (when (> (Math/abs (double deg)) 0.05)
           (str " transform=\"rotate(" (fixed deg 1) " " cxs " " cys ")\""))
         " fill=\"" fill "\""
         (when (< alpha 0.999) (str " fill-opacity=\"" (fixed alpha 3) "\""))
         "/>")))

(defn field->svg
  "Splat field (seed/splat-field's return shape) → an SVG document string.

   Paint order is REVERSED on the way out: the field is finest-first (index 0 is
   the topmost stroke, which is how both compositors read it), and SVG paints in
   document order, so the last splat is written first.

   Options (see `default-opts`): `:mode` :gradient | :flat, `:colors` palette size,
   `:hard-levels` falloff buckets, `:extent` stdevs drawn, `:hard-sharp` the app's
   Hardness, `:scale` a width/height multiplier for upscaled output."
  [{:keys [splats background height width opacity sig-min sig-max]} opts]
  (let [o (merge default-opts
                 {:opacity (double (or opacity 1.0))
                  :sig-min (double (or sig-min 1.0))
                  :sig-max (double (or sig-max 1.0))}
                 opts)
        {:keys [mode colors hard-levels stops extent flat-extent dp scale cull-alpha]} o
        draws (vec (keep #(splat->draw % o) splats))
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
        head (str "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                  "<svg xmlns=\"http://www.w3.org/2000/svg\" "
                  "width=\"" (fixed (* scale w) 0) "\" height=\"" (fixed (* scale h) 0) "\" "
                  "viewBox=\"0 0 " w " " h "\">"
                  "<rect width=\"" w "\" height=\"" h "\" fill=\""
                  (hex-rgb [bg-r bg-g bg-b]) "\"/>")
        body
        (if (= mode :flat)
          {:defs "" :els (map #(el % (hex-rgb (:color %)) (double flat-extent) dp)
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
                      (el d (str "url(#" (gid (key-of i d)) ")") (double extent) dp)))}))]
    (str head (:defs body) (apply str (:els body)) "</svg>")))
