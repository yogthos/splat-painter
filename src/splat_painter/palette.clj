(ns splat-painter.palette
  "Diversity-maximizing color quantization — a port of pixel-mosaic's quantizeColors.
   Build a small palette that maximizes color diversity, then snap each color to the
   nearest palette entry. Colors are [r g b] doubles in 0..1.")

;; --- helpers ----------------------------------------------------------------

(defn dist2
  "Squared Euclidean distance between two [r g b] vectors."
  [c1 c2]
  (let [dr (- (nth c1 0) (nth c2 0))
        dg (- (nth c1 1) (nth c2 1))
        db (- (nth c1 2) (nth c2 2))]
    (+ (* dr dr) (* dg dg) (* db db))))

(defn- coarse-key [c]
  [(int (Math/round (* (nth c 0) 31.0)))
   (int (Math/round (* (nth c 1) 31.0)))
   (int (Math/round (* (nth c 2) 31.0)))])

(defn- key->num [k]
  (+ (* (nth k 0) 1024) (* (nth k 1) 32) (nth k 2)))

;; --- build-palette ----------------------------------------------------------

(defn build-palette
  "Diversity-maximizing palette of up to `k` colors from `colors`.
   Uses frequency-weighted farthest-point sampling: the most common color,
   then repeatedly the candidate farthest from the current palette by
   min-squared-distance (maximin), deterministically.
   Returns a vector of [r g b]. If (< k 2) or (empty? colors), returns []."
  [colors k]
  (let [k (int k)]
    (if (or (< k 2) (empty? colors))
      []
      (let [;; frequency counting with coarse keys (5 bits/channel)
            ;; first-color-wins for each key (deterministic)
            freq+first (loop [cs (seq colors)
                              cnt (transient {})
                              fst (transient {})]
                         (if cs
                           (let [c (first cs)
                                 key (coarse-key c)]
                             (if (get cnt key)
                               (recur (next cs) (assoc! cnt key (inc (get cnt key))) fst)
                               (recur (next cs) (assoc! cnt key 1) (assoc! fst key c))))
                           [(persistent! cnt) (persistent! fst)]))
            candidates (->> (first freq+first)
                            (map (fn [[key cnt]]
                                   {:color (get (second freq+first) key)
                                    :key   key
                                    :count cnt}))
                            (sort-by (fn [{:keys [key count]}] [(- count) (key->num key)]))
                            vec)]
        (if (<= (count candidates) k)
          (mapv :color candidates)
          (loop [palette [(first candidates)]]
            (if (>= (count palette) (min k (count candidates)))
              (mapv :color palette)
              (let [pal-keys (set (map :key palette))
                    [best] (reduce
                             (fn [[best maxmin] cand]
                               (if (contains? pal-keys (:key cand))
                                 [best maxmin]
                                 (let [min-d (reduce (fn [d pc]
                                                      (min d (dist2 (:color pc) (:color cand))))
                                                    ##Inf
                                                    palette)]
                                   (if (> min-d maxmin) [cand min-d] [best maxmin]))))
                             [nil -1.0]
                             candidates)]
                (if best
                  (recur (conj palette best))
                  (mapv :color palette))))))))))

;; --- nearest / quantize -----------------------------------------------------

(defn nearest
  "The palette entry with smallest squared Euclidean distance to `color`."
  [color palette]
  (if (empty? palette)
    color
    (first (reduce (fn [[best d2] c]
                     (let [d (dist2 color c)]
                       (if (< d d2) [c d] [best d2])))
                   [(first palette) (dist2 color (first palette))]
                   (rest palette)))))

(defn quantize
  "Replace each color in `colors` with its nearest neighbor in the palette
   built from the same `colors` and `k`. If (< k 2) or (empty? colors),
   returns `colors` unchanged."
  [colors k]
  (if (or (< (int k) 2) (empty? colors))
    colors
    (let [pal (build-palette colors k)]
      (if (empty? pal)
        colors
        (mapv #(nearest % pal) colors)))))
