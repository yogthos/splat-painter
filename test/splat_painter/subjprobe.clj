(ns splat-painter.subjprobe
  "Diagnostic for splat-painter-olb: is the dark jacket seen as SUBJECT?

   Every guard that is supposed to stop a coverage daub washing across the
   shoulder silhouette keys on subjectness — sfoot (seed.clj:1453) gates the
   Broad growth on the max subject-abs over the grown footprint, and the Broad
   thinning/bgate both read subject-abs too. If the jacket reads as background,
   all of them are looking the other way and Broad grows daubs that reach into it.

   Prints a horizontal transect across the shoulder plus region means. Writes
   nothing. Run: jolt -A:test -m splat-painter.subjprobe"
  (:require [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.wavelet :as wavelet]))

(defn- mean [xs] (/ (reduce + 0.0 xs) (max 1 (count xs))))

(defn- region-stats
  "mean/min/max of subject-abs over a [row0 row1) x [col0 col1) box, stride 2."
  [dmap [row0 row1 col0 col1]]
  (let [vs (for [x (range row0 row1 2) y (range col0 col1 2)]
             (wavelet/subject-abs-at dmap (double x) (double y)))]
    [(mean vs) (reduce min vs) (reduce max vs)]))

(def ^:private regions
  ;; render/field coords at maxside 1024 (Lenin.jpg -> 1024x646), row=x col=y
  {"jacket interior (dark, near map)" [200 340 560 620]
   "jacket interior (dark, lower)"    [420 560 420 560]
   "face"                             [120 260 380 500]
   "wall map (background)"            [170 300 700 820]
   "plaster bust"                     [175 340 395 545]
   ;; the ONLY region where a Broad sweep changes the render (measured: every
   ;; 64x64 block with mean|d| > 5 between Broad 1.0 and 2.5 lies here)
   "top-right corner (Broad acts)"    [0 256 896 1024]})

(defn -main [& args]
  (let [path (or (first args) "img/Lenin.jpg")
        img  (fields/prepare (image/load-image path 1024))
        dmap (:detail img)]
    (println (format "%s  %dx%d   dmap %dx%d"
                     path (:width img) (:height img) (:w dmap) (:h dmap)))
    ;; whole-image histogram: how much of the frame can the Broad dial reach at all?
    ;; mloc = 1 + (Broad-1)*(1-subjAbs), so subjAbs = 1.0 means Broad is a NO-OP there.
    (let [vs (vec (for [x (range 0 (:height img) 4) y (range 0 (:width img) 4)]
                    (wavelet/subject-abs-at dmap (double x) (double y))))
          n  (count vs)
          pct (fn [p] (format "%.3f" (nth (sort vs) (min (dec n) (long (* p n))))))]
      (println (format "subjAbs percentiles  p05 %s  p25 %s  p50 %s  p75 %s  p95 %s"
                       (pct 0.05) (pct 0.25) (pct 0.50) (pct 0.75) (pct 0.95)))
      (println (format "fraction of frame with subjAbs >= 0.999 (Broad fully inert): %.1f%%"
                       (* 100.0 (/ (count (filter #(>= % 0.999) vs)) (double n)))))
      (println (format "fraction with subjAbs >= 0.95: %.1f%%   < 0.70: %.1f%%"
                       (* 100.0 (/ (count (filter #(>= % 0.95) vs)) (double n)))
                       (* 100.0 (/ (count (filter #(< % 0.70) vs)) (double n))))))
    (println)
    (println (format "%-34s %8s %8s %8s" "region" "mean" "min" "max"))
    (doseq [[label box] (sort-by key regions)]
      (let [[m lo hi] (region-stats dmap box)]
        (println (format "%-34s %8.4f %8.4f %8.4f" label m lo hi))))
    (println)
    ;; transect: row 260, walking right out of the jacket, across the silhouette,
    ;; onto the wall map. subject-abs should stay HIGH while inside the subject.
    (println "transect row 260, col 540..760 (jacket -> silhouette -> map)")
    (println (format "%6s %10s" "col" "subjAbs"))
    (doseq [y (range 540 761 10)]
      (println (format "%6d %10.4f" y (wavelet/subject-abs-at dmap 260.0 (double y)))))))
