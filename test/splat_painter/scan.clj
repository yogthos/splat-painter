(ns splat-painter.scan
  "Print a horizontal scanline of the placement signals a stroke's stop rules read,
   so a gap the tracer walks through can be checked against what it could have seen.
   Run: jolt -M:scan <image> <maxside> <row> <col0> <col1>"
  (:require [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.wavelet :as wavelet]))

(defn -main [& [path maxside rs c0s c1s]]
  (let [img (fields/prepare (image/load-image (or path "img/collapse-watch.jpg")
                                              (if maxside (long (Double/parseDouble maxside)) 1024)))
        dmap (:detail img)
        ^doubles px (:pixels img)
        W (long (:width img))
        row (long (Double/parseDouble (or rs "80")))
        c0  (long (Double/parseDouble (or c0s "0")))
        c1  (long (Double/parseDouble (or c1s "60")))]
    (println (format "row %d, cols %d..%d   (edge-floor 0.10, band line-hold lifts under 0.105)" row c0 c1))
    (println (format "%5s %6s %7s %7s %7s" "col" "luma" "edge" "sharp" "mid"))
    (doseq [y (range c0 c1)]
      (let [l (aget px (* 3 (+ (* row W) y)))]
        (println (format "%5d %6.2f %7.3f %7.3f %7.3f  %s"
                         y l
                         (wavelet/edge-at dmap row y)
                         (wavelet/sharp-at dmap row y)
                         (wavelet/mid-at dmap row y)
                         (if (> l 0.7) "." "#")))))))
