(ns splat-painter.pin
  "Print the ACTUAL count + checksums for seed-test/splat-field-golden's fixture.
   Run after an INTENTIONAL spec change to re-pin the golden values:
     rm -rf ~/.jolt/aot-cache && joltc -M:pin
   (the AOT cache invalidates on file SIZE, so same-length literal edits go stale)."
  (:require [splat-painter.seed :as seed]))

(defn- gray-img [H W f]
  {:height H :width W :channels 3
   :pixels (double-array (mapcat (fn [x]
                                   (mapcat (fn [y]
                                             (let [g (double (f x y))] [g g g]))
                                           (range W)))
                                 (range H)))})

(defn -main [& _]
  (let [img (gray-img 48 64 (fn [x y] (if (and (> x 16) (< x 32) (> y 20) (< y 44))
                                        0.9 (* 0.5 (/ (double (+ x y)) 112.0)))))
        {:keys [splats]} (seed/splat-field img {:count 4000 :size 6.0 :stroke 2.5 :detail 0.6
                                                :variation 0.5 :curvature 0.5 :opacity 0.9 :contrast 1.0})
        [sx sy sd sc] (reduce (fn [[sx sy sd sc] {[mx my] :mean [c00 c01 _ c11] :cov [cr cg cb] :color}]
                                [(+ sx mx) (+ sy my) (+ sd (- (* c00 c11) (* c01 c01))) (+ sc cr cg cb)])
                              [0.0 0.0 0.0 0.0] splats)]
    (println "count" (count splats))
    (println (format "sx %.3f  sy %.3f  sd %.3f  sc %.3f" sx sy sd sc))))
