(ns splat-painter.prof
  "Headless profiling. Precomputes everything the app caches at load (structure,
   detail, blur, edge grid), then times the per-render `splat-field` — the cost
   that governs slider responsiveness. Run: joltc -M:prof [image] [size]"
  (:require [splat-painter.image :as image]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.seed :as seed]))

(defn- once-ms [f] (let [t0 (System/nanoTime) r (f)] [r (/ (- (System/nanoTime) t0) 1e6)]))
(defn- avg-ms [n f]
  (dotimes [_ 1] (f))
  (let [t0 (System/nanoTime)] (dotimes [_ n] (f)) (/ (- (System/nanoTime) t0) 1e6 (double n))))

(defn -main [& [path szs]]
  (let [path (or path "DrawingWithGaussians/inputs/eye.jpeg")
        size (Double/parseDouble (or szs "16.0"))
        [img0 load-ms] (once-ms #(image/load-image path 1024))
        _ (println (format "image %dx%d, load %.0f ms" (:width img0) (:height img0) load-ms))
        [sfield an-ms] (once-ms #(structure/analyze img0))
        [dmap d-ms]    (once-ms #(wavelet/detail-map img0))
        [blur b-ms]    (once-ms #(structure/blur-image img0 2))
        img (assoc img0 :structure sfield :detail dmap :blur blur :noise-fields (seed/prep-noise sfield))
        _ (println (format "ONE-TIME load: analyze %.0f  detail %.0f  blur %.0f ms"
                           an-ms d-ms b-ms))
        H  (double (:height img))
        ;; detail-map strength: what does the wavelet map report across the image?
        _  (let [dm dmap dv (:detail dm) dmx (:dmax dm)
                 vals (map (fn [i] (/ (aget ^doubles dv i) (max 1e-9 dmx))) (range (alength ^doubles dv)))
                 n (count vals) srt (sort vals)
                 pct (fn [p] (nth srt (min (dec n) (long (* p n)))))]
             (println (format "detail-map: mean %.2f  p50 %.2f  p90 %.2f  p99 %.2f  (frac>0.5: %.2f)"
                              (/ (reduce + vals) n) (pct 0.5) (pct 0.9) (pct 0.99)
                              (/ (double (count (filter #(> % 0.5) vals))) n))))
        run (fn [s] (let [fld (seed/splat-field img {:size s})
                          f (:splats fld)
                          xs (map (comp first :mean) f)
                          xmin (reduce min xs) xmax (reduce max xs)
                          sigs (sort (map (fn [{[c00 c01 _ c11] :cov}]
                                            (Math/sqrt (Math/sqrt (max (- (* c00 c11) (* c01 c01)) 1e-8)))) f))
                          nn (count sigs)
                          p (fn [q] (nth sigs (min (dec nn) (long (* q nn)))))]
                      (println (format "  size %4.1f -> %5d splats : %.1f ms  | x-cover %.0f..%.0f/%.0f%s | stdev p5 %.1f p50 %.1f p95 %.1f px"
                                       s (count f) (avg-ms 3 #(seed/splat-field img {:size s}))
                                       xmin xmax H
                                       (if (< xmax (* 0.95 H)) "  <-- CUTOFF" "")
                                       (p 0.05) (p 0.5) (p 0.95)))))]
    (println "\n--- per-render splat-field (cached grid) ---")
    (doseq [s [size 24.0 12.0 6.0]] (run s))))
