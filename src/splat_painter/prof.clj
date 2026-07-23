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
        [dmap d-ms]    (once-ms #(wavelet/placement-map img0 sfield))
        [blur b-ms]    (once-ms #(structure/blur-image img0 2))
        [blurh bh-ms]  (once-ms #(structure/blur-image img0 (max 6 (quot (:height img0) 80))))
        img (assoc img0 :structure sfield :detail dmap :blur blur :blur-heavy blurh :noise-fields (seed/prep-noise sfield))
        _ (println (format "ONE-TIME load: analyze %.0f  detail %.0f  blur %.0f ms"
                           an-ms d-ms b-ms))
        H  (double (:height img))
        ;; placement-map stats
        _  (let [pm dmap dv (:detail pm) dmx (max 1e-9 (:dmax pm))
                 vals (map (fn [i] (/ (aget ^doubles dv i) dmx)) (range (alength ^doubles dv)))
                 n (count vals) srt (sort vals)
                 pct (fn [p] (nth srt (min (dec n) (long (* p n)))))]
             (println (format "placement-map: mean %.3f  p50 %.3f  p90 %.3f  p99 %.3f  (frac>0.5: %.3f)"
                              (/ (reduce + vals) n) (pct 0.5) (pct 0.9) (pct 0.99)
                              (/ (double (count (filter #(> % 0.5) vals))) n))))
        ;; dark/bright split: luma at map resolution (LINEAR, not gamma),
        ;; grouped by scene brightness <0.25 vs >=0.25
        _  (let [pm dmap dv (:detail pm)
                 sh (:h pm) sw (:w pm)
                 n (* sh sw)
                 ^doubles lum (structure/luma-of img0 sh sw)
                 dark-vals (double-array n) bright-vals (double-array n)
                 dc (volatile! 0) bc (volatile! 0)]
             (dotimes [i n]
               (if (< (aget lum i) 0.25)
                 (do (aset dark-vals @dc (/ (aget dv i) (max 1e-9 (:dmax pm)))) (vswap! dc inc))
                 (do (aset bright-vals @bc (/ (aget dv i) (max 1e-9 (:dmax pm)))) (vswap! bc inc))))
             (let [nd @dc nb @bc
                   quad (fn [^doubles arr cnt]
                          (if (zero? cnt)
                            {:n 0 :p50 0.0 :p90 0.0 :f26 0.0 :f52 0.0}
                            (let [srt (sort (take cnt (seq arr)))
                                  p   (fn [q] (nth srt (min (dec cnt) (long (* q cnt)))))
                                  frac (fn [thr]
                                         (/ (double (count (filter #(> % thr) srt))) cnt))]
                              {:n cnt :p50 (p 0.5) :p90 (p 0.9)
                               :f26 (frac 0.26) :f52 (frac 0.52)})))]
               (println (format "dark  (luma<0.25):  n=%d p50=%.3f p90=%.3f frac>=0.26=%.3f frac>=0.52=%.3f"
                                (:n (quad dark-vals nd)) (:p50 (quad dark-vals nd))
                                (:p90 (quad dark-vals nd)) (:f26 (quad dark-vals nd)) (:f52 (quad dark-vals nd))))
               (println (format "bright(luma>=0.25): n=%d p50=%.3f p90=%.3f frac>=0.26=%.3f frac>=0.52=%.3f"
                                (:n (quad bright-vals nb)) (:p50 (quad bright-vals nb))
                                (:p90 (quad bright-vals nb)) (:f26 (quad bright-vals nb)) (:f52 (quad bright-vals nb))))))
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
