(ns splat-painter.detailprobe
  "Distribution of the per-splat detail D that w4w keys hardness on.

   D = min(1, Detail * dv * 2.2), so it SATURATES at dv >= 0.455 and a saturated
   signal makes the hardness term a no-op. This prints the histogram of :detail
   over an actual splat field so the choice of signal is measured, not assumed.

   Run: jolt -A:test -m splat-painter.detailprobe [image] [count]"
  (:require [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.seed :as seed]
            [splat-painter.wavelet :as wavelet]))

;; Candidate signals, sampled at a splat's own mean. The one w4w wants must (a) not
;; saturate, and (b) actually separate a detailed foreground from a flat background.
(def ^:private candidates
  {"detail-at (locally normalized)" wavelet/detail-at
   "sharp-at (fine band)"           wavelet/sharp-at
   "subject-abs (absolute)"         wavelet/subject-abs-at})

(defn- stats [vs]
  (let [n (count vs) srt (vec (sort vs))
        pct (fn [p] (nth srt (min (dec n) (long (* p n)))))]
    {:p05 (pct 0.05) :p50 (pct 0.50) :p95 (pct 0.95)
     :sat (/ (count (filter #(>= % 0.999) vs)) (double n))
     :mean (/ (reduce + 0.0 vs) n)}))

(defn- in-box? [[r0 r1 c0 c1] [x y]]
  (and (>= x r0) (< x r1) (>= y c0) (< y c1)))

(defn -main [& args]
  (let [path  (or (first args) "img/Lenin.jpg")
        cnt   (Long/parseLong (or (second args) "60000"))
        img   (fields/prepare (image/load-image path 1024))
        fld   (seed/splat-field img {:count cnt :size 6.0 :stroke 2.4 :detail 1.0
                                     :variation 0.48 :curvature 0.48 :swirl 1.0
                                     :contrast 1.0
                                     :size-broad 0.4 :size-mid 0.7 :size-fine 0.4
                                     :edge-band 1.0})
        ds    (mapv #(double (:detail % 1.0)) (:splats fld))
        n     (count ds)
        srt   (sort ds)
        pct   (fn [p] (nth srt (min (dec n) (long (* p n)))))]
    (println (format "%s  %d splats" path n))
    (println (format "D percentiles  p05 %.3f  p25 %.3f  p50 %.3f  p75 %.3f  p95 %.3f"
                     (pct 0.05) (pct 0.25) (pct 0.50) (pct 0.75) (pct 0.95)))
    (println (format "fraction D >= 0.999 (hardness term inert): %.1f%%"
                     (* 100.0 (/ (count (filter #(>= % 0.999) ds)) (double n)))))
    (doseq [[lo hi] [[0.0 0.2] [0.2 0.4] [0.4 0.6] [0.6 0.8] [0.8 0.999] [0.999 1.01]]]
      (println (format "  D in [%.2f,%.2f): %5.1f%%" lo hi
                       (* 100.0 (/ (count (filter #(and (>= % lo) (< % hi)) ds))
                                   (double n))))))
    ;; which candidate signal actually separates a detailed foreground from a flat
    ;; background WITHOUT saturating? FG = face box, BG = the top-right map corner
    ;; (the only region measuring subjAbs 0.337).
    (let [dmap (:detail img)
          fg-box [60 200 570 700] bg-box [0 140 890 1020]
          means (mapv :mean (:splats fld))]
      (println)
      (println (format "%-32s %6s %6s %6s %6s %7s %7s %7s"
                       "signal" "p05" "p50" "p95" "sat%" "FG" "BG" "FG-BG"))
      (doseq [[label f] (sort-by key candidates)]
        (let [vs (mapv (fn [[x y]] (double (f dmap x y))) means)
              st (stats vs)
              pick (fn [box] (let [sel (keep-indexed
                                         (fn [i m] (when (in-box? box m) (nth vs i)))
                                         means)]
                               (if (seq sel) (/ (reduce + 0.0 sel) (count sel)) Double/NaN)))
              fg (pick fg-box) bg (pick bg-box)]
          (println (format "%-32s %6.3f %6.3f %6.3f %5.1f%% %7.3f %7.3f %7.3f"
                           label (:p05 st) (:p50 st) (:p95 st) (* 100.0 (:sat st))
                           fg bg (- fg bg))))))))
