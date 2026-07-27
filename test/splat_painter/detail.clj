(ns splat-painter.detail
  "Detail-placement diagnostic — where does fine detail go? Dumps the admitted
   level ladder against what the sliders asked for, the per-level candidates-vs-emitted
   survival table, total budget spend, and how much of the image's fine-band detail a
   small stroke reaches at its own scale. Run: jolt -M:detail [image] [maxside] [spec] [size]"
  (:require [splat-painter.image :as image]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.seed :as seed]))

(defn- sigma [{:keys [cov]}]
  (let [c00 (double (nth cov 0)) c01 (double (nth cov 1)) c11 (double (nth cov 3))]
    (Math/sqrt (Math/sqrt (max (- (* c00 c11) (* c01 c01)) 1e-12)))))

(defn- pct [v p] (nth (sort v) (long (* p (dec (count v))))))

(defn ladder-report
  "Print the admitted ladder next to the ladder the sliders asked for."
  [dmap detail size variation curvature stroke tier-muls count H W]
  (let [{:keys [levels total scale]} (seed/layer-params dmap detail size variation curvature
                                                        stroke tier-muls count H W)
        requested (long (max 1 (min 7 (inc (Math/round (* (double detail) 6.0))))))
        nominal (fn [lvl] (let [l (double lvl)]
                            (max 0.7 (if (<= l 4.0) (/ (double size) (Math/pow 2.0 l))
                                         (* (/ (double size) 16.0) (Math/pow 0.7 (- l 4.0)))))))]
    (println (format "ladder: %d of %d requested levels admitted | budget scale %.2fx | %d candidates"
                     (clojure.core/count (remove :band levels)) requested scale total))
    (doseq [{:keys [lvl ssz sp nx th band segs]} levels]
      (println (format "  %-5s lvl %d  sigma %5.2f (nominal %5.2f, x%.2f)  spacing %5.2f  cand %7d  th %5.2f  segs %2d"
                       (if band "BAND" "") lvl ssz (nominal lvl)
                       (/ ssz (nominal lvl)) sp nx th segs)))
    levels))

(defn coverage-report
  "How much of the image's FINE detail gets a stroke at its own scale. For the top
   `frac` of pixels by fine-band (sharp) detail, stamp every emitted splat with
   sigma <= `smax` over a disc of its own sigma and report what fraction of those
   detail pixels any small stroke reaches."
  [splats dmap H W frac smax]
  (let [n (* H W)
        sharp (double-array n)
        _ (dotimes [x H] (dotimes [y W] (aset sharp (+ (* x W) y) (wavelet/sharp-at dmap x y))))
        cut (pct (vec sharp) (- 1.0 frac))
        mask (byte-array n)
        small (filter #(<= (sigma %) smax) splats)]
    (doseq [{:keys [mean] :as s} small]
      (let [mx (double (nth mean 0)) my (double (nth mean 1))
            r (long (Math/ceil (max 1.0 (sigma s))))]
        (doseq [x (range (max 0 (- (long mx) r)) (min H (+ (long mx) r 1)))
                y (range (max 0 (- (long my) r)) (min W (+ (long my) r 1)))]
          (aset mask (+ (* x W) y) (byte 1)))))
    (let [hits (loop [i 0 tot 0 cov 0]
                 (if (>= i n) [tot cov]
                     (if (>= (aget sharp i) cut)
                       (recur (inc i) (inc tot) (+ cov (if (pos? (aget mask i)) 1 0)))
                       (recur (inc i) tot cov))))
          [tot cov] hits]
      (println (format "fine detail: %d px in the top %.0f%% of the sharp band; %.1f%% reached by a stroke of sigma<=%.1f (%d such splats of %d)"
                       tot (* 100.0 frac) (* 100.0 (/ (double cov) (max 1 tot))) smax
                       (clojure.core/count small) (clojure.core/count splats)))
      (/ (double cov) (max 1 tot)))))

(defn -main [& [image maxside spec size]]
  (let [img0 (image/load-image (or image "img/DSC_8428-topaz-rawdenoisea.jpg")
                               (if maxside (long (Double/parseDouble maxside)) 1024))
        H (:height img0) W (:width img0)
        sfield (structure/analyze img0)
        dmap   (wavelet/placement-map img0 sfield)
        light  (structure/bilateral-blur img0 3)
        img (assoc img0 :structure sfield :detail dmap :blur light
                   :blur-drift (structure/blur-image img0 2)
                   :blur-heavy (structure/edge-preserving-blur img0 light (structure/blur-image img0 (max 6 (quot H 80))))
                   :noise-fields (seed/prep-noise sfield))
        size (if size (Double/parseDouble size) (max 4.0 (/ (double H) 50.0)))]
    (doseq [[cnt cutin det] (if spec (read-string spec) [[72000 1.0 0.6]])]
      (let [ctl {:count cnt :size size :detail det :variation 0.5 :curvature 0.5 :stroke 2.5
                 :opacity 0.9 :contrast 1.0 :edge-band cutin}]
        (println (format "\n=== %dx%d  Splats %d  Size %.2f  Cut-in %.1f  Detail %.1f ===" W H (long cnt) size cutin det))
        (let [levels (ladder-report dmap det size 0.5 0.5 2.5 [1.0 1.0 1.0 cutin] cnt H W)
              fld    (seed/splat-field img ctl)
              splats (:splats fld)
              sigs   (mapv sigma splats)]
          (println (format "emitted %d splats  sigma p05 %.2f p50 %.2f p95 %.2f max %.2f"
                           (count splats) (pct sigs 0.05) (pct sigs 0.5) (pct sigs 0.95) (reduce max sigs)))
          ;; Survival per level, by bucketing each emitted sigma to the nearest level
          ;; sigma. CAVEAT: the band overlay sits within a hair of the finest ladder rung
          ;; (1.60 vs 1.71 on the HK frame), so its splats land in that rung's bucket and
          ;; inflate it — read the finest row as "ladder + band". To separate them, diff
          ;; two runs with Cut-in 1.0 and 0.0.
          (let [lsz (mapv :ssz (remove :band levels))
                near (fn [s] (apply min-key #(Math/abs (- (double %) (double s))) lsz))
                by   (frequencies (map near sigs))]
            (doseq [{:keys [lvl ssz nx band]} levels :when (not band)]
              (println (format "  lvl %d sigma %5.2f: %7d candidates -> %6d emitted splats (%.1f%%)"
                               lvl ssz nx (get by ssz 0) (* 100.0 (/ (double (get by ssz 0)) (max 1 nx)))))))
          (println (format "budget spend: %d of %d requested (%.0f%%)"
                           (count splats) (long cnt) (* 100.0 (/ (double (count splats)) cnt))))
          (coverage-report splats dmap H W 0.05 3.0))))))
