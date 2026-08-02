(ns splat-painter.edgeprobe
  (:require [splat-painter.image :as image]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.fields :as fields]
            [splat-painter.structure :as structure]))

(defn sample-box [dmap efa W r0 r1 c0 c1]
  ;; returns [sum-e sum-s sum-ef min-e max-e min-s max-s count]
  (let [acc (double-array 8)]
    (aset acc 3 1.0e9) (aset acc 5 1.0e9)
    (let [nrows (long (- r1 r0)) ncols (long (- c1 c0))]
      (dotimes [ri nrows]
        (dotimes [ci ncols]
          (let [r (long (+ r0 ri)) c (long (+ c0 ci))
                e (double (wavelet/edge-at dmap (double r) (double c)))
                s (double (wavelet/sharp-at dmap (double r) (double c)))
                efr (double (aget efa (unchecked-add (unchecked-multiply r W) c)))]
            (aset acc 0 (unchecked-add (aget acc 0) e))
            (aset acc 1 (unchecked-add (aget acc 1) s))
            (aset acc 2 (unchecked-add (aget acc 2) efr))
            (when (< e (aget acc 3)) (aset acc 3 e))
            (when (> e (aget acc 4)) (aset acc 4 e))
            (when (< s (aget acc 5)) (aset acc 5 s))
            (when (> s (aget acc 6)) (aset acc 6 s))
            (aset acc 7 (unchecked-inc (aget acc 7)))))))
    acc))

(defn report [nm dmap efa W r0 r1 c0 c1]
  (let [acc (sample-box dmap efa W r0 r1 c0 c1)
        n (max 1.0 (aget acc 7))
        em (/ (aget acc 0) n) sm (/ (aget acc 1) n) efm (/ (aget acc 2) n)]
    (println (format "  %-26s edge=%.3f(%.3f-%.3f) sharp=%.3f(%.3f-%.3f) efull=%.3f s/e=%.3f"
                     nm em (aget acc 3) (aget acc 4) sm (aget acc 5) (aget acc 6) efm
                     (/ sm (max 1.0e-9 em))))))

(defn- width-box
  "Percentiles of edge-width over a box from the :edge-width field (array x=row,y=col, W=width)."
  [wa W r0 r1 c0 c1]
  (let [nrows (long (- r1 r0)) ncols (long (- c1 c0))
        vals (java.util.ArrayList.)]
    (dotimes [ri nrows]
      (dotimes [ci ncols]
        (let [r (long (+ r0 ri)) c (long (+ c0 ci))
              v (double (aget ^doubles wa (unchecked-add (unchecked-multiply r W) c)))]
          (when (and (pos? v) (not (Double/isInfinite v)))
            (.add vals v)))))
    (let [n (.size vals)]
      (if (zero? n)
        {:mean Double/NaN :p50 Double/NaN :p90 Double/NaN}
        (let [arr (double-array vals)]
          (java.util.Arrays/sort arr)
          {:mean (/ (reduce + (seq arr)) n)
           :p50  (aget arr (int (* 0.50 n)))
           :p90  (aget arr (int (* 0.90 (dec n))))})))))

(defn report-width [nm wa W r0 r1 c0 c1]
  (let [{:keys [mean p50 p90]} (width-box wa W r0 r1 c0 c1)]
    (println (format "  %-34s mean=%.3f p50=%.3f p90=%.3f px" nm mean p50 p90))))

(defn -main [& _]
  (let [img (fields/prepare (image/load-image "img/Lenin.jpg" 1024))
        dmap (:detail img)
        ef (:edge-full img)
        efa (:edge ef)
        W (long (:w ef))
        ;; edge-width field — the contrast-invariant measure that separates the
        ;; crisp map-frame (offline 2.87px) from the soft ear (6.21px)
        ;; built ON DEMAND, not from fields/prepare: the field costs ~7s on a
        ;; 1024px image and nothing in the render path consumes it, so wiring it
        ;; into image load would tax every open for a diagnostic.
        ew (structure/edge-width-field img)
        ^doubles wa (:width ew)
        We (long (:w ew))]
    (println (format "image %dx%d  dmap %dx%d  edge-full %dx%d  edge-width %dx%d"
                     (:height img) (:width img) (:h dmap) (:w dmap) (:h ef) (:w ef) (:h ew) (:w ew)))
    (println "EDGE-WIDTH FIELD (A1 — confirm it separates crisp from soft):")
    (report-width "map-frame 0-90x60-120 (CRISP)" wa We 0 90 60 120)
    (report-width "ear/neck 95-210x540-625 (soft)" wa We 95 210 540 625)
    (report-width "lapel 300-470x470-560 (soft)" wa We 300 470 470 560)
    (report-width "shoulder 190-330x760-860 (soft)" wa We 190 330 760 860)
    (report-width "desk-edge 560-630x300-520 (v.soft)" wa We 560 630 300 520)
    (println "LEGACY edge/sharp (A0 — energy measures, do NOT discriminate width):")
    (println "SOFT:")
    (report "ear/neck 95-210x540-625" dmap efa W 95 210 540 625)
    (println "CRISP candidates:")
    (report "lapel 300-470x470-560" dmap efa W 300 470 470 560)
    (report "desk-edge 560-630x300-520" dmap efa W 560 630 300 520)
    (report "map-frame 0-90x60-120" dmap efa W 0 90 60 120)
    (report "shoulder-map 190-330x760-860" dmap efa W 190 330 760 860)))
