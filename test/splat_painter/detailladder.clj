(ns splat-painter.detailladder
  "What the Detail dial actually does, rung by rung, across its whole travel.

   Detail's job is to decide how fine the ladder goes. It cannot be read off any
   single render — the dial changes the rung count, hence the budget split, hence
   every rung's size and spacing — so this prints the admitted ladder at each step
   of the slider next to the finest stroke size it reaches. A dial with a gradual
   progression shows the finest size falling smoothly and the entry count rising;
   a staircase shows repeated IDENTICAL rows.

   Run: jolt -A:test -m splat-painter.detailladder [image] [maxside] [size]
   Evidence for splat-painter-a65."
  (:require [splat-painter.image :as image]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.seed :as seed]))

;; the settings the a65 staircase was measured at
(def ^:private a65-muls [0.4 2.2 2.2 1.4])
(def ^:private splats 570000)
(def ^:private tier-muls (atom a65-muls))

(defn- row [dmap detail size H W]
  (let [lp    (seed/layer-params dmap detail size 0.5 0.5 2.0 @tier-muls splats H W)
        ls    (:levels lp)
        ladder (remove :band ls)
        band  (first (filter :band ls))
        finest (reduce min (map :ssz ladder))]
    {:detail detail
     :rungs (clojure.core/count ladder)
     :band? (some? band)
     :finest finest
     :total (long (:total lp))
     :ladder (mapv (fn [l] [(long (:lvl l)) (double (:ssz l)) (long (:nx l))])
                   (sort-by :lvl ladder))
     :bandssz (when band (double (:ssz band)))}))

(defn -main [& [image maxside size muls]]
  (let [img    (image/load-image (or image "img/Lenin.jpg")
                                 (if maxside (long (Double/parseDouble maxside)) 1024))
        H (:height img) W (:width img)
        sfield (structure/analyze img)
        dmap   (wavelet/placement-map img sfield)
        size   (if size (Double/parseDouble size) 6.0)
        _      (when muls (reset! tier-muls (read-string muls)))
        rows   (mapv (fn [d] (row dmap d size H W))
                     (mapv (fn [i] (/ (double i) 20.0)) (range 0 21)))]
    (println (format "%dx%d  Size %.2f  Broad %.1f  Mid/Fine %.1f  Cut-in %.1f  Splats %d"
                     W H size (nth @tier-muls 0) (nth @tier-muls 1) (nth @tier-muls 3) splats))
    (println "Detail rungs band finest    total  ladder [lvl:sigma xcand]")
    (doseq [{:keys [detail rungs band? finest total ladder]} rows]
      (println (format "%5.2f %5d %4s %6.2f %8d  %s"
                       detail rungs (if band? "yes" "-") finest total
                       (clojure.string/join "  "
                         (mapv (fn [[l s n]] (format "%d:%.2f x%d" l s n)) ladder)))))
    ;; the two defects, as numbers: dead travel at the top, and the largest
    ;; single-step jump in the finest stroke size anywhere on the dial
    (let [fin  (mapv :finest rows)
          dead (clojure.core/count
                (take-while (fn [[a b]] (== (double a) (double b)))
                            (reverse (mapv vector fin (rest fin)))))
          jumps (mapv (fn [[a b]] (Math/abs (- (double a) (double b))))
                      (mapv vector fin (rest fin)))
          worst (reduce max jumps)
          wi    (.indexOf ^java.util.List jumps worst)]
      (println (format "\ndead travel at the top: %d of %d steps change nothing (%.0f%% of the dial)"
                       dead (clojure.core/count jumps)
                       (* 100.0 (/ (double dead) (clojure.core/count jumps)))))
      (println (format "largest finest-size step: %.2f px at Detail %.2f -> %.2f (%.2f -> %.2f)"
                       worst (/ (double wi) 20.0) (/ (double (inc wi)) 20.0)
                       (double (nth fin wi)) (double (nth fin (inc wi)))))
      (println (format "distinct ladders over the travel: %d of %d"
                       (clojure.core/count (distinct (mapv :ladder rows)))
                       (clojure.core/count rows))))))
