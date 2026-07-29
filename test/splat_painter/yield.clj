(ns splat-painter.yield
  "Per-level candidate→splat YIELD, and the charge the budget model puts against it.

   `-M:detail` buckets emitted splats by nearest level sigma, which cannot separate
   rungs that land within a hair of each other (the band tier and the finest ladder
   rung routinely do). Here every emitted segment is TAGGED with the level that made
   it — stroke-segments is wrapped to append its `lvl` argument to each row — so the
   attribution is exact in one pass. Band rows are the ones carrying selong>0.

   The charge column is the budget's per-candidate demand term for that tier
   (mid-yield / fine-yield / band frac·band-trace / coverage surv·seg-count). Yield ÷
   charge is the factor the tier is over- or under-charged by, which is what sets how
   much of its own slice it spends.

   Run: jolt -M:yield [image|fixture:dense|fixture:ladder] [maxside] [controls-edn]
     jolt -M:yield img/coyote.jpg 1024 '{:count 550000 :size 20.48 :detail 1.0}'"
  (:require [splat-painter.image :as image]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.fields :as fields]
            [splat-painter.seed :as seed]))

(defn- gray-img [H W f]
  {:height H :width W :channels 3
   :pixels (double-array (mapcat (fn [x]
                                   (mapcat (fn [y] (let [g (double (f x y))] [g g g]))
                                           (range W)))
                                 (range H)))})

(defn- fixture
  "The synthetic images seed-test calibrates against, so a constant can be checked
   against the fixture the test will judge it on."
  [name]
  (case name
    ;; full-contrast 2px checker: every rung has detail to answer
    "dense"  (gray-img 256 256 (fn [x y] (if (odd? (+ (quot x 2) (quot y 2))) 0.9 0.1)))
    ;; coarse ramp + a low-contrast 2-3px checker; detail runs out at the min-phys floor
    "ladder" (gray-img 256 256 (fn [x y]
                                 (let [coarse (* 0.25 (+ 1.0 (Math/sin (* 0.04 (+ x y)))))
                                       fine  (if (odd? (+ (quot x 2) (quot y 3))) 0.12 -0.12)]
                                   (max 0.0 (min 1.0 (+ coarse fine))))))
    (throw (ex-info (str "unknown fixture " name) {}))))

(defn -main [& [path maxside ctl-edn]]
  (let [path (or path "img/DSC_8428-topaz-rawdenoisea.jpg")
        img0 (if (.startsWith path "fixture:")
               (fixture (subs path 8))
               (image/load-image path (if maxside (long (Double/parseDouble maxside)) 1024)))
        H (:height img0) W (:width img0) area (* H W)
        img0   (fields/prepare img0)
        sfield (:structure img0)
        dmap   (:detail img0)
        nf     (:noise-fields img0)
        base {:count 72000 :size (max 4.0 (/ (double H) 50.0)) :detail 0.6
              :variation 0.5 :curvature 0.5 :stroke 2.5 :edge-band 1.0
              :size-broad 1.0 :size-mid 1.0 :size-fine 1.0}
        {:keys [count size detail variation curvature stroke edge-band
                size-broad size-mid size-fine]} (merge base (when ctl-edn (read-string ctl-edn)))
        muls [size-broad size-mid size-fine edge-band]
        lp (seed/layer-params dmap detail size variation curvature stroke muls count H W)
        ;; the tier split layer-params itself uses: broad-end = min(requested levels, 4)
        broad-end (min 4 (max 1 (min 7 (inc (Math/round (* (double detail) 6.0))))))
        px (:pixels img0)
        layered-means #'splat-painter.seed/layered-means
        orig @#'splat-painter.seed/stroke-segments
        segs (with-redefs [splat-painter.seed/stroke-segments
                           (fn [& args]
                             (let [lvl (long (nth args 2))
                                   [rows reason] (apply orig args)]
                               [(mapv (fn [r] (conj r lvl)) rows) reason]))]
               (layered-means dmap nf detail size variation curvature stroke 1.0
                              muls count H W px px))
        band?   (fn [s] (pos? (double (nth s 14))))
        by-lvl  (frequencies (map (fn [s] (if (band? s) :band (long (nth s 15)))) segs))
        natural (fn [l] (Math/ceil (/ (double area) (* (double (:sp l)) (double (:sp l))))))
        pool    (fn [pred] (reduce + 0.0 (map natural (filter pred (remove :band (:levels lp))))))
        mid-pool  (pool (fn [l] (and (>= (long (:lvl l)) 2) (< (long (:lvl l)) broad-end))))
        fine-pool (pool (fn [l] (>= (long (:lvl l)) broad-end)))]
    (println (format "%s  %dx%d  Splats %d  Size %.2f  Detail %.2f  Cut-in %.2f  tiers %.2f/%.2f/%.2f  broad-end %d"
                     path W H (long count) (double size) (double detail) (double edge-band)
                     (double size-broad) (double size-mid) (double size-fine) broad-end))
    (println (format "%-5s %-4s %6s %9s %9s %8s" "tier" "lvl" "sigma" "cand" "splats" "yield"))
    (doseq [{:keys [lvl ssz nx band]} (:levels lp)]
      (let [n (get by-lvl (if band :band (long lvl)) 0)
            tier (cond band "band" (< (long lvl) 2) "cov"
                       (< (long lvl) broad-end) "mid" :else "fine")]
        (println (format "%-5s %-4d %6.2f %9d %9d %8.2f"
                         tier lvl ssz nx n (/ (double n) (max 1 nx))))))
    (let [emitted (fn [pred] (reduce + 0 (map (fn [[k v]] (if (and (integer? k) (pred (long k))) v 0)) by-lvl)))
          mid-em  (emitted (fn [l] (and (>= l 2) (< l broad-end))))
          fine-em (emitted (fn [l] (>= l broad-end)))
          line (fn [nm dem p em thin]
                 (when (pos? p)
                   (println (format "%-5s charge/cand %5.2f   thinned %6.4f   charged %9.0f   emitted %9d   (x%.2f)"
                                    nm (/ (double dem) p) (double thin)
                                    (* (double dem) (double thin)) em
                                    (/ (* (double dem) (double thin)) (max 1 em))))))]
      (line "mid"  (:mid-demand lp)  mid-pool  mid-em  (:cand-thin lp))
      (line "fine" (:fine-demand lp) fine-pool fine-em (:fine-thin lp)))))
