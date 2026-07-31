(in-ns 'jolt.main)

;; Per-level FIELD OVERRIDE, applied after the filter. Lets one tier's spec be
;; varied while the budget solve above it stays exactly as the full ladder left
;; it — e.g. cap level 2's chain length without touching Detail/Stroke.
;; {lvl {:segs 8}} => level 2 emits 8-segment chains.
(def lvl-override (atom {}))

(defn iso-refit
  [m]
  (let [ks  @keep-lvls
        ovr @lvl-override
        sel (if (= ks :all)
              (:levels m)
              (filterv (fn [l] (contains? ks (long (:lvl l)))) (:levels m)))
        tweaked (mapv (fn [l]
                        (let [o (get ovr (long (:lvl l)))]
                          (if o (merge l o) l)))
                      sel)
        levels (first (reduce (fn [acc l]
                                (let [out (first acc) off (long (second acc))]
                                  [(conj out (assoc l :offset off))
                                   (+ off (cellcount l))]))
                              [[] 0] tweaked))
        out    (assoc m
                      :levels levels
                      :nlev   (clojure.core/count levels)
                      :total  (reduce + 0 (map cellcount levels)))]
    (reset! last-params out)
    out))

(defn iso-render-ovr!
  "Filter + override, then save. `ovr` is {lvl {field val}}."
  [lvls ovr path]
  (reset! lvl-override ovr)
  (iso-render! lvls path))

:tweak-ready
