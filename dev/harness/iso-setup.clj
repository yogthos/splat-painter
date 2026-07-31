(in-ns 'jolt.main)

;; --- per-level isolation render harness -------------------------------------
;; Wraps seed/layer-params so a render can show ONE tier alone AT ITS OWN
;; PARAMETERS. Filtering here (rather than moving the Detail slider) is the
;; whole point: Detail changes nlev -> budget split -> every level's ssz and
;; spacing, so a Detail-based "tier control" confounds tier identity with tier
;; parameters. This keeps the full ladder's budget solve intact and only drops
;; levels from the emission list.

;; captured UNCONDITIONALLY — (def x ...) inside (when-not (resolve 'x) ...) is a
;; trap: jolt creates the var at COMPILE time so resolve is truthy and the body
;; never runs. Re-run this file only after (load-file "src/splat_painter/seed.clj"),
;; or orig-lp captures the wrapper and the filter compounds.
(def orig-lp splat-painter.seed/layer-params)

(def keep-lvls (atom :all))   ; :all, or a set of :lvl values e.g. #{0} #{7} #{2 3}
(def last-params (atom nil))  ; the post-filter map, for inspection

(defn cellcount [l] (* (long (:nx l)) (long (:ny l))))

(defn iso-refit
  "Drop non-selected levels and re-derive the three fields that depend on the
   level LIST rather than on any single level: cumulative :offset (finest-first,
   so GPU gl_VertexID order == CPU emission order), :nlev (the GS decodes exactly
   this many slots) and :total (candidate count drawn as GL_POINTS)."
  [m]
  (let [ks @keep-lvls]
    (if (= ks :all)
      (do (reset! last-params m) m)
      (let [kept   (filterv (fn [l] (contains? ks (long (:lvl l)))) (:levels m))
            levels (first (reduce (fn [acc l]
                                    (let [out (first acc) off (long (second acc))]
                                      [(conj out (assoc l :offset off))
                                       (+ off (cellcount l))]))
                                  [[] 0] kept))
            out    (assoc m
                          :levels levels
                          :nlev   (clojure.core/count levels)
                          :total  (reduce + 0 (map cellcount levels)))]
        (reset! last-params out)
        out))))

(alter-var-root
 (var splat-painter.seed/layer-params)
 (fn [_]
   (fn [dmap detail size variation curvature stroke tier-muls count H W]
     (iso-refit (orig-lp dmap detail size variation curvature stroke
                         tier-muls count H W)))))

;; --- GUI-thread render + native-res save ------------------------------------

(def done-flag (atom nil))

(defn iso-render!
  "Set the level filter, then render + save a native-resolution PNG on the GTK
   thread. Blocks the REPL thread on an atom poll (jolt deref has no 3-arg
   timeout). Returns :ok, an error string, or :timeout."
  [lvls path]
  (reset! keep-lvls lvls)
  (reset! done-flag nil)
  (glimmer.core/on-gui
   (fn []
     (try
       (glimmer-gl.gtk/make-current @splat-painter.core/area-atom)
       ((var splat-painter.core/gpu-save-png!) @splat-painter.core/area-atom path)
       (reset! done-flag :ok)
       (catch Exception e (reset! done-flag (str "ERR " e))))))
  (loop [i 0]
    (let [d @done-flag]
      (cond d          d
            (> i 900)  :timeout
            :else      (do (Thread/sleep 100) (recur (inc i)))))))

(defn ladder
  "The unfiltered ladder, for picking which :lvl values exist."
  []
  (reset! keep-lvls :all)
  (let [f (splat-painter.core/field-for-current-controls)]
    (mapv (fn [l] {:lvl (:lvl l) :ssz (:ssz l) :sp (:sp l)
                   :nx (:nx l) :band (:band l) :map-kind (:map-kind l)
                   :segs (:segs l)})
          (:levels @last-params))))

:iso-ready
