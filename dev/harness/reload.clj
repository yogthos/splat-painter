(in-ns 'jolt.main)

;; Hot-reload the generation shader after editing gen.clj. Returns :ok only if the
;; program actually compiled AND linked — a nil from build-gen-program means the
;; GS failed to build, and rendering on would silently reuse the STALE program and
;; produce a confidently wrong conclusion.
(def reload-flag (atom nil))

(defn reload-gen! []
  (load-file "src/splat_painter/gen.clj")
  (in-ns 'jolt.main)
  (reset! reload-flag nil)
  (glimmer.core/on-gui
   (fn []
     (try
       (let [area @splat-painter.core/area-atom]
         (glimmer-gl.gtk/make-current area)
         (let [prog (splat-painter.gen/build-gen-program)]
           (if (and prog (:program prog))
             (do (swap! splat-painter.core/gl-state assoc-in [area :gpu :gen] prog)
                 (reset! reload-flag :ok))
             (reset! reload-flag :COMPILE-FAILED))))
       (catch Exception e (reset! reload-flag (str "ERR " e))))))
  (loop [i 0]
    (let [d @reload-flag]
      (cond d         d
            (> i 600) :timeout
            :else     (do (Thread/sleep 100) (recur (inc i)))))))

:reload-ready
