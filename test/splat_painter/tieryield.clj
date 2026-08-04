(ns splat-painter.tieryield
  "What the per-image tier-yield probe buys: budget spend per image, measured against
   fitted.

   `demand` charges each detail candidate a per-tier constant (mid-yield 1.8, fine-yield
   1.0). Real per-candidate yield is image-dependent, so the same Splats request lands a
   different fraction of the budget on every image — the user-visible bug in
   splat-painter-zig. This prints, per image, the requested count against what the
   shipping GPU path actually emits, BEFORE (dmap pre-loaded with the fitted constants,
   which suppresses the probe) and AFTER (bare dmap, probe runs), plus the measured
   yields themselves.

   The spread of the spend column is the acceptance criterion: it has to narrow without
   the constants being re-fitted per image.

   Run: jolt -A:test -m splat-painter.tieryield [maxside] [image...]"
  (:require [clojure.string :as str]
            [glimmer-gl.gl :as gl]
            [glimmer-gl.offscreen :as off]
            [jolt.ffi :as ffi]
            [splat-painter.image :as image]
            [splat-painter.gpu-fields :as gf]
            [splat-painter.seed :as seed]
            [splat-painter.gen :as gen]
            [splat-painter.shader :as shader]))

;; parity.clj's user-controls, with :detail at the post-a65 default (the dial was remapped
;; onto the achievable ladder depth, so the 0.56 the issue recorded is not the same setting)
(def ^:private controls
  {:count 547000 :size 7.5 :stroke 2.4 :detail 1.0 :variation 0.47
   :curvature 0.47 :contrast 1.0 :size-broad 2.31 :size-mid 0.47
   :size-fine 0.47 :edge-band 0.91 :swirl 0.91})

(def ^:private images
  ["img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg"
   "img/street.jpg"
   "img/A7A01668-topaz-rawdenoise-crop.jpg"
   "img/collapse-watch.jpg"
   "img/bunny.jpeg"])

(defn- gen-rig []
  (let [genp   (gen/build-gen-program)
        tf-buf (gl/gen-one gl/gl-gen-buffers)
        query  (gl/gen-one gl/gl-gen-queries)
        vao    (gl/gen-one gl/gl-gen-vertex-arrays)
        fbo    (gl/gen-one gl/gl-gen-framebuffers)
        tex    (gl/gen-one gl/gl-gen-textures)]
    (when (nil? genp) (throw (Exception. "build-gen-program returned nil")))
    (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
    (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 gl/GL-RGBA32F 1 1 0 gl/GL-RGBA gl/GL-FLOAT ffi/null)
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
    (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0 gl/GL-TEXTURE-2D tex 0)
    (gl/gl-bind-buffer gl/GL-TRANSFORM-FEEDBACK-BUFFER tf-buf)
    (gl/gl-buffer-data gl/GL-TRANSFORM-FEEDBACK-BUFFER
                       (* shader/max-splats 12 (ffi/sizeof :float)) ffi/null
                       gl/GL-DYNAMIC-COPY)
    (gl/gl-bind-vertex-array vao)
    {:gen genp :tf-buf tf-buf :query query :vao vao}))

(defn- spend
  "Emitted splats for `fields` at the reference controls, through the shipping path."
  [rig fields H W]
  (:count (gen/generate! (:gen rig) fields controls (:tf-buf rig) (:query rig) (:vao rig)
                         {:height H :width W})))

(defn- yields
  "The measured mid/fine yields for these fields, plus the layer-params result they
   produced, by running layer-params with a probe that records what it returns."
  [rig fields H W]
  (let [seen (atom nil)
        {:keys [detail size variation curvature stroke
                size-broad size-mid size-fine edge-band count]} controls
        lp (seed/layer-params (:dmap fields) detail size variation curvature stroke
                              [size-broad size-mid size-fine edge-band] count H W
                              (fn [probe-ready broad-end H W]
                                (let [y (gen/tier-probe-yields!
                                         (:gen rig) fields controls (:tf-buf rig) (:query rig)
                                         (:vao rig) (:levels probe-ready) broad-end
                                         (:warp probe-ready) H W)]
                                  (reset! seen y)
                                  y)))]
    (assoc @seen :lp lp)))

(defn- report! [rig progs path maxside]
  (let [img  (image/load-image path maxside)
        H (long (:height img)) W (long (:width img))
        perm (gen/upload-perm!)
        ctx  (gf/make-ctx)
        f    (try (gf/build-fields! ctx progs img perm)
                  (finally (gf/free-ctx! ctx)))
        ;; BEFORE: pre-load the dmap with the fitted constants. layer-params skips the
        ;; probe when the dmap already carries a measurement, so this is exactly the
        ;; shipped open-loop behaviour with no code change.
        before (spend rig (update f :dmap assoc
                                 :mid-ppc @#'splat-painter.seed/mid-yield
                                 :fine-ppc @#'splat-painter.seed/fine-yield)
                      H W)
        after  (spend rig f H W)
        y      (yields rig f H W)
        req    (double (:count controls))
        ;; cand-thin / fine-thin are what the charge ACTS through: at 1.0 the tier is
        ;; inside its slice and demand — measured or fitted — changes nothing.
        {:keys [detail size variation curvature stroke
                size-broad size-mid size-fine edge-band count]} controls
        lp     (seed/layer-params (:dmap f) detail size variation curvature stroke
                                  [size-broad size-mid size-fine edge-band] count H W)]
    (println (format "%-26s %4dx%-4d  yield mid %5.2f  cand-thin fitted %5.3f measured %5.3f   before %7d %5.1f%%   after %7d %5.1f%%"
                     (last (str/split path #"/")) W H
                     (double (:mid-ppc y))
                     (double (:cand-thin lp)) (double (:cand-thin (:lp y)))
                     before (* 100.0 (/ (double before) req))
                     after  (* 100.0 (/ (double after) req))))
    ;; COST: the probe runs inside layer-params, so it runs on every generate! — i.e.
    ;; every drag frame, not once per image load. Two extra passes plus two query
    ;; readbacks, and a readback is a pipeline stall. Timed both ways here because that
    ;; is the number that decides whether this belongs on the drag path at all.
    (let [fitted (update f :dmap assoc
                         :mid-ppc @#'splat-painter.seed/mid-yield
                         :fine-ppc @#'splat-painter.seed/fine-yield)
          time-of (fn [flds]
                    (spend rig flds H W)                    ; warm
                    (let [t (System/nanoTime) n 5]
                      (dotimes [_ n] (spend rig flds H W))
                      (/ (- (System/nanoTime) t) 1e6 n)))
          t-off (time-of fitted)
          t-on  (time-of f)]
      (println (format "  %-24s generate! %.1f ms without probe, %.1f ms with (+%.1f ms/frame)"
                       "" t-off t-on (- t-on t-off))))
    {:before (/ (double before) req) :after (/ (double after) req)}))

(defn -main [& args]
  (let [maxside (long (Long/parseLong (or (first args) "1024")))
        paths   (or (seq (rest args)) images)
        ectx    (off/ensure-current!)]
    (when (:error ectx)
      (println "no offscreen GL:" (:error ectx))
      (System/exit 1))
    (let [rig   (gen-rig)
          progs (gf/build-programs)
          _     (println (format "Splats %d  Size %.2f  Detail %.2f  maxside %d"
                                 (long (:count controls)) (double (:size controls))
                                 (double (:detail controls)) maxside))
          rows  (mapv (fn [p] (report! rig progs p maxside)) paths)
          spr   (fn [k] (let [vs (mapv k rows)]
                          (/ (reduce max vs) (max 1e-9 (reduce min vs)))))]
      (println (format "\nspend spread (max/min): before %.2fx   after %.2fx"
                       (spr :before) (spr :after)))
      (println (format "spend range: before %.1f%%-%.1f%%   after %.1f%%-%.1f%%"
                       (* 100.0 (reduce min (mapv :before rows)))
                       (* 100.0 (reduce max (mapv :before rows)))
                       (* 100.0 (reduce min (mapv :after rows)))
                       (* 100.0 (reduce max (mapv :after rows))))))
    (System/exit 0)))
