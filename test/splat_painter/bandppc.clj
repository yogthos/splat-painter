(ns splat-painter.bandppc
  "What the GPU field path's band paint-per-candidate measurement buys, per image.

   band-level charges the edge-band tier nx·ppc and caps nx at band-share·budget/ppc.
   Without a measured ppc the cap keys on the fallback frac·band-trace = frac·6, which
   is below the nominal frac·band-segs = frac·12 the cap maxes against — so it can
   never bind, and on a ridge-dense image the tier admits far more seeds than its
   band-share ceiling allows. This prints, per image: the measured ppc from the GPU
   probe (gen/probe-band-ppc!), the CPU measurement for comparison, and the band nx +
   layer total with and without the measurement.

   Run: jolt -A:test -m splat-painter.bandppc <maxside> <image>...
   Sizing evidence for splat-painter-g1p."
  (:require [clojure.string :as str]
            [glimmer-gl.gl :as gl]
            [glimmer-gl.offscreen :as off]
            [jolt.ffi :as ffi]
            [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.gpu-fields :as gf]
            [splat-painter.seed :as seed]
            [splat-painter.gen :as gen]
            [splat-painter.shader :as shader]))

;; the settings the g1p measurements were taken at (parity.clj's user-controls)
(def ^:private controls
  {:count 547000 :size 7.5 :stroke 2.4 :detail 0.56 :variation 0.47
   :curvature 0.47 :contrast 1.0 :size-broad 2.31 :size-mid 0.47
   :size-fine 0.47 :edge-band 0.91 :swirl 0.91})

(defn- band-of
  "Band nx + layer total for `dmap` at the reference controls."
  [dmap H W]
  (let [{:keys [count size stroke detail variation curvature
                size-broad size-mid size-fine edge-band]} controls
        lp   (seed/layer-params dmap detail size variation curvature stroke
                                [size-broad size-mid size-fine edge-band] count H W)
        band (first (filter :band (:levels lp)))]
    {:nx (long (or (:nx band) 0)) :total (long (:total lp))}))

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

(defn- report! [rig path maxside]
  (let [img0  (image/load-image path maxside)
        H (long (:height img0)) W (long (:width img0))
        perm  (gen/upload-perm!)
        progs (gf/build-programs)
        ctx   (gf/make-ctx)
        gpuf  (try (gf/build-fields! ctx progs img0 perm)
                   (finally (gf/free-ctx! ctx)))
        raw   (:dmap gpuf)
        ppc   (gen/probe-band-ppc! (:gen rig) gpuf (:tf-buf rig) (:query rig) (:vao rig))
        cpu   (:band-ppc (:detail (fields/prepare img0)))
        without (band-of raw H W)
        with    (band-of (assoc raw :band-ppc ppc) H W)]
    (println (format "%-22s %4dx%-4d  probe ppc %6.3f  cpu ppc %6.3f (x%.3f)"
                     (last (str/split path #"/")) W H
                     ppc (double cpu) (/ ppc (double cpu))))
    (println (format "  %-24s band nx %7d  total %8d" "measured (now)"
                     (:nx with) (:total with)))
    (println (format "  %-24s band nx %7d  total %8d" "fallback (before)"
                     (:nx without) (:total without)))
    (println (format "  %-24s nx x%.3f  total x%.3f" "over-admission removed"
                     (/ (double (:nx without)) (max 1.0 (double (:nx with))))
                     (/ (double (:total without)) (max 1.0 (double (:total with))))))))

(defn -main [& args]
  (let [maxside (long (Long/parseLong (or (first args) "1024")))
        paths   (or (seq (rest args)) ["img/collapse-watch.jpg"])
        ectx    (off/ensure-current!)]
    (when (:error ectx)
      (println "no offscreen GL:" (:error ectx))
      (System/exit 1))
    (let [rig (gen-rig)]
      (doseq [p paths] (report! rig p maxside)))
    (System/exit 0)))
