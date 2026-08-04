(ns splat-painter.band-ppc-test
  "Holds the GPU band paint-per-candidate probe to the CPU measurement it replaces.

   band-level charges the edge-band tier nx·ppc and caps nx at band-share·budget/ppc,
   so ppc is what keeps the tier inside its budget slice. fields/prepare measures it
   with seed/band-paint-per-candidate — the real gate and the real tracer on the real
   image. gpu-fields/build-fields! leaves every field in VRAM and so has no CPU
   nf/blur/blurd arrays to run that measurement against, which left the GPU path (the
   SHIPPING path) on a fallback that can never bind the cap: frac·band-trace = frac·6
   against a nominal frac·band-segs = frac·12. On ridge-dense images the tier then
   over-admits — measured 44% on the text fixture (splat-painter-g1p).

   The probe closes that: the geometry shader places candidate i of a level at
   poshash(i, lvl, 29/31), the same hashed stream seed/band-paint-per-candidate draws
   its probes from, so a band-only generation pass over the probe count measures the
   same quantity and the transform-feedback query already reports it exactly. These
   tests pin that the two agree, and that the shipping path carries the result."
  (:require [clojure.test :refer [deftest is testing]]
            [glimmer-gl.gl :as gl]
            [glimmer-gl.offscreen :as off]
            [jolt.ffi :as ffi]
            [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.gpu-fields :as gf]
            [splat-painter.seed :as seed]
            [splat-painter.gen :as gen]
            [splat-painter.shader :as shader]))

(def ^:private fixture "test/splat_painter/fixtures/eye.jpeg")

(defn- img [] (image/load-image fixture 192))

(defonce ^:private announced (atom false))

(defn- announce! [ctx]
  (when (compare-and-set! announced false true)
    (if-let [err (:error ctx)]
      (println "band-ppc-test: SKIPPED, no offscreen GL —" err)
      (println (format "band-ppc-test: GL on %s"
                       (or (gl/gl-get-string* gl/GL-RENDERER) "?"))))))

(defn- with-gl
  "Run `f` on an offscreen context, or return :skipped when there is no display."
  [f]
  (let [ctx (off/ensure-current!)]
    (announce! ctx)
    (if (:error ctx) :skipped (f))))

(defn- gen-rig
  "The generation objects a probe needs, offscreen: program, TF buffer, query, VAO,
   and a 1x1 FBO. The offscreen context has no default framebuffer (framebuffer 0 is
   incomplete by construction), so the draw needs one bound even though the pass runs
   with rasterizer discard and rasterizes nothing. Mirrors parity.clj's run-gpu."
  []
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

(deftest probe-spec-mirrors-the-shipped-band-level
  (testing "the probe measures the tier band-level charges, not a lookalike"
    ;; The probe hand-builds a one-level params map instead of going through
    ;; layer-params (which sizes nx from the budget — the whole point of the
    ;; measurement, so it cannot be an input to it). That makes the probe a SECOND
    ;; place the band tier's identity is written down, and two copies drift: this
    ;; pins every field that decides WHICH tier the geometry shader runs.
    (let [lvl  (first (:levels (seed/band-probe-spec)))
          band (#'splat-painter.seed/band-level
                {:h 8 :w 8 :dmax 1.0
                 :detail (double-array 64 1.0) :sharp (double-array 64 1.0)
                 :edge (double-array 64 1.0) :mid (double-array 64 1.0)
                 :subject (double-array 64 1.0)}
                1.0 (* 256.0 256.0) 100000 1.0)]
      (is (some? band) "the band tier is live at the reference strength")
      (doseq [k [:lvl :th :sideo :selong :segs :stepf :bendf :map-kind :band]]
        (is (= (get band k) (get lvl k))
            (str "probe level differs from the shipped band level at " k)))
      (is (= 1 (:ny lvl)) "one candidate per row: i is the probe index")
      (is (= 0 (:offset lvl)) "the probe level is the only level in the pass")
      (is (= (:probes (seed/band-probe-spec)) (:nx lvl))
          "nx is the probe count, so ppc = emitted/nx"))))

(deftest gpu-probe-matches-the-cpu-measurement
  (testing "the band-only generation pass measures the CPU's paint-per-candidate"
    ;; Runs both against the SAME fields (CPU-built, uploaded) so this isolates the
    ;; geometry shader's gate+tracer against seed/stroke-segments. Field-construction
    ;; divergence is gpu-fields-test's job.
    (let [r (with-gl
              (fn []
                (let [im     (fields/prepare (img))
                      cpu    (double (:band-ppc (:detail im)))
                      perm   (gen/upload-perm!)
                      fields (gen/upload-fields! im perm)
                      {:keys [gen tf-buf query vao]} (gen-rig)]
                  {:cpu cpu :gpu (gen/probe-band-ppc! gen fields tf-buf query vao)})))]
      (if (= r :skipped)
        (is true "no offscreen GL — skipped")
        (let [{:keys [cpu gpu]} r]
          (println (format "band-ppc-test: probe %.4f vs CPU %.4f (x%.4f)"
                           gpu cpu (/ gpu cpu)))
          (is (pos? cpu) "the CPU measurement is live on this fixture")
          (is (pos? gpu) "the probe emitted something")
          ;; Observed EXACT on this hardware (1170 segments over 128 probes both ways),
          ;; but not asserted as equality: trace lengths are discrete, so one probe
          ;; whose gate or ridge stop lands differently in float32 moves the total by
          ;; up to max-segs — 2.7% here — and the CI macOS runner is the Apple Software
          ;; Renderer, whose float32 trig already diverges enough to reach 3.9e-3 in
          ;; the orientation field (see the parity bound at 1e-2). 10% leaves room for
          ;; a few such probes and none at all for measuring a different tier: the
          ;; defect this replaces is a fallback wrong by 44%, and the mutations below
          ;; the docstring (probe ssz, dropped selong) land far outside it.
          (is (< (Math/abs (- gpu cpu)) (* 0.10 cpu))
              (str "probe " gpu " vs CPU measurement " cpu
                   " (x" (/ gpu cpu) ")")))))))

(deftest gpu-fields-path-carries-band-ppc
  (testing "the shipping path's dmap gets a measured ppc, not the fallback"
    (let [r (with-gl
              (fn []
                (let [im    (img)
                      progs (gf/build-programs)]
                  (when progs
                    (let [ctx    (gf/make-ctx)
                          perm   (gen/upload-perm!)
                          fields (try (gf/build-fields! ctx progs im perm)
                                      (finally (gf/free-ctx! ctx)))
                          {:keys [gen tf-buf query vao]} (gen-rig)
                          out    (gen/with-band-ppc! gen fields tf-buf query vao)]
                      {:before (:band-ppc (:dmap fields))
                       :after  (:band-ppc (:dmap out))})))))]
      (cond
        (= r :skipped) (is true "no offscreen GL — skipped")
        (nil? r)       (is true "field shaders did not compile — skipped")
        :else
        (do (is (nil? (:before r))
                "build-fields! has no CPU arrays to measure with, so it carries none")
            (is (and (:after r) (pos? (double (:after r))))
                (str "the probe fills it in: " (:after r))))))))

(deftest with-band-ppc-leaves-a-measured-dmap-alone
  (testing "the CPU field path already measured it; do not pay for a second pass"
    ;; upload-fields! passes fields/prepare's dmap straight through, :band-ppc and all.
    ;; Re-probing there would cost a draw per image load for a number already in hand,
    ;; and would silently replace the reference measurement the goldens pin.
    (let [r (with-gl
              (fn []
                (let [im     (fields/prepare (img))
                      perm   (gen/upload-perm!)
                      fields (gen/upload-fields! im perm)
                      {:keys [gen tf-buf query vao]} (gen-rig)]
                  {:in  (:band-ppc (:dmap fields))
                   :out (:band-ppc (:dmap (gen/with-band-ppc! gen fields tf-buf query vao)))})))]
      (if (= r :skipped)
        (is true "no offscreen GL — skipped")
        (do (is (pos? (double (:in r))) "prepare measured it")
            (is (= (:in r) (:out r)) "passed through untouched"))))))
