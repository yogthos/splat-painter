(ns splat-painter.tier-ppc-test
  "Holds the mid/fine tier probes to the realized paint-per-candidate they replace.

   `demand` charges every detail tier candidate a FITTED per-tier constant (mid-yield
   1.8, fine-yield 1.0). The real per-candidate yield is image-dependent — measured
   0.53–2.20 across photos at one level — so the budget spends 2.5× differently per
   image and the Splats slider does not mean the same thing from one photo to the
   next. This suite replaces the constants with a per-image MEASUREMENT, keeping the
   constants as the fallback.

   The measurement is a PROBE, not a feedback loop: `splat-field` stays pure, and the
   probe runs at the tier's ADMITTED level geometry (the fallback ladder's own levels
   — yield is per-candidate, so the probe count is arbitrary but the level is the one
   the tier actually paints at). layer-params does the two-pass internally: compute
   the yield-free ladder geometry, probe the detail tiers, then charge the measured
   yields. The probe itself is injected per path — the CPU closure reuses
   seed/emit-levels (the SAME emission reduce layered-means runs), the GPU closure
   reuses the geometry shader via gen/run-gen! — so the two can only disagree by the
   float precision that parity.clj already bounds. These tests pin the pass structure,
   the demand consumption, the probe's agreement with the realized yield, and the
   CPU/GPU agreement."
  (:require [clojure.test :as t :refer [deftest is testing]]
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

(def ^:private reference-controls
  ;; the acceptance measurement's controls (parity user-controls): a shallow ladder
  ;; (one mid level pinned at the min-phys floor, no fine tier) that exercises the
  ;; whole per-image spread the constants could not absorb.
  {:count 547000 :size 7.5 :stroke 2.4 :detail 1.0 :variation 0.47
   :curvature 0.47 :contrast 1.0 :size-broad 2.31 :size-mid 0.47
   :size-fine 0.47 :edge-band 0.91 :swirl 0.91})

(def ^:private deep-controls
  ;; a deep ladder (mid levels at two sizes plus a fine tier on the min-phys floor)
  ;; where the tier-level nx-weighting and the subdivision claims are live.
  {:count 547000 :size 20.48 :stroke 2.0 :detail 1.0 :variation 0.5
   :curvature 0.5 :contrast 1.0 :size-broad 1.0 :size-mid 1.0
   :size-fine 1.0 :edge-band 1.0 :swirl 1.0})

(defn- img [] (image/load-image fixture 192))

(defn- layer-params
  ([dmap ctl] (layer-params dmap ctl (img)))
  ([dmap ctl im]
   (seed/layer-params dmap (:detail ctl) (:size ctl) (:variation ctl)
                      (:curvature ctl) (:stroke ctl)
                      [(:size-broad ctl) (:size-mid ctl) (:size-fine ctl) (:edge-band ctl)]
                      (:count ctl) (:height im) (:width im))))

(defn- area [im] (double (* (:width im) (:height im))))

(defn- raw-nx-of [l im] (long (Math/ceil (/ (area im) (* (:sp l) (:sp l))))))

(defn- tier-of [fb l]
  (cond (:band l) :band
        (>= (:lvl l) (:broad-end fb)) :fine
        (>= (:lvl l) 2) :mid
        :else :cov))

(deftest tier-probe-passes-mirror-the-shipped-ladder
  ;; The probe pass levels must BE the fallback ladder's levels — same ssz, same
  ;; threshold, same map, same step — so the probe measures the geometry the tier
  ;; actually paints at. The whole ladder rides in the pass (band/coverage/detail
  ;; outside the tier at nx 0) so the subdivision claims (lvl ≤ 2 hands cells to the
  ;; finer rung) and the band admission run exactly as they do in the real pass. The
  ;; probe counts are allocated proportional to each level's real candidate pool
  ;; (area/sp²) so the pass total is the nx-weighted tier mean the demand charges.
  (let [im  (fields/prepare (img))
        dmap (:detail im)
        fb   (layer-params dmap deep-controls im)
        geom (mapv (fn [l] (assoc l :nx (raw-nx-of l im))) (:levels fb))
        passes (seed/tier-probe-passes geom (:broad-end fb) 64)]
    (println (format "tier-ppc-test: deep ladder broad-end %d levels %s"
                     (:broad-end fb)
                     (mapv (fn [l] [(:lvl l) (:ssz l) (:nx l) (:th l) (:map-kind l)])
                           (:levels fb))))
    (testing "every pass level reuses the fallback's geometry field-for-field"
      (doseq [tier [:mid :fine]]
        (let [pass (:levels (tier passes))]
          (is (= (count (:levels fb)) (count pass)))
          (doseq [[pl fl] (map vector pass (:levels fb))]
            (is (= (:lvl fl) (:lvl pl)))
            (is (= (:ssz fl) (:ssz pl)))
            (is (= (:th fl) (:th pl)))
            (is (= (:stepf fl) (:stepf pl)))
            (is (= (:bendf fl) (:bendf pl)))
            (is (= (:map-kind fl) (:map-kind pl)))
            (is (= (:sp fl) (:sp pl)))
            (is (= (:sideo fl) (:sideo pl)))
            (is (= (:selong fl) (:selong pl)))))))
    (testing "the tier's levels probe at raw-nx-proportional counts, others at 0"
      (doseq [tier [:mid :fine]]
        (let [pass  (:levels (tier passes))
              own   (filter #(and (not (:band %)) (= tier (tier-of fb %))) pass)
              other (filter #(or (:band %) (not= tier (tier-of fb %))) pass)]
          (is (seq own) (str tier " probe pass carries that tier"))
          (is (every? #(zero? (:nx %)) other)
              (str tier " pass leaves every other level at nx 0"))
          (is (every? #(pos? (:nx %)) own)
              (str tier " pass probes its own levels"))
          ;; PROPORTIONAL to the real pools, which is what makes the pooled
          ;; emitted/total the nx-weighted tier mean `demand` charges (demand sums
          ;; nx·yield over the tier's levels). Asserted as a ratio, not as
          ;; near-equality: this ladder's pools are 9597/900/256, so proportional
          ;; allocation is deliberately far from uniform. The original assertion here
          ;; was |a-b| <= 2 — uniformity — which contradicted this test's own comment
          ;; and could only ever pass on a ladder whose levels had equal pools.
          (let [own  (vec own)
                nxs  (mapv :nx own)
                pool (mapv #(raw-nx-of % im) own)]
            (doseq [[i j] (partition 2 1 (range (count own)))
                    :let [a (double (nxs i)) b (double (nxs j))
                          pa (double (pool i)) pb (double (pool j))]
                    :when (and (pos? b) (pos? pb))]
              ;; ceil rounding is up to 1 probe per level, so compare the ratios with a
              ;; tolerance that scales with the smaller probe count
              (is (< (Math/abs (- (/ a b) (/ pa pb))) (+ 0.5 (/ 2.0 b)))
                  (str tier " probe counts track the real pools: probes " nxs
                       " against pools " pool)))))))
    (testing "offsets are cumulative over the probed levels, finest-first"
      (doseq [tier [:mid :fine]]
        (let [pass (:levels (tier passes))
              acc  (atom 0)]
          (doseq [l pass]
            (is (= @acc (:offset l)))
            (swap! acc + (:nx l)))
          (is (= @acc (:total (tier passes)))))))
    (testing "the passes total the probe pools"
      (doseq [tier [:mid :fine]]
        (is (= (reduce + (map :nx (:levels (tier passes)))) (:total (tier passes))))))))

(deftest demand-consumes-measured-yields
  ;; :mid-ppc / :fine-ppc carried on the dmap REPLACE the fitted constants in demand;
  ;; the constants remain the fallback for a bare dmap. Doubling the mid yield doubles
  ;; the mid demand, so once the tier is past its slice the thinning factor halves and
  ;; the tier's admitted candidate pool halves — the charge is the emission, not a
  ;; fitted mean.
  (let [im      (fields/prepare (img))
        dmap    (:detail im)
        mid-y   (double (deref #'splat-painter.seed/mid-yield))
        fine-y  (double (deref #'splat-painter.seed/fine-yield))
        mid-nx  (fn [lp] (reduce + (map :nx (filter #(and (>= (:lvl %) 2)
                                                          (< (:lvl %) (:broad-end lp)))
                                                    (:levels lp)))))
        mid-raw (fn [lp] (reduce + (map (fn [l] (long (Math/ceil (/ (area im)
                                                                    (* (:sp l) (:sp l))))))
                                        (filter #(and (>= (:lvl %) 2)
                                                      (< (:lvl %) (:broad-end lp)))
                                                (:levels lp)))))
        ;; (not (:band %)) matters: the edge-band tier is lvl 7, so it passes
        ;; (>= lvl broad-end) and lands in the fine filter — but its nx comes from
        ;; band-level's own share cap and cannot respond to :fine-ppc at all. Counting it
        ;; left a fixed 49 candidates in both runs, which is exactly the residual that
        ;; made the halving assertion fail (|349 - 2*199| = 49).
        fine-nx (fn [lp] (reduce + (map :nx (filter #(and (not (:band %))
                                                          (>= (:lvl %) (:broad-end lp)))
                                                    (:levels lp)))))
        ;; cand-thin only exists where the tier is PAST its slice, and at Splats 547000
        ;; it is not — nx comes out equal to the raw pool, the halving below is vacuous
        ;; and the `<` precondition fails. Squeeze the budget so the cap actually binds;
        ;; the yield is per-candidate and budget-invariant, so this changes only whether
        ;; the thinning is observable, not what is measured.
        thin-controls (assoc reference-controls :count 2000)
        base    (layer-params dmap thin-controls im)
        mea     (layer-params (assoc dmap :mid-ppc (* 2.0 mid-y) :fine-ppc fine-y)
                              thin-controls im)]
    (println (format "tier-ppc-test: mid yield %.2f -> %.2f, mid nx %d -> %d (raw %d)"
                     mid-y (* 2.0 mid-y) (mid-nx base) (mid-nx mea) (mid-raw base)))
    (is (pos? (mid-nx base)) "the mid tier exists at the reference controls")
    (is (< (mid-nx base) (mid-raw base))
        "the mid tier is past its slice at these controls (so the halving below is meaningful)")
    ;; demand charges nx·yield, so a doubled yield doubles the mid demand and the
    ;; thinning factor halves — the tier admits half the candidates. This is the whole
    ;; point: the charge is the emission, not a fitted mean.
    (is (<= (Math/abs (- (mid-nx base) (* 2 (mid-nx mea)))) 2)
        (str "doubling the yield halves the thinned pool: " (mid-nx base) " -> " (mid-nx mea)))
    (when (pos? (fine-nx base))
      (let [meaf (layer-params (assoc dmap :mid-ppc mid-y :fine-ppc (* 2.0 fine-y))
                               thin-controls im)]
        (println (format "tier-ppc-test: fine nx %d -> %d (thin %.4f -> %.4f)"
                         (fine-nx base) (fine-nx meaf)
                         (double (:fine-thin base)) (double (:fine-thin meaf))))
        ;; the fine tier is thinned against its OWN slice, so doubling its yield halves
        ;; its pool the same way — but the pool is a per-level ceil, and the fine tier can
        ;; carry several levels, so the rounding error is a unit per level rather than a
        ;; flat 2. Assert the direction always, and the halving only where the thinning
        ;; is actually binding in both runs (fine-thin < 1 means past its slice).
        (is (<= (fine-nx meaf) (fine-nx base))
            "doubling the fine yield cannot grow the fine pool")
        (when (and (< (double (:fine-thin base)) 1.0)
                   (< (double (:fine-thin meaf)) 1.0))
          (let [lvls (count (filter #(>= (:lvl %) (:broad-end base)) (:levels base)))]
            (is (<= (Math/abs (- (fine-nx base) (* 2 (fine-nx meaf)))) (max 2 lvls))
                (str "doubling the fine yield halves its pool: "
                     (fine-nx base) " -> " (fine-nx meaf)))))))))

(deftest probe-measures-the-realized-yield
  ;; The probe's per-candidate yield must be the REALIZED one — the tier's emitted
  ;; segments divided by its candidate pool. The same emission machinery over the FULL
  ;; pool is the realized yield; the probe is that emission over a small sample, so the
  ;; two must agree to the sample's Monte-Carlo error (per-candidate trace lengths are
  ;; heavy-tailed 0..32, so a 64-probe sample lands within ~20%).
  (let [im   (fields/prepare (img))
        dmap (:detail im)
        H (:height im) W (:width im)
        ctl  reference-controls
        nf   (seed/with-swirl (:noise-fields im) (:swirl ctl))
        fb   (layer-params dmap ctl im)
        geom (mapv (fn [l] (assoc l :nx (raw-nx-of l im))) (:levels fb))
        warp (* 0.95 (double (:curvature ctl)))
        muls [(:size-broad ctl) (:size-mid ctl) (:size-fine ctl) (:edge-band ctl)]
        realized (seed/tier-probe-yields dmap nf (:blur im) (:blur-drift im) (:blur-heavy im)
                                         geom (:broad-end fb)
                                         (apply max (map :nx geom)) warp
                                         (:detail ctl) (:variation ctl) (:curvature ctl)
                                         (:stroke ctl) (:swirl ctl) muls H W)
        sampled  (seed/tier-probe-yields dmap nf (:blur im) (:blur-drift im) (:blur-heavy im)
                                         geom (:broad-end fb) 64 warp
                                         (:detail ctl) (:variation ctl) (:curvature ctl)
                                         (:stroke ctl) (:swirl ctl) muls H W)]
    (println (format "tier-ppc-test: mid realized %.3f vs sampled %.3f (x%.3f), fine realized %.3f vs sampled %.3f (x%.3f)"
                     (:mid-ppc realized) (:mid-ppc sampled)
                     (if (pos? (:mid-ppc sampled)) (/ (:mid-ppc realized) (:mid-ppc sampled)) 0.0)
                     (:fine-ppc realized) (:fine-ppc sampled)
                     (if (pos? (:fine-ppc sampled)) (/ (:fine-ppc realized) (:fine-ppc sampled)) 0.0)))
    (is (pos? (:mid-ppc realized)) "the mid tier emits at the reference controls")
    (is (< (Math/abs (- (:mid-ppc realized) (:mid-ppc sampled)))
           (* 0.20 (:mid-ppc realized)))
        (str "sampled " (:mid-ppc sampled) " vs realized " (:mid-ppc realized)))
    (when (pos? (:fine-ppc realized))
      (is (< (Math/abs (- (:fine-ppc realized) (:fine-ppc sampled)))
             (* 0.20 (:fine-ppc realized)))))))

;; --- GL rig (mirrors band-ppc-test) ---------------------------------------

(defonce ^:private announced (atom false))

(defn- announce! [ctx]
  (when (compare-and-set! announced false true)
    (if-let [err (:error ctx)]
      (println "tier-ppc-test: SKIPPED, no offscreen GL —" err)
      (println (format "tier-ppc-test: GL on %s"
                       (or (gl/gl-get-string* gl/GL-RENDERER) "?"))))))

(defn- with-gl [f]
  (let [ctx (off/ensure-current!)]
    (announce! ctx)
    (if (:error ctx) :skipped (f))))

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

(deftest gpu-tier-probe-matches-the-cpu-measurement
  (testing "the geometry shader's gate+tracer measures the CPU's paint-per-candidate"
    ;; Runs both against the SAME fields (CPU-built, uploaded) so this isolates the GS
    ;; against seed/emit-levels. Field-construction divergence is gpu-fields-test's job.
    (let [r (with-gl
              (fn []
                (let [im     (fields/prepare (img))
                      perm   (gen/upload-perm!)
                      fields (gen/upload-fields! im perm)
                      {:keys [gen tf-buf query vao]} (gen-rig)
                      H (:height im) W (:width im)
                      fb     (layer-params (:dmap fields) reference-controls im)
                      geom   (mapv (fn [l] (assoc l :nx (raw-nx-of l im))) (:levels fb))
                      ctl    reference-controls
                      muls   [(:size-broad ctl) (:size-mid ctl) (:size-fine ctl) (:edge-band ctl)]
                      ;; NOT destructured into mid-ppc/fine-ppc: the CPU result below
                      ;; bound the same two names, shadowing these, so :gpu-mid carried
                      ;; the CPU number and the comparison was against itself.
                      gpu    (gen/tier-probe-yields!
                              gen fields ctl tf-buf query vao
                              geom (:broad-end fb)
                              (* 0.95 (double (:curvature ctl))) H W)
                      ;; the CPU arrays from the PREPARED IMAGE. `fields` is
                      ;; upload-fields!' output, where :noise/:blur/... are GL texture
                      ;; ids (longs) — handing those to the CPU tracer is what made this
                      ;; test error rather than compare.
                      cpu    (seed/tier-probe-yields
                              (:detail im) (seed/with-swirl (:noise-fields im) (:swirl ctl))
                              (:blur im) (:blur-drift im) (:blur-heavy im)
                              geom (:broad-end fb) seed/tier-probe-count
                              (* 0.95 (double (:curvature ctl)))
                              (:detail ctl) (:variation ctl)
                              (:curvature ctl) (:stroke ctl)
                              (:swirl ctl) muls H W)]
                  {:gpu-mid (:mid-ppc gpu) :gpu-fine (:fine-ppc gpu)
                   :cpu-mid (:mid-ppc cpu) :cpu-fine (:fine-ppc cpu)})))]
      (if (= r :skipped)
        (is true "no offscreen GL — skipped")
        (let [{:keys [gpu-mid gpu-fine cpu-mid cpu-fine]} r]
          (println (format "tier-ppc-test: mid probe %.4f vs CPU %.4f (x%.4f), fine probe %.4f vs CPU %.4f (x%.4f)"
                           gpu-mid cpu-mid (if (pos? cpu-mid) (/ gpu-mid cpu-mid) 0.0)
                           gpu-fine cpu-fine (if (pos? cpu-fine) (/ gpu-fine cpu-fine) 0.0)))
          (is (pos? cpu-mid) "the CPU measurement is live on this fixture")
          (is (pos? gpu-mid) "the probe emitted something")
          ;; 10% leaves room for a few discrete probe traces landing differently in
          ;; float32 (the Apple Software Renderer's trig already reaches 3.9e-3 in the
          ;; orientation field — parity's bound is 1e-2) and none at all for measuring
          ;; a different tier: the defect this replaces is a constant wrong by ~2×.
          (is (< (Math/abs (- gpu-mid cpu-mid)) (* 0.10 cpu-mid))
              (str "probe " gpu-mid " vs CPU measurement " cpu-mid
                   " (x" (/ gpu-mid cpu-mid) ")"))
          (when (pos? cpu-fine)
            (is (< (Math/abs (- gpu-fine cpu-fine)) (* 0.10 cpu-fine)))))))))

(defn -main [& _]
  (let [results (t/run-tests 'splat-painter.tier-ppc-test)]
    (System/exit (if (pos? (+ (:fail results 0) (:error results 0))) 1 0))))

