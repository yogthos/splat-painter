(ns splat-painter.parity
  "Headless GPU-vs-CPU parity harness for the band-tier hunt (SPEC-9wx).

   Replicates core.clj's gpu-verify! (count ratio + first-divergence index) without
   the GTK app, so the divergence can be measured at any resolution in ~seconds.
   The band side-push override mirrors dev/harness/iso-setup: it alter-var-roots
   seed/layer-params so BOTH paths (layered-means on the CPU, generate! on the GPU)
   see the same level list.

   Run: jolt -A:test -m splat-painter.parity <image> <maxside|0> <shipped|off> [chains]
   maxside 0 = native resolution. `chains` also prints the CPU band chain-length +
   stop-reason distribution (mirrors test/splat_painter/yield.clj)."
  (:require [glimmer-gl.gl :as gl]
            [glimmer-gl.offscreen :as off]
            [jolt.ffi :as ffi]
            [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.seed :as seed]
            [splat-painter.gen :as gen]
            [splat-painter.shader :as shader]))

(def ^:private user-controls
  {:count 547000 :size 7.5 :stroke 2.4 :detail 0.56 :variation 0.47
   :curvature 0.47 :contrast 1.0 :size-broad 2.31 :size-mid 0.47
   :size-fine 0.47 :edge-band 0.91 :swirl 0.91})

;; band side-push override (mirror of iso-setup's alter-var-root)
(def ^:private orig-lp splat-painter.seed/layer-params)
(def ^:private sideo-mode (atom :shipped))

(defn- lp-wrap
  "Apply the band :sideo override to every layer-params result. `:off` zeroes the
   band's side push (soff=0 on both paths); `:shipped` passes through unchanged."
  [& args]
  (let [m (apply orig-lp args)]
    (if (= @sideo-mode :off)
      (update m :levels
              (fn [ls] (mapv (fn [l] (if (:band l) (assoc l :sideo 0.0) l)) ls)))
      m)))

(alter-var-root (var splat-painter.seed/layer-params) (fn [_] lp-wrap))

;; CPU side
(defn- cpu-field [img]
  (seed/splat-field img user-controls))

;; GPU side (offscreen GL, mirrors core.clj ensure-gpu!/generate!)
(defn- run-gpu [img]
  (let [ectx (off/ensure-current!)
        _    (when (:error ectx)
               (throw (Exception. (str "no offscreen GL: " (:error ectx)))))
        H (long (:height img)) W (long (:width img))
        perm   (gen/upload-perm!)
        fields (gen/upload-fields! img perm)
        genp   (gen/build-gen-program)
        tf-buf (gl/gen-one gl/gl-gen-buffers)
        query  (gl/gen-one gl/gl-gen-queries)
        vao    (gl/gen-one gl/gl-gen-vertex-arrays)
        ;; offscreen context has NO default framebuffer (framebuffer 0 is
        ;; incomplete by construction), so bind a 1x1 FBO for the draw — the gen
        ;; pass runs with rasterizer discard, nothing is ever rasterized into it.
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
    (let [{:keys [count total]} (gen/generate! genp fields user-controls tf-buf query vao
                                               {:height H :width W})]
      {:n count :total total :splats (gen/read-splats tf-buf count)})))

;; stats + first divergence (mirror core.clj gpu-verify!)
(defn- splat-stats [splats]
  (reduce (fn [[sx sy sd sc sa] {[mx my] :mean [c00 c01 _ c11] :cov [r g b] :color a :alpha}]
            [(+ sx mx) (+ sy my) (+ sd (- (* c00 c11) (* c01 c01))) (+ sc r g b)
             (+ sa (double (or a 1.0)))])
          [0.0 0.0 0.0 0.0 0.0] splats))

(defn- first-divergence
  "Index of the first record whose mean position differs by > 0.5px, or nil."
  [gpu cpu]
  (let [m (min (count gpu) (count cpu))]
    (loop [i 0]
      (if (>= i m)
        nil
        (let [[gx1 gy1] (:mean (nth gpu i)) [cx1 cy1] (:mean (nth cpu i))]
          (if (or (> (Math/abs (- gx1 cx1)) 0.5) (> (Math/abs (- gy1 cy1)) 0.5))
            i (recur (inc i))))))))

;; CPU band chain stats (mirror of yield.clj's wrapping)
(defn- band-chain-stats
  "Wrap stroke-segments to record [lvl chain-len stop-reason] per chain, run
   layered-means directly (yield.clj's proven pattern — no splat-field emit, so
   no sample-arr path), and return band-tier (lvl 7) chain stats."
  [img]
  (let [H (long (:height img)) W (long (:width img))
        px (:pixels img)
        dmap (:detail img) nf (seed/with-swirl (:noise-fields img) 0.91)
        layered-means #'splat-painter.seed/layered-means
        orig @#'splat-painter.seed/stroke-segments
        chains (atom [])
        _ (with-redefs [splat-painter.seed/stroke-segments
                        (fn [& args]
                          (let [lvl (long (nth args 2))
                                [rows reason] (apply orig args)]
                            (when (seq rows)
                              (swap! chains conj [lvl (clojure.core/count rows) reason]))
                            [(mapv (fn [r] (conj r lvl)) rows) reason]))]
            (layered-means dmap nf 0.56 7.5 0.47 0.47 2.4 0.91
                           [2.31 0.47 0.47 0.91] 547000 H W
                           (or (:blur img) px) (or (:blur-drift img) (or (:blur img) px))))
        band (filterv (fn [[lvl _ _]] (= 7 lvl)) @chains)
        lens (sort (map second band))
        n    (clojure.core/count lens)
        why  (frequencies (map (fn [[_ _ r]] r) band))]
    {:n n
     :p50 (when (pos? n) (nth lens (quot n 2)))
     :p90 (when (pos? n) (nth lens (quot (* 9 n) 10)))
     :max (when (pos? n) (last lens))
     :why why}))

(defn- print-chains [img]
  (let [cs (band-chain-stats img)
        wr (map (fn [[k v]] (format "%s %d (%.1f%%)" (name k) v
                                    (* 100.0 (/ (double v) (max 1 (:n cs))))))
                (sort-by (comp - val) (:why cs)))]
    (println (format "CPU band chains: n=%d p50=%s p90=%s max=%s  stops: %s"
                     (:n cs) (:p50 cs) (:p90 cs) (:max cs)
                     (clojure.string/join "  " wr)))))

;; per-chain trace at the first divergence
(defn- trace-divergence
  "Find the chain owning the first-divergence index and dump both paths' segments
   with the CPU-side field state at each CPU segment position."
  [img cpu-splats gpu-splats di]
  (let [H (long (:height img)) W (long (:width img))
        px (:pixels img)
        dmap (:detail img) nf (seed/with-swirl (:noise-fields img) 0.91)
        layered-means #'splat-painter.seed/layered-means
        orig @#'splat-painter.seed/stroke-segments
        chains (atom [])                        ; [lvl len args] in emission order
        _ (with-redefs [splat-painter.seed/stroke-segments
                        (fn [& args]
                          (let [lvl (long (nth args 2))
                                [rows reason] (apply orig args)]
                            (when (seq rows)
                              (swap! chains conj [lvl (clojure.core/count rows)
                                                  (first rows) args]))
                            [(mapv (fn [r] (conj r lvl)) rows) reason]))]
            (layered-means dmap nf 0.56 7.5 0.47 0.47 2.4 0.91
                           [2.31 0.47 0.47 0.91] 547000 H W
                           (or (:blur img) px) (or (:blur-drift img) (or (:blur img) px))))
        segs   (first (with-redefs [splat-painter.seed/stroke-segments
                                    (fn [& args]
                                      (let [lvl (long (nth args 2))
                                            [rows reason] (apply orig args)]
                                        [(mapv (fn [r] (conj r lvl)) rows) reason]))]
                        [(layered-means dmap nf 0.56 7.5 0.47 0.47 2.4 0.91
                                        [2.31 0.47 0.47 0.91] 547000 H W
                                        (or (:blur img) px) (or (:blur-drift img) (or (:blur img) px)))]))
        starts (reductions + 0 (map second @chains))
        idx    (count (take-while #(<= % di) starts))
        start  (long (if (zero? idx) 0 (nth starts (dec idx))))
        len    (long (nth (nth @chains (dec idx)) 1))
        lvl    (long (nth (nth @chains (dec idx)) 0))
        cargs  (nth (nth @chains (dec idx)) 3)]
    (println (format "divergence chain: %d-th chain, lvl %d, records %d..%d"
                     idx lvl start (dec (+ start len))))
    (println (format "capture self-check: capture segs %d vs real cpu %d"
                     (clojure.core/count segs) (clojure.core/count cpu-splats)))
    (when (< 4654 (clojure.core/count segs))
      (println (format "  capture segs[4654]=%s cpu[4654]=%s"
                       (pr-str (:mean (nth segs 4654))) (pr-str (:mean (nth cpu-splats 4654))))))
    (println (format "cumsum@idx=%d (chains %d) start=%d (arg seed %.1f,%.1f)"
                     idx (clojure.core/count @chains) start
                     (double (nth cargs 3)) (double (nth cargs 4))))
    (println (format "args: %s"
                     (clojure.string/join " "
                       (map-indexed (fn [i a] (str i ":" (cond (number? a) (format "%.4f" (double a))
                                                               (map? a) "MAP" :else "OTHER")))
                                    cargs))))
    (println (format "captured chain first rows: %s"
                     (pr-str (take 3 (map (fn [r] [(double (nth r 0)) (double (nth r 1))])
                                          [(nth (nth @chains (dec idx)) 2)])))))
    (spit "/tmp/chain-args.edn"
          (pr-str {:lvl lvl :seed [(double (nth cargs 3)) (double (nth cargs 4))]
                   :ssz (double (nth cargs 5)) :D (double (nth cargs 6))
                   :sn (double (nth cargs 7)) :tn (double (nth cargs 8))
                   :ds (double (nth cargs 9)) :curvature (double (nth cargs 10))
                   :stroke (double (nth cargs 11)) :hd (long (nth cargs 12))
                   :wd (long (nth cargs 13)) :segs (long (nth cargs 14))
                   :stepf (double (nth cargs 15)) :bendf (double (nth cargs 16))
                   :hb (double (nth cargs 17)) :traw (double (nth cargs 18))
                   :sgate (double (nth cargs 19)) :iw (long (nth cargs 21))
                   :ih (long (nth cargs 22)) :lth (double (nth cargs 23))
                   :melt (double (nth cargs 24)) :mkind (nth cargs 25)
                   :gainv (double (nth cargs 26)) :bph (double (nth cargs 28))
                   :sideo (double (nth cargs 29)) :selong (double (nth cargs 30))}))
    (doseq [k (range start (min (+ start 40) (min (count gpu-splats) (+ start len))))]
      (let [[gx gy] (:mean (nth gpu-splats k))
            [cx cy] (:mean (nth cpu-splats k))
            [th coh] ((var splat-painter.seed/sample-fields) nf cx cy)
            ev ((var splat-painter.wavelet/edge-at) dmap cx cy)]
        (println (format "  [%d] GPU (%.2f, %.2f)  CPU (%.2f, %.2f)  a=%.2f | field@cpu %.1f coh %.3f ev %.3f"
                         k gx gy cx cy (double (or (:alpha (nth gpu-splats k)) 1.0))
                         (* 180.0 (/ th Math/PI)) coh ev))))))

(defn verify-parity
  "Run the CPU + GPU generation at `maxside` with the band push `mode` (:off or
   :shipped). Returns {:gpu n :cpu n :max-div px :di idx} where :max-div is the
   position difference at the first-divergence index (the razor guard: pre-fix
   it jumped ~2.9px at dying ridges, post-fix it stays under ~1px)."
  [path maxside mode]
  (let [img0 (image/load-image path maxside)
        img  (fields/prepare img0)]
    (reset! sideo-mode (if (= mode "off") :off :shipped))
    (let [cpu  (:splats (cpu-field img))
          gpu  (run-gpu img)
          gsplats (:splats gpu) n (:n gpu)
          di   (first-divergence gsplats cpu)
          max-div (if di
                    (let [[gx gy] (:mean (nth gsplats di))
                          [cx cy] (:mean (nth cpu di))]
                      (max (Math/abs (- gx cx)) (Math/abs (- gy cy))))
                    0.0)]
      {:gpu n :cpu (count cpu) :max-div max-div :di di})))

(defn -main
  [& [path maxside mode chains?]]
  (let [path (or path "img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg")
        ms   (if (and maxside (not= maxside "0"))
               (long (Double/parseDouble maxside)) nil)
        img0 (image/load-image path ms)
        _    (println (format "image %s -> %dx%d" path (:width img0) (:height img0)))
        img  (fields/prepare img0)]
    (reset! sideo-mode (if (= mode "off") :off :shipped))
    (println (format "band side push: %s" @sideo-mode))
    (let [lp   (seed/layer-params (:detail img) 0.56 7.5 0.47 0.47 2.4
                                   [2.31 0.47 0.47 0.91] 547000
                                   (:height img) (:width img))
          band (first (filter :band (:levels lp)))]
      (println (format "band level: ssz %.2f sp %.2f nx %d sideo %.2f selong %.1f"
                       (double (:ssz band)) (double (:sp band)) (long (:nx band))
                       (double (:sideo band)) (double (:selong band)))))
    (let [t0 (System/nanoTime)
          cpu (:splats (cpu-field img))
          t1 (System/nanoTime)
          gpu (run-gpu img)
          t2 (System/nanoTime)
          gsplats (:splats gpu) n (:n gpu)
          [gx gy gd gc ga] (splat-stats gsplats)
          [cx cy cd cc ca] (splat-stats cpu)
          di (first-divergence gsplats cpu)]
      (println (format "cpu %d splats in %.1fs; gpu %d splats (of %d candidates) in %.1fs"
                       (count cpu) (/ (- t1 t0) 1e9) n (:total gpu) (/ (- t2 t1) 1e9)))
      (println (format "gpu-verify: count GPU %d / CPU %d  (%.1f%%)"
                       n (count cpu) (* 100.0 (/ (double n) (max 1 (count cpu))))))
      (println (format "  Sigma mean-x  GPU %.1f  CPU %.1f" gx cx))
      (println (format "  Sigma mean-y  GPU %.1f  CPU %.1f" gy cy))
      (println (format "  Sigma det     GPU %.1f  CPU %.1f" gd cd))
      (println (format "  Sigma colour  GPU %.2f  CPU %.2f" gc cc))
      (println (format "  Sigma alpha   GPU %.2f  CPU %.2f" ga ca))
      (println (format "  first-divergence idx: %s of %d" di (min n (count cpu))))
      (when di
        (doseq [k (range (max 0 (- di 1)) (min (min n (count cpu)) (+ di 3)))]
          (println (format "   [%d] GPU %s a=%.2f | CPU %s a=%.2f" k
                           (pr-str (mapv #(format "%.1f" (double %)) (:mean (nth gsplats k))))
                           (double (:alpha (nth gsplats k)))
                           (pr-str (mapv #(format "%.1f" (double %)) (:mean (nth cpu k))))
                           (double (or (:alpha (nth cpu k)) 1.0))))))
      (when (= chains? "chains")
        (print-chains img))
      (when (= chains? "trace")
        (when di (trace-divergence img cpu gsplats di))))))
