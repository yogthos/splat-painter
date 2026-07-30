(ns splat-painter.gpu-fields-test
  "Holds the GPU field passes to their CPU twins in splat-painter.structure.

   fields-test pins the CPU builders so an optimization can't silently repaint
   every picture. This is the other half of that guard: the GPU passes are a
   second implementation of the same math, and two implementations drift unless
   something compares them. glimmer-gl.offscreen gives a context with no window,
   so that comparison runs under `jolt -M:test` instead of only inside the app.

   Tolerances are loose next to fields-test's 1e-6 because the GPU computes in
   binary32 and the CPU in binary64 — the difference is rounding, not behaviour.
   A real divergence (wrong radius, wrong edge handling, a transposed pass) is
   orders of magnitude larger than the bounds here."
  (:require [clojure.test :refer [deftest is testing]]
            [glimmer-gl.gl :as gl]
            [glimmer-gl.offscreen :as off]
            [splat-painter.gpu-fields :as gf]
            [splat-painter.image :as image]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]))

(def ^:private fixture "test/splat_painter/fixtures/eye.jpeg")

(defn- img [] (image/load-image fixture 64))

(defn- chan-of
  "Channel `off` of an image's interleaved RGB :pixels as a fresh ^doubles."
  [im o]
  (let [^doubles px (:pixels im)
        n (* (long (:height im)) (long (:width im)))
        a (double-array n)]
    (dotimes [i n] (aset a i (aget px (+ (* 3 i) (long o)))))
    a))

(defn- max-diff [^doubles a ^doubles b]
  (let [n (min (alength a) (alength b))]
    (loop [i 0 mx 0.0]
      (if (>= i n) mx
          (recur (inc i) (max mx (Math/abs (- (aget a i) (aget b i)))))))))

;; Single-precision rounding over a 2r+1 tap sum. Empirically ~3e-7 at these
;; radii on values in [0,1]; 1e-5 leaves room for a different driver's summation
;; order without leaving room for an actual bug.
(def ^:private float-tol 1e-5)

;; A test that is allowed to skip has to say which way it went, or a broken
;; context reads exactly like a passing suite. Announce the decision once.
(defonce ^:private announced (atom false))

(defn- announce! [ctx]
  (when (compare-and-set! announced false true)
    (if-let [err (:error ctx)]
      (println "gpu-fields-test: SKIPPED, no offscreen GL —" err)
      (let [[maj mnr] (:version ctx)]
        (println (format "gpu-fields-test: GL %d.%d on %s"
                         (long maj) (long mnr)
                         (or (gl/gl-get-string* gl/GL-RENDERER) "?")))))))

(defn- with-gl
  "Run `f` on an offscreen context. Returns :skipped when there is no context, so
   a test asserts nothing rather than failing on a machine with no display."
  [f]
  (let [ctx (off/ensure-current!)]
    (announce! ctx)
    (if (:error ctx) :skipped (f))))

(deftest box-blur-matches-the-cpu
  (testing "the separable box blur every other GPU field is built on"
    (with-gl
      (fn []
        (let [im   (img)
              H    (long (:height im)) W (long (:width im))
              ctx  (gf/make-ctx)
              progs (gf/build-programs)]
          (is (some? progs) "field shaders compile")
          (when progs
            (let [src (gf/upload-rgb! im)
                  dst (gf/new-scratch W H)
                  tmp (gf/new-scratch W H)
                  red (chan-of im 0)]
              ;; 2 is the tensor radius, 3 the light blur, 8 the flow/heavy end.
              ;; 70 is past the old fixed +/-64 loop cap and wider than the image,
              ;; so it also pins that a huge radius clamps instead of truncating —
              ;; the placement map blurs at min(sh,sw)/8, ~96 at full size.
              (doseq [r [2 3 8 70]]
                (gf/box-blur! ctx progs src dst tmp W H r)
                (let [got  (gf/read-channel ctx dst W H 0)
                      want (structure/box-blur red H W r)
                      d    (max-diff got want)]
                  (is (< d float-tol) (str "radius " r " max diff " d)))))))))))

;; --- Di Zenzo tensor ---------------------------------------------------------

(defn- rel-max-diff
  "max |a-b| / (|want| + eps). The tensor's magnitudes span orders of magnitude
   across an image — a flat patch is ~1e-6 where an edge is ~10 — so an absolute
   bound would either pass everything or fail the edges on pure rounding."
  [^doubles got ^doubles want eps]
  (let [n (min (alength got) (alength want))]
    (loop [i 0 mx 0.0]
      (if (>= i n) mx
          (recur (inc i)
                 (max mx (/ (Math/abs (- (aget got i) (aget want i)))
                            (+ (Math/abs (aget want i)) (double eps)))))))))

(defn- angle-max-diff
  "Largest orientation difference, mod π, over texels where `weight` clears
   `thresh`. Theta is an axis, not a vector — θ and θ+π are the same stroke
   direction — and where the tensor is ~0 the angle is atan2(0,0), i.e. noise
   amplified by float32 rounding. Comparing it there would measure nothing."
  [^doubles got ^doubles want ^doubles weight thresh]
  (let [n (min (alength got) (alength want))
        pi Math/PI]
    (loop [i 0 mx 0.0 cnt 0]
      (if (>= i n) [mx cnt]
          (if (< (aget weight i) (double thresh))
            (recur (inc i) mx cnt)
            (let [d (Math/abs (- (aget got i) (aget want i)))
                  d (- d (* pi (Math/floor (/ d pi))))    ; into [0, π)
                  d (min d (- pi d))]                     ; and fold the wrap
              (recur (inc i) (max mx d) (inc cnt))))))))

(defn- run-analyze [im max-side]
  (let [H (long (:height im)) W (long (:width im))
        ctx   (gf/make-ctx)
        progs (gf/build-programs)]
    (when progs
      (let [src (gf/upload-rgb! im)
            r   (gf/analyze! ctx progs src H W max-side)
            {:keys [h w]} r]
        {:ctx ctx :h h :w w
         :theta     (gf/read-channel ctx (:eigen r) w h 0)
         :coherence (gf/read-channel ctx (:eigen r) w h 1)
         :grad2     (gf/read-channel ctx (:eigen r) w h 2)
         :flow-theta (gf/read-channel ctx (:flow r) w h 0)
         :flow-str   (gf/read-channel ctx (:flow r) w h 1)}))))

(defn- check-analyze [max-side]
  (let [im   (img)
        want (structure/analyze im max-side)
        got  (run-analyze im max-side)]
    (is (some? got) "field shaders compile")
    (when got
      (is (= [(:h want) (:w want)] [(:h got) (:w got)])
          "GPU and CPU agree on the reduced tensor grid")
      (testing "tensor energy (a+b), the magnitude everything else is scaled by"
        (let [d (rel-max-diff (:grad2 got) (:grad2 want) 1e-4)]
          (is (< d 1e-3) (str "grad2 relative diff " d))))
      (testing "coherence"
        (let [d (max-diff (:coherence got) (:coherence want))]
          (is (< d 1e-3) (str "coherence max diff " d))))
      (testing "orientation, where there is enough energy to have one"
        ;; gmax-relative gate: only texels carrying real edge structure.
        (let [gmax (areduce ^doubles (:grad2 want) i m 0.0
                            (max m (aget ^doubles (:grad2 want) i)))
              [d cnt] (angle-max-diff (:theta got) (:theta want)
                                      (:grad2 want) (* 1e-3 gmax))]
          (is (pos? cnt) "some texels clear the energy gate")
          (is (< d 1e-3) (str "theta max angular diff " d " over " cnt " texels"))))
      (testing "diffused flow tensor (the edges-seed-the-flow field)"
        (let [[d cnt] (angle-max-diff (:flow-theta got) (:flow-theta want)
                                      (:flow-str want) 0.05)]
          (is (pos? cnt) "some texels clear the flow-strength gate")
          (is (< d 1e-2) (str "flow-theta max angular diff " d " over " cnt " texels")))
        (let [d (max-diff (:flow-str got) (:flow-str want))]
          (is (< d 1e-3) (str "flow-str max diff " d)))))))

(deftest tensor-matches-the-cpu-at-full-res
  (testing "gamma, Sobel, radius-2 smoothing and the eigen decomposition"
    ;; 768 is analyze's default cap and the fixture is 64, so nothing downscales:
    ;; this isolates the maths from the resampling.
    (with-gl #(check-analyze 768))))

(deftest tensor-matches-the-cpu-when-downscaled
  (testing "the nearest-neighbour reduction analyze does above max-side"
    ;; 40 gives a 1.6 ratio — not a power of two, so a sloppy resample shows up.
    (with-gl #(check-analyze 40))))

;; --- reductions --------------------------------------------------------------

(deftest stats-reduce-matches-a-plain-loop
  (testing "the log-step max/sum fold placement-map's normalization rides on"
    ;; Worth its own test: every fused band divides by a dmax and a mean from
    ;; this, so a wrong fold would skew all three maps by one scale factor and
    ;; still look plausible. Odd dimensions are the interesting case — the ragged
    ;; edge must be skipped, not clamped, or texels get double-counted into sum.
    (with-gl
      (fn []
        (let [im    (img)
              H     (long (:height im)) W (long (:width im))
              ctx   (gf/make-ctx)
              progs (gf/build-programs)]
          (when progs
            (let [src  (gf/upload-rgb! im)
                  red  (chan-of im 0)
                  [st sc] (gf/stats! ctx progs src W H 0)
                  got  (gf/read-channel ctx st 1 1 0)
                  got2 (gf/read-channel ctx st 1 1 1)
                  want-max (areduce ^doubles red i m 0.0 (max m (aget ^doubles red i)))
                  want-sum (areduce ^doubles red i s 0.0 (+ s (aget ^doubles red i)))]
              (is (< (Math/abs (- (aget ^doubles got 0) want-max)) 1e-5)
                  (str "max " (aget ^doubles got 0) " vs " want-max))
              ;; relative: a 4096-texel sum is ~1e3, so absolute 1e-5 would be
              ;; stricter than float32 can carry
              (is (< (/ (Math/abs (- (aget ^doubles got2 0) want-sum)) want-sum) 1e-5)
                  (str "sum " (aget ^doubles got2 0) " vs " want-sum))
              (is (seq sc) "the fold actually took intermediate steps"))))))))

;; --- Haar placement map ------------------------------------------------------

(defn- run-placement [im max-side]
  (let [H (long (:height im)) W (long (:width im))
        ctx   (gf/make-ctx)
        progs (gf/build-programs)]
    (when progs
      (let [src (gf/upload-rgb! im)
            t   (gf/analyze! ctx progs src H W max-side)
            p   (gf/placement-map! ctx progs src (:eigen t) H W (:h t) (:w t) max-side 4)
            {:keys [h w]} p]
        (into {:h h :w w}
              (map (fn [k] [k (gf/read-channel ctx (get p k) w h 0)]))
              [:detail :sharp :mid :edge :subject])))))

(defn- check-placement [max-side]
  (let [im   (img)
        sf   (structure/analyze im max-side)
        want (wavelet/placement-map im sf max-side 4)
        got  (run-placement im max-side)]
    (is (some? got) "field shaders compile")
    (when got
      (is (= [(:h want) (:w want)] [(:h got) (:w got)]) "same placement grid")
      ;; These maps are clamped into [0,1] and drive thresholds, so an absolute
      ;; bound is the meaningful one — unlike the tensor, nothing here spans
      ;; orders of magnitude.
      (doseq [k [:detail :sharp :mid :edge :subject]]
        (testing (name k)
          (let [d (max-diff (get got k) (get want k))]
            (is (< d 2e-3) (str (name k) " max diff " d))))))))

(deftest placement-map-matches-the-cpu-at-full-res
  (testing "luma+gamma, the Haar ladder, band snapshots, fusion and the dilation"
    (with-gl #(check-placement 768))))

(deftest placement-map-matches-the-cpu-when-downscaled
  (testing "the same with the nearest reduction in play"
    (with-gl #(check-placement 40))))

(deftest box-blur-replicates-edges
  (testing "corner pixels use the clamped window, not a zero-padded one"
    ;; The CPU clamps its sliding window at the border; a shader that samples
    ;; outside with wrap or zero produces a dark frame that a whole-image max
    ;; diff can hide among 4096 pixels. Compare the corners on their own.
    (with-gl
      (fn []
        (let [im   (img)
              H    (long (:height im)) W (long (:width im))
              ctx  (gf/make-ctx)
              progs (gf/build-programs)]
          (when progs
            (let [src  (gf/upload-rgb! im)
                  dst  (gf/new-scratch W H)
                  tmp  (gf/new-scratch W H)
                  _    (gf/box-blur! ctx progs src dst tmp W H 8)
                  got  (gf/read-channel ctx dst W H 0)
                  want (structure/box-blur (chan-of im 0) H W 8)
                  corners [0 (dec W) (* (dec H) W) (dec (* H W))]]
              (doseq [i corners]
                (is (< (Math/abs (- (aget ^doubles got i) (aget ^doubles want i)))
                       float-tol)
                    (str "corner index " i))))))))))
