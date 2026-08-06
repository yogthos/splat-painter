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
            [splat-painter.fields :as fields]
            [splat-painter.seed :as seed]
            [splat-painter.gen :as gen]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.residual :as residual]
            [jolt.ffi :as ffi]))

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

;; Di Zenzo tensor

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

;; reductions

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

;; Haar placement map

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

;; binned colour fields

(defn- rgb-max-diff
  "max |a-b| between a GPU RGBA texture's three colour channels and a flat
   H*W*3 CPU array."
  [ctx tex w h ^doubles want]
  (reduce max 0.0
          (for [c (range 3)]
            (let [^doubles got (gf/read-channel ctx tex w h c)]
              (areduce got i m 0.0
                       (max m (Math/abs (- (aget got i)
                                           (aget want (+ (* 3 i) c))))))))))

(deftest bilateral-blur-matches-the-cpu
  (testing "the edge-aware colour field, K=9 luma bins"
    (with-gl
      (fn []
        (let [im (img) H (long (:height im)) W (long (:width im))
              ctx (gf/make-ctx) progs (gf/build-programs)]
          (when progs
            (let [src (gf/upload-rgb! im)
                  dst (gf/new-scratch W H)]
              (gf/bilateral-blur! ctx progs src dst W H 3)
              (let [d (rgb-max-diff ctx dst W H (structure/bilateral-blur im 3))]
                (is (< d 1e-4) (str "bilateral max diff " d))))))))))

(deftest dominant-blur-matches-the-cpu
  (testing "the modal-tone coverage colour, including the reduced-res path"
    (with-gl
      (fn []
        (let [im (img) H (long (:height im)) W (long (:width im))
              ctx (gf/make-ctx) progs (gf/build-programs)]
          (when progs
            (let [src (gf/upload-rgb! im)]
              ;; radius 2 keeps step=1 (full res); 6 forces the downsample +
              ;; bilinear expand, which is the path the app actually takes
              (doseq [r [2 6]]
                (let [dst (gf/new-scratch W H)]
                  (gf/dominant-blur! ctx progs src dst W H r)
                  (let [d (rgb-max-diff ctx dst W H (structure/dominant-blur im r))]
                    ;; p=4 means pow() on a blurred weight, so the float32 gap is
                    ;; wider here than for the linear fields
                    (is (< d 1e-3) (str "dominant r=" r " max diff " d))))))))))))

(deftest noise-fields-match-the-cpu
  (testing "the baked orientation field, both ends of the Swirl dial"
    (with-gl
      (fn []
        (let [im (img) H (long (:height im)) W (long (:width im))
              ctx (gf/make-ctx) progs (gf/build-programs)]
          (when progs
            (let [src  (gf/upload-rgb! im)
                  t    (gf/analyze! ctx progs src H W 768)
                  perm (gen/upload-perm!)
                  [n1 n0] (gf/noise-fields! ctx progs (:eigen t) (:flow t) perm
                                            (:w t) (:h t) W H)
                  want (seed/prep-noise (structure/analyze im 768))
                  w (:w t) h (:h t)]
              ;; cos2θ/sin2θ, not θ — that IS the stored representation, and
              ;; comparing components sidesteps the wrap entirely
              (doseq [[tex ch k] [[n1 0 :c2] [n1 1 :s2] [n1 2 :coherence]
                                  [n0 0 :c2s] [n0 1 :s2s]]]
                (let [d (max-diff (gf/read-channel ctx tex w h ch) (get want k))]
                  ;; Perlin runs in float32 here against float64 on the CPU, and
                  ;; the result goes through two atan2 blends. The bound is 1e-2,
                  ;; not the 1e-3 it was: 1e-3 was calibrated on ONE GL
                  ;; implementation (Apple M1 Max hardware) and the CI macOS runner
                  ;; uses the Apple SOFTWARE Renderer, whose float32 trig differs
                  ;; enough to reach 3.9e-3 — c2 3.26e-3, s2 3.58e-3, c2s 3.91e-3,
                  ;; s2s 3.34e-3, while :coherence (no trig) stayed inside 1e-3.
                  ;; These are cos2θ/sin2θ in [-1,1], so 1e-2 is still ~0.3° of
                  ;; angle: a real defect here (transposed components, a dropped
                  ;; blend term) diverges by O(1), not by 1e-2. Mutation-checked at
                  ;; this bound — comparing n1 channel 0 against :s2 instead of :c2
                  ;; fails at 1.41.
                  (is (< d 1e-2) (str (name k) " max diff " d)))))))))))

(defn- solid-image
  "An image with a large perfectly uniform region — the case a photograph
   fixture does not have. Every Sobel tap there is exactly 0, which is where
   atan(0,0) lives."
  [H W]
  {:height H :width W :channels 3
   :pixels (let [a (double-array (* H W 3))]
             ;; left half solid black, right half a gentle ramp, so there is both
             ;; a dead-flat area and a real edge between them
             (dotimes [i (* H W)]
               (let [x (mod i W)
                     v (if (< x (quot W 2)) 0.0 (/ (double (- x (quot W 2))) W))]
                 (aset a (* 3 i) v) (aset a (+ (* 3 i) 1) v) (aset a (+ (* 3 i) 2) v)))
             a)})

(deftest no-field-is-ever-nan
  (testing "flat regions do not produce NaN anywhere in the chain"
    ;; This is deliberately UNGATED. The theta comparison above only looks where
    ;; grad2 clears a fraction of gmax — which is exactly where a zero tensor
    ;; ISN'T — so it cannot see this. A NaN theta poisons a splat's covariance
    ;; and propagates through the geometry shader.
    (with-gl
      (fn []
        (let [im (solid-image 64 64)
              H 64 W 64
              ctx (gf/make-ctx) progs (gf/build-programs)]
          (when progs
            (let [src  (gf/upload-rgb! im)
                  t    (gf/analyze! ctx progs src H W 768)
                  pm   (gf/placement-map! ctx progs src (:eigen t) H W (:h t) (:w t) 768 4)
                  perm (gen/upload-perm!)
                  [n1 n0] (gf/noise-fields! ctx progs (:eigen t) (:flow t) perm
                                            (:w t) (:h t) W H)
                  nan? (fn [^doubles a] (areduce a i acc false
                                                 (or acc (Double/isNaN (aget a i)))))]
              (doseq [[label tex w h] [["eigen" (:eigen t) (:w t) (:h t)]
                                       ["flow" (:flow t) (:w t) (:h t)]
                                       ["noise" n1 (:w t) (:h t)]
                                       ["noise-swirl0" n0 (:w t) (:h t)]
                                       ["detail" (:detail pm) (:w pm) (:h pm)]
                                       ["sharp" (:sharp pm) (:w pm) (:h pm)]
                                       ["mid" (:mid pm) (:w pm) (:h pm)]
                                       ["edge" (:edge pm) (:w pm) (:h pm)]
                                       ["subject" (:subject pm) (:w pm) (:h pm)]]
                      c (range 3)]
                (is (not (nan? (gf/read-channel ctx tex w h c)))
                    (str label " channel " c " contains NaN"))))))))))

(deftest build-fields-restores-the-viewport
  (testing "a pass chain hands back the viewport it was called with"
    ;; Every pass resizes the viewport to its own target and build-fields! runs
    ;; from inside gpu-draw!, which is about to draw the splat quads. Leaving the
    ;; viewport at the last pass's size painted the image at that size in the
    ;; bottom-left corner of the pane. Nothing headless draws to a screen, so
    ;; this asserts the invariant directly rather than the symptom.
    (with-gl
      (fn []
        (let [im (img) H (long (:height im)) W (long (:width im))
              progs (gf/build-programs)]
          (when progs
            (gl/gl-viewport 0 0 640 480)
            (let [ctx  (gf/make-ctx)
                  perm (gen/upload-perm!)]
              (gf/build-fields! ctx progs im perm)
              (let [p (ffi/alloc (* 4 (ffi/sizeof :int)))]
                (gl/gl-get-integerv 0x0BA2 p)
                (let [vp (mapv #(ffi/read p :int (* 4 %)) (range 4))]
                  (ffi/free p)
                  (is (= [0 0 640 480] vp)
                      (str "viewport left at " vp " instead of the caller's"))))
              (gf/free-ctx! ctx))))))))

(deftest heavy-radius-mirrors-the-cpu
  (testing "the coverage colour window is the same on both paths"
    ;; gpu-fields duplicates fields/heavy-radius rather than requiring it, to keep
    ;; this namespace off the CPU analysis stack. Nothing enforced that they agree,
    ;; and they silently diverging is not hypothetical: editing fields/heavy-radius
    ;; to chase a halo produced a byte-identical render, because the shipping path
    ;; reads the copy. It is also the window that sets how far coverage strokes
    ;; pull subject colour into the background, so a mismatch moves the artifact.
    ;; No GL needed — pure arithmetic.
    (doseq [h [64 512 683 1024 2048 100 6]]
      (is (= (fields/heavy-radius {:height h}) (gf/heavy-radius h))
          (str "height " h)))))

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

;; residual weighting

(defn- residual-source
  "The fixture as a repaint layer's source: same pixels, plus an :original that
   matches on the left half and misses badly on the right."
  [im]
  (let [^doubles px (:pixels im)
        W   (long (:width im))
        or* (double-array (alength px))]
    (System/arraycopy px 0 or* 0 (alength px))
    (dotimes [i (quot (alength px) 3)]
      (when (>= (mod i W) (quot W 2))
        (let [b (* 3 i)]
          (dotimes [c 3] (aset or* (+ b c) (max 0.0 (- (aget px (+ b c)) 0.35)))))))
    (assoc im :original or*)))

(deftest residual-aim-matches-the-cpu
  (testing "build-fields! aims the map, and re-uploads what the budget solve reads"
    ;; The two field builders reach the SAME candidate threshold and the SAME budget
    ;; solve, so a residual applied on one path and not the other (or applied to
    ;; different channels) silently paints two different pictures depending on
    ;; whether the field shaders compiled. Nothing else pins that: the placement-map
    ;; tests all run without an :original, where this code path does not execute.
    (with-gl
      (fn []
        (let [im    (residual-source (img))
              progs (gf/build-programs)]
          (when progs
            (let [ctx  (gf/make-ctx)
                  perm (gen/upload-perm!)
                  got  (:dmap (gf/build-fields! ctx progs im perm))
                  base (wavelet/placement-map im (structure/analyze im) 768 4)
                  w    (residual/weights-for im (:h base) (:w base))
                  want (residual/aimed-dmap base w)]
              (gf/free-ctx! ctx)
              (is (some? w) "the fixture produces a residual")
              (is (> (apply max (seq w)) 1.5) "and a boost big enough to be readable")
              ;; the aim runs on the CPU on both paths now, so the only difference
              ;; left is the float32 round trip through the re-uploaded texture and
              ;; the 2e-3 the unweighted maps already differ by
              (doseq [k [:detail :sharp :mid :edge]]
                (testing (name k)
                  (let [d (max-diff (get got k) (get want k))]
                    (is (< d 2e-3) (str (name k) " max diff " d)))))
              (is (< (max-diff (:subject got) (:subject base)) 2e-3)
                  "subjectness is a gate — the aim must leave it alone")
              ;; the budget solve reads these arrays and the geometry shader reads
              ;; the texture they were uploaded into — a divergence here would size
              ;; the candidate pools for a different picture than the one painted
              (is (> (reduce max 0.0 (map #(Math/abs (- %1 %2))
                                          (seq (:detail base)) (seq (:detail got))))
                     1e-3)
                  "the aim actually moved the map"))))))))

(deftest residual-aim-is-skipped-without-an-original
  (testing "a first pass gets the unweighted map, byte for byte the same code path"
    (with-gl
      (fn []
        (let [im    (img)
              progs (gf/build-programs)]
          (when progs
            (let [ctx  (gf/make-ctx)
                  perm (gen/upload-perm!)
                  got  (:dmap (gf/build-fields! ctx progs im perm))
                  want (wavelet/placement-map im (structure/analyze im) 768 4)]
              (gf/free-ctx! ctx)
              (is (nil? (residual/weights-for im (:h want) (:w want))))
              (doseq [k [:detail :sharp :mid :edge]]
                (is (< (max-diff (get got k) (get want k)) 2e-3) (name k))))))))))
