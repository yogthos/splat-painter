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
            [splat-painter.structure :as structure]))

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
              ;; 2 is the tensor radius, 3 the light blur, 8 the flow/heavy end —
              ;; the three the real pipeline actually asks for.
              (doseq [r [2 3 8]]
                (gf/box-blur! ctx progs src dst tmp W H r)
                (let [got  (gf/read-channel ctx dst W H 0)
                      want (structure/box-blur red H W r)
                      d    (max-diff got want)]
                  (is (< d float-tol) (str "radius " r " max diff " d)))))))))))

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
