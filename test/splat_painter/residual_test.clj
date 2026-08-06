(ns splat-painter.residual-test
  "The residual field is what aims a repaint layer. Two properties have to hold or
   the feature is either inert or a regression:

   - weights are >= 1 EVERYWHERE. The field boosts unresolved regions; it never
     suppresses converged ones. Suppression would lower `dv` there, and dv also
     drives the per-stroke fidelity D — so a converged region would start taking
     warped, less faithful strokes as a reward for being right.
   - strength 0, or a composite equal to the original, leaves the placement maps
     BIT-identical. That is the off switch and the first-pass guarantee."
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.residual :as resid]))

(defn- rgb-image
  "H*W*3 pixel array from a fn of [row col] -> [r g b]."
  [H W f]
  (let [p (double-array (* H W 3))]
    (dotimes [r H]
      (dotimes [c W]
        (let [[rr gg bb] (f r c)
              b (* 3 (+ (* r W) c))]
          (aset-double p b rr) (aset-double p (+ b 1) gg) (aset-double p (+ b 2) bb))))
    p))

(def ^:private H 64)
(def ^:private W 64)

(defn- flat [v] (rgb-image H W (fn [_ _] [v v v])))

(defn- right-half-wrong
  "Original is flat 0.5; the composite matches on the left half and is 0.5+e on the right."
  [e]
  (rgb-image H W (fn [_ c] (let [v (if (< c (quot W 2)) 0.5 (+ 0.5 e))] [v v v]))))

(deftest identical-images-weigh-one
  (testing "no error anywhere -> every weight is exactly 1.0"
    (let [w (resid/weight-map (flat 0.5) (flat 0.5) H W 32 32 1.0)]
      (is (= 1024 (alength ^doubles w)))
      (is (every? #(= 1.0 %) (seq w))))))

(deftest zero-strength-weighs-one
  (testing "strength 0 is the off switch even with a large residual"
    (let [w (resid/weight-map (right-half-wrong 0.4) (flat 0.5) H W 32 32 0.0)]
      (is (every? #(= 1.0 %) (seq w))))))

(deftest weights-never-suppress
  (testing "every weight is >= 1.0 — converged regions keep today's map exactly"
    (let [w (resid/weight-map (right-half-wrong 0.3) (flat 0.5) H W 32 32 1.5)]
      (is (every? #(>= (double %) 1.0) (seq w))))))

(deftest error-region-outweighs-converged-region
  (testing "the half that misses the original is boosted; the half that matches is not"
    (let [^doubles w (resid/weight-map (right-half-wrong 0.3) (flat 0.5) H W 32 32 1.0)
          sw   32
          side (fn [c0 c1]
                 (let [n (* 32 (- c1 c0))]
                   (/ (reduce + 0.0 (for [r (range 32) c (range c0 c1)] (aget w (+ (* r sw) c))))
                      (double n))))
          ;; sample away from the seam: the field is blurred, so the middle columns
          ;; are a legitimate ramp between the two regimes, not a failure.
          left  (side 0 12)
          right (side 20 32)]
      (is (> right (* 1.5 left)) (format "left %.3f right %.3f" left right))
      (is (< left 1.05) (format "converged half should sit at ~1.0, got %.3f" left)))))

(deftest weight-is-capped
  (testing "an extreme local error cannot boost without bound"
    (let [;; one hot cell in an otherwise perfect frame: relative error is enormous
          comp' (rgb-image H W (fn [r c] (let [v (if (and (< r 4) (< c 4)) 1.0 0.5)] [v v v])))
          w     (resid/weight-map comp' (flat 0.5) H W 32 32 1.0)]
      (is (<= (apply max (seq w)) (+ 1.0 resid/weight-cap 1e-9))))))

(deftest apply-weights-scales-the-density-maps-only
  (testing "the four density channels scale; subject and the grid metadata do not"
    (let [n    16
          mk   (fn [v] (double-array n v))
          dmap {:h 4 :w 4 :dmax 1.0 :src-h 64 :src-w 64
                :detail (mk 0.2) :sharp (mk 0.3) :mid (mk 0.4) :edge (mk 0.5)
                :subject (mk 0.6)}
          w    (double-array n 2.0)
          out  (resid/weighted-dmap dmap w)]
      (is (every? #(= 0.4 %) (seq (:detail out))))
      (is (every? #(= 0.6 %) (seq (:sharp out))))
      (is (every? #(= 0.8 %) (seq (:mid out))))
      (is (every? #(= 1.0 %) (seq (:edge out))))
      (is (every? #(= 0.6 %) (seq (:subject out))) "subject is a GATE, not a density")
      (is (= 1.0 (:dmax out)) "thresholds stay in the same units")
      (is (= [4 4 64 64] [(:h out) (:w out) (:src-h out) (:src-w out)]))
      (is (every? #(= 0.2 %) (seq (:detail dmap))) "the input dmap is not mutated"))))

(deftest unit-weights-leave-the-maps-untouched
  (testing "an all-ones weight reproduces the input arrays exactly"
    (let [n    16
          orig (double-array (map #(/ (double %) 17.0) (range n)))
          dmap {:h 4 :w 4 :dmax 1.0 :detail orig :sharp orig :mid orig :edge orig}
          out  (resid/weighted-dmap dmap (double-array n 1.0))]
      (is (= (seq orig) (seq (:detail out))))
      (is (= (seq orig) (seq (:sharp out)))))))

(deftest resampling-preserves-the-error-layout
  (testing "the weight grid can be coarser than the image without losing where the error is"
    (let [^doubles w (resid/weight-map (right-half-wrong 0.3) (flat 0.5) H W 8 8 1.0)]
      (is (= 64 (alength w)))
      (is (> (aget w (+ (* 4 8) 7)) (aget w (+ (* 4 8) 0)))))))

(deftest strength-scales-the-boost
  (testing "double the strength, double the excess over 1.0"
    (let [c (right-half-wrong 0.3)
          o (flat 0.5)
          hi (fn [s] (apply max (seq (resid/weight-map c o H W 32 32 s))))
          a  (- (hi 0.5) 1.0)
          b  (- (hi 1.0) 1.0)]
      (is (> a 0.0))
      (is (< (Math/abs (- b (* 2.0 a))) 1e-9)))))
