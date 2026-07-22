(ns splat-painter.noise-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.noise :as n]))

(defn- approx= [tol a b] (< (Math/abs (- (double a) (double b))) tol))

(deftest output-in-unit-range
  (testing "noise2/noise3 stay within [0,1]"
    (is (every? (fn [_]
                  (let [x (* 50 (Math/random)) y (* 50 (Math/random)) z (* 50 (Math/random))]
                    (and (<= 0.0 (n/noise2 x y) 1.0)
                         (<= 0.0 (n/noise3 x y z) 1.0))))
                (range 500)))))

(deftest zero-at-integer-lattice
  ;; classic Perlin is 0 at integer lattice points → remapped to 0.5
  (testing "noise ≈ 0.5 at integer coordinates"
    (is (approx= 1e-9 0.5 (n/noise2 5.0 7.0)))
    (is (approx= 1e-9 0.5 (n/noise3 1.0 2.0 3.0)))))

(deftest smooth
  (testing "nearby inputs give nearby outputs"
    (is (every? (fn [_]
                  (let [x (* 20 (Math/random)) y (* 20 (Math/random))]
                    (approx= 0.06 (n/noise2 x y) (n/noise2 (+ x 0.01) (+ y 0.01)))))
                (range 200)))))

(deftest not-constant
  (testing "the field actually varies across space"
    (let [samples (for [i (range 40)] (n/noise2 (* i 1.7) (* i 0.9)))]
      (is (> (- (apply max samples) (apply min samples)) 0.2)))))

(deftest deterministic
  (testing "fixed permutation → identical output for identical input"
    (is (= (n/noise2 3.14 2.72) (n/noise2 3.14 2.72)))
    (is (= (n/noise3 1.5 9.5 4.25) (n/noise3 1.5 9.5 4.25)))))
