(ns splat-painter.wavelet-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.wavelet :as w]))

(defn- gray-img [H W f]
  {:height H :width W :channels 3
   :pixels (double-array (mapcat (fn [x]
                                   (mapcat (fn [y] (let [g (double (f x y))] [g g g]))
                                           (range W)))
                                 (range H)))})

(deftest detail-high-in-texture-low-in-flat
  (testing "checkerboard half reads high detail, flat half ~0"
    (let [H 32 W 32
          img (gray-img H W (fn [x y] (if (< y 16)
                                        0.5                          ; flat left half
                                        (if (even? (+ x y)) 0.0 1.0)))) ; textured right half
          dm  (w/detail-map img 64 4)]
      (is (> (:dmax dm) 0.0))
      (is (> (w/detail-at dm 16 24) 0.3) "textured region should read high detail")
      (is (< (w/detail-at dm 16 6) 0.15) "flat region should read low detail"))))

(deftest flat-image-has-no-detail
  (testing "solid image → dmax 0, detail-at 0 everywhere"
    (let [img (gray-img 16 16 (fn [_ _] 0.4))
          dm  (w/detail-map img 64 4)]
      (is (= 0.0 (:dmax dm)))
      (is (= 0.0 (w/detail-at dm 8 8))))))

(deftest detail-at-in-range
  (testing "normalized detail stays in [0,1] over a gradient"
    (let [img (gray-img 24 24 (fn [x y] (/ (double (+ x y)) 48.0)))
          dm  (w/detail-map img 64 4)]
      (is (every? (fn [[x y]] (<= 0.0 (w/detail-at dm x y) 1.0))
                  (for [x (range 0 24 4) y (range 0 24 4)] [x y]))))))

(deftest deterministic
  (let [img (gray-img 16 16 (fn [x y] (if (even? (+ x y)) 0.2 0.8)))]
    (is (= (:dmax (w/detail-map img 64 4)) (:dmax (w/detail-map img 64 4))))))
