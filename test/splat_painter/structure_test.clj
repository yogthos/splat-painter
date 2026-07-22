(ns splat-painter.structure-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.structure :as s]))

(defn- approx= [tol a b] (< (Math/abs (- (double a) (double b))) tol))

(defn- solid [H W [r g b :as c]]
  {:height H :width W :channels 3
   :pixels (double-array (apply concat (for [_ (range (* H W))] c)))})

(defn- gradient-img [H W f]
  (let [pixels (double-array (* H W 3))]
    (dotimes [x H]
      (dotimes [y W]
        (let [v (double (f x y))
              base (* 3 (+ (* x W) y))]
          (aset pixels base v)
          (aset pixels (inc base) v)
          (aset pixels (+ 2 base) v))))
    {:height H :width W :channels 3 :pixels pixels}))

(defn- near-angle? [tol a b]
  (let [d (Math/abs (- (mod a Math/PI) (mod b Math/PI)))
        d (min d (- Math/PI d))]
    (< d tol)))

(deftest horizontal-edge-gives-horizontal-stroke
  (testing "step across rows → gradient along x, stroke along y"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< x (/ H 2)) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (s/orient-at sfield (/ H 2) (/ W 2))]
      (is (near-angle? 0.2 theta (/ Math/PI 2.0))
          (str "theta=" theta " should be ~π/2 (horizontal stroke)"))
      (is (> coherence 0.5)
          (str "coherence=" coherence " should be high at a sharp edge")))))

(deftest vertical-edge-gives-vertical-stroke
  (testing "step across cols → gradient along y, stroke along x"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< y (/ W 2)) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (s/orient-at sfield (/ H 2) (/ W 2))]
      (is (near-angle? 0.2 theta 0.0)
          (str "theta=" theta " should be ~0 (vertical stroke)"))
      (is (> coherence 0.5)
          (str "coherence=" coherence " should be high at a sharp edge")))))

(deftest diagonal-edge
  (testing "step along x+y → gradient along (1,1), stroke along (1,-1) = 3π/4"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< (+ x y) H) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (s/orient-at sfield (/ H 2) (/ W 2))]
      (is (near-angle? 0.3 theta (* 3.0 (/ Math/PI 4.0)))
          (str "theta=" theta " should be ~3π/4"))
      (is (> coherence 0.4)
          (str "coherence=" coherence " should be moderate at a diagonal edge")))))

(deftest flat-image-has-no-coherence
  (testing "solid color → no gradient → near-zero coherence"
    (let [img (solid 16 16 [0.5 0.5 0.5])
          sfield (s/analyze img)
          {:keys [coherence]} (s/orient-at sfield 8 8)]
      (is (< coherence 0.1)
          (str "coherence=" coherence " should be near zero for a flat image")))))

(deftest determinism
  (testing "orient-at returns identical results for identical inputs"
    (let [img (gradient-img 16 16 (fn [x y] (if (< x 8) 0.2 0.8)))
          sfield (s/analyze img)]
      (is (= (s/orient-at sfield 8 8)
             (s/orient-at sfield 8 8))))))

(deftest luma-bt601-weights
  (testing "luma uses BT.601: 0.299*R + 0.587*G + 0.114*B"
    (let [img {:height 1 :width 2 :channels 3
               :pixels (double-array [1.0 0.0 0.0   ;; pure red
                                      0.0 1.0 0.0])} ;; pure green
          L (s/luma img)]
      (is (approx= 1e-12 0.299 (aget L 0)) "red pixel luma")
      (is (approx= 1e-12 0.587 (aget L 1)) "green pixel luma"))))

(deftest analyze-returns-expected-keys-and-gmax
  (testing "analyze returns :h :w :jxx :jyy :jxy :gmax with positive gmax on edges"
    (let [img (gradient-img 16 16 (fn [x y] (if (< x 8) 0.2 0.8)))
          sfield (s/analyze img)]
      (is (contains? sfield :h))
      (is (contains? sfield :w))
      (is (contains? sfield :jxx))
      (is (contains? sfield :jyy))
      (is (contains? sfield :jxy))
      (is (contains? sfield :gmax))
      (is (pos? (:gmax sfield)) "gmax should be >0 for a non-flat image"))))

(deftest flat-image-gmax-is-zero
  (testing "analyze of a solid-color image has gmax 0"
    (let [img (solid 16 16 [0.5 0.5 0.5])
          sfield (s/analyze img)]
      (is (approx= 1e-12 0.0 (:gmax sfield))))))
