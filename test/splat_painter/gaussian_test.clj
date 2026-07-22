(ns splat-painter.gaussian-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.edn :as edn]
            [splat-painter.gaussian :as g]))

(defn- approx? [tol a b]
  (if (and (number? a) (number? b))
    (< (Math/abs (- (double a) (double b))) tol)
    (and (sequential? a) (sequential? b)
         (= (count a) (count b))
         (every? true? (map #(approx? tol %1 %2) a b)))))

(deftest covariance-is-RSSR-t
  (testing "Σ = R(θ) diag(sx², sy²) R(θ)ᵀ, the 2DGS construction"
    (is (approx? 1e-9 [4 0 0 4] (g/covariance 2 2 0)))      ; isotropic, no rotation
    (is (approx? 1e-9 [1 0 0 4] (g/covariance 1 2 0)))      ; axis-aligned anisotropic
    ;; 90° rotation swaps the principal axes: diag(1,4) -> diag(4,1)
    (is (approx? 1e-9 [4 0 0 1] (g/covariance 1 2 (/ Math/PI 2))))
    ;; always symmetric and positive-definite
    (let [[c00 c01 c10 c11] (g/covariance 3 1.5 0.7)]
      (is (approx? 1e-12 c01 c10))
      (is (pos? c00))
      (is (pos? c11))
      (is (pos? (- (* c00 c11) (* c01 c10)))))))

(deftest precision-inverts-covariance
  (testing "δᵀ P δ entries are the closed-form inverse of the 2×2 covariance"
    (is (approx? 1e-9 [1 1 0] (g/precision [1 0 0 1])))     ; identity
    (is (approx? 1e-9 [0.25 0.25 0] (g/precision [4 0 0 4])))
    ;; at the mean, δᵀPδ = 0 (the peak of the gaussian)
    (let [[p00 p11 cross] (g/precision [2 0.5 0.5 0.7])]
      (is (approx? 1e-9 0.0 (+ (* p00 0 0) (* cross 0 0) (* p11 0 0)))))))

(deftest rasterize-additive-port-matches-reference
  (testing "the CPU additive rasterizer matches a numpy port of rendering2d.py
            over a full 8×8 grid with 3 anisotropic splats"
    (let [g0 (edn/read-string (slurp "test/splat_painter/golden.edn"))
          {:keys [H W means covs colors out]} g0
          splats (map (fn [m c col]
                        {:mean  m
                         :cov   [(get-in c [0 0]) (get-in c [0 1])
                                 (get-in c [1 0]) (get-in c [1 1])]
                         :color col})
                      means covs colors)
          bg (repeat (* H W 3) 0.0)
          ours (g/rasterize splats bg H W)
          golden (into [] (mapcat identity (apply concat out)))] ; HxWx3 -> flat
      (is (= (count ours) (count golden)))
      (is (approx? 1e-4 ours golden)))))

(deftest composite-port-matches-reference
  (testing "the CPU over-compositor matches a numpy port of the front-to-back
            over-operator over the same 8×8 / 3-splat case at opacity 0.6"
    (let [g0 (edn/read-string (slurp "test/splat_painter/golden_composite.edn"))
          {:keys [H W means covs colors out]} g0
          splats (map (fn [m c col]
                        {:mean  m
                         :cov   [(get-in c [0 0]) (get-in c [0 1])
                                 (get-in c [1 0]) (get-in c [1 1])]
                         :color col})
                      means covs colors)
          bg (repeat (* H W 3) 0.0)
          ours (g/composite splats bg H W 0.6)
          golden (into [] (mapcat identity (apply concat out)))]
      (is (= (count ours) (count golden)))
      (is (approx? 1e-4 ours golden)))))

(deftest composite-over-operator-properties
  (testing "single splat at opacity 1 with no splat in front peaks at its color (T starts at 1)"
    (let [H 5 W 5 bg (repeat (* H W 3) 0.0)
          out (g/composite [{:mean [2 2] :cov [1 0 0 1] :color [0.9 0.1 0.2]}] bg H W 1.0)
          peak (nth out (* 3 (+ (* 2 W) 2)))]
      (is (approx? 1e-9 peak 0.9))))
  (testing "two coincident opacity-0.5 splats: 0.5 + 0.5·0.5 = 0.75 of the color at the peak"
    (let [H 3 W 3 bg (repeat (* H W 3) 0.0)
          out (g/composite [{:mean [1 1] :cov [1 0 0 1] :color [0.4 0.0 0.0]}
                            {:mean [1 1] :cov [1 0 0 1] :color [0.4 0.0 0.0]}] bg H W 0.5)
          peak (nth out (* 3 (+ (* 1 W) 1)))]
      (is (approx? 1e-9 peak 0.3))))
  (testing "compositing never exceeds 1.0 (occlusion bounds the sum), unlike additive"
    (let [H 5 W 5 bg (repeat (* H W 3) 0.0)
          splats (for [_ (range 40)] {:mean [2 2] :cov [1 0 0 1] :color [1.0 1.0 1.0]})
          out (g/composite splats bg H W 1.0)]
      (is (<= (apply max out) 1.0)))))

(deftest rasterize-properties
  (testing "single centered isotropic splat: peak pixel == color, symmetric, brighter than a neighbour"
    (let [H 5 W 5
          bg (repeat (* H W 3) 0.0)
          out (g/rasterize [{:mean [2 2] :cov [1 0 0 1] :color [0.9 0.1 0.2]}] bg H W)
          px (fn [x y] [(nth out (* 3 (+ (* x W) y)))
                        (nth out (+ 1 (* 3 (+ (* x W) y))))
                        (nth out (+ 2 (* 3 (+ (* x W) y))))])]
      ;; peak-normalized: the center pixel carries the full color
      (is (approx? 1e-9 [0.9 0.1 0.2] (px 2 2)))
      ;; 4-way symmetry of an isotropic splat
      (is (approx? 1e-9 (px 1 2) (px 3 2)))
      (is (approx? 1e-9 (px 2 1) (px 2 3)))
      (is (approx? 1e-9 (px 1 2) (px 2 1)))
      ;; falloff: neighbour is dimmer than the center
      (is (< (reduce + (px 1 2)) (reduce + (px 2 2))))))
  (testing "additivity: two coincident splats sum their colors at the peak"
    (let [H 3 W 3 bg (repeat (* H W 3) 0.0)
          out (g/rasterize [{:mean [1 1] :cov [1 0 0 1] :color [0.4 0.0 0.0]}
                            {:mean [1 1] :cov [1 0 0 1] :color [0.3 0.0 0.0]}] bg H W)
          center (nth out (* 3 (+ (* 1 W) 1)))]
      (is (approx? 1e-9 0.7 center)))))
