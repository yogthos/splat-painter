(ns splat-painter.grid-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.grid :as grid]))

(defn- approx= [tol a b]
  (< (Math/abs (- (double a) (double b))) tol))

(deftest optimize-returns-grid
  (testing "optimize returns {:cols :rows :cx :cy} with correct corner count"
    (let [zero-edge (fn [_x _y] 0.0)
          g         (grid/optimize 64 64 64 zero-edge)]
      (is (contains? g :cols))
      (is (contains? g :rows))
      (is (contains? g :cx))
      (is (contains? g :cy))
      (let [n-corners (* (inc (:rows g)) (inc (:cols g)))
            cx        (:cx g)
            cy        (:cy g)]
        (is (= n-corners (alength cx))
            "cx array length matches corner count")
        (is (= n-corners (alength cy))
            "cy array length matches corner count")
        (let [rows (:rows g)
              cols (:cols g)
              step (max 1.0 (/ (min 64.0 (:rows g)) 4.0))
              H    64.0
              W    64.0]
          (doseq [r (range 1 rows) c (range 1 cols)]
            (let [idx    (+ (* r (inc cols)) c)
                  init-x (* (/ (double r) rows) H)
                  init-y (* (/ (double c) cols) W)]
              (is (approx= (* 2.0 step) (aget cx idx) init-x)
                  (str "corner (" r "," c ") x near initial"))
              (is (approx= (* 2.0 step) (aget cy idx) init-y)
                  (str "corner (" r "," c ") y near initial")))))))))

(deftest corners-snap-to-a-vertical-edge
  (testing "vertical edge at y=32 attracts interior corners closer than initial"
    (let [H       64
          W       64
          ;; wide edge so edge-density window (radius 3) can detect it
          edge-fn (fn [x y] (if (< (Math/abs (- y 32.0)) 5.0) 1.0 0.0))
          ;; 49 cells => cols=7,rows=7; no column starts exactly at y=32
          g       (grid/optimize H W 49 edge-fn)
          rows    (:rows g)
          cols    (:cols g)
          cx      (:cx g)
          cy      (:cy g)
          col     (first (sort-by (fn [c]
                                    (Math/abs (- (* (/ (double c) cols) W) 32.0)))
                                  (range 1 cols)))
          mean-init (/ (apply + (for [r (range 1 rows)]
                                  (Math/abs (- (* (/ (double col) cols) W) 32.0))))
                       (double (dec rows)))
          mean-final (/ (apply + (for [r (range 1 rows)]
                                   (let [idx (+ (* r (inc cols)) col)]
                                     (Math/abs (- (aget cy idx) 32.0)))))
                        (double (dec rows)))]
      (is (> mean-init 0.5) "initial corners are off the edge")
      (is (< mean-final mean-init)
          (str "corners moved closer to edge: init=" mean-init " final=" mean-final)))))

(deftest grid-is-deterministic
  (testing "two optimize calls with same inputs give equal corner arrays"
    (let [edge-fn (fn [x y] (if (> x 32.0) 0.5 0.0))
          g1      (grid/optimize 64 64 36 edge-fn)
          g2      (grid/optimize 64 64 36 edge-fn)]
      (is (= (vec (:cx g1)) (vec (:cx g2))) "cx arrays equal")
      (is (= (vec (:cy g1)) (vec (:cy g2))) "cy arrays equal"))))

(deftest cell-centroid-and-size
  (testing "cell-centroid within image and cell-size positive"
    (let [edge-fn (fn [_x _y] 0.0)
          g       (grid/optimize 64 64 64 edge-fn {:iterations 0})]
      (doseq [r (range (:rows g)) c (range (:cols g))]
        (let [[x y] (grid/cell-centroid g r c)
              sz    (grid/cell-size g r c)]
          (is (and (>= x 0.0) (< x 64.0)) "centroid x within bounds")
          (is (and (>= y 0.0) (< y 64.0)) "centroid y within bounds")
          (is (> sz 0.0) "cell size positive"))))))
