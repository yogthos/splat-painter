(ns splat-painter.wavelet-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.wavelet :as w]
            [splat-painter.structure :as structure]))

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

(deftest placement-map-lifts-dark-texture
  ;; The whole point of the luma-relative map: a dark region with high *relative*
  ;; contrast (checkerboard at 0.05/0.15) must read higher detail than a bright
  ;; flat region (solid 0.8) even though the dark checkerboard's absolute Haar
  ;; energy is tiny. Without the luma-relative + edge-fused correction, the dark
  ;; half would be starved of fine splats.
  (let [H 32 W 64
        img (gray-img H W
              (fn [x y]
                (if (< x 16)
                  ;; dark half — strong relative texture
                  (if (even? (+ (int x) (int y))) 0.05 0.15)
                  ;; bright half — flat solid
                  0.8)))
        sfield (structure/analyze img)
        pmap   (w/placement-map img sfield)
        ^doubles P (:detail pmap)
        sw (:w pmap)]
    ;; split the placement map by the same source coords
    (let [midx (long (/ (* 16 (:h pmap)) H))
          dark-vals  (for [ri (range 0 midx)
                           ci (range sw)]
                       (aget P (+ (* ri sw) ci)))
          bright-vals (for [ri (range midx (:h pmap))
                            ci (range sw)]
                        (aget P (+ (* ri sw) ci)))
          dark-mean  (/ (reduce + dark-vals) (count dark-vals))
          bright-mean (/ (reduce + bright-vals) (count bright-vals))]
      (is (> dark-mean 0.26)
          (str "dark textured half mean=" dark-mean " should exceed 0.26"))
      (is (< bright-mean 0.26)
          (str "bright flat half mean=" bright-mean " should stay below 0.26")))))

(deftest placement-map-dmax-is-one
  ;; placement-map :dmax must be exactly 1.0 — the map is pre-normalized so
  ;; detail-at and the GPU sampler need no change.
  (let [img (gray-img 16 16 (fn [x y] (if (even? (+ x y)) 0.2 0.8)))
        sf  (structure/analyze img)
        pm  (w/placement-map img sf)]
    (is (= 1.0 (:dmax pm)) "dmax must be 1.0")))

(deftest placement-map-detail-at-in-range
  ;; detail-at on placement-map (dmax=1.0) must stay in [0,1] — the GPU
  ;; sampler texture fetches these values directly.
  (let [img (gray-img 32 32 (fn [x y] (if (< y 16) 0.5
                                        (if (even? (+ x y)) 0.0 1.0))))
        sf  (structure/analyze img)
        pm  (w/placement-map img sf)]
    (is (every? (fn [[x y]] (<= 0.0 (w/detail-at pm x y) 1.0))
                (for [x (range 0 32 4) y (range 0 32 4)] [x y])))))

(deftest subject-map-saturates-on-genuine-structure
  ;; The subject map ends in a saturating remap so genuine structure clears 0.95
  ;; (the Broad dial must not smear real subjects, per gen.clj's m=1 invariant)
  ;; while flat bokeh collapses toward 0. Pre-remap the dense period-8 stripes
  ;; only reach ~0.905, below the bar — so this assertion discriminates.
  ;; Period-8 (1px line / 7px gap), NOT wide 1px/15px stripes (~0.54) which would
  ;; miscalibrate the constants; and NOT a fine checker which saturates already.
  (let [H 256 W 256
        img (gray-img H W
              (fn [x y]
                (if (and (<= 64 x 191) (<= 64 y 191))
                  (if (zero? (mod y 8)) 0.2 0.5)    ; 1px line / 7px gap, period 8
                  0.5)))                             ; flat bokeh
        sf  (structure/analyze img)
        pm  (w/placement-map img sf)
        mean-subj (fn [r0 r1 c0 c1]
                    (let [xs (for [x (range r0 r1) y (range c0 c1)]
                               (w/subject-abs-at pm x y))]
                      (/ (reduce + xs) (count xs))))
        inside (mean-subj 120 140 120 140)
        flat   (mean-subj 8 28 8 28)]
    (is (>= inside 0.95)
        (str "dense structure subjectness " inside " should clear 0.95"))
    (is (<= flat 0.35)
        (str "flat bokeh subjectness " flat " should be <= 0.35"))
    (is (>= (- inside flat) 0.5)
        (str "separation " (- inside flat) " should be >= 0.5"))))
