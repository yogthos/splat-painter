(ns splat-painter.palette-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.palette :as p]))

(defn- test-colors [n]
  (for [i (range n)]
    [(/ (double (mod (* i 37) 100)) 100.0)
     (/ (double (mod (* i 61) 100)) 100.0)
     (/ (double (mod (* i 13) 100)) 100.0)]))

(deftest build-palette-size
  (testing "palette size <= k and all entries distinct"
    (let [colors (test-colors 100)
          pal (p/build-palette colors 8)]
      (is (<= (count pal) 8))
      (is (= (count (distinct pal)) (count pal))
          "all palette entries should be distinct"))))

(deftest two-clusters
  (testing "two clusters yield a palette with one entry near each"
    (let [near-black (repeat 50 [0.1 0.1 0.1])
          near-white (repeat 50 [0.9 0.9 0.9])
          colors (concat near-black near-white)
          pal (p/build-palette colors 2)]
      (is (= 2 (count pal)))
      (is (< (apply min (map #(p/dist2 % [0.1 0.1 0.1]) pal)) 0.1)
          "one palette entry close to black cluster")
      (is (< (apply min (map #(p/dist2 % [0.9 0.9 0.9]) pal)) 0.1)
          "one palette entry close to white cluster"))))

(deftest quantize-maps-to-palette
  (testing "every quantized color equals nearest palette entry"
    (let [colors (test-colors 50)
          pal (p/build-palette colors 4)]
      (when (pos? (count pal))
        (let [q (p/quantize colors 4)]
          (is (= (count colors) (count q)))
          (is (every? (fn [c] (= c (p/nearest c pal))) q)
              "each output should be its nearest palette entry"))))))

(deftest quantize-off
  (testing "quantize with k<2 returns input unchanged"
    (let [colors (test-colors 10)]
      (is (= (p/quantize colors 1) colors))
      (is (= (p/quantize colors 0) colors)))))

(deftest determinism
  (testing "build-palette is deterministic"
    (let [colors (test-colors 30)]
      (is (= (p/build-palette colors 5)
             (p/build-palette colors 5))))))

(deftest build-palette-maxmin-diversity
  (testing "farthest-point seeding picks diverse colors, not just most frequent"
    ;; Three clusters: 50 red-ish, 30 green-ish, 10 blue-ish.
    ;; If max-min works, palette of 3 has one entry per cluster.
    ;; If broken (freq-only), it picks e.g. two red-ish entries.
    (let [reds    (repeat 50 [0.9 0.1 0.1])
          greens  (repeat 30 [0.1 0.9 0.1])
          blues   (repeat 10 [0.1 0.1 0.9])
          colors  (concat reds greens blues)
          pal     (p/build-palette colors 3)]
      (is (= 3 (count pal)))
      ;; Each cluster gets at least one palette entry close to it
      (is (< (apply min (map #(p/dist2 % [0.9 0.1 0.1]) pal)) 0.1)
          "red cluster has a representative")
      (is (< (apply min (map #(p/dist2 % [0.1 0.9 0.1]) pal)) 0.1)
          "green cluster has a representative")
      (is (< (apply min (map #(p/dist2 % [0.1 0.1 0.9]) pal)) 0.1)
          "blue cluster has a representative"))))
