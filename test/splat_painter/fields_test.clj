(ns splat-painter.fields-test
  "Characterization guard for the per-image field builders. These run once per image
   load and dominate it, so they get optimized — parallelized, restructured — and the
   whole point of such a change is that the OUTPUT must not move. Each test pins a
   checksum of one builder's arrays; any change that alters a field trips it here
   rather than silently repainting every picture."
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.image :as image]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.fields :as fields]
            [splat-painter.seed :as seed]))

(def ^:private fixture "test/splat_painter/fixtures/eye.jpeg")

(defn- img [] (image/load-image fixture 64))

(defn- ck
  "Order-sensitive checksum of a ^doubles array: Σ vᵢ·(i mod 7 + 1). Position-weighted,
   so a permutation (a band handed to the wrong thread) shows up, not just a value drift."
  [^doubles a]
  (let [n (alength a)]
    (loop [i 0 s 0.0]
      (if (>= i n) s
          (recur (inc i) (+ s (* (aget a i) (double (inc (mod i 7))))))))))

(defn- approx= [tol a b] (< (Math/abs (- (double a) (double b))) tol))

(deftest bilateral-blur-is-stable
  (testing "the edge-aware light blur (the most expensive field)"
    (is (approx= 1e-6 15009.381648121473 (ck (structure/bilateral-blur (img) 3))))))

(deftest box-blur-is-stable
  (testing "the separable box blur every other field is built on"
    (is (approx= 1e-6 15050.330980392244 (ck (structure/blur-image (img) 2))))))

(deftest dominant-blur-is-stable
  (testing "the coverage tiers' colour source"
    (is (approx= 1e-6 14044.626831098392 (ck (structure/dominant-blur (img) 6))))))

(deftest structure-tensor-is-stable
  (testing "orientation / coherence / flow"
    (let [sf (structure/analyze (img))]
      (is (approx= 1e-6 26948.95038691415 (ck (:theta sf))))
      (is (approx= 1e-6 12543.827788502585 (ck (:coherence sf))))
      (is (approx= 1e-6 28941.23481404941 (ck (:flow-theta sf)))))))

(deftest placement-map-is-stable
  (testing "the wavelet detail maps that drive placement"
    (let [dm (wavelet/placement-map (img) (structure/analyze (img)))]
      (is (approx= 1e-6 7438.358239256164 (ck (:detail dm))))
      (is (approx= 1e-6 6862.340326887574 (ck (:sharp dm)))))))

(deftest prep-noise-is-stable
  (testing "the baked orientation field, both ends of the Swirl dial"
    (let [nf (seed/prep-noise (structure/analyze (img)))]
      (is (approx= 1e-6 -7986.781693016991 (ck (:c2 nf))))
      (is (approx= 1e-6 -2100.4643701075915 (ck (:s2 nf))))
      (is (approx= 1e-6 -7391.388662699906 (ck (:c2s nf))))
      (is (approx= 1e-6 -1844.471251604193 (ck (:s2s nf)))))))

(deftest prepare-assembles-every-field
  (testing "fields/prepare wires each builder to the right key"
    (let [f (fields/prepare (img))]
      (is (approx= 1e-6 (ck (structure/bilateral-blur (img) 3))          (ck (:blur f))))
      (is (approx= 1e-6 (ck (structure/blur-image (img) 2))              (ck (:blur-drift f))))
      (is (approx= 1e-6 (ck (structure/dominant-blur (img) (fields/heavy-radius (img))))
                        (ck (:blur-heavy f))))
      (is (approx= 1e-6 (ck (:theta (structure/analyze (img))))          (ck (:theta (:structure f)))))
      (is (approx= 1e-6 (ck (:detail (wavelet/placement-map (img) (structure/analyze (img)))))
                        (ck (:detail (:detail f)))))
      (is (approx= 1e-6 (ck (:c2 (seed/prep-noise (structure/analyze (img)))))
                        (ck (:c2 (:noise-fields f))))))))
