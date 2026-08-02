(ns splat-painter.edgeclip-test
  "Pins the per-fragment EDGE CLIP (splat-painter-c8e).

   Gaussian splats are symmetric and have no notion of a boundary they must not
   cross: every guard upstream (the near-edge shrink, the footprint gate, the
   region-consistency colour clamp) limits a stroke's SIZE or COLOUR near an edge
   while its BODY still paints through. `shader/edge-clip-factor` is the shared
   spec — it attenuates a fragment's alpha where the underlying region colour
   disagrees with the stroke's own, and only in the stroke's TAIL, so the mark
   keeps its identity at the core. check.clj pins the GLSL mirror."
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.shader :as shader]))

(def ^:private core-pdf 0.0)          ; the stroke centre
(def ^:private tail-pdf 3.0)          ; well out in the tail (>2 sigma)

(deftest amount-zero-is-exactly-identity
  (testing "clip 0 must not touch a single fragment — the slider ships at 0 and the
            render has to stay byte-identical to before the feature existed"
    (doseq [pdf [0.0 0.5 1.0 3.0 6.0]
            dc  [0.0 0.2 0.5 1.0]]
      (is (== 1.0 (shader/edge-clip-factor 0.0 pdf dc))))))

(deftest the-core-of-a-stroke-is-never-clipped
  (testing "at the centre the stroke keeps full alpha however foreign the ground is —
            otherwise a mark placed ON a boundary would erase itself"
    (doseq [dc [0.0 0.3 0.6 1.0]]
      (is (== 1.0 (shader/edge-clip-factor 1.0 core-pdf dc))))))

(deftest matching-ground-is-never-clipped
  (testing "where the stroke agrees with what it is painting over, nothing changes —
            this is the whole interior of every region"
    (doseq [pdf [0.0 1.0 3.0 6.0]]
      (is (== 1.0 (shader/edge-clip-factor 1.0 pdf 0.0))))))

(deftest per-stroke-jitter-must-not-trip-the-clip
  (testing "tone and temperature jitter make a stroke's colour differ from the
            underlying blur BY DESIGN (variation*0.15 tone, variation*0.10 temp), and
            the region-consistency clamp already tolerates raw within 0.12 of the
            bilateral. If the low threshold sat under that, every stroke everywhere
            would be clipped and the painting would go transparent."
    (is (> shader/clip-lo 0.15)
        "clip-lo must sit above the designed per-stroke colour jitter")
    (doseq [dc [0.0 0.05 0.10 0.15]]
      (is (== 1.0 (shader/edge-clip-factor 1.0 tail-pdf dc))
          (str "jitter-sized disagreement " dc " must not clip")))))

(deftest a-strong-boundary-clips-the-tail
  (testing "skin (0.6) over a dark ground (0.05) is dc 0.55 — a real silhouette"
    (let [f (shader/edge-clip-factor 1.0 tail-pdf 0.55)]
      (is (< f 0.05) "the tail should be almost entirely removed")))
  (testing "and the clip is proportional to the dial"
    (let [full (shader/edge-clip-factor 1.0 tail-pdf 0.55)
          half (shader/edge-clip-factor 0.5 tail-pdf 0.55)]
      (is (< full half 1.0))
      ;; linear in amount: half the dial removes half the alpha the full dial does
      (is (< (Math/abs (- (- 1.0 half) (* 0.5 (- 1.0 full)))) 1e-9)))))

(deftest clip-is-monotone-in-both-arguments
  (testing "more disagreement never clips less"
    (let [ys (map #(shader/edge-clip-factor 1.0 tail-pdf %) (range 0.0 1.01 0.05))]
      (is (= ys (reverse (sort ys))))))
  (testing "further into the tail never clips less"
    (let [ys (map #(shader/edge-clip-factor 1.0 % 0.55) (range 0.0 6.01 0.25))]
      (is (= ys (reverse (sort ys)))))))

(deftest clip-never-leaves-the-unit-interval
  (testing "alpha is multiplied by this, so anything outside [0,1] is a bug"
    (doseq [amt [0.0 0.5 1.0]
            pdf [0.0 1.0 3.0 20.0]
            dc  [0.0 0.5 2.0]]
      (let [f (shader/edge-clip-factor amt pdf dc)]
        (is (<= 0.0 f 1.0))))))
