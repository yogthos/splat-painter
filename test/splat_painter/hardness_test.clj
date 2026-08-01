(ns splat-painter.hardness-test
  "Pins the DETAIL term in per-splat hardness (splat-painter-w4w).

   Hardness used to key on stroke SIZE alone, so a fat stroke in a detailed
   foreground and a fat stroke in smooth background got identical treatment and
   the dial read as a global crispness change. `shader/detail-hardness-scale` is
   the shared spec: it scales how much of the dialled hardness a stroke actually
   receives, from `hard-detail-floor` in a flat region to the full dial in a
   detailed one. The input is absolute subjectness at the stroke mean, sampled in
   seed/splat-field and mirrored by subjAbsAt(px, py) in the generation shader.
   check.clj pins the GLSL mirror of the same line."
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.shader :as shader]))

(deftest detail-scale-spans-floor-to-full
  (testing "a flat-region stroke gets the floor, a fully-detailed one the whole dial"
    (is (== shader/hard-detail-floor (shader/detail-hardness-scale 0.0)))
    (is (== 1.0 (shader/detail-hardness-scale 1.0)))
    (testing "and the floor leaves background SOME crispness, never zero"
      (is (< 0.0 shader/hard-detail-floor 1.0)))))

(deftest detail-scale-is-monotone-and-clamped
  (testing "more detail never means less hardness"
    (let [xs (range 0.0 1.01 0.05)
          ys (map shader/detail-hardness-scale xs)]
      (is (= ys (sort ys)))))
  (testing "out-of-range detail cannot push the scale outside [floor, 1]"
    ;; subject-abs-at already clamps to [0,1], but a future field change could hand
    ;; this something wider; the scale must not amplify hardness past the dial.
    (is (== shader/hard-detail-floor (shader/detail-hardness-scale -0.5)))
    (is (== 1.0 (shader/detail-hardness-scale 3.0)))))

(deftest detail-actually-separates-foreground-from-background
  (testing "at the same size and dial, a detailed stroke ends up crisper"
    ;; the applied form: hard' = 1 + (hard-1)*scale. u_hard_soft is fixed at 1.0
    ;; and the slider floors at 1.0, so (hard-1) >= 0 and the scale can only
    ;; soften -- it must never invert a stroke to softer-than-gaussian.
    (let [apply-scale (fn [hard d]
                        (+ 1.0 (* (- hard 1.0) (shader/detail-hardness-scale d))))
          dialled     2.8]
      (is (> (apply-scale dialled 1.0) (apply-scale dialled 0.0))
          "detailed foreground must be harder than flat background")
      (is (>= (apply-scale dialled 0.0) 1.0)
          "never softer than a pure gaussian")
      (is (== dialled (apply-scale dialled 1.0))
          "full detail spends the whole dial")
      (testing "and the separation is a large fraction of the dial's travel"
        ;; the point of the feature: if this ratio drifts toward 1 the dial is
        ;; global again and w4w has silently regressed.
        (let [fg (- (apply-scale dialled 1.0) 1.0)
              bg (- (apply-scale dialled 0.0) 1.0)]
          (is (<= (/ bg fg) 0.5)
              "background must get at most half the foreground's hardness lift")))))
  (testing "hardness 1.0 (dial off) is unaffected by detail"
    (let [apply-scale (fn [hard d]
                        (+ 1.0 (* (- hard 1.0) (shader/detail-hardness-scale d))))]
      (is (== 1.0 (apply-scale 1.0 0.0)))
      (is (== 1.0 (apply-scale 1.0 1.0))))))

(deftest pack-splats-carries-detail-in-the-third-texel
  (testing "detail rides slot 9 (texel 3i+2 .y), which the render shader reads"
    (let [splats [{:mean [1.0 2.0] :cov [3.0 4.0 4.0 5.0] :color [0.1 0.2 0.3]
                   :alpha 0.7 :detail 0.25}
                  {:mean [6.0 7.0] :cov [8.0 9.0 9.0 10.0] :color [0.4 0.5 0.6]
                   :alpha 0.8 :detail 1.0}]
          p (shader/pack-splats splats)]
      (is (= 24 (count p)) "two splats x 12 floats")
      (is (== 0.7 (nth p 8)))
      (is (== 0.25 (nth p 9)) "first splat's detail")
      (is (== 0.8 (nth p 20)))
      (is (== 1.0 (nth p 21)) "second splat's detail")
      ;; the two remaining slots stay zero -- the next feature to need a per-splat
      ;; float takes them, and a stale non-zero here would be read as data.
      (is (== 0.0 (nth p 10)))
      (is (== 0.0 (nth p 11)))))
  (testing "a splat with no :detail packs as fully detailed"
    ;; every splat the generator emits carries one; this is the safety default for
    ;; hand-built splats in tests and the check harness, and 1.0 reproduces the
    ;; pre-w4w behaviour (the whole dial) rather than silently softening them.
    (let [p (shader/pack-splats [{:mean [0.0 0.0] :cov [1.0 0.0 0.0 1.0]
                                  :color [1.0 1.0 1.0] :alpha 1.0}])]
      (is (== 1.0 (nth p 9))))))
