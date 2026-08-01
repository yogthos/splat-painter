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
      (testing "and the separation is exactly the floor, by construction"
        ;; bg/fg IS hard-detail-floor: fg lift = (dial-1)*1.0, bg lift = (dial-1)*floor.
        ;; Asserting a separate magic number here would just be the floor written
        ;; twice, and the first version of this test did exactly that and then failed
        ;; the moment the floor was retuned. Pin the identity, and bound the floor.
        (let [fg (- (apply-scale dialled 1.0) 1.0)
              bg (- (apply-scale dialled 0.0) 1.0)]
          (is (< (Math/abs (- (/ bg fg) shader/hard-detail-floor)) 1e-9))
          (is (<= shader/hard-detail-floor 0.85)
              "above this the dial is effectively global again and w4w is undone")))))
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

;; --- sharpen detail gate (splat-painter, same request as w4w) ----------------
;; Sharpen is a present pass on the flattened composite, so it rides the subjectness
;; FIELD rather than a per-splat value, but the shape of the scale is the same.

(deftest sharpen-scale-spans-floor-to-full
  (testing "a flat region gets the floor, a fully-detailed one the whole dial"
    (is (== shader/sharp-detail-floor (shader/detail-sharpen-scale 0.0)))
    (is (== 1.0 (shader/detail-sharpen-scale 1.0)))
    (is (< 0.0 shader/sharp-detail-floor 1.0))))

(deftest sharpen-scale-is-monotone-and-clamped
  (testing "more detail never means less sharpening"
    (let [ys (map shader/detail-sharpen-scale (range 0.0 1.01 0.05))]
      (is (= ys (sort ys)))))
  (testing "out-of-range subjectness cannot exceed the dial"
    (is (== shader/sharp-detail-floor (shader/detail-sharpen-scale -0.5)))
    (is (== 1.0 (shader/detail-sharpen-scale 3.0)))))

(deftest sharpen-separates-foreground-from-background
  (testing "detailed foreground sharpens more than flat background"
    (is (> (shader/detail-sharpen-scale 1.0) (shader/detail-sharpen-scale 0.0))))
  (testing "amount 0 stays 0 everywhere -- the pass is skipped, but the scale is
            multiplicative so it must not manufacture sharpening from nothing"
    (doseq [d [0.0 0.5 1.0]]
      (is (== 0.0 (* 0.0 (shader/detail-sharpen-scale d))))))
  (testing "background keeps SOME sharpening -- a hard zero would make flat regions
            visibly different in kind, not degree"
    (is (> (shader/detail-sharpen-scale 0.0) 0.25))))

;; --- the knee ----------------------------------------------------------------

(deftest knee-gives-subject-the-full-dial-and-flat-the-floor
  (testing "absolute subjectness is BIMODAL on real photographs (Lenin percentiles
            p05 0.342, p25 0.946, p50 1.000), so a linear ramp spends its travel
            crossing a nearly empty gap and charges the cost to genuine subject.
            The knee must clear that: anything reading as subject gets the whole
            dial, anything genuinely flat gets the floor."
    (let [[lo hi] shader/detail-knee]
      (is (< 0.0 lo hi 1.0) "knee is a real, ordered, interior range")
      ;; measured region values on img/Lenin.jpg
      (is (== 1.0 (shader/detail-weight 0.946)) "p25 subjectness -> full dial")
      (is (== 1.0 (shader/detail-weight 1.000)) "jacket/face -> full dial")
      (is (== 0.0 (shader/detail-weight 0.337)) "the flat corner -> floor only")
      (is (== 0.0 (shader/detail-weight 0.342)) "p05 subjectness -> floor only"))))

(deftest knee-is-smooth-and-monotone
  (testing "a hard step would band the gate visibly across a gradient"
    (let [xs (range 0.0 1.001 0.02)
          ys (mapv shader/detail-weight xs)]
      (is (= ys (sort ys)) "monotone")
      (is (== 0.0 (first ys)))
      (is (== 1.0 (last ys)))
      ;; smoothstep has zero derivative at both ends — no visible seam where the
      ;; gate starts or stops acting
      (let [d (mapv - (rest ys) (butlast ys))]
        (is (< (first d) (apply max d)) "eases in, not a linear ramp")
        (is (< (last d) (apply max d)) "eases out")))))

(deftest both-dials-share-one-notion-of-foreground
  (testing "hardness and sharpen have independent FLOORS but the same SHAPE, so a
            region that counts as foreground for one counts for the other"
    (doseq [d [0.0 0.2 0.337 0.5 0.7 0.946 1.0]]
      (let [h (/ (- (shader/detail-hardness-scale d) shader/hard-detail-floor)
                 (- 1.0 shader/hard-detail-floor))
            s (/ (- (shader/detail-sharpen-scale d) shader/sharp-detail-floor)
                 (- 1.0 shader/sharp-detail-floor))]
        (is (< (Math/abs (- h s)) 1e-9)
            (str "normalized weight must match at subjectness " d))))))
