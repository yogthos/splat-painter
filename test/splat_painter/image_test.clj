(ns splat-painter.image-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.image :as image]
            [splat-painter.seed :as seed]
            [splat-painter.gaussian :as g]))

(def ^:private fixture "test/splat_painter/fixtures/eye.jpeg")

(deftest load-decodes-eye-jpeg
  (let [img (image/load-image fixture 128)]
    (is (= 128 (:height img)))
    (is (= 128 (:width img)))
    (is (= 3 (:channels img)))
    (is (= (* 128 128 3) (count (:pixels img))))
    (is (every? #(<= 0.0 % 1.0) (:pixels img)))
    ;; the eye photo is not blank — there is real tonal spread
    (let [ps (:pixels img)]
      (is (< (apply min ps) 0.5))
      (is (> (apply max ps) 0.5)))))

(deftest no-scale-keeps-original-size
  (let [img (image/load-image fixture)]            ; max-side nil -> original
    (is (= 512 (:height img)))
    (is (= 512 (:width img)))))

(deftest seed-and-rasterize-a-real-image
  (testing "loading -> seeding -> CPU rasterize produces a non-trivial image"
    (let [img  (image/load-image fixture 64)
          fld  (seed/splat-field img {:count 256 :scale 3.0 :background 0.0})
          out  (g/rasterize (:splats fld) (repeat (* 64 64 3) 0.0) 64 64)]
      (is (pos? (count out)))
      (is (some pos? out))                       ; not all black
      (is (< (apply min out) (apply max out)))))); has contrast
