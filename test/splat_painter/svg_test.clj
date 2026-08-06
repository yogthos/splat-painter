(ns splat-painter.svg-test
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [splat-painter.gaussian :as gauss]
            [splat-painter.shader :as shader]
            [splat-painter.svg :as svg]))

(defn- approx? [tol a b] (< (Math/abs (- (double a) (double b))) (double tol)))

(defn- els [doc] (re-seq #"<ellipse[^/]*/>" doc))

(defn- attr [el name]
  (when-let [m (re-find (re-pattern (str name "=\"([^\"]*)\"")) el)]
    (second m)))

;; ---------------------------------------------------------------- number output

(deftest coordinates-are-compact-and-lossless-to-the-decimal
  (let [f #'svg/fixed]
    (is (= "5" (f 5.0 1)))
    (is (= "5.1" (f 5.14 1)))
    (is (= "5.2" (f 5.15 1)))
    (is (= "0.1" (f 0.1 3)) "trailing zeros dropped")
    (is (= "0.001" (f 0.001 3)) "leading zeros kept")
    (is (= "-0.1" (f -0.06 1)))
    (is (= "0" (f -0.04 1)) "a value that rounds to zero has no sign")
    (is (= "1000" (f 1000.0 0)))
    ;; the tone filter asks for 4 — a table too short for it is an index-out-of-bounds
    ;; at save time, not a rounding error
    (is (= "0.5556" (f (/ 1.0 1.8) 4)))
    (is (= "0.555556" (f (/ 1.0 1.8) 6)))))

;; ---------------------------------------------------------------- geometry

(deftest ellipse-inverts-the-covariance-construction
  (testing "Σ = R diag(sx²,sy²) Rᵀ round-trips to the ellipse that carries it"
    (doseq [[sx sy th] [[4.0 1.0 0.0] [1.0 4.0 0.0] [3.0 2.0 0.7]
                        [5.0 0.5 -1.2] [2.0 2.0 0.3]]]
      (let [{:keys [rx ry]} (svg/ellipse (gauss/covariance sx sy th))]
        (is (approx? 1e-9 (max sx sy) rx) "rx is the MAJOR stdev")
        (is (approx? 1e-9 (min sx sy) ry) "ry is the MINOR stdev")))))

(deftest ellipse-rotation-is-in-svgs-frame
  (testing "the image's x is a ROW and SVG's x is a COLUMN, so the frames are swapped"
    ;; θ=0 with sx>sy puts the long axis along the image's ROW axis, which is SVG's y —
    ;; a quarter turn. This is also the exactly-axis-aligned case where c01 == 0.0 and
    ;; the first eigenvector form degenerates, so it pins the shader's fallback too.
    (is (approx? 1e-9 90.0 (:deg (svg/ellipse (gauss/covariance 4.0 1.0 0.0)))))
    ;; sy>sx: long axis along the image's COLUMN axis = SVG's x, no rotation.
    (is (approx? 1e-9 0.0 (:deg (svg/ellipse (gauss/covariance 1.0 4.0 0.0)))))
    ;; a general angle stays a quarter turn away from the (row,col) angle
    (let [th 0.6
          {:keys [deg]} (svg/ellipse (gauss/covariance 4.0 1.0 th))]
      (is (approx? 1e-6 (- 90.0 (Math/toDegrees th)) deg)))))

(deftest ellipse-sig-matches-the-shaders-scalar-stdev
  (is (approx? 1e-9 2.0 (:sig (svg/ellipse (gauss/covariance 2.0 2.0 0.0)))))
  (is (approx? 1e-9 (Math/sqrt 6.0) (:sig (svg/ellipse (gauss/covariance 6.0 1.0 0.4))))))

;; ---------------------------------------------------------------- falloff

(deftest profile-is-the-render-shaders-gaussian
  (testing "exp(−(r²/2)^hard), peak-normalized at the mean"
    (is (approx? 1e-12 1.0 (svg/profile-alpha 0.0 1.0)))
    (is (approx? 1e-12 1.0 (svg/profile-alpha 0.0 1.7)))
    ;; hard=1 is the plain gaussian
    (is (approx? 1e-12 (Math/exp -0.5) (svg/profile-alpha 1.0 1.0)))
    (is (approx? 1e-12 (Math/exp -4.5) (svg/profile-alpha 3.0 1.0)))
    ;; a harder exponent is FLATTER inside 1σ and falls off faster outside it
    (is (> (svg/profile-alpha 0.8 2.0) (svg/profile-alpha 0.8 1.0)))
    (is (< (svg/profile-alpha 2.0 2.0) (svg/profile-alpha 2.0 1.0)))))

(deftest hardness-mirrors-the-render-vertex-shader
  (let [sharp 1.7 soft 1.0]
    (testing "at the top of the size range the stroke gets the SOFT end"
      (is (approx? 1e-9 soft (svg/hardness 20.0 1.0 20.0 1.0 sharp soft))))
    (testing "at the bottom the sub-pixel antialias ease is the last word"
      ;; sig 1.0 < 2.5, so hard−1 is scaled by sig/2.5 whatever the smoothstep says
      (is (approx? 1e-9 (+ 1.0 (* (- sharp 1.0) (shader/detail-hardness-scale 1.0) (/ 1.0 2.5)))
                   (svg/hardness 1.0 1.0 20.0 1.0 sharp soft))))
    (testing "detail spends the dial on foreground, not on smooth background"
      (let [fg (svg/hardness 5.0 1.0 20.0 1.0 sharp soft)
            bg (svg/hardness 5.0 1.0 20.0 0.0 sharp soft)]
        (is (> fg bg))
        (is (approx? 1e-9 (+ 1.0 (* (- fg 1.0) shader/hard-detail-floor)) bg))))
    (testing "a degenerate size range does not divide by zero"
      (is (pos? (svg/hardness 3.0 4.0 4.0 1.0 sharp soft))))))

;; ---------------------------------------------------------------- palette

(deftest palette-assigns-every-colour
  (let [rows (vec (for [i (range 500)] [(mod (* i 7) 256) (mod (* i 13) 256) (mod (* i 31) 256) i]))
        {:keys [entries index]} (svg/palette rows 64)]
    (is (<= (count entries) 64))
    (is (= 500 (count index)) "every input row lands in exactly one box")
    (is (every? #(< -1 % (count entries)) (vals index)))))

(deftest palette-keeps-a-rare-saturated-accent
  (testing "population-first median cut folds the eye colour into the fur; population×spread does not"
    ;; 2000 near-greys plus 12 saturated greens — the accent is 0.6% of the field.
    (let [greys  (for [i (range 2000)] [(+ 100 (mod i 30)) (+ 100 (mod i 30)) (+ 105 (mod i 30)) i])
          greens (for [i (range 12)] [20 200 60 (+ 2000 i)])
          {:keys [entries index]} (svg/palette (vec (concat greys greens)) 16)
          green-entry (nth entries (index 2000))]
      (is (> (nth green-entry 1) 0.5) "the accent keeps its green")
      (is (< (nth green-entry 0) 0.3) "and does not get averaged into grey"))))

(deftest palette-handles-a-single-colour
  (let [{:keys [entries index]} (svg/palette [[10 20 30 0] [10 20 30 1]] 16)]
    (is (= 1 (count entries)))
    (is (= {0 0, 1 0} index))))

;; ---------------------------------------------------------------- document

(def ^:private field
  {:width 40 :height 20
   :background [0.0 0.0 0.0]
   :opacity 1.0 :sig-min 1.0 :sig-max 4.0
   :splats [{:mean [5.0 10.0] :cov (gauss/covariance 4.0 1.0 0.0)
             :color [1.0 0.0 0.0] :alpha 1.0 :detail 1.0}
            {:mean [6.0 20.0] :cov (gauss/covariance 2.0 2.0 0.0)
             :color [0.0 1.0 0.0] :alpha 0.5 :detail 0.0}
            {:mean [7.0 30.0] :cov (gauss/covariance 3.0 1.0 0.5)
             :color [0.0 0.0 1.0] :alpha 0.25 :detail 0.5}]})

(deftest document-is-one-element-per-splat-in-reverse-paint-order
  (let [doc (svg/field->svg field {:mode :flat})
        es  (els doc)]
    (is (str/starts-with? doc "<?xml"))
    (is (str/ends-with? doc "</svg>"))
    (is (str/includes? doc "viewBox=\"0 0 40 20\""))
    (is (= 3 (count es)))
    (testing "the field is finest-first and SVG paints in document order, so it reverses"
      ;; splat 0 (red) is the topmost stroke — it must be the LAST element written
      (is (= "#0000ff" (attr (first es) "fill")))
      (is (= "#ff0000" (attr (last es) "fill"))))
    (testing "SVG x is the image's column"
      (is (= "10" (attr (last es) "cx")))
      (is (= "5" (attr (last es) "cy"))))))

(deftest alpha-rides-fill-opacity-never-opacity
  (testing "`opacity` composites the element as an isolated group — 15× slower per splat"
    (let [doc (svg/field->svg field {:mode :flat})]
      (is (not (re-find #"<ellipse[^/]*[^-]opacity=" doc)))
      (is (str/includes? doc "fill-opacity=\"0.5\""))
      (is (str/includes? doc "fill-opacity=\"0.25\"")))))

(deftest global-opacity-multiplies-into-the-per-splat-alpha
  (let [doc (svg/field->svg (assoc field :opacity 0.5) {:mode :flat})]
    (is (str/includes? doc "fill-opacity=\"0.25\""))
    (is (str/includes? doc "fill-opacity=\"0.125\""))))

(deftest gradient-mode-emits-one-def-per-used-colour-and-profile
  (let [doc (svg/field->svg field {:mode :gradient :colors 8 :hard-levels 2})
        gs  (re-seq #"<radialGradient id=\"(g\d+)\"" doc)]
    (is (seq gs))
    (is (<= (count gs) 6) "at most (splats × profiles) defs, and deduped")
    (testing "every referenced gradient is defined"
      (let [defined (set (map second gs))
            used    (set (map second (re-seq #"fill=\"url\(#(g\d+)\)\"" doc)))]
        (is (= used (set/intersection used defined)))
        (is (seq used))))
    (testing "the profile runs from opaque at the core to EXACTLY clear at the rim"
      ;; a rim left at the true 0.2% would be a step, and on a base stroke that step
      ;; is a visible disc edge across a smooth sky
      (is (re-find #"offset=\"0\" stop-color=\"[^\"]+\" stop-opacity=\"1\"" doc))
      (is (re-find #"offset=\"1\"[^/]*stop-opacity=\"0\"" doc))
      (is (not (re-find #"offset=\"1\"[^/]*stop-opacity=\"0\.0[0-9]" doc))))))

(deftest invisible-splats-are-culled
  (let [f (update field :splats conj
                  {:mean [1.0 1.0] :cov (gauss/covariance 2.0 2.0 0.0)
                   :color [1.0 1.0 1.0] :alpha 0.001 :detail 1.0})]
    (is (= 3 (count (els (svg/field->svg f {:mode :flat})))))))

(deftest scale-changes-the-canvas-not-the-coordinates
  (testing "the viewBox stays in image pixels — that IS the upscale"
    (let [doc (svg/field->svg field {:mode :flat :scale 4.0})]
      (is (str/includes? doc "width=\"160\" height=\"80\""))
      (is (str/includes? doc "viewBox=\"0 0 40 20\"")))))

(deftest background-is-the-fields-background
  (is (str/includes? (svg/field->svg (assoc field :background [1.0 0.5 0.0]) {:mode :flat})
                     "<rect width=\"40\" height=\"20\" fill=\"#ff8000\"/>")))

(deftest tone-passes-become-one-filter-over-the-whole-picture
  (testing "no filter at the no-op values"
    (let [doc (svg/field->svg field {:mode :flat})]
      (is (not (str/includes? doc "<filter")))
      (is (not (str/includes? doc "<g filter")))))
  (testing "Lift is the gamma transfer, Brightness the linear one, in that order"
    (let [doc (svg/field->svg field {:mode :flat :lift 1.8 :brightness 1.2})]
      ;; shader/fs-src-lift is c^(1/amount); feFunc gamma is amplitude·C^exponent+offset
      (is (str/includes? doc "type=\"gamma\" exponent=\"0.5556\""))
      (is (str/includes? doc "type=\"linear\" slope=\"1.2\""))
      (is (< (str/index-of doc "gamma") (str/index-of doc "linear")))
      (testing "SVG filters default to linearRGB; the shaders work on stored sRGB"
        (is (str/includes? doc "color-interpolation-filters=\"sRGB\"")))
      (testing "the filter covers the background too — the passes run on the composite"
        (is (< (str/index-of doc "<g filter=\"url(#tone)\">") (str/index-of doc "<rect")))
        (is (str/ends-with? doc "</g></svg>")))))
  (testing "either dial alone emits only its own transfer"
    (let [doc (svg/field->svg field {:mode :flat :brightness 1.2})]
      (is (str/includes? doc "type=\"linear\""))
      (is (not (str/includes? doc "type=\"gamma\""))))))

(deftest an-empty-field-still-produces-a-document
  (let [doc (svg/field->svg (assoc field :splats []) {:mode :gradient})]
    (is (str/ends-with? doc "</svg>"))
    (is (empty? (els doc)))))
