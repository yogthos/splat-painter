(ns splat-painter.svg-test
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [splat-painter.gaussian :as gauss]
            [splat-painter.shader :as shader]
            [splat-painter.svg :as svg]))

(defn- approx? [tol a b] (< (Math/abs (- (double a) (double b))) (double tol)))

(defn- els [doc] (re-seq #"<(?:circle|ellipse)[^/]*/>" doc))

(defn- attr [el name]
  (when-let [m (re-find (re-pattern (str name "=\"([^\"]*)\"")) el)]
    (second m)))

;; ---------------------------------------------------------------- number output

(deftest coordinates-are-compact-and-lossless-to-the-decimal
  (let [f #'svg/fixed]
    (is (= "5" (f 5.0 1)))
    (is (= "5.1" (f 5.14 1)))
    (is (= "5.2" (f 5.15 1)))
    (is (= ".1" (f 0.1 3)) "trailing zeros dropped, and so is the leading zero")
    (is (= ".001" (f 0.001 3)) "the zeros INSIDE the fraction are kept")
    (is (= "-.1" (f -0.06 1)))
    (is (= "0" (f -0.04 1)) "a value that rounds to zero has no sign")
    (is (= "1000" (f 1000.0 0)))
    ;; the tone filter asks for 4 — a table too short for it is an index-out-of-bounds
    ;; at save time, not a rounding error
    (is (= ".5556" (f (/ 1.0 1.8) 4)))
    (is (= ".555556" (f (/ 1.0 1.8) 6)))))

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

;; ---------------------------------------------------------------- fidelity

(deftest fidelity-1-is-a-no-op-on-every-knob-it-drives
  (let [o (svg/fidelity->opts 1.0)]
    (is (zero? (:cull-peak o)) "nothing is pruned")
    (is (= 512 (:colors o))    "the full palette")
    (is (= 1.0 (:round o))     "circles only where the splat is already round")
    (is (= 1 (:dp o))          "full coordinate precision")))

(deftest fidelity-moves-every-knob-monotonically
  (let [ks (map svg/fidelity->opts [1.0 0.75 0.5 0.25 0.0])]
    (is (apply < (map :cull-peak ks)) "prunes harder as fidelity drops")
    (is (apply > (map :colors ks))    "and quantizes harder")
    (is (apply < (map :round ks))     "and rounds more strokes to circles"))
  (is (= (svg/fidelity->opts 0.0) (svg/fidelity->opts -3.0)) "clamped below")
  (is (= (svg/fidelity->opts 1.0) (svg/fidelity->opts 9.0)) "and above"))

;; ---------------------------------------------------------------- pruning

(defn- dr [cx cy r alpha]
  {:cx (double cx) :cy (double cy) :rx (double r) :ry (double r)
   :cos 1.0 :sin 0.0 :hard 1.0 :alpha (double alpha) :color [0.5 0.5 0.5]})

;; A backing daub big and flat enough to be opaque over the whole 64x64 canvas, in the
;; backmost slot. Without one the corners are bare ground, the repair pass is right to
;; put everything back, and no pruning test can say anything. The real field has this
;; by construction: the base level covers the image with no gaps.
(def ^:private backing (dr 32 32 400 1.0))

(defn- mask [draws thresh]
  (svg/keep-mask (conj (vec draws) backing) 64 64 thresh 1 3.5))

(defn- kept [draws thresh] (vec (seq (:keep (mask draws thresh)))))

(deftest nothing-is-pruned-at-threshold-zero
  (is (= [1 1 1] (kept [(dr 32 32 8 1.0) (dr 32 32 8 1.0)] 0.0))))

(deftest an-occluded-splat-goes-and-a-visible-one-stays
  ;; index 0 is the TOPMOST stroke; index 1 sits directly behind an opaque one
  (is (= [1 0 1] (kept [(dr 32 32 8 1.0) (dr 32 32 4 1.0)] 0.5)))
  ;; the same splat with nothing over it survives
  (is (= [1 1 1] (kept [(dr 8 8 8 1.0) (dr 32 32 4 1.0)] 0.5))))

(deftest a-small-fully-visible-stroke-outranks-a-large-faint-one
  ;; THE reason the score is a max and not a sum. The liner stroke covers a fraction of
  ;; the glaze's area, so by summed contribution the glaze wins and the detail is
  ;; pruned first — which is backwards. By peak the stroke is opaque and the glaze is
  ;; not, and the stroke stays.
  (let [glaze  (dr 8 8 16 0.08)
        stroke (dr 48 48 1.2 0.95)]
    (is (= [0 1 1] (kept [glaze stroke] 0.2)))))

(deftest a-splat-hidden-at-its-centre-but-exposed-at-its-rim-stays
  ;; reading the peak only at the centre culled this and opened a hole where the rim
  ;; was the last paint on the canvas
  (let [cap  (dr 32 32 3 1.0)       ; small opaque cap over the big one's middle
        wide (dr 32 32 12 0.9)]
    (is (= [1 1 1] (kept [cap wide] 0.5)))))

(deftest a-dropped-splat-does-not-consume-transmittance
  ;; the coverage guarantee: pruning the glaze in front must leave the opaque splat
  ;; behind it reading a clear canvas, so it cannot be pruned in turn
  (let [{:keys [keep residual repaired]} (mask [(dr 32 32 8 0.5) (dr 32 32 8 1.0)] 0.6)]
    (is (= [0 1 1] (vec (seq keep))))
    (is (zero? repaired) "the opaque splat behind covers it, so there is nothing to repair")
    (is (< residual 0.01) "and no background shows through")))

(deftest holes-are-repaired-rather-than-left-dark
  ;; NO backing here: two glazes are the only paint on the canvas, each scores under the
  ;; threshold, and dropping both is a hole straight to the background. Left alone every
  ;; such hole shows the black clear, which reads as the whole picture going darker — so
  ;; the repair pass walks the dropped splats back to front and puts them back.
  (let [{:keys [keep repaired]} (svg/keep-mask [(dr 32 32 8 0.5) (dr 32 32 8 0.5)]
                                               64 64 0.6 1 3.5)]
    (is (= [1 1] (vec (seq keep))))
    (is (= 2 repaired))))

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
      ;; splat 0 is elongated along the image's ROW axis, so it is rotated, and a
      ;; rotated splat carries its centre once in the transform rather than three times
      (is (= "translate(10 5)rotate(90)" (attr (last es) "transform")))
      ;; splat 1 is isotropic — a circle, no rotate, centre in cx/cy
      (is (str/starts-with? (second es) "<circle"))
      (is (= "20" (attr (second es) "cx")))
      (is (= "6" (attr (second es) "cy"))))))

(deftest a-round-splat-is-a-circle-and-a-rotated-one-carries-its-centre-once
  (let [doc (svg/field->svg field {:mode :flat})]
    (testing "rotating a circle does nothing, so both the second radius and the rotate go"
      (is (str/includes? doc "<circle"))
      (is (not (re-find #"<circle[^/]*transform" doc))))
    (testing "and a rotated ellipse writes translate()rotate() instead of cx/cy + rotate(a cx cy)"
      (is (re-find #"<ellipse[^/]*transform=\"translate\([^)]*\)rotate\([^)]*\)\"" doc))
      (is (not (re-find #"rotate\([-0-9.]+ " doc)) "no centre repeated inside rotate()"))))

(deftest alpha-rides-fill-opacity-never-opacity
  (testing "`opacity` composites the element as an isolated group — 15× slower per splat"
    (let [doc (svg/field->svg field {:mode :flat})]
      (is (not (re-find #"<(?:circle|ellipse)[^/]*[^-]opacity=" doc)))
      (is (str/includes? doc "fill-opacity=\".5\""))
      (is (str/includes? doc "fill-opacity=\".25\"")))))

(deftest global-opacity-multiplies-into-the-per-splat-alpha
  (let [doc (svg/field->svg (assoc field :opacity 0.5) {:mode :flat})]
    (is (str/includes? doc "fill-opacity=\".25\""))
    (is (str/includes? doc "fill-opacity=\".13\""))))

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
      (is (str/includes? doc "type=\"gamma\" exponent=\".5556\""))
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
