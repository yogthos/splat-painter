(ns splat-painter.score-test
  "Tests for splat-painter.score, the perceptual render metric. The two
   properties that justify the instrument's existence are pinned here: chroma
   catches hue errors luma cannot see, and SSIM separates preserved detail
   from detail averaged away. Everything is constructed in-test — no renders,
   no /tmp files."
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.score :as s]))

(defn- approx? [tol a b]
  (if (and (number? a) (number? b))
    (< (Math/abs (- (double a) (double b))) tol)
    (and (sequential? a) (sequential? b)
         (= (count a) (count b))
         (every? true? (map #(approx? tol %1 %2) a b)))))

(defn- structured-plane
  "A flat H*W luma plane in the 0..255 scale ssim works in, with strong local
   variance (gradient + checkerboard + sinusoidal texture). Structure is what
   makes the SSIM gap between a plane and its own blur large and unambiguous."
  [W H]
  (let [p (double-array (* W H))]
    (dotimes [y H]
      (dotimes [x W]
        (let [grad (* 255.0 (/ y (dec H)))
              checker (if (zero? (mod (+ x y) 2)) 30.0 0.0)
              tex (* 20.0 (Math/sin (* 0.7 x)) (Math/sin (* 0.9 y)))]
          (aset p (+ (* y W) x) (min 255.0 (+ grad checker (Math/abs tex)))))))
    p))

(defn- band-image
  "Flat H*W*3 0..1 buffer of `colors` horizontal bands, each band a different
   luma AND hue. Vertical structure matters: the scoring path pairs render row
   y with source row y+1, so without vertical variation the registration
   offset produces no baseline at all."
  [W H colors]
  (let [out (double-array (* 3 W H))
        band-h (/ H (count colors))]
    (dotimes [y H]
      (let [[r g b] (nth colors (quot y band-h))]
        (dotimes [x W]
          (let [j (* 3 (+ (* y W) x))]
            (aset out j r) (aset out (inc j) g) (aset out (+ j 2) b)))))
    out))

(defn- hue-rotate
  "Rotate every pixel's chroma around the BT.709 luma axis by `theta`, preserving
   luma exactly: each color c = (Y,Y,Y) + v with Y its luma and v in the 2D null
   space of the luma weights; v is rotated in an orthonormal basis (e1, e2) of
   that null space built in code (normalized cross products), so the rotated
   color stays in the null space to float precision. The weights here MUST match
   luma-plane's — that identity is what keeps mean|d| flat."
  [^doubles px W H theta]
  (let [n    (* W H)
        wr 0.2126 wg 0.7152 wb 0.0722
        e1l (Math/sqrt (+ (* 0.7152 0.7152) (* 0.2126 0.2126)))
        e1x (/ 0.7152 e1l) e1y (/ -0.2126 e1l) e1z 0.0
        e2x (- (* wg e1z) (* wb e1y))
        e2y (- (* wb e1x) (* wr e1z))
        e2z (- (* wr e1y) (* wg e1x))
        e2l (Math/sqrt (+ (* e2x e2x) (* e2y e2y) (* e2z e2z)))
        e2x (/ e2x e2l) e2y (/ e2y e2l) e2z (/ e2z e2l)
        ct (Math/cos theta) st (Math/sin theta)
        out (double-array (* 3 n))]
    (dotimes [i n]
      (let [j  (* 3 i)
            r  (aget px j) g (aget px (inc j)) bl (aget px (+ j 2))
            Y  (+ (* wr r) (* wg g) (* wb bl))
            dr (- r Y) dg (- g Y) db (- bl Y)
            al (+ (* dr e1x) (* dg e1y) (* db e1z))
            be (+ (* dr e2x) (* dg e2y) (* db e2z))
            al2 (- (* al ct) (* be st))
            be2 (+ (* al st) (* be ct))
            nr (+ Y (* al2 e1x) (* be2 e2x))
            ng (+ Y (* al2 e1y) (* be2 e2y))
            nb (+ Y (* al2 e1z) (* be2 e2z))
            nr (if (< nr 0.0) 0.0 (if (> nr 1.0) 1.0 nr))
            ng (if (< ng 0.0) 0.0 (if (> ng 1.0) 1.0 ng))
            nb (if (< nb 0.0) 0.0 (if (> nb 1.0) 1.0 nb))]
        (aset out j nr) (aset out (inc j) ng) (aset out (+ j 2) nb)))
    out))

(deftest luma-plane-keeps-score-py-0-255-scale
  (testing "score.py works in 0..255 pixel units (it loads via PIL); a port
            computing in 0..1 would claim parity on numbers 255x apart"
    (let [g (double-array [0.5 0.5 0.5 0.25 0.25 0.25 1.0 1.0 1.0])
          p (s/luma-plane g 3 1)]
      (is (approx? 1e-9 127.5 (aget p 0)))
      (is (approx? 1e-9 63.75 (aget p 1)))
      (is (approx? 1e-9 255.0 (aget p 2))))))

(deftest oklab-planes-match-published-values
  (testing "srgb->oklab must agree with Ottosson's published values — the
            matrices are hardcoded in score.clj. White pins both matrix
            properties at once: M1 row sums are 1 (gray stays on the diagonal),
            M2 row sums are 0 (gray maps to a=b=0)."
    (is (approx? 1e-6 [1.0 0.0 0.0] (s/srgb->oklab 1.0 1.0 1.0)))
    (is (approx? 1e-12 [0.0 0.0 0.0] (s/srgb->oklab 0.0 0.0 0.0)))
    ;; sRGB #FF0000 -> Oklab (oklab.com): catches transposed matrices, wrong
    ;; channel order, or a gamma error the white test cannot see
    (is (approx? 1e-4 [0.627955 0.224863 0.125846] (s/srgb->oklab 1.0 0.0 0.0)))))

(deftest ssim-identical-scores-one-blurred-below
  (testing "ssim must be called DIRECTLY on two aligned luma planes. The
            scoring path applies score.py's unconditional 1px registration
            offset (render row y-1 == source row y), which is correct for
            render-vs-source but compares two ALIGNED inputs 1px apart: an
            identical pair scores 0.585 there and a blurred copy scores HIGHER
            (0.665) because blurring smooths the dipole the offset creates.
            Both properties below are false through -M:score; they hold on the
            function itself, which is where they belong."
    (let [W 64 H 64
          p (structured-plane W H)
          blur (s/blur-plane p W H 1.5)]
      (is (= 1.0 (s/ssim p p W H)))
      (is (< (s/ssim p blur W H) 0.9))
      (is (< (s/ssim p blur W H) (s/ssim p p W H))))))

(deftest hue-change-at-matched-luma-moves-chroma-not-luma
  (testing "the chroma column exists to catch what luma cannot: a hue rotation
            that preserves BT.709 luma per pixel must leave mean|d| flat while
            the chroma column jumps. The real pair moved luma 8.203 -> 8.217
            (+0.17%) and chroma 0.002 -> 0.048 (~24x); here the rotation is
            exact, so the luma ratio pins tighter than the measured 0.17%."
    (let [W 96 H 72
          src  (band-image W H [[0.75 0.60 0.50] [0.40 0.50 0.60] [0.55 0.45 0.35]])
          hue  (hue-rotate src W H (* 2.0 (/ Math/PI 3.0)))
          base (s/score-buffers src src W H)
          rot  (s/score-buffers src hue W H)
          l0 (:full-mean base) l1 (:full-mean rot)
          c0 (:full-chroma base) c1 (:full-chroma rot)]
      (is (pos? c0) "the registration offset alone must give a small chroma baseline")
      (is (< (/ (Math/abs (- l1 l0)) l0) 1e-3)
          "hue rotation at matched luma must not move the luma columns")
      (is (> (/ c1 c0) 5) "chroma must jump by an order of magnitude, not a few percent")
      (is (> c1 0.03) "and the chroma move must be absolutely significant"))))

(deftest small-image-roi-falls-back-to-whole-plane
  (testing "an image below the 610x455 nose-ROI box must not throw: the ROI
            columns fall back to the whole image and :roi-clamped says so"
    (let [W 40 H 30
          flat (double-array (* 3 W H) 0.5)
          s (s/score-buffers flat flat W H)]
      (is (true? (:roi-clamped s)))
      (is (= 0.0 (:roi-mean s)))
      (is (= 0.0 (:roi-chroma s)))
      (is (= 1.0 (:roi-ssim s)))
      (is (= (:roi-mean s) (:full-mean s))
          "a clamped ROI covers the whole image, so the columns must agree")
      (is (= [[0 29 0 40] true] (s/roi-region 29 40))))))

(deftest roi-region-geometry
  (testing "the nose ROI box is deliberately pinned: it is part of score.py
            parity, and silently drifting it would change every number the
            seed.clj constants are justified with"
    (is (= [[404 454 520 590] false] (s/roi-region 682 1024)))))
