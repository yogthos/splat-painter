(ns splat-painter.structure-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.structure :as s]))

(defn- approx= [tol a b] (< (Math/abs (- (double a) (double b))) tol))

(defn- solid [H W [r g b :as c]]
  {:height H :width W :channels 3
   :pixels (double-array (apply concat (for [_ (range (* H W))] c)))})

(defn- gradient-img [H W f]
  (let [pixels (double-array (* H W 3))]
    (dotimes [x H]
      (dotimes [y W]
        (let [v (double (f x y))
              base (* 3 (+ (* x W) y))]
          (aset pixels base v)
          (aset pixels (inc base) v)
          (aset pixels (+ 2 base) v))))
    {:height H :width W :channels 3 :pixels pixels}))

(defn- tensor-at
  "Read the tensor at a full-image pixel the way the SEED does — straight out of the
   :theta/:coherence arrays. These fixtures are all far under analyze's 768px cap, so
   the tensor grid is the image grid and the index is just x*W+y. (This replaced
   s/orient-at, an accessor no production code used: prep-noise reads the arrays.)"
  [sfield x y]
  (let [W (long (:w sfield))
        i (+ (* (long x) W) (long y))]
    {:theta (aget ^doubles (:theta sfield) i)
     :coherence (aget ^doubles (:coherence sfield) i)
     :grad2 (aget ^doubles (:grad2 sfield) i)}))

(defn- near-angle? [tol a b]
  (let [d (Math/abs (- (mod a Math/PI) (mod b Math/PI)))
        d (min d (- Math/PI d))]
    (< d tol)))

(deftest horizontal-edge-gives-horizontal-stroke
  (testing "step across rows → gradient along x, stroke along y"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< x (/ H 2)) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (tensor-at sfield (quot H 2) (quot W 2))]
      (is (near-angle? 0.2 theta (/ Math/PI 2.0))
          (str "theta=" theta " should be ~π/2 (horizontal stroke)"))
      (is (> coherence 0.5)
          (str "coherence=" coherence " should be high at a sharp edge")))))

(deftest vertical-edge-gives-vertical-stroke
  (testing "step across cols → gradient along y, stroke along x"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< y (/ W 2)) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (tensor-at sfield (quot H 2) (quot W 2))]
      (is (near-angle? 0.2 theta 0.0)
          (str "theta=" theta " should be ~0 (vertical stroke)"))
      (is (> coherence 0.5)
          (str "coherence=" coherence " should be high at a sharp edge")))))

(deftest diagonal-edge
  (testing "step along x+y → gradient along (1,1), stroke along (1,-1) = 3π/4"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< (+ x y) H) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (tensor-at sfield (quot H 2) (quot W 2))]
      (is (near-angle? 0.3 theta (* 3.0 (/ Math/PI 4.0)))
          (str "theta=" theta " should be ~3π/4"))
      (is (> coherence 0.4)
          (str "coherence=" coherence " should be moderate at a diagonal edge")))))

(deftest flat-image-has-no-coherence
  (testing "solid color → no gradient → near-zero coherence"
    (let [img (solid 16 16 [0.5 0.5 0.5])
          sfield (s/analyze img)
          {:keys [coherence]} (tensor-at sfield 8 8)]
      (is (< coherence 0.1)
          (str "coherence=" coherence " should be near zero for a flat image")))))

(deftest determinism
  (testing "reading the tensor twice gives identical results"
    (let [img (gradient-img 16 16 (fn [x y] (if (< x 8) 0.2 0.8)))
          sfield (s/analyze img)]
      (is (= (tensor-at sfield 8 8)
             (tensor-at sfield 8 8))))))

(deftest luma-bt601-weights
  (testing "luma uses BT.601: 0.299*R + 0.587*G + 0.114*B"
    (let [img {:height 1 :width 2 :channels 3
               :pixels (double-array [1.0 0.0 0.0   ;; pure red
                                      0.0 1.0 0.0])} ;; pure green
          L (s/luma img)]
      (is (approx= 1e-12 0.299 (aget L 0)) "red pixel luma")
      (is (approx= 1e-12 0.587 (aget L 1)) "green pixel luma"))))

(deftest analyze-returns-expected-keys-and-gmax
  (testing "analyze returns :h :w :jxx :jyy :jxy :gmax with positive gmax on edges"
    (let [img (gradient-img 16 16 (fn [x y] (if (< x 8) 0.2 0.8)))
          sfield (s/analyze img)]
      (is (contains? sfield :h))
      (is (contains? sfield :w))
      (is (contains? sfield :jxx))
      (is (contains? sfield :jyy))
      (is (contains? sfield :jxy))
      (is (contains? sfield :gmax))
      (is (pos? (:gmax sfield)) "gmax should be >0 for a non-flat image"))))

(deftest flat-image-gmax-is-zero
  (testing "analyze of a solid-color image has gmax 0"
    (let [img (solid 16 16 [0.5 0.5 0.5])
          sfield (s/analyze img)]
      (is (approx= 1e-12 0.0 (:gmax sfield))))))

;; heavy blur bleed: the edge-preserving field must not halo in DARK regions

(defn- disc-on-dark [H W cx cy r bright dark]
  "A filled disc of `bright` (0..1 grey) on a `dark` ground — the silhouette case."
  (let [px (double-array (* H W 3))]
    (dotimes [x H]
      (dotimes [y W]
        (let [inside (<= (+ (* (- x cx) (- x cx)) (* (- y cy) (- y cy))) (* r r))
              v (if inside bright dark)
              b (* 3 (+ (* x W) y))]
          (aset px b v) (aset px (inc b) v) (aset px (+ b 2) v))))
    {:height H :width W :channels 3 :pixels px}))

(defn- grey-at [^doubles arr W i]
  (let [b (* 3 i)] (aget arr b)))

(defn- mean-luma-lift-in-ring
  "Mean (field − raw) luma over dark pixels whose distance from (cx,cy) is in
   [rlo,rhi] — a ring band around the disc."
  [img ^doubles field cx cy rlo rhi dark-tol]
  (let [H (:height img) W (:width img) ^doubles raw (:pixels img)]
    (loop [i 0 sum 0.0 cnt 0]
      (if (== i (* H W))
        (if (pos? cnt) (/ sum cnt) 0.0)
        (let [x (quot i W) y (rem i W)
              dist (Math/sqrt (+ (* (- x cx) (- x cx)) (* (- y cy) (- y cy))))
              raw-l (grey-at raw W i)]
          ;; dark-GROUND pixels: the ground sits exactly at dark-tol, so this must be
          ;; inclusive at the bottom. It was `(< dark-tol raw-l ...)`, which excluded
          ;; every pixel, gave cnt=0, and made the whole halo test read 0.0 -> 0.0.
          (if (and (<= raw-l (+ dark-tol 0.03))
                   (<= rlo dist rhi))
            (recur (inc i) (+ sum (- (grey-at field W i) raw-l)) (inc cnt))
            (recur (inc i) sum cnt)))))))

(deftest dominant-blur-kills-dark-halo-and-keeps-far-field
  (testing "heavy blur bleed across a silhouette: near-subject lift removed, far field untouched"
    ;; A bright disc on near-black. The heavy box blur carries disc brightness across the
    ;; silhouette; the coverage tiers' colour source must not. dominant-blur gets this from
    ;; bin separation rather than from a blend against the raw pixel: a window just outside
    ;; the disc is majority-dark, so the dark bin wins the vote outright and the disc's own
    ;; bin contributes nothing. (This property arrived with edge-preserving-blur, which fed
    ;; the coverage tiers until dominant-blur replaced it; the test follows the job.)
    (let [H 100 W 100 cx 50 cy 50 r 15 bright 0.85 dark 0.04
          img (disc-on-dark H W cx cy r bright dark)
          heavy (s/blur-image img 6)
          out   (s/dominant-blur img 6)
          ring  (fn [^doubles f rlo rhi] (mean-luma-lift-in-ring img f cx cy rlo rhi dark))]
      (let [before-near (ring heavy (+ r 1) (+ r 10))   ; ~10px outside the disc
            after-near  (ring out   (+ r 1) (+ r 10))
            ;; Lifts are in 0..1 units. MEASURED: the box blur lifts this ring 0.0790 and
            ;; dominant-blur cuts that to 0.0222 — a 3.6x suppression, where the
            ;; edge-preserving blur it replaced managed 0.00074. That is a real (small)
            ;; regression on this metric and the cap is set to the measured value, NOT
            ;; loosened to whatever passes: a soft-voted bin cannot fully out-vote a 1/3
            ;; minority at p=4 (0.67^4 vs 0.33^4 leaves the bright bin ~5% of the weight),
            ;; and raising p to bury it posterizes photographs. Accepted on render evidence
            ;; — DSC_8428 is this fixture in the wild, bright signage on near-black, and the
            ;; lights came out punchier rather than haloed. Revisit if a halo ever shows up.
            near-cap    0.03
            before-far  (ring heavy 75 90)
            after-far   (ring out   75 90)]
        (println "HALO-BLEED near-subject: heavy +" before-near " -> dominant +" after-near
                 "| far-field: heavy +" before-far " -> dominant +" after-far)
        ;; GUARD THE FIXTURE: a halo test whose fixture shows no halo passes against
        ;; zero and would stay green with the fix reverted. Assert the precondition.
        (is (> before-near (/ 12.0 255.0))
            (str "fixture must actually exhibit the bleed; heavy lift was " before-near))
        (is (<= after-near near-cap)
            (str "near-subject lift removed to <= " near-cap ": got " after-near
                 " (was " before-near " on the heavy field)"))
        (is (<= (Math/abs (- after-far before-far)) 0.5)
            (str "far field untouched (+/-0.5): after " after-far " vs before " before-far))))))

(deftest dominant-blur-preserves-flat-smoothing
  (testing "on a smooth gradient with no boundary the field stays close to the box blur (bokeh stays seamless)"
    ;; A unimodal window has no minority to out-vote, so the mode IS the mean and the
    ;; filter must be a near no-op — otherwise raising p would band a smooth gradient
    ;; instead of only dropping sub-brush structure.
    (let [img  (gradient-img 64 64 (fn [x y] (* 0.9 (/ (+ x y) 126.0))))
          heavy (s/blur-image img 6)
          out   (s/dominant-blur img 6)
          n (* 64 64)
          diff (loop [i 0 s 0.0]
                 (if (== i n) (/ s n)
                              (let [b (* 3 i)]
                                (recur (inc i)
                                       (+ s (Math/abs (- (aget out b) (aget heavy b))))))))]
      (println "FLAT-SMOOTH mean |out - box| =" diff)
      (is (< diff 0.02)
          (str "flat-region smoothing preserved: mean |out - box| small; got " diff)))))

;; dominant-blur: the COVERAGE tiers' colour source
;; The coverage tiers paint with strokes far larger than a fine feature, so their
;; colour must be the DOMINANT tone of a brush-sized window. The edge-preserving
;; blur they used to sample keeps sub-brush structure by construction: a σ6 stroke
;; centred on a 5px letter got solid black and painted it over a 12px radius, and
;; nothing repaints the paper around it — the smudge around text and line art.

(defn- line-and-block
  "A light field carrying two dark features: a THIN line (3px, far finer than the
   brush) at column `lc`, and a wide BLOCK (40px, far coarser) at columns bl..bl+39."
  [H W lc bl]
  (gradient-img H W (fn [_ y] (if (or (and (>= y lc) (< y (+ lc 3)))
                                      (and (>= y bl) (< y (+ bl 40))))
                                0.08 0.92))))

(defn- grey [^doubles f W x y] (aget f (* 3 (+ (* x W) y))))

(deftest dominant-blur-at-p1-is-the-box-blur
  ;; The identity anchor for the recombination: the hat kernels partition unity, so
  ;; combining bins by w^1 sums back to the plain window mean. If this drifts, the
  ;; binning or the normalisation is wrong and every higher p is wrong with it.
  ;; radius 2 keeps it on the full-resolution path (the downsample kicks in at 3).
  (let [H 48 W 48
        img (gradient-img H W (fn [x y] (* 0.9 (/ (double (+ x (* 2 y))) 144.0))))
        box (s/blur-image img 2)
        dom (s/dominant-blur img 2 1.0)]
    (is (every? (fn [i] (approx= 1e-9 (aget ^doubles box i) (aget ^doubles dom i)))
                (range (* H W 3)))
        "p=1 must reproduce the box blur exactly")))

(deftest dominant-blur-drops-sub-brush-structure-and-keeps-masses
  ;; The fix itself. At the brush's own scale a 3px line is a minority of the window
  ;; and must lose the vote; a 40px block is the majority and must survive. The
  ;; edge-preserving blur keeps BOTH, which is what made the coverage tier stamp
  ;; letter-sized features at brush size.
  (let [H 128 W 128 lc 30 bl 70 rad 12
        img   (line-and-block H W lc bl)
        dom   (s/dominant-blur img rad)
        at    (fn [f y] (grey f W 64 y))]
    (println (format "DOMINANT line %.3f (box %.3f) | mass %.3f | inside-edge %.3f | outside-edge %.3f"
                     (at dom (inc lc)) (grey (s/blur-image img rad) W 64 (inc lc))
                     (at dom (+ bl 20)) (at dom (+ bl 6)) (at dom (- bl 6))))
    ;; guard the fixture: the source really does carry both features
    (is (approx= 0.02 0.08 (grey (:pixels img) W 64 (inc lc))) "fixture: the thin line is dark")
    (is (approx= 0.02 0.08 (grey (:pixels img) W 64 (+ bl 20))) "fixture: the block is dark")
    ;; the thin line drops out — and the edge-preserving field it replaces keeps it
    ;; 0.88 sits between what the mode gives (0.919, i.e. the paper's own tone) and what a
    ;; plain window MEAN gives (0.819 — the line merely diluted, which is the halo). A looser
    ;; bound passes for p=1 and would not test anything.
    (is (> (at dom (inc lc)) 0.88)
        (str "a 3px line must lose the vote in a " rad "px window, not just be diluted by it: got "
             (at dom (inc lc)) " (the window mean is " (grey (s/blur-image img rad) W 64 (inc lc)) ")"))
    (is (< (at (s/bilateral-blur img 3) (inc lc)) 0.3)
        (str "an edge-PRESERVING field of the same window keeps it at full strength, which is "
             "the behaviour being replaced: got " (at (s/bilateral-blur img 3) (inc lc))))
    ;; the 40px mass survives, and neither side bleeds across its boundary
    (is (< (at dom (+ bl 20)) 0.3)
        (str "a 40px mass must survive: got " (at dom (+ bl 20))))
    (is (< (at dom (+ bl 6)) 0.25)
        (str "no bleed IN from the light side just inside the mass: got " (at dom (+ bl 6))))
    (is (> (at dom (- bl 6)) 0.85)
        (str "no bleed OUT onto the light side just outside it: got " (at dom (- bl 6))))))

;; edge-width-field: total_variation / peak_gradient. Must separate a crisp
;; low-contrast edge from a soft high-contrast one — the property :edge/:sharp
;; lack. Amplitude-invariant: doubling the step contrast does not move width.
(deftest edge-width-separates-crisp-from-soft
  (testing "a 1px step is narrower than a ramped transition, regardless of contrast"
    ;; crisp 1px vertical step: left half 0.2, right half 0.9
    (let [crisp (gradient-img 32 32 (fn [x y] (if (< y 16) 0.2 0.9)))
          ;; soft 8px ramp across the same boundary
          soft  (gradient-img 32 32 (fn [x y]
                                       (cond (< y 12) 0.2
                                             (< y 20) (+ 0.2 (* 0.0875 (- y 12)))
                                             :else 0.9)))
          cw (s/edge-width-field crisp)
          sw (s/edge-width-field soft)
          ^doubles cwa (:width cw)
          ^doubles swa (:width sw)
          W (long (:w cw))
          c-at (fn [x y] (aget cwa (+ (* (long x) W) (long y))))
          s-at (fn [x y] (aget swa (+ (* (long x) W) (long y))))]
      ;; both fields cover the full image
      (is (= (* 32 32) (alength cwa)))
      (is (= (* 32 32) (alength swa)))
      ;; at the boundary the SOFT ramp reads WIDER than the crisp step
      (is (< (c-at 16 15) (s-at 16 15))
          (str "crisp edge width " (c-at 16 15) " must be < soft ramp width " (s-at 16 15))))))

(deftest edge-width-is-amplitude-invariant
  (testing "doubling the step contrast does not change the measured width"
    (let [weak (gradient-img 32 32 (fn [x y] (if (< y 16) 0.3 0.5)))   ; 0.2 contrast
          strong (gradient-img 32 32 (fn [x y] (if (< y 16) 0.1 0.9))); 0.8 contrast
          ^doubles wwa (:width (s/edge-width-field weak))
          ^doubles swa (:width (s/edge-width-field strong))
          W 32
          w-at (fn [a x y] (aget a (+ (* (long x) W) (long y))))]
      (is (approx= 0.05 (w-at wwa 16 15) (w-at swa 16 15))
          "width must not depend on contrast amplitude"))))
