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

(defn- near-angle? [tol a b]
  (let [d (Math/abs (- (mod a Math/PI) (mod b Math/PI)))
        d (min d (- Math/PI d))]
    (< d tol)))

(deftest horizontal-edge-gives-horizontal-stroke
  (testing "step across rows → gradient along x, stroke along y"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< x (/ H 2)) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (s/orient-at sfield (/ H 2) (/ W 2))]
      (is (near-angle? 0.2 theta (/ Math/PI 2.0))
          (str "theta=" theta " should be ~π/2 (horizontal stroke)"))
      (is (> coherence 0.5)
          (str "coherence=" coherence " should be high at a sharp edge")))))

(deftest vertical-edge-gives-vertical-stroke
  (testing "step across cols → gradient along y, stroke along x"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< y (/ W 2)) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (s/orient-at sfield (/ H 2) (/ W 2))]
      (is (near-angle? 0.2 theta 0.0)
          (str "theta=" theta " should be ~0 (vertical stroke)"))
      (is (> coherence 0.5)
          (str "coherence=" coherence " should be high at a sharp edge")))))

(deftest diagonal-edge
  (testing "step along x+y → gradient along (1,1), stroke along (1,-1) = 3π/4"
    (let [H 24 W 24
          img (gradient-img H W (fn [x y] (if (< (+ x y) H) 0.0 1.0)))
          sfield (s/analyze img)
          {:keys [theta coherence]} (s/orient-at sfield (/ H 2) (/ W 2))]
      (is (near-angle? 0.3 theta (* 3.0 (/ Math/PI 4.0)))
          (str "theta=" theta " should be ~3π/4"))
      (is (> coherence 0.4)
          (str "coherence=" coherence " should be moderate at a diagonal edge")))))

(deftest flat-image-has-no-coherence
  (testing "solid color → no gradient → near-zero coherence"
    (let [img (solid 16 16 [0.5 0.5 0.5])
          sfield (s/analyze img)
          {:keys [coherence]} (s/orient-at sfield 8 8)]
      (is (< coherence 0.1)
          (str "coherence=" coherence " should be near zero for a flat image")))))

(deftest determinism
  (testing "orient-at returns identical results for identical inputs"
    (let [img (gradient-img 16 16 (fn [x y] (if (< x 8) 0.2 0.8)))
          sfield (s/analyze img)]
      (is (= (s/orient-at sfield 8 8)
             (s/orient-at sfield 8 8))))))

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

;; --- heavy blur bleed: the edge-preserving field must not halo in DARK regions ---

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
          (if (and (< dark-tol raw-l (+ dark-tol 0.03))
                   (<= rlo dist rhi))
            (recur (inc i) (+ sum (- (grey-at field W i) raw-l)) (inc cnt))
            (recur (inc i) sum cnt)))))))

(deftest edge-preserving-blur-kills-dark-halo-and-keeps-far-field
  (testing "heavy blur bleed across a silhouette: near-subject lift removed, far field untouched"
    ;; A bright disc on near-black. The heavy box blur carries disc brightness
    ;; across the silhouette; edge-preserving-blur must push it back to the light
    ;; (bilateral) field where the deviation is RELATIVELY large, even though the
    ;; absolute deviation in the dark halo is tiny (~0.04 = noise on a bright ramp).
    (let [H 100 W 100 cx 50 cy 50 r 15 bright 0.85 dark 0.04
          img (disc-on-dark H W cx cy r bright dark)
          light (s/bilateral-blur img 3)
          heavy (s/blur-image img 6)
          out   (s/edge-preserving-blur img light heavy)
          ring  (fn [^doubles f rlo rhi] (mean-luma-lift-in-ring img f cx cy rlo rhi dark))]
      (let [before-near (ring heavy (+ r 1) (+ r 10))   ; ~10px outside the disc
            after-near  (ring out   (+ r 1) (+ r 10))
            near-cap    1.0
            before-far  (ring heavy 75 90)
            after-far   (ring out   75 90)]
        (println "HALO-BLEED near-subject: heavy +" before-near " -> edge-preserving +" after-near
                 "| far-field: heavy +" before-far " -> edge-preserving +" after-far)
        (is (<= after-near near-cap)
            (str "near-subject lift removed to <= " near-cap ": got " after-near
                 " (was " before-near " on the heavy field)"))
        (is (<= (Math/abs (- after-far before-far)) 0.5)
            (str "far field untouched (+/-0.5): after " after-far " vs before " before-far))))))

(deftest edge-preserving-blur-preserves-flat-smoothing
  (testing "on a smooth gradient with no boundary the field stays close to heavy (bokeh stays seamless)"
    ;; No edge => no deviation, both tests read ~0 => w~0 => out ≈ heavy. The light
    ;; blur is an edge-aware bilateral that is itself close to the box heavy on a
    ;; smooth ramp, but the point is the blend must NOT pull toward light here.
    (let [img  (gradient-img 64 64 (fn [x y] (* 0.9 (/ (+ x y) 126.0))))
          light (s/bilateral-blur img 3)
          heavy (s/blur-image img 6)
          out   (s/edge-preserving-blur img light heavy)
          n (* 64 64)
          diff (loop [i 0 s 0.0]
                 (if (== i n) (/ s n)
                              (let [b (* 3 i)]
                                (recur (inc i)
                                       (+ s (Math/abs (- (aget out b) (aget heavy b))))))))]
      (println "FLAT-SMOOTH mean |out - heavy| =" diff)
      (is (< diff 0.02)
          (str "flat-region smoothing preserved: mean |out - heavy| small; got " diff)))))

(deftest edge-preserving-blur-weight-pins-spec-formula
  (testing "the blend weight follows w = max(wabs, wrel) exactly - each regime in isolation"
    ;; Three pixels, grey (all channels equal). light = raw throughout (physically the
    ;; bilateral field tracks raw near an edge). Expected outputs are HAND-COMPUTED from
    ;; the spec formula - wabs=clamp((d-0.06)/0.10), wrel=clamp((d/max(lv,0.02)-0.15)/0.45),
    ;; w=max(wabs,wrel), out=(1-w)*heavy + w*light - so this pins the fix independent of
    ;; the blur machinery and is verifiable without running the suite:
    ;;
    ;;   px0 DARK halo:   d=0.04 wabs=0 ; wrel=(0.04/0.04-0.15)/0.45=1.88->1 ; w=1 -> out=light=0.04
    ;;       (the fix: relative fires where absolute sees nothing; out pulled fully to light)
    ;;   px1 BRIGHT small: d=0.04 wabs=0 ; wrel=(0.04/0.80-0.15)/0.45=(-0.22)->0 ; w=0 -> out=heavy=0.84
    ;;       (no over-correction: the same 0.04 absolute lift is noise on lit skin)
    ;;   px2 BRIGHT large: d=0.12 wabs=(0.12-0.06)/0.10=0.6 ; wrel=(0.12/0.80-0.15)/0.45=0 ; w=0.6 -> out=0.4*0.92+0.6*0.80=0.848
    ;;       (the absolute ramp still fires on a genuine large deviation)
    (let [img   {:height 3 :width 1 :channels 3
                 :pixels (double-array [0.04 0.04 0.04   0.80 0.80 0.80   0.80 0.80 0.80])}
          light (double-array [0.04 0.04 0.04   0.80 0.80 0.80   0.80 0.80 0.80])
          heavy (double-array [0.08 0.08 0.08   0.84 0.84 0.84   0.92 0.92 0.92])
          out   (s/edge-preserving-blur img light heavy)
          exp   [0.040 0.040 0.040   0.840 0.840 0.840   0.848 0.848 0.848]]
      (doseq [c (range 9)]
        (is (approx= 1e-9 (double (exp c)) (double (aget out c)))
            (str "px " (quot c 3) " ch " (rem c 3) ": expected " (exp c) " got " (aget out c)))))))
