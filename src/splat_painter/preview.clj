(ns splat-painter.preview
  "Headless CPU preview → PNG (native jolt.png, no GL context needed). Composites the
   splat field front-to-back with the SAME size→hardness the shader uses, so the painted
   look — facets, blur, paint order, stroke shape — can be inspected without the GTK app.
   Run: joltc -M:preview <in-image> <out.png> [size]"
  (:require [splat-painter.image :as image]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.seed :as seed]))

(defn -main [& [path out szs counts hards]]
  (let [path (or path "img/collapse-watch.jpg")
        out  (or out "/tmp/ga_preview.png")
        img0 (image/load-image path 1000)
        sfield (structure/analyze img0)
        img  (assoc img0 :structure sfield :detail (wavelet/placement-map img0 sfield)
                    :blur (structure/blur-image img0 2)
                    :blur-heavy (let [l (structure/blur-image img0 2)
                                      h (structure/blur-image img0 (max 6 (quot (:height img0) 80)))]
                                  (structure/edge-preserving-blur img0 l h))
                    :noise-fields (seed/prep-noise sfield))
        size (if szs (Double/parseDouble szs) (max 4.0 (/ (double (:height img)) 50.0)))
        count* (if counts (long (Double/parseDouble counts)) 14000)
        fld  (seed/splat-field img {:size size :count count* :detail 0.6 :variation 0.5
                                    :curvature 0.5 :stroke 2.5 :opacity 0.9 :contrast 1.0})
        {:keys [splats background height width opacity sig-min sig-max]} fld
        H (long height) W (long width)
        hard-sharp (if hards (Double/parseDouble hards) 1.7) hard-soft 1.0
        sig-min (double sig-min) sig-max (double sig-max)
        opacity (double opacity)
        acc (double-array (* H W 3))
        T   (double-array (* H W))
        b0 (double (nth background 0)) b1 (double (nth background 1)) b2 (double (nth background 2))]
    (dotimes [i (* H W)] (aset T i 1.0))
    (println (format "%dx%d  %d splats  size %.1f  sig %.1f..%.1f" W H (count splats) size sig-min sig-max))
    (doseq [{:keys [mean cov color alpha]} splats]
      (let [mx (double (nth mean 0)) my (double (nth mean 1))
            A  (double (or alpha 1.0))
            c00 (double (nth cov 0)) c01 (double (nth cov 1)) c11 (double (nth cov 3))
            det (max (- (* c00 c11) (* c01 c01)) 1e-8)
            p00 (/ c11 det) p11 (/ c00 det) cross (/ (* -2.0 c01) det)
            sig (Math/sqrt (Math/sqrt det))
            tsc (min 1.0 (max 0.0 (/ (- sig sig-min) (max (- sig-max sig-min) 1e-4))))
            ts  (* tsc tsc (- 3.0 (* 2.0 tsc)))
            hard0 (+ (* hard-sharp (- 1.0 ts)) (* hard-soft ts))
            hard (+ 1.0 (* (- hard0 1.0) (min 1.0 (max 0.0 (/ sig 2.5)))))
            rad (long (Math/ceil (* 3.0 (Math/sqrt (max c00 c11)))))
            cr (double (nth color 0)) cg (double (nth color 1)) cb (double (nth color 2))
            x0 (max 0 (- (long mx) rad)) x1 (min (dec H) (+ (long mx) rad))
            y0 (max 0 (- (long my) rad)) y1 (min (dec W) (+ (long my) rad))]
        (loop [x x0]
          (when (<= x x1)
            (let [dx (- (double x) mx)]
              (loop [y y0]
                (when (<= y y1)
                  (let [dy (- (double y) my)
                        pdf (* 0.5 (+ (* p00 dx dx) (* cross dx dy) (* p11 dy dy)))
                        a (* A opacity (Math/exp (- (Math/pow pdf hard))))
                        i (+ (* x W) y)
                        t (aget T i)
                        wa (* t a)
                        b (* 3 i)]
                    (aset acc b (+ (aget acc b) (* wa cr)))
                    (aset acc (+ b 1) (+ (aget acc (+ b 1)) (* wa cg)))
                    (aset acc (+ b 2) (+ (aget acc (+ b 2)) (* wa cb)))
                    (aset T i (* t (- 1.0 a))))
                  (recur (inc y)))))
            (recur (inc x))))))
    (let [pimg (jolt.png/image W H)]
      (dotimes [i (* H W)]
        (let [t (aget T i) b (* 3 i)]
          (jolt.png/put! pimg
                         (* 255.0 (+ (aget acc b) (* t b0)))
                         (* 255.0 (+ (aget acc (+ b 1)) (* t b1)))
                         (* 255.0 (+ (aget acc (+ b 2)) (* t b2))))))
      (jolt.png/write pimg W H out)
      (println (str "wrote " out)))))
