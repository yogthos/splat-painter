(ns splat-painter.precision
  "Float32-emulation of the GS band trace to test whether the CPU/GPU position
   divergence is float32-vs-float64 amplification in the edge-snap / sign
   decisions near a dying ridge.

   Run: jolt -A:test -m splat-painter.precision <image> <maxside>
   Reads /tmp/chain-args.edn (written by splat-painter.parity trace mode) for the
   diverging chain's seed + params."
  (:require [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.seed :as seed]
            [splat-painter.wavelet :as wavelet]))

(def ^:private sample-fields (var splat-painter.seed/sample-fields))
(def ^:private sample-arr (var splat-painter.seed/sample-arr))
(def ^:private edge-at (var splat-painter.wavelet/edge-at))

(defn- f32 [x] (double (float x)))
(defn- clamp01 [x] (max 0.0 (min 1.0 x)))

(defn- bilerp1-32
  "Float32 bilinear of a flat double array (H rows x W cols) at image coords x,y
   via H/srch, W/srcw — mirror of GS fieldBilerp."
  [a H W srch srcw x y]
  (let [fx (f32 (min (double (dec H)) (max 0.0 (* (double x) (/ (double H) (double srch))))))
        fy (f32 (min (double (dec W)) (max 0.0 (* (double y) (/ (double W) (double srcw))))))
        i0 (long (Math/floor fx)) j0 (long (Math/floor fy))
        i1 (min (dec H) (inc i0)) j1 (min (dec W) (inc j0))
        wx (f32 (- fx (double i0))) wy (f32 (- fy (double j0)))
        lerp (fn [u v t] (f32 (+ (f32 (* (f32 (- 1.0 t)) u)) (f32 (* t v)))))]
    (lerp (lerp (f32 (aget a (+ (* i0 W) j0))) (f32 (aget a (+ (* i0 W) j1))) wy)
          (lerp (f32 (aget a (+ (* i1 W) j0))) (f32 (aget a (+ (* i1 W) j1))) wy)
          wx)))

(defn- gs-field
  "GS fieldsAt emulation: per-texel swirl mix then float32 bilinear,
   θ = 0.5·atan2(s,c), coh clamped. Returns [th coh]."
  [nf x y]
  (let [H (long (:h nf)) W (long (:w nf))
        srch (long (or (:src-h nf) H)) srcw (long (or (:src-w nf) W))
        c2 (:c2 nf) s2 (:s2 nf) co (:coherence nf)
        c2s (or (:c2s nf) c2) s2s (or (:s2s nf) s2)
        fx (f32 (min (double (dec H)) (max 0.0 (* (double x) (/ (double H) (double srch))))))
        fy (f32 (min (double (dec W)) (max 0.0 (* (double y) (/ (double W) (double srcw))))))
        i0 (long (Math/floor fx)) j0 (long (Math/floor fy))
        i1 (min (dec H) (inc i0)) j1 (min (dec W) (inc j0))
        wx (f32 (- fx (double i0))) wy (f32 (- fy (double j0)))
        lerp (fn [u v t] (f32 (+ (f32 (* (f32 (- 1.0 t)) u)) (f32 (* t v)))))
        mixv (fn [i j]
               (let [idx (+ (* i W) j)
                     c (f32 (aget c2 idx)) s (f32 (aget s2 idx))
                     cs (f32 (aget c2s idx)) ss (f32 (aget s2s idx))]
                 [(f32 (+ (f32 (* 0.91 c)) (f32 (* 0.09 cs))))
                  (f32 (+ (f32 (* 0.91 s)) (f32 (* 0.09 ss))))]))
        [c00 s00] (mixv i0 j0) [c01 s01] (mixv i0 j1)
        [c10 s10] (mixv i1 j0) [c11 s11] (mixv i1 j1)
        cbl (lerp (lerp c00 c01 wy) (lerp c10 c11 wy) wx)
        sbl (lerp (lerp s00 s01 wy) (lerp s10 s11 wy) wx)
        th  (f32 (* 0.5 (Math/atan2 sbl cbl)))
        cob (bilerp1-32 co H W srch srcw x y)]
    [th (max 0.0 (min 1.0 cob))]))

(defn- gs-edge [dmap x y]
  (let [H (long (:h dmap)) W (long (:w dmap))
        srch (long (or (:src-h dmap) H)) srcw (long (or (:src-w dmap) W))]
    (if-let [e (:edge dmap)]
      (min 1.0 (bilerp1-32 e H W srch srcw x y))
      0.0)))

(defn- gs-snap
  "GS edgeSnap emulation (h=1.75, guard max(e)<0.12). Returns [x y d em ep]."
  [nf dmap x y gain]
  (let [[th _] (gs-field nf x y)
        nx (- (Math/sin th)) ny (Math/cos th)
        e0 (gs-edge dmap x y)
        ep (gs-edge dmap (+ x (* nx 1.75)) (+ y (* ny 1.75)))
        em (gs-edge dmap (- x (* nx 1.75)) (- y (* ny 1.75)))]
    (if (< (max e0 ep em) 0.12)
      [x y 0.0 0.0 0.0]
      (let [den (- (+ em ep) (* 2.0 e0))
            num (- em ep)
            d (if (< (Math/abs den) (* 0.5 (Math/abs num)))
                (if (>= ep em) 1.0 -1.0)
                (if (< (Math/abs den) 1e-9) 0.0
                    (max -1.0 (min 1.0 (/ num (* 2.0 den))))))]
        [(f32 (max 0.0 (min (double (dec (:h dmap))) (+ (double x) (* nx 1.75 d gain)))))
         (f32 (max 0.0 (min (double (dec (:w dmap))) (+ (double y) (* ny 1.75 d gain)))))
         d em ep num den]))))

(defn- cpu-snap
  "Real CPU edge-snap (float64). Returns [x y d em ep]."
  [dmap nf x y h hd wd gain]
  (let [[th _] (sample-fields nf x y)
        nx (- (Math/sin th)) ny (Math/cos th)
        e0 (edge-at dmap x y)
        ep (edge-at dmap (+ x (* nx h)) (+ y (* ny h)))
        em (edge-at dmap (- x (* nx h)) (- y (* ny h)))]
    (if (< (max e0 ep em) 0.12)
      [x y 0.0 0.0 0.0]
      (let [den (- (+ em ep) (* 2.0 e0))
            num (- em ep)
            d (if (< (Math/abs den) (* 0.5 (Math/abs num)))
                (if (>= ep em) 1.0 -1.0)
                (if (< (Math/abs den) 1e-9) 0.0
                    (max -1.0 (min 1.0 (/ num (* 2.0 den))))))]
        [(max 0.0 (min (double hd) (+ x (* nx h d gain))))
         (max 0.0 (min (double wd) (+ y (* ny h d gain))))
         d em ep num den]))))

(defn- sample-rgb32
  "GS sampleRGB emulation: float32 bilinear of a flat H*W*3 array at grid x,y
   (x=row, y=col), clamped — mirror of the CPU sample-arr."
  [arr width height x y]
  (let [W (long width) H (long height)
        gx (f32 (max 0.0 (min (double (dec H)) (double x))))
        gy (f32 (max 0.0 (min (double (dec W)) (double y))))
        x0 (int (Math/floor gx)) y0 (int (Math/floor gy))
        x1 (min (long (dec H)) (inc x0)) y1 (min (long (dec W)) (inc y0))
        fx (- gx (double x0)) fy (- gy (double y0))
        tex (fn [xi yi] (let [b (* 3 (+ (* xi W) yi))]
                          [(f32 (aget arr b)) (f32 (aget arr (+ b 1))) (f32 (aget arr (+ b 2)))]))
        [r00 g00 b00] (tex x0 y0) [r10 g10 b10] (tex x1 y0)
        [r01 g01 b01] (tex x0 y1) [r11 g11 b11] (tex x1 y1)
        lx (fn [a b t] (f32 (+ a (f32 (* t (f32 (- b a)))))))]
    [(lx (lx r00 r10 fx) (lx r01 r11 fx) fy)
     (lx (lx g00 g10 fx) (lx g01 g11 fx) fy)
     (lx (lx b00 b10 fx) (lx b01 b11 fx) fy)]))

(defn- replay-band
  "Replay the band tier's stroke-segments in float64 (CPU) and float32 (GS
   emulation). Returns a map of per-step record vectors."
  [img args]
  (let [nf (:noise-fields img) dmap (:detail img)
        px (:pixels img) blur-px (or (:blur img) px) blurd-px (or (:blurd img) px)
        H (long (:height img)) W (long (:width img))
        seed-pos (get args :seed) seed-x (double (nth seed-pos 0)) seed-y (double (nth seed-pos 1))
        ssz (double (get args :ssz 1.0)) ds (double (get args :ds 1.0))
        stepf (double (get args :stepf 0.8)) segs (long (get args :segs 32))
        bph (double (get args :bph 0.5)) sideo (double (get args :sideo 0.0))
        selong (double (get args :selong 0.0))
        lth (double (get args :lth 0.15)) gainv (double (get args :gainv 1.0))
        nf64 (seed/with-swirl nf 0.91)
        run
        (fn [f32?]
          (let [gx (if f32? (f32 seed-x) seed-x) gy (if f32? (f32 seed-y) seed-y)
                th0 (if f32?
                      (let [[t _] (gs-field nf gx gy)] t)
                      (let [[t _] (sample-fields nf64 gx gy)] t))
                nx0 (- (Math/sin th0)) ny0 (Math/cos th0)
                rung (fn [hh]
                       (let [p+ (if f32? (sample-rgb32 blur-px W H (+ gx (* nx0 hh)) (+ gy (* ny0 hh)))
                                    (sample-arr blur-px W H (+ gx (* nx0 hh)) (+ gy (* ny0 hh))))
                             p- (if f32? (sample-rgb32 blur-px W H (- gx (* nx0 hh)) (- gy (* ny0 hh)))
                                    (sample-arr blur-px W H (- gx (* nx0 hh)) (- gy (* ny0 hh))))]
                         (max (Math/abs (- (nth p+ 0) (nth p- 0)))
                              (Math/abs (- (nth p+ 1) (nth p- 1)))
                              (Math/abs (- (nth p+ 2) (nth p- 2))))))
                h1 (max 1.75 (* 0.8 ssz)) h2 (max 3.0 (* 1.5 ssz)) h3 (max 5.0 (* 2.5 ssz))
                d1 (rung h1) d2 (rung h2) d3 (rung h3)
                dmax (max d1 d2 d3)
                crisp? (>= d1 (* 0.75 dmax))
                soft-ramp? (and (not crisp?) (>= dmax 0.15))
                snapped (if f32?
                          (let [s (gs-snap nf dmap gx gy 0.65)] [(nth s 0) (nth s 1)])
                          (let [s (cpu-snap dmap nf64 gx gy 1.75 (dec H) (dec W) 0.65)] [(nth s 0) (nth s 1)]))
                sx (nth snapped 0) sy (nth snapped 1)
                [thS _] (if f32? (gs-field nf sx sy) (sample-fields nf64 sx sy))
                snx (- (Math/sin thS)) sny (Math/cos thS)
                d0 (+ (* (- gx sx) snx) (* (- gy sy) sny))
                side (cond (> d0 1e-9) 1.0 (< d0 -1e-9) -1.0 :else (double ds))
                soff (if (pos? selong) (* sideo (+ 0.6 (* 2.55 bph bph))) sideo)
                trace
                (loop [k 0 px sx py sy dxp 0.0 dyp 0.0 fade 1.0 out []]
                  (let [[th coh] (if f32? (gs-field nf px py) (sample-fields nf64 px py))
                        dx0 (if f32? (f32 (Math/cos th)) (Math/cos th))
                        dy0 (if f32? (f32 (Math/sin th)) (Math/sin th))
                        ev  (if f32? (gs-edge dmap px py) (edge-at dmap px py))
                        prev-len (Math/sqrt (+ (* dxp dxp) (* dyp dyp)))
                        pxp (if (> prev-len 1e-6) (/ dxp prev-len) dxp)
                        pyp (if (> prev-len 1e-6) (/ dyp prev-len) dyp)
                        dotn (+ (* dx0 pxp) (* dy0 pyp))
                        ridge? (and (pos? k) (< ev 0.10))
                        corner? (and (pos? k) (< (Math/abs dotn) 0.90))
                        sgn (cond (zero? k) ds (< dotn 0.0) -1.0 :else 1.0)
                        dx (if f32? (f32 (* sgn dx0)) (* sgn dx0))
                        dy (if f32? (f32 (* sgn dy0)) (* sgn dy0))
                        [dx dy] (if (pos? k)
                                  (let [mx (if f32? (f32 (+ (f32 (* 0.35 dx)) (f32 (* 0.65 dxp))))
                                             (+ (* 0.35 dx) (* 0.65 dxp)))
                                        my (if f32? (f32 (+ (f32 (* 0.35 dy)) (f32 (* 0.65 dyp))))
                                             (+ (* 0.35 dy) (* 0.65 dyp)))
                                        ml (Math/sqrt (+ (* mx mx) (* my my)))]
                                    (if (> ml 1e-6) [(/ mx ml) (/ my ml)] [dx dy]))
                                  [dx dy])
                        L (* ssz stepf)
                        nxr (if f32? (f32 (max 0.0 (min (double (dec H)) (+ px (f32 (* L dx))))))
                                   (max 0.0 (min (double (dec H)) (+ px (* L dx)))))
                        nyr (if f32? (f32 (max 0.0 (min (double (dec W)) (+ py (f32 (* L dy))))))
                                   (max 0.0 (min (double (dec W)) (+ py (* L dy)))))
                        [nxs nys sd sem sep snum sden] (if f32?
                                                             (gs-snap nf dmap nxr nyr 0.85)
                                                             (cpu-snap dmap nf64 nxr nyr 1.75 (dec H) (dec W) 0.85))
                        rec {:k k :px px :py py :th th :coh coh :ev ev :dot dotn
                             :sgn sgn :raw [nxr nyr] :snap [nxs nys] :sd sd
                             :sem sem :sep sep :sn snum :sdn sden :side side
                             :soft-ramp? soft-ramp? :dmax dmax :d1 d1 :d3 d3}]
                    (if (or ridge? corner? (>= k (dec segs)))
                      (conj out rec)
                      (recur (inc k) nxs nys dx dy fade (conj out rec)))))
                ]
            trace))]
    {:cpu (run false) :gs (run true)}))

(defn -main
  [& [path maxside]]
  (let [args {:seed [641.30810546875 587.1103515625] :ssz 1.150211725889332
              :ds 1.0 :stepf 0.8 :segs 32 :bph 0.7201259944122285
              :sideo 0.0 :selong 2.6 :lth 0.15 :gainv 1.0}
        img0 (image/load-image (or path "img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg")
                               (if (and maxside (not= maxside "0")) (long (Double/parseDouble maxside)) nil))
        img  (fields/prepare img0)
        {:keys [cpu gs]} (replay-band img args)]
    (println (format "seed (%.2f, %.2f) ssz %.3f ds %.1f"
                     (double (nth (:seed args) 0)) (double (nth (:seed args) 1))
                     (double (:ssz args)) (double (:ds args))))
    (doseq [i (range (min (count cpu) (count gs)))]
      (let [c (nth cpu i) g (nth gs i)]
        (println (format "k=%-2d CPU snap d %.4f em %.5f ep %.5f num %.5f den %.6f | GS snap d %.4f em %.5f ep %.5f"
                         (:k c) (:sd c) (:sem c) (:sep c) (:sn c) (:sdn c) (:sd g) (:sem g) (:sep g))))))
)
