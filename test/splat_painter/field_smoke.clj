(ns splat-painter.field-smoke
  "Dev check for the GPU field passes: run one on the live context and compare it
   numerically against the CPU function it is meant to replace. Needs a current GL
   context, so it runs inside the app — GA_PAINTER_FIELD_SMOKE=1 on the first render.

   Readback here is deliberate and dev-only (~8.6M floats/s); the shipping path
   never pulls these back."
  (:require [glimmer-gl.gl :as gl]
            [splat-painter.gpu-fields :as gf]
            [splat-painter.structure :as structure]))

(defn- stats
  "max |a-b| and mean |a-b| between two same-length ^doubles."
  [^doubles a ^doubles b]
  (let [n (min (alength a) (alength b))]
    (loop [i 0 mx 0.0 acc 0.0]
      (if (>= i n)
        [mx (/ acc (max 1 n))]
        (let [d (Math/abs (- (aget a i) (aget b i)))]
          (recur (inc i) (max mx d) (+ acc d)))))))

(defn- chan-of [img off]
  (let [H (long (:height img)) W (long (:width img))
        ^doubles px (:pixels img)
        n (* H W)
        a (double-array n)]
    (dotimes [i n] (aset a i (aget px (+ (* 3 i) off))))
    a))

(defn run!
  "Compare the GPU separable box blur against structure/box-blur on the red channel."
  [img]
  (let [H (long (:height img)) W (long (:width img))
        ctx   (gf/make-ctx)
        progs (gf/build-programs)]
    (if-not progs
      (println "field-smoke: shader failed to compile (see GL info log)")
      (let [src (gf/upload-rgb! img)
            dst (gf/new-scratch W H)
            tmp (gf/new-scratch W H)]
        (doseq [r [2 3 8]]
          (let [t0  (System/nanoTime)
                _   (gf/box-blur! ctx progs src dst tmp W H r)
                _   (gl/gl-finish)
                gms (/ (- (System/nanoTime) t0) 1e6)
                got (gf/read-channel ctx dst W H 0)
                t1  (System/nanoTime)
                want (structure/box-blur (chan-of img 0) H W r)
                cms (/ (- (System/nanoTime) t1) 1e6)
                [mx mean] (stats got want)]
            (println (format "field-smoke box r=%-2d  GPU %6.1f ms  CPU %6.1f ms  |  max diff %.9f  mean %.9f"
                             r gms cms mx mean))))
        (println "field-smoke: done")))))
