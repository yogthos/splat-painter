(ns splat-painter.layer-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.core :refer [rgba->image]]
            [jolt.ffi :as ffi]))

;; rgba->image consumes a raw RGBA byte buffer exactly as glReadPixels returns it
;; (origin bottom-left, iw*ih*4 bytes). These helpers write/read that buffer via the
;; same ffi byte API the app uses, so the test exercises the real signedness path
;; (a byte of 200 must decode to 200/255, not wrap to a negative).

(defn- put-rgba! [buf iw r c rr gg bb]
  ;; one RGBA byte at buffer row r, col c (row-major, bottom-up glReadPixels layout)
  (let [o (+ (* 4 (* r iw)) (* 4 c))]
    (ffi/write buf :uint8 o            rr)
    (ffi/write buf :uint8 (+ o 1)      gg)
    (ffi/write buf :uint8 (+ o 2)      bb)
    (ffi/write buf :uint8 (+ o 3)      255)))

(defn- px [img x y k]
  ;; image pixel channel k (0=r 1=g 2=b) at row x, col y — flat base 3*(x*W+y)
  (aget ^doubles (:pixels img) (+ (* 3 (+ (* x (:width img)) y)) k)))

(deftest rgba->image-flips-rows-keeps-channels-and-sign
  (testing "a 2x2 bottom-up RGBA buffer becomes a top-down H*W*3 double image"
    (let [iw 2 ih 2
          buf (ffi/alloc (* iw ih 4))]
      ;; buffer row 0 = BOTTOM of the capture = image row (ih-1) = row 1
      (put-rgba! buf iw 0 0 200   0   0)   ; bottom-left  -> image row1 col0 = red(200)
      (put-rgba! buf iw 0 1   0 200   0)   ; bottom-right -> image row1 col1 = green(200)
      ;; buffer row 1 = TOP of the capture = image row 0
      (put-rgba! buf iw 1 0   0   0 200)   ; top-left     -> image row0 col0 = blue(200)
      (put-rgba! buf iw 1 1 200 200 200)   ; top-right    -> image row0 col1 = grey(200)
      (let [img (rgba->image buf iw ih)]
        (is (= ih  (:height img)))
        (is (= iw  (:width img)))
        (is (= 3   (:channels img)))
        (is (= (* iw ih 3) (count (:pixels img))))
        ;; row flip: image row 0 came from buffer row 1 (blue, the TOP of capture)
        (is (and (zero? (px img 0 0 0)) (zero? (px img 0 0 1)) (> (px img 0 0 2) 0.7)))
        ;; image row 1 came from buffer row 0 (red, the BOTTOM of capture)
        (is (and (> (px img 1 0 0) 0.7) (zero? (px img 1 0 1)) (zero? (px img 1 0 2))))
        ;; signedness: a byte of 200 decodes to 200/255, NOT a wrapped negative
        (is (< (Math/abs (- (px img 0 1 0) (/ 200.0 255.0))) 1e-9))
        ;; channel order RGB + layout base 3*(x*W+y): the grey pixel's green is at
        ;; index 3*(0*2+1)+1 = 4, and it reads 200/255
        (is (< (Math/abs (- (px img 0 1 1) (/ 200.0 255.0))) 1e-9)))
      (ffi/free buf))))
