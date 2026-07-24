(ns splat-painter.layer-test
  (:require [clojure.test :refer [deftest is testing]]
            [splat-painter.core :as core :refer [rgba->image]]
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

;; --- layer stack: the pure vector + settings helpers --------------------------
;; The GL capture/commit fns can't run headless, but the index bookkeeping they
;; share (insert-at-active, remove-at-j, snapshot/restore of every slider) is
;; exercised here against dummy entries and the real control atoms.

(deftest settings-snapshot-round-trip
  (testing "snapshot then restore reproduces every control atom exactly"
    (let [saved (core/snapshot-settings)]
      (is (= 14 (count saved)))                  ; count size broad mid fine detail
      (core/restore-settings! (vec (repeat 14 0.37))) ; variation curvature stroke contrast
      (is (every? #(== 0.37 %) (core/snapshot-settings))) ; hardness tex-streak/grain/edge
      (core/restore-settings! saved)
      (is (= saved (core/snapshot-settings))))))

(deftest insert-layer-appends-at-top-inserts-in-middle
  (let [e (fn [x] {:tex x})]
    (is (= [(e :a)]             (core/insert-layer [] 0 (e :a))))          ; append on empty
    (is (= [(e :a) (e :b)]      (core/insert-layer [(e :a)] 1 (e :b))))    ; append at top
    (is (= [(e :a) (e :x) (e :b)]
          (core/insert-layer [(e :a) (e :b)] 1 (e :x))))))                 ; insert middle

(deftest remove-layer-drops-index-j
  (let [e (fn [x] {:tex x})]
    (is (= [(e :b)]               (core/remove-layer [(e :a) (e :b)] 0)))
    (is (= [(e :a)]               (core/remove-layer [(e :a) (e :b)] 1)))
    (is (= [(e :a) (e :c)]        (core/remove-layer [(e :a) (e :b) (e :c)] 1)))))

(deftest layer-stack-navigation
  (testing "add commits under the live top; select swaps live with layers[j]"
    ;; index convention (documented once in core/commit-active!): with N committed
    ;; layers and active in [0,N], the live pass sits between layers[active-1] and
    ;; layers[active]; below = layers[0..active), above = layers[active..N).
    (let [insert   core/insert-layer
          remove   core/remove-layer
          commit   (fn [id] {:tex id :opacity 0.6 :settings [id]})
          v (atom []) a (atom 0)
          tex      (fn [coll] (vec (map :tex coll)))]
      ;; three add-layer! operations: commit the live pass at `active`, then push
      ;; active to the new top (count).
      (swap! v insert @a (commit :c0)) (reset! a (count @v))   ; v=[c0] a=1
      (swap! v insert @a (commit :c1)) (reset! a (count @v))   ; v=[c0,c1] a=2
      (swap! v insert @a (commit :c2)) (reset! a (count @v))   ; v=[c0,c1,c2] a=3
      (is (= [:c0 :c1 :c2] (tex @v)))
      (is (= 3 @a))
      ;; select-layer! 1: commit at active(==count)=append, then drop index 1.
      (swap! v insert @a (commit :c3))                          ; v=[c0,c1,c2,c3]
      (swap! v remove 1)                                        ; v=[c0,c2,c3]
      (reset! a 1)
      (is (= [:c0 :c2 :c3] (tex @v)))                           ; below=[c0] above=[c2,c3]
      (is (= [:c0] (tex (subvec @v 0 @a))))
      (is (= [:c2 :c3] (tex (subvec @v @a))))
      ;; select-layer! 0: commit at active(1)=INSERT MIDDLE, then drop index 0.
      (swap! v insert @a (commit :c4))                          ; insert@1 -> [c0,c4,c2,c3]
      (is (= [:c0 :c4 :c2 :c3] (tex @v)))
      (swap! v remove 0)                                        ; [c4,c2,c3]
      (reset! a 0)
      (is (= [:c4 :c2 :c3] (tex @v)))                           ; below=[] above=[c4,c2,c3]
      (is (= 0 @a)))))
