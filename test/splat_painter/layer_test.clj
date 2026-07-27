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
      (is (= 16 (count saved)))                  ; count size broad mid fine detail
      (core/restore-settings! (vec (repeat 16 0.37))) ; variation curvature stroke contrast
                                                 ; hardness cutin swirl tex-streak/grain/edge
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

;; --- clear-layers! / on-image-loaded: the texture-leak + stack-reset paths ------
;; area-atom is nil in this headless runner, so clear-layers! skips the GL
;; make-current + delete path (it gates on @area-atom) and on-image-loaded's
;; request-render! no-ops. The atom bookkeeping is what these two exercise.

(deftest clear-layers-resets-stack-headlessly
  (testing "clear-layers! resets layers-atom + active to 0 without touching GL"
    (let [prev-layers @core/layers-atom
          prev-active @core/active-layer-atom]
      (reset! core/layers-atom [{:tex 1 :opacity 0.6 :settings [1]}
                                {:tex 2 :opacity 0.5 :settings [2]}
                                {:tex 3 :opacity 0.4 :settings [3]}])
      (reset! core/active-layer-atom 2)
      (#'splat-painter.core/clear-layers!)
      (is (empty? @core/layers-atom))
      (is (zero? @core/active-layer-atom))
      (reset! core/layers-atom prev-layers)
      (reset! core/active-layer-atom prev-active))))

(deftest loading-an-image-drops-the-previous-stack
  (testing "on-image-loaded clears a pre-existing committed layer stack (regression)"
    ;; prepare-image runs for real on the 512x512 fixture (~2s): structure tensor,
    ;; bilateral + box blurs, the wavelet map, the noise fields. Do not stub it.
    (let [prev-snap   (core/snapshot-settings)
          prev-layers @core/layers-atom
          prev-active @core/active-layer-atom
          prev-image  @core/image-atom
          prev-size   @core/size-atom
          prev-status @core/status-atom]
      (reset! core/layers-atom [{:tex 1 :opacity 0.6 :settings [1]}
                                {:tex 2 :opacity 0.5 :settings [2]}])
      (reset! core/active-layer-atom 2)
      (#'splat-painter.core/on-image-loaded "test/splat_painter/fixtures/eye.jpeg")
      (is (empty? @core/layers-atom))
      (is (zero? @core/active-layer-atom))
      (is (some? @core/image-atom))
      (core/restore-settings! prev-snap)
      (reset! core/layers-atom prev-layers)
      (reset! core/active-layer-atom prev-active)
      (reset! core/image-atom prev-image)
      (reset! core/size-atom prev-size)
      (reset! core/status-atom prev-status))))

(deftest failed-load-leaves-existing-layers-untouched
  (testing "a failed load (catch branch) must not clear the committed layer stack"
    ;; pins the constraint that clear-layers! runs only AFTER prepare-image succeeds:
    ;; a bad path throws inside load/prepare, takes the catch branch, and the existing
    ;; image + its layers survive (only status-atom is rewritten).
    (let [prev-snap   (core/snapshot-settings)
          prev-layers @core/layers-atom
          prev-active @core/active-layer-atom
          prev-image  @core/image-atom
          prev-status @core/status-atom
          seeded      [{:tex 1 :opacity 0.6 :settings [1]}
                       {:tex 2 :opacity 0.5 :settings [2]}]]
      (reset! core/layers-atom seeded)
      (reset! core/active-layer-atom 1)
      (#'splat-painter.core/on-image-loaded "test/splat_painter/fixtures/does-not-exist.jpeg")
      (is (= seeded @core/layers-atom))                         ; stack survives a failed load
      (is (= 1 @core/active-layer-atom))                        ; active index untouched
      (is (some? (re-find #"failed to load" @core/status-atom)))
      (core/restore-settings! prev-snap)
      (reset! core/layers-atom prev-layers)
      (reset! core/active-layer-atom prev-active)
      (reset! core/image-atom prev-image)
      (reset! core/status-atom prev-status))))

(deftest field-tex-ids-excludes-perm-and-non-textures
  (testing "field-tex-ids returns exactly the per-image texture ids, never :perm or :dmax"
    ;; mirrors gen/upload-fields!: 8 per-image textures allocated via
    ;; new-tex, plus :perm (the shared Perlin texture, uploaded ONCE by upload-perm!
    ;; and reused across images), :dmax (a float), :dmap (a map), dim vectors and
    ;; :H/:W. Freeing :perm corrupts every later generate!; freeing (long :dmax) would
    ;; delete an unrelated texture id. Both must be excluded by an explicit allow-list.
    (let [fields {:detail 22 :subject 23 :noise 24 :noise-swirl0 29 :blur 25 :blur-drift 26
                  :blur-heavy 27 :raw 28 :perm 3 :dmax 1.0 :dmap {:h 64 :w 64}
                  :detail-dim [64.0 64.0] :detail-src [1024.0 1024.0]
                  :noise-dim [32.0 32.0] :noise-src [1024.0 1024.0] :H 1024 :W 1024}
          ids (core/field-tex-ids fields)]
      (is (= [22 23 24 29 25 26 27 28] ids))   ; every per-image texture key, in order
      (is (not-any? #(= 3 %) ids))            ; :perm shared texture never freed
      (is (not-any? #(= 1 %) ids)))))         ; :dmax (1.0) never freed
