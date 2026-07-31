(ns splat-painter.sharpen-test
  "Pins the sharpen present-pass SHADER (splat-painter.shader/build-program-sharpen):
   the edge gate, the letterbox clamp, and the amount-0 identity. Offscreen GL in
   the gpu-fields-test style — skips cleanly when there is no context. The offline
   gate-threshold measurement these constants come from lives in
   dev/harness/sharpen/ (measure_gate.py, simulate_pass.py).

   Everything here is bottom-up GL order (row 0 = framebuffer y = texture v = 0),
   matching how the app renders the composite texture the pass samples."
  (:require [clojure.test :refer [deftest is testing]]
            [glimmer-gl.gl :as gl]
            [glimmer-gl.offscreen :as off]
            [splat-painter.shader :as shader]
            [jolt.ffi :as ffi]))

(def ^:private GL-RGBA8 0x8058)

;; A test that is allowed to skip has to say which way it went, or a broken
;; context reads exactly like a passing suite. Announce the decision once.
(defonce ^:private announced (atom false))

(defn- announce! [ctx]
  (when (compare-and-set! announced false true)
    (if-let [err (:error ctx)]
      (println "sharpen-test: SKIPPED, no offscreen GL —" err)
      (let [[maj mnr] (:version ctx)]
        (println (format "sharpen-test: GL %d.%d on %s"
                         (long maj) (long mnr)
                         (or (gl/gl-get-string* gl/GL-RENDERER) "?")))))))

(defn- with-gl
  "Run `f` on an offscreen context. Returns :skipped when there is no context, so
   a test asserts nothing rather than failing on a machine with no display."
  [f]
  (let [ctx (off/ensure-current!)]
    (announce! ctx)
    (if (:error ctx) :skipped (f))))

;; Deterministic byte patterns — a tiny LCG, no RNG dependency.
(defn- lcg-bytes [n seed]
  (mapv (fn [i] (mod (unchecked-add (unchecked-multiply 1103515245 (+ seed i)) 12345)
                     256))
        (range n)))

(defn- upload-grey-texture
  "W*H grey bytes (bottom-up) as an RGBA8 texture with NEAREST filtering — the
   same format/filtering as the app's composite target."
  [W H ^bytes grey-bytes]
  (let [tex (gl/gen-one gl/gl-gen-textures)
        buf (ffi/alloc (* W H 4))]
    (dotimes [i (* W H)]
      (let [b (bit-and (aget grey-bytes i) 0xff)]
        (ffi/write buf :uint8 (* 4 i)       b)
        (ffi/write buf :uint8 (+ (* 4 i) 1) b)
        (ffi/write buf :uint8 (+ (* 4 i) 2) b)
        (ffi/write buf :uint8 (+ (* 4 i) 3) 255)))
    (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
    (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 GL-RGBA8 (int W) (int H) 0
                        gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-NEAREST)
    (ffi/free buf)
    tex))

(defn- make-target!
  "An FBO + RGBA8 texture to draw the pass into, W*H; leaves the FBO bound."
  [W H]
  (let [fbo (gl/gen-one gl/gl-gen-framebuffers)
        tex (gl/gen-one gl/gl-gen-textures)]
    (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
    (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 GL-RGBA8 (int W) (int H) 0
                        gl/GL-RGBA gl/GL-UNSIGNED-BYTE ffi/null)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-NEAREST)
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
    (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0
                                  gl/GL-TEXTURE-2D tex 0)
    {:fbo fbo :tex tex}))

(defn- run-pass!
  "Run the sharpen pass exactly as gpu-draw!'s :present branch does: clear the
   bound target to opaque black, draw the 6-vertex attribute-less rect sampling
   `src-tex`, with `rect` = [ox oy dw dh] in framebuffer px. Returns the readback
   as a byte-array of W*H grey values (R channel)."
  [prog vao src-tex W H rect amount]
  (let [{:keys [locs]} prog]
    (gl/gl-viewport 0 0 (int W) (int H))
    (gl/gl-clear-color 0.0 0.0 0.0 1.0)
    (gl/gl-clear gl/GL-COLOR-BUFFER-BIT)
    (gl/gl-disable gl/GL-BLEND)
    (gl/gl-use-program (:program prog))
    (gl/gl-active-texture gl/GL-TEXTURE0)
    (gl/gl-bind-texture gl/GL-TEXTURE-2D src-tex)
    (gl/gl-uniform-1i (:u_src locs) 0)
    (gl/gl-uniform-2f (:u_viewport locs) (double W) (double H))
    (gl/gl-uniform-4f (:u_rect locs) (double (rect 0)) (double (rect 1))
                      (double (rect 2)) (double (rect 3)))
    (gl/gl-uniform-1f (:u_amount locs) (double amount))
    (gl/gl-bind-vertex-array vao)
    (gl/gl-draw-arrays gl/GL-TRIANGLES 0 6)
    (let [buf (ffi/alloc (* W H 4))
          out (byte-array (* W H))]
      (gl/gl-read-pixels 0 0 (int W) (int H) gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
      (dotimes [i (* W H)]
        (aset out i (unchecked-byte (ffi/read buf :uint8 (* 4 i)))))
      (ffi/free buf)
      out)))

(defn- ubyte [^bytes bs i] (bit-and (aget bs i) 0xff))

(defn- variance [^bytes bs]
  (let [n (alength bs)
        m (/ (areduce bs i s 0.0 (+ s (ubyte bs i))) n)]
    (/ (areduce bs i s 0.0 (let [d (- (ubyte bs i) m)] (+ s (* d d)))) n)))

(deftest amount-zero-is-identity
  (testing "the pass at amount 0 reproduces the source texture byte-for-byte"
    (with-gl
      (fn []
        (let [W 64 H 64
              src (byte-array (map unchecked-byte (lcg-bytes (* W H) 7)))
              tex (upload-grey-texture W H src)
              _   (make-target! W H)          ; binds its FBO
              vao (gl/gen-one gl/gl-gen-vertex-arrays)
              prog (shader/build-program-sharpen)]
          (is (some? prog) "sharpen program links")
          (when prog
            (let [got (run-pass! prog vao tex W H [0 0 W H] 0.0)
                  bad (reduce (fn [n i] (if (= (ubyte src i) (ubyte got i)) n (inc n)))
                              0 (range (* W H)))]
              (is (zero? bad) (str bad " pixels changed at amount 0")))))))))

(deftest step-edge-gets-steeper
  (testing "a soft step edge has a larger maximum across-edge jump after the pass"
    (with-gl
      (fn []
        (let [W 64 H 64
              ;; vertical smoothstep ramp 0.1 -> 0.9 over x in [28,31)
              src (byte-array (* W H))]
          (dotimes [y H]
            (dotimes [x W]
              (let [t (max 0.0 (min 1.0 (/ (- x 28) 3.0)))
                    s (+ 0.1 (* 0.8 (- (* 3 t t) (* 2 t t t))))]
                (aset src (+ (* y W) x)
                      (unchecked-byte (Math/round (double (* 255 s))))))))
          (let [tex  (upload-grey-texture W H src)
                _    (make-target! W H)
                vao  (gl/gen-one gl/gl-gen-vertex-arrays)
                prog (shader/build-program-sharpen)]
            (is (some? prog) "sharpen program links")
            (when prog
              (let [got  (run-pass! prog vao tex W H [0 0 W H] 1.0)
                    jump (fn [bs]
                           (reduce (fn [mx y]
                                     (reduce (fn [m x]
                                               (max m (Math/abs (- (ubyte bs (+ (* y W) x))
                                                                   (ubyte bs (+ (* y W) (dec x)))))))
                                             mx (range 1 W)))
                                   0 (range 1 (dec H))))
                    before (jump src)
                    after  (jump got)]
                (is (> after (* 1.15 before))
                    (str "max across-edge jump before=" before " after=" after))))))))))

(deftest gate-protects-flat-texture
  (testing "mid-grey + single-pixel grain (the canvas/bristle stand-in) barely moves"
    (with-gl
      (fn []
        (let [W 64 H 64
              ;; mid grey 128 + checkerboard +/-10 (pure single-pixel grain) + a
              ;; deterministic +/-2 white-noise component so the gate's residual
              ;; leakage is exercised too, not just an exact zero.
              noise (lcg-bytes (* W H) 99)
              src   (byte-array (* W H))]
          (dotimes [y H]
            (dotimes [x W]
              (let [i     (+ (* y W) x)
                    check (if (even? (+ x y)) 10 -10)
                    white (- (mod (nth noise i) 5) 2)]
                (aset src i (unchecked-byte (+ 128 check white))))))
          (let [tex  (upload-grey-texture W H src)
                _    (make-target! W H)
                vao  (gl/gen-one gl/gl-gen-vertex-arrays)
                prog (shader/build-program-sharpen)]
            (is (some? prog) "sharpen program links")
            (when prog
              (let [got   (run-pass! prog vao tex W H [0 0 W H] 1.5)
                    ratio (/ (variance got) (variance src))]
                ;; the gate ON keeps the grain's variance rise small; with the
                ;; gate forced to 1.0 the same run rises ~(1+amount)^2 = 6.25x —
                ;; that mutation must turn this test red.
                (is (< ratio 2.0)
                    (str "flat-texture variance ratio " ratio))))))))))

(deftest letterbox-clamp-holds
  (testing "a fragment one texel inside the image rect ignores the black bar outside"
    (with-gl
      (fn []
        (let [W 64 H 64
              ;; the composite as the app produces it: opaque black bars, grey 128
              ;; image inside the rect (8,8,48,48)
              src (byte-array (* W H))]
          (dotimes [y H]
            (dotimes [x W]
              (when (and (>= x 8) (< x 56) (>= y 8) (< y 56))
                (aset src (+ (* y W) x) (unchecked-byte 128)))))
          (let [tex  (upload-grey-texture W H src)
                _    (make-target! W H)
                vao  (gl/gen-one gl/gl-gen-vertex-arrays)
                prog (shader/build-program-sharpen)]
            (is (some? prog) "sharpen program links")
            (when prog
              (let [got (run-pass! prog vao tex W H [8 8 48 48] 1.5)
                    at  (fn [x y] (ubyte got (+ (* y W) x)))]
                ;; without the clamp the border rings: 0.5+1.5*(0.5-0.1875)=0.9688
                ;; (~247) at the corner, 0.6875 (~175) mid-edge.
                (is (<= (Math/abs (- (at 8 20) 128)) 1)
                    (str "left-edge pixel rang: " (at 8 20)))
                (is (<= (Math/abs (- (at 20 8) 128)) 1)
                    (str "bottom-edge pixel rang: " (at 20 8)))
                (is (<= (Math/abs (- (at 55 55) 128)) 1)
                    (str "top-right corner pixel rang: " (at 55 55)))
                (is (zero? (at 4 20))
                    "the bar itself stays black")))))))))
