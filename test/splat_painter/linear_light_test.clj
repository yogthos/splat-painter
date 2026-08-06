(ns splat-painter.linear-light-test
  "Pins the GA_PAINTER_LINEAR_LIGHT colour decode in the generation shader.

   The flag makes the composite target sRGB (encode-on-write); the emit side must
   then hand it LINEAR colour or the write encode double-encodes the gamma values
   already in the field textures (measured mean|d| 8.5 -> 55.5 on loki). The decode
   is a #define-injected block that touches ONLY the emitted colour — the decision
   probes (THIN_GAIN, drift) sample the same textures for luma signals and their
   thresholds were measured on gamma-domain values, so they must keep seeing raw
   samples. These tests pin: the decode is gated and late, the injected define
   lands after #version (GLSL rejects a define before it — the compile failure the
   injection first hit), and the flag-on shader compiles+links on the real driver."
  (:require [clojure.test :refer [deftest is]]
            [jolt.ffi :as ffi]
            [glimmer-gl.offscreen :as off]
            [splat-painter.gen :as gen]))

(ffi/defcfn c-setenv   "setenv"   [:string :string :int] :int)
(ffi/defcfn c-unsetenv "unsetenv" [:string] :int)

(defn- with-env [name value f]
  (c-setenv name value 1)
  (try (f) (finally (c-unsetenv name))))

(deftest decode-is-gated-and-late
  (let [gs (:gs-src (gen/sources))]
    (is (re-find #"(?s)#ifdef GA_PAINTER_LINEAR_LIGHT\nvec3 srgbToLinear\(vec3 c\)\{.*?\n\}\n#endif"
                 gs)
        "srgbToLinear lives inside its own #ifdef, so flag-off compiles without it")
    (is (re-find #"(?s)vec3 color = clamp\(.*?\n#ifdef GA_PAINTER_LINEAR_LIGHT\n  color = srgbToLinear\(color\);.*?\n#endif"
                 gs)
        "the decode runs on the final emitted colour, after every tonal decision")
    (is (= 2 (count (re-seq #"#ifdef GA_PAINTER_LINEAR_LIGHT" gs)))
        "exactly two gated regions: the function and the call site — nothing else
         (wrapping sampleRGB would make three and re-tune the decision probes)")
    (is (= 2 (count (re-seq #"#endif" gs))) "#endif count matches, no stray pair")))

(deftest define-lands-after-version
  (let [on  (#'gen/flag-gs-src true)
        off (#'gen/flag-gs-src false)]
    (is (re-find #"(?s)#version 330 core\n#define GA_PAINTER_LINEAR_LIGHT\nlayout\(points\) in;"
                 on)
        "the define sits between #version and the first statement — GLSL rejects
         any statement before #version")
    (is (= (:gs-src (gen/sources)) off)
        "flag off compiles the base source unchanged (byte-identical by construction)")))

(deftest flag-on-shader-compiles-on-the-real-driver
  ;; Offscreen, mirroring band-ppc-test's rig: tests create real GL 4.1 contexts.
  ;; This is the regression pin for the define-before-#version compile failure.
  (let [ctx (off/ensure-current!)]
    (if (:error ctx)
      (println "linear-light-test: SKIPPED, no offscreen GL —" (:error ctx))
      (do (with-env "GA_PAINTER_LINEAR_LIGHT" "1"
            #(is (some? (gen/build-gen-program))
                 "flag-on gen program compiles + links"))
          (is (some? (gen/build-gen-program))
              "flag-off gen program compiles + links")))))
