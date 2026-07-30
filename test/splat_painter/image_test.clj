(ns splat-painter.image-test
  (:require [clojure.test :refer [deftest is testing]]
            [jolt.ffi :as ffi]
            [splat-painter.image :as image]
            [splat-painter.seed :as seed]
            [splat-painter.gaussian :as g]))

(def ^:private fixture "test/splat_painter/fixtures/eye.jpeg")

(deftest load-decodes-eye-jpeg
  (let [img (image/load-image fixture 128)]
    (is (= 128 (:height img)))
    (is (= 128 (:width img)))
    (is (= 3 (:channels img)))
    (is (= (* 128 128 3) (count (:pixels img))))
    (is (every? #(<= 0.0 % 1.0) (:pixels img)))
    ;; the eye photo is not blank — there is real tonal spread
    (let [ps (:pixels img)]
      (is (< (apply min ps) 0.5))
      (is (> (apply max ps) 0.5)))))

(deftest no-scale-keeps-original-size
  (let [img (image/load-image fixture)]            ; max-side nil -> original
    (is (= 512 (:height img)))
    (is (= 512 (:width img)))))

(deftest seed-and-rasterize-a-real-image
  (testing "loading -> seeding -> CPU rasterize produces a non-trivial image"
    (let [img  (image/load-image fixture 64)
          fld  (seed/splat-field img {:count 256 :scale 3.0 :background 0.0})
          out  (g/rasterize (:splats fld) (repeat (* 64 64 3) 0.0) 64 64)]
      (is (pos? (count out)))
      (is (some pos? out))                       ; not all black
      (is (< (apply min out) (apply max out)))))); has contrast

(deftest read-gerror-returns-nil-on-null-slot
  ;; A zeroed GError** slot (NULL GError*) must read back as nil, not
  ;; dereference address 0+offset into a native fault. The message pointer sits
  ;; at byte offset 8, after the two 4-byte fields — reading 16 fetched one
  ;; pointer past the struct and faulted.
  (let [slot (ffi/alloc (ffi/sizeof :pointer))]
    (ffi/write slot :pointer 0 ffi/null)
    (is (nil? (#'splat-painter.image/read-gerror slot)))
    (ffi/free slot)))

(deftest load-image-error-carries-the-real-glib-message
  ;; A non-image file: the gdk-pixbuf loader rejects it and sets a GError with
  ;; a real message. Tests run from the project root, so deps.edn is present —
  ;; no fixture needed. Do not assert exact GLib wording (locale/version drift):
  ;; assert the message is present and is NOT the "unknown error" fallback that
  ;; read-gerror produced when it read the message from the wrong struct offset.
  (let [msg (try (image/load-image "deps.edn")
                 (catch clojure.lang.ExceptionInfo e (ex-message e)))]
    (is msg "load-image should throw an ex-info with a message")
    (is (not (re-find #"unknown error" msg))
        "must not fall back to 'unknown error' — that means the GLib message was lost")
    ;; anchor on the path, not on any ": " — the "image:" prefix would satisfy
    ;; a bare #": .+" even if GLib's portion came back empty
    (is (re-find #"deps\.edn: \S" msg)
        "carries a non-empty GLib portion after the path")))

