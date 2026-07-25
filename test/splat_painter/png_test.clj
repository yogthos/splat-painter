(ns splat-painter.png-test
  (:require [clojure.test :refer [deftest is]]
            [jolt.ffi :as ffi]
            [splat-painter.png :as png]))

(deftest save-rgba-bottom-up-does-not-fault-on-success
  ;; Defect under test: save-rgba-bottom-up! used to read the GError slot
  ;; UNCONDITIONALLY after pixbuf-savev, and read-gerror dereferenced the
  ;; NULL slot at offset 16 — a native "invalid memory reference" fault on
  ;; every SUCCESSFUL save. Here we drive the success path with a tiny valid
  ;; RGBA buffer and assert the call returns normally (or, in an environment
  ;; without a PNG encoder, throws a proper ex-info save-failure — NOT a fault).
  (let [iw 4 ih 4 n (* iw ih 4)
        buf  (ffi/alloc n)
        path (str "/tmp/splat-painter-save-test-"
                  (System/currentTimeMillis) ".png")]
    (dotimes [i n]
      (ffi/write buf :uint8 i 128))
    (try
      (png/save-rgba-bottom-up! buf iw ih path)
      (is (.exists (java.io.File. path))
          "a successful save writes the PNG file")
      (catch clojure.lang.ExceptionInfo e
        ;; Legitimate save failure (e.g. this gdk-pixbuf build has no PNG
        ;; encoder). That is acceptable — the bug was a native FAULT on the
        ;; success path, not a real ex-info save failure.
        (is (re-find #"save failed" (ex-message e))))
      (finally
        (ffi/free buf)
        (let [f (java.io.File. path)] (when (.exists f) (.delete f)))))))
