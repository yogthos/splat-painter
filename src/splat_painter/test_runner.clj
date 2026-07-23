(ns splat-painter.test-runner
  "Entry point for `joltc -M:test`. Requires each splat-painter test namespace and
  runs clojure.test against it; exits non-zero if anything failed."
  (:require [clojure.test :as t]
            [splat-painter.gaussian-test]
            [splat-painter.seed-test]
            [splat-painter.image-test]
            [splat-painter.structure-test]
            [splat-painter.noise-test]
            [splat-painter.wavelet-test]))

(defmethod t/report :error [m]
  (t/with-test-out
    (t/inc-report-counter :error)
    (println "\nERROR in" (t/testing-vars-str m))
    (when (seq t/*testing-contexts*) (println (t/testing-contexts-str)))
    (when-let [e (:actual m)]
      (when (instance? Throwable e)
        (println "  ->" (.getName (class e)) ":" (ex-message e))
        (when-let [d (ex-data e)] (prn d))))))

(defn -main [& _]
  (let [results (t/run-tests 'splat-painter.gaussian-test
                             'splat-painter.seed-test
                             'splat-painter.image-test
                             'splat-painter.structure-test
                             'splat-painter.noise-test
                             'splat-painter.wavelet-test)
        failed (+ (:fail results 0) (:error results 0))]
    (println (format "\n%d tests, %d passed, %d failed"
                     (:test results 0) (:pass results 0) failed))
    (when (pos? failed) (System/exit 1))))
