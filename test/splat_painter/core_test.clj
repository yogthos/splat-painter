(ns splat-painter.core-test
  "The headless override contract: a scripted render (GA_PAINTER_SAVE_PNG) has no UI
   to drag, so every live slider must be forceable from the environment or it cannot
   be varied from a script. Curvature had no override, which is how the gap surfaced."
  (:require [clojure.test :refer [deftest is testing]]
            [jolt.ffi :as ffi]
            [splat-painter.core :as core]))

(ffi/defcfn c-setenv   "setenv"   [:string :string :int] :int)
(ffi/defcfn c-unsetenv "unsetenv" [:string] :int)

(defn- with-env [name value f]
  (c-setenv name value 1)
  (try (f) (finally (c-unsetenv name))))

;; [env var, accessor, backing atom, env string, expected value]. One row per live
;; slider in the control panel; the accessors are private, hence the var lookups.
(def ^:private overrides
  [["GA_PAINTER_COUNT"      #'core/cur-count      #'core/count-atom      "4321"  4321]
   ["GA_PAINTER_SIZE"       #'core/cur-size       #'core/size-atom       "7.5"   7.5]
   ["GA_PAINTER_DETAIL"     #'core/cur-detail     #'core/detail-atom     "0.25"  0.25]
   ["GA_PAINTER_STROKE"     #'core/cur-stroke     #'core/stroke-atom     "3.25"  3.25]
   ["GA_PAINTER_VAR"        #'core/cur-var        #'core/variation-atom  "0.75"  0.75]
   ["GA_PAINTER_CURV"       #'core/cur-curv       #'core/curvature-atom  "0.125" 0.125]
   ["GA_PAINTER_BROAD"      #'core/cur-broad      #'core/broad-atom      "2.5"   2.5]
   ["GA_PAINTER_MID"        #'core/cur-mid        #'core/mid-atom        "0.4"   0.4]
   ["GA_PAINTER_FINE"       #'core/cur-fine       #'core/fine-atom       "1.6"   1.6]
   ["GA_PAINTER_CUTIN"      #'core/cur-cutin      #'core/cutin-atom      "0.0"   0.0]
   ["GA_PAINTER_SWIRL"      #'core/cur-swirl      #'core/swirl-atom      "0.5"   0.5]
   ["GA_PAINTER_CONTRAST"   #'core/cur-contrast   #'core/contrast-atom   "1.4"   1.4]
   ["GA_PAINTER_BRIGHTNESS" #'core/cur-brightness #'core/brightness-atom "1.3"   1.3]
   ["GA_PAINTER_LIFT"       #'core/cur-lift       #'core/lift-atom       "1.8"   1.8]
   ["GA_PAINTER_HARDNESS"   #'core/cur-hardness   #'core/hardness-atom   "1.2"   1.2]
   ["GA_PAINTER_AA"         #'core/cur-aa         #'core/aa-atom         "1.0"   1.0]
   ["GA_PAINTER_TEX_STREAK" #'core/cur-tex-streak #'core/tex-streak-atom "0.3"   0.3]
   ["GA_PAINTER_TEX_GRAIN"  #'core/cur-tex-grain  #'core/tex-grain-atom  "0.2"   0.2]
   ["GA_PAINTER_TEX_EDGE"   #'core/cur-tex-edge   #'core/tex-edge-atom   "0.05"  0.05]])

(deftest env-overrides-win
  (testing "each GA_PAINTER_* override replaces its slider atom"
    (doseq [[var-name accessor _atom-var s expected] overrides]
      (with-env var-name s
        #(is (= expected (accessor)) var-name)))))

(deftest accessors-fall-back-to-the-atom
  (testing "with the env unset the accessor reads the live slider"
    (doseq [[var-name accessor atom-var _s _expected] overrides]
      (c-unsetenv var-name)
      (is (= @@atom-var (accessor)) var-name))))

(deftest the-extension-picks-the-writer
  (testing "a scripted GA_PAINTER_SAVE_PNG=/tmp/a.svg is a vector save"
    (is (core/svg-path? "/tmp/a.svg"))
    (is (core/svg-path? "/tmp/A.SVG"))
    (is (not (core/svg-path? "/tmp/a.png")))
    (is (not (core/svg-path? "/tmp/svg.png")) "the extension, not the name")
    (is (not (core/svg-path? nil)))))

(deftest re-picking-the-active-save-format-still-re-renders
  ;; Clicking the box that is already on UNCHECKS it in GTK, and glimmer.ratom only
  ;; notifies on a real change — so a plain reset! to the format it already holds
  ;; would publish nothing, and the box would sit unchecked while the atom disagreed.
  (let [seen  (atom [])
        watch (fn [r] (swap! seen conj @r))
        orig  @core/save-format-atom]
    (try
      (reset! core/save-format-atom :png)
      (swap! (:watches core/save-format-atom) conj watch)   ; glimmer.ratom's watcher set
      (#'core/set-save-format! :svg)
      (is (= :svg @core/save-format-atom))
      (is (= [:svg] @seen) "a real change publishes once")
      (reset! seen [])
      (#'core/set-save-format! :svg)
      (is (= :svg @core/save-format-atom) "the format is unchanged")
      (is (seq @seen) "but the toggles were told to re-render anyway")
      (finally
        (swap! (:watches core/save-format-atom) disj watch)
        (reset! core/save-format-atom orig)))))

(deftest every-settings-atom-has-an-override
  (testing "no slider is reachable only through the UI"
    (let [covered (set (map (fn [[_ _ a _ _]] a) overrides))]
      (doseq [a @#'core/settings-atoms]
        (is (contains? (set (map deref covered)) a)
            "settings-atoms entry with no GA_PAINTER_* override")))))
