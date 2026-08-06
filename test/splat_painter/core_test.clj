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
    (is (core/svg-path? "/tmp/a.svgz") "and so is the gzipped flavour")
    (is (not (core/svg-path? "/tmp/a.png")))
    (is (not (core/svg-path? "/tmp/svg.png")) "the extension, not the name")
    (is (not (core/svg-path? nil))))
  (testing "and only .svgz gets gzipped"
    (is (core/gz-path? "/tmp/a.svgz"))
    (is (core/gz-path? "/tmp/A.SVGZ"))
    (is (not (core/gz-path? "/tmp/a.svg")))
    (is (not (core/gz-path? nil)))))

(deftest the-svg-box-offers-the-gzipped-extension
  ;; .svgz is the same document ~7x smaller and browsers and Inkscape open it directly,
  ;; so it is what Save… defaults to. Typing .svg over it still writes plain — the
  ;; extension saved is what picks the writer, not this.
  (is (= "svgz" (core/save-ext :svg)))
  (is (= "png" (core/save-ext :png)))
  (is (core/gz-path? (str "splats." (core/save-ext :svg)))))

(defn- toolbar-prop [tag k]
  (->> (#'core/toolbar) (filter vector?) (filter #(= tag (first %))) first second k))

(deftest toggle-handlers-are-called-with-the-widgets-value
  ;; THE contract that broke SVG export: :on-toggled is value-bearing — glimmer-gl
  ;; registers gtk_check_button_get_active as its value fn, so glimmer.widget calls
  ;; (handler active) and a zero-arg handler is an arity crash on the first real
  ;; click. Nothing catches that by calling the handler directly, so this pulls it
  ;; out of the actual toolbar tree and calls it the way the signal will.
  (let [h    (toolbar-prop :checkbutton :on-toggled)
        orig @core/save-format-atom]
    (try
      (is (some? h) "the SVG box still has a toggle handler")
      (h 1) (is (= :svg @core/save-format-atom) "checked (GTK's 1) is the vector save")
      (h 0) (is (= :png @core/save-format-atom) "unchecked is PNG")
      (testing "a boolean would do too, if the value fn ever stops being a C int"
        (h true)  (is (= :svg @core/save-format-atom))
        (h false) (is (= :png @core/save-format-atom)))
      (finally (reset! core/save-format-atom orig)))))

(deftest the-format-box-shows-the-format
  (let [orig @core/save-format-atom]
    (try
      (reset! core/save-format-atom :svg)
      (is (true? (toolbar-prop :checkbutton :active)))
      (reset! core/save-format-atom :png)
      (is (false? (toolbar-prop :checkbutton :active)))
      (finally (reset! core/save-format-atom orig)))))

(deftest every-settings-atom-has-an-override
  (testing "no slider is reachable only through the UI"
    (let [covered (set (map (fn [[_ _ a _ _]] a) overrides))]
      (doseq [a @#'core/settings-atoms]
        (is (contains? (set (map deref covered)) a)
            "settings-atoms entry with no GA_PAINTER_* override")))))
