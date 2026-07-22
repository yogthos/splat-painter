(ns splat-painter.core
  "The splat-painter app: a file picker, a control panel, and a GtkGLArea that
  renders the loaded image as a field of 2D gaussian splats.

  Architecture mirrors glimmer-gl's demo app: the UI is one reactive glimmer
  hiccup tree (the sliders are :scale widgets, the GL pane is a :gl-area), while
  the GLArea's realize/render/resize handlers are plain fns doing raw GL. The
  splat field is regenerated CPU-side (splat-painter.seed) whenever the image or a
  control changes and re-uploaded as an RGBA32F texture; the fragment shader
  (splat-painter.shader) is the GPU twin of splat-painter.gaussian/rasterize.

  The one piece of raw FFI in the UI is the Open Image dialog: GTK4 removed
  synchronous dialogs, so GtkFileDialog's async open is driven through a
  :collect-safe foreign-callable standing in for GAsyncReadyCallback."
  (:require [glimmer.core   :as ui]
            [glimmer.ratom  :as r]
            [glimmer-gl.gtk :as glx]        ; registers :gl-area + :scale
            [glimmer-gl.gl  :as gl]
            [splat-painter.shader    :as shader]
            [splat-painter.image     :as image]
            [splat-painter.seed      :as seed]
            [splat-painter.structure :as structure]
            [splat-painter.wavelet   :as wavelet]
            [splat-painter.png       :as png]
            [jolt.ffi       :as ffi]))

;; --- reactive controls (the panel re-renders on change) ----------------------
(defonce count-atom    (r/atom 14000))  ; splat budget (max strokes) — higher = more detail, slower
(defonce size-atom     (r/atom 16.0))   ; base (coarsest) splat stdev; finer levels halve it
(defonce stroke-atom   (r/atom 2.5))
(defonce detail-atom   (r/atom 0.6))    ; how many fine detail levels are added
(defonce variation-atom (r/atom 0.5))   ; per-stroke size/tone jitter
(defonce curvature-atom (r/atom 0.5))   ; Perlin warp — how much strokes bend/curve off-grid
(defonce opacity-atom  (r/atom 0.9))
(defonce palette-atom  (r/atom 0.0))
(defonce contrast-atom (r/atom 1.0))
(defonce hardness-atom (r/atom 1.7))   ; edge crispness of detail strokes; large strokes always
                                       ; soften to a round gaussian (u_hard_soft fixed at 1.0)
(defonce status-atom   (r/atom "no image loaded — click Open Image…"))

;; --- non-reactive image / GL state -------------------------------------------
;; Need a sized internal-format for macOS core-profile FBO completeness.
(def ^:private GL-RGBA8 0x8058)
(defonce image-atom (atom nil))   ; {:height :width :channels :pixels} or nil
(defonce gl-state   (atom {}))    ; per-GLArea GL handles, keyed by area pointer
(defonce saved?-atom (atom false)) ; one-shot headless save via GA_PAINTER_SAVE_PNG
(defonce viewport   (atom [800 600]))
(defonce area-atom  (atom nil))   ; the GLArea widget, for queue-render / dialog parent

;; --- GtkFileDialog (GTK4 async file picker) ----------------------------------
(ffi/defcfn gtk-file-dialog-new        "gtk_file_dialog_new"        [] :pointer)
(ffi/defcfn gtk-file-dialog-set-title  "gtk_file_dialog_set_title"  [:pointer :string] :void)
(ffi/defcfn gtk-file-dialog-open       "gtk_file_dialog_open"
  [:pointer :pointer :pointer :pointer :pointer] :void)
(ffi/defcfn gtk-file-dialog-open-finish "gtk_file_dialog_open_finish"
  [:pointer :pointer :pointer] :pointer)
(ffi/defcfn g-file-get-path  "g_file_get_path" [:pointer] :string)
(ffi/defcfn g-object-unref   "g_object_unref"  [:pointer] :void)
(ffi/defcfn g-error-free     "g_error_free"    [:pointer] :void)

;; --- GtkFileDialog save variant -----------------------------------------------
(ffi/defcfn gtk-file-dialog-save        "gtk_file_dialog_save"
  [:pointer :pointer :pointer :pointer :pointer] :void)
(ffi/defcfn gtk-file-dialog-save-finish "gtk_file_dialog_save_finish"
  [:pointer :pointer :pointer] :pointer)
(ffi/defcfn gtk-file-dialog-set-initial-name "gtk_file_dialog_set_initial_name"
  [:pointer :string] :void)

;; --- render requests ---------------------------------------------------------
;; glimmer.ratom atoms aren't IRef-watchable, so we don't add-watch. Control
;; sliders only reset their atoms; the Render button (or loading an image) calls
;; request-render!, and on-render rebuilds the field from the current atoms — so
;; you tune the sliders, then hit Render to see the result.
(defn- field-for-current-controls []
  (when-let [img @image-atom]
    (seed/splat-field img {:count     @count-atom
                           :size      @size-atom
                           :stroke    @stroke-atom
                           :detail    @detail-atom
                           :variation @variation-atom
                           :curvature @curvature-atom
                           :opacity   @opacity-atom
                           :palette   (int @palette-atom)
                           :contrast  @contrast-atom})))

(defn- request-render! []
  (when-let [a @area-atom] (glx/gtk-gl-area-queue-render a)))

(defn- on-image-loaded [path]
  (try
    (let [img0   (image/load-image path 1024)
          ;; precomputed once so live slider drags don't recompute: edge-orientation
          ;; tensor, a light blur (smooth average colour), the wavelet detail map, and
          ;; the Perlin flow fields. Placement is coarse-to-fine layers (splat-painter.seed),
          ;; no deforming grid.
          sfield (structure/analyze img0)
          img    (assoc img0 :structure sfield
                             :blur   (structure/blur-image img0 2)
                             :detail (wavelet/detail-map img0)
                             :noise-fields (seed/prep-noise sfield))]
      (reset! image-atom img)
      ;; Size is the base (flat-region) stroke stdev; detail shrinks it locally. Seed
      ;; it relative to the image so default strokes are visible brush marks.
      (reset! size-atom (max 4.0 (/ (double (:height img)) 50.0)))
      (reset! status-atom (format "%s  —  %d×%d" path (:width img) (:height img)))
      (request-render!))
    (catch Throwable e
      (reset! status-atom (str "failed to load: " (ex-message e))))))

;; GAsyncReadyCallback body: src is the GtkFileDialog, res the GAsyncResult.
(defn- handle-open-result [dialog res]
  (let [errslot (ffi/alloc (ffi/sizeof :pointer))
        gfile   (gtk-file-dialog-open-finish dialog res errslot)]
    (if (ffi/null? gfile)
      (when-let [ep (ffi/read errslot :pointer 0)]
        (let [mp (ffi/read ep :pointer 16)]
          (when-not (ffi/null? mp)
            (reset! status-atom (str "open failed: " (ffi/ptr->string mp))))
          (g-error-free ep)))
      (let [path (g-file-get-path gfile)]
        (g-object-unref gfile)
        (on-image-loaded path)))
    (ffi/free errslot)
    (g-object-unref dialog)))

(defn- open-image-dialog! []
  (if-not @area-atom
    (reset! status-atom "window not ready yet")
    (let [root   (glx/gtk-widget-get-root @area-atom)
          dialog (gtk-file-dialog-new)]
      (gtk-file-dialog-set-title dialog "Open image")
      (let [cb (ffi/foreign-callable
                 (fn [src res _ud] (handle-open-result src res))
                 [:pointer :pointer :pointer] :void :collect-safe)]
        (gtk-file-dialog-open dialog root ffi/null cb ffi/null)))))

;; --- save (offscreen render → glReadPixels → PNG) ----------------------------
(defn- ensure-export-targets! [area iw ih]
  "Lazily create an FBO + RGBA8 color texture in gl-state for export, resizing
   the texture to iw*ih on each call (needed when image size changes). Caches
   and reuses the FBO/texture across saves — no glDelete needed."
  (let [st   (get @gl-state area)
        fbo  (or (:export-fbo st)  (gl/gen-one gl/gl-gen-framebuffers))
        ctex (or (:export-tex st) (gl/gen-one gl/gl-gen-textures))]
    (gl/gl-bind-texture gl/GL-TEXTURE-2D ctex)
    (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 GL-RGBA8 (int iw) (int ih) 0
                        gl/GL-RGBA gl/GL-UNSIGNED-BYTE ffi/null)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-NEAREST)
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
    (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0
                                  gl/GL-TEXTURE-2D ctex 0)
    (swap! gl-state update area assoc :export-fbo fbo :export-tex ctex)
    [fbo ctex]))

(defn- save-png! [path]
  (when-let [area @area-atom]
    (glx/make-current area)
    (if-let [st (get @gl-state area)]
      (let [fld (field-for-current-controls)]
        (if-not fld
          (reset! status-atom "no image to save")
          (let [iw (int (:width fld)) ih (int (:height fld))
                splats (:splats fld)
                n (count splats)
                {:keys [locs tex]} st
                bg (:background fld)
                ;; GTK4 GLArea renders into its own framebuffer, not 0.
                ;; Save the current binding and restore it after.
                prev-fbo (let [p (ffi/alloc (ffi/sizeof :int))]
                           (gl/gl-get-integerv gl/GL-FRAMEBUFFER-BINDING p)
                           (let [v (ffi/read p :int 0)] (ffi/free p) v))]
            (ensure-export-targets! area iw ih)
            (when (not= (gl/gl-check-framebuffer-status gl/GL-FRAMEBUFFER)
                        gl/GL-FRAMEBUFFER-COMPLETE)
              (reset! status-atom "save failed: framebuffer incomplete"))
            ;; render offscreen at native image resolution
            (gl/gl-viewport 0 0 iw ih)
            (upload-splat-texture! tex splats)
            (gl/gl-use-program (get-in st [:prog :program]))
            (gl/gl-active-texture gl/GL-TEXTURE0)
            (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
            (gl/gl-uniform-1i (:u_splats locs) 0)
            (gl/gl-uniform-1i (:u_count locs) (int n))
            ;; u_viewport = u_image => scale=1, no letterbox; whole framebuffer = image
            (gl/gl-uniform-2f (:u_viewport locs) (double iw) (double ih))
            (gl/gl-uniform-2f (:u_image locs) (double iw) (double ih))
            (gl/gl-uniform-3f (:u_bg locs) (double (nth bg 0)) (double (nth bg 1)) (double (nth bg 2)))
            (gl/gl-uniform-1f (:u_opacity locs) (double (or (:opacity fld) 1.0)))
            (gl/gl-uniform-1f (:u_hard_sharp locs) (double @hardness-atom))
            (gl/gl-uniform-1f (:u_hard_soft locs) 1.0)   ; large strokes = round gaussian (fixed)
            (gl/gl-uniform-1f (:u_sig_min locs) (double (or (:sig-min fld) 1.0)))
            (gl/gl-uniform-1f (:u_sig_max locs) (double (or (:sig-max fld) 1.0)))
            (gl/gl-clear-color 0.0 0.0 0.0 1.0)
            (gl/gl-clear gl/GL-COLOR-BUFFER-BIT)
            (gl/gl-bind-vertex-array (:vao st))
            (gl/gl-draw-arrays gl/GL-TRIANGLE-STRIP 0 4)
            ;; readback: RGBA 4-byte aligned, no PACK_ALIGNMENT issue
            (let [buf (ffi/alloc (* iw ih 4))]
              (gl/gl-read-pixels 0 0 iw ih gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
              (try
                (png/save-rgba-bottom-up! buf iw ih path)
                (reset! status-atom (format "saved %s  (%d×%d)" path iw ih))
                (catch Throwable e
                  (reset! status-atom (str "save failed: " (ex-message e))))
                (finally (ffi/free buf))))
            ;; restore GTK's framebuffer + window viewport, repaint
            (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev-fbo)
            (let [[w h] @viewport] (gl/gl-viewport 0 0 w h))
            (request-render!))))
      (reset! status-atom "GL not ready"))))

(defn- handle-save-result [dialog res]
  (let [errslot (ffi/alloc (ffi/sizeof :pointer))
        gfile   (gtk-file-dialog-save-finish dialog res errslot)]
    (if (ffi/null? gfile)
      (when-let [ep (ffi/read errslot :pointer 0)]
        (let [mp (ffi/read ep :pointer 16)]
          (when-not (ffi/null? mp)
            (reset! status-atom (str "save canceled/failed: " (ffi/ptr->string mp))))
          (g-error-free ep)))
      (let [path (g-file-get-path gfile)]
        (g-object-unref gfile)
        (save-png! path)))
    (ffi/free errslot)
    (g-object-unref dialog)))

(defn- save-image-dialog! []
  (if-not @area-atom
    (reset! status-atom "window not ready yet")
    (let [root   (glx/gtk-widget-get-root @area-atom)
          dialog (gtk-file-dialog-new)]
      (gtk-file-dialog-set-title dialog "Save PNG")
      (gtk-file-dialog-set-initial-name dialog "splats.png")
      (let [cb (ffi/foreign-callable
                 (fn [src res _ud] (handle-save-result src res))
                 [:pointer :pointer :pointer] :void :collect-safe)]
        (gtk-file-dialog-save dialog root ffi/null cb ffi/null)))))

;; --- GL plumbing -------------------------------------------------------------
(def ^:private quad-verts [-1.0 -1.0   1.0 -1.0   -1.0 1.0   1.0 1.0])

(defn- upload-splat-texture! [tex splats]
  ;; TILED layout so N can exceed GL_MAX_TEXTURE_SIZE: (2*TILE_W) x ceil(N/TILE_W), splats
  ;; laid out row-major (TILE_W per row). pad the buffer to a full rectangle of texels.
  (let [n      (count splats)
        tw     shader/tile-w
        rows   (max 1 (long (Math/ceil (/ (double n) (double tw)))))
        floats (shader/pack-splats splats)          ; n*8 floats, splat-major
        need   (* rows tw 8)
        padded (if (< (count floats) need)
                 (into floats (repeat (- need (count floats)) 0.0))
                 floats)
        ptr    (gl/write-floats padded)]
    (gl/gl-active-texture gl/GL-TEXTURE0)
    (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
    (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 gl/GL-RGBA32F (* 2 tw) rows 0
                        gl/GL-RGBA gl/GL-FLOAT ptr)
    (ffi/free ptr)
    n))

;; Build the per-GLArea GL objects once on realize. Kept shallow (one let) so the
;; closer count is obvious.
(defn- realize-gl! [area prog]
  (let [locs (:locs prog)
        vao  (gl/gen-one gl/gl-gen-vertex-arrays)
        vbo  (gl/gen-one gl/gl-gen-buffers)
        tex  (gl/gen-one gl/gl-gen-textures)
        qptr (gl/write-floats quad-verts)]
    (gl/gl-bind-buffer gl/GL-ARRAY-BUFFER vbo)
    (gl/gl-buffer-data gl/GL-ARRAY-BUFFER (* 8 (ffi/sizeof :float)) qptr gl/GL-STATIC-DRAW)
    (ffi/free qptr)
    (gl/gl-bind-vertex-array vao)
    (let [pos (:a_pos locs)]
      (gl/gl-enable-vertex-attrib-array pos)
      (gl/gl-vertex-attrib-pointer pos 2 gl/GL-FLOAT gl/GL-FALSE (* 2 (ffi/sizeof :float)) 0))
    (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-S gl/GL-CLAMP-TO-EDGE)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-T gl/GL-CLAMP-TO-EDGE)
    ;; back the texture with 1 texel so the sampler isn't 'unloadable' before the
    ;; first image loads (silences a macOS GL warning on the initial frame)
    (let [zptr (gl/write-floats [0.0 0.0 0.0 0.0])]
      (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 gl/GL-RGBA32F 1 1 0 gl/GL-RGBA gl/GL-FLOAT zptr)
      (ffi/free zptr))
    (swap! gl-state assoc area {:prog prog :locs locs :vao vao :vbo vbo :tex tex})
    (println "splat-painter: GL ready — program" (:program prog))))

(defn on-realize [area]
  (reset! area-atom area)
  (glx/make-current area)
  (when-let [err (glx/gl-area-error-message area)]
    (println "GLArea context error:" err))
  (if-let [prog (shader/build-program)]
    (realize-gl! area prog)
    (println "splat-painter: failed to build GL program (see info log above)")))

(defn on-resize [_area w h]
  (reset! viewport [w h])
  (gl/gl-viewport 0 0 w h))

(defn on-render [area]
  (when-let [st (get @gl-state area)]
    (let [{:keys [locs tex]} st
          [w h]   @viewport
          fld     (field-for-current-controls)
          splats  (:splats fld)
          n (do (when splats (upload-splat-texture! tex splats)) (or (count splats) 0))]
      (gl/gl-clear-color 0.05 0.06 0.09 1.0)
      (gl/gl-clear gl/GL-COLOR-BUFFER-BIT)
      (gl/gl-use-program (get-in st [:prog :program]))
      (gl/gl-active-texture gl/GL-TEXTURE0)
      (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
      (let [iw (if fld (:width fld) w)
            ih (if fld (:height fld) h)
            bg (if fld (:background fld) [0.0 0.0 0.0])]
        (gl/gl-uniform-1i (:u_splats locs) 0)
        (gl/gl-uniform-1i (:u_count locs) (int n))
        (gl/gl-uniform-2f (:u_viewport locs) (double w) (double h))
        (gl/gl-uniform-2f (:u_image locs) (double iw) (double ih))
        (gl/gl-uniform-3f (:u_bg locs) (double (nth bg 0)) (double (nth bg 1)) (double (nth bg 2)))
        (gl/gl-uniform-1f (:u_opacity locs) (double (or (:opacity fld) 1.0)))
        (gl/gl-uniform-1f (:u_hard_sharp locs) (double @hardness-atom))
        (gl/gl-uniform-1f (:u_hard_soft locs) 1.0)   ; large strokes = round gaussian (fixed)
        (gl/gl-uniform-1f (:u_sig_min locs) (double (or (:sig-min fld) 1.0)))
        (gl/gl-uniform-1f (:u_sig_max locs) (double (or (:sig-max fld) 1.0))))
      (gl/gl-bind-vertex-array (:vao st))
      (gl/gl-draw-arrays gl/GL-TRIANGLE-STRIP 0 4)
      ;; headless save hook: set GA_PAINTER_SAVE_PNG=/path/to/out.png to export
      (when-let [p (System/getenv "GA_PAINTER_SAVE_PNG")]
        (when (and (not @saved?-atom) fld)
          (reset! saved?-atom true)
          (save-png! p))))))

;; --- reactive control panel --------------------------------------------------
;; Layout rule that keeps the sidebar narrow: NO widget inside the sidebar ever
;; sets :hexpand. In GTK a GtkBox reports itself as "wanting to expand" to its
;; parent whenever any child expands (gtk_widget_compute_expand propagates up the
;; tree). So a :scale with :hexpand true made the control-panel :vbox compete
;; with the :gl-area for the root :hbox's free width — splitting the window and
;; letting the sidebar balloon past its :width-request (which is only a minimum).
;; Instead each :scale gets an explicit :width-request, which sizes the track
;; without expanding. The control-panel then stays at its natural width (~180px)
;; and the :gl-area (the only :hexpand widget) fills everything else.
;;
;; Sliders are live: every value-changed queues a render so the image updates as
;; you drag. GTK coalesces queue-render calls to one render per frame-clock tick.
(defn- slider [label lo hi step value-atom]
  [:hbox {:spacing 8}
   [:label {:label label :width-chars 9 :xalign 0.0}]
   [:scale {:min lo :max hi :step step :value @value-atom :digits 2
            :width-request 120
            :on-value #(do (reset! value-atom %) (request-render!))}]])

(defn- control-panel []
  [:vbox {:spacing 6 :margin 8 :width-request 180}
   [:hbox {:spacing 6}
    [:button {:label "Open Image…" :on-click open-image-dialog!}]
    [:button {:label "Render"      :on-click request-render!}]
    [:button {:label "Save PNG…"   :on-click save-image-dialog!}]]
   ;; cap the path so the label can't widen the sidebar; ellipsize the tail
   [:label {:label @status-atom :xalign 0.0 :halign :start
            :max-width-chars 22 :ellipsize :end}]
   [:separator {}]
   [slider "Splats"    1000 48000 500   count-atom]     ; budget: higher = more detail, slower
   [slider "Size"      6    50    0.5   size-atom]
   [slider "Detail"    0.0  1.0   0.02  detail-atom]
   [slider "Variation" 0.0  1.0   0.02  variation-atom]
   [slider "Curvature" 0.0  1.0   0.02  curvature-atom]
   [slider "Stroke"    0.0  4.0   0.05  stroke-atom]
   [slider "Opacity"   0.1  1.0   0.02  opacity-atom]
   [slider "Palette"   0    64    1     palette-atom]
   [slider "Contrast"  0.5  2.0   0.05  contrast-atom]
   [slider "Hardness"  1.0  4.0   0.05  hardness-atom]])   ; detail-stroke crispness (big strokes stay round)

(defn app []
  [:hbox {:spacing 0}
   [control-panel]
   [:separator {:orientation :vertical}]
   [:gl-area {:version [3 2] :depth-buffer false :hexpand true :vexpand true
              :on-realize on-realize
              :on-render  on-render
              :on-resize  on-resize}]])

(defn -main [& args]
  ;; an optional image path on the command line loads immediately — handy for
  ;; smoke-testing the full render path without clicking the dialog
  (when (seq args) (on-image-loaded (first args)))
  (let [quit-ms (some-> (System/getenv "GA_PAINTER_QUIT_MS") Integer/parseInt)]
    (apply ui/run app
           :app-id "dev.jolt.splat-painter"
           :title  "splat-painter • gaussian splats"
           :width  1100 :height 720
           (when quit-ms [:auto-quit-ms quit-ms]))))
