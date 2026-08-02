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
  (:require [clojure.string]
            [glimmer.core   :as ui]
            [glimmer.ratom  :as r]
            [glimmer-gl.gtk :as glx]        ; registers :gl-area + :scale
            [glimmer-gl.gl  :as gl]
            [splat-painter.shader    :as shader]
            [splat-painter.gen       :as gen]
            [splat-painter.image     :as image]
            [splat-painter.fields    :as fields]
            [splat-painter.gpu-fields :as gf]
            [splat-painter.seed      :as seed]
            [jolt.ffi       :as ffi]))

;; --- reactive controls (the panel re-renders on change) ----------------------
(defonce count-atom    (r/atom 72000))  ; splat budget (max strokes) — higher = more detail, slower
(defonce size-atom     (r/atom 16.0))   ; base (coarsest) splat stdev; finer levels halve it
(defonce stroke-atom   (r/atom 2.5))
(defonce detail-atom   (r/atom 0.6))    ; how many fine detail levels are added
(defonce variation-atom (r/atom 0.5))   ; per-stroke size/tone jitter
(defonce curvature-atom (r/atom 0.5))   ; Perlin warp — how much strokes bend/curve off-grid
(defonce contrast-atom (r/atom 1.0))
(defonce broad-atom    (r/atom 1.0))   ; per-tier size multipliers: background/large
(defonce mid-atom      (r/atom 1.0))   ; features vs mid structure vs the finest
(defonce fine-atom     (r/atom 1.0))   ; details, each loosened/focused independently
(defonce cutin-atom    (r/atom 1.0))   ; edge-band tier strength — restate silhouettes from their own sides; 0 = off
(defonce swirl-atom    (r/atom 1.0))   ; share of the image-INDEPENDENT Perlin field in the flow
                                       ; orientation + position warp; 0 = structure only
(defonce hardness-atom (r/atom 1.7))   ; edge crispness of detail strokes; large strokes always
(defonce aa-atom       (r/atom 0.0))   ; ANTIALIAS present pass: edge-directed smoothing, runs after Sharpen
(defonce clip-atom     (r/atom 0.0))   ; EDGE CLIP: stroke tails stop at a silhouette; 0 = off, byte-identical
                                       ; soften to a round gaussian (u_hard_soft fixed at 1.0)
(defonce sharpen-atom  (r/atom 0.0))   ; final-output edge-gated unsharp on the composite; 0 = off
;; paint-texture experiment (render-time, quad shader only): pigment-like variance
;; WITHIN each stroke. All zero == the clean-gaussian render.
(defonce tex-streak-atom (r/atom 0.18)) ; bristle tonal streaks along the drag
(defonce tex-grain-atom  (r/atom 0.10)) ; canvas-tooth brightness+chroma mottle
(defonce tex-edge-atom   (r/atom 0.25)) ; edge raggedness (contour breaks up)
(defonce status-atom   (r/atom "no image loaded — click Open Image…"))
;; layered repainting: layers-atom is the committed stack (bottom-first). Each
;; entry {:tex <GL id> :opacity <double> :settings <14-vector>} holds a layer's
;; SOLO render — its own full splat pass captured over a TRANSPARENT clear, so the
;; texture is premultiplied RGBA the alpha-aware blit composites directly.
;; active-layer-atom is the index [0..(count layers)] where the LIVE pass sits: it
;; is NOT stored, it is regenerated every frame from image-atom + the current
;; sliders. Layers below `active` blit UNDER the live pass, layers above blit OVER
;; it (see gpu-draw! for the compositing order). Empty stack + active 0 == today's
;; single-pass render (bit-identical — no blits, splats at the fixed 0.9).
(defonce layers-atom         (r/atom []))
(defonce active-layer-atom   (r/atom 0))    ; live pass index: 0 == bottom (single pass)
(defonce layer-opacity-atom  (r/atom 0.6)) ; glaze strength of the active pass over its base

;; --- non-reactive image / GL state -------------------------------------------
;; Need a sized internal-format for macOS core-profile FBO completeness.
(def ^:private GL-RGBA8 0x8058)
;; The splat field is built on the GPU via transform feedback (splat-painter.gen) —
;; ~50× faster than CPU seed/splat-field, which stays the tested reference: the
;; goldens pin it, GA_PAINTER_GPU_VERIFY compares the two fields numerically, and
;; `jolt -M:preview` renders one to PNG with no GL context at all.
(defonce image-atom (atom nil))   ; {:height :width :channels :pixels} or nil
(defonce gl-state   (atom {}))    ; per-GLArea GL handles, keyed by area pointer
(defonce saved?-atom (atom false)) ; one-shot headless save via GA_PAINTER_SAVE_PNG
(defonce viewport   (atom [800 600]))
(defonce area-atom  (atom nil))   ; the GLArea widget, for queue-render / dialog parent
(defonce image-path-atom (atom nil)) ; source path of the loaded image (dialog-free save)
;; GtkFileDialog's async callbacks are :collect-safe foreign-callables. GTK holds
;; only the raw pointer, so we must keep a Clojure reference alive or GC (triggered
;; by e.g. the save path's big pixel-buffer alloc) reclaims the trampoline mid-call
;; and the callback faults on return — "invalid memory reference". One slot per
;; dialog kind, overwritten on the next open (the prior callback has already run).
(defonce ^:private open-cb-atom (atom nil))
(defonce ^:private save-cb-atom (atom nil))

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

;; --- GSettings schema guard ----------------------------------------------------
;; GtkFileDialog ABORTS the process (GLib-GIO-ERROR -> SIGTRAP, uncatchable) when
;; GLib can't find the GSettings schemas. On Homebrew macOS they live under
;; /opt/homebrew/share, which a plain shell launch doesn't have in XDG_DATA_DIRS.
(ffi/defcfn c-setenv "setenv" [:string :string :int] :int)
(ffi/defcfn c-getenv "getenv" [:string] :pointer)

(defn- getenv* [name]
  (let [p (c-getenv name)]
    (when-not (ffi/null? p) (ffi/ptr->string p))))

(defn- readable? [path]
  (try (do (slurp path) true) (catch Throwable _ false)))

(def ^:private schema-dirs ["/opt/homebrew/share" "/usr/local/share" "/usr/share"])

(defn- schemas-available? []
  (some #(readable? (str % "/glib-2.0/schemas/gschemas.compiled")) schema-dirs))

(defn- ensure-schema-path!
  "Make GLib find gschemas.compiled at GTK/GSettings init. Must run before GTK init.
  We can't just set the var when it's empty: a caller that already exports
  XDG_DATA_DIRS *without* Homebrew's prefix (e.g. some terminals do) would still
  abort the file dialog even though the schemas are on disk. So PREPEND our dirs
  (overwrite=1, existing entries preserved) and also set GSETTINGS_SCHEMA_DIR — the
  direct lever GSettings honours — at the concrete schema dirs."
  []
  (letfn [(prepend! [var dirs]
            (let [ours (clojure.string/join ":" dirs)
                  cur  (getenv* var)]
              (c-setenv var (if (or (nil? cur) (zero? (count cur))) ours (str ours ":" cur)) 1)))]
    (prepend! "XDG_DATA_DIRS" schema-dirs)
    (prepend! "GSETTINGS_SCHEMA_DIR" (map #(str % "/glib-2.0/schemas") schema-dirs))))

;; --- GtkFileDialog save variant -----------------------------------------------
(ffi/defcfn gtk-file-dialog-save        "gtk_file_dialog_save"
  [:pointer :pointer :pointer :pointer :pointer] :void)
(ffi/defcfn gtk-file-dialog-save-finish "gtk_file_dialog_save_finish"
  [:pointer :pointer :pointer] :pointer)
(ffi/defcfn gtk-file-dialog-set-initial-name "gtk_file_dialog_set_initial_name"
  [:pointer :string] :void)
;; glimmer-gl v0.0.3 doesn't bind glDeleteTextures, so declare it locally. Used by
;; delete-layer-textures! to free committed-layer textures on drop/reset/replace.
(ffi/defcfn gl-delete-textures "glDeleteTextures" [:int :pointer] :void)

;; --- render requests ---------------------------------------------------------
;; glimmer.ratom atoms aren't IRef-watchable, so we don't add-watch. Each slider's
;; :on-value resets its atom and calls request-render!, so the image updates live
;; as you drag; loading an image does the same. on-render rebuilds the field from
;; the current atoms. (GTK coalesces queued renders to one per frame-clock tick.)
;; test/headless overrides so a fixed count/size can be forced without the sliders
;; forward references: defined further down, used by the render/callback wiring above.
;; A cold AOT compile resolves a file top-down, so without these a clean checkout
;; cannot compile core (a warm ~/.jolt/aot-cache hides it).
(declare clear-layers! gpu-save-png!)

(defn- cur-count  [] (or (some-> (System/getenv "GA_PAINTER_COUNT")  Double/parseDouble long) @count-atom))
(defn- cur-size   [] (or (some-> (System/getenv "GA_PAINTER_SIZE")   Double/parseDouble)      @size-atom))
(defn- cur-detail [] (or (some-> (System/getenv "GA_PAINTER_DETAIL") Double/parseDouble)      @detail-atom))
(defn- cur-stroke [] (or (some-> (System/getenv "GA_PAINTER_STROKE") Double/parseDouble)      @stroke-atom))
(defn- cur-var    [] (or (some-> (System/getenv "GA_PAINTER_VAR")    Double/parseDouble)      @variation-atom))
(defn- cur-curv   [] (or (some-> (System/getenv "GA_PAINTER_CURV")   Double/parseDouble)      @curvature-atom))
(defn- cur-broad  [] (or (some-> (System/getenv "GA_PAINTER_BROAD")  Double/parseDouble)      @broad-atom))
(defn- cur-mid    [] (or (some-> (System/getenv "GA_PAINTER_MID")    Double/parseDouble)      @mid-atom))
(defn- cur-fine   [] (or (some-> (System/getenv "GA_PAINTER_FINE")   Double/parseDouble)      @fine-atom))
(defn- cur-cutin  [] (or (some-> (System/getenv "GA_PAINTER_CUTIN")  Double/parseDouble)      @cutin-atom))
(defn- cur-swirl  [] (or (some-> (System/getenv "GA_PAINTER_SWIRL")  Double/parseDouble)      @swirl-atom))
(defn- cur-contrast [] (or (some-> (System/getenv "GA_PAINTER_CONTRAST") Double/parseDouble)  @contrast-atom))
(defn- cur-hardness [] (or (some-> (System/getenv "GA_PAINTER_HARDNESS") Double/parseDouble)  @hardness-atom))
(defn- cur-aa       [] (or (some-> (System/getenv "GA_PAINTER_AA")       Double/parseDouble)  @aa-atom))
(defn- cur-clip     [] (or (some-> (System/getenv "GA_PAINTER_CLIP")     Double/parseDouble)  @clip-atom))
(defn- cur-sharpen [] (or (some-> (System/getenv "GA_PAINTER_SHARPEN") Double/parseDouble)  @sharpen-atom))
(defn- cur-tex-streak [] (or (some-> (System/getenv "GA_PAINTER_TEX_STREAK") Double/parseDouble) @tex-streak-atom))
(defn- cur-tex-grain  [] (or (some-> (System/getenv "GA_PAINTER_TEX_GRAIN")  Double/parseDouble) @tex-grain-atom))
(defn- cur-tex-edge   [] (or (some-> (System/getenv "GA_PAINTER_TEX_EDGE")   Double/parseDouble) @tex-edge-atom))
(defn- cur-layer-opacity [] (or (some-> (System/getenv "GA_PAINTER_LAYER_OPACITY") Double/parseDouble) @layer-opacity-atom))

(defn- field-for-current-controls
  "The CPU reference field. image-atom now holds RAW pixels — the shipping path
   builds its fields on the GPU — so this builds the CPU ones on demand. Only
   GA_PAINTER_GPU_VERIFY and the diagnostics come through here, never a render."
  []
  (when-let [img0 @image-atom]
    (let [img (if (:structure img0) img0 (fields/prepare img0))]
      (seed/splat-field img {:count     (cur-count)
                           :size      (cur-size)
                           :stroke    (cur-stroke)
                           :detail    (cur-detail)
                           :variation (cur-var)
                           :curvature (cur-curv)
                           :contrast  (cur-contrast)
                           :size-broad (cur-broad)
                           :size-mid   (cur-mid)
                           :size-fine  (cur-fine)
                           :edge-band  (cur-cutin)
                           :swirl      (cur-swirl)}))))

(defn- request-render! []
  (when-let [a @area-atom] (glx/gtk-gl-area-queue-render a)))

(defn- prepare-image
  "Attach the precomputed placement fields to a loaded image map (orientation tensor,
  the three colour sources, the wavelet detail map, the Perlin flow fields) so live
  slider drags don't recompute them. Reused by on-image-loaded (from a file) and
  add-layer! (from a captured render), so both paths build identical fields — and it
  is `fields/prepare` so the tests and diagnostics build them identically too."
  [img0]
  (fields/prepare img0))

(defn- on-image-loaded [path]
  (try
    ;; Just the pixels. Every field is derived in ensure-gpu! on the render
    ;; callback, which is the only place a GL context exists — -main loads the
    ;; first image before ui/run, so nothing here may touch GL or depend on it.
    (let [img (image/load-image path 1024)]
      (clear-layers!)               ; new file: drop the previous image's committed layers + free their textures
      (reset! image-atom img)
      (reset! image-path-atom path)
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
        _       (ffi/write errslot :pointer 0 ffi/null)  ; GError out-param must start NULL (ffi/alloc doesn't zero)
        gfile   (gtk-file-dialog-open-finish dialog res errslot)]
    (if (ffi/null? gfile)
      (let [ep (ffi/read errslot :pointer 0)]
        (when-not (ffi/null? ep)
          ;; GError is { GQuark domain(4); gint code(4); gchar *message; } — message is
          ;; at offset 8 on LP64, NOT 16. Reading 16 fetched a garbage pointer 8 bytes
          ;; past the struct; ptr->string on it faulted ("invalid memory reference") on
          ;; the error/cancel branch — the crash when a file dialog was dismissed.
          (let [mp (ffi/read ep :pointer 8)]
            (when-not (ffi/null? mp)
              (reset! status-atom (str "open failed: " (ffi/ptr->string mp))))
            (g-error-free ep))))
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
        (reset! open-cb-atom cb)   ; retain past the async callback (see the atom's docstring)
        (gtk-file-dialog-open dialog root ffi/null cb ffi/null)))))

;; --- save (offscreen render → glReadPixels → PNG) ----------------------------
(defn- ensure-export-targets!
  "Lazily create an FBO + RGBA8 color texture in gl-state for export, resizing
   the texture to iw*ih on each call (needed when image size changes). Caches
   and reuses the FBO/texture across saves — no glDelete needed."
  [area iw ih]
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

(defn- ensure-render-targets!
  "The two IMAGE-RESOLUTION FBO + RGBA8 texture pairs render-final! ping-pongs
   between: `composite` is what the splat passes draw into, `post` is what a
   final-output pass (sharpen) writes. Separate from ensure-export-targets!,
   which is the destination the layer captures read back.

   Reallocates ONLY when the size actually changes. ensure-export-targets!
   re-runs glTexImage2D unconditionally, which is fine for a save; this runs on
   every frame, so an unconditional realloc would churn a multi-megabyte texture
   through the driver on every step of a slider drag.

   LINEAR filtering: the pane blit resamples these to the widget size. The
   sharpen pass samples at exact texel centres, where LINEAR and NEAREST agree,
   so this costs it nothing."
  [area iw ih]
  (let [st (get @gl-state area)
        mk (fn [fbo-k tex-k]
             (let [fbo (or (fbo-k st) (gl/gen-one gl/gl-gen-framebuffers))
                   tex (or (tex-k st) (gl/gen-one gl/gl-gen-textures))]
               (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
               (when-not (= [iw ih] (:render-size st))
                 (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 GL-RGBA8 (int iw) (int ih) 0
                                     gl/GL-RGBA gl/GL-UNSIGNED-BYTE ffi/null)
                 (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-LINEAR)
                 (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-LINEAR)
                 (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-S gl/GL-CLAMP-TO-EDGE)
                 (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-T gl/GL-CLAMP-TO-EDGE)
                 (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
                 (gl/gl-framebuffer-texture-2d gl/GL-FRAMEBUFFER gl/GL-COLOR-ATTACHMENT0
                                               gl/GL-TEXTURE-2D tex 0))
               [fbo tex]))
        [cfbo ctex] (mk :composite-fbo :composite-tex)
        [pfbo ptex] (mk :post-fbo :post-tex)]
    (swap! gl-state update area assoc
           :composite-fbo cfbo :composite-tex ctex
           :post-fbo pfbo :post-tex ptex :render-size [iw ih])
    {:composite [cfbo ctex] :post [pfbo ptex]}))

(defn- handle-save-result [dialog res]
  (let [errslot (ffi/alloc (ffi/sizeof :pointer))
        _       (ffi/write errslot :pointer 0 ffi/null)  ; GError out-param must start NULL (ffi/alloc doesn't zero)
        gfile   (gtk-file-dialog-save-finish dialog res errslot)]
    (if (ffi/null? gfile)
      (let [ep (ffi/read errslot :pointer 0)]
        (when-not (ffi/null? ep)
          ;; GError is { GQuark domain(4); gint code(4); gchar *message; } — message is
          ;; at offset 8 on LP64, NOT 16. Reading 16 fetched a garbage pointer 8 bytes
          ;; past the struct; ptr->string on it faulted ("invalid memory reference") on
          ;; the error/cancel branch — the crash when a file dialog was dismissed.
          (let [mp (ffi/read ep :pointer 8)]
            (when-not (ffi/null? mp)
              (reset! status-atom (str "save canceled/failed: " (ffi/ptr->string mp))))
            (g-error-free ep))))
      (let [path (g-file-get-path gfile)]
        (g-object-unref gfile)
        (when @area-atom
          (glx/make-current @area-atom)
          (gpu-save-png! @area-atom path))))
    (ffi/free errslot)
    (g-object-unref dialog)))

(defn- save-image-dialog! []
  (cond
    (not @area-atom)
    (reset! status-atom "window not ready yet")

    ;; no GSettings schemas anywhere -> opening the dialog would ABORT the process;
    ;; save directly next to the source image instead.
    (not (schemas-available?))
    (let [path (str (or @image-path-atom "splat-painter") "-splats.png")]
      (glx/make-current @area-atom)
      (gpu-save-png! @area-atom path))

    :else
    (let [root   (glx/gtk-widget-get-root @area-atom)
          dialog (gtk-file-dialog-new)]
      (gtk-file-dialog-set-title dialog "Save PNG")
      (gtk-file-dialog-set-initial-name dialog "splats.png")
      (let [cb (ffi/foreign-callable
                 (fn [src res _ud] (handle-save-result src res))
                 [:pointer :pointer :pointer] :void :collect-safe)]
        (reset! save-cb-atom cb)   ; retain past the async callback (see the atom's docstring)
        (gtk-file-dialog-save dialog root ffi/null cb ffi/null)))))

;; --- GL plumbing -------------------------------------------------------------
(def ^:private quad-verts [-1.0 -1.0   1.0 -1.0   -1.0 1.0   1.0 1.0])

;; --- GPU generation path (transform feedback) --------------------------------
(defn- read-fbo-binding []
  (let [p (ffi/alloc (ffi/sizeof :int))]
    (gl/gl-get-integerv gl/GL-FRAMEBUFFER-BINDING p)
    (let [v (ffi/read p :int 0)] (ffi/free p) v)))

(defonce ^:private gpu-fields-ctx (atom nil))

(defn- build-image-fields!
  "The per-image field textures. Builds them with GPU passes when the shaders
   compile, otherwise falls back to computing them on the CPU and uploading.

   The fallback is not ceremony: gpu-fields needs GLSL that a driver could
   reject, and an image that will not load is worse than one that loads slowly.
   GA_PAINTER_CPU_FIELDS forces it, which is also how the two paths get compared."
  [img perm]
  (let [forced? (some? (System/getenv "GA_PAINTER_CPU_FIELDS"))
        progs   (when-not forced?
                  (or (:progs @gpu-fields-ctx)
                      (when-let [p (gf/build-programs)]
                        (swap! gpu-fields-ctx assoc :progs p) p)))]
    (if-not progs
      (gen/upload-fields! (fields/prepare img) perm)
      ;; the ctx is remade per load because it captures the caller's viewport and
      ;; framebuffer to restore; free it so we don't leak an FBO+VAO per image
      (let [ctx (gf/make-ctx)]
        (try (gf/build-fields! ctx progs img perm)
             (finally (gf/free-ctx! ctx)))))))

(defn- gpu-controls []
  {:count (cur-count) :size (cur-size) :stroke (cur-stroke) :detail (cur-detail)
   :variation (cur-var) :curvature (cur-curv) :contrast (cur-contrast)
   :size-broad (cur-broad) :size-mid (cur-mid) :size-fine (cur-fine)
   :edge-band (cur-cutin) :swirl (cur-swirl)})

(defn field-tex-ids
  "The per-image GL texture ids owned by a `fields` map (as produced by
   gen/upload-fields!), in fixed allocation order. This is an explicit allow-list:
   it excludes :perm (the shared Perlin texture, uploaded once by upload-perm!
   and reused across images) and the non-texture entries (:dmax float, :dmap map,
   the dim/src vectors, :H/:W). Only ids in this list may be freed on image swap
   — deleting :perm corrupts every later generate!, and deleting :dmax (a long)
   would release an unrelated texture id."
  [fields]
  (mapv fields [:detail :subject :noise :noise-swirl0 :blur :blur-drift :blur-heavy :raw]))

(defn- delete-field-textures!
  "Free the per-image texture objects of `fields`. Caller must have made the GL
   context current. No-op when `fields` is nil (first call, nothing to free)."
  [fields]
  (when fields
    (let [ids (field-tex-ids fields)]
      (when (pos? (count ids))
        (let [p (gl/write-ints (mapv int ids))]
          (gl-delete-textures (count ids) p)
          (ffi/free p))))))

(defn- quad-vao!
  "Fullscreen-quad VAO/VBO bound to `pos` (vs-src's a_pos)."
  [pos]
  (let [vao  (gl/gen-one gl/gl-gen-vertex-arrays)
        vbo  (gl/gen-one gl/gl-gen-buffers)
        qptr (gl/write-floats quad-verts)]
    (gl/gl-bind-buffer gl/GL-ARRAY-BUFFER vbo)
    (gl/gl-buffer-data gl/GL-ARRAY-BUFFER (* 8 (ffi/sizeof :float)) qptr gl/GL-STATIC-DRAW)
    (ffi/free qptr)
    (gl/gl-bind-vertex-array vao)
    (gl/gl-enable-vertex-attrib-array pos)
    (gl/gl-vertex-attrib-pointer pos 2 gl/GL-FLOAT gl/GL-FALSE (* 2 (ffi/sizeof :float)) 0)
    vao))

(defn- ensure-gpu!
  "Lazily build the GPU-generation objects (gen program, buffer-render program, perm
   texture, transform-feedback buffer + query + VAO, texture-buffer view) and (re)upload
   the per-image field textures when the image changes. Returns the :gpu state map."
  [area]
  (let [st  (get @gl-state area)
        gpu (:gpu st)
        gpu (if (:gen gpu)
              gpu
              (let [tf-buf (gl/gen-one gl/gl-gen-buffers)
                    render (shader/build-program-buf)]
                (gl/gl-bind-buffer gl/GL-TRANSFORM-FEEDBACK-BUFFER tf-buf)
                (gl/gl-buffer-data gl/GL-TRANSFORM-FEEDBACK-BUFFER
                                   (* shader/max-splats 12 (ffi/sizeof :float)) ffi/null gl/GL-DYNAMIC-COPY)
                {:gen     (gen/build-gen-program)
                 :render  render
                 :quad    (shader/build-program-quad)
                 :blit    (shader/build-program-blit)
                 :sharpen (shader/build-program-sharpen)
                 :aa      (shader/build-program-aa)
                 :perm    (gen/upload-perm!)
                 :tf-buf  tf-buf
                 :query   (gl/gen-one gl/gl-gen-queries)
                 :gen-vao (gl/gen-one gl/gl-gen-vertex-arrays)
                 :buf-tex (gl/gen-one gl/gl-gen-textures)
                 ;; fullscreen-quad VAO — only the GA_PAINTER_LOOP_RENDER variant
                 ;; draws attributes; the quad renderer is gl_VertexID-only (gen-vao).
                 :quad-vao (quad-vao! (:a_pos (:locs render)))}))
        img @image-atom
        gpu (if (identical? img (:fields-img gpu))
              gpu
              (do (delete-field-textures! (:fields gpu))   ; free outgoing per-image textures
                  (assoc gpu :fields (build-image-fields! img (:perm gpu)) :fields-img img)))]
    (swap! gl-state assoc-in [area :gpu] gpu)
    gpu))

(defn- blit-layers!
  "Composite committed layers[lo..hi) src-over (premultiplied) into the current FBO,
  then restore unit-0 buf-tex + the quad program (the blit leaves TEXTURE9 + the blit
  program bound). When base-opaque?, the bottom layer (vector index 0) is forced to
  u_alpha 1.0 — it is the original opaque underpaint; only the BELOW range ever
  contains it. Otherwise every layer uses its stored glaze opacity. No-op when lo >= hi."
  [blit quad gen-vao buf-tex tf-buf layers lo hi base-opaque? vw vh iw ih]
  (when (< lo hi)
    (let [blocs (:locs blit)]
      (gl/gl-enable gl/GL-BLEND)
      (gl/gl-blend-func gl/GL-ONE-FACTOR gl/GL-ONE-MINUS-SRC-ALPHA)
      (gl/gl-use-program (:program blit))
      (doseq [i (range lo hi)
              :let [{:keys [tex opacity]} (layers i)]]
        (gl/gl-active-texture (+ gl/GL-TEXTURE0 9))
        (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
        (gl/gl-uniform-1i (:u_layer blocs) 9)
        (gl/gl-uniform-1f (:u_alpha blocs)
                          (double (if (and base-opaque? (zero? i)) 1.0 opacity)))
        (gl/gl-uniform-2f (:u_viewport blocs) (double vw) (double vh))
        (gl/gl-uniform-2f (:u_image blocs) (double iw) (double ih))
        (gl/gl-bind-vertex-array gen-vao)            ; no attribs — gl_VertexID only
        (gl/gl-draw-arrays gl/GL-TRIANGLES 0 6))
      (gl/gl-active-texture gl/GL-TEXTURE0)
      (gl/gl-bind-texture gl/GL-TEXTURE-BUFFER buf-tex)
      (gl/gl-tex-buffer gl/GL-TEXTURE-BUFFER gl/GL-RGBA32F tf-buf)
      (gl/gl-use-program (:program quad)))))

(defn- gpu-draw!
  "Generate the splat field on the GPU and composite it into the currently-bound
   framebuffer. `vw/vh` = framebuffer viewport pixels (letterbox target), `iw/ih` =
   image pixels. Renders one blended quad per splat (Σ quad-areas work — no
   pixels×splats loop, no GPU-watchdog hang at high counts); set
   GA_PAINTER_LOOP_RENDER to force the old per-pixel loop shader for A/B compares.
   Returns {:count survivors :total candidates :n rendered}.

   Optional opts map (trailing arg):
     nil / {}        full composite — committed layers below `active` blit under the
                     live splat pass, committed layers above blit over it (all src-over,
                     premultiplied), opaque-black clear. What render-final! wraps.

   Every caller passes vw/vh == iw/ih: this always draws at native image resolution,
   and the letterbox fit exists only in blit-to-pane!. The vw/vh parameters are kept
   because the quad shader's mapping is written in terms of them.
     {:solo true}    capture the live pass ALONE: transparent clear, NO blits, splats
                     at the fixed 0.9. Used by commit-active! to grab a layer's solo
                     texture (premultiplied RGBA over the transparent clear).
     {:blits-below true} composite ONLY the layers under `active` over opaque black
                     (no splats) — select-layer! rebuilds the live pass's source image
                     from this below-composite."
  [area vw vh iw ih & [opts]]
  (let [{:keys [gen render quad blit fields tf-buf query gen-vao buf-tex quad-vao]} (ensure-gpu! area)
        {:keys [solo blits-below] :or {solo false blits-below false}} opts
        {:keys [count total sig-min sig-max]}
        (if blits-below
          {:count 0 :total 0 :sig-min 0.0 :sig-max 0.0}   ; no splats drawn — skip gen
          (gen/generate! gen fields (gpu-controls) tf-buf query gen-vao {:height ih :width iw}))
        n (min count shader/max-splats)]
    (gl/gl-finish)   ; make the transform-feedback writes visible to the texelFetch below
    (when (and (not blits-below) (System/getenv "GA_PAINTER_GPU_TIME"))
      (dotimes [_ 3] (gen/generate! gen fields (gpu-controls) tf-buf query gen-vao {:height ih :width iw}) (gl/gl-finish))
      (let [t1 (System/nanoTime) reps 8]
        (dotimes [_ reps] (gen/generate! gen fields (gpu-controls) tf-buf query gen-vao {:height ih :width iw}) (gl/gl-finish))
        (println (format "gpu-gen: %d->%d  %.1f ms/gen (warm avg of %d)" total count
                         (/ (- (System/nanoTime) t1) 1e6 reps) reps))))
    ;; the background fills the whole pane (the loop shader painted u_bg outside the
    ;; image rect too), so clear to it, then scissor the splat quads to the image rect
    ;; so stroke tails can't bleed into the letterbox bars.
    (if solo
      (gl/gl-clear-color 0.0 0.0 0.0 0.0)   ; solo capture: transparent — premultiplied splats accumulate correct dst alpha
      (gl/gl-clear-color 0.0 0.0 0.0 1.0))  ; opaque black (single-pass + composite base)
    (gl/gl-clear gl/GL-COLOR-BUFFER-BIT)
    (gl/gl-active-texture gl/GL-TEXTURE0)
    (gl/gl-bind-texture gl/GL-TEXTURE-BUFFER buf-tex)
    (gl/gl-tex-buffer gl/GL-TEXTURE-BUFFER gl/GL-RGBA32F tf-buf)
    (if (and (System/getenv "GA_PAINTER_LOOP_RENDER") (not solo) (not blits-below))
      (let [locs (:locs render)]
        (gl/gl-use-program (:program render))
        (gl/gl-uniform-1i (:u_splats locs) 0)
        (gl/gl-uniform-1i (:u_count locs) (int n))
        (gl/gl-uniform-2f (:u_viewport locs) (double vw) (double vh))
        (gl/gl-uniform-2f (:u_image locs) (double iw) (double ih))
        (gl/gl-uniform-3f (:u_bg locs) 0.0 0.0 0.0)
        (gl/gl-uniform-1f (:u_opacity locs) 0.9)   ; fixed stroke alpha (slider removed)
        (gl/gl-uniform-1f (:u_hard_sharp locs)
                          ;; short-stroke regimes render fine marks as SOFT dabs — hard
                          ;; plateau-edged dots don't merge and bead contours into pearls
                          (+ 1.0 (* (- (double (cur-hardness)) 1.0)
                                    (min 1.0 (+ 0.4 (* 0.24 (cur-stroke)))))))
        (gl/gl-uniform-1f (:u_hard_soft locs) 1.0)
        (gl/gl-uniform-1f (:u_sig_min locs) (double sig-min))
        (gl/gl-uniform-1f (:u_sig_max locs) (double sig-max))
        (gl/gl-bind-vertex-array quad-vao)
        (gl/gl-draw-arrays gl/GL-TRIANGLE-STRIP 0 4))
      (let [locs    (:locs quad)
            scale   (min (/ (double vw) iw) (/ (double vh) ih))
            dw      (long (Math/ceil (* iw scale))) dh (long (Math/ceil (* ih scale)))
            ox      (long (Math/floor (* 0.5 (- vw dw)))) oy (long (Math/floor (* 0.5 (- vh dh))))
            active  (long @active-layer-atom)
            layers  @layers-atom
            nlayers (clojure.core/count layers)]   ; `count` below is the shadowed splat count
        (gl/gl-use-program (:program quad))
        (gl/gl-uniform-1i (:u_splats locs) 0)
        (gl/gl-uniform-1i (:u_count locs) (int n))
        (gl/gl-uniform-2f (:u_viewport locs) (double vw) (double vh))
        (gl/gl-uniform-2f (:u_image locs) (double iw) (double ih))
        ;; the live pass: full 0.9 for the bottom (active 0) or a solo capture; else
        ;; a cur-layer-opacity glaze — a live preview of this pass's stored opacity.
        (gl/gl-uniform-1f (:u_opacity locs)
                          (cond solo 0.9 (zero? active) 0.9 :else (* 0.9 (cur-layer-opacity))))
        (gl/gl-uniform-1f (:u_hard_sharp locs)
                          ;; short-stroke regimes render fine marks as SOFT dabs — hard
                          ;; plateau-edged dots don't merge and bead contours into pearls
                          (+ 1.0 (* (- (double (cur-hardness)) 1.0)
                                    (min 1.0 (+ 0.4 (* 0.24 (cur-stroke)))))))
        (gl/gl-uniform-1f (:u_hard_soft locs) 1.0)
        (gl/gl-uniform-1f (:u_sig_min locs) (double sig-min))
        (gl/gl-uniform-1f (:u_sig_max locs) (double sig-max))
        (gl/gl-uniform-1f (:u_tex_streak locs) (double (cur-tex-streak)))
        (gl/gl-uniform-1f (:u_tex_grain locs)  (double (cur-tex-grain)))
        (gl/gl-uniform-1f (:u_tex_edge locs)   (double (cur-tex-edge)))
        ;; EDGE CLIP: bind the bilateral (the ground each stroke paints over) so the
        ;; fragment shader can stop a stroke at a silhouette instead of feathering it
        ;; through — see shader/edge-clip-factor. Unit 0 is the splat buffer, so this
        ;; takes unit 2; restore the active unit afterwards like blit-to-pane! does.
        (gl/gl-uniform-1f (:u_clip locs) (double (cur-clip)))
        (gl/gl-uniform-2f (:u_hw locs) (double ih) (double iw))
        ;; ALWAYS point u_blurTex at unit 2 and bind there, even when the clip is off.
        ;; Leaving it at its default 0 puts a sampler2D and u_splats (a samplerBuffer)
        ;; on the SAME unit, which is invalid: the draw is dropped and the frame comes
        ;; out black. The shader still skips the fetch when u_clip is 0, so this costs
        ;; one bind per draw and nothing per fragment.
        (gl/gl-active-texture (+ gl/GL-TEXTURE0 2))
        (gl/gl-bind-texture gl/GL-TEXTURE-2D (:blur fields))
        (gl/gl-uniform-1i (:u_blurTex locs) 2)
        (gl/gl-active-texture gl/GL-TEXTURE0)
        (gl/gl-enable gl/GL-SCISSOR-TEST)
        (gl/gl-scissor (int ox) (int oy) (int dw) (int dh))
        ;; COMPOSITING ORDER — all src-over, premultiplied, one scissor block so no
        ;; committed layer nor the live splats bleed into the letterbox bars:
        ;;  1. committed layers UNDER the live pass  (layers[0..active)); the bottom
        ;;     layer (index 0 = the original opaque pass) is forced to u_alpha 1.0.
        (when-not solo
          (blit-layers! blit quad gen-vao buf-tex tf-buf layers 0 active true vw vh iw ih))
        ;;  2. the live splat pass (skipped in the blits-below source rebuild).
        (when-not blits-below
          (gl/gl-enable gl/GL-BLEND)
          (gl/gl-blend-func gl/GL-ONE-FACTOR gl/GL-ONE-MINUS-SRC-ALPHA)
          (gl/gl-use-program (:program quad))
          (gl/gl-bind-vertex-array gen-vao)            ; no attribs — gl_VertexID only
          (gl/gl-draw-arrays gl/GL-TRIANGLES 0 (int (* 6 n))))
        ;;  3. committed layers OVER the live pass  (layers[active..nlayers)); each at
        ;;     its own stored opacity (the i==0 base rule is for the BELOW range only).
        (when (and (not solo) (not blits-below))
          (blit-layers! blit quad gen-vao buf-tex tf-buf layers active nlayers false vw vh iw ih))
        (gl/gl-disable gl/GL-SCISSOR-TEST)
        (gl/gl-disable gl/GL-BLEND)))
    {:count count :total total :n n}))

(defn- present-pass!
  "Draw one full-frame present pass (`prog-key` = :sharpen or :aa) from `src-tex`
   into the currently-bound framebuffer, at iw*ih. The target IS the image here —
   the letterbox exists only in the pane blit — so the clamp rect is the whole
   texture. Both passes share vs-src-sharpen's geometry and uniform names, so this
   is one body rather than two that must be kept in step."
  [area iw ih src-tex amount prog-key]
  (let [gpu     (ensure-gpu! area)
        prog    (get gpu prog-key)
        gen-vao (:gen-vao gpu)
        locs    (:locs prog)]
    (gl/gl-viewport 0 0 (int iw) (int ih))
    (gl/gl-clear-color 0.0 0.0 0.0 1.0)
    (gl/gl-clear gl/GL-COLOR-BUFFER-BIT)
    (gl/gl-disable gl/GL-BLEND)
    (gl/gl-use-program (:program prog))
    (gl/gl-active-texture gl/GL-TEXTURE0)
    (gl/gl-bind-texture gl/GL-TEXTURE-2D src-tex)
    (gl/gl-uniform-1i (:u_src locs) 0)
    (gl/gl-uniform-2f (:u_viewport locs) (double iw) (double ih))
    (gl/gl-uniform-4f (:u_rect locs) 0.0 0.0 (double iw) (double ih))
    (gl/gl-uniform-1f (:u_amount locs) (double amount))
    ;; SHARPEN also gates on absolute subjectness, so the dial spends itself on
    ;; foreground detail instead of crunching bokeh and canvas texture — the same
    ;; field the generation shader gates hardness on. The AA pass shares this body
    ;; and has no such uniform, so this is conditional on the location existing.
    (when-let [subj-loc (:u_subjTex locs)]
      (when (>= (long subj-loc) 0)
        (let [fields (:fields gpu)]
          (gl/gl-active-texture (+ gl/GL-TEXTURE0 1))
          (gl/gl-bind-texture gl/GL-TEXTURE-2D (:subject fields))
          (gl/gl-uniform-1i subj-loc 1)
          (gl/gl-uniform-2f (:u_detailDim locs)
                            (double (nth (:detail-dim fields) 0))
                            (double (nth (:detail-dim fields) 1)))
          (gl/gl-uniform-2f (:u_detailSrc locs)
                            (double (nth (:detail-src fields) 0))
                            (double (nth (:detail-src fields) 1)))
          (gl/gl-active-texture gl/GL-TEXTURE0))))
    (gl/gl-bind-vertex-array gen-vao)            ; no attribs — gl_VertexID only
    (gl/gl-draw-arrays gl/GL-TRIANGLES 0 6)))

(defn- render-final!
  "Render the finished picture at NATIVE IMAGE RESOLUTION into a texture; returns
   [tex result] and leaves that texture's FBO bound, so a caller can read it back
   directly. THE definition of what a render is: the on-screen pane and the saved
   PNG both come from here, so what you are looking at is what you will get.

   Before this, the pane composited at WIDGET resolution — 1766x1488 device px
   against a 1024x1024 image on a Retina display. The splats were rasterized at a
   different scale than the export, and a final-output pass ran at a different
   physical radius, so nothing judged on screen was the thing written to disk."
  [area iw ih]
  (let [{[cfbo ctex] :composite [pfbo ptex] :post} (ensure-render-targets! area iw ih)
        sharpen (double (cur-sharpen))
        aa      (double (cur-aa))]
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER cfbo)
    (gl/gl-viewport 0 0 (int iw) (int ih))
    (let [result (gpu-draw! area iw ih iw ih)
          ;; PRESENT-PASS ORDER: sharpen, then antialias. Sharpen is gated on local
          ;; gradient so it targets silhouettes, and a painted silhouette is scalloped
          ;; at sub-pixel scale by construction (overlapping strokes at slightly
          ;; different angles) — amplifying that is the staircase. Antialias smooths
          ;; ALONG the edge afterwards, so it cleans up after sharpen instead of being
          ;; undone by it. The per-splat hardness easing cannot do this job: it runs
          ;; during rasterization, before the composite exists.
          ;; Each pass ping-pongs between the two targets, so no third one is needed.
          [tex fbo] (if (pos? sharpen)
                      (do (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER pfbo)
                          (present-pass! area iw ih ctex sharpen :sharpen)
                          [ptex cfbo])
                      [ctex pfbo])]
      (if (pos? aa)
        (do (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER fbo)
            (present-pass! area iw ih tex aa :aa)
            [(if (= fbo cfbo) ctex ptex) result])
        [tex result]))))

(defn- blit-to-pane!
  "Resample the finished native-resolution texture into the currently-bound
   framebuffer, letterboxed into a vw*vh pane. The layer blit program at full
   opacity with blending OFF — a straight resample, not a composite."
  [area tex vw vh iw ih]
  (let [{:keys [blit gen-vao]} (ensure-gpu! area)
        locs (:locs blit)]
    (gl/gl-viewport 0 0 (int vw) (int vh))
    (gl/gl-clear-color 0.0 0.0 0.0 1.0)
    (gl/gl-clear gl/GL-COLOR-BUFFER-BIT)
    (gl/gl-disable gl/GL-BLEND)
    (gl/gl-use-program (:program blit))
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 9))
    (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
    (gl/gl-uniform-1i (:u_layer locs) 9)
    (gl/gl-uniform-1f (:u_alpha locs) 1.0)
    (gl/gl-uniform-2f (:u_viewport locs) (double vw) (double vh))
    (gl/gl-uniform-2f (:u_image locs) (double iw) (double ih))
    (gl/gl-bind-vertex-array gen-vao)            ; no attribs — gl_VertexID only
    (gl/gl-draw-arrays gl/GL-TRIANGLES 0 6)
    (gl/gl-active-texture gl/GL-TEXTURE0)))

(defn- splat-stats [splats]
  (reduce (fn [[sx sy sd sc sa] {[mx my] :mean [c00 c01 _ c11] :cov [r g b] :color a :alpha}]
            [(+ sx mx) (+ sy my) (+ sd (- (* c00 c11) (* c01 c01))) (+ sc r g b)
             (+ sa (double (or a 1.0)))])
          [0.0 0.0 0.0 0.0 0.0] splats))

(defn- gpu-verify! [area tf-buf n]
  ;; numerical m11 check: compare the GPU-generated field to the CPU golden reference
  ;; for the current controls (aggregate Σmean/Σdet/Σcolour — exact match is impossible
  ;; across double↔float + Perlin/hash, so we report both for a closeness read).
  ;;
  ;; THIS ONLY PRINTS. It asserts nothing, and neither does anything else: gpu-fields-test
  ;; compares FIELDS, never the splat field, and check.clj pins GLSL source text. Tracer
  ;; parity is a convention, not a gate — do not read a count ratio here as pass/fail.
  ;;
  ;; TOLERATED DIVERGENCE (splat-painter-9wx, characterized 2026-08-01, deliberately not
  ;; fixed). At a dying ridge, edge-snap's Newton step has a degenerate denominator
  ;; (den 0.0046 vs numerator 0.0091 — the raw position lands within 0.02px of the probe
  ;; midpoint); float32-vs-float64 drift flips its sign, the step saturates at ±1.0, and
  ;; the paths pull ~2.98px in OPPOSITE directions. The CPU is the one misbehaving; the
  ;; GPU, which is also the shipping path, pulls the right way.
  ;;
  ;; It shows up as a COUNT RATIO ABOVE 100% and is masked by the band tier's side push,
  ;; which keeps band strokes off the ridge where the razor lives. As shipped that leaves
  ;; parity at 100.0% (425709/425696). Reduce or remove the push — which splat-painter-d37
  ;; does — and it opens to single-digit percent (121.4% with the push gone entirely).
  ;; That is expected, not a regression. Every fix tried changed the render for a defect
  ;; that never reaches the screen (+5.5% to +13% splats), which is why it stands.
  (let [gpu (gen/read-splats tf-buf n)
        cpu (:splats (field-for-current-controls))
        [gx gy gd gc ga] (splat-stats gpu)
        [cx cy cd cc ca] (splat-stats cpu)]
    (println (format "gpu-verify: count GPU %d / CPU %d  (%.1f%%)" n (count cpu)
                     (* 100.0 (/ (double n) (max 1 (count cpu))))))
    (println (format "  Σmean-x  GPU %.1f  CPU %.1f" gx cx))
    (println (format "  Σmean-y  GPU %.1f  CPU %.1f" gy cy))
    (println (format "  Σdet     GPU %.1f  CPU %.1f" gd cd))
    (println (format "  Σcolour  GPU %.2f  CPU %.2f" gc cc))
    (println (format "  Σalpha   GPU %.2f  CPU %.2f" ga ca))
    ;; temp diagnostic: locate the first record where the two fields diverge
    (when (System/getenv "GA_PAINTER_GPU_DIAG")
      (let [m (min (count gpu) (count cpu))
            di (loop [i 0]
                 (if (>= i m)
                   nil
                   (let [[gx1 gy1] (:mean (nth gpu i)) [cx1 cy1] (:mean (nth cpu i))]
                     (if (or (> (Math/abs (- gx1 cx1)) 0.5) (> (Math/abs (- gy1 cy1)) 0.5))
                       i (recur (inc i))))))]
        (println "  first-divergence idx:" di "of" m)
        (when (and (nil? di) (< (count gpu) (count cpu)))
          (let [g (nth gpu (dec (count gpu))) c (nth cpu (count gpu))
                sig (fn [{[c00 c01 _ c11] :cov}] (Math/sqrt (Math/sqrt (max 1e-8 (- (* c00 c11) (* c01 c01))))))]
            (println (format "  last GPU rec: mean %s sig %.2f | first MISSING cpu rec: mean %s sig %.2f"
                             (pr-str (mapv #(format "%.0f" (double %)) (:mean g))) (sig g)
                             (pr-str (mapv #(format "%.0f" (double %)) (:mean c))) (sig c)))))
        (when di
          (doseq [k (range (max 0 (- di 1)) (min m (+ di 3)))]
            (println (format "   [%d] GPU %s a=%.2f | CPU %s a=%.2f" k
                             (pr-str (mapv #(format "%.1f" (double %)) (:mean (nth gpu k))))
                             (double (:alpha (nth gpu k)))
                             (pr-str (mapv #(format "%.1f" (double %)) (:mean (nth cpu k))))
                             (double (or (:alpha (nth cpu k)) 1.0))))))))))

(defn- save-rgba-jolt! [buf iw ih path]
  ;; buf = iw*ih*4 RGBA bytes, bottom-up (glReadPixels); write top-down via the native
  ;; jolt.png encoder (no gdk-pixbuf — its from-data path faults post-GL here).
  (let [pimg (jolt.png/image iw ih)]
    (dotimes [r ih]
      (let [src (* 4 (* (- ih 1 r) iw))]
        (dotimes [c iw]
          (let [o (+ src (* 4 c))]
            (jolt.png/put! pimg
                           (double (ffi/read buf :uint8 o))
                           (double (ffi/read buf :uint8 (+ o 1)))
                           (double (ffi/read buf :uint8 (+ o 2))))))))
    (jolt.png/write pimg iw ih path)))

(defn rgba->image
  "Convert an RGBA byte buffer (as glReadPixels returns it — origin BOTTOM-left,
  iw*ih*4 bytes) into an image map {:height ih :width iw :channels 3 :pixels} whose
  :pixels is a flat H*W*3 double-array, row-major top-down, values 0..1. Image row x
  (top-down) is read from buffer row (ih-1-x) — the bottom-up flip — and channels RGB
  are taken from the RGBA texels and divided by 255. `buf` is a raw pointer read with
  the unsigned :uint8 byte API, so a byte of 200 decodes to 200/255 (not -56). Pure,
  no GL. add-layer! feeds the captured render through this into prepare-image."
  [buf iw ih]
  (let [pixels (double-array (* ih iw 3))]
    (dotimes [x ih]
      (let [src (* 4 (* (- ih 1 x) iw))]   ; buffer row (ih-1-x) = top-down image row x
        (dotimes [y iw]
          (let [o    (+ src (* 4 y))
                base (* 3 (+ (* x iw) y))] ; flat base 3*(row*W + col)
            (aset-double pixels base       (/ (double (ffi/read buf :uint8 o))       255.0))
            (aset-double pixels (+ base 1) (/ (double (ffi/read buf :uint8 (+ o 1))) 255.0))
            (aset-double pixels (+ base 2) (/ (double (ffi/read buf :uint8 (+ o 2))) 255.0))))))
    {:height ih :width iw :channels 3 :pixels pixels}))

(defn- gpu-save-png! [area path]
  (when-let [img @image-atom]
    (let [iw (int (:width img)) ih (int (:height img))
          prev-fbo (read-fbo-binding)]
      ;; render-final! leaves ITS framebuffer bound and the picture is already at
      ;; native resolution, so the readback comes straight off it — no separate
      ;; export target, and by construction the same pixels the pane is showing.
      (let [[_tex {:keys [count total n]}] (render-final! area iw ih)
            buf (ffi/alloc (* iw ih 4))]
        (println (format "gpu-save: %d candidates -> %d survivors (render %d)" total count n))
        (when (System/getenv "GA_PAINTER_GPU_VERIFY")   ; recomputes the CPU field — dev only
          (gpu-verify! area (:tf-buf (:gpu (get @gl-state area))) n))
        (gl/gl-read-pixels 0 0 iw ih gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
        (save-rgba-jolt! buf iw ih path)
        (println "gpu-save: wrote" path)
        (ffi/free buf))
      (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev-fbo)
      (let [[w h] @viewport] (gl/gl-viewport 0 0 w h)))))

;; --- layer stack: pure helpers (testable, no GL/atoms beyond the slider reset) ---
;; A committed layer is {:tex <GL id> :opacity <double> :settings <14-vector>}. The
;; index convention (documented here, the single source of truth): with N committed
;; layers and active in [0,N], the LIVE pass sits between layers[active-1] and
;; layers[active] — compositing blits layers[0..active) UNDER it and
;; layers[active..N) OVER it. insert-at-active keeps that invariant on commit.

(def ^:private settings-atoms
  "The slider atoms a committed layer must snapshot so re-selecting it reproduces the
   pass deterministically. Positional — snapshot-settings / restore-settings! use the
   same order. (count size broad mid fine detail variation curvature stroke contrast
   hardness aa cutin clip swirl tex-streak tex-grain tex-edge.)"
  [count-atom size-atom broad-atom mid-atom fine-atom detail-atom variation-atom
   curvature-atom stroke-atom contrast-atom hardness-atom aa-atom cutin-atom clip-atom swirl-atom
   tex-streak-atom tex-grain-atom tex-edge-atom])

(defn snapshot-settings
  "Snapshot every placement slider atom into a 17-vector. Stored on a committed layer
   so re-selecting it restores the exact field that generated it."
  []
  (vec (for [a settings-atoms] @a)))

(defn restore-settings!
  "Reset every placement slider atom from a snapshot-settings vector."
  [snap]
  (dotimes [i (count settings-atoms)]
    (reset! (settings-atoms i) (snap i))))

(defn insert-layer
  "Insert entry into the committed-layers vector at index `active` (append when active
   is past the end, else splice in place). Pure — no GL, no atoms."
  [layers active entry]
  (if (= active (count layers))
    (conj layers entry)
    (vec (concat (subvec layers 0 active) [entry] (subvec layers active)))))

(defn remove-layer
  "Drop the entry at index j from the committed-layers vector. Pure."
  [layers j]
  (vec (concat (subvec layers 0 j) (subvec layers (inc j)))))

(defn- delete-layer-textures!
  "Free the GL texture objects of `entries` (committed layer maps). Caller must have
   made the GL context current."
  [entries]
  (doseq [{:keys [tex]} entries :when (and tex (pos? (long tex)))]
    (let [p (gl/write-ints [(int tex)])]
      (gl-delete-textures 1 p)
      (ffi/free p))))

(defn- clear-layers!
  "Drop every committed layer, freeing its GL texture, and reset the active index to 0.
   No-ops on the GL side when no GLArea is realized, so headless callers are safe."
  []
  (when-let [a @area-atom]
    (glx/make-current a)
    (delete-layer-textures! @layers-atom))
  (reset! layers-atom [])
  (reset! active-layer-atom 0))

(defn- commit-active!
  "Capture the live splat pass ALONE (:solo — transparent clear, no blits, splats at the
   fixed 0.9) offscreen at native res, upload it to a fresh RGBA8/LINEAR/CLAMP texture,
   and insert {:tex :opacity :settings} into layers-atom at the active index (append
   when active is a new top layer). The :settings snapshot reproduces the pass from
   image-atom + the sliders. GUI-thread only (GL context)."
  []
  (let [area    @area-atom
        img     @image-atom
        iw      (int (:width img)) ih (int (:height img))
        prev-fbo (read-fbo-binding)]
    (glx/make-current area)
    (ensure-export-targets! area iw ih)
    (gl/gl-viewport 0 0 iw ih)
    (gpu-draw! area iw ih iw ih {:solo true})
    (let [buf (ffi/alloc (* iw ih 4))
          tex (gl/gen-one gl/gl-gen-textures)]
      (gl/gl-read-pixels 0 0 iw ih gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
      (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
      (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 GL-RGBA8 iw ih 0 gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
      (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-LINEAR)
      (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-LINEAR)
      (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-S gl/GL-CLAMP-TO-EDGE)
      (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-T gl/GL-CLAMP-TO-EDGE)
      (ffi/free buf)
      (let [active (long @active-layer-atom)
            entry  {:tex      tex
                    :opacity  (if (zero? active) 1.0 (cur-layer-opacity))
                    :settings (snapshot-settings)}]
        (swap! layers-atom insert-layer active entry)))
    (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev-fbo)
    (let [[w h] @viewport] (gl/gl-viewport 0 0 w h))))

(defn add-layer!
  "Commit the live pass into the stack (commit-active!), then capture the FULL composite
   (a normal non-solo 1:1 draw) and make it the next pass's source image — so repainting
   smooths the shapes while the per-layer opacity lets the pass below show through like
   a glaze. Active moves to the new top; sliders are left as-is. GUI-thread only (GL
   context); also the headless multi-pass driver (GA_PAINTER_PASSES)."
  []
  (if-not (and @image-atom @area-atom (get @gl-state @area-atom))
    (reset! status-atom "load an image first")
    (let [area     @area-atom
          img      @image-atom
          iw       (int (:width img)) ih (int (:height img))
          prev-fbo (read-fbo-binding)]
      (glx/make-current area)
      (ensure-export-targets! area iw ih)
      (gl/gl-viewport 0 0 iw ih)
      (gpu-draw! area iw ih iw ih)               ; the full composite (below + live + above — live drawn ONCE)
      (let [buf  (ffi/alloc (* iw ih 4))
            _    (gl/gl-read-pixels 0 0 iw ih gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
            img' (rgba->image buf iw ih)]
        (ffi/free buf)
        (commit-active!)                         ; snapshot the live pass (needs the OLD image fields)
        (reset! image-atom img'))
      (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev-fbo)
      (let [[w h] @viewport] (gl/gl-viewport 0 0 w h))
      (reset! active-layer-atom (count @layers-atom))
      (reset! status-atom (format "layer %d — repainting from render" (inc (count @layers-atom))))
      (request-render!))))

(defn select-layer!
  "Make committed layer j (0-indexed) the active layer to re-edit. Saves the current
   live pass back into the stack (commit-active!), restores layers[j]'s settings +
   opacity, removes its stored solo render (the live regen with restored settings
   reproduces it), and rebuilds the live pass's source image — the composite of layers
   below j (or the original file when j == 0). No-op if j == active or j is out of
   [0, (count layers)]. GUI-thread only (GL context)."
  [j]
  (let [layers @layers-atom
        active (long @active-layer-atom)]
    (when (and (integer? j) (<= 0 j (count layers)) (not= j active))
      (let [area @area-atom]
        (glx/make-current area)
        ;; 1. save the current live pass + its settings back into the stack. The live is
        ;;    committed at `active`, so indices >= active shift up one — the entry at the
        ;;    post-commit index j is the layer to make live.
        (commit-active!)
        ;; 2. layers[j] becomes the live pass: restore its settings + opacity, drop it.
        (let [target (@layers-atom j)
              {:keys [settings opacity]} target]
          (restore-settings! settings)
          (reset! layer-opacity-atom (or opacity 0.6))
          (swap! layers-atom remove-layer j)
          (delete-layer-textures! [target])   ; free the dropped layer's GL texture
          (reset! active-layer-atom j)
          ;; 3. rebuild the live pass's SOURCE image = everything painted below it.
          (if (zero? j)
            ;; bottom layer: source is the original file (reload fields, but do NOT touch
            ;; the slider atoms — on-image-loaded would reset size-atom).
            (when @image-path-atom
              (reset! image-atom (image/load-image @image-path-atom 1024)))
            ;; else: composite the layers below j over opaque black, read it back.
            (let [iw      (int (:width @image-atom)) ih (int (:height @image-atom))
                  prev-fbo (read-fbo-binding)]
              (ensure-export-targets! area iw ih)
              (gl/gl-viewport 0 0 iw ih)
              (gpu-draw! area iw ih iw ih {:blits-below true})
              (let [buf (ffi/alloc (* iw ih 4))]
                (gl/gl-read-pixels 0 0 iw ih gl/GL-RGBA gl/GL-UNSIGNED-BYTE buf)
                (reset! image-atom (rgba->image buf iw ih))
                (ffi/free buf))
              (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER prev-fbo)
              (let [[w h] @viewport] (gl/gl-viewport 0 0 w h))))
          (reset! status-atom (format "layer %d of %d (editing)" (inc j) (inc (count @layers-atom))))
          (request-render!))))))

(defn- reset-layers!
  "Drop every committed layer (freeing each layer's GL texture), reset the active index
   to 0, and reload the original image from disk (rebuilt fields, fresh single-pass
   render)."
  []
  (if (empty? @layers-atom)
    (reset! status-atom "no layers to reset")
    (do (clear-layers!)
        (when @image-path-atom (on-image-loaded @image-path-atom)))))

(defn on-realize [area]
  (reset! area-atom area)
  (glx/make-current area)
  (when-let [err (glx/gl-area-error-message area)]
    (println "GLArea context error:" err))
  ;; every GL object is built lazily by ensure-gpu! on the first draw; this just
  ;; registers the area so the render/save paths know the context exists.
  (swap! gl-state assoc area {})
  (println "splat-painter: GL ready"))

(defn on-resize [_area w h]
  (reset! viewport [w h])
  (gl/gl-viewport 0 0 w h))

(defn on-render [area]
  (when (and (System/getenv "GA_PAINTER_TF_SMOKE") (not @saved?-atom))
    (reset! saved?-atom true)
    (require 'splat-painter.tf-smoke)
    ((resolve 'splat-painter.tf-smoke/run!)))
  (when-let [_st (get @gl-state area)]
    (let [[w h] @viewport
          img   @image-atom]
      (if-not img
        (do (gl/gl-clear-color 0.05 0.06 0.09 1.0) (gl/gl-clear gl/GL-COLOR-BUFFER-BIT))
        ;; render at native resolution, then resample into the pane. The pane is a
        ;; VIEWER of the finished picture, not a second renderer at another scale.
        (let [iw (int (:width img)) ih (int (:height img))
              pane-fbo (read-fbo-binding)
              [tex _]  (render-final! area iw ih)]
          (gl/gl-bind-framebuffer gl/GL-FRAMEBUFFER pane-fbo)
          (blit-to-pane! area tex w h iw ih)))
      (when-let [p (System/getenv "GA_PAINTER_SAVE_PNG")]
        (when (and (not @saved?-atom) img)
          (reset! saved?-atom true)
          ;; layered repainting: compose N passes before saving. Each add-layer!
          ;; captures the current composite offscreen and makes it the next pass's
          ;; source, so (dec passes) of them stack under the final gpu-save-png!.
          ;; GA_PAINTER_PASSES unset/1 == the original single-pass save (bit-identical).
          (let [passes (or (some-> (System/getenv "GA_PAINTER_PASSES") Integer/parseInt) 1)]
            (dotimes [_ (dec passes)] (add-layer!)))
          (gpu-save-png! area p))))))

;; --- reactive control panel --------------------------------------------------
;; Layout rule that keeps the sidebar narrow: NO widget inside the sidebar ever
;; sets :hexpand. In GTK a GtkBox reports itself as "wanting to expand" to its
;; parent whenever any child expands (gtk_widget_compute_expand propagates up the
;; tree). So a :scale with :hexpand true made the control-panel :vbox compete
;; with the :gl-area for the content row's free width — splitting the window and
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

(defn- toolbar []
  ;; the action buttons + status live in a bar ACROSS THE TOP: stacking them in
  ;; the sidebar forced its width to the four-button row (~2× the slider column),
  ;; stealing canvas from the render. The status label ellipsizes instead of
  ;; expanding, so the :gl-area stays the tree's only :hexpand widget (see the
  ;; layout rule above) and long paths can't stretch the window.
  [:hbox {:spacing 6 :margin 6}
   [:button {:label "Open Image…"  :on-click open-image-dialog!}]
   [:button {:label "Save PNG…"    :on-click save-image-dialog!}]
   [:button {:label "Add Layer"    :on-click add-layer!}]
   [:button {:label "Reset Layers" :on-click reset-layers!}]
   [:button {:label "◀" :on-click #(select-layer! (dec @active-layer-atom))}]
   [:label {:label (format "layer %d/%d" (inc @active-layer-atom) (inc (count @layers-atom)))}]
   [:button {:label "▶" :on-click #(select-layer! (inc @active-layer-atom))}]
   [:label {:label @status-atom :xalign 0.0 :halign :start :ellipsize :end}]])

(defn- control-panel []
  [:vbox {:spacing 6 :margin 8 :width-request 160}
   [slider "Splats"    1000 600000 1000 count-atom]     ; budget: higher = more detail, slower
   [slider "Size"      6    50    0.5   size-atom]
   [slider "Broad"     0.4  2.5   0.05  broad-atom]    ; bokeh: low-detail looseness, subjects untouched
   [slider "Mid"       0.4  2.5   0.05  mid-atom]      ; mid-structure size
   [slider "Fine"      0.4  2.5   0.05  fine-atom]     ; finest-detail size (lower = tighter)
   [slider "Detail"    0.0  1.0   0.02  detail-atom]
   [slider "Variation" 0.0  1.0   0.02  variation-atom]
   [slider "Curvature" 0.0  1.0   0.02  curvature-atom]
   [slider "Stroke"    1.0  4.0   0.05  stroke-atom]   ; <1 degenerates chains to bead dots
   [slider "Contrast"  0.5  2.0   0.05  contrast-atom]
   [slider "Hardness"  1.0  4.0   0.05  hardness-atom]   ; detail-stroke crispness (big strokes stay round)
   [slider "Antialias" 0.0  1.0   0.05  aa-atom]        ; smooths along edges AFTER Sharpen (kills the staircase); 0 = off
   [slider "Sharpen"   0.0  1.5   0.05  sharpen-atom]    ; final-output edge-gated unsharp; 0 = off
   [slider "Cut-in"    0.0  1.5   0.05  cutin-atom]     ; edge-band tier: restate silhouettes from their own sides; 0 = off
   [slider "EdgeClip"  0.0  1.0   0.05  clip-atom]     ; stroke tails stop at a silhouette instead of feathering through; 0 = off
   [slider "Swirl"     0.0  1.0   0.02  swirl-atom]    ; Perlin share of the flow + position warp; 0 = image structure only
   [:separator {}]
   [slider "Streak"    0.0  0.6   0.02  tex-streak-atom]  ; bristle tonal grooves along the drag
   [slider "Grain"     0.0  0.5   0.02  tex-grain-atom]   ; canvas-tooth brightness/colour mottle
   [slider "EdgeTex"   0.0  0.7   0.02  tex-edge-atom]  ; edge raggedness (contour breaks up)
   [:separator {}]
   [slider "Layer Op"  0.05 1.0   0.05  layer-opacity-atom]]) ; glaze: how much the pass below shows through

(defn app []
  [:vbox {:spacing 0}
   [toolbar]
   [:separator {}]
   [:hbox {:spacing 0}
    [control-panel]
    [:separator {:orientation :vertical}]
    [:gl-area {:version [3 2] :depth-buffer false :hexpand true :vexpand true
               :on-realize on-realize
               :on-render  on-render
               :on-resize  on-resize}]]])

(defn -main [& args]
  ;; GLib must see the GSettings schemas BEFORE GTK initializes, or the file
  ;; dialogs hard-abort the process on machines where XDG_DATA_DIRS is unset.
  (ensure-schema-path!)
  ;; an optional image path on the command line loads immediately — handy for
  ;; smoke-testing the full render path without clicking the dialog
  (when (seq args) (on-image-loaded (first args)))
  (let [quit-ms (some-> (System/getenv "GA_PAINTER_QUIT_MS") Integer/parseInt)]
    (apply ui/run app
           :app-id "dev.jolt.splat-painter"
           :title  "splat-painter • gaussian splats"
           :width  1100 :height 720
           (when quit-ms [:auto-quit-ms quit-ms]))))
