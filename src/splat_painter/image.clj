(ns splat-painter.image
  "Decode an image file to a flat double pixel buffer via gdk-pixbuf, with an
   optional downscale. This is the load+resize fit.py does with PIL, here via
   gdk_pixbuf_new_from_file_at_scale. Returns {:height :width :channels :pixels}
   in the flat H*W*3 0..1-double shape splat-painter.seed consumes."
  (:require [jolt.ffi :as ffi]))

;; GError layout on 64-bit: { guint32 domain; gint code; gchar* message; }.
;; The message pointer is 8-byte aligned, so it sits at byte offset 16.
(def ^:private gerror-msg-off 16)

(ffi/defcfn pixbuf-from-file-at-scale
  "gdk_pixbuf_new_from_file_at_scale" [:string :int :int :int :pointer] :pointer)
(ffi/defcfn pixbuf-width      "gdk_pixbuf_get_width"      [:pointer] :int)
(ffi/defcfn pixbuf-height     "gdk_pixbuf_get_height"     [:pointer] :int)
(ffi/defcfn pixbuf-channels   "gdk_pixbuf_get_n_channels" [:pointer] :int)
(ffi/defcfn pixbuf-rowstride  "gdk_pixbuf_get_rowstride"  [:pointer] :int)
(ffi/defcfn pixbuf-pixels-len "gdk_pixbuf_get_pixels_with_length" [:pointer :pointer] :pointer)
(ffi/defcfn g-object-unref    "g_object_unref" [:pointer] :void)
(ffi/defcfn g-error-free      "g_error_free"   [:pointer] :void)

(defn- read-gerror
  "If `slot` holds a non-null GError*, read and free its message; else nil."
  [slot]
  (when-let [err (ffi/read slot :pointer 0)]
    (let [msg-ptr (ffi/read err :pointer gerror-msg-off)
          msg (when-not (ffi/null? msg-ptr) (ffi/ptr->string msg-ptr))]
      (g-error-free err)
      msg)))

(defn load-image
  "Load `path` (JPEG/PNG/...) into a flat 0..1 double pixel buffer. If `max-side`
   is given, the longest edge is scaled to it with aspect preserved. Returns
   {:height :width :channels :pixels}. Throws on a decode error."
  ([path] (load-image path nil))
  ([path max-side]
   (let [errslot (ffi/alloc (ffi/sizeof :pointer))
         ;; With preserve_aspect=1, passing max-side for both axes fits the
         ;; image in a max-side box so the longest edge becomes max-side.
         ;; -1 means 'no constraint' → original size.
         [tw th] (if max-side [(int max-side) (int max-side)] [-1 -1])
         pb (pixbuf-from-file-at-scale path tw th 1 errslot)]
     (if (ffi/null? pb)
       (let [msg (or (read-gerror errslot) "unknown error")]
         (ffi/free errslot)
         (throw (ex-info (str "image: failed to load " path ": " msg) {:path path})))
       (let [w  (pixbuf-width pb)
             h  (pixbuf-height pb)
             ch (pixbuf-channels pb)
             rs (pixbuf-rowstride pb)
             lenp (ffi/alloc (ffi/sizeof :int))
             px   (pixbuf-pixels-len pb lenp)
             _    (ffi/read lenp :int 0)
             pixels (double-array (* h w 3))]
         ;; Copy row-by-row honoring rowstride (rows may be padded), normalize
         ;; 8-bit -> 0..1, and force 3 channels (gray is replicated, alpha dropped).
         (loop [y 0 dst 0]
           (when (< y h)
             (let [row-off (* y rs)]
               (loop [x 0 d dst]
                 (when (< x w)
                   (let [src (+ row-off (* x ch))]
                     (dotimes [k 3]
                       (let [c (ffi/read px :uint8 (+ src (min k (dec ch))))]
                         (aset pixels (+ d k) (/ (double c) 255.0)))))
                   (recur (inc x) (+ d 3)))))
             (recur (inc y) (+ dst (* w 3)))))
         (g-object-unref pb)
         (ffi/free lenp)
         (ffi/free errslot)
         ;; :pixels stays a Java double-array (not a persistent vector): the seed
         ;; and structure passes read it via aget in tight per-pixel loops, and a
         ;; 3M-element persistent vector under nth is orders of magnitude slower.
         {:height h :width w :channels 3 :pixels pixels})))))
