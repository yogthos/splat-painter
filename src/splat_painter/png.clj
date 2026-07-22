(ns splat-painter.png
  "Encode an RGBA pixel buffer (as produced by glReadPixels, origin bottom-left)
   to a PNG file via gdk-pixbuf. gdk_pixbuf_flip corrects the vertical origin flip:
   gdk-pixbuf treats row 0 as the top, glReadPixels produces row 0 as the bottom."
  (:require [jolt.ffi :as ffi]))

;; GError layout on 64-bit: { guint32 domain; gint code; gchar* message; }.
;; The message pointer is 8-byte aligned, so it sits at byte offset 16.
(def ^:private gerror-msg-off 16)

;; (data, colorspace=0=RGB, has_alpha=1, bits_per_sample=8, width, height, rowstride, destroy_fn, destroy_data)
(ffi/defcfn pixbuf-new-from-data "gdk_pixbuf_new_from_data"
  [:pointer :int :int :int :int :int :int :pointer :pointer] :pointer)
;; (pixbuf, horizontal) -> new pixbuf; horizontal=FALSE (0) => vertical flip
(ffi/defcfn pixbuf-flip  "gdk_pixbuf_flip"  [:pointer :int] :pointer)
;; (pb, filename, type, keys, vals, err) -> gboolean
(ffi/defcfn pixbuf-savev "gdk_pixbuf_savev"
  [:pointer :string :string :pointer :pointer :pointer] :int)
(ffi/defcfn png-object-unref "g_object_unref" [:pointer] :void)
(ffi/defcfn png-error-free   "g_error_free"   [:pointer] :void)

(defn- read-gerror
  "If `slot` holds a non-null GError*, read and free its message; else nil."
  [slot]
  (when-let [err (ffi/read slot :pointer 0)]
    (let [msg-ptr (ffi/read err :pointer gerror-msg-off)
          msg (when-not (ffi/null? msg-ptr) (ffi/ptr->string msg-ptr))]
      (png-error-free err)
      msg)))

(defn save-rgba-bottom-up!
  "Write `buf` (a raw pointer to iw*ih*4 RGBA bytes, origin bottom-left as from
   glReadPixels) to `path` as PNG. Does NOT free `buf` (caller owns it).
   Throws ex-info on failure."
  [buf iw ih path]
  (let [pb (pixbuf-new-from-data buf 0 1 8 (int iw) (int ih) (int (* iw 4))
                                 ffi/null ffi/null)]
    (when (ffi/null? pb)
      (throw (ex-info "png: gdk_pixbuf_new_from_data returned NULL" {})))
    (let [pb2     (pixbuf-flip pb 0)  ; vertical flip (horizontal=FALSE=0)
          errslot (ffi/alloc (ffi/sizeof :pointer))
          ok      (pixbuf-savev pb2 path "png" ffi/null ffi/null errslot)]
      (let [msg (or (read-gerror errslot) "unknown")]
        (png-object-unref pb2)
        (png-object-unref pb)
        (ffi/free errslot)
        (when (zero? ok)
          (throw (ex-info (str "png: save failed: " msg) {:path path})))))))
