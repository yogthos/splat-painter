(ns splat-painter.gzip
  "gzip output, via zlib's gz* stream API. A `.svgz` IS a gzip-wrapped SVG — every
   browser and Inkscape open one directly — and on this exporter's output that is worth
   about 7×, far more than any encoding trick left in the file. The SVG is one long
   stream of near-identical elements, which is the best case DEFLATE has.

   zlib rather than assembling the container by hand: `compress2` would give a ZLIB
   stream (RFC 1950), not a GZIP one (RFC 1952), and converting between them means
   stripping two header bytes, dropping the Adler-32 trailer, and appending a CRC-32
   and length of your own. `gzopen`/`gzwrite`/`gzclose` write the real thing.

   zlib needs no declaring in deps.edn: its symbols resolve in the bare runtime with no
   GTK loaded (checked), and the app already depends on it transitively through
   gdk-pixbuf → libpng."
  (:require [jolt.ffi :as ffi]))

(ffi/defcfn z-gzopen  "gzopen"  [:string :string] :pointer)
(ffi/defcfn z-gzwrite "gzwrite" [:pointer :pointer :uint] :int)
(ffi/defcfn z-gzclose "gzclose" [:pointer] :int)

(def ^:private max-write
  "gzwrite takes an unsigned int. A document this size is not a thing here — the
   biggest painting measured is 40 MB — so this is an assertion, not a chunking loop."
  0x7fffffff)

(defn spit-gz!
  "Write `s` to `path` as a gzip stream at maximum compression. Returns the number of
   UNCOMPRESSED bytes written; the file on disk is the deflate of those."
  [path s]
  (let [n (alength (.getBytes ^String s "UTF-8"))]
    (when (>= n max-write)
      (throw (ex-info "document too large for a single gzwrite" {:bytes n :path path})))
    ;; own the buffer rather than taking string->ptr's, so the free is ours to make.
    ;; +1 for the NUL write-bytes lands after the payload.
    (let [buf (ffi/alloc (inc n))]
      (try
        (ffi/write-bytes buf s)
        (let [f (z-gzopen path "wb9")]
          (when (ffi/null? f)
            (throw (ex-info "could not open for writing" {:path path})))
          (let [wrote (z-gzwrite f buf n)
                rc    (z-gzclose f)]      ; close flushes, so its status matters too
            (when (or (not= wrote n) (not= rc 0))
              (throw (ex-info "gzip write failed" {:path path :wrote wrote :want n :close rc})))))
        (finally (ffi/free buf)))
      n)))
