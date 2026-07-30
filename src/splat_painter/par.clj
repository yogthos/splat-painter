(ns splat-painter.par
  "Band-parallel loops for the per-image field builders.

   Loading an image runs six field builders over every pixel and used to do it on
   one core: 12.6s on a 1MP photo, all of it before the first stroke is painted.
   Every one of those loops is a disjoint write per index, so splitting the index
   range into contiguous bands across threads is a pure scheduling change — each
   band reads shared source arrays and writes only its own slice, so the result is
   identical to the serial loop, byte for byte. splat-painter.fields-test pins that.

   NOTE: (.availableProcessors (Runtime/getRuntime)) reports 1 under jolt even on a
   10-core machine, so it cannot size the pool — futures really are parallel (a 4×
   workload measured 303ms across futures vs 683ms serially), the count is just
   wrong. Hence the fallback below.")

(def threads
  "Bands to split a parallel loop into. Rebindable (alter-var-root) for benchmarking."
  (let [n (try (.availableProcessors (Runtime/getRuntime)) (catch Throwable _ 0))]
    (if (> (long n) 1) (long n) 8)))

(def ^:private min-work
  "Total elements below which the loop runs serially: thread hand-off costs more than
   the work saved, and the small fixtures the tests use are all under it."
  4096)

(defn dobands!
  "Call (f lo hi) over contiguous bands covering [0,n), in parallel, and block until
   every band is done. `f` MUST confine its writes to indices in [lo,hi) — bands run
   concurrently.

   `work` is the TOTAL element count the loop touches, which is what decides whether
   threading pays — not `n`, the number of bands. A row-banded blur over a 1024×683
   image has n=683 but work=699392; thresholding on n instead ran every such loop
   serially while looking parallel. Defaults to n for loops where they coincide."
  ([n f] (dobands! n n f))
  ([n work f]
   (let [n (long n)]
    (if (< (long work) min-work)
      (f 0 n)
      (let [k    (long (min threads n))
            step (long (quot (+ n (dec k)) k))
            fs   (doall
                   (for [b (range k)
                         :let [lo (* (long b) step)
                               hi (min n (+ lo step))]
                         :when (< lo hi)]
                     (future (f lo hi))))]
        (doseq [fut fs] (deref fut))
        nil)))))
