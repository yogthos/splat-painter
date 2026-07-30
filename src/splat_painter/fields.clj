(ns splat-painter.fields
  "The per-image precomputed placement fields, in ONE place.

   Everything the seed reads that does not change with a slider — the orientation
   tensor, the three colour sources, the wavelet placement map, the Perlin flow —
   is built once when an image is loaded so a live drag only re-runs placement.

   This lived inline in core/prepare-image and was hand-copied into seed-test
   (\"mirror core/on-image-loaded EXACTLY\"), detail, preview and prof — five copies
   of a definition that has to agree or the tests and diagnostics measure a
   different painting than the app makes. It is a function now."
  (:require [splat-painter.structure :as structure]
            [splat-painter.wavelet :as wavelet]
            [splat-painter.seed :as seed]))

(defn heavy-radius
  "Window for the coverage tiers' colour source — a brush-sized neighbourhood,
   scaled to the image so it means the same thing at any resolution."
  [image]
  (max 6 (quot (long (:height image)) 80)))

(defn prepare
  "Attach the precomputed placement fields to a loaded image map. Returns the
   assoc'd map; pure, so the app, the tests and the diagnostics all build the
   same painting."
  [img0]
  ;; The six builders below are the whole cost of loading an image. Only :detail and
  ;; :noise-fields need the tensor, so the three colour fields start immediately and
  ;; the rest follow analyze — wall time is the longest chain, not the sum. Each
  ;; builder writes its own arrays and reads only img0/sfield, so this is scheduling
  ;; alone: fields-test pins that every field comes out identical.
  (let [;; EDGE-AWARE light blur: smooth within a shade region, no mixing across
        ;; boundaries — the box blur fed strokes near edges contaminated colours
        ;; (pale splotches in dark areas, smudges on smooth skin)
        light  (future (structure/bilateral-blur img0 3))
        ;; the drift/dry-out probes keep a FORGIVING box blur: on the razor-sharp
        ;; bilateral field any probe wobble across a boundary trips the lift
        ;; threshold instantly and fragments contour chains into bead dashes
        drift  (future (structure/blur-image img0 2))
        heavy  (future (structure/dominant-blur img0 (heavy-radius img0)))
        ;; PIXEL-SCALE edge strength. The placement maps below are all built at
        ;; <=768 with blurs on top of the tensor's own, so none of them can say
        ;; whether a 4px feature is still there — a stroke asking "has my ridge
        ;; ended?" always heard yes. This is the signal that answers it.
        efull  (future (structure/full-edge img0))
        sfield (structure/analyze img0)
        detail (future (wavelet/placement-map img0 sfield))
        noise  (future (seed/prep-noise sfield))]
    (assoc img0 :structure sfield
                :edge-full @efull
                :blur   @light
                :blur-drift @drift
                ;; the smooth colour field the COVERAGE tiers paint with, at their own
                ;; scale: the DOMINANT tone of a brush-sized window, so structure finer
                ;; than the brush drops out of it instead of being stamped by it. The
                ;; edge-preserving blur this replaced kept sub-brush structure by
                ;; construction, so a σ6 base stroke centred on a 5px letter painted
                ;; solid black over a 12px radius and nothing repainted the paper
                ;; around it — letters fattened, counters filled, gaps closed.
                :blur-heavy @heavy
                :detail @detail
                :noise-fields @noise)))
