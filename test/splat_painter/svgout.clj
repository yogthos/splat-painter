(ns splat-painter.svgout
  "Headless image → SVG, the CPU-path twin of splat-painter.preview. Same analysis,
   same splat field, same controls; the field goes to splat-painter.svg instead of a
   rasterizer, so the vector output can be diffed against preview's PNG.

   Run: jolt -M:svg <in-image> <out.svg> [size] [count] [opts-edn]

     jolt -M:svg examples/loki-original.jpg /tmp/a.svg
     jolt -M:svg examples/loki-original.jpg /tmp/a.svg 13 20000 '{:colors 1024}'
     jolt -M:svg examples/loki-original.jpg /tmp/a.svg - - '{:mode :flat}'

   `opts-edn` is splat-painter.svg's option map (see its `default-opts`). `-` or an
   empty string means \"use the default\" for any positional argument.

   The SVG is written in image pixels (viewBox = the analysed image), so rendering it
   at any width is the upscale — nothing in the field is resolution-bound."
  (:require [clojure.string :as str]
            [splat-painter.image :as image]
            [splat-painter.fields :as fields]
            [splat-painter.seed :as seed]
            [splat-painter.svg :as svg]))

(defn- arg [s]
  (let [s (when s (str/trim s))]
    (when (and (seq s) (not= s "-")) s)))

(defn -main [& [path out szs counts opts-edn]]
  (let [[szs counts opts-edn] (map arg [szs counts opts-edn])
        path  (or (arg path) "img/collapse-watch.jpg")
        out   (or (arg out) "/tmp/ga_out.svg")
        img0  (image/load-image path 1000)
        img   (fields/prepare img0)
        size  (if szs (Double/parseDouble szs) (max 4.0 (/ (double (:height img)) 50.0)))
        n     (if counts (long (Double/parseDouble counts)) 14000)
        fld   (seed/splat-field img {:size size :count n :detail 0.6 :variation 0.5
                                     :curvature 0.5 :stroke 2.5 :opacity 0.9 :contrast 1.0})
        opts  (if opts-edn (read-string opts-edn) {})
        _     (println (format "%dx%d  %d splats  size %.1f  sig %.1f..%.1f  opts %s"
                               (:width fld) (:height fld) (count (:splats fld))
                               size (:sig-min fld) (:sig-max fld) (pr-str opts)))
        doc   (svg/field->svg fld opts)]
    (spit out doc)
    (println (format "wrote %s  (%.2f MB)" out (/ (count doc) 1048576.0)))))
