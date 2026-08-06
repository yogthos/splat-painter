# splat-painter

A GUI tool that uses a field of 2D Gaussian splats to create an oil-painting
look built by progressive refinement. The painting starts wth an opaque underpainting
of large soft strokes, then successively finer, more translucent layers of brush strokes
traced along the image's edges.

Built in [Jolt](https://github.com/jolt-lang/jolt) (Clojure on Chez
Scheme) with [glimmer](https://github.com/jolt-lang/glimmer) (GTK4) and
[glimmer-gl](https://github.com/jolt-lang/glimmer-gl) (OpenGL).

The covariance math (Σ = R·diag(s²)·Rᵀ, closed-form 2×2 precision) follows the 2D
Gaussian splatting formulation of [DrawingWithGaussians](https://github.com/belkakari/DrawingWithGaussians)
and 3DGS.

## Examples

Each row below shows the source image, then the splat-painted result.

| Source | Splat-painted |
| --- | --- |
| ![](examples/hk-original.jpg) | ![](examples/hk.png) |
| ![](examples/loki-original.jpg) | ![](examples/loki.jpg) |
| ![](examples/phone-original.jpg) | ![](examples/phone.jpg) |
| ![](examples/photog-original.jpg) | ![](examples/photog.jpg) |

## How it paints

Analysis is done once per image load on the CPU, a few seconds on a 1MP photo
(`splat-painter.fields/prepare`, whose builders run across threads):

- **colour structure tensor** (Di Zenzo, per-channel Sobel, gamma-corrected) gives
  per-pixel edge orientation, coherence, and strength. Chroma edges count like luma edges
- **Haar wavelet detail maps** at three scales (aggregate / mid bands / fine bands),
  luma-relative so dark regions keep their detail, fused with locally-normalized
  edge strength
- light + heavy blur colour fields

The rendering is done on the GPU using a vertex+geometry transform-feedback pass which turns
candidate positions into splats with up to seven coarse-to-fine levels. The base level
fully covers the image to ensure there are no gaps by construction. Then, each finer level
adds progressive details where its scale-matched detail map says so.
The broad tiers subdivide, while medium layers overlap and mix. Fine seeds trace brush strokes along the edge tangent as chains of tapered gaussian segments. Ridge-snapping keeps each stroke's colour on its own side of the edge, so it fades out like a drying brush. Mid levels make short translucent glazes, and the finest levels are impasto liner strokes, long thin lines a couple of pixels wide that follow contours from the orientation field.

## Run

```sh
jolt -M:run                       # open the window, click "Open Image…"
jolt -M:run path/to/image.jpeg    # load an image immediately
```

**Save…** writes the picture you are looking at. The **SVG** box beside it picks the
default extension — checked is vector, unchecked is PNG — but whatever extension you
actually type is what decides, so a `.svg` filename over a PNG default does the
obvious thing. The SVG default is `.svgz` (gzipped, ~7× smaller, opens the same
everywhere); type `.svg` to get it plain. **Fidelity** is the size/quality trade.
See "Vector output".

Sliders (live):

- **Splats**, stroke budget (higher = finer, more faithful)
- **Size**, base stroke stdev in px, each finer level halves it
- **Broad**, bokeh dial: loosens only the LOW-detail regions (few large smooth
  daubs, thinned to keep coverage) while the wavelet-detected subjects keep their
  tight underpainting, smooth the background without touching the subject
- **Mid / Fine**, per-tier size multipliers for the mid/fine stroke levels
- **Detail**, how many finer levels are painted (up to seven)
- **Variation**, per-stroke size/tone jitter
- **Curvature**, Perlin bend of stroke traces (gated off on strong edges)
- **Swirl**, how much of the placement noise is the image-INDEPENDENT Perlin field.
  Two things ride on it: the flat-region flow that orients strokes where the tensor has
  no opinion, and the coherence (not the size) of the position warp that pushes seeds
  off the level lattice. At 1.0 (the default) both come from one smooth Perlin field, so
  neighbouring strokes turn and drift together, that is the organic look, and also what
  swims structure around in hazy, low-contrast regions where the detail map does not
  pin strokes down. At 0 the orientation comes from the edge-seeded flow (nearby edges
  voted into the flat areas) and the warp from each seed's own hash: the lattice still
  breaks, but nothing carries the photo's shapes with it. Coherent edges are unaffected
  either way, the tensor already owns those.
- **Stroke**, stroke length (chain step scaling)
- **Contrast**, per-channel contrast
- **Hardness**, edge crispness of detail strokes (tiny marks stay soft, antialiased)
- **Cut-in**, edge-band tier strength: restates silhouettes from their own sides, so
  a coat or hair against an out-of-focus background loses its grey fringe. 0 turns it
  off. The effect saturates around 1.0 (the default)

## Test & check

```sh
jolt -M:test      # unit + golden-field regression tests (incl. GPU field passes)
jolt -M:check     # headless: shader GLSL structure, packing, full pipeline
jolt -M:preview   # CPU render to PNG (no GL needed)
jolt -M:svg       # CPU field to SVG (no GL needed) — see "Vector output"
jolt -M:prof      # analysis/placement profiling
jolt -M:detail    # detail-placement ladder, survival and coverage report
jolt -M:yield     # per-level splats-per-candidate vs what the budget charges for it
jolt -M:scan      # scanline of the placement signals a stroke's stop rules read
jolt -M:pin       # print the golden fixture's actual checksums (for re-pinning)
```

Dev/debug entry points live under `test/`. Only the app ships from `src/`.

The GPU field passes (`splat-painter.gpu-fields`) are tested against their CPU
twins on a windowless GL context, `glimmer-gl.offscreen` asks GDK for a context
bound to the display rather than a surface, so `-M:test` covers render-to-texture
code with no window. On a machine with no display those tests print why they
skipped rather than failing.

Headless overrides (for scripting/testing): `GA_PAINTER_SAVE_PNG`,
`GA_PAINTER_QUIT_MS`, and one per slider, `GA_PAINTER_COUNT`, `GA_PAINTER_SIZE`,
`GA_PAINTER_DETAIL`, `GA_PAINTER_STROKE`, `GA_PAINTER_VAR`, `GA_PAINTER_CURV`,
`GA_PAINTER_BROAD/MID/FINE`, `GA_PAINTER_CUTIN`, `GA_PAINTER_SWIRL`,
`GA_PAINTER_CONTRAST`, `GA_PAINTER_HARDNESS`, `GA_PAINTER_TEX_STREAK/GRAIN/EDGE`.
Then `GA_PAINTER_GPU_VERIFY`, `GA_PAINTER_LOOP_RENDER`, `GA_PAINTER_TF_SMOKE`.
`core-test` pins the one-per-slider rule. `GA_PAINTER_SAVE_PNG` writes whatever its
path's extension asks for, so `=out.svg` scripts a vector save, and
`GA_PAINTER_SVG_FIDELITY` sets the dial for it.

The CPU generator (`splat-painter.seed/splat-field`) stays the tested reference,
the goldens pin it, `GA_PAINTER_GPU_VERIFY` compares the two fields numerically,
and `jolt -M:preview` renders one to PNG with no GL context. There is no CPU
render path in the app: `GA_PAINTER_LOOP_RENDER` swaps the quad renderer for the
pixels×splats loop, which only completes at low splat counts.

## Vector output

A splat IS an ellipse — eigen-decomposing Σ = R·diag(sx²,sy²)·Rᵀ gives a centre, two
radii and a rotation, which SVG expresses exactly — so the field can leave as vectors
instead of pixels. `splat-painter.svg/field->svg` takes the same field map
`seed/splat-field` returns (and `gen/read-splats` reads back off the GPU) and returns
an SVG document. It is pure: no GL, no I/O.

In the app it is **Save…** with the SVG box checked: the live pass renders, the field
comes back off the transform-feedback buffer, and the document is written. Headless,
`GA_PAINTER_SAVE_PNG` takes an `.svg`/`.svgz` path too — the extension picks the
writer. There is also a CPU-path harness that needs no GL at all:

```sh
jolt -M:svg examples/loki-original.jpg /tmp/a.svgz            # gzipped, from the extension
jolt -M:svg examples/loki-original.jpg /tmp/a.svg             # plain
jolt -M:svg examples/loki-original.jpg /tmp/a.svg - - '{:mode :flat}'
jolt -M:svg examples/loki-original.jpg /tmp/a.svg - - '{:fidelity 0.5}'
rsvg-convert -w 4000 -o /tmp/a-4x.png /tmp/a.svgz             # the upscale
```

The viewBox stays in analysed image pixels, so rendering the same file at 4× IS the
upscale: stroke edges and liner traces stay crisp where a bicubic enlargement of the
PNG goes soft. Nothing in the field is resolution-bound.

What SVG has no primitive for is the gaussian FALLOFF across the ellipse, and that is
the whole design problem. Three ways to get a per-splat colour with a soft profile,
measured with `rsvg-convert` on 20k splats at 1000×1000:

| | colour | defs | render |
| --- | --- | --- | --- |
| shared `<mask>`, objectBoundingBox units | exact | 1 | ~100 s |
| `<radialGradient>` per quantized colour | palette | N | ~1.1 s |
| flat ellipse at an iso-contour | exact | 0 | ~0.5 s |

Masks are out — a masked element forces the renderer into an offscreen buffer, 5 ms
each. A gradient cannot take its colour from the element referencing it: `currentColor`
and `var()` in a `<stop>` both resolve against the gradient's own ancestors (checked in
Chromium and librsvg), so one shared gradient paints one colour, full stop. Hence
`:mode :gradient` (the default): median-cut the field's colours and emit one gradient
per (palette entry, hardness bucket) actually used.

The other 15× is `fill-opacity`, not `opacity`. The `opacity` attribute is defined to
composite the element as an isolated group, so renderers allocate a scratch surface per
ellipse — 16 s against 1.1 s on the same 20k. Per-splat alpha is exact either way; only
the colour is quantized.

Against `-M:preview`'s raster of the same field (`examples/loki-original.jpg`, 17.4k
splats, mean abs error per channel):

| `:colors` | MAE | SVG | gzipped |
| --- | --- | --- | --- |
| 64 | 2.85 | 2.2 MB | 373 KB |
| 256 | 1.88 | 2.6 MB | 405 KB |
| 512 (default) | 1.62 | 2.9 MB | 427 KB |
| 1024 | 1.47 | 3.4 MB | 446 KB |
| 2048 | 1.39 | 4.4 MB | 506 KB |

Past ~1024 the remaining error is the piecewise-linear gradient ramp, not the palette:
6, 10 and 16 stops all measure the same, so the default is 8. The palette splits boxes
by population × spread rather than population alone — plain median cut folds a rare
saturated accent (a cat's green eye against grey fur) into the nearest grey box.

`:mode :flat` drops the gradients entirely and draws each splat as a solid ellipse at
its half-intensity contour. It is not a faithful raster (MAE 5.0) — it is a different
picture, a palette-knife look with exact unquantized colour, and the smallest file.

### Fidelity

Element count is file size — nothing else comes close — and at a high splat budget a
faithful export runs to tens of megabytes. The **Fidelity** slider beside the SVG box
trades that off. 1.0 keeps every splat and is exactly what the exporter produced before
the dial existed; below that it prunes, quantizes and rounds. On a 166k-splat painting:

| Fidelity | splats kept | `.svgz` | plain `.svg` | MAE vs the PNG |
| --- | --- | --- | --- | --- |
| 1.0 | 100% | 2.4 MB | 17.6 MB | 1.85/255 |
| 0.9 (default) | 75% | 1.8 MB | 13.1 MB | 1.90/255 |
| 0.5 | 49% | 1.2 MB | 8.4 MB | 2.48/255 |
| 0.1 | 33% | 0.8 MB | 5.1 MB | 5.51/255 |

The dial's first move is nearly free — a quarter of the splats in a dense painting
never reach the image at all.

What decides is each splat's PEAK contribution, max over its footprint of α·T: how
strongly it shows through everything painted in front of it, at the one pixel where it
shows most. That is the 3DGS pruning score (accumulated opacity × transmittance) that
[LightGaussian](https://lightgaussian.github.io/) introduced, with the max
[RadSplat](https://arxiv.org/pdf/2311.17245) substitutes for LightGaussian's sum. The
max is what retains detail: a sum is an AREA measure, so it ranks a one-pixel liner
stroke below a barely-visible fat glaze and prunes the marks the painting is made of.
The literature's other finding lands the same way — occlusion-aware pruning beats
heuristics on isolated attributes like area or opacity, which is why this measures
transmittance rather than either.

Coverage is safe by construction: a dropped splat does not consume transmittance, so
pruning the strokes in front of a base daub leaves it reading a clear canvas and it
cannot then be pruned itself. That bounds how much background can show but does not
drive it to zero, and every hole shows the black clear — which reads as the whole
picture going a couple of levels darker, not as local damage. So a repair pass walks
the dropped splats back to front and re-instates any still sitting over bare ground.
Without it, Fidelity 0.25 lost 4 levels of mean luminance; with it, 0.4.

Under the dial the encoding is squeezed losslessly: a round splat is a `<circle>` (no
second radius, no rotate), a rotated one is `translate()rotate()` rather than a
`rotate()` that repeats the centre three times, and numbers drop their leading zero.
That is about 10% before any splat is pruned.

An export is the painting's GEOMETRY and COLOUR. Lift and Brightness come along —
both are per-channel point operations on the composite, so they are exactly one
`feComponentTransfer` gamma and one linear over the whole picture, which costs one
offscreen surface rather than one per element. Sharpen and Antialias do not: both gate
on a local gradient, which `feConvolveMatrix` cannot express. Neither does the fragment
shader's bristle texture (streak/grain/ragged edge) — nothing per-element survives the
element count.

Saving from the app exports THE LIVE PASS. A committed layer is stored as a texture,
not a splat field — `commit-active!` captures its pixels and the field that made them
is gone — so a stacked painting cannot be re-serialized without regenerating every
layer against its own source composite. The status line says so when the stack is not
empty.

Against the app's own PNG save of the same render (64k splats, default texture dials,
which the SVG does not reproduce) the export lands at 1.7/255 mean abs error.

`.svgz` is the default and is worth more than every encoding trick in the file put
together — an SVG is one long run of near-identical elements, which is the best case
DEFLATE has, and it comes out around 7×. It is written through zlib's `gzopen`/
`gzwrite` (`splat-painter.gzip`), so the file is a real RFC 1952 stream with no
timestamp, and the same painting saves to the same bytes every time. Chromium, librsvg
and Inkscape all open one straight off disk; a web SERVER has to send it with
`Content-Encoding: gzip`.

## REPL-driven development

`jolt nrepl-server` (default port 7888, writes `.nrepl-port`) resolves `deps.edn`
and parks the main thread on a pump, so an eval can start the GTK loop and jolt
marshals the blocking main loop onto the main thread. Connect any editor / nREPL client:

```clojure
(require 'splat-painter.core)
(splat-painter.core/-main "img/street.jpg")   ; window opens; this returns

;; the control atoms are the sliders, reset! one like a drag:
(reset! splat-painter.core/broad-atom 2.5)
;; GTK is single-threaded: marshal the re-render (any widget/GL touch) onto the
;; main loop. Plain data (the atoms) is fine to touch from the REPL thread.
(glimmer.core/on-gui #(#'splat-painter.core/request-render!))

;; hot-reload: redefine a fn/def and the next render uses it, no restart,
(alter-var-root #'splat-painter.seed/splat-budget (constantly 300000))
(glimmer.core/on-gui #(#'splat-painter.core/request-render!))
```

`glimmer.core/reload!` re-renders the mounted panel after you redefine the `app`
component. Quit through the app (close from its own menu / auto-quit) rather than
destroying the window from the REPL.

## Build

A standalone binary (no `jolt` needed to run it) is compiled with `jolt build`:

```sh
jolt build -m splat-painter.core -o splat-painter --opt
./splat-painter path/to/image.jpeg
```

The binary dlopens GTK4, OpenGL, and gdk-pixbuf at runtime, so those must be
installed on the target, `brew install gtk4 gdk-pixbuf` on macOS,
`apt install libgtk-4-1 libgdk-pixbuf-2.0-0` on Linux. macOS binaries are unsigned.
Clear quarantine before first run with `xattr -d com.apple.quarantine ./splat-painter`.

Prebuilt binaries for macOS (arm64) and Linux (x86_64) are attached to each
[tagged release](https://github.com/yogthos/splat-painter/releases). The
`release` GitHub Actions workflow builds them when a `v*` tag is pushed.

## Dependencies

Glimmer and glimmer-gl are git deps pinned in `deps.edn`. gdk-pixbuf (image decode)
is declared as a `:jolt/native` lib. GTK4/OpenGL/GLib come in transitively.
