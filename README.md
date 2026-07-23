# splat-painter

A GUI tool that repaints an image as a field of 2D Gaussian splats — an oil-painting
look built by progressive refinement: an opaque underpainting of large soft strokes,
then successively finer, more translucent layers of brush strokes traced along the
image's edges. Built in [Jolt](https://github.com/jolt-lang/jolt) (Clojure on Chez
Scheme) with [glimmer](https://github.com/jolt-lang/glimmer) (GTK4) and
[glimmer-gl](https://github.com/jolt-lang/glimmer-gl) (OpenGL).

The covariance math (Σ = R·diag(s²)·Rᵀ, closed-form 2×2 precision) follows the 2D
Gaussian splatting formulation of DrawingWithGaussians / 3DGS.

## How it paints

Analysis (once per image load, CPU):

- **colour structure tensor** (Di Zenzo — per-channel Sobel, gamma-corrected) gives
  per-pixel edge orientation, coherence, and strength; chroma edges (lips vs skin)
  count like luma edges
- **Haar wavelet detail maps** at three scales (aggregate / mid bands / fine bands),
  luma-relative so dark regions keep their detail, fused with locally-normalized
  edge strength
- light + heavy blur colour fields (the heavy one edge-preserving)

Generation (per render, GPU): a vertex+geometry transform-feedback pass turns
candidate positions into splats — up to seven coarse-to-fine levels. The base level
fully covers the image (no gaps by construction); each finer level places only where
its scale-matched detail map says so — the broad/mid tiers subdivide, while from
level 3 up the layers overlap and mix. Fine seeds trace **brush strokes**: chains of
tapered gaussian segments stepped along the edge tangent (ridge-snapped, colour from
the stroke's own side of the edge, fading out like a drying brush). Mid levels make
short translucent glazes; the finest levels are **impasto liner strokes** — long thin
lines at a couple-of-pixels width that follow contours while the orientation field
stays coherent, carry nearly opaque paint on strong edges, and keep to their own side
of the ridge so the two sides' colours meet at the edge instead of crossing it.

Rendering (GPU): one blended quad per splat (premultiplied over, back-to-front —
the buffer is already in paint order), so cost scales with painted area, not
pixels × splats.

The CPU implementation of the same pipeline is the tested reference; the GPU is
verified to produce the identical field (exact survivor counts, aggregate sums to
float precision — see `test/splat_painter/check.clj` and `GA_PAINTER_GPU_VERIFY`).

## Run

```sh
joltc -M:run                       # open the window, click "Open Image…"
joltc -M:run path/to/image.jpeg    # load an image immediately
```

Sliders (live):

- **Splats** — stroke budget (higher = finer, more faithful)
- **Size** — base stroke stdev in px; each finer level halves it
- **Broad** — bokeh dial: loosens only the LOW-detail regions (few large smooth
  daubs, thinned to keep coverage) while the wavelet-detected subjects keep their
  tight underpainting — smooth the background without touching the subject
- **Mid / Fine** — per-tier size multipliers for the mid/fine stroke levels
- **Detail** — how many finer levels are painted (up to seven)
- **Variation** — per-stroke size/tone jitter
- **Curvature** — Perlin bend of stroke traces (gated off on strong edges)
- **Stroke** — stroke length (chain step scaling)
- **Contrast** — per-channel contrast
- **Hardness** — edge crispness of detail strokes (tiny marks stay soft — antialiased)

**Save PNG…** exports at the input's native resolution. If GSettings schemas are
missing (the GTK file dialog would abort), it saves `<image>-splats.png` next to the
source instead.

Headless overrides (for scripting/testing): `GA_PAINTER_SAVE_PNG`,
`GA_PAINTER_QUIT_MS`, `GA_PAINTER_COUNT`, `GA_PAINTER_SIZE`, `GA_PAINTER_DETAIL`,
`GA_PAINTER_STROKE`, `GA_PAINTER_VAR`, `GA_PAINTER_BROAD/MID/FINE`,
`GA_PAINTER_CPU_GEN` (CPU reference path), `GA_PAINTER_GPU_VERIFY`,
`GA_PAINTER_LOOP_RENDER`, `GA_PAINTER_TF_SMOKE`.

## Test & check

```sh
joltc -M:test      # unit + golden-field regression tests
joltc -M:check     # headless: shader GLSL structure, packing, full pipeline
joltc -M:preview   # CPU render to PNG (no GL needed)
joltc -M:prof      # analysis/placement profiling
joltc -M:pin       # print the golden fixture's actual checksums (for re-pinning)
```

Dev/debug entry points live under `test/`; only the app ships from `src/`.

## REPL-driven development

`joltc nrepl-server` (default port 7888, writes `.nrepl-port`) resolves `deps.edn`
and parks the main thread on a pump, so an eval can start the GTK loop and jolt
marshals the blocking main loop onto the main thread — the window comes up and the
launching eval returns, leaving the REPL live. Connect any editor / nREPL client:

```clojure
(require 'splat-painter.core)
(splat-painter.core/-main "img/street.jpg")   ; window opens; this returns

;; the control atoms are the sliders — reset! one like a drag:
(reset! splat-painter.core/broad-atom 2.5)
;; GTK is single-threaded: marshal the re-render (any widget/GL touch) onto the
;; main loop. Plain data (the atoms) is fine to touch from the REPL thread.
(glimmer.core/on-gui #(#'splat-painter.core/request-render!))

;; hot-reload: redefine a fn/def and the next render uses it, no restart —
(alter-var-root #'splat-painter.seed/splat-budget (constantly 300000))
(glimmer.core/on-gui #(#'splat-painter.core/request-render!))
```

`glimmer.core/reload!` re-renders the mounted panel after you redefine the `app`
component. Quit through the app (close from its own menu / auto-quit) rather than
destroying the window from the REPL — a raw `gtk_window_destroy` trips a teardown
abort.

## Build

A standalone binary (no `joltc` needed to run it) is compiled with `joltc build`:

```sh
joltc build -m splat-painter.core -o splat-painter --opt
./splat-painter path/to/image.jpeg
```

The binary dlopens GTK4, OpenGL, and gdk-pixbuf at runtime, so those must be
installed on the target — `brew install gtk4 gdk-pixbuf` on macOS,
`apt install libgtk-4-1 libgdk-pixbuf-2.0-0` on Linux. macOS binaries are unsigned;
clear quarantine before first run with `xattr -d com.apple.quarantine ./splat-painter`.

Prebuilt binaries for macOS (arm64) and Linux (x86_64) are attached to each
[tagged release](https://github.com/yogthos/splat-painter/releases); the
`release` GitHub Actions workflow builds them when a `v*` tag is pushed.

## Dependencies

glimmer and glimmer-gl are git deps pinned in `deps.edn`. gdk-pixbuf (image decode)
is declared as a `:jolt/native` lib. GTK4/OpenGL/GLib come in transitively. macOS
OpenGL is 4.1 (no compute shaders); GPU generation uses geometry-shader +
transform-feedback compaction instead.

## Layout

- `src/splat_painter/gaussian.clj` — covariance/precision + CPU reference rasterizers
- `src/splat_painter/image.clj` — gdk-pixbuf load + downscale
- `src/splat_painter/structure.clj` — colour structure tensor, blurs, flow fields
- `src/splat_painter/wavelet.clj` — Haar detail maps (aggregate/mid/sharp/edge)
- `src/splat_painter/noise.clj` — Perlin noise (Ken Perlin's reference permutation)
- `src/splat_painter/seed.clj` — the shared placement + brush-stroke spec (CPU reference)
- `src/splat_painter/gen.clj` — GPU splat generation (vertex+geometry, transform feedback)
- `src/splat_painter/shader.clj` — render shaders (per-splat quads + loop fallbacks)
- `src/splat_painter/png.clj` — PNG export
- `src/splat_painter/core.clj` — the glimmer app (dialogs, control panel, GL loop)
