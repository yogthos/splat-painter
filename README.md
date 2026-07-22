# splat-painter

A small GUI tool that renders an image as a field of 2D Gaussian splats — a
stylized, "drawn" look — with live controls. Built in
[Jolt](https://github.com/jolt-lang/jolt) (Clojure on Chez Scheme) with
[glimmer](https://github.com/jolt-lang/glimmer) (GTK4) and
[glimmer-gl](https://github.com/jolt-lang/glimmer-gl) (OpenGL).

The Gaussian math is a faithful port of
[DrawingWithGaussians](DrawingWithGaussians/drawingwithgaussians/rendering2d.py):
an additive rasterizer where each splat contributes a peak-normalized
`exp(-½ δᵀ Σ⁻¹ δ)` ellipse and the pixel is the background plus the sum of all
contributions. See [blog.md](blog.md) for the design and findings.

## Run

```sh
joltc -M:run                       # open the window, click "Open Image…"
joltc -M:run path/to/image.jpeg    # load an image immediately
```

Splat orientation is edge-driven — each splat is aligned and elongated along the
local image structure (see [blog.md](blog.md)), so there are no manual
angle/aspect controls. Sliders (all live — dragging repaints):

- **Splats** — density (target count)
- **Size** — base splat stdev in px
- **Stroke** — stroke elongation (every splat is a stroke; edges elongate more)
- **Detail** — shrink strokes in high-gradient areas
- **Opacity** — per-splat alpha
- **Sharpness** — color blend toward the sharp raw pixel at edges (vs. the smooth blur)
- **Palette** — diversity color quantization (0 = off)
- **Contrast** — per-channel contrast
- **Hardness** — stroke edge sharpness; higher = crisper, more discrete brush marks

**Save PNG…** exports the current render at the input's native resolution
(offscreen framebuffer → gdk-pixbuf), independent of the window size.

## Test & check

```sh
joltc -M:test      # math + seed + image-load + a numpy golden rasterization
joltc -M:check     # headless: shader GLSL, splat packing, full pipeline
```

## Dependencies

glimmer and glimmer-gl are pulled from `../jolt-lang/` via `deps.edn`
`:local/root`. gdk-pixbuf (image decode) is declared as a `:jolt/native` lib in
`deps.edn`. GTK4/OpenGL/GLib come in transitively.

## Layout

- `src/splat_painter/gaussian.clj` — covariance/precision + additive CPU rasterizer (port of `rendering2d.py`)
- `src/splat_painter/image.clj` — gdk-pixbuf load + downscale to a flat pixel buffer
- `src/splat_painter/structure.clj` — Sobel → structure tensor → per-splat edge orientation/coherence
- `src/splat_painter/palette.clj` — diversity-maximizing color quantization (port of pixel-mosaic)
- `src/splat_painter/seed.clj` — image → splat field: structure-oriented covariance + edge-aware color
- `src/splat_painter/shader.clj` — the splat rasterizer as a GLSL fragment shader
- `src/splat_painter/core.clj` — the glimmer app (file dialog, control panel, GLArea loop)
- `src/splat_painter/check.clj` — headless sanity check
- `test/splat_painter/golden.edn` — numpy reference rasterization for the port-fidelity test
