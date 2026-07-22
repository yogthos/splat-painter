# splat-painter — building a 2D Gaussian-splat stylizer in Jolt

A log of how this tool came together, design decisions, and the interesting bits
discovered along the way. Newest entries at the bottom.

## What it is

A GTK4 desktop app that turns an image into a field of 2D Gaussian splats and
renders them back as a stylized, "drawn" version of the picture, with live
controls for splat count, size, brushstroke aspect, rotation, strength, and
background. Built in [Jolt](https://github.com/jolt-lang/jolt) (a Clojure on
Chez Scheme) with the [glimmer](https://github.com/jolt-lang/glimmer) reactive
GTK4 toolkit and its [glimmer-gl](https://github.com/jolt-lang/glimmer-gl)
OpenGL extension.

The Gaussian math is a faithful port of
[DrawingWithGaussians](DrawingWithGaussians/drawingwithgaussians/rendering2d.py)
— specifically its additive rasterizer: each splat contributes a
peak-normalized `exp(-½ δᵀ Σ⁻¹ δ)` ellipse, and the final pixel is the
background plus the sum of all splat contributions. That additive model is what
makes the stylization work without sorting or per-splat opacity.

## Reference algorithm (what we ported)

`rendering2d.py` is the whole rasterizer in ~40 lines. Per gaussian it inverts
the 2×2 covariance by the closed-form `1/det` expression (faster than a general
matrix inverse), accumulates `½(p₀₀ dx² + cross·dx·dy + p₁₁ dy²)` over a pixel
grid, peak-normalizes each splat over the grid, exponentiates, then folds in
color with a single `intensityᵀ @ colors` matmul and adds the background. We
keep that exact math; only the execution substrate changes (CPU Clojure for
tests + a GLSL fragment shader for the live view).

## Architecture

Three layers, each a direct port of the reference's pieces:

- `splat-painter.gaussian` — the pure math. `covariance` builds `Σ = R(θ) diag(s²) Rᵀ`
  (the 2D analog of 3DGS's covariance construction), `precision` is the closed-form
  2×2 inverse, `rasterize` is the additive CPU rasterizer. No GL, no I/O — so it
  tests in isolation.
- `splat-painter.image` — loads a file with gdk-pixbuf (`gdk_pixbuf_new_from_file_at_scale`,
  the same load+resize `fit.py` does with PIL), copies row-by-row honoring
  rowstride, normalizes 8-bit → 0..1 doubles, forces 3 channels.
- `splat-painter.seed` — turns the image into a splat field: a stratified grid of
  means (deterministic, so sliders don't reshuffle), each splat's color sampled
  from the pixel under it, covariance shaped by the controls. This is the same
  parameterization the reference optimizer *learns* (means, scale→covariance,
  color) — we just set it directly from image pixels so the field resembles the
  picture instantly, with no gradient descent.
- `splat-painter.shader` — the GPU twin of `rasterize`: a fullscreen fragment
  shader that loops over every splat (stored in an RGBA32F texture, two texels
  each) and does the identical precision + peak-normalized exp + additive color.
- `splat-painter.core` — the glimmer app: a file picker, a slider panel, and a
  GtkGLArea. The panel is reactive hiccup; the GLArea handlers do raw GL.

## Findings & gotchas

### The additive model needs no sorting — but it blows out for direct-from-pixel seeding

The first instinct with "gaussian splatting" is the 3DGS front-to-back sorted
alpha-compositing pipeline. But `rendering2d.py` is *additive*:
`pixel = background + Σ_i intensity_i · color_i`. No opacity, no depth sort. That
is the whole reason a single fullscreen fragment shader can render it: each
pixel is an independent sum, so a per-pixel `for` loop over splats is an exact
port, not an approximation.

That additive model is correct *in the fitting regime* — the optimizer learns
colors/background that compensate for the overlap. But when you seed thousands
of splats directly from image pixels (no optimization), they massively overlap
and the sum runs far past 1.0. A 64×64 eye image with 1200 splats rasterized
additively to **max 22.06, mean 8.05** — pure white, the "white square" bug. A
CPU probe confirmed it decisively:

```
source       min=0.000 max=1.000 mean=0.306
additive     min=0.030 max=22.060 mean=8.051   <- blowout -> white square
alpha o=1    min=0.004 max=0.923 mean=0.295    <- faithful (mean ≈ source)
alpha o=.6   min=0.004 max=0.920 mean=0.296
```

The fix is the standard 2DGS/3DGS over-operator (`diff_surfel_rasterization`,
the Art repo's rasterizer): `C = Σ cᵢ αᵢ Tᵢ`, `Tᵢ = Π(1−αⱼ)`, `final = C + T·bg`,
where each splat *occludes* the ones behind it instead of adding to them. The
live GLSL shader now composites front-to-back; the additive CPU rasterizer stays
as the tested line-for-line port of `rendering2d.py` (its numpy golden still
validates it), and a sibling `splat-painter.gaussian/composite` ports the
over-operator for the CPU side. This also splits color (raw pixel) from opacity
(a render-time alpha uniform) — the standard gaussian-splatting parameterization,
which is why a new **Opacity** slider replaces the old Strength control.

### The visible grid: a perfect lattice dots through

Seeding means on exact cell centers (`(+ i 0.5) * spacing`) lays down a perfect
lattice. Under over-compositing each splat's peak (intensity 1.0) lands on a grid
point, so the peaks form a visible dotted grid across the image. The fix is
**deterministic stratified jitter**: each mean gets a per-cell pseudo-random
offset in `[-0.5, 0.5)` cell units, computed from a cheap integer hash
(`i*73856093 ^ j*19349663 ^ k*83492791`). The hash is per-cell deterministic, so
a given splat count always yields the *same* field — dragging a slider never
reshuffles the splats, and there's no RNG state to carry. A test asserts the
jitter actually perturbs means (adjacent splats get closer than one cell
spacing, impossible on an exact lattice).

### The sidebar was eating the window: hexpand on slider scales

The control panel took half the viewport because every slider's `:scale` had
`:hexpand true`. Inside the root `:hbox`, an expanding slider competes with the
expanding `:gl-area` for free width — GTK splits it ~evenly, so the sidebar
grabbed half the window. The fix: only the `:gl-area` hexpands; the sidebar and
its scales leave `:hexpand` off (or explicit `false`) so the sidebar sizes to its
natural width (~250px) and the image fills the rest. The one trap: the status
label held the full file path, whose natural width would re-widen the sidebar —
capped with `:max-width-chars 28 :ellipsize :end` so long paths truncate.

### What's tested vs. what isn't (the render-path honesty gap)

Both CPU rasterizers are pinned by **numpy goldens** now: the additive
`rasterize` matches a line-for-line numpy `rendering2d.py` port over 192 output
floats at 1e-4 (`golden.edn`), and the over-compositing `composite` matches a
numpy front-to-back over-operator over the same case (`golden_composite.edn`),
plus three invariants (single splat peaks at color; two coincident opacity-0.5
splats give 0.75·color; 40 stacked white splats never exceed 1.0).

The **GPU fragment shader** (the actual render path the user sees) is *not*
numerically tested — only structurally: `check.clj` asserts the GLSL carries the
right uniforms and the over-compositing expression (`acc += wa * t1.yzw`,
`T *= (1.0 - a)`), and the app boots a real GL context that compiles+links it.
Running GLSL headlessly needs an offscreen OpenGL context (pbuffers / EGL / a
software renderer) that this project doesn't set up, and that's out of scope for
a stylizer tool. The mitigation: the shader's per-splat math mirrors the CPU
`composite` exactly (same precision formula, same peak-normalized `exp(-pdf)`,
same over-operator), so the now-golden-tested CPU path is the reference the GLSL
is hand-checked against. A real divergence would still surface as a visibly wrong
image at runtime, caught by `joltc -M:run`.

### Peak-normalization is free on the GPU

The reference peak-normalizes each splat over the pixel grid
(`exp(z - max(z))`) — partly for numerical stability in the batched matmul. On
the GPU, evaluating `exp(-pdf)` per pixel at the mean gives `exp(0) = 1.0`
already, and `pdf ≥ 0` everywhere so nothing overflows. So the per-grid
subtraction simply isn't needed: the shader's `exp(-pdf)` is peak-normalized to
1.0 at the mean by construction, matching the reference exactly.

### Faithful-port verification with a numpy golden

The strongest proof the port is correct: a line-for-line numpy reimplementation
of `rendering2d.py` rasters a fixed 8×8 / 3-splat case, and
`splat-painter.gaussian-test` asserts the Clojure `rasterize` matches every one of
the 192 output floats within 1e-4 (`test/splat_painter/golden.edn`). If the port
drifts, that test fails immediately.

### Jolt: a missing `:require` looks like "Unknown class g"

Twice I wrote a namespace with a docstring but forgot the `(:require ...)`
clause (seed.clj, shader.clj). The symptom is a runtime
`Unhandled exception: Unknown class <alias>` at the call site — jolt resolves a
namespace-qualified symbol like `g/covariance` against Java interop when no alias
is bound, so it reads as an unknown class rather than "alias not found." Easy to
chase in circles on; the fix is always "add the `:require`."

### Jolt: glimmer reactive atoms aren't `add-watch`-able

`glimmer.ratom` cells aren't Clojure `IRef`s, so `add-watch` throws
"not a watchable reference." The demo app never watches — it just derefs in the
hiccup and lets reconciliation re-render. For the GL pane (which is imperative,
not reconciled), we instead have each slider's `:on-value` reset its atom *and*
call `gtk_gl_area_queue_render`, and `on-render` rebuilds the field from the
current controls. Renders are on-demand, so this is cheap and avoids the watch
problem entirely.

### Jolt: `defcfn` symbols trip the LSP, but compile fine

Every `ffi/defcfn`-bound name (`pixbuf-width`, `gtk-file-dialog-new`, …) shows
as an "unresolved symbol" in the editor's tree-sitter diagnostics — the macro
isn't expanded for analysis. They resolve at compile time; ignore the warnings.

### Jolt: `ffi/null` is a value, not a function

`jolt.ffi/null` is the null pointer itself (a boxed long address), not a fn
that returns it. Writing `(ffi/null)` to pass a NULL to a C call therefore
throws `class java.lang.Long cannot be cast to class clojure.lang.IFn` at the
call site — not "unknown var." The correct form is the bare value: pass
`ffi/null` as an argument directly, exactly as glimmer does
(`gtk-window-set-child parent ffi/null`). It bit the `GtkFileDialog.open` call,
which wants NULLs for its cancellable and user-data slots.

### GTK4 has no synchronous file dialog

`gtk_dialog_run` is gone in GTK4; every dialog is async. `GtkFileDialog.open`
takes a `GAsyncReadyCallback`. jolt's `ffi/foreign-callable` (with the
`:collect-safe` flag, since GTK invokes it from the blocking `g_application_run`
loop) gives a real C function pointer for that callback, so the Open Image
dialog is pure jolt — no C shim. The `GError*` message is read at byte offset 16
of the struct (its layout is `{guint32 domain; gint code; gchar* message;}` with
8-byte pointer alignment).

## How to run

```sh
joltc -M:run            # open the window, click "Open Image…"
joltc -M:run path/to/img.jpeg   # load an image immediately
joltc -M:test           # unit tests (math + seed + image + golden)
joltc -M:check          # headless: shader GLSL, packing, full pipeline
GA_PAINTER_QUIT_MS=5000 joltc -M:run   # smoke test: auto-close after 5s
```

## Styling pass: structure-tensor brushstrokes

The v1 field looked flat — a uniform mosaic, not a painting. Two reasons, both in
the seeding rather than the renderer:

- the input was downscaled to 384px before any color was sampled, so detail was
  gone before rendering started;
- every splat shared *one* covariance, built once from the global Size/Aspect/
  Rotation. A grid of identical blobs.

The reference (DrawingWithGaussians) gets its painterly look the opposite way:
`init_gaussians` seeds *random* means, sigmas, and colors and lets Adam discover
orientation/elongation/color over thousands of steps. We set splats directly from
pixels with no optimization, so we inherit none of that structure. To get it back
without an optimizer, derive the structure from the image itself.

### The structure tensor

`splat-painter.structure` runs a Sobel pass over luminance, forms the per-pixel
tensor `J = [[gx² gxgy][gxgy gy²]]`, and box-blurs it (separable, radius 2) for a
coherent orientation. The minor eigenvector of J is the local isophote — the edge
tangent, the direction a brushstroke should follow. From J at a point:

    φ = ½·atan2(2·jxy, jxx − jyy)      gradient (major-axis) angle
    θ = φ + π/2                        stroke direction (perpendicular)
    coherence = (λ₁ − λ₂)/(λ₁ + λ₂)    0 = isotropic, 1 = a clean edge

Each splat then gets its own covariance: long axis along θ, elongation
`1 + Stroke·coherence`, base size shrunk by `1 − Detail·‖∇‖` in busy regions.
Flat areas stay round and large; edges become thin strokes that trace the
contour. This is the classic painterly-rendering trick (Litwinowicz/Hertzmann) —
neither reference repo does an image-analysis *init* (both lean on gradient
descent), so the tensor pass is the one genuinely new piece; underneath it still
builds on the ported `covariance` and over-compositing math.

The tensor depends only on the image, so it's computed once at load and cached on
the image map — dragging a slider re-seeds (cheap, O(splats)) without re-running
Sobel.

Controls changed to match: **Aspect** and **Rotation** are gone (orientation is
now per-splat and edge-driven); **Stroke** (elongation) and **Detail** (size
adaptivity) replace them, and the **Opacity** earlier entries kept promising is
finally wired to the shader. The input load also went from 384px to 1024px.

### Gotcha: a flipped Sobel sign hiding behind a stale cache

The diagonal-edge test wanted θ ≈ 3π/4 and got π/4 — a sign flip in `gx` made
`jxy` negative and rotated every diagonal stroke the wrong way. Cheap to catch
with a test, but only if you write it (orientation is undirected, so the test
compares angles mod π). The chase got muddier because jolt's compile cache kept
serving the pre-fix `.clj`; `rm -rf .cpcache` before re-running is the tell, and
worth doing before trusting any "tests pass" on freshly edited code.

## More natural color: region sampling, contrast, palette (ported from pixel-mosaic)

The brushstrokes had the right *shape* but noisy *color* — each splat took a
single nearest pixel, so it speckled and, worse, a stroke straddling an edge
grabbed whatever pixel sat under its center. [pixel-mosaic](https://github.com/yogthos/pixel-mosaic),
an edge-aware pixelation library, already solved this for blocks; we ported its
color assignment.

Per splat, sample the pixels in the splat's footprint and take the **average** in
smooth regions, blending toward the **luminance-median** as edge strength rises —
the median picks a real pixel color instead of averaging across a boundary. Edge
strength is free: it's the structure-tensor coherence from the previous pass. A
**Sharpness** control scales the blend (0 = soft/all-average, 1 = crisp).

Two more ports on top: a per-channel **Contrast** about 0.5, and optional
**Palette** quantization (`splat-painter.palette`) — diversity-maximizing
farthest-point seeding over a coarse-keyed color histogram, then snap each splat
to its nearest palette color for a cohesive poster look. Palette 0 = off.

### Gotcha: jolt stalls compiling a nested `for` with array writes

The region sampler started as a nested `for` over the window; joltc hung
*compiling* it — not a runtime cost, the compile itself never returned. Rewriting
the two loops as `loop/recur` filling a `double-array` compiled instantly and runs
faster anyway. Separately, aliasing the palette namespace as `palette` collided
with the `:palette` control key inside the destructure — the alias is `p`.

## Making it look painted (and fast enough to drag)

First real-image runs exposed two coupled problems: the controls were unusable
(seconds per slider drag) and the output looked like a smooth smear, not brushwork.

### Profiling: jolt has no primitive-array fast path

Headless timing (`joltc -M:prof`) on a 1000px photo: `structure/analyze` 10 s
one-time, `splat-field` 1.4 s *per slider drag*. First guess was the 3M-element
`:pixels` persistent vector under `nth`, so it moved to a Java `double-array` with
`aget`. It barely moved the needle (1433 → 1433 ms). The lesson: jolt/Chez does
**not** do JVM-style primitive-array specialization — `aget` on a `double-array`
is no faster than `nth` on a vector, and the numeric ops box. So the only lever is
**doing less work**, not faster access:

- **Structure tensor at reduced resolution.** Orientation varies slowly, so the
  Sobel + tensor run on a ≤384px downscale, not the full 1024px. `analyze`
  10.3 s → 1.75 s, and the coarser tensor is a *smoother* flow field, which helps
  the painted look. `orient-at` maps full-image coords into the small grid.
- **Precomputed blur instead of per-splat region scans.** The edge-aware colour
  was sampling a window and sorting for a median *per splat, every render*. Now a
  blur is computed once at load; per-splat colour is a single `blend(blur, raw)`
  lookup (smooth in flat regions, the sharp raw pixel at edges). `splat-field`
  1433 → 38 ms at 6000 splats.

(The real fix for the numeric floor is teaching jolt's compiler to honour
primitive type hints — a language change, tracked separately.)

### Brushstrokes, not blobs

Even fast, the field looked blurry: in flat regions coherence ≈ 0, so splats stayed
round and soft and blended into a smear. Three changes turned it into brushwork:

- **Every splat is a stroke.** A coherence floor (`min-coh`) means even flat areas
  elongate along the (smoothed) flow field, so the whole image is directional
  marks, not round dots.
- **A hard-edged stroke profile.** The shader alpha went from `exp(-pdf)` to
  `exp(-pdf^hardness)`; `hardness > 1` flattens the core and steepens the edge, so
  strokes read as discrete brush marks instead of soft gaussians that blend. Exposed
  as a **Hardness** slider.
- **Sane default size.** The default stdev was `H/32` (32px on a 1024px image —
  huge). Now it's ~0.6× the grid spacing, so strokes are marks, not overlapping
  blobs.

## Where the detail lives: wavelet-adaptive splats + noise-varied strokes

Two follow-on problems once it looked painted: the strokes read as a regular grid
(uniform size, aligned direction), and every region got the same density no matter
how much detail was there.

### Adaptive size + density from a wavelet detail map

`splat-painter.wavelet` ports [wavescope-mcp](https://github.com/yogthos/wavescope-mcp)'s
Haar transform to 2D — where wavescope decomposes a 1D line-importance signal, we
decompose the luminance and sum `|detail|` across Haar scales into a per-cell detail
energy map: high in textured/edgy regions, ~0 in flat ones. The seed then subdivides
each base grid cell into `n×n` sub-cells, `n` (1–4) driven by the local detail —
high-detail cells get up to 16 small strokes, flat cells stay one large stroke. **Size**
became the *minimum* stroke size (the floor in busy areas); flat areas draw their size
from the coarse grid spacing. So the eye's iris gets dense fine strokes while the sky
gets a few big ones.

### Perlin flow for stroke variation

`splat-painter.noise` ports [perlin-flow](https://github.com/yogthos/perlin-flow)'s Perlin
noise (Ken Perlin's reference permutation, so no 32-bit RNG to port). Each stroke's
orientation blends the structure-tensor edge angle with a smooth Perlin *flow* angle,
weighted by `variation·(1−coherence)`: on strong edges the stroke follows the contour;
in flat regions it follows the organic flow field, so the background reads as flowing
brushwork instead of a rigid hatch. Per-stroke size and tone get a cheap deterministic
hash jitter. A **Variation** slider scales all of it.

### The speed came from the compiler

None of this would be draggable if the analysis were slow. `structure/analyze` went
10.3 s → 167 ms across the session: computing the structure tensor at reduced
resolution, and then teaching jolt itself to honour `^double`/`^doubles` primitive
hints — `Math/*` lowering to native `flsqrt`/`flatan`, `aget`/`aset` as inlined ops,
and `double-array` backed by an unboxed Chez flvector so `(aget ^doubles a i)` reads a
raw flonum and the surrounding arithmetic stays `fl+`/`fl*`. (That work lives in the
jolt repo; here it just means the sliders are live.)

## Edge-aware placement: wavelet detail, edge-seeded flow, a deforming grid

To make the field read as brushwork rather than a stamped lattice, placement,
size, and orientation all became image-driven:

- **Wavelet detail map** (`splat-painter.wavelet`, a 2D Haar port of wavescope-mcp):
  where the image has fine texture, cells subdivide into more, smaller strokes;
  flat areas stay coarse. Size range is genuinely adaptive.
- **Edge-seeded flow** (`splat-painter.structure/flow-at`): a heavily-blurred copy of
  the structure tensor diffuses the edges' orientation into the surrounding flat
  areas, so a stroke in a low-detail region follows the *nearby feature*. A smooth
  Perlin *vector* flow (two decorrelated channels → `atan2`, never a scalar
  `noise·π` which bands into stripes) fills truly featureless areas.
- **Deforming grid** (`splat-painter.grid`, a port of pixel-mosaic's
  `optimizeGridCorners`): the base grid's interior corners are nudged (damped,
  Jacobi, 2 iterations) so cell boundaries snap onto image edges — the cells
  conform to structure and the strokes placed in them align with contours.

### The gaussian-splat / flat-region tension

One honest limitation surfaced: overlapping *gaussian* splats can't tile a
featureless region cleanly. With a hard edge profile adjacent splats show seams (a
faint weave); soft, they blur. pixel-mosaic avoids this by filling each cell with a
*solid* colour block — no falloff, no seam. A gaussian painter trades that clean
flat fill for soft, overlapping strokes. Pushing **Variation** (organic size /
flow) and letting detailed regions carry the crisp strokes is the lever that reads
the residual flat-area texture as canvas rather than a grid.

### Size as base stdev, and never truncating the field

Two things had to change together to get a working Size control without gaps.

First, size became *continuous*, not a floor. Each stroke's stdev is
`smin + (smax - smin)·(1 - detail·D)` where `D` is the local wavelet detail and
`smax` is the Size slider. Flat regions get the full Size; the most detailed get
~0.28× it. Density follows size — spacing ≈ 0.6× stdev — so strokes overlap
everywhere and there are no gaps, at any Size.

Second came a bug worth calling out: the fragment shader brute-force loops over
*every* splat per pixel, so the field is hard-capped at `MAX_SPLATS` (16384). The
first attempt to respect that cap did `(take 16000 all)` — which kept the first
rows and dropped the last, cutting the bottom off the image whenever a small Size
pushed the count over budget. The fix is to never truncate: compute the total,
and if it exceeds the budget, scale *all* stroke sizes up uniformly (`total ∝
1/stdev²`, so a couple of `sqrt(total/budget)` steps converge) until the whole
field fits. Coverage stays full because every cell still gets at least one stroke;
only the strokes get a touch larger. Verified headless — at Size 6 (the worst
case) splats still span `0..H`.

### Precompute everything position-dependent

The noise-varied strokes regressed per-render cost badly: `noise2` is ~30 ops and
the loop called it four times per splat across ~14k splats. But those fields —
final blended orientation (edge-seeded flow + Perlin fill + sharp-edge blend),
coherence, and the size/tone Perlin channels — depend only on *position*, and
they're smooth. So `prep-noise` computes them once at the tensor resolution and
folds the two orientation blends (two `atan2`/`cos`/`sin` each) into that one pass.
The hot loop drops to a handful of `aget`s plus one covariance `cos`/`sin`. Same
for the deforming grid: it's cached at load, `edge-strength-fn` precomputes a
normalized-gradient array the corner optimizer ags instead of recomputing per
sample, and the optimizer runs a single damped Jacobi pass. Net: one-time load
~2.4s, per-render 55–112ms (was 308ms), so slider drags stay responsive.

## What's not here (yet)

The reference's gradient-descent *fit* (the Adam + densify/prune optimization in
`fit.py`/`gaussian.py`) is deliberately out of scope — this is the interactive
"set the splats straight from the image" stylizer, not the reconstructor. The
faithful additive rasterizer and scale+rotation covariance are ported; the
optimizer is a follow-up if you want densification-driven reconstruction on top.
