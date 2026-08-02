# Painting-diagnostics harness

Tools for attributing a visual defect to a specific tier / constant, instead of
guessing at it. Everything here is dev-only; nothing in `src/` depends on it.

The reason it exists: this project's artifacts (a dark rod by the nose, gaps in
fingers, soft edges) are all produced by several tiers painting over each other,
so "turn a slider and look" attributes almost nothing. Six such attempts failed
before the per-level isolation render found the cause in one pass.

## Start a REPL-driven session

    jolt nrepl-server 7888          # writes .nrepl-port; keep it running
    python3 dev/harness/nrepl.py '(+ 1 2)'

`nrepl.py` is a minimal bencode client; run it from the repo root (it reads
`.nrepl-port` from CWD). See the `repl-driven-dev-workflow` bd memory for
launching the GUI without blocking the eval.

## Per-level isolation render — the attribution tool

    python3 dev/harness/nrepl.py -f dev/harness/iso-setup.clj
    python3 dev/harness/nrepl.py '(iso-render! #{2} "/tmp/level2-only.png")'
    python3 dev/harness/nrepl.py '(iso-render! :all "/tmp/all.png")'

`alter-var-root`s `seed/layer-params` to keep only the chosen `:lvl` values,
recomputing `:offset` cumulatively plus `:nlev` and `:total`, so each tier renders
ALONE AT ITS OWN PARAMETERS. The budget solve above it is untouched.

**Use this rather than the Detail slider.** Detail sets `nlev`, which changes the
budget split, which changes every level's `ssz` and spacing — so no Detail-based
comparison isolates a tier. That confound invalidated a day of earlier work.

Levels at a typical Size 7.5: `7` = edge band, `2` = fine, `1` = broad, `0` = base.

`iso-tweak.clj` adds `lvl-override`, which merges arbitrary fields into a kept
level (`{2 {:segs 8}}`), so one tier's spec can be varied with everything else
fixed. CAVEAT: only fields the GS actually reads as uniforms take effect — `:segs`,
`:th`, `:sideo`, `:selong`, `:nx` do. If an override changes nothing, check that
before concluding the parameter is irrelevant. (`:traw` used to be the standing
example of this and is gone from the level map entirely as of bd 6zj — both paths
now derive it from the per-stroke size at the call site.)

`reload.clj` hot-reloads the generation shader after editing `gen.clj`, guarded on
`(:program prog)` so a compile failure cannot silently render with the stale
program.

## Full restart render (`ab.sh`)

    dev/harness/ab.sh /tmp/out.png

Starts a server, launches the app, applies `controls-user.clj`, renders at native
resolution, tears down. Needed because **field-level and ladder-level changes are
not hot-reloadable** — the placement maps and `layer-params` ladder are built at
image load, in both the CPU and GPU paths.

**`ab.sh` KILLS the nREPL server on exit.** Do not mix it with a persistent-REPL
sweep in the same run; that mistake silently cost one whole sweep.

**Apply controls AFTER the image loads, not before.** `on-image-loaded`
(`core.clj:214`) does `(reset! size-atom (max 4.0 (/ height 50.0)))`, and images load
at maxside 1024 — so it clobbers Size to **20.48** on every load, whatever you set
first. A REPL sweep that applies `controls-user.clj` and then loads the image renders
every frame at Size 20.48 instead of the user's 7.5, silently, and every number from it
is worthless. The env-var `jolt -M:run` path is immune (`cur-size` reads
`GA_PAINTER_SIZE` ahead of the atom). This is what the md5-vs-baseline control catches,
which is why you run it first.

## CPU/GPU parity, headless (`test/splat_painter/parity.clj`)

    jolt -A:test -m splat-painter.parity <image> <maxside|0> <shipped|off> [chains]

Replicates `core/gpu-verify!` — count ratio plus first-divergence index — without
launching the GTK app, so it runs in seconds at any resolution. `maxside 0` is
native. The third argument overrides the band tier's side push (`off` removes it),
which is what exposes the on-ridge tracer divergence documented in
splat-painter-9wx; `chains` also prints the CPU band chain-length and stop-reason
distribution. `precision.clj` alongside it replays a single chain in float32 to
separate a precision difference from a formula difference.

Note that a non-100% ratio here is not automatically a bug: tracer parity is a
convention, not a gate (nothing asserts it — `gpu-verify!` only prints, and
`gpu-fields-test` compares FIELDS, never the splat field). See 9wx for the
tolerated band.

## Blog ablations (`blog-ablations-NOT-SHIPPED.patch`)

Recovered from a scratch worktree, never merged. Adds `SP_ABL_ROUND` / `SP_ABL_GRID` /
`SP_ABL_NOWAV` / `SP_ABL_NOLAYER` / `SP_ABL_NOCHAIN` env flags that rewind ONE idea out
of the generation shader each — round dabs instead of tensor-oriented, a stratified
lattice instead of hashed placement, flat density instead of the wavelet detail map, no
glaze ladder, one dab per seed instead of a traced chain. Off by default, so the shipped
shader stays byte-identical.

**It is stale and will half-apply if used as-is.** The flags work by
`clojure.string/replace` against exact shader source lines, and a non-matching anchor is
a SILENT no-op — the flag looks like it ran and changed nothing. At least one anchor has
already rotted: `abl-nolayer?` matches `float tcap = (lvl <= 1) ? 0.35 : ...`, which is
`0.60` on main since 5fd17fa. Re-check every anchor against the current `gen.clj` before
trusting a rendered comparison. If this gets revived, anchor the replacements on
something that fails loudly when it misses.

## Can the dial even reach here? (`test/splat_painter/subjprobe.clj`)

    jolt -A:test -m splat-painter.subjprobe img/Lenin.jpg

Prints the absolute-subjectness histogram plus per-region means. **Run this before
attributing anything to the Broad dial.** The coverage growth is
`mloc = 1 + (Broad−1)·(1−subjAbs)`, so wherever `subjAbs` saturates to 1.0 the dial
is a mathematical no-op and no sweep of it means anything. Saturation is heavily
image-dependent — fraction of frame fully inert: Lenin 64.8%, A7A01535 39.2%,
collapse-watch 27.5%, crow 26.4%. On Lenin the whole subject AND the wall map read
≥0.94, so Broad only acts in one corner. A day of olb work went into sweeping a
control that could not affect the region under investigation.

## Cold compile without destroying your cache

    JOLT_CACHE_DIR=/tmp/coldcheck jolt -M:check     # 4.3s cold, 0.3s warm on re-run
    JOLT_AOT_CACHE=OFF            jolt -M:check     # 3.2s, writes no cache

Use one of these — NOT `rm -rf ~/.jolt/aot-cache` — for the `stroke-segments`
compiler-pathology check (~4s healthy, >10s is the pathology). Both are
non-destructive, keep the developer's warm cache, and work under a sandbox; the
`rm` form is a destructive write to a home-directory path and gets DENIED for
headless agents, which stalled one delegated run.

Note the separate, already-fixed issue: AOT cache invalidation on file SIZE was a
real bug up to jolt 0.5.11 and is gone as of 0.5.12 (re-confirmed on 0.5.17). Do
not clear the cache before a red/green check on its account.

## Metrics

**Check what your ROI actually contains before you trust a number from it.** The
olb ROI was labelled "a narrow jacket strip by the map" and sat on the collar and
tie; every wash/chroma number taken through it measured the wrong part of the
frame. Crop the box and look at it once — `scratchpad/sheet.py` tiles a labelled
ROI from the source beside each render for exactly this.

`distbleed.py` — colour/luma error inside a silhouette BINNED BY DISTANCE from the
boundary. A bleed confined to 2–4px at an edge is invisible in a whole-ROI mean and
obvious as a decay profile; it is also the only way to tell a real bleed from a
uniform offset. It found that Cut-in and Sharpen both *suppress* the silhouette
halo (Cut-in by 2.5×, Sharpen 1.2 nearly to zero) — the opposite of the standing
hypothesis.

`worst.py` — ranks 64×64 blocks by mean|d| against a registered source, so the
regions that actually deviate pick themselves instead of being guessed at.

`edgewidth.py` — **use this, not `distbleed.py`, to judge a silhouette.** Width is
`total_variation / peak_gradient` per scanline, which is TRANSLATION-INVARIANT, and
reported relative to the source. `distbleed` bins error by distance from the
boundary, so a SUB-PIXEL difference in where a render puts its edge moves signal
between bins — and renders do not share a sub-pixel edge position (integer
registration measured (1,1), (1,0) and (0,1) across a single Cut-in sweep). Its
1–2px bin is therefore dominated by registration rather than by the painter: it
produced a non-monotonic Cut-in curve whose jumps line up exactly with the offset
changes, and a byte-identical file read +0.0119 or +0.0831 depending only on which
file was listed first. Registering every file against the first one is the specific
trap; `distbleed` now registers each independently, but the 1–2px bin is still not
evidence at sub-pixel scale.


`score.py` — fidelity vs the source: ROI and whole-image mean|d|, plus "holes"
(source-bright pixels the render made much darker).

Edge SHARPNESS, the metric that matches "fine detail looks soft" — inline, since
it is three lines:

    grad  = hypot(central-difference x, central-difference y)
    band  = grad_source > percentile(grad_source, 85)
    score = mean(grad_render[band]) / mean(grad_source[band])

1.0 would match the source. Shipping is ~0.72 on faces after the band-th work.

`diffmap.py` — signed luma difference as an image (blue = render too dark).

**Register before you measure.** The render sits 1px above a Lanczos downscale of
the source; uncompensated, every edge contributes a dipole that is comparison
error, not painter error. It inflated one set of numbers by 27%. Find the offset by
minimising mean|d| over a few px before trusting any difference metric.

## Discipline that this codebase keeps rewarding

- Mutation-check any test or metric before trusting it. A tolerance tuned on local
  GPU hardware is not valid on the CI renderer (Apple Software Renderer), and a
  parity test that passes may simply not discriminate.
- Verify a control reproduces the baseline byte-identically (md5) before reading
  anything into a sweep.
- Sweep past the optimum. `heavy-radius` and `band-th` both have a peak, not a
  ramp — going further makes things worse, which is itself the evidence that the
  mechanism is understood.
