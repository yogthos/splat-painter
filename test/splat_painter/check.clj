(ns splat-painter.check
  "Headless sanity check: the shader emits well-formed GLSL (the splat loop, the
  precision formula, every uniform), splat packing matches the texture layout,
  the full image -> seed -> pack pipeline runs end to end, and the glimmer-gl
  widgets register. Needs no GL context and no display. Run with `joltc -M:check`."
  (:require [clojure.string :as str]
            [splat-painter.shader :as shader]
            [splat-painter.gen :as gen]
            [splat-painter.seed :as seed]
            [splat-painter.image :as image]
            [splat-painter.core :as core]
            [glimmer.widget :as w]
            [glimmer-gl.gtk]))            ; loading registers :gl-area / :scale

(defn- assert-contains [src needle label]
  (assert (str/includes? src needle) (str label " missing from shader: " needle))
  true)
(defn- assert-not-contains [src needle label]
  (assert (not (str/includes? src needle)) (str label " should be gone from shader: " needle))
  true)

(defn- hiccup-nodes
  "Depth-first walk over a glimmer widget tree. A node is either a hiccup element
  ([:tag props & children]) or a function-call form ([fn & args]) that yields a
  node when applied — the latter is how control-panel/slider compose. We realize
  (apply) call-forms so the full tree is walked."
  [root]
  (letfn [(realize [node]
            (if (and (vector? node) (seq node)
                     (not (keyword? (first node))) (ifn? (first node)))
              (apply (first node) (rest node))
              node))
          (children [node]
            (let [n (realize node)]
              (when (and (vector? n) (keyword? (first n)))
                (->> n rest (drop-while map?) (filter vector?)))))]
    (tree-seq #(and (vector? %) (seq %)) children root)))

(defn -main [& _]
  ;; the samplerBuffer LOOP variant (GA_PAINTER_LOOP_RENDER): consumes the
  ;; transform-feedback stream directly, and carries the reference over-compositing
  ;; math the quad renderer reproduces per splat.
  (let [{:keys [vs-src fs-src-buf]} (shader/sources)]
    (println "shader: vs" (count vs-src) "chars, fs" (count fs-src-buf) "chars")
    (assert-contains fs-src-buf "uniform samplerBuffer u_splats;" "samplerBuffer u_splats")
    (assert-contains fs-src-buf "uniform int  u_count;"          "u_count")
    (assert-contains fs-src-buf "uniform vec2 u_viewport;"       "u_viewport")
    (assert-contains fs-src-buf "uniform vec2 u_image;"          "u_image")
    (assert-contains fs-src-buf "uniform vec3 u_bg;"             "u_bg")
    (assert-contains fs-src-buf "texelFetch(u_splats, 3 * i);"     "flat texelFetch texel0")
    (assert-contains fs-src-buf "texelFetch(u_splats, 3 * i + 1);" "flat texelFetch texel1")
    (assert-contains fs-src-buf "texelFetch(u_splats, 3 * i + 2);" "flat texelFetch texel2 (alpha)")
    (assert-contains fs-src-buf "float det = max(c00 * c11 - c01 * c01, 1e-8);" "det floor")
    (assert-contains fs-src-buf "float p00 = c11 / det, p11 = c00 / det, cross = -2.0 * c01 / det;"
                    "precision (2x2 inverse)")
    (assert-contains fs-src-buf "uniform float u_opacity;"          "u_opacity")
    (assert-contains fs-src-buf "uniform float u_hard_sharp;"       "u_hard_sharp")
    (assert-contains fs-src-buf "uniform float u_hard_soft;"        "u_hard_soft")
    (assert-contains fs-src-buf "float hardness = mix(u_hard_sharp, u_hard_soft, ts);" "size-scaled hardness")
    (assert-contains fs-src-buf "acc += wa * t1.yzw;"             "over-composite color accumulation")
    (assert-contains fs-src-buf "T *= (1.0 - a);"                 "transmittance update")
    (assert-contains fs-src-buf "frag = vec4(acc + T * u_bg, 1.0);" "background weighted by T")
    (assert-contains vs-src "void main()"                    "vertex main"))

  ;; the per-splat quad renderer (no pixels×splats loop — the 48k-hang fix)
  (let [{:keys [vs-src-quad fs-src-quad]} (shader/sources)]
    (println "render (per-splat quad variant):")
    (assert-contains vs-src-quad "int splat  = (u_count - 1) - (gl_VertexID / 6);" "quad back-to-front order")
    (assert-contains vs-src-quad "vec2 he = (3.5 + 5.0 * edgeAmt) * sqrt(vec2(c00, c11));" "quad marginal-stdev extents (edge-tex dilated)")
    (assert-contains vs-src-quad "float ts  = clamp((sig - u_sig_min) / max(u_sig_max - u_sig_min, 1e-4), 0.0, 1.0);" "quad size→hardness")
    (assert-contains fs-src-quad "float a = v_alpha * u_opacity * exp(-pow(pdf, v_hard));" "quad per-splat alpha formula")
    (assert-contains fs-src-quad "vec3  base = mix(vec3(lum), v_color, sat);" "quad grain inherits stroke hue")
    (assert-contains fs-src-quad "vec3  col = clamp(base * bright, 0.0, 1.0);" "quad paint-texture colour")
    (assert-contains fs-src-quad "frag = vec4(col * a, a);" "quad premultiplied output"))

  ;; the layer blit (alpha-aware: each committed pass composited src-over at its
  ;; stored opacity under/over the live splat pass): attribute-less, same letterbox
  ;; rect as the quad VS, premultiplied content scaled by u_alpha
  (let [{:keys [vs-src-blit fs-src-blit]} (shader/sources)]
    (println "render (layer blit variant):")
    (assert-contains vs-src-blit "int corner = gl_VertexID;" "blit attribute-less (gl_VertexID only)")
    (assert-contains vs-src-blit "float scale = min(u_viewport.x / u_image.x, u_viewport.y / u_image.y);" "blit letterbox mapping")
    (assert-contains vs-src-blit "v_uv = (pane - org) / (u_image * scale);" "blit texcoord from letterbox rect")
    (assert-contains fs-src-blit "uniform sampler2D u_layer;" "blit u_layer sampler")
    (assert-contains fs-src-blit "uniform float u_alpha;" "blit per-blit opacity gain")
    (assert-contains fs-src-blit "uniform float u_encode;" "blit sRGB re-encode toggle")
    (assert-contains fs-src-blit "if (u_encode > 0.5) rgb = srgbEncode(rgb);" "blit conditional sRGB re-encode")
    (assert-contains fs-src-blit "frag = vec4(rgb, c.a * u_alpha);" "blit alpha-aware premultiplied output"))

  ;; the GPU generation shader must MIRROR seed/splat-record + layered-means + noise
  (let [{:keys [vs-src gs-src]} (gen/sources)]
    (println "generation (vertex + geometry, transform feedback):")
    (assert-contains vs-src "v_id = gl_VertexID;" "gen VS passes vertex id")
    (assert-contains gs-src "out vec4 o_a;" "gen TF varying o_a")
    (assert-contains gs-src "out vec4 o_b;" "gen TF varying o_b")
    ;; placement (layered-means): threshold discard + jitter + Perlin warp gate
    (assert-contains gs-src "float thd = th * (0.75 + 0.5 * hash01(i*43 + lvl, j, 19));" "dithered placement threshold")
    (assert-contains gs-src "if (lvl > 0 && dv * gain < thd && !thinAdmit) return;" "subject-gated threshold discard (thin-bright admission bypass)")
    (assert-contains gs-src "float thp = THIN_GAIN * (dot(sampleRGB(u_blurTex, cx, cy), LUMA)" "thin-bright admission signal (mirror seed/thin-gain)")
    (assert-contains gs-src "float sgate = subjectAt(cx, cy);" "wavelet subjectness gate")
    (assert-contains gs-src "if (hash01(i*61 + lvl, j, 43) >= bminp*bminp) return;" "bokeh-adaptive broad thinning")
    (assert-contains gs-src "if (ev < 0.10) { reason = RSN_RIDGE; break; }" "feature-following: ridge died -> clean break (:ridge)")
    (assert-contains gs-src "abs(dx0*dxp + dy0*dyp) < 0.90" "feature-following: clean bend break at a corner (:corner)")
    (assert-contains gs-src "dmx > 0.60" "chroma BACKSTOP only (runaway); mild dry-out + racc dropped for detail tiers")
    (assert-contains gs-src "reason = chroma ? RSN_CHROMA : (lh ? RSN_LH : RSN_DRIFT);" "stop reason tracked in an int (GLSL has no loop returns)")
    (assert-contains gs-src "if (fade < 0.15) {" "no emission after the brush lifts (clean break, reason set)")
    (assert-contains gs-src "float al = lal2 * fade * (1.0 - 0.65 * tt * tt) * ha;" "both-ends taper × glaze × dry-out alpha")
    (assert-contains gs-src "float body = (liner ? clamp((ev - 0.25) / 0.45, 0.0, 1.0)" "impasto body gated on the physical liner predicate (ev cached once)")
    (assert-contains gs-src "float Ev = (lvl <= 3) ? edgeNear(cx, cy, 0.75 * ssz) : edgeAt(cx, cy);" "footprint-sensed edge strength")
    (assert-contains gs-src "float cy = float(u_W) * poshash(i, lvl, 31);" "avalanche-hashed candidate y")
    (assert-contains gs-src "float cx = float(u_H) * poshash(i, lvl, 29);" "avalanche-hashed candidate positions")
    (assert-contains gs-src "uint wang32(uint v){" "Wang avalanche hash")
    (assert-contains gs-src "float aw = (lvl >= 2 && ssz < 3.5) ? 0.0 : u_warp * (1.0 - D) * ssz;" "gen warp zeroed on physical liner-scale levels")
    ;; SWIRL (mirror seed/warp-noise + seed/with-swirl): the warp's spatial coherence is
    ;; the dial, its amplitude is not — Perlin at 1, per-seed avalanche hash at 0.
    (assert-contains gs-src "return u_swirl * noise2(fx, fy) + (1.0 - u_swirl) * poshash(i, lvl, salt);" "gen warp noise mixes Perlin with the seed's own hash")
    (assert-contains gs-src "float x2 = (aw < 0.2) ? x : x + aw * warpNoise(i, lvl, 61, 0.06*x, 0.06*y);" "gen position warp goes through warpNoise")
    ;; the orientation half of the dial: the geometry shader mixes the two baked fields
    ;; per fetch (the CPU mixes the arrays once — linear either way, so they agree)
    (assert-contains gs-src "uniform sampler2D u_noiseSTex;" "gen swirl-free orientation field")
    (assert-contains gs-src "vec3 v00 = mix(texelFetch(u_noiseSTex, ivec2(j0, i0), 0).xyz, texelFetch(u_noiseTex, ivec2(j0, i0), 0).xyz, u_swirl);" "gen fieldsAt mixes the two orientation fields by Swirl")
    ;; Round 2: liner? is a pure physical-size predicate (lvl>=2 && ssz<3.5), mirrored in the GS.
    (assert-contains gs-src "bool liner = (lvl >= 2 && ssz2 < 3.5);" "gen physical liner predicate (mirror seed/liner-scale?)")
    (assert-not-contains gs-src "(lvl >= 4) || (lvl >= 2" "old lvl>=4 liner disjunct removed (raw-floor/melt still key on lvl)")
    (assert-contains gs-src "float sfoot = sabs;" "footprint-sensed Broad growth gate")
    (assert-contains gs-src "float mloc  = 1.0 + (u_broad - 1.0) * (1.0 - sfoot);" "growth gated by grown footprint")
    ;; hash01 (uint wrap == CPU mod 2^32) + Perlin (permutation texture)
    (assert-contains gs-src "uint(a)*73856093u + uint(b)*19349663u + uint(salt)*83492791u" "gen hash01 constants")
    (assert-contains gs-src "float noise2(float x, float y){ return noise3(x, y, 0.0); }" "gen noise2")
    ;; splat-record: elongation, covariance, colour blend
    (assert-contains gs-src "float e   = 1.0 + min(u_stroke, 1.5) * coh * (0.25 + 0.75 * D);" "gen capped elongation")
    (assert-contains gs-src "float L = ssz2 * stepf;" "stroke-length chain step (stepf final)")
    (assert-contains gs-src "float c00 = sx2*c*c + sy2*s*s;" "gen covariance c00")
    (assert-contains gs-src "float tcap2 = min(tcap, 0.3 + 0.7 * min(1.0, 3.0 / max(csz, 1e-6)));" "colour specificity follows brush size")
    (assert-contains gs-src "float t = min(tcap2, max(traw, clamp(0.15 + 0.85 * max(coh0, D), 0.0, 1.0)));" "gen colour blend with per-level raw floor + specificity cap")
    (assert-contains gs-src "* max(1.0, ssz * szf / 8.0)) * Ev);" "sigma-aware near-edge shrink")
    (assert-contains gs-src "float tcap = (lvl <= 1) ? 0.60 : (ssz2 < 3.5) ? 1.0 : (ssz2 < 8.0) ? 0.7 : 0.35;" "progressive colour-specificity ceiling")
    (assert-contains gs-src "float mapAt(int sel, float x, float y){" "scale-matched map selector")
    (assert-contains gs-src "vec4 fieldBilerp(sampler2D tex, float x, float y, vec2 dim, vec2 src){" "field maps sampled bilinearly (fieldBilerp)")
    (assert-contains (:fs-src-sharpen (shader/sources)) "float dscale = SHARP_DETAIL_FLOOR + (1.0 - SHARP_DETAIL_FLOOR) * smoothstep(DETAIL_KNEE.x, DETAIL_KNEE.y, subj);" "sharpen gated on subjectness (detail-sharpen-scale)")
    (assert-contains gs-src "vec4 t = fieldBilerp(u_detailTex, x, y, u_detailDim, u_detailSrc);" "detail/sharp/mid maps via fieldBilerp")
    (assert-contains gs-src "fieldBilerp(u_subjTex, x, y, u_detailDim, u_detailSrc).r" "absolute subjectness via fieldBilerp")
    (assert-contains gs-src "vec3 r11 = texelFetch(tex, ivec2(y1, x1), 0).rgb;" "colour sampled 4-tap bilinear (sampleRGB), not nearest")
    (assert-contains gs-src "if (fdv >= u_th[k-1] * (0.75 + 0.5 * hash01(i*47 + lvl, j, 23))) return;" "dithered subdivision claim")
    (assert-contains gs-src "uniform sampler2D u_blurHTex;" "gen heavy-blur texture")
    (assert-contains gs-src "float hb = (lvl <= 1) ? 1.0 : 0.0;" "broad strokes use heavy blur")
    (assert-contains gs-src "o_a = vec4(px, py, c00, c01);" "gen output o_a layout")
    (assert-contains gs-src "o_b = vec4(c11, color.r, color.g, color.b);" "gen output o_b layout")
    (assert-contains gs-src "o_c = vec4(alpha, clamp(subjAbsAt(px, py), 0.0, 1.0), 0.0, 0.0);" "gen output o_c (stroke-taper alpha + subjectness)")
    ;; region-consistency clamp (mirror seed/splat-field): the bilateral defines the
    ;; region; raw specificity trusted only when consistent with it — an
    ;; edge-straddling raw sample is pulled to the region colour, not bled across.
    (assert-contains gs-src "vec3  bilat = sampleRGB(u_blurTex, hx, hy);" "region: bilateral defines the colour region")
    (assert-contains gs-src "float wcl   = clamp((dcl - 0.12) / 0.15, 0.0, 1.0) * sizew;" "region-consistency clamp weight")
    (assert-contains gs-src "raw = mix(raw, bilat, wcl);" "raw pulled to bilateral region colour")
    ;; the brush-stroke trace (mirror of seed/stroke-segments)
    (assert-contains gs-src "layout(points, max_vertices = 32) out;" "gen GS emits stroke chains (32 liner segments)")
    (assert-contains gs-src "float sz = ssz2 * (1.0 - 0.45 * tt * sqrt(tt)) * hw;" "both-ends stroke width taper (seed-jittered size)")
    (assert-contains gs-src "float lal = (band || lvl <= 1) ? 1.0 : (ssz2 >= 8.0) ? 1.0 : 0.85;" "per-physical-size glaze alpha; band tier opaque by role")
    (assert-contains gs-src "float soff = band ? u_sideo[k] * (0.6 + 2.55 * bph * bph) : u_sideo[k];" "band push distribution crowds the near zone (squared jitter)")
    (assert-contains gs-src "float traw = (lvl <= 1) ? 0.0 : (ssz2 < 1.5) ? 0.85 : (ssz2 < 3.5) ? 0.7 : (ssz2 < 8.0) ? 0.45 : 0.0;" "per-physical-size raw floor")
    (assert-contains gs-src "if (lvl >= 2) traw *= 1.0 - 0.7 * sharpAt(cx, cy);" "density-scaled traw: trust region colour where fine detail crowds")
    (assert-contains gs-src "if (lvl > 0 && lvl <= 2 && k > 0 && u_selong[k-1] <= 0.0) {" "subdivision claim gated to broad/mid tiers, and never claimed by the edge-band overlay")
    ;; EDGE-BAND tier (mirror seed/layer-params + splat-record + stroke-segments): placed
    ;; off the raw edge channel, always takes a side, pushed clear of the ridge, and born
    ;; elongated rather than inheriting the local tensor's anisotropy.
    (assert-contains gs-src "if (sel == 3) return t.b;" "map select 3 = raw edge channel, unnormalized (mirror wavelet/edge-at)")
    (assert-contains gs-src "float se  = (selong > 0.0) ? selong : sqrt(e);" "forced elongation replaces the coherence-derived one for the band tier")
    (assert-contains gs-src "side = (dd > 1e-9) ? 1.0 : (dd < -1e-9) ? -1.0 : (band ? dirsign : 0.0);" "a band seed on the ridge falls back to its direction hash, so both sides get restated")
    (assert-contains gs-src "vec2 so0 = sideOffset(x2, y2, side, soff * ssz2);" "head pushed off the ridge by the level's own side offset")
    (assert-contains gs-src "px = clamp(px + sidem * soff * ssz2 * (-dy), 0.0, float(u_H - 1));" "the side push is re-applied after every in-trace ridge snap")
    (assert-contains gs-src "vec3 headBlur = sampleRGB(u_blurDTex, band ? x2 : cpx + bax, band ? y2 : cpy + bay);" "band drift reference taken at the pushed head, not the pre-snap seed")
    (assert-contains gs-src "float bend = u_curv * 0.9 * bendf * clamp((ssz2 - 2.5) / 2.5, 0.0, 1.0)" "coherence-gated Perlin stroke bend")
    (assert-contains gs-src "float bph = hash01(i*67 + lvl, j, 53);" "per-seed bend phase hash (decorrelates neighbours)")
    (assert-contains gs-src "(1.0 - clamp((ev - 0.3) / 0.3, 0.0, 1.0))" "bend gated by wavelet edge map (ev cached once)")
    (assert-contains gs-src "noise2(0.05*px + 89.0*bph, 0.05*py + 57.0*bph)" "per-seed phase offsets the Perlin bend")
    (assert-contains gs-src "vec2 edgeSnap(float x, float y, float gain){" "edge-ridge snap (damped corrector gain)")
    (assert-contains gs-src "if (snapE) { vec2 sp3 = edgeSnap(px, py, liner ? 0.85 : 0.65); px = sp3.x; py = sp3.y; }" "per-step ridge correction (aggressive on liners, 0.85)")
    (assert-contains gs-src "float mx = 0.35*dx + 0.65*dxp, my = 0.35*dy + 0.65*dyp;" "liner direction momentum")
    (assert-contains gs-src "if (detail && reason == RSN_CHROMA) al *= 0.5;" "stub-glaze demotes only the chroma-backstop chain (reason-based, not length)")
    (assert-contains gs-src "float tt = float(q) / float(max(1, finalLen - 1));" "taper follows the ACTUAL traced length captured in phase 0")
    (assert-contains gs-src "bool liner = (lvl >= 2 && ssz2 < 3.5);" "liner discipline keys on physical stroke size")
    ;; impasto side gate + boundary-side brush-load (mirror seed/stroke-segments):
    ;; the side keys on the LINER discipline (sigma-keyed), not the level index;
    ;; the brush-load is a three-tier decision — no boundary, geometric side, then
    ;; the colour test only at a genuine STEP edge (min dp dm < 0.3*dsides).
    ;; ...and the side is gated on softRamp exactly as seed.clj gates it — the GS used to
    ;; take a side unconditionally AND apply it before softRamp was even computed, so on
    ;; every soft silhouette the GPU pushed the stroke 0.55 sigma off the ridge and the CPU
    ;; reference did not (splat-painter-hr5).
    (assert-contains gs-src "if (snapE && liner && (band || !softRamp)) {"
                     "impasto side keys on liner, not level, and is suppressed on a soft ramp")
    (assert-contains gs-src "float h1 = max(1.75, 0.8 * ssz2);" "probe ladder: rung 1 (narrow)")
    (assert-contains gs-src "float h2 = max(3.0, 1.5 * ssz2);" "probe ladder: rung 2 (mid)")
    (assert-contains gs-src "float h3 = max(5.0, 2.5 * ssz2);" "probe ladder: rung 3 (wide, clears a soft ramp)")
    (assert-contains gs-src "float dmax = max(d1, max(d2, d3));" "sharpness probe: all three rungs, take dmax")
    (assert-contains gs-src "float disp = h1;" "disp = h1 (the nearest offset; crisp samples at h1)")
    ;; the classification is hoisted above the side offset (it keys it), so the ladder
    ;; reads as a predicate there and the brush-load branch consumes it below
    (assert-contains gs-src "bool softRamp = (dmax >= 0.15) && (d1 < 0.75 * dmax);"
                     "sharpness measure: soft ramp when d1 < 0.75*dmax, classified before the side offset")
    (assert-contains gs-src "} else if (softRamp) {" "brush-load reads the hoisted soft-ramp classification")
    (assert-contains gs-src "bax = side * disp * nx0; bay = side * disp * ny0;" "geometric side wins the brush-load")
    (assert-contains gs-src "if (min(dp, dm) < 0.3 * dsides) {" "colour-test brush-load only at a genuine step edge")
    ;; bokeh melt: absolute subjectness drives the broad tier; local-relative sgate
    ;; keeps driving fine placement (it saturates to 1 on smooth bokeh)
    (assert-contains gs-src "float sabs  = subjAbsAt(cx, cy);" "absolute subjectness sample")
    (assert-contains gs-src "float melt = (lvl <= 1) ? clamp((u_broad - 1.0) / 1.5, 0.0, 1.0) * (1.0 - sfoot) : 0.0;" "broad-tier bokeh melt (footprint-gated)")
    (assert-contains gs-src "float bgate = bg0 * bg0;" "Broad-tied mid/fine bokeh gate")
    (assert-contains gs-src "float wsl = softRamp ? 1.0 : (detail ? 1.0" "canvas re-mix incl. melted broad chains; DETAIL tiers re-load colour per segment")
    (assert-contains gs-src "float theta = tc.x, coh0 = tc.y * cohmul;" "melt rounds coherence (and colour) off")
    (assert-contains gs-src "uniform sampler2D u_subjTex;" "absolute-subjectness texture")
    ;; line-hold: liner strokes lift when the sharp map under the brush drops
    (assert-contains gs-src "float mv = (liner && q > 0) ? mapAt(u_sharp[k], px, py) * gain : 1.0;" "line-hold sharp-map sample (liners only, after the seed)")
    (assert-contains gs-src "if (lh) fade = 0.0;" "line-hold brush lift (lh gated on liner && q>0 && mv<0.35*th)")
    ;; liner tier keeps only a hint of the head taper (chains hand off into one rod)
    (assert-contains gs-src "float hw = liner ? 0.8  + 0.2  * smoothstep(0.0, 0.18, tt)" "liner-muted head width taper")
    (assert-contains gs-src "float ha = liner ? 0.75 + 0.25 * smoothstep(0.0, 0.15, tt)" "liner-muted head alpha taper")
    (assert-contains gs-src "float dv = mapAt(u_sharp[k], cx, cy);" "scale-matched detail map per level")
    (assert-contains gs-src "int   segs  = u_segs[k];" "per-level segment count")
    (assert-contains gs-src "if (q >= segs) break;" "trace loops the per-level SEGS cap (no coherence gate, no span target)")
    (assert-contains gs-src "float sgn = (q == 0) ? dirsign : ((dx0*dxp + dy0*dyp) < 0.0 ? -1.0 : 1.0);" "sign-continuous tangent"))

  (println "pack-splats:")
  (let [splats [{:mean [1.0 2.0] :cov [4.0 0.5 0.5 9.0] :color [0.1 0.2 0.3] :alpha 0.7 :detail 0.25}]
        packed (shader/pack-splats splats)]
    (println "  1 splat ->" (count packed) "floats (want 12)")
    (assert (= 12 (count packed)))
    (assert (= [1.0 2.0 4.0 0.5 9.0 0.1 0.2 0.3 0.7 0.25 0.0 0.0] packed))
    (println "  layout [mean_x mean_y c00 c01  c11 r g b  alpha detail 0 0]: OK"))

  (println "pipeline (load eye.jpeg -> seed -> pack):")
  (let [img   (image/load-image "test/splat_painter/fixtures/eye.jpeg" 64)
        fld   (seed/splat-field img {:count 256 :size 3.0})
        n     (count (:splats fld))
        packed (shader/pack-splats (:splats fld))]
    (println (format "  image %dx%d -> %d splats -> %d texture floats"
                     (:width img) (:height img) n (count packed)))
    (assert (pos? n))
    (assert (= (* 3 n 4) (count packed)))))

  (println "widgets registered:"
           (every? #(contains? @w/specs %) [:gl-area :scale]))
  (assert (every? #(contains? @w/specs %) [:gl-area :scale]) "widgets not registered")
  (println "layout invariants (sidebar narrow, sliders live):")
  (let [tree     (core/app)
        kw-node? (fn [n] (and (vector? n) (keyword? (first n))))
        nodes    (filter kw-node? (hiccup-nodes tree))
        props    (fn [n] (let [p (second n)] (if (map? p) p {})))
        scales   (filter #(= :scale (first %)) nodes)
        hexp     (filter #(:hexpand (props %)) nodes)]
    (assert (seq scales) "expected at least one :scale in the control panel")
    (assert (every? #(contains? (props %) :on-value) scales)
            "every slider must wire :on-value (live repaint)")
    (assert (every? #(not (:hexpand (props %))) scales)
            "sliders must not :hexpand — it propagates up and balloons the sidebar")
    (assert (every? #(contains? (props %) :width-request) scales)
            "every slider needs :width-request so the track has size without :hexpand")
    (assert (= 1 (count hexp)) "exactly one :hexpand widget (the GL area)")
    (assert (= :gl-area (first (first hexp))) "the sole :hexpand must be :gl-area")
    (println (format "  %d sliders (on-value wired, no :hexpand, :width-request set); %d :hexpand widget(s)"
                     (count scales) (count hexp))))

  (println "check: ok")
