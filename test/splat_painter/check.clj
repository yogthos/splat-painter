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
  (let [{:keys [vs-src fs-src]} (shader/sources)]
    (println "shader: vs" (count vs-src) "chars, fs" (count fs-src) "chars")
    (assert-contains fs-src "uniform sampler2D u_splats;"    "u_splats")
    (assert-contains fs-src "uniform int  u_count;"          "u_count")
    (assert-contains fs-src "uniform vec2 u_viewport;"       "u_viewport")
    (assert-contains fs-src "uniform vec2 u_image;"          "u_image")
    (assert-contains fs-src "uniform vec3 u_bg;"             "u_bg")
    (assert-contains fs-src "texelFetch(u_splats, ivec2(3 * col,     row), 0);" "texelFetch texel0")
    (assert-contains fs-src "texelFetch(u_splats, ivec2(3 * col + 1, row), 0);" "texelFetch texel1")
    (assert-contains fs-src "texelFetch(u_splats, ivec2(3 * col + 2, row), 0);" "texelFetch texel2 (alpha)")
    (assert-contains fs-src "float det = max(c00 * c11 - c01 * c01, 1e-8);" "det floor")
    (assert-contains fs-src "float p00 = c11 / det, p11 = c00 / det, cross = -2.0 * c01 / det;"
                    "precision (2x2 inverse)")
    (assert-contains fs-src "uniform float u_opacity;"          "u_opacity")
    (assert-contains fs-src "uniform float u_hard_sharp;"       "u_hard_sharp")
    (assert-contains fs-src "uniform float u_hard_soft;"        "u_hard_soft")
    (assert-contains fs-src "float hardness = mix(u_hard_sharp, u_hard_soft, ts);" "size-scaled hardness")
    (assert-contains fs-src "float a = t2.x * u_opacity * exp(-pow(pdf, hardness));" "per-splat alpha × opacity × size-hardness")
    (assert-contains fs-src "acc += wa * t1.yzw;"             "over-composite color accumulation")
    (assert-contains fs-src "T *= (1.0 - a);"                 "transmittance update")
    (assert-contains fs-src "frag = vec4(acc + T * u_bg, 1.0);" "background weighted by T")
    (assert-contains vs-src "void main()"                    "vertex main"))

  ;; the samplerBuffer render variant (consumes the transform-feedback stream directly)
  (let [{:keys [fs-src-buf]} (shader/sources)]
    (println "render (texture-buffer variant):")
    (assert-contains fs-src-buf "uniform samplerBuffer u_splats;" "samplerBuffer u_splats")
    (assert-contains fs-src-buf "texelFetch(u_splats, 3 * i);"     "flat texelFetch texel0")
    (assert-contains fs-src-buf "texelFetch(u_splats, 3 * i + 1);" "flat texelFetch texel1")
    (assert-contains fs-src-buf "texelFetch(u_splats, 3 * i + 2);" "flat texelFetch texel2 (alpha)")
    (assert-contains fs-src-buf "float hardness = mix(u_hard_sharp, u_hard_soft, ts);" "size-scaled hardness"))

  ;; the per-splat quad renderer (no pixels×splats loop — the 48k-hang fix)
  (let [{:keys [vs-src-quad fs-src-quad]} (shader/sources)]
    (println "render (per-splat quad variant):")
    (assert-contains vs-src-quad "int splat  = (u_count - 1) - (gl_VertexID / 6);" "quad back-to-front order")
    (assert-contains vs-src-quad "vec2 he = (3.5 + 2.0 * u_tex_edge) * sqrt(vec2(c00, c11));" "quad marginal-stdev extents (edge-tex dilated)")
    (assert-contains vs-src-quad "float ts  = clamp((sig - u_sig_min) / max(u_sig_max - u_sig_min, 1e-4), 0.0, 1.0);" "quad size→hardness")
    (assert-contains fs-src-quad "float a = v_alpha * u_opacity * exp(-pow(pdf, v_hard));" "quad per-splat alpha formula")
    (assert-contains fs-src-quad "vec3  col = clamp(v_color * bright + chroma, 0.0, 1.0);" "quad paint-texture colour")
    (assert-contains fs-src-quad "frag = vec4(col * a, a);" "quad premultiplied output"))

  ;; the GPU generation shader must MIRROR seed/splat-record + layered-means + noise
  (let [{:keys [vs-src gs-src]} (gen/sources)]
    (println "generation (vertex + geometry, transform feedback):")
    (assert-contains vs-src "v_id = gl_VertexID;" "gen VS passes vertex id")
    (assert-contains gs-src "out vec4 o_a;" "gen TF varying o_a")
    (assert-contains gs-src "out vec4 o_b;" "gen TF varying o_b")
    ;; placement (layered-means): threshold discard + jitter + Perlin warp gate
    (assert-contains gs-src "float thd = th * (0.75 + 0.5 * hash01(i*43 + lvl, j, 19));" "dithered placement threshold")
    (assert-contains gs-src "if (lvl > 0 && dv * gain < thd) return;" "subject-gated threshold discard")
    (assert-contains gs-src "float sgate = subjectAt(cx, cy);" "wavelet subjectness gate")
    (assert-contains gs-src "if (hash01(i*61 + lvl, j, 43) >= bminp*bminp) return;" "bokeh-adaptive broad thinning")
    (assert-contains gs-src "else if (dmx > ((lvl >= 4) ? 0.3 : 0.22)) fade *= 0.4;" "stroke fades at colour drift")
    (assert-contains gs-src "if (dmx > 0.45) fade = 0.0;" "brush lifts at colour-region exit")
    (assert-contains gs-src "if (fade < 0.15) break;" "no emission after the brush lifts")
    (assert-contains gs-src "float al = lal2 * fade * (1.0 - 0.65 * tt * tt);" "taper × glaze × dry-out alpha")
    (assert-contains gs-src "float body = ((lvl >= 4) ? clamp((edgeAt(px, py) - 0.25) / 0.45, 0.0, 1.0) : 0.0)" "impasto body on strong edges")
    (assert-contains gs-src "float Ev = (lvl <= 3) ? edgeNear(cx, cy, 0.75 * ssz) : edgeAt(cx, cy);" "footprint-sensed edge strength")
    (assert-contains gs-src "float cy = float(u_W) * poshash(i, lvl, 31);" "avalanche-hashed candidate y")
    (assert-contains gs-src "float cx = float(u_H) * poshash(i, lvl, 29);" "avalanche-hashed candidate positions")
    (assert-contains gs-src "uint wang32(uint v){" "Wang avalanche hash")
    (assert-contains gs-src "float aw = u_warp * (1.0 - D) * ssz;" "gen warp amplitude")
    ;; hash01 (uint wrap == CPU mod 2^32) + Perlin (permutation texture)
    (assert-contains gs-src "uint(a)*73856093u + uint(b)*19349663u + uint(salt)*83492791u" "gen hash01 constants")
    (assert-contains gs-src "float noise2(float x, float y){ return noise3(x, y, 0.0); }" "gen noise2")
    ;; splat-record: elongation, covariance, colour blend
    (assert-contains gs-src "float e   = 1.0 + min(u_stroke, 1.5) * coh * (0.25 + 0.75 * D);" "gen capped elongation")
    (assert-contains gs-src "float L = ssz2 * stepf * (0.4 + 0.24 * u_stroke);" "stroke-length chain step")
    (assert-contains gs-src "float c00 = sx2*c*c + sy2*s*s;" "gen covariance c00")
    (assert-contains gs-src "float t = min(tcap, max(traw, clamp(0.15 + 0.85 * max(coh0, D), 0.0, 1.0)));" "gen colour blend with per-level raw floor + specificity cap")
    (assert-contains gs-src "float tcap = (lvl <= 1) ? 0.35 : (lvl <= 3) ? 0.7 : 1.0;" "progressive colour-specificity ceiling")
    (assert-contains gs-src "float mapAt(int sel, float x, float y){" "scale-matched map selector")
    (assert-contains gs-src "if (fdv >= u_th[k-1] * (0.75 + 0.5 * hash01(i*47 + lvl, j, 23))) return;" "dithered subdivision claim")
    (assert-contains gs-src "uniform sampler2D u_blurHTex;" "gen heavy-blur texture")
    (assert-contains gs-src "float hb = (lvl <= 1) ? 1.0 : 0.0;" "broad strokes use heavy blur")
    (assert-contains gs-src "o_a = vec4(px, py, c00, c01);" "gen output o_a layout")
    (assert-contains gs-src "o_b = vec4(c11, color.r, color.g, color.b);" "gen output o_b layout")
    (assert-contains gs-src "o_c = vec4(alpha, 0.0, 0.0, 0.0);" "gen output o_c (stroke-taper alpha)")
    ;; the brush-stroke trace (mirror of seed/stroke-segments)
    (assert-contains gs-src "layout(points, max_vertices = 8) out;" "gen GS emits stroke chains")
    (assert-contains gs-src "float sz = ssz2 * (1.0 - 0.45 * tt * sqrt(tt));" "stroke width taper (seed-jittered size)")
    (assert-contains gs-src "float lal = (lvl <= 1) ? 1.0 : (lvl <= 3) ? 0.85 : (lvl <= 5) ? 0.65 : 0.55;" "per-level glaze alpha")
    (assert-contains gs-src "float traw = (lvl <= 1) ? 0.0 : (lvl <= 3) ? 0.45 : (lvl <= 5) ? 0.7 : 0.85;" "per-level raw floor incl. lvl-6 step")
    (assert-contains gs-src "if (lvl > 0 && lvl <= 2 && k > 0) {" "subdivision claim gated to broad/mid tiers")
    (assert-contains gs-src "float bend = u_curv * 0.9 * bendf * (1.0 - 0.7*tc.y) * (noise2(0.05*px, 0.05*py) - 0.5);" "coherence-gated Perlin stroke bend")
    (assert-contains gs-src "vec2 edgeSnap(float x, float y){" "edge-ridge snap")
    (assert-contains gs-src "if (snapE) { vec2 sp3 = edgeSnap(px, py); px = sp3.x; py = sp3.y; }" "per-step ridge correction")
    (assert-contains gs-src "float dv = mapAt(u_sharp[k], cx, cy);" "scale-matched detail map per level")
    (assert-contains gs-src "int   segs  = u_segs[k];" "per-level segment count")
    (assert-contains gs-src "float sgn = (q == 0) ? dirsign : ((dx0*dxp + dy0*dyp) < 0.0 ? -1.0 : 1.0);" "sign-continuous tangent"))

  (println "pack-splats:")
  (let [splats [{:mean [1.0 2.0] :cov [4.0 0.5 0.5 9.0] :color [0.1 0.2 0.3] :alpha 0.7}]
        packed (shader/pack-splats splats)]
    (println "  1 splat ->" (count packed) "floats (want 12)")
    (assert (= 12 (count packed)))
    (assert (= [1.0 2.0 4.0 0.5 9.0 0.1 0.2 0.3 0.7 0.0 0.0 0.0] packed))
    (println "  layout [mean_x mean_y c00 c01  c11 r g b  alpha 0 0 0]: OK"))

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
