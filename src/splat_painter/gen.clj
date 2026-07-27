(ns splat-painter.gen
  "GPU splat GENERATION — the whole splat-field built on the GPU, replacing the CPU
   splat-painter.seed/splat-field per render. A vertex+geometry transform-feedback
   program draws one GL_POINT per candidate cell (splat-painter.seed/layer-params gives
   the per-level grid), the geometry shader threshold-tests each cell and, for the
   survivors, jitters + Perlin-warps the centre, samples the precomputed fields, runs
   the splat-record math, and EmitVertex()s the 8-float splat record. Transform
   feedback CAPTURES only the survivors into a buffer (variable-count compaction with
   no compute/SSBO — see splat-painter.tf-smoke), and a query reports the count. That
   buffer is then bound as a samplerBuffer and consumed by the render shader with no
   readback. The GLSL mirrors splat-painter.seed/splat-record + layered-means and
   splat-painter.noise (hash01 + Perlin), verified against the CPU golden (visually +
   count) — the CPU path stays the tested reference.

   Field textures (uploaded once per image load, all RGBA32F, texelFetch nearest):
     u_detail  W_d×H_d  .r = wavelet detail; normalized by u_dmax
     u_noise   W_n×H_n  .rgba = (theta, coherence, snoise, tnoise) from prep-noise
     u_blur    W×H      .rgb = smooth base colour (light); u_blurH = heavy (broad strokes)
     u_raw     W×H      .rgb = crisp pixel colour
     u_perm    512×1    .r = Perlin permutation (0..255)
   Reduced-res maps (detail/noise) carry their src dims so the shader maps a full-image
   (x,y) into them exactly like the CPU (round(x·dim/src))."
  (:require [glimmer-gl.gl :as gl]
            [splat-painter.seed :as seed]
            [splat-painter.noise :as noise]
            [jolt.ffi :as ffi]))

;; 8, not 7: the ladder itself tops out at 7 rungs (nlev = 1+round(detail·6)) and the
;; EDGE-BAND overlay rides on top of it. At 7 the `pad` below would silently TRUNCATE
;; the finest-first vector, dropping the COARSEST slot — the base fill — and leaving
;; the image unpainted underneath.
(def ^:private max-levels 8)

;; --- generation program: vertex + geometry (transform feedback) --------------
(def ^:private vs-src
  "#version 330 core
flat out int v_id;
void main(){ v_id = gl_VertexID; }")

(def ^:private gs-src
  "#version 330 core
layout(points) in;
layout(points, max_vertices = 32) out;
flat in int v_id[];

// captured by transform feedback: 12 floats = 3 RGBA texels per splat, matching the
// render shaders' t0=(mean_x,mean_y,c00,c01) t1=(c11,r,g,b) t2=(alpha,0,0,0).
out vec4 o_a;
out vec4 o_b;
out vec4 o_c;

const int   ML      = 8;
const float MIN_COH = 0.28;
const int   SEGS    = 32;     // max segments per brush stroke (liner chains, seed/layer-params segs-of cap)

// per-level placement (finest-first slots), from seed/layer-params
uniform int   u_nlev;
uniform float u_ssz[ML];
uniform float u_sp[ML];
uniform float u_th[ML];
uniform int   u_nx[ML];
uniform int   u_ny[ML];
uniform int   u_off[ML];
uniform int   u_lvl[ML];
// scale-relative stroke behaviour per level slot (seed/layer-params: seg-count,
// step-frac, bend-frac, sharp-level?)
uniform int   u_segs[ML];
uniform float u_stepf[ML];
uniform float u_bendf[ML];
uniform int   u_sharp[ML];
// EDGE-BAND tier (mirror seed/layer-params): u_sideo is the side push in units of the
// level's own stdev (0.55 for an ordinary liner, band-sideo for the band); u_selong > 0
// forces the elongation instead of deriving it from the local tensor, and doubles as
// the marker for `is this slot the band overlay`.
uniform float u_sideo[ML];
uniform float u_selong[ML];
uniform float u_warp;
uniform int   u_H;
uniform int   u_W;

// controls
uniform float u_stroke;
uniform float u_variation;
uniform float u_contrast;
uniform float u_detail;   // the Detail slider (deff scale)
uniform float u_curv;     // Curvature raw 0..1 — Perlin bend of the stroke trace
uniform float u_broad;    // Broad slider — bokeh-adaptive broad-tier multiplier

// fields
uniform sampler2D u_detailTex;
uniform sampler2D u_subjTex;  // absolute subjectness (dmap grid, .r)
uniform float u_dmax;
uniform vec2  u_detailDim;   // (H_d, W_d)
uniform vec2  u_detailSrc;   // (src_h, src_w) = image (H, W)
uniform sampler2D u_noiseTex;
uniform vec2  u_noiseDim;
uniform vec2  u_noiseSrc;
uniform sampler2D u_blurTex;  // W×H edge-aware light blur (fine strokes' smooth base)
uniform sampler2D u_blurDTex; // W×H FORGIVING box blur — drift/dry-out probes only
uniform sampler2D u_blurHTex; // W×H HEAVY blur (base + broadest level: colour at their scale)
uniform sampler2D u_rawTex;   // W×H
uniform sampler2D u_permTex;  // 512×1

// --- helpers (mirror splat-painter.noise / seed) ---------------------------
int perm(int i){ return int(texelFetch(u_permTex, ivec2(i, 0), 0).r + 0.5); }
float fade(float t){ return t*t*t*(t*(t*6.0-15.0)+10.0); }
float lerpf(float t, float a, float b){ return a + t*(b-a); }
float grad(int h, float x, float y, float z){
  int hh = h & 15;
  float u = hh < 8 ? x : y;
  float v = hh < 4 ? y : (hh == 12 || hh == 14 ? x : z);
  return ((hh & 1) == 0 ? u : -u) + ((hh & 2) == 0 ? v : -v);
}
float noise3(float x, float y, float z){
  float fx = floor(x), fy = floor(y), fz = floor(z);
  int X = int(fx) & 255, Y = int(fy) & 255, Z = int(fz) & 255;
  float xf = x - fx, yf = y - fy, zf = z - fz;
  float u = fade(xf), v = fade(yf), w = fade(zf);
  int A  = perm(X) + Y,   AA = perm(A) + Z,     AB = perm(A + 1) + Z;
  int B  = perm(X+1) + Y, BA = perm(B) + Z,     BB = perm(B + 1) + Z;
  float n = lerpf(w,
    lerpf(v, lerpf(u, grad(perm(AA),   xf,       yf,       zf),
                      grad(perm(BA),   xf - 1.0, yf,       zf)),
             lerpf(u, grad(perm(AB),   xf,       yf - 1.0, zf),
                      grad(perm(BB),   xf - 1.0, yf - 1.0, zf))),
    lerpf(v, lerpf(u, grad(perm(AA+1), xf,       yf,       zf - 1.0),
                      grad(perm(BA+1), xf - 1.0, yf,       zf - 1.0)),
             lerpf(u, grad(perm(AB+1), xf,       yf - 1.0, zf - 1.0),
                      grad(perm(BB+1), xf - 1.0, yf - 1.0, zf - 1.0))));
  return (1.0 + n) / 2.0;
}
float noise2(float x, float y){ return noise3(x, y, 0.0); }

// Wang avalanche hash (mirror of seed/wang32): position generation needs real bit
// mixing — a linear mix lays the candidate points on Marsaglia lines.
uint wang32(uint v){
  v = (v ^ 61u) ^ (v >> 16);
  v = v * 9u;
  v = v ^ (v >> 4);
  v = v * 668265261u;
  v = v ^ (v >> 15);
  return v;
}
float poshash(int n, int lvl, int salt){
  // top 23 bits only: exactly representable in float32 AND the CPU's doubles
  return float(wang32(wang32(uint(n)*2u + uint(lvl)) ^ (uint(salt) * 2654435769u)) >> 9) / 8388608.0;
}

float hash01(int a, int b, int salt){
  uint h = uint(a)*73856093u + uint(b)*19349663u + uint(salt)*83492791u; // wraps mod 2^32
  return float(h) / 4294967296.0;
}

// bilinear-map a full-image (x,y) into a reduced field grid (dim=(rows,cols),
// src=(src_h,src_w)), CLAMPED at the borders — matches CPU wavelet/bilerp1 exactly:
// floor-based texel selection with lerp (NO GL_LINEAR — hardware filtering would not
// match the CPU). texelFetch(col,row) as before; fx blends the ROW axis, fy the COL.
vec4 fieldBilerp(sampler2D tex, float x, float y, vec2 dim, vec2 src){
  float gx = clamp(x * dim.x / src.x, 0.0, dim.x - 1.0);
  float gy = clamp(y * dim.y / src.y, 0.0, dim.y - 1.0);
  float x0f = floor(gx); float y0f = floor(gy);
  int x0 = int(x0f); int y0 = int(y0f);
  int x1 = min(x0 + 1, int(dim.x) - 1); int y1 = min(y0 + 1, int(dim.y) - 1);
  float fx = gx - x0f; float fy = gy - y0f;
  vec4 v00 = texelFetch(tex, ivec2(y0, x0), 0);   // row x0, col y0
  vec4 v10 = texelFetch(tex, ivec2(y0, x1), 0);   // row x1, col y0
  vec4 v01 = texelFetch(tex, ivec2(y1, x0), 0);   // row x0, col y1
  vec4 v11 = texelFetch(tex, ivec2(y1, x1), 0);   // row x1, col y1
  vec4 r0 = mix(v00, v10, fx);                    // along the ROW axis (x0->x1)
  vec4 r1 = mix(v01, v11, fx);
  return mix(r0, r1, fy);                         // along the COL axis (y0->y1)
}
float detailAt(float x, float y){
  vec4 t = fieldBilerp(u_detailTex, x, y, u_detailDim, u_detailSrc);
  return u_dmax > 0.0 ? min(1.0, t.r / u_dmax) : 0.0;
}
// the SHARP fine-band map (texel .g) — what the finest levels threshold against
float sharpAt(float x, float y){
  vec4 t = fieldBilerp(u_detailTex, x, y, u_detailDim, u_detailSrc);
  return u_dmax > 0.0 ? min(1.0, t.g / u_dmax) : 0.0;
}
// scale-matched map select (mirror of seed/level-map-kind): 0 = aggregate (.r),
// 1 = MID band (.a — face-feature frequencies), 2 = sharp fine-band (.g)
float mapAt(int sel, float x, float y){
  vec4 t = fieldBilerp(u_detailTex, x, y, u_detailDim, u_detailSrc);
  // sel 3 = the EDGE-BAND tier's raw edge channel (.b), returned UNNORMALIZED —
  // mirroring wavelet/edge-at, which unlike the three wavelet bands does not divide
  // by dmax. Routing it through the tail below would silently rescale the tier.
  if (sel == 3) return t.b;
  float v = (sel == 2) ? t.g : (sel == 1) ? max(t.a, t.g) : t.r;  // mid = union with sharp
  return u_dmax > 0.0 ? min(1.0, v / u_dmax) : 0.0;
}
// raw edge strength (texel .b) — the band where broad fill strokes must not tread
float edgeAt(float x, float y){
  return fieldBilerp(u_detailTex, x, y, u_detailDim, u_detailSrc).b;
}
// ABSOLUTE subjectness (mirror wavelet/subject-abs-at): raw globally-scaled
// fine-band energy + edge strength — 0 in bokeh, high on real structure. Drives
// the broad tier's bokeh adaptation; the locally-normalized maps (which light
// bokeh up to full 'detail') keep driving fine-stroke placement.
float subjAbsAt(float x, float y){
  return min(1.0, fieldBilerp(u_subjTex, x, y, u_detailDim, u_detailSrc).r);
}
// MAX edge strength over centre + 4 diagonal taps at radius d (mirror seed/edge-near):
// a stroke answers for edges anywhere under its BODY — centre-sampled Ev let daubs
// seeded just off a silhouette ribbon mixed colour along it (the ghost veil).
float edgeNear(float x, float y, float d){
  float e = edgeAt(x, y);
  e = max(e, edgeAt(x + d, y + d));
  e = max(e, edgeAt(x - d, y - d));
  e = max(e, edgeAt(x + d, y - d));
  e = max(e, edgeAt(x - d, y + d));
  return e;
}
// wavelet SUBJECTNESS (mirror seed/subject-at): smoothed aggregate detail (centre
// + 4 diagonal taps) finds the detailed subjects; the raw centre term keeps thin
// isolated features (wires) alive. 0 = flat bokeh, 1 = detailed subject.
float subjectAt(float x, float y){
  float r  = float(u_H) / 24.0;
  float p0 = detailAt(x, y);
  float ps = 0.2 * (p0 + detailAt(x + r, y + r) + detailAt(x - r, y - r)
                       + detailAt(x + r, y - r) + detailAt(x - r, y + r));
  float s  = clamp((ps - 0.05) / 0.30, 0.0, 1.0);
  return max(s, clamp((p0 - 0.10) / 0.35, 0.0, 1.0));
}

// BILINEAR orientation-field sample (mirrors seed/sample-fields exactly: same
// continuous coord fx = x·dim/src, same floor/clamp). The texture stores cos2θ /
// sin2θ / coherence — double-angle components interpolate undirected orientations
// (0 ≡ π) smoothly, where nearest-sampled raw angles stair-step along contours.
// Returns (theta, coherence).
vec2 fieldsAt(float x, float y){
  float fx = clamp(x * u_noiseDim.x / u_noiseSrc.x, 0.0, u_noiseDim.x - 1.0);
  float fy = clamp(y * u_noiseDim.y / u_noiseSrc.y, 0.0, u_noiseDim.y - 1.0);
  int i0 = int(fx); int i1 = min(i0 + 1, int(u_noiseDim.x) - 1); float wx = fx - float(i0);
  int j0 = int(fy); int j1 = min(j0 + 1, int(u_noiseDim.y) - 1); float wy = fy - float(j0);
  vec3 v00 = texelFetch(u_noiseTex, ivec2(j0, i0), 0).xyz;
  vec3 v01 = texelFetch(u_noiseTex, ivec2(j1, i0), 0).xyz;
  vec3 v10 = texelFetch(u_noiseTex, ivec2(j0, i1), 0).xyz;
  vec3 v11 = texelFetch(u_noiseTex, ivec2(j1, i1), 0).xyz;
  vec3 b = mix(mix(v00, v01, wy), mix(v10, v11, wy), wx);
  return vec2(0.5 * atan(b.y, b.x), clamp(b.z, 0.0, 1.0));
}
// mirror of seed/edge-snap: move a fine-stroke position onto the local edge ridge
// (parabolic peak of edge strength sampled across the tangent). Without it, seeds
// scattered across a thin line braid parallel wobbly strands beside it.
// `gain` damps the corrector (mirror seed): the SEED snap converges at 0.65;
// liner-chain steps correct gently — a strong per-step lateral corrector fought
// the direction momentum and scalloped thin traced lines into wobble.
vec2 edgeSnap(float x, float y, float gain){
  vec2 tc = fieldsAt(x, y);
  float nx = -sin(tc.x), ny = cos(tc.x);
  float h = 1.75;
  float e0 = edgeAt(x, y);
  float ep = edgeAt(x + nx*h, y + ny*h);
  float em = edgeAt(x - nx*h, y - ny*h);
  if (max(e0, max(ep, em)) < 0.12) return vec2(x, y);
  float den = (em + ep) - 2.0*e0;
  float d = (abs(den) < 1e-9) ? 0.0 : clamp((em - ep) / (2.0*den), -1.0, 1.0);
  return vec2(clamp(x + nx*h*d*gain, 0.0, float(u_H - 1)),
              clamp(y + ny*h*d*gain, 0.0, float(u_W - 1)));
}

// IMPASTO meeting line (mirror seed/stroke-segments offset): back a bodied liner
// stroke's centre off the ridge toward its colour-sample side, so the two sides'
// opaque paints MEET at the edge instead of alternating across it.
vec2 sideOffset(float x, float y, float side, float mag){
  if (side == 0.0) return vec2(x, y);
  vec2 tc = fieldsAt(x, y);
  float nx = -sin(tc.x), ny = cos(tc.x);
  return vec2(clamp(x + side*mag*nx, 0.0, float(u_H - 1)),
              clamp(y + side*mag*ny, 0.0, float(u_W - 1)));
}

vec3 sampleRGB(sampler2D tex, float x, float y){   // W×H, 4-tap bilinear (mirror seed/sample-arr)
  float gx = clamp(x, 0.0, float(u_H - 1));
  float gy = clamp(y, 0.0, float(u_W - 1));
  float x0f = floor(gx); float y0f = floor(gy);
  int x0 = int(x0f); int y0 = int(y0f);
  int x1 = min(x0 + 1, u_H - 1); int y1 = min(y0 + 1, u_W - 1);
  float fx = gx - x0f; float fy = gy - y0f;
  vec3 r00 = texelFetch(tex, ivec2(y0, x0), 0).rgb;   // row x0, col y0
  vec3 r10 = texelFetch(tex, ivec2(y0, x1), 0).rgb;   // row x1, col y0
  vec3 r01 = texelFetch(tex, ivec2(y1, x0), 0).rgb;   // row x0, col y1
  vec3 r11 = texelFetch(tex, ivec2(y1, x1), 0).rgb;   // row x1, col y1
  vec3 r0 = mix(r00, r10, fx);                        // along the ROW axis (x0->x1)
  vec3 r1 = mix(r01, r11, fx);
  return mix(r0, r1, fy);                             // along the COL axis (y0->y1)
}

// splat-record (mirror seed/splat-record) + emit one captured record. `alpha` is the
// stroke taper (1.0 for base fills, fading toward a fine stroke's tail).
// (hx,hy) = the chain HEAD's position — the colour-sample point for EVERY segment:
// one stroke = one brush-load of paint (per-segment sampling alternated the two
// sides' colours along contours into a bead necklace).
// `cohmul` rounds MELTED bokeh strokes off (coherence → 0 kills the elongation
// and pulls the colour toward the smooth blur): an elongated needle on a soft
// gradient always reads as a directional streak, however faithful its colour.
void emitSplat(float px, float py, float hx, float hy, float csz, float D, float sn, float tn, float alpha, float hb, float traw, float tcap, float cohmul, float selong){
  vec2  tc    = fieldsAt(px, py);
  float theta = tc.x, coh0 = tc.y * cohmul;
  vec3  bilat = sampleRGB(u_blurTex, hx, hy);
  vec3  blur  = (hb > 0.5) ? sampleRGB(u_blurHTex, hx, hy) : bilat;
  vec3  raw   = sampleRGB(u_rawTex,  hx, hy);
  // REGION-CONSISTENCY clamp (mirror seed/splat-field): the bilateral DEFINES the
  // region; raw specificity is only trusted when consistent with it. In-region
  // micro-contrast keeps full raw pop; an edge-straddling raw sample is pulled to
  // the region colour so it cannot bleed across a boundary at high raw weight.
  // SIZE-AWARE: the clamp exists because a FAT stroke sampling one raw pixel that
  // disagrees with its region smears a blotch across its whole sigma. A 1-2px dab is
  // the size of the feature it is painting — its raw sample IS the detail, not an
  // outlier — and clamping it erased fine detail exactly where detail lives (an eye
  // or a glasses rim easily exceeds dcl 0.27, which pinned raw to the bilateral blur
  // and is why small features read soft however sharp the mark). Fade the clamp in
  // with stroke size: off at <=1.5px, full by 4.5px.
  vec3  dr    = abs(raw - bilat);
  float dcl   = max(dr.r, max(dr.g, dr.b));
  float sizew = clamp((csz - 1.5) / 3.0, 0.0, 1.0);
  float wcl   = clamp((dcl - 0.12) / 0.15, 0.0, 1.0) * sizew;
  raw = mix(raw, bilat, wcl);
  float coh = MIN_COH + (1.0 - MIN_COH) * coh0;
  float e   = 1.0 + min(u_stroke, 1.5) * coh * (0.25 + 0.75 * D);
  // selong > 0 REPLACES the coherence-derived elongation (mirror seed/splat-record):
  // the EDGE-BAND tier is BORN long-and-thin, so its across-edge sigma is a property
  // of the tier (csz/selong) rather than of whatever coherence happens to sit there.
  float se  = (selong > 0.0) ? selong : sqrt(e);
  float s0  = csz * (1.0 + u_variation * 0.5 * (2.0 * sn));
  float sx  = s0 * se;
  float sy  = s0 / se;
  float sx2 = sx*sx, sy2 = sy*sy;
  float c = cos(theta), s = sin(theta);
  float c00 = sx2*c*c + sy2*s*s;
  float c01 = (sx2 - sy2)*c*s;
  float c11 = sx2*s*s + sy2*c*c;
  // floored by the level's rawness (traw), ceilinged by its specificity cap
  // (tcap) — progressive colour: broad averaged, fine specific (mirror seed).
  // The cap also follows the BRUSH SIZE: a fat brush cannot place a
  // pixel-specific highlight — past ~4px stdev the ceiling eases to averaged.
  float tcap2 = min(tcap, 0.3 + 0.7 * min(1.0, 3.0 / max(csz, 1e-6)));
  float t = min(tcap2, max(traw, clamp(0.15 + 0.85 * max(coh0, D), 0.0, 1.0)));
  vec3 color0 = mix(blur, raw, t);
  vec3 colorAc = (u_contrast == 1.0) ? color0 : clamp((color0 - 0.5)*u_contrast + 0.5, 0.0, 1.0);
  float tone = 1.0 + u_variation * 0.15 * (2.0 * tn);
  // per-stroke TEMPERATURE (mirror seed/splat-record): each brush-load leans warm
  // (R up, B down) or cool. sn is the per-stroke seed noise, so this is exact parity.
  float temp = u_variation * 0.10 * (2.0 * sn);
  vec3 color = clamp(colorAc * tone * vec3(1.0 + temp, 1.0, 1.0 - temp), 0.0, 1.0);
  o_a = vec4(px, py, c00, c01);
  o_b = vec4(c11, color.r, color.g, color.b);
  o_c = vec4(alpha, 0.0, 0.0, 0.0);
  EmitVertex();
  EndPrimitive();
}

void main(){
  int v = v_id[0];
  // decode candidate index -> finest-first level slot k. CONSTANT loop bound (ML) so
  // the compiler can fully unroll: with a uniform bound (u_nlev) + break, Apple's GLSL
  // mis-executed the last slot in this geometry shader — the base level (slot 4 at five
  // levels) emitted NOTHING, which truncated the underpainting and left gaps.
  int k = 0;
  for (int m = 0; m < ML; m++) {
    if (m < u_nlev && v >= u_off[m] && v < u_off[m] + u_nx[m] * u_ny[m]) { k = m; break; }
  }
  int local = v - u_off[k];
  int i = local / u_ny[k];
  int j = local - i * u_ny[k];
  int lvl = u_lvl[k];
  float sp = u_sp[k], ssz = u_ssz[k], th = u_th[k];
  // white-noise candidate position (mirror of seed/layered-means): hashed,
  // aperiodic — any lattice's row frequency reads as parallel stripes.
  float cx = float(u_H) * poshash(i, lvl, 29);
  float cy = float(u_W) * poshash(i, lvl, 31);
  // wavelet subjectness (mirror of seed/layered-means): the LOCAL-relative gate
  // for mid/fine placement, and the ABSOLUTE gate for the bokeh-adaptive broad
  // tier (the local-relative one saturates to 1 on smooth bokeh, which made
  // Broad growth/thinning/melt inert exactly where they exist to act).
  float sgate = subjectAt(cx, cy);
  float sabs  = subjAbsAt(cx, cy);
  // Broad growth gated by the whole GROWN FOOTPRINT (mirror seed): a daub centred
  // in the bokeh just off a silhouette reads flat, grows, and its body reaches
  // back across the edge — 8 taps at the grown radius on the (wide, blurred)
  // subject map catch the nearby subject that centre sampling misses.
  float sfoot = sabs;
  if (lvl <= 1 && u_broad > 1.0) {
    float m0 = 1.0 + (u_broad - 1.0) * (1.0 - sabs);
    float d  = 1.2 * ssz * m0;
    float dd = 0.7071 * d;
    sfoot = max(sfoot, max(subjAbsAt(cx + d, cy), subjAbsAt(cx - d, cy)));
    sfoot = max(sfoot, max(subjAbsAt(cx, cy + d), subjAbsAt(cx, cy - d)));
    sfoot = max(sfoot, max(subjAbsAt(cx + dd, cy + dd), subjAbsAt(cx - dd, cy - dd)));
    sfoot = max(sfoot, max(subjAbsAt(cx + dd, cy - dd), subjAbsAt(cx - dd, cy + dd)));
  }
  float mloc  = 1.0 + (u_broad - 1.0) * (1.0 - sfoot);
  // broad tier: flat regions thin candidates by (bmin/m)² as the kept seeds grow
  // ×m — few LARGE daubs = smooth bokeh; at full subjectness m=1 and the Broad
  // dial has no effect on the detailed subjects.
  if (lvl <= 1) {
    float bminp = min(1.0, u_broad) / mloc;
    if (hash01(i*61 + lvl, j, 43) >= bminp*bminp) return;
    ssz *= mloc;
  }
  // mid/fine strokes belong where the wavelets see detail: flat bokeh keeps only
  // the big smooth daubs. The ABSOLUTE gate rides the Broad slider (mirror seed):
  // past 1.0 it thins mid/fine marks out of truly flat regions — isolated dark
  // flecks on a melted wash — leaving them at Broad <= 1 where strokes are the
  // wanted effect.
  // SQUARED (mirror seed): the linear form bottomed out too weak to ever block
  // accents whose locally-normalized maps light on bokeh noise. Gates LEVEL 1
  // too — its chains were the fibrous filament texture over empty regions.
  float bg0 = 1.0 - clamp((u_broad - 1.0) / 1.5, 0.0, 1.0) * (1.0 - min(1.0, sabs / 0.35));
  float bgate = bg0 * bg0;
  float gain = (lvl >= 2) ? (0.25 + 0.75 * sgate) * bgate : (lvl == 1) ? bgate : 1.0;
  // each level reads the map matched to ITS scale (mirror of seed/layered-means).
  // The cutoff is DITHERED ±25% per seed: a hard threshold on a map that oscillates
  // around it dashes contours into bead necklaces; dithering turns the knife edge
  // into a smooth density ramp.
  float dv = mapAt(u_sharp[k], cx, cy);
  float thd = th * (0.75 + 0.5 * hash01(i*43 + lvl, j, 19));
  if (lvl > 0 && dv * gain < thd) return;         // not detailed enough -> discard
  // SUBDIVISION (mirror of seed/layered-means), broad/mid tiers only: a cell claimed
  // by the next-finer level (slot k-1 — slots are finest-first) is not painted by
  // this coarser level; dithered so the handoff interleaves. From level 3 up there
  // is NO claim — the fine glazes overlap the mid strokes and mix. The finer level
  // is always >= 2 here, so its claim carries the same subject gate.
  // u_selong[k-1] <= 0: the EDGE-BAND overlay is not a rung of the ladder, so it never
  // claims a cell away from the level below it (mirror seed/layered-means claimed?).
  if (lvl > 0 && lvl <= 2 && k > 0 && u_selong[k-1] <= 0.0) {
    float fdv = mapAt(u_sharp[k-1], cx, cy) * (0.25 + 0.75 * sgate) * bgate;
    if (fdv >= u_th[k-1] * (0.75 + 0.5 * hash01(i*47 + lvl, j, 23))) return;
  }

  // hashed positions need no jitter — they ARE the noise
  float x = cx, y = cy;
  float D = min(1.0, u_detail * dv * 2.2);
  // no Perlin warp on liner-scale levels (mirror seed/liner-scale?): fine seeds land exactly on
  // the detail they trace; noise variation belongs to large/medium strokes
  float aw = (lvl >= 2 && ssz < 3.5) ? 0.0 : u_warp * (1.0 - D) * ssz;   // no Perlin warp on liner-scale levels (mirror seed/liner-threshold)
  float x2 = (aw < 0.2) ? x : x + aw * noise2(0.06*x, 0.06*y);
  float y2 = (aw < 0.2) ? y : y + aw * noise2(41.3 + 0.06*x, 17.9 + 0.06*y);
  x2 = clamp(x2, 0.0, float(u_H - 1));
  y2 = clamp(y2, 0.0, float(u_W - 1));

  // per-seed jitter is hashed (independent per stroke), mirroring seed/layered-means.
  // Size jitter applies at SEED level to the whole chain (size AND step) so chains
  // stay self-overlapping at any Variation; broad levels keep 40% of both jitters.
  float sn0 = hash01(i*31 + lvl, j, 11) - 0.5;
  // MELT (mirror seed): how much a flat-region broad stroke sinks into the wash;
  // footprint-gated so strokes touching a subject keep their identity. Computed
  // here so the size jitter below can mute with it.
  float melt = (lvl <= 1) ? clamp((u_broad - 1.0) / 1.5, 0.0, 1.0) * (1.0 - sfoot) : 0.0;
  float szf = max(0.75, 1.0 + u_variation * sn0 * ((lvl <= 1) ? 0.4 * (1.0 - melt) : 1.0));
  // near a strong edge: the mid fill levels don't paint at all (their boundary-band
  // chains ribbon mixed colour along silhouettes as a ghost veil — the edge belongs
  // to base coverage below and fine strokes above), and base daubs SHRINK so their
  // soft tails can't reach across the silhouette.
  // edge strength sensed over the stroke's FOOTPRINT (taps at 0.75·size), not just
  // its centre; level 1's opaque heavy-blur ribbons suppress at 90% vs level 2's 75%
  float Ev = (lvl <= 3) ? edgeNear(cx, cy, 0.75 * ssz) : edgeAt(cx, cy);
  if ((lvl == 1 || lvl == 2) && Ev > 0.45
      && hash01(i*53 + lvl, j, 37) < ((lvl == 1) ? 0.9 : 0.75)) return;
  // fat strokes shrink near edges so soft tails can't cross the silhouette; the
  // fine liner strokes (lvl>=4) ARE the edge's own paint and keep their size.
  // The BASE is the COVERAGE layer: it shrinks gently (>=0.75x, spacing still
  // seals) so paint always reaches the boundary — no unpainted moat.
  // σ-aware (mirror seed): small strokes keep the gentle coefficients, but a fat
  // low-budget daub centred on a thin bright feature would smear it 20px past the
  // silhouette as an opaque ghost cloud — past ~8px stdev the shrink strengthens.
  float ssz2 = ssz * szf * (1.0 - min(0.7, ((lvl == 0) ? 0.25 : (lvl <= 3) ? 0.45 : 0.1)
                                           * max(1.0, ssz * szf / 8.0)) * Ev);
  float snoise = 0.0;
  // tone jitter follows the PHYSICAL stroke scale (mirror seed): any small mark
  // keeps 15% regardless of which level painted it — full jitter on small
  // strokes beads contours into speckle.
  float tnoise = (hash01(i*37 + lvl, j, 13) - 0.5)
               * ((lvl <= 1) ? 0.25 * (1.0 - melt) : (lvl >= 4) ? 0.15
                  : (0.15 + 0.85 * clamp((ssz2 - 2.5) / 2.5, 0.0, 1.0)));

  // broad strokes (base + level 1) colour from the HEAVY blur — smoothed at their scale
  float hb = (lvl <= 1) ? 1.0 : 0.0;
  // colour-rawness floor rises with fineness (seed/raw-floor): small strokes paint
  // faithful colour — a half-blur blend at feature scale softens the feature away
  // EXPERIMENT: colour ladder keyed on PHYSICAL size, not level index. Once the
  // monotone ladder ends at level 2-3, index-keyed tiering gives the FINEST strokes
  // mid-tier averaged colour (t capped at 0.7) — blur at exactly the scale we want
  // fidelity. Size-keyed restores small-stroke faithful colour.
  // COVERAGE tiers (lvl<=1) take the coverage constant by ROLE, not size: a small
  // base daub must still paint faithful, fully-covering colour (seed/raw-floor). At
  // Size 6 the base lands ~6.7px — under the 8px size threshold — so size-keying
  // gave it mid-tier averaged colour (t 0.45) that smeared edge colour outward.
  float traw = (lvl <= 1) ? 0.0 : (ssz2 < 1.5) ? 0.85 : (ssz2 < 3.5) ? 0.7 : (ssz2 < 8.0) ? 0.45 : 0.0;
  // DENSITY-SCALED traw (mirror seed, spec-lip-band): in a feature-DENSE region
  // (sharpAt high) a single raw sample is unreliable — it lands on a shadow or lip
  // line 3-5px from the next feature and paints a foreign dark mark (the band above
  // the upper lip). Trust the bilateral (region) colour more there; an isolated
  // crisp feature (low sharpAt) keeps full raw fidelity.
  if (lvl >= 2) traw *= 1.0 - 0.7 * sharpAt(cx, cy);
  // colour-specificity ceiling per level (mirror seed/spec-cap): broad layers
  // paint AVERAGED colour, mids halfway, fine layers fully specific
  float tcap = (lvl <= 1) ? 0.35 : (ssz2 < 3.5) ? 1.0 : (ssz2 < 8.0) ? 0.7 : 0.35;
  if (lvl == 0) {                                 // base fill: one full-alpha splat
    emitSplat(x2, y2, x2, y2, ssz2, D, snoise, tnoise, 1.0, hb, traw, tcap, 1.0 - melt, 0.0);
    return;
  }

  // fine level: TRACE A BRUSH STROKE (mirror seed/stroke-segments) — SEGS segments
  // stepped along the orientation field (the edge tangent), sign-continuous, bent by
  // smooth Perlin noise scaled by Curvature, size+alpha tapering toward the tail.
  float dirsign = hash01(i*41 + lvl, j, 17) < 0.5 ? 1.0 : -1.0;
  float bph = hash01(i*67 + lvl, j, 53);         // per-seed bend phase: decorrelates neighbours
  int   segs  = u_segs[k];                       // scale-relative stroke behaviour:
  float stepf = u_stepf[k];                      // broad levels stroke long and curl,
  float bendf = u_bendf[k];                      // fine levels make short precise marks
  bool snapE = (lvl >= 2);                       // fine strokes glue to the edge ridge
  // LINER discipline keys on PHYSICAL stroke size, not level index (mirror seed):
  // which level paints fine detail depends on the sliders — any small accent
  // chain must follow the original detail exactly or it reads as waviness.
  bool liner = (lvl >= 2 && ssz2 < 3.5);              // physical (mirror seed/liner-scale?); lvl<2 never a liner
  // colour samples the PRE-snap position (one side of the edge); geometry snaps.
  // On-ridge colour is the sides' mix and paints silhouettes as drawn outlines.
  float cpx = x2, cpy = y2;
  if (snapE) { vec2 sp2 = edgeSnap(x2, y2, 0.65); x2 = sp2.x; y2 = sp2.y; }
  // EDGE-BAND tier (mirror seed/stroke-segments): selong > 0 marks the overlay slot.
  float selong = u_selong[k];
  bool  band   = selong > 0.0;
  // which side of the ridge did this seed come from? (mirror seed/stroke-segments)
  float side = 0.0;
  // IMPASTO side keys on the LINER discipline (sigma-keyed), not the level index:
  // small lvl 2-3 chains snap onto the ridge like lvl>=4 liners and keep their side.
  // A BAND seed that lands exactly on the ridge falls back to its own direction hash,
  // so the two sides get restated in roughly equal numbers with no side detection.
  if (snapE && liner) {
    vec2 tcs = fieldsAt(x2, y2);
    float nsx = -sin(tcs.x), nsy = cos(tcs.x);
    float dd = (cpx - x2)*nsx + (cpy - y2)*nsy;
    side = (dd > 1e-9) ? 1.0 : (dd < -1e-9) ? -1.0 : (band ? dirsign : 0.0);
  }
  // how far off the ridge, in units of the level's own stdev. A liner nudges 0.55 sigma;
  // the BAND is pushed clear — at selong 2.6 its across-body sigma is ssz/2.6, so even
  // the smallest push here keeps the stroke off the ridge it restates. Jittered by the
  // seed's bend hash (bendf is 0 for the band, so bph is otherwise unused) so band
  // strokes TILE the band instead of stacking into one hard line — SQUARED, so the
  // distribution crowds the near zone, where the measured outward excess actually is
  // (+19.8 luma at 1px out, +2.8 at 10px). (mirror seed/stroke-segments)
  float soff = band ? u_sideo[k] * (0.6 + 2.55 * bph * bph) : u_sideo[k];
  vec2 so0 = sideOffset(x2, y2, side, soff * ssz2);
  x2 = so0.x; y2 = so0.y;
  // side sign in the stroke's MOTION frame (mirror seed): stays consistent
  // through field sign flips that would wobble a per-step theta resample
  float sidem = side * dirsign;
  float px = x2, py = y2, dxp = 0.0, dyp = 0.0;
  // progressive refinement: finer layers GLAZE (translucent touches over the
  // accumulated underpainting) instead of overwriting it; the overlapping fine
  // tier glazes lightest so stacked strokes MIX (mirror seed/level-alpha)
  // paint translucency by PHYSICAL size (mirror seed/level-alpha): broad = opaque
  // coverage, mid = glaze, DABS near-opaque because they are placed sparsely and each
  // must stand on its own — glaze alpha there just averages neighbours into mush.
  // coverage tiers (lvl<=1) are fully opaque by role — size-keying at small Size
  // gave the base 0.85 alpha and let the black background through around silhouettes.
  // the BAND tier is opaque by ROLE, like the coverage tiers (mirror seed): it exists to
  // COVER the outward bleed, and a glaze there averages the bleed back in.
  float lal = (band || lvl <= 1) ? 1.0 : (ssz2 >= 8.0) ? 1.0 : (ssz2 >= 2.5) ? 0.85 : 0.95;
  float fade = 1.0;
  // NO tensor-coherence gate on chain length (mirror seed/stroke-segments). Tried and
  // removed: coherence does not discriminate a line from a smooth gradient, because a
  // gradient is rank-1 and so reads as perfectly ORIENTED. Measured on the test
  // portrait: strong-edge median 0.95 vs FLAT median 0.72. It was a no-op where it was
  // meant to act (0.05/255 on the render), so the nominal segs stands.
  // BOUNDARY-SIDE BRUSH-LOAD (mirror seed/stroke-segments): where the two sides
  // across the tangent genuinely differ (a boundary, not a thin LINE feature),
  // the brush-load samples ~0.7 sigma on the stroke's OWN colour side — chains
  // parallel to a boundary otherwise alternate sides per seed and tile the
  // contour into colour capsules (the regular wavy scallops).
  float bax = 0.0, bay = 0.0; bool softRamp = false;
  {
    vec2 tc0 = fieldsAt(cpx, cpy);
    float nx0 = -sin(tc0.x), ny0 = cos(tc0.x);
    // SHARPNESS, not contrast (mirror seed). Probe ALL THREE rungs of the ridge
    // (no early-out) and keep the fraction of the total transition completing within
    // h1: crisp = (d1 >= 0.75*dmax). A hard step / 1px-AA edge reads d1~=d3 (ratio ~1
    // -> CRISP, two opaque paints meet, impasto kept); a wide ramp reads d1<<d3 (ratio
    // ~0.35 -> SOFT RAMP, no meeting line -> paint LOCAL colour, damp the body).
    // Contrast-invariant, so an 8px and a 24px ramp both classify soft — the fixture
    // stays at the spec's 8px. dmax<0.15 -> a thin LINE feature / flat. cp/cm are taken
    // at h1 (the nearest offset) for the crisp step-edge test. Unrolled, NOT a
    // loop+break: Apple GL 4.1 mis-executes the last slot of a uniform-bound loop.
    float h1 = max(1.75, 0.8 * ssz2);
    float h2 = max(3.0, 1.5 * ssz2);
    float h3 = max(5.0, 2.5 * ssz2);
    vec3 cp1 = sampleRGB(u_blurTex, cpx + nx0*h1, cpy + ny0*h1);   // rung 1
    vec3 cm1 = sampleRGB(u_blurTex, cpx - nx0*h1, cpy - ny0*h1);
    float d1 = max(abs(cp1.r - cm1.r), max(abs(cp1.g - cm1.g), abs(cp1.b - cm1.b)));
    vec3 cp2 = sampleRGB(u_blurTex, cpx + nx0*h2, cpy + ny0*h2);   // rung 2
    vec3 cm2 = sampleRGB(u_blurTex, cpx - nx0*h2, cpy - ny0*h2);
    float d2 = max(abs(cp2.r - cm2.r), max(abs(cp2.g - cm2.g), abs(cp2.b - cm2.b)));
    vec3 cp3 = sampleRGB(u_blurTex, cpx + nx0*h3, cpy + ny0*h3);   // rung 3
    vec3 cm3 = sampleRGB(u_blurTex, cpx - nx0*h3, cpy - ny0*h3);
    float d3 = max(abs(cp3.r - cm3.r), max(abs(cp3.g - cm3.g), abs(cp3.b - cm3.b)));
    float dmax = max(d1, max(d2, d3));
    float disp = h1;
    vec3 cp = cp1, cm = cm1;                 // nearest-offset colour probes (for crisp)
    float dsides = dmax;
    if (dmax < 0.15) {
      bax = 0.0; bay = 0.0;                  // thin LINE feature / flat: on-ridge colour
    } else if (d1 < 0.75 * dmax) {
      bax = 0.0; bay = 0.0; softRamp = true;   // SOFT RAMP: paint LOCAL colour (wsl=1 below), damp body
    } else if (side != 0.0) {
      bax = side * disp * nx0; bay = side * disp * ny0;   // geometric side wins
    } else {                                  // colour test only at a genuine STEP edge
      vec3 c0 = sampleRGB(u_blurTex, cpx, cpy);
      vec3 dpa = abs(cp - c0);
      vec3 dma = abs(cm - c0);
      float dp = max(dpa.r, max(dpa.g, dpa.b));
      float dm = max(dma.r, max(dma.g, dma.b));
      if (min(dp, dm) < 0.3 * dsides) {
        float sidec = (dp < dm) ? 1.0 : -1.0;
        bax = sidec * disp * nx0;
        bay = sidec * disp * ny0;
      }
    }
  }
  // drift reference + probes read the FORGIVING box field (mirror seed): on the
  // razor-sharp bilateral paint field any probe wobble across a boundary trips
  // the lift instantly and dashes contour chains into beads
  // the drift reference belongs where the stroke actually STARTS (mirror seed): for a
  // band stroke that is the pushed-off-ridge head (x2,y2). Referencing the pre-snap
  // seed, which can sit on the FAR side of the boundary, would make the chroma backstop
  // fire on step 1 and the band would never trace at all.
  vec3 headBlur = sampleRGB(u_blurDTex, band ? x2 : cpx + bax, band ? y2 : cpy + bay);
  // FEATURE-FOLLOWING TRACER (mirror seed/stroke-segments): a stroke ends where the
  // FEATURE ends or turns — CLEAN breaks, no segment emitted at the break — so a contour
  // chunks into the strokes a draughtsman draws (the top of an eye one line, the bottom
  // another). Detail tiers (lvl>=2) replace the old fixed-span cap, TURN-KILL, the racc
  // path-roughness accumulator and the coherence-follow fade with GEOMETRIC stops + a
  // single chroma BACKSTOP; broad/coverage tiers keep their two-tier dry-out. LINE-HOLD
  // stays. The walk is fully deterministic, so phase 0 counts the traced length and
  // captures the stop reason; phase 1 re-walks the identical path and emits with the
  // ACTUAL-length taper. (mirror seed two-phase.) GLSL has no returns from a loop, so
  // the stop reason rides in an int and every stop is a clean break.
  const int RSN_CAP = 0, RSN_RIDGE = 1, RSN_CORNER = 2, RSN_CHROMA = 3, RSN_LH = 4, RSN_DRIFT = 5;
  int reason   = RSN_CAP;
  int finalLen = 0;
  for (int phase = 0; phase < 2; phase++) {
    px = x2; py = y2; dxp = 0.0; dyp = 0.0; fade = 1.0;
    int emitted = 0;
    for (int q = 0; q < SEGS; q++) {
      if (q >= segs) break;                       // per-level SEGS cap (mirror seed kmax = segs-1)
      vec2  tc  = fieldsAt(px, py);
      float dx0 = cos(tc.x), dy0 = sin(tc.x);
      float ev  = edgeAt(px, py);
      bool  detail = (lvl >= 2);
      // GEOMETRIC STOPS — clean break, no segment emitted (mirror seed):
      //   edge-floor 0.10: the ridge died, the feature has ended -> :ridge
      //   bend-cos 0.90: |dot(field-dir, prev-step)| < 0.90 (~26deg/step) -> a :corner
      if (q > 0 && detail) {
        if (ev < 0.10) { reason = RSN_RIDGE; break; }
        if (abs(dx0*dxp + dy0*dyp) < 0.90) { reason = RSN_CORNER; break; }
      }
      // COLOUR is a BACKSTOP, not a stop signal (mirror seed): along a real feature the
      // colour SHOULD change, so the old two-tier dry-out + racc fired after 2-4 segs in
      // detailed areas and WAS the 'short disjointed lines' artifact. Detail tiers keep
      // only a HARD stop at runaway 0.60 (the stroke wandered into a foreign colour
      // region); broad/coverage tiers keep their two-tier dry-out unchanged.
      float dmx = 0.0;
      if (q > 0) {
        vec3 cb = sampleRGB(u_blurDTex, px + bax, py + bay);
        vec3 dcl = abs(cb - headBlur);
        dmx = max(dcl.r, max(dcl.g, dcl.b));
      }
      bool chroma = (detail && q > 0 && dmx > 0.60);
      if (q > 0) {
        if (chroma) fade = 0.0;                   // detail tiers: only the chroma backstop
        else if (!detail) {
          if (dmx > 0.18) fade = 0.0; else if (dmx > 0.22) fade *= 0.4;   // broad two-tier dry-out
        }
      }
      // LINE-HOLD stays — the one non-geometric stop (mirror seed): a liner that walked
      // off ITS OWN placement map has left the feature, so lift. It reads the level's map
      // (the right signal); the old coherence-follow fade is removed (a gradient is
      // rank-1 = perfectly oriented, so coherence never separated a line from bokeh).
      float mv = (liner && q > 0) ? mapAt(u_sharp[k], px, py) * gain : 1.0;
      bool  lh = (liner && q > 0 && mv < 0.35 * th);
      if (lh) fade = 0.0;
      else if (liner && q > 0 && mv < 0.7 * th) fade *= 0.5;
      if (fade < 0.15) {
        reason = chroma ? RSN_CHROMA : (lh ? RSN_LH : RSN_DRIFT);
        break;                                   // brush lifted — emit nothing at this step
      }
      // EMIT (phase 1 only). tt follows the ACTUAL traced length captured in phase 0
      // (finalLen), so a corner-broken stroke still gets a full head->tail profile.
      if (phase == 1) {
        float tt = float(q) / float(max(1, finalLen - 1));
        float body = (liner ? clamp((ev - 0.25) / 0.45, 0.0, 1.0) : (0.4 + 0.6 * sgate))
                   * (softRamp ? 0.35 : 1.0);   // SOFT RAMP: a gradient has no meeting line
        float lal2 = lal + max(0.0, 0.9 - lal) * body;
        float hw = liner ? 0.8  + 0.2  * smoothstep(0.0, 0.18, tt)
                         : 0.55 + 0.45 * smoothstep(0.0, 0.18, tt);
        float ha = liner ? 0.75 + 0.25 * smoothstep(0.0, 0.15, tt)
                         : 0.5  + 0.5  * smoothstep(0.0, 0.15, tt);
        float sz = ssz2 * (1.0 - 0.45 * tt * sqrt(tt)) * hw;
        float al = lal2 * fade * (1.0 - 0.65 * tt * tt) * ha;
        // DETAIL tiers RE-LOAD colour each segment (wsl=1): one brush-load per stroke is
        // right for a gestural mark, but at detail scale (~8px span) it drags the head's
        // colour onto the next feature (nostril/philtrum/lip line 3-5px apart on a face).
        float wsl = softRamp ? 1.0 : (detail ? 1.0 : ((melt > 0.0) ? 0.85 * melt * tt : 0.0));
        // stub-glaze (mirror seed/stub-glaze): only a chain that died to the chroma
        // BACKSTOP — it FAILED to follow a feature — is demoted to glaze (x0.5). A
        // ridge/corner/cap/line-hold stop is a COMPLETE short stroke at full alpha.
        if (detail && reason == RSN_CHROMA) al *= 0.5;
        // offset drops for re-loading tiers: a CARRIED brush-load (wsl<1) samples
        // cpx+bax to escape the AA ramp; a per-segment RELOAD (wsl>=1) samples its own
        // position, where the fixed offset would draw the dark ground through a lit shape.
        emitSplat(px, py, cpx + bax*(wsl < 1.0 ? 1.0 : 0.0) + wsl*(px - cpx),
                        cpy + bay*(wsl < 1.0 ? 1.0 : 0.0) + wsl*(py - cpy),
                  sz, D, snoise, tnoise, al, hb, traw, tcap, 1.0 - melt, selong);
      }
      emitted++;
      // bend gated by coherence, physical size AND the wavelet edge map (mirror seed);
      // per-seed phase decorrelates neighbours into natural wobble instead of a weave
      float bend = u_curv * 0.9 * bendf * clamp((ssz2 - 2.5) / 2.5, 0.0, 1.0)
                 * (1.0 - 0.7*tc.y)
                 * (1.0 - clamp((ev - 0.3) / 0.3, 0.0, 1.0))
                 * (noise2(0.05*px + 89.0*bph, 0.05*py + 57.0*bph) - 0.5);
      float cb = cos(bend), sb = sin(bend);
      float sgn = (q == 0) ? dirsign : ((dx0*dxp + dy0*dyp) < 0.0 ? -1.0 : 1.0);
      float dx1 = sgn*dx0, dy1 = sgn*dy0;
      float dx = cb*dx1 - sb*dy1, dy = sb*dx1 + cb*dy1;
      if (liner && q > 0) {
        float mx = 0.35*dx + 0.65*dxp, my = 0.35*dy + 0.65*dyp;
        float ml = sqrt(mx*mx + my*my);
        if (ml > 1e-6) { dx = mx/ml; dy = my/ml; }
      }
      float L = ssz2 * stepf;
      px = clamp(px + L*dx, 0.0, float(u_H - 1));
      py = clamp(py + L*dy, 0.0, float(u_W - 1));
      if (snapE) { vec2 sp3 = edgeSnap(px, py, liner ? 0.85 : 0.65); px = sp3.x; py = sp3.y; }
      if (side != 0.0) {
        px = clamp(px + sidem * soff * ssz2 * (-dy), 0.0, float(u_H - 1));
        py = clamp(py + sidem * soff * ssz2 * ( dx), 0.0, float(u_W - 1));
      }
      dxp = dx; dyp = dy;
    }
    if (phase == 0) finalLen = emitted;
  }
}")

(def ^:private gen-uniform-names
  ["u_nlev" "u_warp" "u_H" "u_W" "u_stroke" "u_variation" "u_contrast" "u_detail" "u_curv" "u_broad"
   "u_ssz" "u_sp" "u_th" "u_nx" "u_ny" "u_off" "u_lvl"
   "u_segs" "u_stepf" "u_bendf" "u_sharp" "u_sideo" "u_selong"
   "u_detailTex" "u_subjTex" "u_dmax" "u_detailDim" "u_detailSrc"
   "u_noiseTex" "u_noiseDim" "u_noiseSrc" "u_blurTex" "u_blurDTex" "u_blurHTex" "u_rawTex" "u_permTex"])

(defn sources
  "Return {:vs-src :gs-src} — pure, no GL context (for headless inspection/tests)."
  []
  {:vs-src vs-src :gs-src gs-src})

(defn build-gen-program
  "Compile + link the VS+GS transform-feedback generation program, capturing
   o_a,o_b interleaved (8 floats/splat). Returns {:program :locs} or nil."
  []
  (when-let [prog (gl/make-tf-program vs-src gs-src ["o_a" "o_b" "o_c"])]
    {:program prog
     :locs (into {} (map (fn [n] [(keyword n) (gl/gl-get-uniform-location prog n)]))
                 gen-uniform-names)}))

;; --- field texture upload (once per image) -----------------------------------
(defn- new-tex []
  (let [t (gl/gen-one gl/gl-gen-textures)]
    (gl/gl-bind-texture gl/GL-TEXTURE-2D t)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MIN-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-MAG-FILTER gl/GL-NEAREST)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-S gl/GL-CLAMP-TO-EDGE)
    (gl/gl-tex-parameter-i gl/GL-TEXTURE-2D gl/GL-TEXTURE-WRAP-T gl/GL-CLAMP-TO-EDGE)
    t))

(defn- upload-rgba! [tex tw th ^"[D" ptr]
  (gl/gl-bind-texture gl/GL-TEXTURE-2D tex)
  (gl/gl-tex-image-2d gl/GL-TEXTURE-2D 0 gl/GL-RGBA32F (int tw) (int th) 0
                      gl/GL-RGBA gl/GL-FLOAT ptr)
  (ffi/free ptr))

;; Build an RGBA32F buffer (W wide, H tall; texel (row=xi,col=yi)) directly from
;; source double-arrays with a getter that returns [r g b a] for flat index xi*W+yi.
(defn- rgba-ptr [H W getter]
  (let [n (* (long H) (long W))
        ptr (ffi/alloc (* n 4 (ffi/sizeof :float)))]
    (dotimes [idx n]
      (let [o (* idx 16)                     ; 4 floats × 4 bytes
            [r g b a] (getter idx)]
        (ffi/write ptr :float o        (double r))
        (ffi/write ptr :float (+ o 4)  (double g))
        (ffi/write ptr :float (+ o 8)  (double b))
        (ffi/write ptr :float (+ o 12) (double a))))
    ptr))

(defn upload-fields!
  "Upload the per-image field textures. `img` must carry :structure/:detail/:blur/
   :noise-fields (the same caches the CPU path uses). Returns a map of texture ids +
   dims for the generation uniforms. `perm-tex` is uploaded once and reused."
  [img perm-tex]
  (let [H (long (:height img)) W (long (:width img))
        ^doubles raw (:pixels img)
        ^doubles blur (or (:blur img) (:pixels img))
        ^doubles blurd (or (:blur-drift img) blur)
        ^doubles blurh (or (:blur-heavy img) blur)
        dmap (:detail img)
        Hd (long (:h dmap)) Wd (long (:w dmap))
        ^doubles dd (:detail dmap)
        ^doubles ds (or (:sharp dmap) dd)
        ^doubles de (or (:edge dmap) (double-array (alength dd)))
        ^doubles dm2 (or (:mid dmap) dd)
        ^doubles dsu (or (:subject dmap) dd)
        nf (:noise-fields img)
        Hn (long (:h nf)) Wn (long (:w nf))
        ^doubles c2 (:c2 nf) ^doubles s2 (:s2 nf) ^doubles co (:coherence nf)
        detail-t (new-tex) subj-t (new-tex) noise-t (new-tex) blur-t (new-tex) blurd-t (new-tex) blurh-t (new-tex) raw-t (new-tex)]
    ;; .r = aggregate map, .g = sharp fine-band, .b = raw edge strength, .a = MID band
    (upload-rgba! detail-t Wd Hd (rgba-ptr Hd Wd (fn [i] [(aget dd i) (aget ds i) (aget de i) (aget dm2 i)])))
    ;; ABSOLUTE subjectness (.r) — the broad tier's bokeh gate (same grid as detail)
    (upload-rgba! subj-t   Wd Hd (rgba-ptr Hd Wd (fn [i] [(aget dsu i) 0.0 0.0 1.0])))
    ;; orientation as double-angle components (cos2θ, sin2θ) + coherence — the GS
    ;; bilinearly blends the components (fieldsAt), never the raw angle.
    (upload-rgba! noise-t  Wn Hn (rgba-ptr Hn Wn (fn [i] [(aget c2 i) (aget s2 i) (aget co i) 0.0])))
    (upload-rgba! blur-t   W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget blur b) (aget blur (+ b 1)) (aget blur (+ b 2)) 1.0]))))
    (upload-rgba! blurd-t  W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget blurd b) (aget blurd (+ b 1)) (aget blurd (+ b 2)) 1.0]))))
    (upload-rgba! blurh-t  W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget blurh b) (aget blurh (+ b 1)) (aget blurh (+ b 2)) 1.0]))))
    (upload-rgba! raw-t    W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget raw b) (aget raw (+ b 1)) (aget raw (+ b 2)) 1.0]))))
    {:detail detail-t :subject subj-t :noise noise-t :blur blur-t :blur-drift blurd-t :blur-heavy blurh-t :raw raw-t :perm perm-tex
     :dmap dmap                              ; the CPU detail map, for layer-params' budget
     :dmax (double (:dmax dmap))
     :detail-dim [(double Hd) (double Wd)] :detail-src [(double H) (double W)]
     :noise-dim  [(double Hn) (double Wn)] :noise-src  [(double (or (:src-h nf) H)) (double (or (:src-w nf) W))]
     :H H :W W}))

(defn upload-perm!
  "Upload Ken Perlin's 512 permutation entries as a 512×1 RGBA32F texture (.r = value)."
  []
  (let [^ints p noise/perm512
        t (new-tex)]
    (upload-rgba! t 512 1 (rgba-ptr 1 512 (fn [i] [(double (aget p i)) 0.0 0.0 1.0])))
    t))

;; --- run generation ----------------------------------------------------------
(defn- set-1fv! [loc xs]
  (let [ptr (gl/write-floats xs)] (gl/gl-uniform-1fv loc (count xs) ptr) (ffi/free ptr)))
(defn- set-1iv! [loc xs]
  (let [ptr (gl/write-ints xs)] (gl/gl-uniform-1iv loc (count xs) ptr) (ffi/free ptr)))

(defn sig-range
  "Approximate the field's stdev range from the level ssz values — the GPU path has no
   CPU-side splats to reduce over, and this only feeds the size→hardness smoothstep
   easing. The lower bound uses the 0.75 size-jitter shrink clamp; the upper includes
   the bokeh-adaptive Broad growth (flat-region daubs reach ×broad the nominal)."
  [levels variation broad]
  (let [sszs (map :ssz levels)
        v (double variation)]
    [(* (reduce min sszs) 0.75)
     (* (reduce max sszs) (max 1.0 (double broad)) (+ 1.0 (* 0.5 v)))]))

(defn read-splats
  "Read `n` generated splats back from the transform-feedback buffer as a vector of
   {:mean [x y] :cov [c00 c01 c01 c11] :color [r g b] :alpha a} — the same shape as the CPU
   splat-field, for numerical verification against the golden reference. `tf-buf` must
   be bound to GL_TRANSFORM_FEEDBACK_BUFFER."
  [tf-buf n]
  (let [nf  (* n 12)
        ptr (ffi/alloc (* nf (ffi/sizeof :float)))]
    (gl/gl-bind-buffer gl/GL-TRANSFORM-FEEDBACK-BUFFER tf-buf)
    (gl/gl-get-buffer-sub-data gl/GL-TRANSFORM-FEEDBACK-BUFFER 0
                               (* nf (ffi/sizeof :float)) ptr)
    (let [fl (gl/read-floats ptr nf)]
      (ffi/free ptr)
      (mapv (fn [i]
              (let [b (* i 12)
                    mx (nth fl b) my (nth fl (+ b 1))
                    c00 (nth fl (+ b 2)) c01 (nth fl (+ b 3)) c11 (nth fl (+ b 4))]
                {:mean [mx my] :cov [c00 c01 c01 c11]
                 :color [(nth fl (+ b 5)) (nth fl (+ b 6)) (nth fl (+ b 7))]
                 :alpha (nth fl (+ b 8))}))
            (range n)))))

(defn generate!
  "Run the generation pass into `tf-buf` (a GL buffer bound to binding point 0). Sets
   all uniforms from `controls` + `fields`, draws `total` candidate points through the
   VS+GS with rasterizer discard, and returns {:count n :sig-min :sig-max}. Needs a
   current context, a bound (complete) framebuffer, the gen program, a bound VAO, and
   a query object `q`."
  [gen fields controls tf-buf q vao {:keys [height width]}]
  (let [{:keys [program locs]} gen
        {:keys [count size stroke detail variation curvature contrast
                size-broad size-mid size-fine edge-band]} controls
        H (long height) W (long width)
        params (seed/layer-params (:dmap fields) detail size variation curvature stroke
                                  [(double (or size-broad 1.0)) (double (or size-mid 1.0))
                                   (double (or size-fine 1.0)) (double (or edge-band 1.0))]
                                  count H W)
        {:keys [nlev warp levels total]} params
        ;; finest-first level arrays, padded to max-levels
        pad (fn [xs d] (vec (take max-levels (concat xs (repeat d)))))
        ssz (pad (map :ssz levels) 0.0)
        sp  (pad (map :sp levels) 1.0)
        th  (pad (map :th levels) -1.0)
        nx  (pad (map :nx levels) 0)
        ny  (pad (map :ny levels) 1)
        off (pad (map :offset levels) 0)
        lvl (pad (map :lvl levels) 0)
        sgs (pad (map :segs levels) 1)
        stf (pad (map :stepf levels) 1.0)
        bnf (pad (map :bendf levels) 1.0)
        shp (pad (map (fn [l] (case (:map-kind l) :sharp 2 :mid 1 :edge 3 0)) levels) 0)
        ;; side push / forced elongation per slot (mirror seed/layer-params). Padding
        ;; defaults match an ordinary liner: 0.55 sigma push, no forced elongation.
        sdo (pad (map (fn [l] (double (or (:sideo l) 0.55))) levels) 0.55)
        sel (pad (map (fn [l] (double (or (:selong l) 0.0))) levels) 0.0)
        [sig-min sig-max] (sig-range levels variation (or size-broad 1.0))]
    (gl/gl-use-program program)
    ;; per-level + controls
    (gl/gl-uniform-1i (:u_nlev locs) (int nlev))
    (gl/gl-uniform-1f (:u_warp locs) (double warp))
    (gl/gl-uniform-1i (:u_H locs) (int H))
    (gl/gl-uniform-1i (:u_W locs) (int W))
    (gl/gl-uniform-1f (:u_stroke locs) (double stroke))
    (gl/gl-uniform-1f (:u_variation locs) (double variation))
    (gl/gl-uniform-1f (:u_contrast locs) (double contrast))
    (gl/gl-uniform-1f (:u_detail locs) (double detail))
    (gl/gl-uniform-1f (:u_curv locs) (double curvature))
    (gl/gl-uniform-1f (:u_broad locs) (double (or size-broad 1.0)))
    (set-1fv! (:u_ssz locs) ssz)
    (set-1fv! (:u_sp locs) sp)
    (set-1fv! (:u_th locs) th)
    (set-1iv! (:u_nx locs) nx)
    (set-1iv! (:u_ny locs) ny)
    (set-1iv! (:u_off locs) off)
    (set-1iv! (:u_lvl locs) lvl)
    (set-1iv! (:u_segs locs) sgs)
    (set-1fv! (:u_stepf locs) stf)
    (set-1fv! (:u_bendf locs) bnf)
    (set-1iv! (:u_sharp locs) shp)
    (set-1fv! (:u_sideo locs) sdo)
    (set-1fv! (:u_selong locs) sel)
    ;; field textures on units 1..5 (unit 0 is the render splat buffer)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 1)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:detail fields)) (gl/gl-uniform-1i (:u_detailTex locs) 1)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 2)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:noise fields))  (gl/gl-uniform-1i (:u_noiseTex locs) 2)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 3)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:blur fields))   (gl/gl-uniform-1i (:u_blurTex locs) 3)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 4)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:raw fields))    (gl/gl-uniform-1i (:u_rawTex locs) 4)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 5)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:perm fields))   (gl/gl-uniform-1i (:u_permTex locs) 5)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 6)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:blur-heavy fields)) (gl/gl-uniform-1i (:u_blurHTex locs) 6)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 7)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:subject fields))    (gl/gl-uniform-1i (:u_subjTex locs) 7)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 8)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:blur-drift fields)) (gl/gl-uniform-1i (:u_blurDTex locs) 8)
    (gl/gl-uniform-1f (:u_dmax locs) (double (:dmax fields)))
    (gl/gl-uniform-2f (:u_detailDim locs) (double (nth (:detail-dim fields) 0)) (double (nth (:detail-dim fields) 1)))
    (gl/gl-uniform-2f (:u_detailSrc locs) (double (nth (:detail-src fields) 0)) (double (nth (:detail-src fields) 1)))
    (gl/gl-uniform-2f (:u_noiseDim locs) (double (nth (:noise-dim fields) 0)) (double (nth (:noise-dim fields) 1)))
    (gl/gl-uniform-2f (:u_noiseSrc locs) (double (nth (:noise-src fields) 0)) (double (nth (:noise-src fields) 1)))
    ;; capture survivors via transform feedback
    (gl/gl-bind-vertex-array vao)
    (gl/gl-bind-buffer-base gl/GL-TRANSFORM-FEEDBACK-BUFFER 0 tf-buf)
    (gl/gl-enable gl/GL-RASTERIZER-DISCARD)
    (gl/gl-begin-query gl/GL-TRANSFORM-FEEDBACK-PRIMITIVES-WRITTEN q)
    (gl/gl-begin-transform-feedback gl/GL-POINTS)
    (gl/gl-draw-arrays gl/GL-POINTS 0 (int total))
    (gl/gl-end-transform-feedback)
    (gl/gl-end-query gl/GL-TRANSFORM-FEEDBACK-PRIMITIVES-WRITTEN)
    (gl/gl-disable gl/GL-RASTERIZER-DISCARD)
    (let [n (gl/get-query-object-uiv q gl/GL-QUERY-RESULT)]
      {:count n :sig-min sig-min :sig-max sig-max :total total})))
