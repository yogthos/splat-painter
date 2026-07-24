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

(def ^:private max-levels 7)

;; --- generation program: vertex + geometry (transform feedback) --------------
(def ^:private vs-src
  "#version 330 core
flat out int v_id;
void main(){ v_id = gl_VertexID; }")

(def ^:private gs-src
  "#version 330 core
layout(points) in;
layout(points, max_vertices = 8) out;
flat in int v_id[];

// captured by transform feedback: 12 floats = 3 RGBA texels per splat, matching the
// render shaders' t0=(mean_x,mean_y,c00,c01) t1=(c11,r,g,b) t2=(alpha,0,0,0).
out vec4 o_a;
out vec4 o_b;
out vec4 o_c;

const int   ML      = 7;
const float MIN_COH = 0.28;
const int   SEGS    = 8;      // max segments per brush stroke (seed/seg-count liner lines)

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
uniform sampler2D u_blurTex;  // W×H light blur (fine strokes' smooth base)
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

// nearest-map a full-image (x,y) into a reduced field grid (dim=(rows,cols),
// src=(src_h,src_w)), matching CPU round()=floor(v+0.5), then texelFetch(col,row).
ivec2 fieldTexel(float x, float y, vec2 dim, vec2 src){
  int xi = int(floor(x * dim.x / src.x + 0.5));
  int yi = int(floor(y * dim.y / src.y + 0.5));
  xi = clamp(xi, 0, int(dim.x) - 1);
  yi = clamp(yi, 0, int(dim.y) - 1);
  return ivec2(yi, xi);
}
float detailAt(float x, float y){
  vec4 t = texelFetch(u_detailTex, fieldTexel(x, y, u_detailDim, u_detailSrc), 0);
  return u_dmax > 0.0 ? min(1.0, t.r / u_dmax) : 0.0;
}
// the SHARP fine-band map (texel .g) — what the finest levels threshold against
float sharpAt(float x, float y){
  vec4 t = texelFetch(u_detailTex, fieldTexel(x, y, u_detailDim, u_detailSrc), 0);
  return u_dmax > 0.0 ? min(1.0, t.g / u_dmax) : 0.0;
}
// scale-matched map select (mirror of seed/level-map-kind): 0 = aggregate (.r),
// 1 = MID band (.a — face-feature frequencies), 2 = sharp fine-band (.g)
float mapAt(int sel, float x, float y){
  vec4 t = texelFetch(u_detailTex, fieldTexel(x, y, u_detailDim, u_detailSrc), 0);
  float v = (sel == 2) ? t.g : (sel == 1) ? max(t.a, t.g) : t.r;  // mid = union with sharp
  return u_dmax > 0.0 ? min(1.0, v / u_dmax) : 0.0;
}
// raw edge strength (texel .b) — the band where broad fill strokes must not tread
float edgeAt(float x, float y){
  return texelFetch(u_detailTex, fieldTexel(x, y, u_detailDim, u_detailSrc), 0).b;
}
// ABSOLUTE subjectness (mirror wavelet/subject-abs-at): raw globally-scaled
// fine-band energy + edge strength — 0 in bokeh, high on real structure. Drives
// the broad tier's bokeh adaptation; the locally-normalized maps (which light
// bokeh up to full 'detail') keep driving fine-stroke placement.
float subjAbsAt(float x, float y){
  return min(1.0, texelFetch(u_subjTex, fieldTexel(x, y, u_detailDim, u_detailSrc), 0).r);
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

vec3 sampleRGB(sampler2D tex, float x, float y){   // W×H, sample-arr nearest (int trunc)
  int xi = clamp(int(x), 0, u_H - 1);
  int yi = clamp(int(y), 0, u_W - 1);
  return texelFetch(tex, ivec2(yi, xi), 0).rgb;
}

// splat-record (mirror seed/splat-record) + emit one captured record. `alpha` is the
// stroke taper (1.0 for base fills, fading toward a fine stroke's tail).
// (hx,hy) = the chain HEAD's position — the colour-sample point for EVERY segment:
// one stroke = one brush-load of paint (per-segment sampling alternated the two
// sides' colours along contours into a bead necklace).
// `cohmul` rounds MELTED bokeh strokes off (coherence → 0 kills the elongation
// and pulls the colour toward the smooth blur): an elongated needle on a soft
// gradient always reads as a directional streak, however faithful its colour.
void emitSplat(float px, float py, float hx, float hy, float csz, float D, float sn, float tn, float alpha, float hb, float traw, float tcap, float cohmul){
  vec2  tc    = fieldsAt(px, py);
  float theta = tc.x, coh0 = tc.y * cohmul;
  vec3  blur = (hb > 0.5) ? sampleRGB(u_blurHTex, hx, hy) : sampleRGB(u_blurTex, hx, hy);
  vec3  raw  = sampleRGB(u_rawTex,  hx, hy);
  float coh = MIN_COH + (1.0 - MIN_COH) * coh0;
  float e   = 1.0 + min(u_stroke, 1.5) * coh * (0.25 + 0.75 * D);
  float se  = sqrt(e);
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
  float mloc  = 1.0 + (u_broad - 1.0) * (1.0 - sabs);
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
  float bgate = 1.0 - clamp((u_broad - 1.0) / 1.5, 0.0, 1.0) * (1.0 - min(1.0, sabs / 0.35));
  float gain = (lvl >= 2) ? (0.25 + 0.75 * sgate) * bgate : 1.0;
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
  if (lvl > 0 && lvl <= 2 && k > 0) {
    float fdv = mapAt(u_sharp[k-1], cx, cy) * (0.25 + 0.75 * sgate) * bgate;
    if (fdv >= u_th[k-1] * (0.75 + 0.5 * hash01(i*47 + lvl, j, 23))) return;
  }

  // hashed positions need no jitter — they ARE the noise
  float x = cx, y = cy;
  float D = min(1.0, u_detail * dv * 2.2);
  float aw = u_warp * (1.0 - D) * ssz;
  float x2 = (aw < 0.2) ? x : x + aw * noise2(0.06*x, 0.06*y);
  float y2 = (aw < 0.2) ? y : y + aw * noise2(41.3 + 0.06*x, 17.9 + 0.06*y);
  x2 = clamp(x2, 0.0, float(u_H - 1));
  y2 = clamp(y2, 0.0, float(u_W - 1));

  // per-seed jitter is hashed (independent per stroke), mirroring seed/layered-means.
  // Size jitter applies at SEED level to the whole chain (size AND step) so chains
  // stay self-overlapping at any Variation; broad levels keep 40% of both jitters.
  float sn0 = hash01(i*31 + lvl, j, 11) - 0.5;
  float szf = max(0.75, 1.0 + u_variation * sn0 * ((lvl <= 1) ? 0.4 : 1.0));
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
  // MELT (mirror seed/layered-means): how much a flat-region broad stroke should
  // sink into the wash — grows only past Broad 1.0 and only where subjectness is
  // low, so Broad at max makes bokeh strokes invisible while detailed regions
  // keep their brushwork. Mutes the tone jitter and drives the chain re-mix.
  float melt = (lvl <= 1) ? clamp((u_broad - 1.0) / 1.5, 0.0, 1.0) * (1.0 - sabs) : 0.0;
  float tnoise = (hash01(i*37 + lvl, j, 13) - 0.5)
               * ((lvl <= 1) ? 0.25 * (1.0 - melt) : (lvl >= 4) ? 0.15 : 1.0);

  // broad strokes (base + level 1) colour from the HEAVY blur — smoothed at their scale
  float hb = (lvl <= 1) ? 1.0 : 0.0;
  // colour-rawness floor rises with fineness (seed/raw-floor): small strokes paint
  // faithful colour — a half-blur blend at feature scale softens the feature away
  float traw = (lvl <= 1) ? 0.0 : (lvl <= 3) ? 0.45 : (lvl <= 5) ? 0.7 : 0.85;
  // fine colour rawness follows the local detail density (mirror seed): a crisp
  // raw-colour mark never pops at full contrast on soft ground
  if (lvl >= 4) traw *= 0.6 + 0.4 * sgate;
  // colour-specificity ceiling per level (mirror seed/spec-cap): broad layers
  // paint AVERAGED colour, mids halfway, fine layers fully specific
  float tcap = (lvl <= 1) ? 0.35 : (lvl <= 3) ? 0.7 : 1.0;
  if (lvl == 0) {                                 // base fill: one full-alpha splat
    emitSplat(x2, y2, x2, y2, ssz2, D, snoise, tnoise, 1.0, hb, traw, tcap, 1.0 - melt);
    return;
  }

  // fine level: TRACE A BRUSH STROKE (mirror seed/stroke-segments) — SEGS segments
  // stepped along the orientation field (the edge tangent), sign-continuous, bent by
  // smooth Perlin noise scaled by Curvature, size+alpha tapering toward the tail.
  float dirsign = hash01(i*41 + lvl, j, 17) < 0.5 ? 1.0 : -1.0;
  int   segs  = u_segs[k];                       // scale-relative stroke behaviour:
  float stepf = u_stepf[k];                      // broad levels stroke long and curl,
  float bendf = u_bendf[k];                      // fine levels make short precise marks
  bool snapE = (lvl >= 2);                       // fine strokes glue to the edge ridge
  // colour samples the PRE-snap position (one side of the edge); geometry snaps.
  // On-ridge colour is the sides' mix and paints silhouettes as drawn outlines.
  float cpx = x2, cpy = y2;
  if (snapE) { vec2 sp2 = edgeSnap(x2, y2, 0.65); x2 = sp2.x; y2 = sp2.y; }
  // which side of the ridge did this seed come from? (mirror seed/stroke-segments)
  float side = 0.0;
  if (snapE && lvl >= 4) {
    vec2 tcs = fieldsAt(x2, y2);
    float nsx = -sin(tcs.x), nsy = cos(tcs.x);
    float dd = (cpx - x2)*nsx + (cpy - y2)*nsy;
    side = (dd > 1e-9) ? 1.0 : (dd < -1e-9) ? -1.0 : 0.0;
  }
  vec2 so0 = sideOffset(x2, y2, side, 0.55 * ssz2);
  x2 = so0.x; y2 = so0.y;
  // side sign in the stroke's MOTION frame (mirror seed): stays consistent
  // through field sign flips that would wobble a per-step theta resample
  float sidem = side * dirsign;
  float px = x2, py = y2, dxp = 0.0, dyp = 0.0;
  // progressive refinement: finer layers GLAZE (translucent touches over the
  // accumulated underpainting) instead of overwriting it; the overlapping fine
  // tier glazes lightest so stacked strokes MIX (mirror seed/level-alpha)
  float lal = (lvl <= 1) ? 1.0 : (lvl <= 3) ? 0.85 : (lvl <= 5) ? 0.65 : 0.55;
  float fade = 1.0;
  vec3 headBlur = sampleRGB(u_blurTex, x2, y2);
  for (int q = 0; q < SEGS; q++) {
    if (q >= segs || fade < 0.15) break;
    if (q > 0) {
      // TWO-TIER dry-out (mirror seed): gradual drift DRIES the brush (x0.4);
      // a LARGE mismatch (>0.45) means the stroke EXITED its colour region — a
      // chain escaping a curved silhouette would paint its dark brush-load into
      // the background — so the painter LIFTS the brush and emits nothing.
      // the broad tier lifts IMMEDIATELY on any real colour change (0.18): its
      // opaque underpainting chains at the largest sizes must never carry paint
      // across a boundary (one escaped segment = a huge wrong-colour ghost cloud)
      vec3 cb = sampleRGB(u_blurTex, px, py);
      vec3 dcl = abs(cb - headBlur);
      float dmx = max(dcl.r, max(dcl.g, dcl.b));
      if (dmx > ((lvl <= 1) ? 0.18 : 0.45)) fade = 0.0;
      else if (dmx > ((lvl >= 4) ? 0.3 : 0.22)) fade *= 0.4;
    }
    vec2 tc = fieldsAt(px, py);
    // follow the line only while there IS a line (mirror seed/stroke-segments):
    // when coherence collapses (busy texture, letter junctions) the liner stroke
    // runs dry fast — long chains wandering through dense detail smear it.
    // ...but a strong edge under the brush keeps the line alive: real ink lines
    // push THROUGH junctions, where coherence dips while edge energy stays high
    if (q > 0 && lvl >= 4 && tc.y < 0.35 && edgeAt(px, py) < 0.5) fade *= 0.5;
    // LINE-HOLD (mirror seed/stroke-segments): a liner stroke exists to trace the
    // fine structure it was seeded on. When the sharp fine-band map under the
    // brush falls below the level's own placement threshold, the stroke has
    // WALKED OFF its line — a chain escaping a silhouette tangentially would drag
    // its bodied paint into the featureless background (ghost tendrils) — lift.
    if (q > 0 && lvl >= 4) {
      float mv = mapAt(2, px, py) * (0.25 + 0.75 * sgate);
      if (mv < 0.35 * th) fade = 0.0;
      else if (mv < 0.7 * th) fade *= 0.5;
    }
    if (fade < 0.15) break;                        // brush lifted — emit nothing
    float tt = float(q) / float(segs - 1);
    // IMPASTO body (mirror seed/stroke-segments): fine liner strokes on a strong
    // edge paint nearly opaque — the contour is defined by thin bodied lines whose
    // soft shoulders blend; off-edge texture strokes keep the light mixing glaze.
    float body = ((lvl >= 4) ? clamp((edgeAt(px, py) - 0.25) / 0.45, 0.0, 1.0) : 0.0)
               * (0.4 + 0.6 * sgate);
    float lal2 = lal + (0.9 - lal) * body;
    // BOTH-ENDS taper (mirror seed/stroke-segments): quick lift-on at the head on top
    // of the existing tail dry-out, so the mark tapers at BOTH ends like a real
    // stroke. The LINER tier keeps only a hint (overlapping chains hand off — strong
    // per-chain taper lumps the handoffs into a string of tadpoles).
    float hw = (lvl >= 4) ? 0.8  + 0.2  * smoothstep(0.0, 0.18, tt)
                          : 0.55 + 0.45 * smoothstep(0.0, 0.18, tt);
    float ha = (lvl >= 4) ? 0.75 + 0.25 * smoothstep(0.0, 0.15, tt)
                          : 0.5  + 0.5  * smoothstep(0.0, 0.15, tt);
    float sz = ssz2 * (1.0 - 0.45 * tt * sqrt(tt)) * hw;  // width tapers at both ends
    float al = lal2 * fade * (1.0 - 0.65 * tt * tt) * ha; // alpha: lift-on × glaze × dry-out
    // the brush-load RE-MIXES with the canvas (mirror seed): the colour-sample
    // point slides up to 35% toward the current position along the stroke.
    // MELTED broad chains re-mix much harder — one brush-load carried across a
    // smooth gradient reads as a feathery streak on the wash.
    float wsl = (lvl >= 4) ? 0.35 * tt : ((melt > 0.0) ? 0.85 * melt * tt : 0.0);
    emitSplat(px, py, cpx + wsl*(px - cpx), cpy + wsl*(py - cpy), sz, D, snoise, tnoise, al, hb, traw, tcap, 1.0 - melt);
    // bend gated by coherence: straight strongly-oriented edges trace straight
    float bend = u_curv * 0.9 * bendf * (1.0 - 0.7*tc.y) * (noise2(0.05*px, 0.05*py) - 0.5);
    float cb = cos(bend), sb = sin(bend);
    float dx0 = cos(tc.x), dy0 = sin(tc.x);
    float sgn = (q == 0) ? dirsign : ((dx0*dxp + dy0*dyp) < 0.0 ? -1.0 : 1.0);
    float dx1 = sgn*dx0, dy1 = sgn*dy0;
    float dx = cb*dx1 - sb*dy1, dy = sb*dx1 + cb*dy1;
    // DIRECTION MOMENTUM (mirror seed): liner strokes carry 65% of the previous
    // step's direction — a field-noise-driven turn every step waves the line
    if (lvl >= 4 && q > 0) {
      float mx = 0.35*dx + 0.65*dxp, my = 0.35*dy + 0.65*dyp;
      float ml = sqrt(mx*mx + my*my);
      if (ml > 1e-6) { dx = mx/ml; dy = my/ml; }
    }
    // the Stroke slider is stroke LENGTH: it scales the chain step (2.5 ≈ 1.0)
    float L = ssz2 * stepf * (0.4 + 0.24 * u_stroke);
    px = clamp(px + L*dx, 0.0, float(u_H - 1));
    py = clamp(py + L*dy, 0.0, float(u_W - 1));
    if (snapE) { vec2 sp3 = edgeSnap(px, py, (lvl >= 4) ? 0.35 : 0.65); px = sp3.x; py = sp3.y; }
    // side offset along the stroke's OWN motion perpendicular (mirror seed) —
    // the path is a stable frame; a per-step theta resample wobbled the line
    if (side != 0.0) {
      px = clamp(px + sidem * 0.55 * ssz2 * (-dy), 0.0, float(u_H - 1));
      py = clamp(py + sidem * 0.55 * ssz2 * ( dx), 0.0, float(u_W - 1));
    }
    dxp = dx; dyp = dy;
  }
}")

(def ^:private gen-uniform-names
  ["u_nlev" "u_warp" "u_H" "u_W" "u_stroke" "u_variation" "u_contrast" "u_detail" "u_curv" "u_broad"
   "u_ssz" "u_sp" "u_th" "u_nx" "u_ny" "u_off" "u_lvl"
   "u_segs" "u_stepf" "u_bendf" "u_sharp"
   "u_detailTex" "u_subjTex" "u_dmax" "u_detailDim" "u_detailSrc"
   "u_noiseTex" "u_noiseDim" "u_noiseSrc" "u_blurTex" "u_blurHTex" "u_rawTex" "u_permTex"])

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
        detail-t (new-tex) subj-t (new-tex) noise-t (new-tex) blur-t (new-tex) blurh-t (new-tex) raw-t (new-tex)]
    ;; .r = aggregate map, .g = sharp fine-band, .b = raw edge strength, .a = MID band
    (upload-rgba! detail-t Wd Hd (rgba-ptr Hd Wd (fn [i] [(aget dd i) (aget ds i) (aget de i) (aget dm2 i)])))
    ;; ABSOLUTE subjectness (.r) — the broad tier's bokeh gate (same grid as detail)
    (upload-rgba! subj-t   Wd Hd (rgba-ptr Hd Wd (fn [i] [(aget dsu i) 0.0 0.0 1.0])))
    ;; orientation as double-angle components (cos2θ, sin2θ) + coherence — the GS
    ;; bilinearly blends the components (fieldsAt), never the raw angle.
    (upload-rgba! noise-t  Wn Hn (rgba-ptr Hn Wn (fn [i] [(aget c2 i) (aget s2 i) (aget co i) 0.0])))
    (upload-rgba! blur-t   W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget blur b) (aget blur (+ b 1)) (aget blur (+ b 2)) 1.0]))))
    (upload-rgba! blurh-t  W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget blurh b) (aget blurh (+ b 1)) (aget blurh (+ b 2)) 1.0]))))
    (upload-rgba! raw-t    W  H  (rgba-ptr H W (fn [i] (let [b (* i 3)] [(aget raw b) (aget raw (+ b 1)) (aget raw (+ b 2)) 1.0]))))
    {:detail detail-t :subject subj-t :noise noise-t :blur blur-t :blur-heavy blurh-t :raw raw-t :perm perm-tex
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
                size-broad size-mid size-fine]} controls
        H (long height) W (long width)
        params (seed/layer-params (:dmap fields) detail size variation curvature stroke
                                  [(double (or size-broad 1.0)) (double (or size-mid 1.0))
                                   (double (or size-fine 1.0))]
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
        shp (pad (map (fn [l] (case (:map-kind l) :sharp 2 :mid 1 0)) levels) 0)
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
    ;; field textures on units 1..5 (unit 0 is the render splat buffer)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 1)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:detail fields)) (gl/gl-uniform-1i (:u_detailTex locs) 1)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 2)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:noise fields))  (gl/gl-uniform-1i (:u_noiseTex locs) 2)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 3)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:blur fields))   (gl/gl-uniform-1i (:u_blurTex locs) 3)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 4)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:raw fields))    (gl/gl-uniform-1i (:u_rawTex locs) 4)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 5)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:perm fields))   (gl/gl-uniform-1i (:u_permTex locs) 5)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 6)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:blur-heavy fields)) (gl/gl-uniform-1i (:u_blurHTex locs) 6)
    (gl/gl-active-texture (+ gl/GL-TEXTURE0 7)) (gl/gl-bind-texture gl/GL-TEXTURE-2D (:subject fields))    (gl/gl-uniform-1i (:u_subjTex locs) 7)
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
