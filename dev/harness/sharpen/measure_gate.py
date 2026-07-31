#!/usr/bin/env python3
"""Step 1: measure Sobel gate thresholds for the sharpen pass.

Sobel field on the RENDER computed exactly as the shader will:
luma (Rec.709), [1 2 1]-weighted Sobel, /8, on 0..1 values, edge-clamped taps.
Regions:
 - bokeh box (right-hand out-of-focus background)
 - strong-edge band: source central-difference gradient > its 85th percentile
   (the band the sharpness metric uses), shifted by the known 1px registration
   (render row i == source row i+1)
 - flat stroke-texture region: source sobel below its 25th percentile
   (no source edges -> any render gradient there is canvas grain / bristle /
   ragged-contour texture the gate must reject)
"""
import numpy as np
from PIL import Image

REN = '/private/tmp/claude-501/-Users-yogthos-src-splat-painter/61ecf884-c16f-4cf0-a9e5-35710884657a/scratchpad/baseline-sharpen.png'
SRC = '/Users/yogthos/src/splat-painter/img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg'

def luma(path, size=None):
    im = Image.open(path).convert('RGB')
    if size: im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)/255.0
    return 0.2126*a[:,:,0] + 0.7152*a[:,:,1] + 0.0722*a[:,:,2]

def sobel(g):
    p = np.pad(g, 1, mode='edge')
    gx = (p[:-2,2:] + 2*p[1:-1,2:] + p[2:,2:]) - (p[:-2,:-2] + 2*p[1:-1,:-2] + p[2:,:-2])
    gy = (p[2:,:-2] + 2*p[2:,1:-1] + p[2:,2:]) - (p[:-2,:-2] + 2*p[:-2,1:-1] + p[:-2,2:])
    return np.hypot(gx, gy)/8.0

def cdiff(g):
    return np.hypot(*np.gradient(g))

r = luma(REN)
s = luma(SRC, (1024,1024))
gr = sobel(r)
band_src = cdiff(s) > np.percentile(cdiff(s), 85)   # (1024,1024), source coords
# registration: render[i,j] ~ source[i+1,j]
band_r = band_src[1:, :]                               # band for render rows 0..1022
soft_r = sobel(s)[1:, :] < np.percentile(sobel(s), 25) # no source edge at all
gr_r = gr[:-1, :]                                      # render field, aligned

BX = (790, 1010); BY = (140, 300)   # bokeh box: right background, x0,x1 / y0,y1
BY2 = (460, 620)                    # second bokeh strip, lower right
bok1 = gr[BY[0]:BY[1], BX[0]:BX[1]].ravel()
bok2 = gr[BY2[0]:BY2[1], BX[0]:BX[1]].ravel()
bok = np.concatenate([bok1, bok2])
band_vals = gr_r[band_r]
soft_vals = gr_r[soft_r]

def stats(name, v):
    q = np.percentile(v, [10, 25, 50, 90, 99])
    print(f'{name:34s} n={v.size:8d} median={np.median(v):.4f}  p10={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} p90={q[3]:.4f} p99={q[4]:.4f}')
    return q

print('== render Sobel magnitude (shader-exact) by region ==')
qb  = stats(f'bokeh box x{BX} y{BY}+{BY2}', bok)
qs  = stats('strong-edge band (src p85)', band_vals)
qf  = stats('flat stroke-texture (src<p25)', soft_vals)
qa  = stats('whole render', gr.ravel())

print()
lo_bok, lo_flat = qb[3], qf[3]     # p90 of each reject population
hi_band = qs[0]                    # p10 of the accept population
print(f'for >=90% rejection:  lo >= p90(bokeh) = {lo_bok:.4f}   lo >= p90(flat-texture) = {lo_flat:.4f}')
print(f'for >=90% acceptance: hi <= p10(band)  = {hi_band:.4f}')
print(f'clean split (p90 reject < p10 accept)? bokeh: {lo_bok < hi_band}  flat-texture: {lo_flat < hi_band}')

# candidate thresholds: evaluate actual accept/reject fractions
for lo, hi in [(0.01,0.03),(0.015,0.04),(0.02,0.05),(0.02,0.06),(0.03,0.08)]:
    def frac_below(v,t): return (v < t).mean()
    def frac_above(v,t): return (v > t).mean()
    # smoothstep gate value mean per population (what the pass actually applies)
    def gate(v):
        t = np.clip((v-lo)/(hi-lo), 0, 1); return (t*t*(3-2*t)).mean()
    print(f'lo={lo:.3f} hi={hi:.3f} | bokeh<lo: {frac_below(bok,lo)*100:5.1f}%  flat<lo: {frac_below(soft_vals,lo)*100:5.1f}%  '
          f'band>hi: {frac_above(band_vals,hi)*100:5.1f}% | mean gate: bokeh {gate(bok):.3f} flat {gate(soft_vals):.3f} band {gate(band_vals):.3f}')
