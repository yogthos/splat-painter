#!/usr/bin/env python3
"""Simulate the exact sharpen pass offline on the baseline render.

Mirrors the planned GLSL: 3x3 taps (edge-clamped), binomial blur /16,
luma high-pass, Sobel gate smoothstep(lo,hi), out = clamp(c + amount*gate*hp).
Evaluates the A3 metrics for candidate (lo,hi) x amount.
"""
import numpy as np
from PIL import Image

REN = '/private/tmp/claude-501/-Users-yogthos-src-splat-painter/61ecf884-c16f-4cf0-a9e5-35710884657a/scratchpad/baseline-sharpen.png'
SRC = '/Users/yogthos/src/splat-painter/img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg'

def load(path, size=None):
    im = Image.open(path).convert('RGB')
    if size: im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)/255.0

def luma(a): return 0.2126*a[:,:,0] + 0.7152*a[:,:,1] + 0.0722*a[:,:,2]

def cdiff(g): return np.hypot(*np.gradient(g))

def sharpen(c, amount, lo, hi):
    lum = luma(c)
    p = np.pad(lum, 1, mode='edge')
    t = [p[0:-2,0:-2], p[0:-2,1:-1], p[0:-2,2:],
         p[1:-1,0:-2], p[1:-1,1:-1], p[1:-1,2:],
         p[2:,0:-2],  p[2:,1:-1],  p[2:,2:]]
    blur = (t[0] + 2*t[1] + t[2] + 2*t[3] + 4*t[4] + 2*t[5] + t[6] + 2*t[7] + t[8])/16.0
    hp = lum - blur
    gx = (t[2] + 2*t[5] + t[8]) - (t[0] + 2*t[3] + t[6])
    gy = (t[6] + 2*t[7] + t[8]) - (t[0] + 2*t[1] + t[2])
    g = np.hypot(gx, gy)/8.0
    x = np.clip((g - lo)/(hi - lo), 0.0, 1.0)
    gate = x*x*(3 - 2*x)
    return np.clip(c + amount*gate[:,:,None]*hp[:,:,None], 0.0, 1.0)

def sobel(g):
    p = np.pad(g, 1, mode='edge')
    gx = (p[:-2,2:] + 2*p[1:-1,2:] + p[2:,2:]) - (p[:-2,:-2] + 2*p[1:-1,:-2] + p[2:,:-2])
    gy = (p[2:,:-2] + 2*p[2:,1:-1] + p[2:,2:]) - (p[:-2,:-2] + 2*p[:-2,1:-1] + p[:-2,2:])
    return np.hypot(gx, gy)/8.0

base = load(REN)
src = load(SRC, (1024,1024))
src_lum = luma(src)
gsrc = cdiff(src_lum)
band = gsrc > np.percentile(gsrc, 85)
band_r = band[1:, :]          # registration: render[i] ~ source[i+1]
BX = (790, 1010); BYs = [(140, 300), (460, 620)]
bok_mask = np.zeros((1024,1024), bool)
for by in BYs: bok_mask[by[0]:by[1], BX[0]:BX[1]] = True

src_lum_r = src_lum[1:, :]
gmean_src_band = gsrc[1:, :][band_r].mean()
bg0 = sobel(luma(base))[bok_mask].mean()
d0 = np.abs(luma(base)[:-1,:] - src_lum_r).mean()

print(f'base: sharpness={cdiff(luma(base))[:-1,:][band_r].mean()/gmean_src_band:.4f}  bg={bg0:.5f}  mean|d|={d0:.5f}')
print()
hdr = f'{"lo":>6} {"hi":>6} {"amt":>5} {"sharp":>7} {"bg_ratio":>9} {"mean|d|":>9}'
print(hdr)
for lo, hi in [(0.010,0.030),(0.015,0.040),(0.020,0.050),(0.010,0.040),(0.015,0.030),(0.020,0.060)]:
    for amt in (0.3, 0.6, 1.0, 1.5):
        out = sharpen(base, amt, lo, hi)
        ol = luma(out)
        sharp = cdiff(ol)[:-1,:][band_r].mean()/gmean_src_band
        bgr = sobel(ol)[bok_mask].mean()/bg0
        md = np.abs(ol[:-1,:] - src_lum_r).mean()
        print(f'{lo:6.3f} {hi:6.3f} {amt:5.2f} {sharp:7.4f} {bgr:9.4f} {md:9.5f}')
    print()
