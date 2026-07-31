#!/usr/bin/env python3
"""A3: sharpen amount sweep — face edge sharpness, bokeh background gradient
energy ratio, whole-image mean|d|, all against the Lanczos-downscaled source
with the 1px registration (render row i == source row i+1)."""
import numpy as np
from PIL import Image

SRC = '/Users/yogthos/src/splat-painter/img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg'
RENDERS = [(0.0, '/tmp/sharpen-post.png'), (0.3, '/tmp/sharpen-0.3.png'),
           (0.6, '/tmp/sharpen-0.6.png'), (1.0, '/tmp/sharpen-1.0.png'),
           (1.5, '/tmp/sharpen-1.5.png')]
BX = (790, 1010); BYs = [(140, 300), (460, 620)]   # the Step-1 bokeh box

def luma(path, size=None):
    im = Image.open(path).convert('RGB')
    if size: im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)/255.0
    return 0.2126*a[:,:,0] + 0.7152*a[:,:,1] + 0.0722*a[:,:,2]

def cdiff(g): return np.hypot(*np.gradient(g))

def sobel(g):
    p = np.pad(g, 1, mode='edge')
    gx = (p[:-2,2:] + 2*p[1:-1,2:] + p[2:,2:]) - (p[:-2,:-2] + 2*p[1:-1,:-2] + p[2:,:-2])
    gy = (p[2:,:-2] + 2*p[2:,1:-1] + p[2:,2:]) - (p[:-2,:-2] + 2*p[:-2,1:-1] + p[:-2,2:])
    return np.hypot(gx, gy)/8.0

s = luma(SRC, (1024,1024))
gsrc = cdiff(s)
band = gsrc > np.percentile(gsrc, 85)
band_r = band[1:, :]
src_r = s[1:, :]
gmean_src = gsrc[1:, :][band_r].mean()
bok = np.zeros((1024,1024), bool)
for by in BYs: bok[by[0]:by[1], BX[0]:BX[1]] = True

rows = []
for amt, path in RENDERS:
    r = luma(path)
    sharp = cdiff(r)[:-1, :][band_r].mean() / gmean_src
    bg = sobel(r)[bok].mean()
    md = np.abs(r[:-1, :] - src_r).mean()
    rows.append((amt, sharp, bg, md))

bg0 = rows[0][2]
print(f'{"amount":>7} {"face sharpness":>15} {"bg energy ratio":>16} {"mean|d|":>10}')
for amt, sharp, bg, md in rows:
    print(f'{amt:7.1f} {sharp:15.4f} {bg/bg0:16.4f} {md:10.5f}')
print()
hit = next(((a, s_, b/bg0) for a, s_, b, _ in rows if s_ >= 0.85), None)
if hit:
    print(f'bar: sharpness first reaches >=0.85 at amount {hit[0]} (={hit[1]:.4f}); '
          f'background energy x{hit[2]:.4f} there — {"PASS (<1.15)" if hit[2] < 1.15 else "FAIL"}')
else:
    best = max(rows, key=lambda r: r[1])
    print(f'bar: 0.85 NOT reached; max {best[1]:.4f} at amount {best[0]}, bg x{best[2]/bg0:.4f}')
