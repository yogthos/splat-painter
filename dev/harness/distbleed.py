#!/usr/bin/env python3
"""Colour/luma error inside the dark side of a silhouette, BINNED BY DISTANCE from
the boundary. A whole-ROI mean hides a bleed confined to a few pixels at the edge;
if the bleed is real it decays with distance, and that shape is the evidence.

Boundary is taken from the SOURCE (so it is identical for every render): the dark
mask is source luma < DARK, and distance is the Euclidean distance transform of
that mask -- distance 1 = immediately inside the dark region.

Reports, per distance bin: dLuma (render - source), dC (chroma change) and dHue
(shift projected onto the light side's own chroma direction, so + = took on the
background's tone).

Usage: distbleed.py ROW0 ROW1 COL0 COL1 render1.png [render2.png ...]
"""
import sys
import numpy as np
from PIL import Image


def distance_transform_edt(mask, maxd=48):
    """Chessboard-ish distance to the nearest False cell, by iterated 4-neighbour
    erosion. Exact enough for binning at 1px granularity and avoids a scipy dep."""
    m = mask.copy()
    d = np.zeros(mask.shape, dtype=np.float64)
    for step in range(1, maxd + 1):
        if not m.any():
            break
        d[m] = step
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        e[0, :] = False
        e[-1, :] = False
        e[:, 0] = False
        e[:, -1] = False
        m = e
    return d

SRC = '/Users/yogthos/src/splat-painter/img/Lenin.jpg'
DARK = 0.25
BINS = [(1, 2), (2, 4), (4, 6), (6, 9), (9, 13), (13, 20), (20, 40)]


def load(p, size=None):
    im = Image.open(p).convert('RGB')
    if size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def opp(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return np.stack([r - g, 0.5 * (r + g) - b], axis=-1)


def luma(a):
    return 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]


def main():
    r0, r1, c0, c1 = (int(v) for v in sys.argv[1:5])
    paths = sys.argv[5:]
    ref = Image.open(paths[0])
    src = load(SRC, ref.size)
    sl, so = luma(src), opp(src)

    S = sl[r0:r1, c0:c1]
    dark = S < DARK
    dist = distance_transform_edt(dark)
    light = ~dark
    # the bleeding tone: mean chroma of the LIGHT side within 6px of the boundary
    ldist = distance_transform_edt(light)
    near_light = light & (ldist <= 6)
    mdir = opp(src)[r0:r1, c0:c1][near_light].mean(axis=0)
    mdir = mdir / (np.linalg.norm(mdir) + 1e-12)
    print(f'box rows {r0}-{r1} cols {c0}-{c1}   dark px {int(dark.sum())}   '
          f'light-side chroma dir [{mdir[0]:+.3f} {mdir[1]:+.3f}]')

    Sopp = so[r0:r1, c0:c1]
    for p in paths:
        R = load(p)
        rl = luma(R)[r0:r1, c0:c1]
        ro = opp(R)[r0:r1, c0:c1]
        dl = rl - S
        dc = np.linalg.norm(ro, axis=-1) - np.linalg.norm(Sopp, axis=-1)
        dh = (ro - Sopp) @ mdir
        print(f'\n{p.split("/")[-1]}')
        print(f'  {"dist(px)":>9s} {"n":>7s} {"dLuma":>9s} {"dC":>9s} {"dHue":>9s}')
        for lo, hi in BINS:
            m = dark & (dist >= lo) & (dist < hi)
            if m.sum() < 20:
                continue
            print(f'  {f"{lo}-{hi}":>9s} {int(m.sum()):7d} {dl[m].mean():+9.4f} '
                  f'{dc[m].mean():+9.4f} {dh[m].mean():+9.4f}')


if __name__ == '__main__':
    main()
