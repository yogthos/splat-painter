#!/usr/bin/env python3
"""Edge sharpness (README definition) measured separately in a DETAILED foreground
box and a SMOOTH background box, which is the whole point of w4w: the Hardness dial
should move the foreground far more than the background.

  grad  = hypot(central difference x, y)
  band  = source grad > its 85th percentile within the box
  score = mean(grad_render[band]) / mean(grad_source[band])

Registration: the render sits 1px above a Lanczos downscale of the source, so the
render is compared shifted (found once, per file, over a +/-2px search).

Usage: crisp.py render1.png [render2.png ...]
"""
import sys
import numpy as np
from PIL import Image

SRC = '/Users/yogthos/src/splat-painter/img/Lenin.jpg'
# face: the most detailed region of the frame (worst.py ranks it top vs source)
FG = (60, 200, 570, 700)
# top-right map corner: the only region measuring subjAbs 0.337, i.e. genuinely flat
BG = (0, 140, 890, 1020)


def luma(p, size=None):
    im = Image.open(p).convert('RGB')
    if size:
        im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64) / 255.0
    return 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]


def grad(a):
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def score(r, s, box):
    r0, r1, c0, c1 = box
    gs, gr = grad(s[r0:r1, c0:c1]), grad(r[r0:r1, c0:c1])
    band = gs > np.percentile(gs, 85)
    return gr[band].mean() / gs[band].mean()


def main():
    ref = Image.open(sys.argv[1])
    s = luma(SRC, ref.size)
    print(f'{"render":22s} {"FG(face)":>9s} {"BG(flat)":>9s} {"FG/BG":>7s} {"WHOLE":>7s}')
    for p in sys.argv[1:]:
        r = luma(p)
        best = None
        for dx in (-2, -1, 0, 1, 2):
            rr = np.roll(r, dx, axis=0)
            m = np.abs(rr[4:-4] - s[4:-4]).mean()
            if best is None or m < best[0]:
                best = (m, rr)
        r = best[1]
        fg, bg = score(r, s, FG), score(r, s, BG)
        # WHOLE is the overall-strength column: raising the floor trades separation
        # for global crispness, and without this the trade is invisible.
        whole = score(r, s, (4, s.shape[0] - 4, 4, s.shape[1] - 4))
        print(f'{p.split("/")[-1]:22s} {fg:9.4f} {bg:9.4f} {fg/bg:7.3f} {whole:7.4f}')


if __name__ == '__main__':
    main()
