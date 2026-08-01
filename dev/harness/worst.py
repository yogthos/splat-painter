#!/usr/bin/env python3
"""Where does the render deviate most from the source? Ranks blocks by mean|d|
after finding the best integer registration offset, so the 1px shift between the
render and a Lanczos downscale of the source does not masquerade as error.

Usage: worst.py render.png [--block N] [--top N]
"""
import sys
import numpy as np
from PIL import Image

SRC = '/Users/yogthos/src/splat-painter/img/Lenin.jpg'


def load(p, size=None):
    im = Image.open(p).convert('RGB')
    if size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def main():
    p = sys.argv[1]
    block = int(sys.argv[sys.argv.index('--block') + 1]) if '--block' in sys.argv else 64
    top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 14
    r = load(p)
    s = load(SRC, (r.shape[1], r.shape[0]))

    best, bo = None, None
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            rr = np.roll(np.roll(r, dx, axis=0), dy, axis=1)[4:-4, 4:-4]
            ss = s[4:-4, 4:-4]
            m = np.abs(rr - ss).mean()
            if best is None or m < best:
                best, bo = m, (dx, dy)
    dx, dy = bo
    r = np.roll(np.roll(r, dx, axis=0), dy, axis=1)
    print(f'{p.split("/")[-1]}: registration offset (row,col)={bo}, '
          f'whole-image mean|d| {best:.3f}/255')

    d = np.abs(r - s).mean(axis=2)
    H, W = d.shape
    blocks = []
    for x in range(4, H - block - 4, block):
        for y in range(4, W - block - 4, block):
            blocks.append((d[x:x + block, y:y + block].mean(), x, y))
    blocks.sort(reverse=True)
    print(f'worst {top} {block}x{block} blocks vs source:')
    for m, x, y in blocks[:top]:
        print(f'   rows {x:4d}-{x+block:4d}  cols {y:4d}-{y+block:4d}   mean|d| {m:6.2f}')


if __name__ == '__main__':
    main()
