#!/usr/bin/env python3
"""Where does a control actually act? Whole-image diff between two renders plus a
coarse block map of the biggest-changing regions, so an effect that is real but
outside a hand-picked ROI still shows up.

Usage: whereis.py A.png B.png [--top N]
"""
import sys
import numpy as np
from PIL import Image

BLOCK = 64


def load(p):
    return np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 12
    d = np.abs(a - b).mean(axis=2)
    print(f'{sys.argv[1].split("/")[-1]} vs {sys.argv[2].split("/")[-1]}')
    print(f'  whole-image mean|d| {d.mean():.4f}/255   max {d.max():.1f}   '
          f'pct>5 {(d > 5).mean() * 100:.2f}%   pct>15 {(d > 15).mean() * 100:.2f}%')
    H, W = d.shape
    blocks = []
    for x in range(0, H - BLOCK + 1, BLOCK):
        for y in range(0, W - BLOCK + 1, BLOCK):
            blocks.append((d[x:x + BLOCK, y:y + BLOCK].mean(), x, y))
    blocks.sort(reverse=True)
    print(f'  top {top} {BLOCK}x{BLOCK} blocks by mean|d| (row, col):')
    for m, x, y in blocks[:top]:
        print(f'    rows {x:4d}-{x+BLOCK:4d}  cols {y:4d}-{y+BLOCK:4d}   mean|d| {m:6.3f}')


if __name__ == '__main__':
    main()
