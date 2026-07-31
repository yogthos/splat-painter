#!/usr/bin/env python3
"""Signed luma difference (render - source) over the nose/lip ROI.

Blue = render too DARK, red = render too BRIGHT, grey = faithful. The defect
should appear as a solid blue slab where the render keeps painting shadow the
source has already faded out of.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw

X0, Y0, X1, Y1 = 520, 405, 610, 455
ZOOM = 10
GAIN = 4.0          # amplify the difference so a 15/255 error is visible

# the render sits 1px above my Lanczos downscale of the source (measured by
# minimising mean|d| over +-3px); without compensating, every edge shows a
# blue/red dipole that is MY registration error, not the painter's.
DY = -1

def luma(path, shift=False):
    im = Image.open(path).convert('RGB')
    dy = DY if shift else 0
    a = np.asarray(im.crop((X0, Y0 + dy, X1, Y1 + dy)), dtype=np.float64)
    return 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]

def tile(d):
    h, w = d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    neg = np.clip(-d * GAIN, 0, 255)      # render darker than source
    pos = np.clip(d * GAIN, 0, 255)       # render brighter than source
    rgb[:, :, 2] = neg
    rgb[:, :, 0] = pos
    rgb[:, :, 1] = 0
    base = 40.0
    rgb = np.clip(rgb + base, 0, 255)
    im = Image.fromarray(rgb.astype(np.uint8))
    return im.resize((w * ZOOM, h * ZOOM), Image.NEAREST)

def main():
    src = luma(sys.argv[1])
    labels, paths = sys.argv[2::2], sys.argv[3::2]
    tiles = []
    for lab, p in zip(labels, paths):
        d = luma(p, shift=True) - src
        # report the worst over-dark region, excluding the jacket columns (x>=75)
        core = d[:, :70]
        print(f'{lab:22s} mean|d|={np.abs(core).mean():6.2f}  '
              f'worst-dark={core.min():7.2f}  '
              f'px over 20 too dark={(core < -20).sum():5d}')
        tiles.append((lab, tile(d)))
    w, h = tiles[0][1].size
    pad = 28
    sheet = Image.new('RGB', (w, (h + pad) * len(tiles)), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (lab, t) in enumerate(tiles):
        y = i * (h + pad)
        dr.text((6, y + 8), lab + '   (blue = render too dark)', fill=(255, 220, 120))
        sheet.paste(t, (0, y + pad))
    sheet.save('diffsheet.png')
    print('wrote diffsheet.png')

if __name__ == '__main__':
    main()
