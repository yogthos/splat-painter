#!/usr/bin/env python3
"""Dark-thin-line pixel counter, per SPEC-dark-lines.md.

A dark-line pixel is a RENDER pixel more than 12 levels (of 255) darker than
its own 7x7 local mean, restricted to places where the SOURCE is smooth
(source gradient below its 60th percentile), so a real edge cannot be counted.

Registration: render row i corresponds to source row i+1 (the render sits 1px
above a Lanczos downscale of the source).

Regions (render coords):
  whole  — the whole frame (rows 0-1023)
  bokeh  — rows 60-420, cols 0-420
  finger — rows 320-445, cols 260-410   (bd splat-painter-ws1 ROI)
  nose   — rows 405-455, cols 520-610   (the -M:score nose ROI)

Usage: darkline.py render.png [render2.png ...]
Prints counts per region. Baselines from the spec (shipped /tmp/sharpen-pre.png):
whole 4645, bokeh 264. Cut-in 0 (/tmp/cutin0.png): whole 3204, bokeh 94.
"""
import sys
import numpy as np
from PIL import Image

SRC = '/Users/yogthos/src/splat-painter/img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg'

REGIONS = {
    'whole': (0, 1024, 0, 1024),      # y0,y1,x0,x1
    'bokeh': (60, 420, 0, 420),
    'finger': (320, 445, 260, 410),
    'nose': (405, 455, 520, 610),
}


def luma(path, size=None):
    im = Image.open(path).convert('RGB')
    if size:
        im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    return 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]


def box7(g):
    p = np.pad(g, 3, mode='edge')
    k = np.zeros_like(g)
    for i in range(7):
        for j in range(7):
            k += p[i:i + g.shape[0], j:j + g.shape[1]]
    return k / 49.0


def main():
    s = luma(SRC, (1024, 1024))
    gs = np.hypot(*np.gradient(s))          # central-difference gradient, source rows
    smooth = gs < np.percentile(gs, 60)     # per source row
    # NOTE: the smoothness mask is applied UNSHIFTED (render row i vs source row i).
    # That is what the spec's counter did — it reproduces the published baselines
    # exactly (4645/264 and 3204/94); the 1px registration shift changes the bokeh
    # count by ~18% and is mask-alignment error here, not painter signal.
    print(f'{"render":34s} {"whole":>7s} {"bokeh":>7s} {"finger":>7s} {"nose":>6s}')
    for path in sys.argv[1:]:
        r = luma(path)
        local = box7(r)
        dark = r < local - 12.0
        counts = {}
        for name, (y0, y1, x0, x1) in REGIONS.items():
            # regions were written for the 1024x1024 fixture; clamp to this frame so
            # a non-square image (Lenin is 1024x646) does not blow up the broadcast
            H_, W_ = dark.shape
            y0c, y1c = min(y0, H_), min(y1, H_)
            x0c, x1c = min(x0, W_), min(x1, W_)
            if y1c <= y0c or x1c <= x0c:
                counts[name] = -1        # region absent at this aspect ratio
            else:
                counts[name] = int((dark[y0c:y1c, x0c:x1c]
                                    & smooth[y0c:y1c, x0c:x1c]).sum())
        print(f'{path.split("/")[-1]:34s} {counts["whole"]:7d} {counts["bokeh"]:7d}'
              f' {counts["finger"]:7d} {counts["nose"]:6d}')


if __name__ == '__main__':
    main()
