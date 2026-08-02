#!/usr/bin/env python3
"""Edge WIDTH at a silhouette — a translation-invariant measure of the artifact.

Why not distbleed: that bins error by distance from the boundary, so a sub-pixel
shift in where the render puts the edge moves signal between bins. Renders do not
share a common sub-pixel edge position (integer registration measured (1,1), (1,0)
and (0,1) across one sweep), so its 1-2px bin is dominated by registration, not by
the painter. It produced a non-monotonic Cut-in curve whose jumps line up exactly
with the offset changes.

Edge width does not care where the edge is. Along each scanline crossing the
boundary:

    width = total_variation / peak_gradient

For an ideal step this is 1 px; for a transition smeared over N px it is ~N. Both
terms are shift-invariant, so translating the profile changes nothing. Reported as
the render's width relative to the source's over the same scanlines: 1.0 means the
render's edge is exactly as tight as the source's, 2.0 means twice as smeared.

Usage: edgewidth.py ROW0 ROW1 COL0 COL1 render1.png [render2.png ...]
"""
import sys
import numpy as np
from PIL import Image

SRC = '/Users/yogthos/src/splat-painter/img/Lenin.jpg'
MIN_CONTRAST = 0.12       # ignore scanlines with no real edge in them


def luma(p, size=None):
    im = Image.open(p).convert('RGB')
    if size:
        im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64) / 255.0
    return 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]


def widths(img, box):
    """Per-scanline width = total variation / peak gradient, across the COL axis."""
    r0, r1, c0, c1 = box
    roi = img[r0:r1, c0:c1]
    d = np.diff(roi, axis=1)
    tv = np.abs(d).sum(axis=1)
    peak = np.abs(d).max(axis=1)
    contrast = roi.max(axis=1) - roi.min(axis=1)
    ok = (contrast > MIN_CONTRAST) & (peak > 1e-6)
    return tv[ok] / peak[ok], ok


def main():
    box = tuple(int(v) for v in sys.argv[1:5])
    paths = sys.argv[5:]
    ref = Image.open(paths[0])
    s = luma(SRC, ref.size)
    sw, ok = widths(s, box)
    print(f'box {box}   scanlines used {int(ok.sum())}   '
          f'source edge width {sw.mean():.3f} px')
    print(f'{"render":18s} {"width":>8s} {"vs source":>10s}')
    for p in paths:
        rw, ok2 = widths(luma(p), box)
        n = min(len(sw), len(rw))
        print(f'{p.split("/")[-1]:18s} {rw.mean():8.3f} {rw.mean() / sw.mean():10.3f}')


if __name__ == '__main__':
    main()
