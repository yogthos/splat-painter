#!/usr/bin/env python3
"""Score a render against the source: local (nose ROI) AND whole-image.

A change that only improves the ROI is not a fix -- it has to not wreck the rest
of the painting. Whole-image error is reported alongside so a local win bought
with a global loss is visible immediately.

Registration: the render sits 1px above a Lanczos downscale of the source
(measured by minimising mean|d|); without compensating, every edge contributes a
dipole that is comparison error, not painter error.
"""
import sys
import numpy as np
from PIL import Image

X0, Y0, X1, Y1 = 520, 405, 610, 455
DY = -1


def luma_full(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)
    return 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]


def score(src_path, path):
    s = luma_full(src_path)
    r = luma_full(path)
    # measured alignment: render row y-1 corresponds to source row y, so compare
    # r[:-1] against s[1:]. Row i of d is source row i+1.
    d_full = r[:-1, :] - s[1:, :]
    roi = d_full[Y0 - 1:Y1 - 1, X0:X1][:, :70]
    return {
        'roi_mean': np.abs(roi).mean(),
        'roi_dark': int((roi < -20).sum()),
        'roi_worst': roi.min(),
        'full_mean': np.abs(d_full).mean(),
        'full_rms': float(np.sqrt((d_full ** 2).mean())),
    }


def main():
    src = sys.argv[1]
    labels, paths = sys.argv[2::2], sys.argv[3::2]
    print(f'{"variant":24s} {"ROI mean|d|":>11s} {"ROI dark":>9s} {"ROI worst":>10s}'
          f' {"FULL mean|d|":>12s} {"FULL rms":>9s}')
    base = None
    for lab, p in zip(labels, paths):
        s = score(src, p)
        if base is None:
            base = s
        dr = s['roi_mean'] - base['roi_mean']
        df = s['full_mean'] - base['full_mean']
        print(f'{lab:24s} {s["roi_mean"]:11.3f} {s["roi_dark"]:9d} {s["roi_worst"]:10.1f}'
              f' {s["full_mean"]:12.3f} {s["full_rms"]:9.3f}'
              f'   ({dr:+.3f} ROI, {df:+.3f} full)')


if __name__ == '__main__':
    main()
