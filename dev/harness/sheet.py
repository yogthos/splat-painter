#!/usr/bin/env python3
"""Tile a labelled ROI crop from the source and several renders, so the thing being
measured can be looked at. Usage: sheet.py out.png ROW0 ROW1 COL0 COL1 ZOOM label=path ...
"""
import sys
from PIL import Image, ImageDraw

SRC = '/Users/yogthos/src/splat-painter/img/Lenin.jpg'


def main():
    out = sys.argv[1]
    r0, r1, c0, c1, zoom = (int(v) for v in sys.argv[2:7])
    items = [a.split('=', 1) for a in sys.argv[7:]]

    def crop(im):
        c = im.crop((c0, r0, c1, r1))
        return c.resize(((c1 - c0) * zoom, (r1 - r0) * zoom), Image.NEAREST)

    ref = Image.open(items[0][1])
    src = Image.open(SRC).convert('RGB').resize(ref.size, Image.LANCZOS)
    tiles = [('source', crop(src))]
    tiles += [(lbl, crop(Image.open(p).convert('RGB'))) for lbl, p in items]

    w, h = tiles[0][1].size
    pad = 24
    cols = len(tiles)
    sheet = Image.new('RGB', (w * cols, h + pad), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for i, (lbl, t) in enumerate(tiles):
        sheet.paste(t, (i * w, pad))
        d.text((i * w + 6, 7), lbl, fill=(255, 215, 110))
    sheet.save(out)
    print('wrote', out, sheet.size)


if __name__ == '__main__':
    main()
