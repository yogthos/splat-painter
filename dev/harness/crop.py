#!/usr/bin/env python3
"""Crop the nose ROI at 10x from each isolation render and tile them side by side."""
import sys
from PIL import Image, ImageDraw

X0, Y0, X1, Y1 = 520, 405, 610, 455
ZOOM = 10

def crop(path):
    im = Image.open(path).convert('RGB')
    c = im.crop((X0, Y0, X1, Y1))
    return c.resize(((X1 - X0) * ZOOM, (Y1 - Y0) * ZOOM), Image.NEAREST)

def main():
    labels = sys.argv[1::2]
    paths = sys.argv[2::2]
    out = paths[-1] if len(paths) > len(labels) else None
    tiles = [(l, crop(p)) for l, p in zip(labels, paths)]
    w, h = tiles[0][1].size
    pad = 28
    sheet = Image.new('RGB', (w, (h + pad) * len(tiles)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, (label, t) in enumerate(tiles):
        y = i * (h + pad)
        d.text((6, y + 8), label, fill=(255, 220, 120))
        sheet.paste(t, (0, y + pad))
    sheet.save('/private/tmp/claude-501/-Users-yogthos-src-splat-painter/61ecf884-c16f-4cf0-a9e5-35710884657a/scratchpad/iso-sheet.png')
    print('wrote iso-sheet.png', sheet.size)

if __name__ == '__main__':
    main()
