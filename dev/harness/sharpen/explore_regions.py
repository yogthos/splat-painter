#!/usr/bin/env python3
"""Coarse tile map of gradient energy to pick bokeh + flat-texture boxes."""
import numpy as np
from PIL import Image

REN = '/private/tmp/claude-501/-Users-yogthos-src-splat-painter/61ecf884-c16f-4cf0-a9e5-35710884657a/scratchpad/baseline-sharpen.png'
SRC = '/Users/yogthos/src/splat-painter/img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg'

def luma(a):
    return 0.2126*a[:,:,0] + 0.7152*a[:,:,1] + 0.0722*a[:,:,2]

r = luma(np.asarray(Image.open(REN).convert('RGB'), dtype=np.float64)/255.0)
s = luma(np.asarray(Image.open(SRC).convert('RGB').resize((1024,1024), Image.LANCZOS), dtype=np.float64)/255.0)

def sobel(g):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # separable conv, edge-padded (matches shader clamp)
    p = np.pad(g, 1, mode='edge')
    gx = (p[:-2,2:] + 2*p[1:-1,2:] + p[2:,2:]) - (p[:-2,:-2] + 2*p[1:-1,:-2] + p[2:,:-2])
    gy = (p[2:,:-2] + 2*p[2:,1:-1] + p[2:,2:]) - (p[:-2,:-2] + 2*p[:-2,1:-1] + p[:-2,2:])
    return np.hypot(gx, gy)/8.0

gr, gs = sobel(r), sobel(s)
T = 64  # tile size
print('mean render-sobel per %dpx tile (rows=y, cols=x), x1000:' % T)
m = gr.reshape(1024//T, T, 1024//T, T).mean(axis=(1,3))
np.set_printoptions(linewidth=250, suppress=True)
print((m*1000).astype(int))
print('mean source-sobel per tile, x1000:')
ms = gs.reshape(1024//T, T, 1024//T, T).mean(axis=(1,3))
print((ms*1000).astype(int))
print('render-sobel percentiles overall:', [round(float(np.percentile(gr,q)),4) for q in (10,50,90,99)])
print('source-sobel p85:', round(float(np.percentile(gs,85)),4))
