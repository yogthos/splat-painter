import sys
from PIL import Image

def gray(p, size=None):
    im = Image.open(p).convert('L')
    if size and im.size != size: im = im.resize(size, Image.LANCZOS)
    return im

def runs(ps, w, y0, y1):
    """Maximal runs of columns with no source ink — the spaces between marks."""
    out, start = [], None
    for x in range(w):
        blank = min(ps[x, y] for y in range(y0, y1)) > 200
        if blank and start is None: start = x
        elif not blank and start is not None:
            out.append((start, x)); start = None
    if start is not None: out.append((start, w))
    return out

def report(render, src='img/collapse-watch.jpg', y0=77, y1=96):
    r = gray(render); s = gray(src, r.size)
    pr, ps = r.load(), s.load(); w, _ = r.size
    rs = runs(ps, w, y0, y1)
    def ink(a, b):
        return sum(255 - pr[x, y] for x in range(a, b) for y in range(y0, y1)) / max(1,(b-a)*(y1-y0))
    # narrow = inside a word (the ones that close up); wide = between words
    narrow = [(a,b) for a,b in rs if b-a <= 6]
    wide   = [(a,b) for a,b in rs if b-a > 6]
    ni = sum(ink(a,b)*(b-a) for a,b in narrow)/max(1,sum(b-a for a,b in narrow))
    wi = sum(ink(a,b)*(b-a) for a,b in wide)/max(1,sum(b-a for a,b in wide))
    return ni, wi, len(narrow), len(wide)

for p in sys.argv[1:]:
    ni, wi, n, m = report(p)
    print("%-26s narrow-gap ink %6.2f (%d gaps)   wide-gap ink %6.2f (%d)" %
          (p.split('/')[-1], ni, n, wi, m))
