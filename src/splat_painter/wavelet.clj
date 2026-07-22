(ns splat-painter.wavelet
  "Multi-scale image DETAIL via the Haar wavelet — a 2D port of wavescope-mcp's
   haar.ts (approx=(a+b)/2, detail=a−b). Where wavescope decomposes a 1D
   line-importance signal to find structural boundaries, we decompose the 2D
   luminance to find where fine texture/edges live at multiple scales. The
   per-cell detail energy (summed |detail| across Haar levels) then drives
   adaptive seeding: more, smaller splats where detail is high; fewer, larger
   ones where the image is flat.

   Coordinate convention matches splat-painter.structure: x=row, y=col; the map is
   computed at a reduced resolution and carries :src-h/:src-w so callers map
   full-image coords into it (see detail-at)."
  (:require [splat-painter.structure :as structure]))

;; --- Haar 1D (a faithful port of haar.ts haarFwd1d) --------------------------
;; Odd-length: the last element is dropped. Operates on a strided slice of a flat
;; ^doubles array so the same code does rows (stride 1) and columns (stride W).
(defn- haar-detail-energy!
  "One forward Haar level over `src` (Hc×Wc, row-major), writing the LL
   approximation into `ll` (Hc/2 × Wc/2) and ADDING |LH|+|HL|+|HH| detail energy
   for each half-cell into `acc` over its `block`×`block` region of the full
   (H×W) grid at offset (or0,oc0). Returns [hc2 wc2] (the LL dims)."
  [^doubles src Hc Wc ^doubles ll ^doubles acc H W block]
  (let [hc2 (quot Hc 2) wc2 (quot Wc 2)]
    (dotimes [r hc2]
      (dotimes [c wc2]
        (let [r0 (* 2 r) r1 (inc r0) c0 (* 2 c) c1 (inc c0)
              a (aget src (+ (* r0 Wc) c0)) b (aget src (+ (* r0 Wc) c1))
              d (aget src (+ (* r1 Wc) c0)) e (aget src (+ (* r1 Wc) c1))
              ;; 2x2 Haar: LL=mean, LH=vertical, HL=horizontal, HH=diagonal detail
              ll-v (* 0.25 (+ a b d e))
              lh (- (+ a b) (+ d e))       ; vertical detail (top row − bottom row)
              hl (- (+ a d) (+ b e))       ; horizontal detail (left col − right col)
              hh (- (+ a e) (+ b d))       ; diagonal detail
              energy (+ (Math/abs lh) (Math/abs hl) (Math/abs hh))]
          (aset ll (+ (* r wc2) c) ll-v)
          ;; scatter this half-cell's detail energy over its block in the full grid
          (let [br (* r block) bc (* c block)]
            (dotimes [i block]
              (dotimes [j block]
                (let [gr (+ br i) gc (+ bc j)]
                  (when (and (< gr H) (< gc W))
                    (let [gi (+ (* gr W) gc)]
                      (aset acc gi (+ (aget acc gi) energy)))))))))))
    [hc2 wc2]))

(defn detail-map
  "Haar multi-scale detail energy of `image`'s luminance at ≤`max-side`, over
   `levels` scales. Returns {:h Ht :w Wt :detail ^doubles :dmax :src-h :src-w}:
   detail[i] is the summed |Haar detail| across scales covering cell i — high in
   textured/edgy regions, ~0 in flat ones. dmax is the max (for normalization)."
  ([image] (detail-map image 384 4))
  ([image max-side levels]
   (let [H (:height image) W (:width image)
         scale (min 1.0 (/ (double max-side) (double (max H W))))
         sh (max 1 (long (Math/round (* H scale))))
         sw (max 1 (long (Math/round (* W scale))))
         ^doubles lum (structure/luma-of image sh sw)   ; reduced-res luminance
         n (* sh sw)
         acc (double-array n)]
     ;; lum is a fresh array we own and haar-detail-energy! only READS src, so we
     ;; can thread it (then each level's LL) directly with no copy.
     (loop [lev 1 ch sh cw sw ^doubles src lum block 2]
       (when (and (<= lev levels) (>= ch 2) (>= cw 2))
         (let [hc2 (quot ch 2) wc2 (quot cw 2)
               ll (double-array (* hc2 wc2))]
           (haar-detail-energy! src ch cw ll acc sh sw block)
           (recur (inc lev) hc2 wc2 ll (* 2 block)))))
     ;; the per-level block-scatter above gives `acc` a blocky (grid-of-2^level)
     ;; structure; smoothing it removes that block pattern so the detail-driven splat
     ;; size/density varies gradually — no cross-hatch weave in the output.
     (let [^doubles acc (structure/box-blur acc sh sw (max 2 (quot (min sh sw) 24)))
           dmax (loop [i 0 m 0.0] (if (< i n) (recur (inc i) (max m (aget acc i))) m))]
       {:h sh :w sw :detail acc :dmax dmax :src-h H :src-w W}))))

(defn detail-at
  "Normalized detail ∈ [0,1] at full-image coords (x=row, y=col), sampled from the
   reduced-res map. 0 = flat, 1 = the most detailed cell."
  [dmap x y]
  (let [H (:h dmap) W (:w dmap)
        src-h (long (or (:src-h dmap) H)) src-w (long (or (:src-w dmap) W))
        ^doubles d (:detail dmap)
        dmax (double (:dmax dmap))
        xi (min (dec H) (max 0 (long (Math/round (* (double x) (/ (double H) src-h))))))
        yi (min (dec W) (max 0 (long (Math/round (* (double y) (/ (double W) src-w))))))]
    (if (pos? dmax) (min 1.0 (/ (aget d (+ (* xi W) yi)) dmax)) 0.0)))
