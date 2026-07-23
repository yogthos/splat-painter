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

(defn placement-map
  "Luma-relative, edge-fused placement map at ≤`max-side` pixels.
   Returns {:h :w :detail ^doubles :dmax :src-h :src-w} with :dmax 1.0
   (pre-normalized, so detail-at and the GPU sampler need no change).

   Three components fused per cell:
   - g: globally-normalized Haar detail energy (today's behavior — bright- region
        values can't regress, acting as a floor)
   - l: local-relative Haar detail — acc(i) divided by its wide local mean +
        a fraction of the global mean, so dark regions with high *relative*
        contrast (dark fur, dark foliage) read high even though their absolute
        Haar energy is small
   - E: structure-tensor edge strength sqrt(grad2/gmax), resampled from the
        sfield grid — puts fine splats ON contours (gravestone text, silhouette
        edges) even where texture energy is low; the tensor's theta/coherence
        then elongate those splats along the edge

   The 0.30*mean(acc) floor in l(i)'s denominator keeps truly flat regions
   (deep black background, sky) from being noise-amplified past the thresholds."
  ([image sfield] (placement-map image sfield 512 4))
  ([image sfield max-side levels]
   (let [H (:height image) W (:width image)
         scale (min 1.0 (/ (double max-side) (double (max H W))))
         sh (max 1 (long (Math/round (* H scale))))
         sw (max 1 (long (Math/round (* W scale))))
         n       (* sh sw)
         ;; gamma-corrected luma at placement-map resolution
         ^doubles lum (structure/luma-of image sh sw)
         _       (dotimes [i n] (aset lum i (Math/pow (aget lum i) 0.4545)))
         ;; Haar multi-scale detail energy (same pipeline as detail-map)
         acc     (double-array n)]
     (loop [lev 1 ch sh cw sw ^doubles src lum block 2]
       (when (and (<= lev levels) (>= ch 2) (>= cw 2))
         (let [hc2 (quot ch 2) wc2 (quot cw 2)
               ll  (double-array (* hc2 wc2))]
           (haar-detail-energy! src ch cw ll acc sh sw block)
           (recur (inc lev) hc2 wc2 ll (* 2 block)))))
     ;; small box-blur to remove Haar block pattern (radius /40 — crisper than /24)
     (let [blur-r  (max 1 (quot (min sh sw) 40))
           ^doubles acc (structure/box-blur acc sh sw blur-r)
           ;; g(i) = global normalization (today's behavior floor)
           dmax    (loop [i 0 m 0.0] (if (< i n) (recur (inc i) (max m (aget acc i))) m))
           ;; m(i) = wide local mean of acc for luma-relative normalization
           wide-r  (max 2 (quot (min sh sw) 8))
           ^doubles m-acc (structure/box-blur acc sh sw wide-r)
           ;; denominator floor for l(i)
           mean-acc (/ (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (aget acc i))) s)) n)
           ;; E(i) = edge strength sqrt(grad2/gmax) from structure tensor, resampled
           ;; nearest-neighbour from the tensor grid to the placement-map grid
           sf-h    (:h sfield) sf-w (:w sfield)
           ^doubles sf-grad2 (:grad2 sfield)
           sf-gmax (max (double (:gmax sfield)) 1e-12)
           sf-srch (double (or (:src-h sfield) sf-h))
           sf-srcw (double (or (:src-w sfield) sf-w))
           E-arr   (double-array n)]
       (dotimes [ri sh]
         (dotimes [ci sw]
           (let [x     (* ri (/ (double H) sh))
                 y     (* ci (/ (double W) sw))
                 sfi   (min (dec sf-h) (max 0 (long (Math/round (* x (/ sf-h sf-srch))))))
                 sfj   (min (dec sf-w) (max 0 (long (Math/round (* y (/ sf-w sf-srcw))))))
                 sfidx (+ (* sfi sf-w) sfj)
                 g2    (aget sf-grad2 sfidx)]
             (aset E-arr (+ (* ri sw) ci)
                   (Math/sqrt (/ (max 0.0 g2) sf-gmax))))))
       ;; P(i) = fused placement
       (let [P (double-array n)]
         (dotimes [i n]
           (let [g (if (pos? dmax) (/ (aget acc i) dmax) 0.0)
                 l (min 1.0 (/ (aget acc i) (+ (* 2.0 (aget m-acc i)) (* 0.30 mean-acc) 1e-12)))
                 E (aget E-arr i)
                 v (min 1.0 (max g l (* 0.85 E)))]
             (aset P i v)))
         ;; final light smoothing so transitions are gradual
         (let [smooth-r (max 1 (quot (min sh sw) 50))
               ^doubles P (structure/box-blur P sh sw smooth-r)]
           {:h sh :w sw :detail P :dmax 1.0 :src-h H :src-w W}))))))

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
