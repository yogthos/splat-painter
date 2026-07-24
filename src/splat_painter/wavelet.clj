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
   (deep black background, sky) from being noise-amplified past the thresholds.

   ALSO returns :sharp — the same normalize+fuse pipeline over ONLY the finest two
   Haar bands with minimal smoothing. The finest placement levels threshold against
   it (and derive their D from it): the aggregate map is deliberately smoothed for
   gradual stroke-density transitions, which blurs away exactly the text-scale
   structure the smallest strokes exist to preserve. Scale-matched maps = the
   coarse-to-fine pass paints, at each scale, what exists AT that scale."
  ([image sfield] (placement-map image sfield 768 4))
  ([image sfield max-side levels]
   (let [H (:height image) W (:width image)
         scale (min 1.0 (/ (double max-side) (double (max H W))))
         sh (max 1 (long (Math/round (* H scale))))
         sw (max 1 (long (Math/round (* W scale))))
         n       (* sh sw)
         ;; gamma-corrected luma at placement-map resolution
         ^doubles lum (structure/luma-of image sh sw)
         _       (dotimes [i n] (aset lum i (Math/pow (aget lum i) 0.4545)))
         ;; Haar multi-scale detail energy (same pipeline as detail-map); snapshot the
         ;; FINE-BAND energy (levels 1-2) before the coarse bands are added.
         acc     (double-array n)
         ;; band snapshots: after band 1 (finest only), band 2, band 3 — the FINE map
         ;; uses bands 1-2, the MID map bands 2-3 (face features — eyelids, lips,
         ;; nostrils — are mid-frequency: they score near-zero on the finest bands
         ;; and drown in the smoothed aggregate).
         [acc-1 acc-fine acc-3]
         (loop [lev 1 ch sh cw sw ^doubles src lum block 2 s1 nil s2 nil s3 nil]
           (if (and (<= lev levels) (>= ch 2) (>= cw 2))
             (let [hc2 (quot ch 2) wc2 (quot cw 2)
                   ll  (double-array (* hc2 wc2))]
               (haar-detail-energy! src ch cw ll acc sh sw block)
               (recur (inc lev) hc2 wc2 ll (* 2 block)
                      (if (= lev 1) (aclone acc) s1)
                      (if (= lev 2) (aclone acc) s2)
                      (if (= lev 3) (aclone acc) s3)))
             [(or s1 (aclone acc)) (or s2 (aclone acc)) (or s3 (aclone acc))]))
         acc-mid (let [m (double-array n)]
                   (dotimes [i n] (aset m i (- (aget ^doubles acc-3 i) (aget ^doubles acc-1 i))))
                   m)
         ;; E(i) = edge strength sqrt(grad2/gmax) from structure tensor, resampled
         ;; nearest-neighbour from the tensor grid to the placement-map grid
         sf-h    (:h sfield) sf-w (:w sfield)
         ^doubles sf-grad2 (:grad2 sfield)
         sf-gmax (max (double (:gmax sfield)) 1e-12)
         sf-srch (double (or (:src-h sfield) sf-h))
         sf-srcw (double (or (:src-w sfield) sf-w))
         E-arr   (double-array n)
         _ (dotimes [ri sh]
             (dotimes [ci sw]
               (let [x     (* ri (/ (double H) sh))
                     y     (* ci (/ (double W) sw))
                     sfi   (min (dec sf-h) (max 0 (long (Math/round (* x (/ sf-h sf-srch))))))
                     sfj   (min (dec sf-w) (max 0 (long (Math/round (* y (/ sf-w sf-srcw))))))
                     sfidx (+ (* sfi sf-w) sfj)
                     g2    (aget sf-grad2 sfidx)]
                 (aset E-arr (+ (* ri sw) ci)
                       (Math/sqrt (/ (max 0.0 g2) sf-gmax))))))
         ;; the E used for FUSION is locally normalized (like the luma-relative
         ;; detail): an eyelid crease has low ABSOLUTE gradient next to hard
         ;; background edges — local prominence is the signal. Raw E stays in
         ;; :edge for the mid-suppression/shrink logic.
         ^doubles E-fuse
         (let [^doubles mE (structure/box-blur E-arr sh sw (max 2 (quot (min sh sw) 8)))
               meanE (/ (loop [i 0 acc 0.0] (if (< i n) (recur (inc i) (+ acc (aget E-arr i))) acc)) n)
               out (double-array n)]
           (dotimes [i n]
             (let [e (aget E-arr i)
                   ln (min 1.0 (/ (* 0.85 e) (+ (* 2.0 (aget mE i)) (* 0.30 meanE) 1e-12)))]
               (aset out i (max e ln))))
           out)
         ;; normalize+fuse one energy band: global g, local-relative l, edge E — the
         ;; blur/smooth radii set how gradual the resulting map is.
         ;; e-sq? squares the edge term: E has the tensor blur's width, so fusing it
         ;; raw seeds fine strokes in a BAND around each edge — parallel offset 'echo'
         ;; lines in otherwise-flat surroundings. E² concentrates the band to the core.
         fuse (fn [^doubles raw blur-r smooth-r e-sq?]
                (let [^doubles a (structure/box-blur raw sh sw blur-r)
                      dmax    (loop [i 0 m 0.0] (if (< i n) (recur (inc i) (max m (aget a i))) m))
                      wide-r  (max 2 (quot (min sh sw) 8))
                      ^doubles m-a (structure/box-blur a sh sw wide-r)
                      mean-a  (/ (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (aget a i))) s)) n)
                      P (double-array n)]
                  (dotimes [i n]
                    (let [g (if (pos? dmax) (/ (aget a i) dmax) 0.0)
                          l (min 1.0 (/ (aget a i) (+ (* 2.0 (aget m-a i)) (* 0.30 mean-a) 1e-12)))
                          E0 (aget E-fuse i)
                          E (if e-sq? (* E0 E0) E0)
                          v (min 1.0 (max g l (* 0.85 E)))]
                      (aset P i v)))
                  (structure/box-blur P sh sw smooth-r)))
         ;; aggregate: smoothed for gradual density transitions (radius /40 blur, /50 final)
         P  (fuse acc (max 1 (quot (min sh sw) 40)) (max 1 (quot (min sh sw) 50)) false)
         ;; sharp: fine bands only, minimal smoothing, E² — text/eye-scale structure
         ;; survives and fine strokes hug the edge cores
         Ps (fuse acc-fine 1 1 true)
         ;; mid: bands 2-3, light smoothing, plain E — the map the mid levels place by
         Pm (fuse acc-mid (max 1 (quot (min sh sw) 60)) 1 false)
         ;; ABSOLUTE subjectness: is there real structure here at all? The fused
         ;; placement maps normalize LOCALLY so dark low-contrast texture still
         ;; receives strokes — but that same normalization lights smooth bokeh up
         ;; to full 'detail', which left the broad tier's bokeh-adaptive machinery
         ;; (Broad growth / thinning / melt) inert on soft backgrounds. This map
         ;; answers from RAW globally-scaled signals only: the fine-band Haar
         ;; energy and the tensor edge strength. Bokeh ≈ sensor noise scores ~0;
         ;; any actual texture or contour scores high.
         subj (let [^doubles af acc-fine
                    out (double-array n)]
                (dotimes [i n]
                  (aset out i (min 1.0 (max (/ (aget af i) 0.35)
                                            (/ (aget E-arr i) 0.30)))))
                (structure/box-blur out sh sw (max 2 (quot (min sh sw) 24))))]
     {:h sh :w sw :detail P :sharp Ps :mid Pm :edge E-arr :subject subj
      :dmax 1.0 :src-h H :src-w W})))

(defn subject-abs-at
  "ABSOLUTE subjectness ∈ [0,1] at full-image coords — 0 in bokeh/flat regions,
   high wherever real fine structure or edges live (see placement-map :subject).
   Drives the broad tier's bokeh adaptation; the locally-normalized maps keep
   driving fine-stroke placement. Falls back to :detail for maps without it."
  [dmap x y]
  (let [H (:h dmap) W (:w dmap)
        src-h (long (or (:src-h dmap) H)) src-w (long (or (:src-w dmap) W))
        ^doubles d (or (:subject dmap) (:detail dmap))
        xi (min (dec H) (max 0 (long (Math/round (* (double x) (/ (double H) src-h))))))
        yi (min (dec W) (max 0 (long (Math/round (* (double y) (/ (double W) src-w))))))]
    (min 1.0 (aget d (+ (* xi W) yi)))))

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

(defn edge-at
  "Raw structure-tensor edge strength ∈ [0,1] at full-image coords, from the map's
   :edge channel (sqrt(grad2/gmax), no normalization) — the band where broad fill
   strokes must not tread."
  [dmap x y]
  (if-let [e (:edge dmap)]
    (let [H (:h dmap) W (:w dmap)
          src-h (long (or (:src-h dmap) H)) src-w (long (or (:src-w dmap) W))
          ^doubles d e
          xi (min (dec H) (max 0 (long (Math/round (* (double x) (/ (double H) src-h))))))
          yi (min (dec W) (max 0 (long (Math/round (* (double y) (/ (double W) src-w))))))]
      (aget d (+ (* xi W) yi)))
    0.0))

(defn mid-at
  "The mid placement levels' map: the UNION (max) of the :mid band map (Haar bands
   2-3 — face-feature-scale structure) and the :sharp fine-band map, so mid strokes
   serve smooth features AND fine texture (dark fur, small text) alike. Falls back
   to :detail when the maps are absent."
  [dmap x y]
  (if-let [m (:mid dmap)]
    (max (detail-at (assoc dmap :detail m) x y) (sharp-at dmap x y))
    (detail-at dmap x y)))

(defn sharp-at
  "Like detail-at but over the :sharp fine-band map — what the finest placement
   levels threshold against. Falls back to :detail when the map has no :sharp."
  [dmap x y]
  (if-let [s (:sharp dmap)]
    (detail-at (assoc dmap :detail s) x y)
    (detail-at dmap x y)))
