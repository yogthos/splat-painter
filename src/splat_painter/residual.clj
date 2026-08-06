(ns splat-painter.residual
  "Where is the painting still wrong?

   A repaint layer takes the previous COMPOSITE as its source image, so nothing
   downstream of the first pass knows anything about the photograph any more. Every
   later layer therefore spends its stroke budget uniformly over a picture whose
   errors are not uniform: a face that never resolved and a background that
   converged two passes ago get the same density of candidates.

   This derives the missing signal — a per-cell weight on the placement maps, built
   from the error between the current composite and the ORIGINAL image — and hands
   it to fields/prepare and gpu-fields/build-fields!, the two places the maps are
   built. Both then feed the same weighted arrays to the budget solve AND to the
   candidate threshold, which is what makes this a REDISTRIBUTION rather than an
   addition: boosting a region raises its share of the map above threshold, the
   budget solve sees a larger surviving fraction and widens the spacing to stay
   under the same splat count, and the strokes come out of the regions that were
   already right.

   Two invariants, both pinned by residual-test:

   - Weights are >= 1, never < 1. Suppressing a converged region would also lower
     its `dv`, and dv drives the per-stroke fidelity D (min(1, Detail*dv*2.2)) and
     the Perlin warp amplitude (1-D) — so a region would be punished with looser,
     warpier strokes for having converged. Boost-only gets the redistribution from
     the budget solve without that side effect.
   - No residual (first pass) or strength 0 leaves the maps bit-identical. The
     off switch has to be exact, not approximate, or every A/B is unreadable.")

(def weight-cap
  "Ceiling on the RELATIVE excess a cell may contribute, before strength scales it.
   Local error ratios are heavy-tailed — a single blown highlight can read 20x the
   frame mean — and without a cap one such cell would take the whole boost and the
   rest of the unresolved region would be indistinguishable from converged."
  2.0)

(def default-strength
  "How hard the residual aims the next layer. 0 == today's uniform behaviour.

   2.0 is a measured peak, not a guess. Swept {0, 0.5, 1, 2, 3, 5} at three
   committed layers on loki / hk / photog, scored with `jolt -M:score`; full-frame
   mean|d| went 10.173/6.369/19.145 at 0 to 9.868/5.749/18.409 at 2.0, and turned
   back UP on all three at 3.0 (9.899/5.995/18.926). rms, Oklab ΔL and SSIM move
   the same way. 5.0 is much worse than doing nothing at all (loki 11.179): past
   the peak the boost is wide enough that the budget solve's compensating spacing
   increase costs more coverage than the aim recovers.

   Not a slider: the placement maps are built once per image load (ensure-gpu!
   rebuilds fields only when the image map's identity changes), so a live dial
   would either do nothing until the next Add Layer or force a full field rebuild
   on every drag. GA_PAINTER_RESIDUAL overrides it for sweeps."
  2.0)

(defn strength
  "The configured residual strength, env override first (see default-strength)."
  []
  (or (some-> (System/getenv "GA_PAINTER_RESIDUAL") Double/parseDouble) default-strength))

(defn- box-blur-1d
  "Separable box blur of a single-channel H*W array, edge-replicate. Local rather
   than structure/box-blur so this namespace stays free of the analysis stack —
   fields/prepare already pulls that in, gpu-fields deliberately does not."
  ^doubles [^doubles src H W r]
  (let [H (long H) W (long W) r (long r)]
    (if (<= r 0)
      src
      (let [tmp (double-array (* H W))
            out (double-array (* H W))
            w   (double (inc (* 2 r)))]
        (dotimes [y H]
          (let [row (* y W)]
            (loop [x 0]
              (when (< x W)
                (let [acc (loop [k (- r) a 0.0]
                            (if (> k r)
                              a
                              (recur (inc k) (+ a (aget src (+ row (min (dec W) (max 0 (+ x k)))))))))]
                  (aset tmp (+ row x) (/ acc w)))
                (recur (inc x))))))
        (dotimes [x W]
          (loop [y 0]
            (when (< y H)
              (let [acc (loop [k (- r) a 0.0]
                          (if (> k r)
                            a
                            (recur (inc k) (+ a (aget tmp (+ (* (min (dec H) (max 0 (+ y k))) W) x))))))]
                (aset out (+ (* y W) x) (/ acc w)))
              (recur (inc y)))))
        out))))

(defn error-field
  "Per-pixel error between `composite` and `original` (both flat H*W*3 arrays of
   0..1 RGB, row-major), as a fresh ^doubles of length H*W.

   Luma-WEIGHTED but per-channel absolute: |dr|*0.299 + |dg|*0.587 + |db|*0.114.
   A plain |Δluma| would let a hue error cancel itself — too much red against too
   little blue reads as a perfect match — and hue error is exactly what a repaint
   layer accumulates, since every pass resamples its colours from the last render."
  ^doubles [^doubles composite ^doubles original H W]
  (let [n   (* (long H) (long W))
        out (double-array n)]
    (dotimes [i n]
      (let [b  (* 3 i)
            dr (Math/abs (- (aget composite b)       (aget original b)))
            dg (Math/abs (- (aget composite (+ b 1)) (aget original (+ b 1))))
            db (Math/abs (- (aget composite (+ b 2)) (aget original (+ b 2))))]
        (aset out i (+ (* 0.299 dr) (* 0.587 dg) (* 0.114 db)))))
    out))

(defn- box-sum
  "Sum of src over the half-open box [y0,y1) x [x0,x1) of a W-wide field."
  ^double [^doubles src W y0 y1 x0 x1]
  (let [W (long W) y1 (long y1) x1 (long x1) x0 (long x0)]
    (loop [y (long y0) acc 0.0]
      (if (>= y y1)
        acc
        (recur (inc y)
               (loop [x x0 a acc]
                 (if (>= x x1)
                   a
                   (recur (inc x) (+ a (aget src (+ (* y W) x)))))))))))

(defn- resample
  "Area-average an H*W field down to sh*sw. Nearest sampling would let the coarse
   grid land between error features and miss them entirely; the placement grid is
   ~3/4 of the image side, so the boxes are small and this is cheap."
  ^doubles [^doubles src H W sh sw]
  (let [H (long H) W (long W) sh (long sh) sw (long sw)]
    (if (and (= H sh) (= W sw))
      (aclone src)
      (let [out (double-array (* sh sw))]
        (dotimes [r sh]
          (dotimes [c sw]
            (let [y0 (long (Math/floor (* (/ (double r) sh) H)))
                  y1 (min H (max (inc y0) (long (Math/floor (* (/ (double (inc r)) sh) H)))))
                  x0 (long (Math/floor (* (/ (double c) sw) W)))
                  x1 (min W (max (inc x0) (long (Math/floor (* (/ (double (inc c)) sw) W)))))
                  n  (max 1 (* (- y1 y0) (- x1 x0)))]
              (aset out (+ (* r sw) c) (/ (box-sum src W y0 y1 x0 x1) (double n))))))
        out))))

(defn weight-map
  "Placement-map weights for a repaint layer: a fresh ^doubles of sh*sw, high where
   `composite` still misses `original` and low where it has converged.

   The error is measured RELATIVE to the frame mean, not absolutely. An absolute
   threshold would be a per-image constant nobody can set: a soft portrait's worst
   region and a high-contrast street scene's best region sit at the same absolute
   error, so only 'worse than this picture's average' means the same thing on both.

   Returns all-ones when the composite matches the original (nothing to aim at) or
   when strength is 0 — the two cases that must leave the maps bit-identical."
  ^doubles [^doubles composite ^doubles original H W sh sw strength]
  (let [sh (long sh) sw (long sw)
        n  (* sh sw)
        s  (double strength)]
    (if (zero? s)
      (double-array n 1.0)
      (let [H (long H) W (long W)
            ;; blur at FULL resolution before the downsample: the area average
            ;; below is a box of ~H/sh pixels, which aliases a one-pixel-wide
            ;; error line into whichever cell happens to contain it. Radius is
            ;; tied to the reduction so this means the same thing at any size.
            e0   (box-blur-1d (error-field composite original H W) H W
                              (max 1 (quot H (* 2 (max 1 sh)))))
            ;; and again on the reduced grid, at the same relative width the
            ;; detail map uses, so the boost is a smooth region rather than a
            ;; speckle field that would dither strokes in and out per candidate
            e    (box-blur-1d (resample e0 H W sh sw) sh sw (max 1 (quot (min sh sw) 48)))
            mean (/ (loop [i 0 a 0.0] (if (< i n) (recur (inc i) (+ a (aget e i))) a)) (double n))
            out  (double-array n)]
        (if (< mean 1e-9)
          (double-array n 1.0)
          (do (dotimes [i n]
                (aset out i (+ 1.0 (* s (min weight-cap
                                             (max 0.0 (- (/ (aget e i) mean) 1.0)))))))
              out))))))

(defn aimed-dmap
  "A copy of `dmap` whose four DENSITY channels are scaled by `w`. Pure; the input
   arrays are untouched, because both field paths hand the same dmap to the budget
   solve AND to the texture upload.

   Scaling, deliberately, after measuring the alternative. A boost raises the map's
   ABSOLUTE values, and dv drives more than admission: D = min(1, Detail*dv*2.2) is
   the per-stroke fidelity, and the Perlin warp is warp*(1-D)*ssz. So a boosted
   region gets strokes that are more faithful and less warped, not just more of
   them. The cost is that it also raises detail-fraction, which the budget solve
   charges for — a repaint layer paints ~17% less at the same Splats.

   A rank transfer (histogram-match the map onto the residual ordering) removes that
   cost exactly: it preserves detail-fraction at EVERY threshold by construction, so
   the solve sizes identical candidate pools, and the shortfall fell from 17.4% to
   3.1%. It also scored WORSE at every strength swept {0.5, 1, 2, 4} — best 10.038
   against scaling's 9.868 on loki at three layers — because conserving the
   distribution means robbing converged regions of fidelity to pay unresolved ones,
   and it cannot raise D anywhere. The budget cost buys more than it costs. See
   splat-painter-0i2.

   :subject is deliberately left alone. It is a GATE (which tier may paint here at
   all — the bokeh-adaptive Broad thinning reads it), not a density; moving it would
   let residual error turn background into subject and pull the broad tier's whole
   growth/melt behaviour off its footprint."
  [dmap ^doubles w]
  (let [aim (fn [^doubles a]
              (when a
                (let [n   (alength a)
                      out (double-array n)]
                  (dotimes [i n] (aset out i (* (aget a i) (aget w (min i (dec (alength w)))))))
                  out)))]
    (cond-> dmap
      (:detail dmap) (assoc :detail (aim (:detail dmap)))
      (:sharp dmap)  (assoc :sharp  (aim (:sharp dmap)))
      (:mid dmap)    (assoc :mid    (aim (:mid dmap)))
      (:edge dmap)   (assoc :edge   (aim (:edge dmap))))))

(defn weights-for
  "The weight array for `img`'s placement grid (sh*sw), or nil when `img` carries no
   :original — the single-pass case, where there is no residual by construction and
   nothing may change. `img` is the layer's SOURCE image (the previous composite);
   :original is the file it is ultimately trying to reproduce."
  ^doubles [img sh sw]
  (when-let [^doubles orig (:original img)]
    (let [^doubles pix (:pixels img)]
      (when (= (alength pix) (alength orig))
        (weight-map pix orig (:height img) (:width img) sh sw (strength))))))
