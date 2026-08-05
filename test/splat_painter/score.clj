(ns splat-painter.score
  "Perceptual render scoring — the harness dev/harness/score.py never grew into.

   score.py reports mean|d| and RMS over BT.709 luma and nothing else. Two blind
   spots that matter here, because this codebase is tuned by measurement (the
   seed.clj constants justify their values with numbers from this instrument):

   - Chroma error scores zero. A stroke laying the wrong hue at the right
     lightness is invisible to a luma-only metric, yet the structure tensor
     deliberately makes chroma edges count like luma edges (the Di Zenzo comment
     in gpu_fields.clj) — the painter detects chroma structure it never scores
     itself on.
   - Per-pixel L1 cannot separate preserved detail from detail averaged away.
     Smearing fine structure is the characteristic way a placement change goes
     wrong, and mean|d| barely moves when it does.

   This port keeps the score.py columns (same registration, same ROI, same
   definitions — the correctness proof is that the luma numbers match on the
   same pair) and adds Oklab lightness error, Oklab chroma error and luma SSIM
   for the ROI and the whole image. ΔL and chroma are separate columns
   deliberately: averaging a hue miss and a brightness miss into one number
   recreates the problem being fixed.

   Run: jolt -M:score <source> <label> <render> [<label> <render> ...]"
  (:require [splat-painter.image :as image]))

;; Registration and ROI — measured knowledge from score.py, ported verbatim.
;; The render sits 1px above a Lanczos downscale of the source (measured by
;; minimising mean|d|): render row y-1 == source row y. Without compensating,
;; every edge contributes a dipole that is comparison error, not painter error.
;; The nose-ROI crop is what the whole-image split exists to catch a local win
;; bought with a global loss against.
(def ^:private roi-x0 520)
(def ^:private roi-y0 405)
(def ^:private roi-x1 610)
(def ^:private roi-y1 455)
(def ^:private roi-w 70)                ; score.py: d_full[Y0-1:Y1-1, X0:X1][:, :70]
(def ^:private dy -1)

;; --- luma (BT.709, ported from score.py) ---------------------------------

(defn luma-plane
  "BT.709 luma of a flat H*W*3 0..1 buffer, in the 0..255 scale score.py works
   in (it loads via PIL as 0..255, so its mean|d|/RMS are pixel-value units).
   A port that computed in 0..1 and claimed parity would be comparing numbers
   255x apart."
  [^doubles px W H]
  (let [n (* W H) ^doubles out (double-array n)]
    (dotimes [i n]
      (let [j (* 3 i)]
        (aset out i (* 255.0 (+ (* 0.2126 (aget px j))
                                (* 0.7152 (aget px (inc j)))
                                (* 0.0722 (aget px (+ j 2))))))))
    out))

(defn- aligned-diff
  "d[y][x] = render luma row y minus source luma row y+1 — an (H-1)*W plane.
   Row i of the plane pairs render row i with source row i+1, exactly score.py's
   d_full = r[:-1, :] - s[1:, :]."
  [^doubles rl ^doubles sl W H]
  (let [dh (dec H) ^doubles d (double-array (* dh W))]
    (dotimes [y dh]
      (let [r0 (* y W) s0 (* (inc y) W)]
        (dotimes [x W]
          (aset d (+ r0 x) (- (aget rl (+ r0 x)) (aget sl (+ s0 x)))))))
    d))

;; --- Oklab ----------------------------------------------------------------
;; Ottosson 2020. M1 takes linear sRGB to LMS cone responses; the cube root of
;; LMS is taken sign-preservingly (a naive x^(1/3) NaNs on negative LMS
;; intermediates — Math/cbrt returns the real root); M2 takes cbrt(LMS) to
;; [L a b]. Stored flattened (row i at [3i,3i+2]); M1 row sums are 1 (gray stays
;; on the diagonal), M2 row sums are 0 (gray maps to a=b=0) — the white test in
;; score-test pins both via srgb->oklab.
(def ^:private m1 ^doubles (double-array [0.4122214708 0.5363325363 0.0514459929
                                          0.2119034982 0.6806995451 0.1073969566
                                          0.0883024619 0.2817188376 0.6299787005]))
(def ^:private m2 ^doubles (double-array [0.2104542553 0.7936177850 -0.0040720468
                                          1.9779984951 -2.4285922050 0.4505937099
                                          0.0259040371 0.7827717662 -0.8086757660]))

(defn cbrt
  "Sign-preserving cube root: sign(x)·|x|^(1/3). Math/cbrt returns the real root
   for negative input — the Oklab trap is using pow(x, 1/3), which NaNs."
  [x]
  (Math/cbrt (double x)))

(defn- srgb->linear [c]
  (if (<= c 0.04045)
    (/ c 12.92)
    (Math/pow (/ (+ c 0.055) 1.055) 2.4)))

(defn srgb->oklab
  "sRGB 0..1 triple -> Oklab [L a b]."
  [r g b]
  (let [lr (srgb->linear r) lg (srgb->linear g) lb (srgb->linear b)
        l  (+ (* (aget m1 0) lr) (* (aget m1 1) lg) (* (aget m1 2) lb))
        m  (+ (* (aget m1 3) lr) (* (aget m1 4) lg) (* (aget m1 5) lb))
        s  (+ (* (aget m1 6) lr) (* (aget m1 7) lg) (* (aget m1 8) lb))
        cl (cbrt l) cm (cbrt m) cs (cbrt s)]
    [(+ (* (aget m2 0) cl) (* (aget m2 1) cm) (* (aget m2 2) cs))
     (+ (* (aget m2 3) cl) (* (aget m2 4) cm) (* (aget m2 5) cs))
     (+ (* (aget m2 6) cl) (* (aget m2 7) cm) (* (aget m2 8) cs))]))

(defn oklab-planes
  "Flat H*W*3 0..1 buffer -> three flat H*W ^doubles planes [L a b]. Converting
   once and summing regions afterwards beats re-converting per region: the nose
   ROI sits inside the full image, so the full pass already covers every pixel
   the ROI would ask for again."
  [^doubles px W H]
  (let [n (* W H)
        ^doubles pl (double-array n) ^doubles pa (double-array n) ^doubles pb (double-array n)]
    (dotimes [i n]
      (let [j  (* 3 i)
            r  (aget px j) g (aget px (inc j)) b (aget px (+ j 2))
            lr (srgb->linear r) lg (srgb->linear g) lb (srgb->linear b)
            l  (+ (* (aget m1 0) lr) (* (aget m1 1) lg) (* (aget m1 2) lb))
            m  (+ (* (aget m1 3) lr) (* (aget m1 4) lg) (* (aget m1 5) lb))
            s  (+ (* (aget m1 6) lr) (* (aget m1 7) lg) (* (aget m1 8) lb))
            cl (cbrt l) cm (cbrt m) cs (cbrt s)]
        (aset pl i (+ (* (aget m2 0) cl) (* (aget m2 1) cm) (* (aget m2 2) cs)))
        (aset pa i (+ (* (aget m2 3) cl) (* (aget m2 4) cm) (* (aget m2 5) cs)))
        (aset pb i (+ (* (aget m2 6) cl) (* (aget m2 7) cm) (* (aget m2 8) cs)))))
    [pl pa pb]))

(defn- lab-err-region
  "Mean |ΔL| and mean chroma distance over aligned region [r0,r1)×[c0,c1) of the
   precomputed Oklab planes: render pixel (y,x) paired with source (y+1,x), same
   pairing as the luma difference. Chroma is euclidean distance in the (a,b)
   plane (sqrt, not Math/hypot — hypot is per-pixel slow here)."
  [^doubles rl ^doubles ra ^doubles rb ^doubles sl ^doubles sa ^doubles sb W r0 r1 c0 c1]
  (let [n (* (- r1 r0) (- c1 c0))]
    ;; the row loop RETURNS its partials and the outer loop adds them: `recur`
    ;; binds to the nearest enclosing `loop`, so an inner (recur (inc y) ...)
    ;; rebinds x rather than advancing y, and never terminates.
    (loop [y r0 dl 0.0 ch 0.0]
      (if (< y r1)
        (let [[rdl rch]
              (loop [x c0 d 0.0 c 0.0]
                (if (< x c1)
                  (let [i   (+ (* y W) x) si (+ (* (inc y) W) x)
                        dll (Math/abs (- (aget rl i) (aget sl si)))
                        da  (- (aget ra i) (aget sa si))
                        db  (- (aget rb i) (aget sb si))]
                    (recur (inc x) (+ d dll)
                           (+ c (Math/sqrt (+ (* da da) (* db db))))))
                  [d c]))]
          (recur (inc y) (+ dl (double rdl)) (+ ch (double rch))))
        [(/ dl (double n)) (/ ch (double n))]))))

(defn- region-stats
  "Luma error stats over region [r0,r1)×[c0,c1) of an (H-1)*W difference plane:
   mean|d|, RMS, count of pixels darker than -20 and the worst (most negative)
   pixel — the score.py ROI/full columns."
  [^doubles d W r0 r1 c0 c1]
  (let [n (* (- r1 r0) (- c1 c0))]
    ;; same shape as lab-err-region: the row loop returns its partials rather than
    ;; recurring across loops, which would rebind x to (inc y) and spin forever.
    (loop [y r0 sad 0.0 ssq 0.0 dark 0 worst 1e300]
      (if (< y r1)
        (let [[rsad rssq rdark rworst]
              (loop [x c0 s 0.0 q 0.0 dk 0 w 1e300]
                (if (< x c1)
                  (let [v  (aget d (+ (* y W) x))
                        av (Math/abs v)]
                    (recur (inc x) (+ s av) (+ q (* v v))
                           (if (< v -20.0) (inc dk) dk)
                           (if (< v w) v w)))   ; min would box here
                  [s q dk w]))]
          (recur (inc y) (+ sad (double rsad)) (+ ssq (double rssq))
                 (+ dark (long rdark))
                 (if (< (double rworst) worst) (double rworst) worst)))
        {:mean (/ sad (double n))
         :rms  (Math/sqrt (/ ssq (double n)))
         :dark dark
         :worst worst}))))

;; --- SSIM over luma -------------------------------------------------------
;; Standard Wang et al. SSIM with a separable Gaussian window (11 taps,
;; σ=1.5 — the paper's default), C1=(0.01·L)², C2=(0.03·L)² with L=255 matching
;; the 0..255 luma scale. Local means, variances and covariance come from
;; Gaussian-weighted sums: μ=g⊗x, σx²=g⊗(x²)−μx², σxy=g⊗(xy)−μx·μy. The mean is
;; taken over the valid interior where the window fully fits, so border effects
;; from the clamped blur never enter the number.

(defn gaussian-kernel
  "1D Gaussian taps with radius r=ceil(3σ), normalized to sum 1."
  [sigma]
  (let [r (long (Math/ceil (* 3.0 sigma)))
        taps (double-array (mapv (fn [k] (Math/exp (/ (* (- k) k) (* 2.0 sigma sigma))))
                                 (range (- r) (inc r))))
        s (reduce + 0.0 (seq taps))]
    (dotimes [i (alength taps)] (aset taps i (/ (aget taps i) s)))
    {:r r :taps taps}))

(defn blur-plane
  "Separable Gaussian blur of a flat H*W plane. Border pixels are index-clamped
   (cheap, and SSIM only averages the valid interior so the clamped band is
   excluded from the number)."
  [^doubles plane W H sigma]
  (let [{:keys [r taps]} (gaussian-kernel sigma)
        ^doubles taps taps
        n  (alength taps)
        w  (long W) h (long H) rr (long r)
        wm (dec w) hm (dec h)
        ^doubles tmp (double-array (* w h))
        ^doubles out (double-array (* w h))]
    ;; horizontal pass. The index clamp must be inline ifs, not max/min: generic
    ;; max/min/dec per tap was what made the first port ~3 orders of magnitude
    ;; too slow (the aget then ran with boxed index arithmetic).
    (dotimes [y h]
      (let [row (* y w)]
        (dotimes [x w]
          (loop [k 0 acc 0.0]
            (if (< k n)
              (let [ci (- (+ x rr) k)
                    ci (if (< ci 0) 0 (if (> ci wm) wm ci))]
                (recur (inc k) (+ acc (* (aget taps k) (aget plane (+ row ci))))))
              (aset tmp (+ row x) acc))))))
    ;; vertical pass
    (dotimes [y h]
      (let [row (* y w)]
        (dotimes [x w]
          (loop [k 0 acc 0.0]
            (if (< k n)
              (let [ci (- (+ y rr) k)
                    ci (if (< ci 0) 0 (if (> ci hm) hm ci))]
                (recur (inc k) (+ acc (* (aget taps k) (aget tmp (+ (* ci w) x))))))
              (aset out (+ row x) acc))))))
    out))

(defn- crop-plane
  "Copy rows [r0,r1)×cols [c0,c1) of a flat W-wide plane into a fresh flat
   (r1-r0)*(c1-c0) plane."
  [^doubles p W r0 r1 c0 c1]
  (let [ch (- r1 r0) cw (- c1 c0)
        ^doubles out (double-array (* ch cw))]
    (dotimes [y ch]
      (dotimes [x cw]
        (aset out (+ (* y cw) x) (aget p (+ (* (+ r0 y) W) (+ c0 x))))))
    out))

(defn ssim
  "SSIM between two aligned luma planes x,y (same flat H*W dims), averaged over
   the valid interior [r,H-r)×[r,W-r) where the Gaussian window fully fits."
  [^doubles x ^doubles y W H]
  (let [sigma 1.5
        r (long (Math/ceil (* 3.0 sigma)))
        c1 (* 0.01 0.01 255.0 255.0)
        c2 (* 0.03 0.03 255.0 255.0)
        npx (* W H)
        ;; the valid interior is [r,H-r) x [r,W-r) — the band where the window
        ;; fully fits. Each axis takes its OWN limit: one shared bound silently
        ;; scores a square crop of a non-square image.
        y-hi (- H r)
        x-hi (- W r)
        ^doubles mx (blur-plane x W H sigma)
        ^doubles my (blur-plane y W H sigma)
        ^doubles x2 (double-array npx) ^doubles y2 (double-array npx) ^doubles xy (double-array npx)
        _ (dotimes [i npx]
            (let [xi (aget x i) yi (aget y i)]
              (aset x2 i (* xi xi))
              (aset y2 i (* yi yi))
              (aset xy i (* xi yi))))
        ;; the second moments must be LOCAL MEANS too — sigma_x^2 = E[x^2] - mu_x^2
        ;; over the window. Subtracting mu^2 from the raw per-pixel x^2 is not a
        ;; variance at all (it is zero-mean only where the plane is locally flat),
        ;; so these three planes get blurred exactly like mx/my do.
        ^doubles bx2 (blur-plane x2 W H sigma)
        ^doubles by2 (blur-plane y2 W H sigma)
        ^doubles bxy (blur-plane xy W H sigma)
        ^doubles vx (double-array npx) ^doubles vy (double-array npx) ^doubles vxy (double-array npx)
        _ (dotimes [i npx]
            (let [mxi (aget mx i) myi (aget my i)]
              (aset vx i (- (aget bx2 i) (* mxi mxi)))
              (aset vy i (- (aget by2 i) (* myi myi)))
              (aset vxy i (- (aget bxy i) (* mxi myi)))))]
    ;; the row accumulator is threaded through the inner loop's RETURN VALUE, not
    ;; by recurring across loops: `recur` binds to the nearest enclosing `loop`, so
    ;; an inner (recur (inc y) ...) rebinds x instead of advancing y and spins
    ;; forever.
    (loop [y r ssum 0.0 n 0]
      (if (>= y y-hi)
        (if (zero? n) 1.0 (/ ssum (double n)))
        (let [[rsum rn]
              (loop [x r s 0.0 c 0]
                (if (>= x x-hi)
                  [s c]
                  (let [i   (+ (* y W) x)
                        mxi (aget mx i) myi (aget my i)
                        num (* (+ (* 2.0 mxi myi) c1) (+ (* 2.0 (aget vxy i)) c2))
                        den (* (+ (* mxi mxi) (* myi myi) c1) (+ (aget vx i) (aget vy i) c2))]
                    (recur (inc x) (+ s (/ num den)) (inc c)))))]
          (recur (inc y) (+ ssum (double rsum)) (+ n (long rn))))))))

;; --- the whole-image / ROI split ------------------------------------------

(defn- aligned-luma
  "The aligned luma planes for SSIM: render rows 0..H-2 and source rows 1..H-1,
   each a flat (H-1)*W plane. SSIM must compare the same slices the error does,
   or the registration offset shows up as an SSIM dipole."
  [^doubles rl ^doubles sl W H]
  (let [dh (dec H) n (* dh W)
        ^doubles r (double-array n) ^doubles s (double-array n)]
    (dotimes [y dh]
      (let [d0 (* y W) s0 (* (inc y) W)]
        (dotimes [x W]
          (aset r (+ d0 x) (aget rl (+ d0 x)))
          (aset s (+ d0 x) (aget sl (+ s0 x))))))
    [r s]))

(defn- ssim-region
  "SSIM over a rectangular region [r0,r1)×[c0,c1) of the aligned luma planes:
   crop so the Gaussian window never samples outside the region."
  [^doubles ra ^doubles sa W r0 r1 c0 c1]
  (let [^doubles cr (crop-plane ra W r0 r1 c0 c1)
        ^doubles cs (crop-plane sa W r0 r1 c0 c1)]
    (ssim cr cs (- c1 c0) (- r1 r0))))

(defn roi-region
  "The nose-ROI plane-region [r0 r1 c0 c1] for a dh×W difference plane, clamped
   to the image. Returns [region clamped?]. An image smaller than the ROI box
   collapses the requested range to empty; the fallback is the whole plane so a
   small image still scores, with clamped?=true for the caller to say so."
  [dh W]
  (let [r0 (min (- roi-y0 1) dh)
        r1 (min (- roi-y1 1) dh)
        c0 (min roi-x0 W)
        c1 (min (+ roi-x0 roi-w) W)]
    (if (and (< r0 r1) (< c0 c1))
      [[r0 r1 c0 c1] false]
      [[0 dh 0 W] true])))

(defn score-buffers
  "Score `render` against `src` — both flat H*W*3 0..1 double buffers (the shape
   image/load-image produces). Returns the score.py luma columns (roi/full
   mean|d|, ROI dark and worst, full RMS) plus the columns score.py was blind
   to: Oklab lightness error (mean |ΔL|), Oklab chroma error (mean euclidean
   distance in the a,b plane, kept separate so a hue miss can't hide inside a
   brightness number) and SSIM over luma — for the nose ROI and the whole image.
   :roi-clamped is true when the image was smaller than the ROI box and the ROI
   columns therefore cover the whole image.

   Registration is score.py's DY=-1: aligned pixel (y,x) is render row y vs
   source row y+1, for y in [0,H-1)."
  [^doubles src ^doubles render W H]
  (let [dh (dec H)
        ^doubles sl (luma-plane src W H)
        ^doubles rl (luma-plane render W H)
        ^doubles d  (aligned-diff rl sl W H)
        al (aligned-luma rl sl W H)
        op (oklab-planes src W H)
        rp (oklab-planes render W H)
        ^doubles ra (nth al 0) ^doubles sa (nth al 1)
        ^doubles srl (nth op 0) ^doubles sra (nth op 1) ^doubles srb (nth op 2)
        ^doubles rrl (nth rp 0) ^doubles rra (nth rp 1) ^doubles rrb (nth rp 2)
        [rois roi-clamped?] (roi-region dh W)
        full [0 dh 0 W]
        rstat  (fn [[r0 r1 c0 c1]] (region-stats d W r0 r1 c0 c1))
        laberr (fn [[r0 r1 c0 c1]] (lab-err-region rrl rra rrb srl sra srb W r0 r1 c0 c1))
        ssim-at (fn [[r0 r1 c0 c1]] (ssim-region ra sa W r0 r1 c0 c1))
        rf (rstat rois) ff (rstat full)
        rlab (laberr rois) flab (laberr full)]
    {:roi-mean   (:mean rf)   :roi-rms (:rms rf)
     :roi-dark   (:dark rf)   :roi-worst (:worst rf)
     :roi-lab-l  (nth rlab 0) :roi-chroma (nth rlab 1) :roi-ssim (ssim-at rois)
     :full-mean  (:mean ff)   :full-rms (:rms ff)
     :full-lab-l (nth flab 0) :full-chroma (nth flab 1) :full-ssim (ssim-at full)
     :roi-clamped roi-clamped?}))

;; --- CLI ------------------------------------------------------------------

(defn -main [& args]
  (let [[src & pairs] args]
    (if (or (nil? src) (empty? pairs) (odd? (count pairs)))
      (do (println "usage: jolt -M:score <source> <label> <render> [<label> <render> ...]")
          (System/exit 1))
      (let [base-img (image/load-image src)
            W (long (:width base-img)) H (long (:height base-img))]
        (println (format "%s" (str "source " src " (" W "x" H ") — luma in 0..255 scale,"
                                   " ΔL and chroma in Oklab units, SSIM ∈ [0,1]")))
        (let [[_ roi-clamped?] (roi-region (dec H) W)]
          (when roi-clamped?
            (println (str "score: image is " W "x" H ", smaller than the " roi-x1 "x" roi-y1
                          " nose ROI — ROI columns fall back to the whole image"))))
        (println (format "%-24s %11s %9s %10s %9s %8s %11s %9s %12s %9s %8s %11s %9s"
                         "variant" "ROI mean|d|" "ROI dark" "ROI worst" "ROI rms"
                         "ROI ΔL" "ROI chroma" "ROI SSIM"
                         "FULL mean|d|" "FULL rms" "FULL ΔL" "FULL chroma" "FULL SSIM"))
        (loop [base nil pairs pairs]
          (when (seq pairs)
            (let [[lab p & rest] pairs
                  rimg (image/load-image p)]
              (when (or (not= W (:width rimg)) (not= H (:height rimg)))
                (throw (ex-info (str "score: " p " is " (:width rimg) "x" (:height rimg)
                                     " but source is " W "x" H " — pair must be same size")
                                {:path p})))
              (let [s (score-buffers (:pixels base-img) (:pixels rimg) W H)
                    b (or base s)
                    dr (- (:roi-mean s) (:roi-mean b))
                    df (- (:full-mean s) (:full-mean b))]
                ;; jolt's format has no '+' flag (score.py gets one free from
                ;; python's {:+.3f}), so the sign on the delta columns is explicit.
                ;; A negative value already carries its own '-' from %.3f.
                (let [signed (fn [v] (str (if (neg? (double v)) "" "+") (format "%.3f" (double v))))]
                  (println (format "%-24s %11.3f %9d %10.1f %9.3f %8.3f %11.3f %9.4f %12.3f %9.3f %8.3f %11.3f %9.4f   (%s ROI, %s full)"
                                   lab (:roi-mean s) (:roi-dark s) (:roi-worst s) (:roi-rms s)
                                   (:roi-lab-l s) (:roi-chroma s) (:roi-ssim s)
                                   (:full-mean s) (:full-rms s) (:full-lab-l s) (:full-chroma s) (:full-ssim s)
                                   (signed dr) (signed df))))
                (recur (or base s) rest)))))))))
