(ns splat-painter.gaussian
  "2D Gaussian splat math — a faithful port of DrawingWithGaussians'
   rendering2d.py (the additive rasterizer) and the scale+rotation covariance
   construction from 2D/3D Gaussian Splatting (Σ = R diag(s²) Rᵀ). Pure Clojure;
   no GL, no I/O. The same arithmetic drives the CPU rasterizer used in tests
   and the GLSL fragment shader used for the live view.

   A splat is a plain map:

     {:mean  [x y]            ; center in pixel coords (x=row, y=col)
      :cov   [c00 c01 c10 c11]; symmetric 2×2 covariance
      :color [r g b]}         ; 0..1 RGB

   The rasterizer is additive: out[pixel] = background[pixel] +
   Σ_splats peak-normalized exp(-½ δᵀ Σ⁻¹ δ) · color. No per-splat opacity and
   no sorting, exactly as in rendering2d.py.")

(def ^:private det-eps 1e-8)   ; rendering2d.py _DET_EPS — floors a near-singular
                               ; covariance so the precision matrix can't overflow.

(defn covariance
  "Σ = R(θ) diag(sx², sy²) R(θ)ᵀ — the 2D analog of 3DGS's R S Sᵀ Rᵀ covariance
   construction (scene/gaussian_model.py). `sx`,`sy` are per-axis scales (stdev),
   `theta` the in-plane rotation in radians. Returns the symmetric 2×2 covariance
   as [c00 c01 c10 c11]."
  [sx sy theta]
  (let [sx2 (* sx sx) sy2 (* sy sy)
        c (Math/cos theta) s (Math/sin theta)]
    [(+ (* sx2 c c) (* sy2 s s))     ; c00
     (* (- sx2 sy2) c s)             ; c01 = c10
     (* (- sx2 sy2) c s)             ; c10
     (+ (* sx2 s s) (* sy2 c c))]))  ; c11

(defn precision
  "Closed-form inverse entries of a symmetric 2×2 covariance [c00 c01 c10 c11],
   as used in δᵀ P δ (rendering2d.py lines 85-91). Returns [p00 p11 cross] where
   δᵀPδ = p00·dx² + cross·dx·dy + p11·dy². The determinant is floored at
   det-eps so a near-singular covariance can't blow up the precision matrix."
  [cov]
  (let [[c00 c01 c10 c11] cov
        det (max (- (* c00 c11) (* c01 c10)) det-eps)]
    [(/ c11 det) (/ c00 det) (/ (- (+ c10 c01)) det)]))

(defn- splat-intensities
  "Peak-normalized exp(-pdf) of one splat over the H×W pixel grid, flat and
   row-major (length H*W). x indexes rows, y columns — matching the reference's
   xg=arange(H)[:,None], yg=arange(W)[None,:]. The peak-normalization subtracts
   the per-splat max so the brightest pixel lands at intensity 1.0
   (jax_stable_exp in the original)."
  [mean cov H W]
  (let [[mx my] mean
        [p00 p11 cross] (precision cov)
        z (for [x (range H) y (range W)
                :let [dx (- x mx) dy (- y my)
                      pdf (* 0.5 (+ (* p00 dx dx) (* cross dx dy) (* p11 dy dy)))]]
            (- pdf))                 ; z = -pdf
        zmax (reduce max z)]
    (into [] (for [zi z] (Math/exp (- zi zmax))))))

(defn rasterize
  "Additive 2D-gaussian rasterizer — port of rendering2d.py. `splats` is a seq of
   {:mean [x y] :cov [..] :color [r g b]}; `background` is a flat H*W*3 vector;
   returns a flat H*W*3 vector of floats. For each pixel, starts from the
   background and adds every splat's peak-normalized contribution times its
   color — the additive form background + Σ intensity·color."
  [splats background H W]
  (let [pre (for [{:keys [mean cov color]} splats]
              {:ints (splat-intensities mean cov H W) :color color})
        pixel (fn [i]
                (let [base (* i 3)
                      acc [(double (nth background base))
                           (double (nth background (+ base 1)))
                           (double (nth background (+ base 2)))]]
                   (reduce (fn [[r g b] {:keys [ints color]}]
                             (let [a (nth ints i)]
                               [(+ r (* a (nth color 0)))
                                (+ g (* a (nth color 1)))
                                (+ b (* a (nth color 2)))]))
                           acc pre)))]
    (into [] (mapcat identity (for [i (range (* H W))] (pixel i))))))

(defn composite
  "Front-to-back over-alpha compositing — the 2DGS/3DGS render equation
   (diff_surfel_rasterization, the Art repo's rasterizer):
     C = Σᵢ cᵢ·αᵢ·Tᵢ,   Tᵢ = Πⱼ<ᵢ (1−αⱼ),   final = C + T·background
   with αᵢ = peak-normalized intensity · `opacity`. Same inputs as `rasterize`
   (flat H*W*3 background, returns flat H*W*3). Unlike the additive rasterizer,
   each splat occludes the ones behind it, so thousands of overlapping splats
   reconstruct the image instead of summing past white. `opacity` ∈ (0,1] is the
   per-splat alpha gain (1.0 = fully opaque at the peak)."
  [splats background H W opacity]
  (let [pre (for [{:keys [mean cov color]} splats]
              {:ints (splat-intensities mean cov H W) :color color})
        pixel (fn [i]
                (let [base (* i 3)
                      b0 (double (nth background base))
                      b1 (double (nth background (+ base 1)))
                      b2 (double (nth background (+ base 2)))]
                  (loop [s pre T 1.0 cr 0.0 cg 0.0 cb 0.0]
                    (if (or (not s) (<= T 1e-4))
                      [(+ cr (* T b0)) (+ cg (* T b1)) (+ cb (* T b2))]
                      (let [{:keys [ints color]} (first s)
                            a (* (nth ints i) opacity)
                            wa (* T a)]
                        (recur (next s) (* T (- 1.0 a))
                               (+ cr (* wa (double (nth color 0))))
                               (+ cg (* wa (double (nth color 1))))
                               (+ cb (* wa (double (nth color 2))))))))))]
    (into [] (mapcat identity (for [i (range (* H W))] (pixel i))))))
