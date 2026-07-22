(ns splat-painter.grid
  "Edge-aware grid: a regular grid of corners is nudged so each cell boundary follows the
   image's edges (a port of pixel-mosaic's optimizeGridCorners). Deforms once per image; the
   resulting cells conform to structure so splats placed in them align with contours.

   Corner arrays are ^doubles cx, cy of length (rows+1)*(cols+1), row-major:
   corner (r,c) at index r*(cols+1)+c. An `edge-fn` is a function (edge-fn x y) -> strength
   in [0,1] (0 flat, 1 strong edge), in the project's (x=row, y=col) convention.")

(def ^:private edge-hit-thresh 0.15)

;; --- corner access -----------------------------------------------------------

(defn- get-x [cx r c cols1] (aget cx (+ (* r cols1) c)))
(defn- get-y [cy r c cols1] (aget cy (+ (* r cols1) c)))
(defn- set-xy! [cx cy r c cols1 x y]
  (let [idx (+ (* r cols1) c)]
    (aset cx idx x)
    (aset cy idx y)))

;; --- edge scoring ------------------------------------------------------------

(defn- edge-density
  "Average edge strength in a (2r+1)^2 window around (px,py). Clamps to image bounds."
  [edge-fn H W px py radius]
  (let [r  (int radius)
        x0 (max 0 (- (int px) r))
        x1 (min (dec H) (+ (int px) r))
        y0 (max 0 (- (int py) r))
        y1 (min (dec W) (+ (int py) r))
        n  (* (inc (- x1 x0)) (inc (- y1 y0)))]
    (if (zero? n)
      0.0
      (let [total (loop [x x0 total 0.0]
                    (if (> x x1)
                      total
                      (recur (inc x)
                        (+ total
                           (loop [y y0 s 0.0]
                             (if (> y y1)
                               s
                               (recur (inc y) (+ s (edge-fn x y)))))))))]
        (/ total n)))))

(defn- edge-alignment
  "How well the straight segment (x1,y1)->(x2,y2) follows edges. Returns 0..1."
  [edge-fn H W x1 y1 x2 y2]
  (let [dx  (- (double x2) (double x1))
        dy  (- (double y2) (double y1))
        len (Math/sqrt (+ (* dx dx) (* dy dy)))]
    (if (< len 1.0)
      0.0
      (let [num-samples (max 3 (long (Math/ceil (* len 0.5))))
            result (loop [i 0 hits 0 max-run 0 run 0]
                     (if (> i num-samples)
                       [hits max-run]
                       (let [t   (/ (double i) num-samples)
                             px  (max 0 (min (dec H) (long (+ x1 (* dx t)))))
                             py  (max 0 (min (dec W) (long (+ y1 (* dy t)))))
                             s   (edge-fn px py)
                             on? (> s edge-hit-thresh)
                             nr  (if on? (inc run) 0)]
                         (recur (inc i)
                                (if on? (inc hits) hits)
                                (max max-run nr)
                                nr))))
            hits  (first result)
            max-r (second result)]
        (+ (* 0.4 (/ (double hits) (inc num-samples)))
           (* 0.4 (/ (double max-r) (inc num-samples)))
           (if (> (edge-fn (int x1) (int y1)) edge-hit-thresh) 0.2 0.0))))))

;; --- corner scoring ----------------------------------------------------------

(defn- corner-score
  "Score for placing a corner at (px,py) given its connected neighbours."
  [edge-fn H W px py neighbours cx cy cols1]
  (+ (edge-density edge-fn H W px py 2.0)
     (reduce + (for [[r c] neighbours]
                 (edge-alignment edge-fn H W px py
                                 (get-x cx r c cols1)
                                 (get-y cy r c cols1))))))

;; --- optimize ----------------------------------------------------------------

(defn optimize
  "Build an (H*W) grid of ~`target-cells` cells and optimize interior corners so cell
   boundaries snap to edges. Returns {:cx ^doubles :cy ^doubles :cols :rows}.
   Options: :iterations (2), :step (base-cell/4), :damping (0.6)."
  [H W target-cells edge-fn & [opts]]
  (let [iterations (long (or (:iterations opts) 2))
        damping    (double (or (:damping opts) 0.6))
        ar         (/ (double W) (double H))
        cols       (max 1 (long (Math/round (Math/sqrt (* target-cells ar)))))
        rows       (max 1 (long (Math/round (/ target-cells (double cols)))))
        cols1      (inc cols)
        n-corners  (* (inc rows) cols1)
        hh         (/ (double H) rows)
        ww         (/ (double W) cols)
        step       (double (or (:step opts) (max 1.0 (/ (min hh ww) 4.0))))
        cx         (double-array n-corners)
        cy         (double-array n-corners)
        offsets    (vec (for [dr [-1 0 1] dc [-1 0 1]
                              :when (not (and (zero? dr) (zero? dc)))]
                          [(* step (double dr)) (* step (double dc))]))]
    ;; initial regular grid
    (dotimes [r (inc rows)]
      (dotimes [c (inc cols)]
        (set-xy! cx cy r c cols1 (* (double r) hh) (* (double c) ww))))
    ;; optimize iterations (Jacobi: snapshot, then move all)
    (dotimes [_iter iterations]
      (let [snap-x (double-array n-corners)
            snap-y (double-array n-corners)]
        (dotimes [i n-corners]
          (aset snap-x i (aget cx i))
          (aset snap-y i (aget cy i)))
        (doseq [r (range 1 rows) c (range 1 cols)]
          (let [idx        (+ (* r cols1) c)
                cur-x      (aget snap-x idx)
                cur-y      (aget snap-y idx)
                neighbours [[(dec r) c] [(inc r) c] [r (dec c)] [r (inc c)]]
                cur-score  (corner-score edge-fn H W cur-x cur-y neighbours
                                         snap-x snap-y cols1)
                best       (reduce
                             (fn [[bx by bs] [dx dy]]
                               (let [nx (+ cur-x dx) ny (+ cur-y dy)]
                                 (if (or (<= nx 0.0) (>= nx (dec H))
                                         (<= ny 0.0) (>= ny (dec W)))
                                   [bx by bs]
                                   (let [ns (corner-score edge-fn H W nx ny neighbours
                                                          snap-x snap-y cols1)]
                                     (if (> ns bs) [nx ny ns] [bx by bs])))))
                             [cur-x cur-y cur-score]
                             offsets)
                bx (first best)
                by (second best)
                nx (+ cur-x (* damping (- bx cur-x)))
                ny (+ cur-y (* damping (- by cur-y)))]
            (aset cx idx nx)
            (aset cy idx ny)))))
    {:cx cx :cy cy :cols cols :rows rows}))

;; --- cell access -------------------------------------------------------------

(defn cell-quad
  "The 4 corner [x y] points of cell (r,c): [TL TR BR BL]."
  [grid r c]
  (let [cols1 (inc (:cols grid)) cx (:cx grid) cy (:cy grid)]
    [[(get-x cx r c cols1) (get-y cy r c cols1)]
     [(get-x cx r (inc c) cols1) (get-y cy r (inc c) cols1)]
     [(get-x cx (inc r) (inc c) cols1) (get-y cy (inc r) (inc c) cols1)]
     [(get-x cx (inc r) c cols1) (get-y cy (inc r) c cols1)]]))

(defn cell-centroid
  "Mean [x y] of the 4 cell corners."
  [grid r c]
  (let [quad (cell-quad grid r c)]
    [(/ (reduce + (map first quad)) 4.0)
     (/ (reduce + (map second quad)) 4.0)]))

(defn cell-size
  "Mean edge length of the cell perimeter (for splat stdev scaling)."
  [grid r c]
  (let [[tl tr br bl] (cell-quad grid r c)
        dist (fn [[x1 y1] [x2 y2]]
               (Math/sqrt (+ (* (- x2 x1) (- x2 x1))
                             (* (- y2 y1) (- y2 y1)))))]
    (/ (+ (dist tl tr) (dist tr br) (dist br bl) (dist bl tl)) 4.0)))
