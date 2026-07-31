(in-ns 'jolt.main)

;; The user's reported settings, read off the screenshot's slider positions
;; against the ranges in core.clj (~line 915). Approximate, but both legs of the
;; A/B use the identical values, so the comparison is exact even if the absolute
;; settings are slightly off from the user's.
(doseq [[a v] {splat-painter.core/count-atom     547000
               splat-painter.core/size-atom      7.5
               splat-painter.core/broad-atom     2.31
               splat-painter.core/mid-atom       0.47
               splat-painter.core/fine-atom      0.47
               splat-painter.core/detail-atom    0.56
               splat-painter.core/variation-atom 0.47
               splat-painter.core/curvature-atom 0.47
               splat-painter.core/stroke-atom    2.4
               splat-painter.core/contrast-atom  1.0
               splat-painter.core/hardness-atom  1.63
               splat-painter.core/cutin-atom     0.91
               splat-painter.core/swirl-atom     0.91}]
  (reset! a v))
:controls-set
