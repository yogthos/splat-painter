(ns splat-painter.tf-smoke
  "Platform de-risk: prove macOS GL 4.1 can do geometry-shader + transform-feedback
   stream COMPACTION — the mechanism GPU splat generation needs (variable count out,
   no compute/SSBO). Draws 10 GL_POINTS; the geometry shader emits only the even ids;
   transform feedback captures the survivors; a query reports how many were written.
   Expect count 5 and values [0 2 4 6 8]. Called from core/on-realize when
   GA_PAINTER_TF_SMOKE is set (needs a current GL context)."
  (:require [glimmer-gl.gl :as gl]
            [jolt.ffi :as ffi]))

(def ^:private vs-src
  "#version 330 core
flat out int v_id;
void main(){ v_id = gl_VertexID; gl_Position = vec4(0.0,0.0,0.0,1.0); }")

(def ^:private gs-src
  "#version 330 core
layout(points) in;
layout(points, max_vertices = 1) out;
flat in int v_id[];
out float o_val;
void main(){
  if ((v_id[0] & 1) == 0) { o_val = float(v_id[0]); EmitVertex(); EndPrimitive(); }
}")

(defn- drain-errors! [] (loop [i 0] (when (and (< i 32) (not (zero? (gl/gl-get-error)))) (recur (inc i)))))

(defn run! []
  (println "tf-smoke: linking VS+GS transform-feedback program…")
  (drain-errors!)
  (let [prog (gl/make-tf-program vs-src gs-src ["o_val"])]
    (if-not prog
      (println "tf-smoke: FAILED to link TF program (see log above)")
      (let [n     10
            cap   (* n (ffi/sizeof :float))
            vao   (gl/gen-one gl/gl-gen-vertex-arrays)
            buf   (gl/gen-one gl/gl-gen-buffers)
            q     (gl/gen-one gl/gl-gen-queries)]
        (gl/gl-bind-vertex-array vao)
        (gl/gl-bind-buffer gl/GL-TRANSFORM-FEEDBACK-BUFFER buf)
        (gl/gl-buffer-data gl/GL-TRANSFORM-FEEDBACK-BUFFER cap ffi/null gl/GL-DYNAMIC-COPY)
        (gl/gl-bind-buffer-base gl/GL-TRANSFORM-FEEDBACK-BUFFER 0 buf)
        (gl/gl-use-program prog)
        (gl/gl-enable gl/GL-RASTERIZER-DISCARD)
        (gl/gl-begin-query gl/GL-TRANSFORM-FEEDBACK-PRIMITIVES-WRITTEN q)
        (gl/gl-begin-transform-feedback gl/GL-POINTS)
        (gl/gl-draw-arrays gl/GL-POINTS 0 n)
        (gl/gl-end-transform-feedback)
        (gl/gl-end-query gl/GL-TRANSFORM-FEEDBACK-PRIMITIVES-WRITTEN)
        (gl/gl-disable gl/GL-RASTERIZER-DISCARD)
        (gl/gl-finish)
        (let [err   (gl/gl-get-error)
              count (gl/get-query-object-uiv q gl/GL-QUERY-RESULT)
              ptr   (ffi/alloc cap)]
          (gl/gl-get-buffer-sub-data gl/GL-TRANSFORM-FEEDBACK-BUFFER 0
                                     (* (max 1 count) (ffi/sizeof :float)) ptr)
          (let [vals (gl/read-floats ptr (min n count))]
            (ffi/free ptr)
            (println (format "tf-smoke: glError=0x%x  written=%d  vals=%s" err count (pr-str vals)))
            (if (and (zero? err) (= count 5) (= vals [0.0 2.0 4.0 6.0 8.0]))
              (println "tf-smoke: PASS — GS+TF compaction works on this GL")
              (println "tf-smoke: FAIL — unexpected result"))))))))
