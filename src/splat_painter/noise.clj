(ns splat-painter.noise
  "Classic Perlin improved-noise (2D/3D) — a port of perlin-flow's water.cljs
   (fade/lerp/grad/noise3). We use Ken Perlin's reference 256-permutation instead
   of perlin-flow's mulberry32 seeded shuffle, so there's no 32-bit RNG / Math.imul
   to port and the field is fixed and deterministic.

   Used to break the rigid grid look of the splat field: a smooth flow field steers
   brushstroke orientation in flat regions, and independent noise channels vary each
   stroke's size, tone, and position so no two strokes are identical.")

;; Ken Perlin's reference permutation (Improved Noise, 2002), doubled to 512 so
;; the (p[X]+Y) index never needs a wrap.
(def ^:private perm-base
  [151 160 137 91 90 15 131 13 201 95 96 53 194 233 7 225 140 36 103 30 69 142
   8 99 37 240 21 10 23 190 6 148 247 120 234 75 0 26 197 62 94 252 219 203 117
   35 11 32 57 177 33 88 237 149 56 87 174 20 125 136 171 168 68 175 74 165 71
   134 139 48 27 166 77 146 158 231 83 111 229 122 60 211 133 230 220 105 92 41
   55 46 245 40 244 102 143 54 65 25 63 161 1 216 80 73 209 76 132 187 208 89 18
   169 200 196 135 130 116 188 159 86 164 100 109 198 173 186 3 64 52 217 226 250
   124 123 5 202 38 147 118 126 255 82 85 212 207 206 59 227 47 16 58 17 182 189
   28 42 223 183 170 213 119 248 152 2 44 154 163 70 221 153 101 155 167 43 172 9
   129 22 39 253 19 98 108 110 79 113 224 232 178 185 112 104 218 246 97 228 251
   34 242 193 238 210 144 12 191 179 162 241 81 51 145 235 249 14 239 107 49 192
   214 31 181 199 106 157 184 84 204 176 115 121 50 45 127 4 150 254 138 236 205
   93 222 114 67 29 24 72 243 141 128 195 78 66 215 61 156 180])

(def ^:private p (int-array (concat perm-base perm-base)))

(defn- fade [t] (* t t t (+ (* t (- (* t 6.0) 15.0)) 10.0)))
(defn- lerp [t a b] (+ a (* t (- b a))))

(defn- grad [hash x y z]
  (let [h (bit-and hash 15)
        u (if (< h 8) x y)
        v (if (< h 4) y (if (or (== h 12) (== h 14)) x z))]
    (+ (if (zero? (bit-and h 1)) u (- u))
       (if (zero? (bit-and h 2)) v (- v)))))

(defn noise3
  "Perlin noise at (x,y,z), remapped from [-1,1] to [0,1]."
  [x y z]
  (let [fx (Math/floor x) fy (Math/floor y) fz (Math/floor z)
        X (bit-and (long fx) 255)
        Y (bit-and (long fy) 255)
        Z (bit-and (long fz) 255)
        xf (- x fx) yf (- y fy) zf (- z fz)
        u (fade xf) v (fade yf) w (fade zf)
        A  (+ (aget p X) Y)
        AA (+ (aget p A) Z)
        AB (+ (aget p (inc A)) Z)
        B  (+ (aget p (inc X)) Y)
        BA (+ (aget p B) Z)
        BB (+ (aget p (inc B)) Z)
        n (lerp w
                (lerp v
                      (lerp u (grad (aget p AA) xf yf zf)
                              (grad (aget p BA) (- xf 1.0) yf zf))
                      (lerp u (grad (aget p AB) xf (- yf 1.0) zf)
                              (grad (aget p BB) (- xf 1.0) (- yf 1.0) zf)))
                (lerp v
                      (lerp u (grad (aget p (inc AA)) xf yf (- zf 1.0))
                              (grad (aget p (inc BA)) (- xf 1.0) yf (- zf 1.0)))
                      (lerp u (grad (aget p (inc AB)) xf (- yf 1.0) (- zf 1.0))
                              (grad (aget p (inc BB)) (- xf 1.0) (- yf 1.0) (- zf 1.0)))))]
    (/ (+ 1.0 n) 2.0)))

(defn noise2
  "2D Perlin noise at (x,y) in [0,1]."
  [x y]
  (noise3 x y 0.0))
