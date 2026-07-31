#!/bin/bash
# A/B render harness: start a fresh nrepl-server + app, set the user's slider
# settings, render to $1, tear down. Needed because the subject map is built at
# image-load time in BOTH the CPU and GPU field paths, so a code change to it
# cannot be hot-reloaded into a running app the way the gen shader can.
set -e
OUT="$1"
SP="$(cd "$(dirname "$0")" && pwd)"
cd "$SP/../.."

kill $(lsof -ti :7888) 2>/dev/null || true
sleep 2
rm -f .nrepl-port
env -u GA_PAINTER_COUNT -u GA_PAINTER_SIZE -u GA_PAINTER_BROAD -u GA_PAINTER_MID \
    -u GA_PAINTER_FINE -u GA_PAINTER_DETAIL -u GA_PAINTER_VAR -u GA_PAINTER_CURV \
    -u GA_PAINTER_STROKE -u GA_PAINTER_CONTRAST -u GA_PAINTER_HARDNESS \
    -u GA_PAINTER_CUTIN -u GA_PAINTER_SWIRL \
    nohup jolt nrepl-server 7888 > "$SP/ab-server.log" 2>&1 &
for i in $(seq 1 90); do [ -f .nrepl-port ] && break; sleep 1; done

python3 "$SP/nrepl.py" -t 900 "(require 'splat-painter.core 'splat-painter.seed 'splat-painter.gen 'glimmer.core 'glimmer-gl.gtk) :ok" > /dev/null
python3 "$SP/nrepl.py" -t 120 '(do (future ((var splat-painter.core/-main) "img/A7A01535-topaz-rawdenoise-sharpen-crop.jpg")) :launched)' > /dev/null
for i in $(seq 1 40); do
  R=$(python3 "$SP/nrepl.py" -t 60 '[(some? @splat-painter.core/area-atom) (some? @splat-painter.core/image-atom)]' 2>&1)
  echo "$R" | grep -q '\[true true\]' && break
  sleep 3
done

python3 "$SP/nrepl.py" -t 300 -f "$SP/iso-setup.clj" > /dev/null
python3 "$SP/nrepl.py" -t 300 -f "$SP/controls-user.clj" > /dev/null
python3 "$SP/nrepl.py" -t 600 "(iso-render! :all \"$OUT\")"
kill $(lsof -ti :7888) 2>/dev/null || true
echo "wrote $OUT"
