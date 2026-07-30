Prior art: Dynamic Auto-Painter, which is pre-llm and does a generative painting simulation, using noise and vector flow fields, etc. Not sure if it uses gaussian splats.

https://www.mediachance.com/dap/newsin7.html
https://www.mediachance.com/dap/new_v9.html

It even has node-based filter graph editor thing for custom filter pipeline functions
https://mediachance.com/reactor/index.html

create an oil-painting look built by progressive refinement. The painting starts wth an opaque underpainting of large soft strokes, then successively finer, more translucent layers of brush strokes traced along the image’s edges.You can customize the brush stroke process and parameter thresholds and color palettes across stages or layers.
The community creates scripts to simulate different painting styles.

It’s at the very least good at simplifying secondary detail in an image, and softening the sharp-all-over photo look, while maintaining integrity of prominent edges and contrasts.

It can apply brush stroke images (often sampled from real paintings), and the stroke textures add their own kind of organic noise. Fun to watch it paint.

The manual might be a valuable resource of parameters to consider.

https://www.mediachance.com/files/DAP8manual.pdf

I think the next level would be a 3d analysis layer. When the figurative masters paint, they soften and sharpen edges (often very subtly), rhythmically (like every couple inches sometimes). This “pops” the form off the flat ground. But it is not just according to shadows and light-leaks crossing shape boundaries (cast shadow, rim lighting, etc), which is what I think the field/noise analysis gives. It’s more about how the form “turns” in a perceived 3d space.

This is a close up of a hand painted by Sargeant. Notice the sharp/soft edge frequency at the small scale (barely percievable unless you get up close and analyze it)

could probably do a lot more progressive analysis on top of what splat painter is doing now.
The main levers I've been exploiting are stroke length and variation along with the opacity, to do detail refinement
adding texture would be the next logical step, so each stroke could have volume to it and have different opacity based on that for example

Yeah, that’s probably at least 80% of the desired “effect”.

Interestingly auto-painter works in a bump-map of a canvas texture to add more organic noise.

And another interesting natural painting app Rebelle even applies ray-tracing over paint stroke bump map to accentuate brush texture by casting shadows on the micro-texture, streaks.

And  zbrush was fully 3d strokes, so also has those impasto textures in strokes.