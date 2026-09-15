// Render script template — referenced by render.json `renderScripts`.
// Each export is a hook called by the render graph at the matching phase.
// Available hooks: value(ctx) → number|number[], geometry(pass, ctx),
// setup(ctx), teardown(ctx). Define only what you need.

export function value(ctx) {
    // Return a scalar or array consumed by a pipeline param binding.
    return [1.0, 0.0, 0.0, 1.0];
}

export function geometry(pass, ctx) {
    // Issue draw calls on the active pass encoder.
    // pass.draw(3); pass.drawIndexed(indexCount);
}
