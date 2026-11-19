// Gameplay script template — attach to an entity via ScriptComponent.script = "scripts/<name>.js"
// Lifecycle: init(ctx) once on first update, update(ctx) per frame, dispose() on unload.

export function init(ctx) {
    // ctx.on('mousedown' | 'keydown', handler) — input events
    // ctx.time, ctx.dt — seconds since start / last frame
}

export function update(ctx) {
    // Read component fields: const v = ctx.getField('Transform', 'position');
    // Write component fields: ctx.setField('Transform', 'position', [x, y, z]);
}

export function dispose() {
    // cleanup (event handlers auto-removed)
}
