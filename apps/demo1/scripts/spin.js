let boost = 0;
let angle = 0;
let lastTime = 0;

export function init(ctx) {
    ctx.on('mousedown', () => { boost += 6.0; });
}

export function update(ctx) {
    const rot = ctx.getField('Transform', 'rotation');
    if (!rot) return;

    const dt = lastTime === 0 ? 0 : ctx.time - lastTime;
    lastTime = ctx.time;

    // base speed + decaying boost from mouse clicks
    const speed = 1.0 + boost;
    boost *= Math.exp(-dt * 2.0);

    angle += speed * dt;
    const half = angle * 0.5;
    const s = Math.sin(half);
    const c = Math.cos(half);

    ctx.setField('Transform', 'rotation', [0, s, 0, c]);
}
