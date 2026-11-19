// Demo5 director: tumbles the container box and cycles the 10 glow balls'
// point-light colors (+ matching emissive so their surfaces glow in-color).
// Runs on the Box entity (ctx.eid = Box). Reaches the balls via ctx.scene.

function hslToRgb(h, l) {
    const c = 1 - Math.abs(2 * l - 1);
    const hp = h * 6;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r = 0, g = 0, b = 0;
    if (hp < 1)      { r = c; g = x; b = 0; }
    else if (hp < 2) { r = x; g = c; b = 0; }
    else if (hp < 3) { r = 0; g = c; b = x; }
    else if (hp < 4) { r = 0; g = x; b = c; }
    else if (hp < 5) { r = x; g = 0; b = c; }
    else             { r = c; g = 0; b = x; }
    const m = l - c / 2;
    return [r + m, g + m, b + m];
}

export function update(ctx) {
    const t = ctx.time;
    const scene = ctx.scene;

    // Tumble the box end-over-end around the X axis (gravity-relative tumble).
    const a = t * 0.6;
    scene.setField(ctx.eid, 'Transform', 'rotation', [Math.sin(a * 0.5), 0, 0, Math.cos(a * 0.5)]);

    // Cycle the 10 glow balls: light color + matching emissive.
    let i = 0;
    for (const [key, eid] of scene.entityKeyMap) {
        if (!key.startsWith('GlowBall')) continue;
        const h = (t * 0.12 + i / 10) % 1;
        const c = hslToRgb(h, 0.6);
        scene.setField(eid, 'LightComponent', 'color', c);
        scene.setField(eid, 'PbrMaterial', 'emissive', [c[0] * 2.5, c[1] * 2.5, c[2] * 2.5]);
        i++;
    }
}
