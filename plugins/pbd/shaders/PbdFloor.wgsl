// Floor visualization: a flat grid-aligned quad at y = 0.
// No vertex buffer — 6 hardcoded quad vertices generated in the vertex shader.

struct Camera {
    vp:  mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;

const HALF_SIZE: f32 = 8.0;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) coord: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p: vec2f;
    if (vi == 0u) { p = vec2f(-HALF_SIZE, -HALF_SIZE); }
    else if (vi == 1u) { p = vec2f( HALF_SIZE, -HALF_SIZE); }
    else if (vi == 2u) { p = vec2f( HALF_SIZE,  HALF_SIZE); }
    else if (vi == 3u) { p = vec2f(-HALF_SIZE, -HALF_SIZE); }
    else if (vi == 4u) { p = vec2f( HALF_SIZE,  HALF_SIZE); }
    else { p = vec2f(-HALF_SIZE,  HALF_SIZE); }

    var out: VOut;
    out.pos = cam.vp * vec4f(p.x, 0.0, p.y, 1.0);
    out.coord = p;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    let cellScale = 1.0;
    let cx = floor(in.coord.x * cellScale);
    let cz = floor(in.coord.y * cellScale);
    let checker = (cx + cz) - 2.0 * floor((cx + cz) / 2.0);

    var baseCol = vec3f(0.18, 0.18, 0.22);
    if (checker > 0.5) {
        baseCol = vec3f(0.10, 0.10, 0.13);
    }

    let gridLine = min(
        fract(in.coord.x * cellScale),
        fract(in.coord.y * cellScale),
    );
    let gridIntensity = 1.0 - smoothstep(0.0, 0.03, gridLine);
    let gridColor = vec3f(0.4, 0.45, 0.55);
    let col = mix(baseCol, gridColor, gridIntensity * 0.6);

    return vec4f(col, 1.0);
}
