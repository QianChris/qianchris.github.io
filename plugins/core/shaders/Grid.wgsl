struct Camera {
    vp:     mat4x4f,
    ivp:    mat4x4f,
    camPos: vec4f,
};
struct TimeInput {
    time: f32, dt: f32, frame: f32, _pad: f32,
    mouse: vec4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(2) var<uniform> timeInput: TimeInput;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) ndc: vec2f,
};

// fullscreen triangle
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    let xy = p[vi];
    var out: VOut;
    out.pos = vec4f(xy, 1.0, 1.0);
    out.ndc = xy;
    return out;
}

struct FOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
};

// anti-aliased grid line coverage for a given spacing
fn gridLine(coord: vec2f, spacing: f32) -> f32 {
    let g = coord / spacing;
    let d = abs(fract(g - 0.5) - 0.5) / fwidth(g);
    let line = min(d.x, d.y);
    return 1.0 - min(line, 1.0);
}

@fragment fn fs(in: VOut) -> FOut {
    // reconstruct a world-space ray from NDC through the inverse view-projection
    let nearP = cam.ivp * vec4f(in.ndc, 0.0, 1.0);
    let farP  = cam.ivp * vec4f(in.ndc, 1.0, 1.0);
    let near = nearP.xyz / nearP.w;
    let far  = farP.xyz / farP.w;
    let dir = far - near;

    var out: FOut;

    // intersect ray with y = 0 plane
    if (abs(dir.y) < 1e-6) { discard; }
    let t = -near.y / dir.y;
    if (t < 0.0 || t > 1.0) { discard; }

    let hit = near + dir * t;

    // depth: project hit back to clip space
    let clip = cam.vp * vec4f(hit, 1.0);
    out.depth = clip.z / clip.w;

    let coord = hit.xz;

    // two line frequencies: fine (1 unit) + coarse (10 units)
    let fine = gridLine(coord, 1.0);
    let coarse = gridLine(coord, 10.0);
    var g = max(fine * 0.4, coarse * 0.8);

    // axis lines (x and z)
    let aw = fwidth(coord) * 1.5;
    let xAxis = 1.0 - min(abs(coord.y) / aw.y, 1.0);
    let zAxis = 1.0 - min(abs(coord.x) / aw.x, 1.0);

    var color = vec3f(0.45);
    color = mix(color, vec3f(0.35, 0.55, 1.0), zAxis);   // Z axis -> blue
    color = mix(color, vec3f(1.0, 0.35, 0.4), xAxis);    // X axis -> red
    g = max(g, max(xAxis, zAxis));

    // distance fade
    let dist = length(hit.xz - cam.camPos.xz);
    let fade = 1.0 - smoothstep(20.0, 80.0, dist);
    let alpha = g * fade;

    if (alpha < 0.01) { discard; }

    out.color = vec4f(color, alpha);
    return out;
}
