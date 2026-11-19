struct Camera {
    vp:  mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var equirect: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var out: VOut;
    out.pos = vec4f(p[vi], 1.0, 1.0);
    out.uv = p[vi];
    return out;
}

const PI = 3.14159265359;
const INV_PI = 0.31830988618;

@fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
    let nearP = cam.ivp * vec4f(uv, 0.0, 1.0);
    let farP  = cam.ivp * vec4f(uv, 1.0, 1.0);
    let dir = normalize(farP.xyz / farP.w - nearP.xyz / nearP.w);

    let u = atan2(dir.z, dir.x) * (0.5 * INV_PI) + 0.5;
    let v = acos(clamp(dir.y, -1.0, 1.0)) * INV_PI;

    return textureSampleLevel(equirect, samp, vec2f(u, v), 0.0);
}
