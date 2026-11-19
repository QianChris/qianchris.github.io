struct PerEntity {
    model:      mat4x4f,
    animParams: vec4f,
};
struct Camera {
    vp:    mat4x4f,
    ivp:   mat4x4f,
    camPos: vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> u: PerEntity;
@group(2) @binding(0) var samp: sampler;
@group(2) @binding(1) var tex: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0)       uv:  vec2f,
};

@vertex fn vs(
    @location(0) pos: vec3f,
    @location(2) uv:  vec2f,
) -> VOut {
    var out: VOut;
    out.pos = cam.vp * u.model * vec4f(pos, 1.0);
    out.uv  = uv;
    return out;
}

@fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
    let row   = u.animParams.x;
    let frame = u.animParams.y;
    let cols  = u.animParams.z;
    let rows  = u.animParams.w;

    let cellW = 1.0 / cols;
    let cellH = 1.0 / rows;
    let cellX = frame * cellW;
    let cellY = row   * cellH;
    let finalUv = vec2f(cellX + uv.x * cellW, cellY + uv.y * cellH);
    return textureSample(tex, samp, finalUv);
}
