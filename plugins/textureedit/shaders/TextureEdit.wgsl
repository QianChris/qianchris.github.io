struct Camera { vp: mat4x4f, ivp: mat4x4f, pos: vec4f, view: mat4x4f, proj: mat4x4f };
struct PerEntity { model: mat4x4f, color: vec4f, params: vec4f };

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> perEntity: PerEntity;
@group(2) @binding(0) var samp: sampler;
@group(2) @binding(1) var tex: texture_2d<f32>;

struct VOut {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
};

@vertex fn vs(
    @location(0) pos: vec3f,
    @location(2) uv:  vec2f,
) -> VOut {
    var out: VOut;
    out.position = camera.vp * perEntity.model * vec4f(pos, 1.0);
    out.uv = uv;
    return out;
}

@fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(tex, samp, uv);
}
