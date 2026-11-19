struct PerEntity {
    model:  mat4x4f,
    color:  vec4f,
    params: vec4f,
};
struct Camera {
    vp:  mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> u: PerEntity;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) local: vec2f,
};

@vertex fn vs(
    @location(0) quad:   vec2f,
    @location(1) center: vec3f,
) -> VOut {
    let clip = cam.vp * u.model * vec4f(center, 1.0);
    let offset = vec2f(quad.x * u.params.y, quad.y * u.params.x);
    var out: VOut;
    out.pos   = vec4f(clip.xy + offset * clip.w, clip.z, clip.w);
    out.local = quad;
    return out;
}

@fragment fn fs(@location(0) local: vec2f) -> @location(0) vec4f {
    if (length(local) > 1.0) { discard; }
    return u.color;
}
