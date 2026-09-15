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
struct TimeInput {
    time: f32, dt: f32, frame: f32, _pad: f32,
    mouse: vec4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(2) var<uniform> timeInput: TimeInput;
@group(1) @binding(0) var<uniform> u: PerEntity;

@vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
    let world = u.model * vec4f(pos, 1.0);
    return cam.vp * world;
}

@fragment fn fs() -> @location(0) vec4f {
    return u.color;
}
