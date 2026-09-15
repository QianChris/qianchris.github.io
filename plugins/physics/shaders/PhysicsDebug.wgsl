struct Camera {
    vp:     mat4x4f,
    ivp:    mat4x4f,
    camPos: vec4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

@vertex fn vs(@location(0) position: vec3f, @location(1) color: vec4f) -> VOut {
    var out: VOut;
    out.pos = cam.vp * vec4f(position, 1.0);
    out.color = color;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    return in.color;
}
