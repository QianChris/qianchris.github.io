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

@vertex fn vs(
    @location(0) quad: vec2f,
    @location(1) start: vec3f,
    @location(2) end: vec3f,
) -> @builtin(position) vec4f {
    let mvp = cam.vp * u.model;
    let clipA = mvp * vec4f(start, 1.0);
    let clipB = mvp * vec4f(end, 1.0);

    let aspect = u.params.y;
    let sA = vec2f(clipA.x / clipA.w * aspect, clipA.y / clipA.w);
    let sB = vec2f(clipB.x / clipB.w * aspect, clipB.y / clipB.w);
    let seg = sB - sA;
    let slen = length(seg);
    let d = select(vec2f(1, 0), seg / slen, slen > 0.0001);
    let perp = vec2f(-d.y, d.x);

    let t = quad.x * 0.5 + 0.5;
    let clipP = mix(clipA, clipB, t);
    let sPos = mix(sA, sB, t) + perp * quad.y * u.params.x;
    let ndc = vec2f(sPos.x / aspect, sPos.y);

    return vec4f(ndc * clipP.w, clipP.z, clipP.w);
}

@fragment fn fs() -> @location(0) vec4f {
    return u.color;
}
