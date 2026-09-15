// PBF container wireframe — 12 box edges as a 24-vertex line list.
// No vertex buffer: corner positions are derived from the boxParams UBO and
// the vertex index (bit 0/1/2 of the corner id select boxMax on x/y/z).

struct Camera {
    vp:   mat4x4f,
    ivp:  mat4x4f,
    pos:  vec4f,
    view: mat4x4f,
    proj: mat4x4f,
};

struct BoxParams {
    boxMin: vec3f,
    _pad0:  f32,
    boxMax: vec3f,
    _pad1:  f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> B: BoxParams;

fn cornerPos(c: u32) -> vec3f {
    return vec3f(
        select(B.boxMin.x, B.boxMax.x, (c & 1u) != 0u),
        select(B.boxMin.y, B.boxMax.y, (c & 2u) != 0u),
        select(B.boxMin.z, B.boxMax.z, (c & 4u) != 0u),
    );
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    // Corner ids: bit0 = +x, bit1 = +y, bit2 = +z. Bottom ring 0-1-3-2,
    // top ring 4-5-7-6, verticals connect them.
    var edges = array<vec2u, 12>(
        vec2u(0u, 1u), vec2u(1u, 3u), vec2u(3u, 2u), vec2u(2u, 0u),
        vec2u(4u, 5u), vec2u(5u, 7u), vec2u(7u, 6u), vec2u(6u, 4u),
        vec2u(0u, 4u), vec2u(1u, 5u), vec2u(3u, 7u), vec2u(2u, 6u),
    );
    let e = edges[vi / 2u];
    let c = select(e.x, e.y, (vi % 2u) == 1u);
    return cam.vp * vec4f(cornerPos(c), 1.0);
}

@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(0.55, 0.55, 0.60, 1.0);
}
