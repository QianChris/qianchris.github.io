// PBD soft-body rendering: surface mesh of the deforming cube.
// No vertex buffer — vertex_index looks up surfaceIndices[vi] for a particle
// index, then reads positions[particleIndex] for the deformed world position.
// Flat per-triangle normals (computed from 3 corners) give two-sided Lambert.

struct Camera {
    vp:  mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<storage, read> positions: array<vec4f>;
@group(1) @binding(1) var<storage, read> surfaceIndices: array<u32>;

const SURFACE_COLOR: vec3f = vec3f(0.35, 0.65, 0.95);
const FLOOR_Y: f32 = 0.0;
const LIGHT_DIR: vec3f = normalize(vec3f(0.5, 1.0, 0.3));
const AMBIENT: vec3f = vec3f(0.28, 0.28, 0.32);

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) normal: vec3f,
    @location(1) bary: vec3f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    let tri = vi / 3u;
    let local = vi % 3u;

    let ia = surfaceIndices[tri * 3u + 0u];
    let ib = surfaceIndices[tri * 3u + 1u];
    let ic = surfaceIndices[tri * 3u + 2u];

    let pa = positions[ia].xyz;
    let pb = positions[ib].xyz;
    let pc = positions[ic].xyz;

    let n = normalize(cross(pb - pa, pc - pa));

    var worldPos: vec3f;
    var bary: vec3f;
    if (local == 0u) { worldPos = pa; bary = vec3f(1.0, 0.0, 0.0); }
    else if (local == 1u) { worldPos = pb; bary = vec3f(0.0, 1.0, 0.0); }
    else { worldPos = pc; bary = vec3f(0.0, 0.0, 1.0); }

    var out: VOut;
    out.pos = cam.vp * vec4f(worldPos, 1.0);
    out.normal = n;
    out.bary = bary;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let lambert = abs(dot(n, LIGHT_DIR));
    let shading = mix(0.45, 1.0, lambert);

    let edgeDist = min(min(in.bary.x, in.bary.y), in.bary.z);
    let edgeIntensity = 1.0 - smoothstep(0.0, 0.04, edgeDist);
    let edgeColor = vec3f(0.7, 0.85, 1.0);

    var col = SURFACE_COLOR * (AMBIENT + vec3f(shading));
    col = mix(col, edgeColor, edgeIntensity * 0.5);

    return vec4f(col, 1.0);
}
