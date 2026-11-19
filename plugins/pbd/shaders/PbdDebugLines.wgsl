// PBD debug overlay — visualize one cluster's structure.
//
// Draws a "star" of lines from the selected cluster's current center of mass
// to each of its member particles. The cluster index is selected on the CPU
// (PbdManager.debugCluster) and written into the pbdParams UBO's debugCluster
// field each frame. Toggle with C, cycle with [ / ].
//
// Topology: line-list. Vertex count = clusterSize * 2.
// Even vertices = current COM; odd vertices = member particle position.

struct Camera {
    vp:  mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};

struct Params {
    dt: f32,
    time: f32,
    gravityY: f32,
    damping: f32,
    solverIterations: u32,
    particleCount: u32,
    restitution: f32,
    clusterCount: u32,
    atomFactor: f32,
    stiffness: f32,
    debugCluster: u32,
    _pad1: f32,
};

struct Cluster {
    restCom: vec3f,
    count: u32,
    offset: u32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<storage, read> positions: array<vec4f>;
@group(1) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(1) @binding(2) var<storage, read> clusterIndices: array<u32>;
@group(1) @binding(3) var<uniform> params: Params;

const CENTER_COLOR: vec3f = vec3f(1.0, 0.55, 0.15);
const MEMBER_COLOR: vec3f = vec3f(0.35, 1.0, 0.55);

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
};

@vertex fn vs(@builtin(vertex_index) vid: u32) -> VOut {
    let cid = params.debugCluster;
    let cl = clusters[cid];
    let pair = vid / 2u;
    let isCenter = (vid % 2u) == 0u;

    // Current COM — computed live from member positions (matches solve shader).
    var com = vec3f(0.0);
    for (var i = 0u; i < cl.count; i++) {
        let pid = clusterIndices[cl.offset + i];
        com += positions[pid].xyz;
    }
    com = com / f32(cl.count);

    var worldPos: vec3f;
    if (isCenter) {
        worldPos = com;
    } else {
        let pid = clusterIndices[cl.offset + pair];
        worldPos = positions[pid].xyz;
    }

    var out: VOut;
    out.pos = cam.vp * vec4f(worldPos, 1.0);
    out.color = select(MEMBER_COLOR, CENTER_COLOR, isCenter);
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
