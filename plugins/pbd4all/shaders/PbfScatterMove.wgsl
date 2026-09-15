// PBF kernel: move pos/vel/pred into sorted order (counting sort, step 6).
// The A/B buffer parity flips after this pass — particle index == sorted slot
// for the rest of the substep, so all neighbour loops walk contiguous memory.

struct Params {
    boxMin: vec3f,
    dt: f32,
    boxMax: vec3f,
    h: f32,
    gridDimX: i32,
    gridDimY: i32,
    gridDimZ: i32,
    h2: f32,
    clampMin: vec3f,
    poly6: f32,
    clampMax: vec3f,
    spikyGrad: f32,
    gravity: f32,
    mass: f32,
    rho0: f32,
    invRho0: f32,
    cfmEps: f32,
    sCorrK: f32,
    sCorrWq: f32,
    xsphC: f32,
    n: u32,
    nCells: u32,
    nBoundary: u32,
    omega: f32,
    invDt: f32,
    volume: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read> slot: array<u32>;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> vel: array<vec4f>;
@group(0) @binding(3) var<storage, read> pred: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> pos2: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> vel2: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> pred2: array<vec4f>;
@group(0) @binding(7) var<uniform> P: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= P.n) { return; }
    let s = slot[i];
    pos2[s] = pos[i];
    vel2[s] = vel[i];
    pred2[s] = pred[i];
}
