// PBF kernel: assign each particle its destination slot in the sorted order
// (counting sort, step 5). slot[i] = cellStart[cell] + atomic cursor.

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

fn gridDim() -> vec3i { return vec3i(P.gridDimX, P.gridDimY, P.gridDimZ); }
fn cellOf(p: vec3f) -> vec3i {
    let c = vec3i(floor((p - P.boxMin) / P.h));
    return clamp(c, vec3i(0), gridDim() - vec3i(1));
}
fn cellIndex(c: vec3i) -> u32 {
    let g = gridDim();
    return u32((c.z * g.y + c.y) * g.x + c.x);
}

@group(0) @binding(0) var<storage, read> pred: array<vec4f>;
@group(0) @binding(1) var<storage, read> cellStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> slot: array<u32>;
@group(0) @binding(4) var<uniform> P: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= P.n) { return; }
    let cell = cellIndex(cellOf(pred[i].xyz));
    slot[i] = cellStart[cell] + atomicAdd(&cursor[cell], 1u);
}
