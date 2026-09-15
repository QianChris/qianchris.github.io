// PBF kernel: add block offsets into cellStart (counting sort, step 4).
// cellStart[nCells] receives the total particle count (exclusive scan end).

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

@group(0) @binding(0) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(1) var<storage, read> blockSum: array<u32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    let i = gid.x;
    if (i > P.nCells) { return; }
    if (i == P.nCells) {
        cellStart[i] = blockSum[(P.nCells + 255u) / 256u];
        return;
    }
    cellStart[i] = cellStart[i] + blockSum[wid.x];
}
