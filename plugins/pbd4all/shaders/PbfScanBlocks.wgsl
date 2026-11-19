// PBF kernel: exclusive scan of the per-block sums (counting sort, step 3).
// Single workgroup, sequential over blocks — block count is small (nCells/256).

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

@group(0) @binding(0) var<storage, read_write> blockSum: array<u32>;
@group(0) @binding(1) var<uniform> P: Params;

@compute @workgroup_size(1)
fn main() {
    let nBlocks = (P.nCells + 255u) / 256u;
    var run = 0u;
    for (var b = 0u; b < nBlocks; b++) {
        let v = blockSum[b];
        blockSum[b] = run;
        run = run + v;
    }

    blockSum[nBlocks] = run;
}
