// PBF kernel: 256-wide workgroup Hillis–Steele scan of the cell histogram
// (counting sort, step 2). Per-block sums go to blockSum for the next stage.

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

@group(0) @binding(0) var<storage, read> cellCount: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSum: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    let i = gid.x;
    let t = lid.x;
    tmp[t] = select(0u, cellCount[i], i < P.nCells);
    workgroupBarrier();

    for (var off = 1u; off < 256u; off = off << 1u) {
        var v = tmp[t];
        if (t >= off) { v = v + tmp[t - off]; }
        workgroupBarrier();
        tmp[t] = v;
        workgroupBarrier();
    }
    if (i < P.nCells) {
        cellStart[i] = select(tmp[t - 1u], 0u, t == 0u);
    }
    if (t == 255u) { blockSum[wid.x] = tmp[255]; }
}
