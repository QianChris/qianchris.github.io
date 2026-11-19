// PBD Step 3a — apply Shape Matching corrections to predicted positions.
//
// Runs every solver iteration. Only modifies `predicted` — does NOT touch
// `positions` or `velocities`. The velocity integration happens once, after
// all iterations, in PbdIntegrate.wgsl. This separation is critical: if
// velocity were recomputed mid-loop, the gravity contribution (baked into
// `predicted` by the predict pass) would be lost after iteration 1, and the
// cube would fall at a near-constant speed regardless of gravity magnitude.

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

struct AtomicCorr {
    x: atomic<i32>,
    y: atomic<i32>,
    z: atomic<i32>,
    _pad: atomic<i32>,
};

@group(0) @binding(0) var<storage, read_write> predicted: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> corrections: array<AtomicCorr>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.particleCount) { return; }

    let cnt = atomicLoad(&counts[i]);
    let invScale = 1.0 / params.atomFactor;

    var corr = vec3f(0.0);
    if (cnt > 0u) {
        let cx = f32(atomicLoad(&corrections[i].x));
        let cy = f32(atomicLoad(&corrections[i].y));
        let cz = f32(atomicLoad(&corrections[i].z));
        corr = vec3f(cx, cy, cz) / f32(cnt) * invScale;
    }

    // Clear for next solve iteration.
    atomicStore(&corrections[i].x, 0);
    atomicStore(&corrections[i].y, 0);
    atomicStore(&corrections[i].z, 0);
    atomicStore(&counts[i], 0u);

    // Apply Jacobi-averaged correction to predicted (NOT to positions).
    var pred = predicted[i];
    pred = vec4f(pred.xyz + corr, pred.w);

    // Floor clamp during solving to prevent tunneling.
    if (pred.y < 0.0) { pred.y = 0.0; }

    predicted[i] = pred;
}
