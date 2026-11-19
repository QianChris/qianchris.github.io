// PBD Step 2 — Shape Matching solve with atomic accumulation.
//
// One thread per cluster. Each cluster:
//   1. Computes current center of mass from `predicted` positions.
//   2. Builds covariance A = sum( (p_i - com) * (q_i)^T ), where q_i is the
//      precomputed rest offset (relative to rest COM).
//   3. Extracts rotation R via polar decomposition (8 Newton iterations).
//   4. For each particle in the cluster: goal = com + R * q_i; correction =
//      (goal - p_i) * stiffness. The correction is encoded as fixed-point
//      (i32 round(value * atomFactor)) and atomicAdd'd into `corrections`,
//      with a matching atomicAdd(1) into `counts`.
//
// The Integrate pass averages by count, applies, and clears — no graph
// coloring is required because no thread writes `predicted` directly.
//
// See public/apps/demo9_softBody/wgsl_shape_matching_pbd_atomic.md for the
// full algorithm rationale.

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

struct AtomicCorr {
    x: atomic<i32>,
    y: atomic<i32>,
    z: atomic<i32>,
    _pad: atomic<i32>,
};

@group(0) @binding(0) var<storage, read_write> predicted: array<vec4f>;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> clusterIndices: array<u32>;
@group(0) @binding(3) var<storage, read> restOffsets: array<vec3f>;
@group(0) @binding(4) var<storage, read_write> corrections: array<AtomicCorr>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

fn atomicAddF32(addr: ptr<storage, atomic<i32>, read_write>, value: f32) {
    atomicAdd(addr, i32(round(value * params.atomFactor)));
}

fn outerProduct(a: vec3f, b: vec3f) -> mat3x3f {
    return mat3x3f(a * b.x, a * b.y, a * b.z);
}

fn inverseMat3(m: mat3x3f) -> mat3x3f {
    let a = m[0][0]; let b = m[0][1]; let c = m[0][2];
    let d = m[1][0]; let e = m[1][1]; let f = m[1][2];
    let g = m[2][0]; let h = m[2][1]; let i = m[2][2];

    let A = e * i - f * h;
    let B = c * h - b * i;
    let C = b * f - c * e;
    let D = f * g - d * i;
    let E = a * i - c * g;
    let F = c * d - a * f;
    let G = d * h - e * g;
    let H = b * g - a * h;
    let I = a * e - b * d;

    let det = a * A + b * D + c * G;
    let invDet = select(1e20, 1.0 / det, abs(det) > 1e-12);

    return mat3x3f(
        vec3f(A, B, C) * invDet,
        vec3f(D, E, F) * invDet,
        vec3f(G, H, I) * invDet,
    );
}

// Polar decomposition A = R * S via Newton iteration:
//   R_{k+1} = 0.5 * (R_k + transpose(inverse(R_k)))
fn polarDecompose(A: mat3x3f) -> mat3x3f {
    var R = A + mat3x3f(
        vec3f(1e-6, 0.0, 0.0),
        vec3f(0.0, 1e-6, 0.0),
        vec3f(0.0, 0.0, 1e-6),
    );
    for (var iter = 0; iter < 8; iter++) {
        let invRt = inverseMat3(transpose(R));
        R = 0.5 * (R + invRt);
    }
    return R;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cid = gid.x;
    if (cid >= params.clusterCount) { return; }

    let cl = clusters[cid];
    let n = f32(cl.count);

    // 1. Current center of mass from predicted positions.
    var com = vec3f(0.0);
    for (var i = 0u; i < cl.count; i++) {
        let pid = clusterIndices[cl.offset + i];
        com += predicted[pid].xyz;
    }
    com = com / n;

    // 2. Covariance A = sum( (p_i - com) * (q_i)^T ).
    //    q_i is the rest offset (already relative to rest COM).
    var A = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    for (var i = 0u; i < cl.count; i++) {
        let pid = clusterIndices[cl.offset + i];
        let pi = predicted[pid].xyz - com;
        let qi = restOffsets[cl.offset + i];
        A = A + outerProduct(pi, qi);
    }

    // 3. Extract optimal rotation R.
    let R = polarDecompose(A);

    // 4. Compute goal positions and accumulate corrections (Jacobi style).
    for (var i = 0u; i < cl.count; i++) {
        let pid = clusterIndices[cl.offset + i];
        let qi = restOffsets[cl.offset + i];
        let goal = com + R * qi;
        let corr = (goal - predicted[pid].xyz) * params.stiffness;

        atomicAddF32(&corrections[pid].x, corr.x);
        atomicAddF32(&corrections[pid].y, corr.y);
        atomicAddF32(&corrections[pid].z, corr.z);
        atomicAdd(&counts[pid], 1u);
    }
}
