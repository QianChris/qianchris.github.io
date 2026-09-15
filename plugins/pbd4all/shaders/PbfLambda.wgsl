// PBF kernel: density constraint evaluation.
// rho = Σ m·poly6 (self + fluid + boundary psi);  C = rho/rho0 − 1 (enforced
// only when compressed);  λ = −C / (‖Σ∇C‖² + Σ‖∇C_k‖² + ε_CFM).
// Boundary contribution via Akinci et al. psi weights (static particles with
// precomputed psi = rho0 / Σ poly6).

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
fn poly6(r2: f32) -> f32 {
    let t = P.h2 - r2;
    return select(0.0, P.poly6 * t * t * t, r2 < P.h2);
}

@group(0) @binding(0) var<storage, read> pred: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> lambda: array<f32>;
@group(0) @binding(2) var<storage, read_write> density: array<f32>;
@group(0) @binding(3) var<storage, read> cellStart: array<u32>;
@group(0) @binding(4) var<storage, read> bpos: array<vec4f>;
@group(0) @binding(5) var<storage, read> bpsi: array<f32>;
@group(0) @binding(6) var<storage, read> bcellStart: array<u32>;
@group(0) @binding(7) var<uniform> P: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= P.n) { return; }
    let pi = pred[i].xyz;
    let c = cellOf(pi);
    let x0 = max(c.x - 1, 0);
    let x1 = min(c.x + 1, P.gridDimX - 1);
    var rho = P.mass * poly6(0.0);
    var gradSum = vec3f(0.0);
    var sumGrad2 = 0.0;
    for (var dz = -1; dz <= 1; dz++) {
        let z = c.z + dz;
        if (z < 0 || z >= P.gridDimZ) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let y = c.y + dy;
            if (y < 0 || y >= P.gridDimY) { continue; }
            let rowBase = u32((z * P.gridDimY + y) * P.gridDimX);
            let b = cellStart[rowBase + u32(x0)];
            let e = cellStart[rowBase + u32(x1) + 1u];
            for (var j = b; j < e; j++) {
                if (j == i) { continue; }
                let rij = pi - pred[j].xyz;
                let r2 = dot(rij, rij);
                if (r2 >= P.h2) { continue; }
                rho += P.mass * poly6(r2);
                if (r2 > 1.0e-12) {
                    let r = sqrt(r2);
                    let hr = P.h - r;
                    let g = (P.volume * P.spikyGrad * hr * hr / r) * rij;
                    gradSum += g;
                    sumGrad2 += dot(g, g);
                }
            }
            if (P.nBoundary > 0u) {
                let bb = bcellStart[rowBase + u32(x0)];
                let be = bcellStart[rowBase + u32(x1) + 1u];
                for (var k = bb; k < be; k++) {
                    let rij = pi - bpos[k].xyz;
                    let r2 = dot(rij, rij);
                    if (r2 >= P.h2 || r2 < 1.0e-12) { continue; }
                    rho += bpsi[k] * poly6(r2);
                    let r = sqrt(r2);
                    let hr = P.h - r;
                    gradSum += (bpsi[k] * P.invRho0 * P.spikyGrad * hr * hr / r) * rij;
                }
            }
        }
    }
    density[i] = rho;
    let C = rho * P.invRho0 - 1.0;
    if (C > 0.0) {
        lambda[i] = -C / (dot(gradSum, gradSum) + sumGrad2 + P.cfmEps + 1.0e-12);
    } else {
        lambda[i] = 0.0;
    }
}
