// PBD Step 1 — predict positions.
// v += g·dt;  p_pred = p + v·dt.  Pinned particles (v.w > 0.5) frozen.

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

@group(0) @binding(0) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> predicted: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.particleCount) { return; }

    let p = positions[i];
    let v = velocities[i];

    if (v.w > 0.5) {
        predicted[i] = p;
        return;
    }

    let accel = vec3f(0.0, params.gravityY, 0.0);
    let newVel = v.xyz + accel * params.dt;
    let newPos = p.xyz + newVel * params.dt;

    predicted[i] = vec4f(newPos, p.w);
}
