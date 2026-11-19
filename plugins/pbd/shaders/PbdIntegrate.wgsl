// PBD Step 3b — integrate velocity from final predicted positions.
//
// Runs ONCE per frame, after all solver iterations have completed. The
// predict pass baked gravity into `predicted`; solve+apply refined `predicted`
// via shape-matching corrections; this pass now derives the frame's velocity
// from the total position delta (predicted - start_of_frame_position) / dt.
//
// Standard PBD integration:
//   v = (p_pred - p) / dt
//   apply damping + floor reflection
//   p = p_pred

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.particleCount) { return; }

    let p = positions[i];      // start-of-frame position
    let v = velocities[i];     // start-of-frame velocity (only .w pinned flag is used)
    let pred = predicted[i];   // final predicted position after all solve+apply iters

    // Pinned particles: accept predicted, keep velocity (.w holds pinned flag).
    if (v.w > 0.5) {
        positions[i] = pred;
        return;
    }

    // Velocity from total frame position delta. This captures the full gravity
    // contribution (predict's g·dt step) plus all constraint corrections.
    var newVel = (pred.xyz - p.xyz) / max(params.dt, 1e-6);

    // Floor bounce — only triggered when the particle is at floor level and
    // still moving down (post-correction velocity).
    if (pred.y < 0.001 && newVel.y < -0.3) {
        newVel.y = -newVel.y * params.restitution;
        newVel.x = newVel.x * 0.8;
        newVel.z = newVel.z * 0.8;
    }

    newVel = newVel * params.damping;

    positions[i] = pred;
    velocities[i] = vec4f(newVel, v.w);
}
