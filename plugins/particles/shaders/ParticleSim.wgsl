// GPU-driven particle simulation.
// One invocation per pool slot: integrates live particles under gravity + force
// fields, recycles dead ones onto the dead-list, and appends survivors to the
// alive-list while bumping the indirect-draw instance count.

struct Particle {
    p0: vec4f,   // pos.xyz, life (remaining)
    p1: vec4f,   // vel.xyz, maxLife
    cs: vec4f,   // startColor rgba
    ce: vec4f,   // endColor rgba
    ss: vec4f,   // startSize, endSize, gravityScale, seed
};

struct DeadList {
    count: atomic<u32>,
    indices: array<u32>,
};

struct DrawArgs {
    vertexCount:   u32,
    instanceCount: atomic<u32>,
    firstVertex:   u32,
    firstInstance: u32,
};

struct ForceField {
    posType: vec4f,   // pos.xyz, type (0 point, 1 vortex, 2 directional, 3 drag)
    dirStr:  vec4f,   // dir.xyz, strength
    shape:   vec4f,   // radius, falloff, _pad, _pad
};

struct Sim {
    header:  vec4f,   // dt, time, drag, ffCount
    gravity: vec4f,   // xyz, capacity(u32 bits)
    fields:  array<ForceField, 8>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> dead: DeadList;
@group(0) @binding(2) var<storage, read_write> alive: array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgs: DrawArgs;
@group(0) @binding(4) var<uniform> sim: Sim;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let capacity = bitcast<u32>(sim.gravity.w);
    if (i >= capacity) { return; }

    var p = particles[i];
    if (p.p0.w <= 0.0) { return; }   // already dead

    let dt = sim.header.x;
    var pos = p.p0.xyz;
    var vel = p.p1.xyz;

    var accel = sim.gravity.xyz * p.ss.z;

    let ffCount = bitcast<u32>(sim.header.w);
    for (var f = 0u; f < ffCount; f = f + 1u) {
        let ff = sim.fields[f];
        let ftype = bitcast<u32>(ff.posType.w);
        let strength = ff.dirStr.w;
        if (ftype == 2u) {                       // directional
            accel += normalize(ff.dirStr.xyz + vec3f(0.0, 0.0001, 0.0)) * strength;
            continue;
        }
        if (ftype == 3u) {                       // drag toward zero velocity
            accel -= vel * strength;
            continue;
        }
        let toField = ff.posType.xyz - pos;
        let dist = length(toField) + 1e-4;
        let radius = ff.shape.x;
        if (dist > radius) { continue; }
        let atten = pow(clamp(1.0 - dist / radius, 0.0, 1.0), ff.shape.y);
        let dir = toField / dist;
        if (ftype == 0u) {                       // point attract (+) / repel (-)
            accel += dir * strength * atten;
        } else if (ftype == 1u) {                // vortex around field axis
            let axis = normalize(ff.dirStr.xyz + vec3f(0.0, 0.0001, 0.0));
            accel += cross(axis, dir) * strength * atten;
        }
    }

    // Semi-implicit Euler with linear drag.
    vel += accel * dt;
    vel *= max(0.0, 1.0 - sim.header.z * dt);
    pos += vel * dt;

    let life = p.p0.w - dt;
    if (life <= 0.0) {
        // recycle: mark dead + push slot back to the dead-list
        p.p0.w = 0.0;
        particles[i] = p;
        let slot = atomicAdd(&dead.count, 1u);
        dead.indices[slot] = i;
        return;
    }

    p.p0 = vec4f(pos, life);
    p.p1 = vec4f(vel, p.p1.w);
    particles[i] = p;

    // append to the alive list and grow the indirect instance count
    let idx = atomicAdd(&drawArgs.instanceCount, 1u);
    alive[idx] = i;
}
