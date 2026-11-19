// GPU-driven particle emission.
// One invocation per particle to spawn this frame; pops a free slot from the
// dead-list stack and initialises it from the emitter's parameters.

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

struct Emit {
    origin:     vec4f,   // xyz, _pad
    direction:  vec4f,   // xyz, spread
    shapeR:     vec4f,   // shape(u32 bits), radius, speedMin, speedMax
    half:       vec4f,   // halfExtents.xyz, gravityScale
    life:       vec4f,   // lifeMin, lifeMax, startSize, endSize
    startColor: vec4f,
    endColor:   vec4f,
    info:       vec4f,   // emitCount(u32 bits), seedBase, time, _pad
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> dead: DeadList;
@group(0) @binding(2) var<uniform> emit: Emit;

fn pcg(state: ptr<function, u32>) -> f32 {
    var x = *state * 747796405u + 2891336453u;
    let word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    *state = x;
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

fn randVec3(state: ptr<function, u32>) -> vec3f {
    return vec3f(pcg(state), pcg(state), pcg(state)) * 2.0 - 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let emitCount = bitcast<u32>(emit.info.x);
    if (i >= emitCount) { return; }

    // Pop a free slot (guard against underflow when the pool is exhausted).
    let cap = arrayLength(&dead.indices);
    let before = atomicSub(&dead.count, 1u);
    if (before == 0u || before > cap) {
        atomicAdd(&dead.count, 1u);
        return;
    }
    let slot = dead.indices[before - 1u];

    var rng = bitcast<u32>(emit.info.y) + i * 2654435761u + slot * 40503u;

    let shape = bitcast<u32>(emit.shapeR.x);
    let radius = emit.shapeR.y;
    var pos = emit.origin.xyz;
    if (shape == 1u) {               // sphere
        pos += normalize(randVec3(&rng) + vec3f(0.0001)) * radius * pow(pcg(&rng), 0.3333);
    } else if (shape == 2u) {        // box
        pos += randVec3(&rng) * emit.half.xyz;
    }

    let spread = emit.direction.w;
    let dir = normalize(emit.direction.xyz + randVec3(&rng) * spread + vec3f(0.0, 0.0001, 0.0));
    let speed = mix(emit.shapeR.z, emit.shapeR.w, pcg(&rng));
    let vel = dir * speed;

    let life = mix(emit.life.x, emit.life.y, pcg(&rng));

    var p: Particle;
    p.p0 = vec4f(pos, life);
    p.p1 = vec4f(vel, life);
    p.cs = emit.startColor;
    p.ce = emit.endColor;
    p.ss = vec4f(emit.life.z, emit.life.w, emit.half.w, pcg(&rng));
    particles[slot] = p;
}
