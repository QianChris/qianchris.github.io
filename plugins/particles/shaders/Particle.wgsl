// GPU-driven particle rendering.
// Instanced billboards: 6 verts per particle, instance list + pool are read-only
// storage, instance count comes from the indirect draw args filled by the sim.

struct Particle {
    p0: vec4f,   // pos.xyz, life (remaining)
    p1: vec4f,   // vel.xyz, maxLife
    cs: vec4f,   // startColor rgba
    ce: vec4f,   // endColor rgba
    ss: vec4f,   // startSize, endSize, gravityScale, seed
};

struct Camera {
    vp:  mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<storage, read> alive: array<u32>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) local: vec2f,
    @location(1) color: vec4f,
};

@vertex fn vs(
    @location(0) quad: vec2f,
    @builtin(instance_index) inst: u32,
) -> VOut {
    let p = particles[alive[inst]];

    let t = clamp(1.0 - p.p0.w / max(p.p1.w, 1e-4), 0.0, 1.0);
    let size = mix(p.ss.x, p.ss.y, t);
    let color = mix(p.cs, p.ce, t);

    // Camera-facing basis from the inverse view-projection (already uploaded).
    let right = normalize((cam.ivp * vec4f(1.0, 0.0, 0.0, 0.0)).xyz);
    let up    = normalize((cam.ivp * vec4f(0.0, 1.0, 0.0, 0.0)).xyz);

    let world = p.p0.xyz + (right * quad.x + up * quad.y) * size;

    var out: VOut;
    out.pos   = cam.vp * vec4f(world, 1.0);
    out.local = quad;
    out.color = color;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    let r = length(in.local);
    if (r > 1.0) { discard; }
    let soft = smoothstep(1.0, 0.6, r);
    return vec4f(in.color.rgb, in.color.a * soft);
}
