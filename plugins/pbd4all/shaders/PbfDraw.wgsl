// PBF fluid rendering — camera-facing billboard spheres.
// Each particle renders as a 6-vertex quad expanded by `radius` along the
// camera right/up basis (extracted from the view matrix rows). The fragment
// stage reconstructs the sphere normal from the quad-local coords, writes
// corrected sphere depth via @builtin(frag_depth), and colours particles by
// speed (slow = deep blue, fast = near-white) with Blinn-Phong shading.
//
// (Deviation from the Particles4All source: no uv.y flip in the normal
// reconstruction — the source's flip inverts the sphere normal field.)

struct Camera {
    vp:   mat4x4f,
    ivp:  mat4x4f,
    pos:  vec4f,
    view: mat4x4f,
    proj: mat4x4f,
};

struct ViewParams {
    radius:   f32,
    speedMax: f32,
    _pad0:    f32,
    _pad1:    f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<storage, read> pos: array<vec4f>;
@group(1) @binding(1) var<storage, read> vel: array<vec4f>;
@group(1) @binding(2) var<uniform> VP: ViewParams;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) local: vec2f,
    @location(1) colour: vec3f,
    @location(2) viewCentre: vec3f,
    @location(3) radius: f32,
};

struct FsOut {
    @location(0) colour: vec4f,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var quad = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                               vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    let q = quad[vi];
    let r = VP.radius;
    let c = pos[ii].xyz;

    // Camera right/up = rows 0/1 of the view matrix (basis vectors).
    let v3 = mat3x3f(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz);
    let basis = transpose(v3);
    let camRight = basis[0];
    let camUp = basis[1];

    let world = c + (camRight * q.x + camUp * q.y) * r;

    var o: VsOut;
    o.clip = cam.vp * vec4f(world, 1.0);
    o.local = q;
    o.viewCentre = (cam.view * vec4f(c, 1.0)).xyz;
    o.radius = r;
    let speed = length(vel[ii].xyz);
    let t = clamp(speed / max(VP.speedMax, 1.0e-3), 0.0, 1.0);
    let slow = vec3f(0.10, 0.35, 0.85);
    let fast = vec3f(0.75, 0.92, 1.00);
    o.colour = mix(slow, fast, t);
    return o;
}

@fragment
fn fs(in: VsOut) -> FsOut {
    let uv = in.local;
    let r2 = dot(uv, uv);
    if (r2 > 1.0) { discard; }
    let n = vec3f(uv, sqrt(1.0 - r2));

    let viewPos = in.viewCentre + n * in.radius;
    let clip = cam.proj * vec4f(viewPos, 1.0);
    var o: FsOut;
    o.depth = clamp(clip.z / clip.w, 0.0, 1.0);

    let lightDir = normalize(vec3f(0.4, 0.8, 0.6));
    let diff = max(dot(n, lightDir), 0.0);
    let viewDir = normalize(-viewPos);
    let halfV = normalize(lightDir + viewDir);
    let spec = pow(max(dot(n, halfV), 0.0), 48.0);
    o.colour = vec4f(in.colour * (0.25 + 0.75 * diff) + vec3f(0.6) * spec, 1.0);
    return o;
}
