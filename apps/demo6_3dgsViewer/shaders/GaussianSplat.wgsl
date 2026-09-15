// 3D Gaussian Splatting renderer (EWA splatting).
//
// Camera UBO (group 0 binding 0): vp / ivp / pos / view / proj  (uniform-layouts.json "camera").
// Splat data (group 1, named "splat" layout):
//   binding 0: centers     array<vec4f>   (xyz + pad)
//   binding 1: colors      array<vec4f>   (rgba 0..1, opacity pre-multiplied into alpha)
//   binding 2: covariances array<f32>     (6 per splat: xx, xy, xz, yy, yz, zz)
//   binding 3: sortIndex   array<u32>     (back-to-front order)
//
// Each splat = one instanced 6-vertex quad (two triangles), generated from
// @builtin(vertex_index); @builtin(instance_index) selects the splat via
// sortIndex. Group 0 (frame / camera) is bound by the render graph before the
// geometry hook runs.

struct Camera {
    vp:   mat4x4f,
    ivp:  mat4x4f,
    pos:  vec4f,
    view: mat4x4f,
    proj: mat4x4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var<storage, read> centers: array<vec4f>;
@group(1) @binding(1) var<storage, read> colors: array<vec4f>;
@group(1) @binding(2) var<storage, read> covariances: array<f32>;
@group(1) @binding(3) var<storage, read> sortIndex: array<u32>;

// GsEntity uniform: world model matrix + viewport (px) + splat scale.
struct SplatUniform {
    model: mat4x4f,     // GsEntity Transform (position/rotation/scale)
    viewport: vec2f,    // render target size in physical pixels
    splatScale: f32,    // global splat size multiplier
    _pad: f32,
};
@group(1) @binding(4) var<uniform> su: SplatUniform;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) corner: vec2f,   // quad corner in [-1,1], for fragment gaussian
};

// Two-triangle quad, CCW. vertex_index 0..5 -> corner.
fn quadCorner(vid: u32) -> vec2f {
    switch vid {
        case 0u: { return vec2f(-1.0, -1.0); }
        case 1u: { return vec2f( 1.0, -1.0); }
        case 2u: { return vec2f( 1.0,  1.0); }
        case 3u: { return vec2f(-1.0, -1.0); }
        case 4u: { return vec2f( 1.0,  1.0); }
        default: { return vec2f(-1.0,  1.0); }
    }
}

@vertex
fn vs(@builtin(vertex_index) vid: u32,
      @builtin(instance_index) iid: u32) -> VOut {
    var out: VOut;
    var bad: bool = false;

    let sid = sortIndex[iid];
    let centerL = centers[sid];
    let color = colors[sid];

    // GsEntity model transform: center -> world space.
    let centerW = (su.model * vec4f(centerL.xyz, 1.0)).xyz;

    // View-space center.
    let pView = (cam.view * vec4f(centerW, 1.0)).xyz;
    let d = -pView.z;              // positive depth in front of camera
    if (d <= 0.001) { bad = true; }

    // Symmetric 3D covariance (xx, xy, xz, yy, yz, zz) in LOCAL space.
    let base = sid * 6u;
    let cxx = covariances[base + 0u];
    let cxy = covariances[base + 1u];
    let cxz = covariances[base + 2u];
    let cyy = covariances[base + 3u];
    let cyz = covariances[base + 4u];
    let czz = covariances[base + 5u];
    let sigmaL = mat3x3f(
        cxx, cxy, cxz,
        cxy, cyy, cyz,
        cxz, cyz, czz
    );

    // Model-transform the covariance: Sigma_w = M3 * Sigma_l * M3^T.
    let M3 = mat3x3f(su.model[0].xyz, su.model[1].xyz, su.model[2].xyz);
    let sigma = M3 * sigmaL * transpose(M3);

    // Rotate world covariance into view space: Rv = mat3x3(view); Sigma_v = Rv * Sigma * Rv^T.
    let Rv = mat3x3f(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz);
    let sigmaV = Rv * sigma * transpose(Rv);

    // Pixel-space focal: NDC focal (proj diagonal) scaled by half viewport.
    // cov2D eigenvalues come out in pixels^2 so splat size tracks screen pixels
    // and scales as 1/d (close = big, far = small).
    let focalX = abs(cam.proj[0][0]) * 0.5 * su.viewport.x;
    let focalY = abs(cam.proj[1][1]) * 0.5 * su.viewport.y;

    // Jacobian of the perspective projection (view -> screen px) at pView.
    // screen.x = focalX * pView.x / d, screen.y = focalY * pView.y / d.
    let inv_d = 1.0 / d;
    let inv_d2 = inv_d * inv_d;
    let J = mat3x3f(
        focalX * inv_d,            0.0,                        0.0,
        0.0,                       focalY * inv_d,            0.0,
        focalX * pView.x * inv_d2, focalY * pView.y * inv_d2,  0.0
    );
    // cov2D (3x3, only top-left 2x2 used) = J * Sigma_v * J^T, in pixels^2.
    let cov2D = J * sigmaV * transpose(J);
    let a = cov2D[0][0];
    let b = cov2D[0][1];
    let c = cov2D[1][1];

    // Eigen-decomposition of symmetric [[a, b], [b, c]] (px^2).
    let trace = a + c;
    let halfDiff = (a - c) * 0.5;
    let disc = sqrt(max(halfDiff * halfDiff + b * b, 0.0));
    let l1 = trace * 0.5 + disc;   // larger eigenvalue (px^2)
    let l2 = trace * 0.5 - disc;   // smaller eigenvalue (px^2)
    var v1: vec2f;
    if (abs(b) > 1e-6) {
        v1 = normalize(vec2f(b, l1 - a));
    } else if (a >= c) {
        v1 = vec2f(1.0, 0.0);
    } else {
        v1 = vec2f(0.0, 1.0);
    }
    let v2 = vec2f(-v1.y, v1.x);

    // Quad half-extent = 3*sigma (px); covers ~99.7% of the gaussian energy.
    let radius1 = 3.0 * su.splatScale * sqrt(max(l1, 0.0));
    let radius2 = 3.0 * su.splatScale * sqrt(max(l2, 0.0));

    // Project center to NDC.
    let clip = cam.vp * vec4f(centerW, 1.0);
    let ndc = clip.xyz / max(abs(clip.w), 1e-6) * sign(clip.w);

    let corner = quadCorner(vid);
    // Quad offset in pixels, then to NDC (divide by half viewport).
    let offPx = v1 * (corner.x * radius1) + v2 * (corner.y * radius2);
    let offNdc = offPx / (0.5 * su.viewport);

    out.color = color;
    out.corner = corner;
    if (bad) {
        out.pos = vec4f(2.0, 2.0, 2.0, 1.0);   // off-screen (clipped)
    } else {
        out.pos = vec4f(ndc.xy + offNdc, ndc.z, 1.0);
    }
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let A = dot(in.corner, in.corner);   // corner in [-1,1] -> A in [0,2]
    if (A > 1.0) { discard; }
    // r/sigma = 3*sqrt(A) (quad edge at 3 sigma). Real gaussian:
    let w = exp(-0.5 * 9.0 * A);          // = exp(-4.5 * A): 1 at center, ~0.011 at edge
    let alpha = w * in.color.a;
    return vec4f(in.color.rgb * alpha, alpha);   // premultiplied
}
