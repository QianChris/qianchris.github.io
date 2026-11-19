struct PerEntity {
    model:        mat4x4f,
    normalMatrix: mat3x3f,
    entityId:     u32,
};

struct Light {
    posOrDir: vec4f,     // xyz = dir(directional)/pos(point), w = 0 dir / 1 point
    color:    vec4f,     // rgb = color, w = intensity
    viewProj: mat4x4f,   // directional shadow VP (identity for point/non-shadow)
    params:   vec4f,     // x = range, y = castShadow, z = shadowMapIndex, w = 0
};

struct LightData {
    ambient: vec4f,
    count:   vec4f,          // x = light count, y = dir shadow count, z = point shadow count
    lights:  array<Light, 16>,
};

struct ShadowPass {
    lightIdx: u32,           // index of the light being rendered (light0..light15)
    face:     u32,           // cube face 0..5 (point lights); 0 for directional
};

// frameShadow bind group: camera(0), light(1), time(2), pointShadowFaces(3).
@group(0) @binding(1) var<uniform> lightData: LightData;
@group(0) @binding(3) var<uniform> pointShadowFaces: array<mat4x4f, 96>;

@group(1) @binding(0) var<uniform> u: PerEntity;

// Per-face selector, set once per shadow render pass.
@group(2) @binding(0) var<uniform> shadowPass: ShadowPass;

@vertex
fn vs(
    @location(0) inPos: vec3f,
) -> @builtin(position) vec4f {
    let light = lightData.lights[shadowPass.lightIdx];
    let isPoint = light.posOrDir.w > 0.5;
    var vp: mat4x4f;
    if (isPoint) {
        let base = u32(light.params.z) * 6u;
        vp = pointShadowFaces[base + shadowPass.face];
    } else {
        vp = light.viewProj;
    }
    return vp * u.model * vec4f(inPos, 1.0);
}
