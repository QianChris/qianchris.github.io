// SkinnedPbr.wgsl — PBR with per-vertex GPU skinning. Mirrors core:PbrShader
// but the vertex shader applies joint matrices (storage buffer) weighted by
// JOINTS_0/WEIGHTS_0 instead of the entity model matrix. The skin matrix is
// already world-space (jointWorld × inverseBind), so u.model/normalMatrix are
// not applied to skinned vertices.

struct PerEntity {
    model:        mat4x4f,
    normalMatrix: mat3x3f,
    entityId:     u32,
};

struct Camera {
    vp:      mat4x4f,
    ivp:     mat4x4f,
    camPos:  vec4f,
};

struct Light {
    posOrDir: vec4f,
    color:    vec4f,
    viewProj: mat4x4f,
    params:   vec4f,
};

struct LightData {
    ambient: vec4f,
    count:   vec4f,
    lights:  array<Light, 16>,
};

struct TimeInput {
    time: f32, dt: f32, frame: f32, _pad: f32,
    mouse: vec4f,
};

struct Material {
    baseColor: vec4f,
    matParams: vec4f,
    emissive:  vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> lightData: LightData;
@group(0) @binding(2) var<uniform> timeInput: TimeInput;
@group(0) @binding(3) var shadowMap2D: texture_depth_2d_array;
@group(0) @binding(4) var shadowSampler: sampler_comparison;
@group(0) @binding(5) var shadowPoint2D: texture_depth_2d_array;
@group(0) @binding(6) var<uniform> pointShadowFaces: array<mat4x4f, 96>;

@group(1) @binding(0) var<uniform> u: PerEntity;

@group(2) @binding(0) var<uniform> mat: Material;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var texBaseColor: texture_2d<f32>;
@group(2) @binding(3) var texMetalRough: texture_2d<f32>;
@group(2) @binding(4) var texOcclusion: texture_2d<f32>;
@group(2) @binding(5) var texEmissive: texture_2d<f32>;
@group(2) @binding(6) var texNormal: texture_2d<f32>;

// Skinning joint matrices (one mat4 per joint), published by SkinningSystem
// via registerGpuResourceSet('animation', { 'animation.joint-matrices': buffer }).
@group(3) @binding(0) var<storage, read> jointMatrices: array<mat4x4f>;

struct VSOutput {
    @builtin(position) pos: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) worldNormal: vec3f,
    @location(2) uv: vec2f,
    @location(3) worldTangent: vec3f,
    @location(4) worldBitangent: vec3f,
};

@vertex
fn vs(
    @location(0) inPos: vec3f,
    @location(1) inNormal: vec3f,
    @location(2) inUv: vec2f,
    @location(3) inJoints: vec4u,
    @location(4) inWeights: vec4f,
    @location(5) inTangent: vec4f,
) -> VSOutput {
    let skinMat =
        jointMatrices[inJoints.x] * inWeights.x
      + jointMatrices[inJoints.y] * inWeights.y
      + jointMatrices[inJoints.z] * inWeights.z
      + jointMatrices[inJoints.w] * inWeights.w;

    let world = skinMat * vec4f(inPos, 1.0);
    var out: VSOutput;
    out.pos = cam.vp * world;
    out.worldPos = world.xyz;
    out.worldNormal = normalize((skinMat * vec4f(inNormal, 0.0)).xyz);
    out.worldTangent = normalize((skinMat * vec4f(inTangent.xyz, 0.0)).xyz);
    out.worldBitangent = cross(out.worldNormal, out.worldTangent) * inTangent.w;
    out.uv = inUv;
    return out;
}

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.0001;
const MIN_ROUGHNESS: f32 = 0.04;

fn dGGX(alpha2: f32, NoH: f32) -> f32 {
    let f = (NoH * alpha2 - NoH) * NoH + 1.0;
    return alpha2 / (PI * f * f);
}

fn vGGX(alpha2: f32, NoV: f32, NoL: f32) -> f32 {
    let gv = NoV + sqrt((NoV - NoV * alpha2) * NoV + alpha2);
    let gl = NoL + sqrt((NoL - NoL * alpha2) * NoL + alpha2);
    return 1.0 / (gv * gl);
}

fn fSchlick(f0: vec3f, VoH: f32) -> vec3f {
    let p = pow(1.0 - VoH, 5.0);
    return p + f0 * (1.0 - p);
}

fn specularBrdf(f0: vec3f, alpha2: f32, NoV: f32, NoL: f32, NoH: f32, VoH: f32) -> vec3f {
    let D = dGGX(alpha2, NoH);
    let G = vGGX(alpha2, NoV, NoL);
    let F = fSchlick(f0, VoH);
    return F * (D * G);
}

fn calcLight(f0: vec3f, diffuseColor: vec3f, alpha2: f32, NoV: f32, N: vec3f, V: vec3f, L: vec3f, intensity: f32, lightColor: vec3f) -> vec3f {
    let H = normalize(L + V);
    let NoL = max(dot(N, L), 0.0);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);
    let spec = specularBrdf(f0, alpha2, NoV, NoL, NoH, VoH);
    let diffuse = (1.0 - fSchlick(f0, VoH)) * (diffuseColor / PI);
    return (diffuse + spec) * NoL * intensity * lightColor;
}

const SHADOW_BIAS: f32 = 0.001;

fn cubeFaceIndex(dir: vec3f) -> u32 {
    let ax = abs(dir.x);
    let ay = abs(dir.y);
    let az = abs(dir.z);
    if (ax >= ay && ax >= az) {
        return select(1u, 0u, dir.x > 0.0);
    } else if (ay >= ax && ay >= az) {
        return select(3u, 2u, dir.y > 0.0);
    } else {
        return select(5u, 4u, dir.z > 0.0);
    }
}

fn computeShadow(i: u32, worldPos: vec3f) -> f32 {
    let light = lightData.lights[i];
    let isPoint = light.posOrDir.w > 0.5;
    let shadowMapIndex = u32(light.params.z);
    if (isPoint) {
        let dir = worldPos - light.posOrDir.xyz;
        let dist = length(dir);
        let range = max(light.params.x, EPSILON);
        if (dist > range) { return 1.0; }
        let face = cubeFaceIndex(dir);
        let faceSlot = shadowMapIndex * 6u + face;
        let faceVP = pointShadowFaces[faceSlot];
        let clip = faceVP * vec4f(worldPos, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0) {
            return 1.0;
        }
        return textureSampleCompareLevel(shadowPoint2D, shadowSampler, uv, i32(faceSlot), ndc.z - SHADOW_BIAS);
    } else {
        let clip = light.viewProj * vec4f(worldPos, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) {
            return 1.0;
        }
        return textureSampleCompareLevel(shadowMap2D, shadowSampler, uv, i32(shadowMapIndex), ndc.z - SHADOW_BIAS);
    }
}

@fragment
fn fs(in: VSOutput) -> @location(0) vec4f {
    let baseColor = mat.baseColor * textureSample(texBaseColor, samp, in.uv);
    let metalRough = textureSample(texMetalRough, samp, in.uv);
    let occlusion = textureSample(texOcclusion, samp, in.uv).r;
    let emissiveTex = textureSample(texEmissive, samp, in.uv).rgb;

    let metallic = mat.matParams.x * metalRough.b;
    let roughness = max(mat.matParams.y * metalRough.g, MIN_ROUGHNESS);
    let ao = mat.matParams.z * occlusion;

    var N = normalize(in.worldNormal);
    let tn = textureSample(texNormal, samp, in.uv).rgb * 2.0 - 1.0;
    let T = normalize(in.worldTangent);
    let B = normalize(in.worldBitangent);
    N = normalize(T * tn.x + B * tn.y + N * tn.z);

    let V = normalize(cam.camPos.xyz - in.worldPos);
    let NoV = max(dot(N, V), EPSILON);

    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;

    let f0 = mix(vec3f(0.04), baseColor.rgb, metallic);
    let diffuseColor = mix(baseColor.rgb * (1.0 - f0), vec3f(0.0), vec3f(metallic));

    var color = vec3f(0.0);
    let lightCount = u32(lightData.count.x);
    for (var i = 0u; i < lightCount; i = i + 1u) {
        let light = lightData.lights[i];
        let isPoint = light.posOrDir.w > 0.5;
        let intensity = light.color.w;
        var factor = 1.0;
        if (light.params.y > 0.5 && mat.matParams.w > 0.5) {
            factor = computeShadow(i, in.worldPos);
        }
        if (isPoint) {
            let toLight = light.posOrDir.xyz - in.worldPos;
            let dist = length(toLight);
            let L = toLight / max(dist, EPSILON);
            let range = max(light.params.x, EPSILON);
            let d = dist / range;
            let attenuation = clamp(1.0 - d * d, 0.0, 1.0);
            color += calcLight(f0, diffuseColor, alpha2, NoV, N, V, L, intensity, light.color.rgb) * attenuation * factor;
        } else {
            let L = normalize(-light.posOrDir.xyz);
            color += calcLight(f0, diffuseColor, alpha2, NoV, N, V, L, intensity, light.color.rgb) * factor;
        }
    }

    color += baseColor.rgb * lightData.ambient.rgb * lightData.ambient.w * ao;
    color += mat.emissive.rgb * emissiveTex;
    color = color / (color + vec3f(1.0));

    return vec4f(color, baseColor.a);
}
