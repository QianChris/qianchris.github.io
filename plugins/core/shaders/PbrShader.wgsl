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
    posOrDir: vec4f,     // xyz = dir(directional)/pos(point), w = 0 dir / 1 point
    color:    vec4f,     // rgb = color, w = intensity
    viewProj: mat4x4f,   // directional shadow VP (identity for point/non-shadow)
    params:   vec4f,     // x = range, y = castShadow, z = shadowMapIndex, w = 0
};

struct LightData {
    ambient: vec4f,          // rgb = color, w = intensity multiplier
    count:   vec4f,          // x = light count, y = dirShadowCount, z = pointShadowCount
    lights:  array<Light, 16>,
};

struct TimeInput {
    time: f32, dt: f32, frame: f32, _pad: f32,
    mouse: vec4f,
};

struct Material {
    baseColor: vec4f,
    matParams: vec4f,  // x=metallic, y=roughness, z=ao, w=shadowReceive
    emissive:  vec4f,  // rgb=emissive, a=pad
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
    @location(5) inTangent: vec4f,
) -> VSOutput {
    let world = u.model * vec4f(inPos, 1.0);
    var out: VSOutput;
    out.pos = cam.vp * world;
    out.worldPos = world.xyz;
    out.worldNormal = normalize(u.normalMatrix * inNormal);
    out.worldTangent = normalize(u.normalMatrix * inTangent.xyz);
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

// Dominant-axis face index (0..5) for a direction, in the same order as
// LightSystem.CUBE_FACE_AXES (+X, -X, +Y, -Y, +Z, -Z). Point shadows use a
// 2d-array (6 layers per light) instead of a cube texture, so this face index
// is only ever compared against our own render-side face ordering — there is no
// WebGPU cube-face convention to satisfy.
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

// Per-light shadow factor (0 = fully shadowed, 1 = lit) at worldPos.
fn computeShadow(i: u32, worldPos: vec3f) -> f32 {
    let light = lightData.lights[i];
    let isPoint = light.posOrDir.w > 0.5;
    let shadowMapIndex = u32(light.params.z);

    if (isPoint) {
        let dir = worldPos - light.posOrDir.xyz;
        let dist = length(dir);
        let range = max(light.params.x, EPSILON);
        if (dist > range) { return 1.0; }  // beyond the light's reach → unshadowed
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

// SHADOW_DEBUG: 0=off, 1=shadowFactor, 2=shadowUV, 3=lightNdcZ, 4=shadowMapDepth
const SHADOW_DEBUG: i32 = 0;

fn firstShadowLight() -> i32 {
    let n = u32(lightData.count.x);
    for (var i = 0u; i < n; i = i + 1u) {
        if (lightData.lights[i].params.y > 0.5) { return i32(i); }
    }
    return -1;
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

    // normal mapping
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

    // accumulate all active lights; each shadow-casting light gets its own factor.
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

    // ambient
    color += baseColor.rgb * lightData.ambient.rgb * lightData.ambient.w * ao;

    // emissive
    color += mat.emissive.rgb * emissiveTex;
    color = color / (color + vec3f(1.0));

    // ── shadow debug visualization (first shadow-casting light) ──
    if (SHADOW_DEBUG > 0) {
        let sIdx = firstShadowLight();
        if (sIdx < 0) {
            return vec4f(1.0, 0.0, 1.0, 1.0);  // magenta = no shadow light
        }
        let uIdx = u32(sIdx);
        let sFactor = computeShadow(uIdx, in.worldPos);
        let shadowLight = lightData.lights[uIdx];
        if (SHADOW_DEBUG == 1) {
            return vec4f(vec3f(sFactor), 1.0);
        }
        if (shadowLight.posOrDir.w > 0.5) {
            // point light: no single UV/NDC; show factor only in debug mode 1.
            return vec4f(vec3f(sFactor), 1.0);
        }
        let clip = shadowLight.viewProj * vec4f(in.worldPos, 1.0);
        let ndc = clip.xyz / clip.w;
        let suv = ndc.xy * 0.5 + 0.5;
        if (SHADOW_DEBUG == 2) {
            return vec4f(suv, 0.0, 1.0);
        }
        if (SHADOW_DEBUG == 3) {
            return vec4f(vec3f(ndc.z), 1.0);
        }
        if (SHADOW_DEBUG == 4) {
            let layer = i32(shadowLight.params.z);
            let dims = textureDimensions(shadowMap2D, 0);
            let tc = clamp(vec2i(vec2f(dims) * suv), vec2i(0, 0), vec2i(dims) - 1);
            let mapDepth = textureLoad(shadowMap2D, tc, layer, 0);
            return vec4f(vec3f(mapDepth), 1.0);
        }
    }

    return vec4f(color, baseColor.a);
}
