// Deferred lighting pass: a fullscreen triangle that reads the GBuffer + the
// light UBO + shadow maps and computes PBR lighting per pixel. The PBR math and
// per-light shadow sampling are the same as the forward PbrShader, so deferred
// and forward render the same lighting — the difference is geometry is rasterized
// once (GBuffer) and lighting is a fullscreen pass independent of scene depth.

struct Camera {
    vp:     mat4x4f,
    ivp:    mat4x4f,
    camPos: vec4f,
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

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> lightData: LightData;
@group(0) @binding(3) var shadowMap2D: texture_depth_2d_array;
@group(0) @binding(4) var shadowSampler: sampler_comparison;
@group(0) @binding(5) var shadowPoint2D: texture_depth_2d_array;
@group(0) @binding(6) var<uniform> pointShadowFaces: array<mat4x4f, 96>;

// GBuffer (group 1): A=pos+metallic, B=normal+roughness, C=albedo+ao, D=emissive.
@group(1) @binding(0) var gbufferA: texture_2d<f32>;
@group(1) @binding(1) var gbufferB: texture_2d<f32>;
@group(1) @binding(2) var gbufferC: texture_2d<f32>;
@group(1) @binding(3) var gbufferD: texture_2d<f32>;
@group(1) @binding(4) var gbufSampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var out: VSOut;
    out.pos = vec4f(p[vi], 0.0, 1.0);
    out.uv = p[vi] * 0.5 + 0.5;
    return out;
}

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.0001;
const MIN_ROUGHNESS: f32 = 0.04;
const SHADOW_BIAS: f32 = 0.001;

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
fn calcLight(f0: vec3f, diffuseColor: vec3f, alpha2: f32, NoV: f32, N: vec3f, V: vec3f, L: vec3f, intensity: f32, lightColor: vec3f) -> vec3f {
    let H = normalize(L + V);
    let NoL = max(dot(N, L), 0.0);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);
    let D = dGGX(alpha2, NoH);
    let G = vGGX(alpha2, NoV, NoL);
    let F = fSchlick(f0, VoH);
    let spec = F * (D * G);
    let diffuse = (1.0 - fSchlick(f0, VoH)) * (diffuseColor / PI);
    return (diffuse + spec) * NoL * intensity * lightColor;
}

// Dominant-axis face index matching LightSystem.CUBE_FACE_AXES (+X,-X,+Y,-Y,+Z,-Z).
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

// Per-light shadow factor — identical logic to the forward PbrShader so the same
// shadow maps (rendered by the Shadow phase) are reused here.
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
fn fs(in: VSOut) -> @location(0) vec4f {
    // Exact per-pixel GBuffer read (lighting target and GBuffer are both viewport-sized).
    let tc = vec2i(i32(in.pos.x), i32(in.pos.y));
    let gA = textureLoad(gbufferA, tc, 0);
    let gB = textureLoad(gbufferB, tc, 0);
    let gC = textureLoad(gbufferC, tc, 0);
    let gD = textureLoad(gbufferD, tc, 0);

    let albedo = gC.rgb;
    let ao = gC.w;
    // Background: the GBuffer was cleared to zero → ao == 0 marks empty pixels.
    if (ao < 0.5) {
        return vec4f(lightData.ambient.rgb * lightData.ambient.w, 1.0);
    }

    let worldPos = gA.xyz;
    let metallic = gA.w;
    let N = normalize(gB.xyz);
    let roughness = max(gB.w, MIN_ROUGHNESS);
    let emissive = gD.rgb;

    let V = normalize(cam.camPos.xyz - worldPos);
    let NoV = max(dot(N, V), EPSILON);
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let diffuseColor = mix(albedo * (1.0 - f0), vec3f(0.0), vec3f(metallic));

    var color = vec3f(0.0);
    let lightCount = u32(lightData.count.x);
    for (var i = 0u; i < lightCount; i = i + 1u) {
        let light = lightData.lights[i];
        let isPoint = light.posOrDir.w > 0.5;
        let intensity = light.color.w;

        // Per-light shadow (shadowReceive is implicitly on — all demo5 materials receive).
        var factor = 1.0;
        if (light.params.y > 0.5) {
            factor = computeShadow(i, worldPos);
        }

        if (isPoint) {
            let toLight = light.posOrDir.xyz - worldPos;
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

    // ambient + emissive, then Reinhard tonemap.
    color += albedo * lightData.ambient.rgb * lightData.ambient.w * ao;
    color += emissive;
    color = color / (color + vec3f(1.0));
    return vec4f(color, 1.0);
}
