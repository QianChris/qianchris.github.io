struct PerEntity {
    model: mat4x4f,
    normalMatrix: mat3x3f,
    entityId: u32,
};

struct Camera {
    vp: mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};

struct Light {
    posOrDir: vec4f,
    color: vec4f,
    viewProj: mat4x4f,
    params: vec4f,
};

struct LightData {
    ambient: vec4f,
    count: vec4f,
    lights: array<Light, 16>,
};

struct Material {
    baseColor: vec4f,
    matParams: vec4f,
    emissive: vec4f,
};

struct EnvironmentData {
    sh: array<vec4f, 9>,
    params: vec4f,
    rotation: vec4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lightData: LightData;
@group(0) @binding(3) var shadowMap2D: texture_depth_2d_array;
@group(0) @binding(4) var shadowSampler: sampler_comparison;
@group(0) @binding(5) var shadowPoint2D: texture_depth_2d_array;
@group(0) @binding(6) var<uniform> pointShadowFaces: array<mat4x4f, 96>;

@group(1) @binding(0) var<uniform> entity: PerEntity;

@group(2) @binding(0) var<uniform> material: Material;
@group(2) @binding(1) var materialSampler: sampler;
@group(2) @binding(2) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(3) var metalRoughTexture: texture_2d<f32>;
@group(2) @binding(4) var occlusionTexture: texture_2d<f32>;
@group(2) @binding(5) var emissiveTexture: texture_2d<f32>;
@group(2) @binding(6) var normalTexture: texture_2d<f32>;

@group(3) @binding(0) var<uniform> environment: EnvironmentData;
@group(3) @binding(1) var environmentSampler: sampler;
@group(3) @binding(2) var environmentSpecular: texture_cube<f32>;
@group(3) @binding(3) var environmentBrdfLut: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f,
    @location(1) worldNormal: vec3f,
    @location(2) uv: vec2f,
    @location(3) worldTangent: vec3f,
    @location(4) worldBitangent: vec3f,
};

@vertex fn vs(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(5) tangent: vec4f,
) -> VertexOutput {
    let world = entity.model * vec4f(position, 1.0);
    var out: VertexOutput;
    out.position = camera.vp * world;
    out.worldPosition = world.xyz;
    out.worldNormal = normalize(entity.normalMatrix * normal);
    out.worldTangent = normalize(entity.normalMatrix * tangent.xyz);
    out.worldBitangent = cross(out.worldNormal, out.worldTangent) * tangent.w;
    out.uv = uv;
    return out;
}

const PI = 3.14159265359;
const EPSILON = 0.0001;
const MIN_ROUGHNESS = 0.04;
const SHADOW_BIAS = 0.001;

fn distributionGgx(alphaSquared: f32, noH: f32) -> f32 {
    let denominator = (noH * alphaSquared - noH) * noH + 1.0;
    return alphaSquared / (PI * denominator * denominator);
}

fn visibilityGgx(alphaSquared: f32, noV: f32, noL: f32) -> f32 {
    let view = noV + sqrt((noV - noV * alphaSquared) * noV + alphaSquared);
    let light = noL + sqrt((noL - noL * alphaSquared) * noL + alphaSquared);
    return 1.0 / max(view * light, EPSILON);
}

fn fresnelSchlick(f0: vec3f, cosine: f32) -> vec3f {
    let factor = pow(1.0 - cosine, 5.0);
    return factor + f0 * (1.0 - factor);
}

fn fresnelSchlickRoughness(f0: vec3f, noV: f32, roughness: f32) -> vec3f {
    let grazing = max(vec3f(1.0 - roughness), f0);
    return f0 + (grazing - f0) * pow(1.0 - noV, 5.0);
}

fn directLight(
    f0: vec3f,
    diffuseColor: vec3f,
    alphaSquared: f32,
    noV: f32,
    normal: vec3f,
    view: vec3f,
    light: vec3f,
    intensity: f32,
    color: vec3f,
) -> vec3f {
    let halfVector = normalize(light + view);
    let noL = max(dot(normal, light), 0.0);
    let noH = max(dot(normal, halfVector), 0.0);
    let voH = max(dot(view, halfVector), 0.0);
    let distribution = distributionGgx(alphaSquared, noH);
    let visibility = visibilityGgx(alphaSquared, noV, noL);
    let fresnel = fresnelSchlick(f0, voH);
    let specular = fresnel * distribution * visibility;
    let diffuse = (vec3f(1.0) - fresnel) * diffuseColor / PI;
    return (diffuse + specular) * noL * intensity * color;
}

fn cubeFaceIndex(direction: vec3f) -> u32 {
    let axis = abs(direction);
    if (axis.x >= axis.y && axis.x >= axis.z) {
        return select(1u, 0u, direction.x > 0.0);
    }
    if (axis.y >= axis.x && axis.y >= axis.z) {
        return select(3u, 2u, direction.y > 0.0);
    }
    return select(5u, 4u, direction.z > 0.0);
}

fn computeShadow(index: u32, worldPosition: vec3f) -> f32 {
    let light = lightData.lights[index];
    let shadowIndex = u32(light.params.z);
    if (light.posOrDir.w > 0.5) {
        let direction = worldPosition - light.posOrDir.xyz;
        if (length(direction) > max(light.params.x, EPSILON)) { return 1.0; }
        let face = cubeFaceIndex(direction);
        let faceSlot = shadowIndex * 6u + face;
        let clip = pointShadowFaces[faceSlot] * vec4f(worldPosition, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
            return 1.0;
        }
        return textureSampleCompareLevel(
            shadowPoint2D,
            shadowSampler,
            uv,
            i32(faceSlot),
            ndc.z - SHADOW_BIAS,
        );
    }

    let clip = light.viewProj * vec4f(worldPosition, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) {
        return 1.0;
    }
    return textureSampleCompareLevel(
        shadowMap2D,
        shadowSampler,
        uv,
        i32(shadowIndex),
        ndc.z - SHADOW_BIAS,
    );
}

fn rotateEnvironment(direction: vec3f) -> vec3f {
    let sine = environment.rotation.x;
    let cosine = environment.rotation.y;
    return vec3f(
        cosine * direction.x - sine * direction.z,
        direction.y,
        sine * direction.x + cosine * direction.z,
    );
}

fn diffuseIrradiance(normal: vec3f) -> vec3f {
    let n = rotateEnvironment(normal);
    var irradiance = environment.sh[0].rgb * 0.282095;
    irradiance += environment.sh[1].rgb * (0.488603 * n.y);
    irradiance += environment.sh[2].rgb * (0.488603 * n.z);
    irradiance += environment.sh[3].rgb * (0.488603 * n.x);
    irradiance += environment.sh[4].rgb * (1.092548 * n.x * n.y);
    irradiance += environment.sh[5].rgb * (1.092548 * n.y * n.z);
    irradiance += environment.sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0));
    irradiance += environment.sh[7].rgb * (1.092548 * n.x * n.z);
    irradiance += environment.sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(irradiance, vec3f(0.0));
}

fn environmentLight(
    normal: vec3f,
    view: vec3f,
    noV: f32,
    baseColor: vec3f,
    metallic: f32,
    roughness: f32,
    f0: vec3f,
) -> vec3f {
    let fresnel = fresnelSchlickRoughness(f0, noV, roughness);
    let diffuseWeight = (vec3f(1.0) - fresnel) * (1.0 - metallic);
    let diffuse = diffuseIrradiance(normal) * baseColor * diffuseWeight;

    let reflection = rotateEnvironment(reflect(-view, normal));
    let prefiltered = textureSampleLevel(
        environmentSpecular,
        environmentSampler,
        reflection,
        roughness * environment.params.w,
    ).rgb;
    let dfg = textureSample(
        environmentBrdfLut,
        environmentSampler,
        vec2f(noV, roughness),
    ).rg;
    let specular = prefiltered * (f0 * dfg.x + dfg.y);
    return environment.params.x * (
        diffuse * environment.params.y + specular * environment.params.z
    );
}

fn toneMapAces(color: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(color: vec3f) -> vec3f {
    let safe = max(color, vec3f(0.0));
    let low = safe * 12.92;
    let high = 1.055 * pow(safe, vec3f(1.0 / 2.4)) - 0.055;
    return clamp(select(high, low, safe <= vec3f(0.0031308)), vec3f(0.0), vec3f(1.0));
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let sampledBaseColor = material.baseColor * textureSample(baseColorTexture, materialSampler, in.uv);
    let sampledMetalRough = textureSample(metalRoughTexture, materialSampler, in.uv);
    let metallic = clamp(material.matParams.x * sampledMetalRough.b, 0.0, 1.0);
    let roughness = clamp(material.matParams.y * sampledMetalRough.g, MIN_ROUGHNESS, 1.0);
    let ao = clamp(material.matParams.z * textureSample(occlusionTexture, materialSampler, in.uv).r, 0.0, 1.0);

    var normal = normalize(in.worldNormal);
    let tangentNormal = textureSample(normalTexture, materialSampler, in.uv).rgb * 2.0 - 1.0;
    normal = normalize(
        normalize(in.worldTangent) * tangentNormal.x
        + normalize(in.worldBitangent) * tangentNormal.y
        + normal * tangentNormal.z
    );

    let view = normalize(camera.camPos.xyz - in.worldPosition);
    let noV = max(dot(normal, view), EPSILON);
    let alpha = roughness * roughness;
    let alphaSquared = alpha * alpha;
    let f0 = mix(vec3f(0.04), sampledBaseColor.rgb, metallic);
    let diffuseColor = sampledBaseColor.rgb * (1.0 - metallic);

    var color = environmentLight(
        normal,
        view,
        noV,
        sampledBaseColor.rgb,
        metallic,
        roughness,
        f0,
    ) * ao;

    let lightCount = u32(lightData.count.x);
    for (var i = 0u; i < lightCount; i = i + 1u) {
        let light = lightData.lights[i];
        var shadow = 1.0;
        if (light.params.y > 0.5 && material.matParams.w > 0.5) {
            shadow = computeShadow(i, in.worldPosition);
        }
        if (light.posOrDir.w > 0.5) {
            let toLight = light.posOrDir.xyz - in.worldPosition;
            let distance = length(toLight);
            let direction = toLight / max(distance, EPSILON);
            let normalizedDistance = distance / max(light.params.x, EPSILON);
            let attenuation = clamp(1.0 - normalizedDistance * normalizedDistance, 0.0, 1.0);
            color += directLight(
                f0, diffuseColor, alphaSquared, noV, normal, view, direction,
                light.color.w, light.color.rgb,
            ) * attenuation * shadow;
        } else {
            color += directLight(
                f0, diffuseColor, alphaSquared, noV, normal, view,
                normalize(-light.posOrDir.xyz), light.color.w, light.color.rgb,
            ) * shadow;
        }
    }

    color += material.emissive.rgb * textureSample(emissiveTexture, materialSampler, in.uv).rgb;
    return vec4f(linearToSrgb(toneMapAces(color)), sampledBaseColor.a);
}
