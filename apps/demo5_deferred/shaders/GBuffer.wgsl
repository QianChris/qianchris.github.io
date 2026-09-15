struct PerEntity {
    model:        mat4x4f,
    normalMatrix: mat3x3f,
    entityId:     u32,
};

struct Camera {
    vp:     mat4x4f,
    ivp:    mat4x4f,
    camPos: vec4f,
};

struct Material {
    baseColor: vec4f,
    matParams: vec4f,  // x=metallic, y=roughness, z=ao, w=shadowReceive
    emissive:  vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> u: PerEntity;

@group(2) @binding(0) var<uniform> mat: Material;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var texBaseColor: texture_2d<f32>;
@group(2) @binding(3) var texMetalRough: texture_2d<f32>;
@group(2) @binding(4) var texOcclusion: texture_2d<f32>;
@group(2) @binding(6) var texNormal: texture_2d<f32>;

struct VSOutput {
    @builtin(position) pos: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) worldNormal: vec3f,
    @location(2) uv: vec2f,
};

@vertex
fn vs(
    @location(0) inPos: vec3f,
    @location(1) inNormal: vec3f,
    @location(2) inUv: vec2f,
) -> VSOutput {
    let world = u.model * vec4f(inPos, 1.0);
    var out: VSOutput;
    out.pos = cam.vp * world;
    out.worldPos = world.xyz;
    out.worldNormal = normalize(u.normalMatrix * inNormal);
    out.uv = inUv;
    return out;
}

const MIN_ROUGHNESS: f32 = 0.04;

// GBuffer layout (4 MRT, all rgba16float):
//   A = worldPos.xyz + metallic
//   B = worldNormal.xyz + roughness
//   C = albedo.rgb + ao            (ao==0 marks background, written by the clear)
//   D = emissive.rgb + 0
struct GBufferOut {
    @location(0) a: vec4f,
    @location(1) b: vec4f,
    @location(2) c: vec4f,
    @location(3) d: vec4f,
};

@fragment
fn fs(in: VSOutput) -> GBufferOut {
    let baseColor = mat.baseColor * textureSample(texBaseColor, samp, in.uv);
    let metalRough = textureSample(texMetalRough, samp, in.uv);
    let occlusion = textureSample(texOcclusion, samp, in.uv).r;

    let metallic = mat.matParams.x * metalRough.b;
    let roughness = max(mat.matParams.y * metalRough.g, MIN_ROUGHNESS);
    let ao = mat.matParams.z * occlusion;

    var o: GBufferOut;
    o.a = vec4f(in.worldPos, metallic);
    o.b = vec4f(normalize(in.worldNormal), roughness);
    o.c = vec4f(baseColor.rgb, ao);
    o.d = vec4f(mat.emissive.rgb, 0.0);
    return o;
}
