struct FaceParams {
    faceRoughness: vec4f,
};

@group(0) @binding(0) var sourceSampler: sampler;
@group(0) @binding(1) var sourceEnvironment: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: FaceParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var out: VertexOutput;
    out.position = vec4f(positions[vertexIndex], 0.0, 1.0);
    // Render-target Y points down; cube face coordinates use V=0 at the top.
    out.uv = vec2f(
        positions[vertexIndex].x * 0.5 + 0.5,
        0.5 - positions[vertexIndex].y * 0.5,
    );
    return out;
}

const PI = 3.14159265359;
const SAMPLE_COUNT = 64u;

fn radicalInverseVdc(bitsInput: u32) -> f32 {
    var bits = bitsInput;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(index: u32, count: u32) -> vec2f {
    return vec2f(f32(index) / f32(count), radicalInverseVdc(index));
}

fn importanceSampleGgx(xi: vec2f, normal: vec3f, roughness: f32) -> vec3f {
    let alpha = max(roughness * roughness, 0.001);
    let phi = 2.0 * PI * xi.x;
    let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    let halfTangent = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(normal.z) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * halfTangent.x + bitangent * halfTangent.y + normal * halfTangent.z);
}

fn faceDirection(face: u32, uv: vec2f) -> vec3f {
    let p = uv * 2.0 - 1.0;
    switch face {
        case 0u: { return normalize(vec3f(1.0, -p.y, -p.x)); }
        case 1u: { return normalize(vec3f(-1.0, -p.y, p.x)); }
        case 2u: { return normalize(vec3f(p.x, 1.0, p.y)); }
        case 3u: { return normalize(vec3f(p.x, -1.0, -p.y)); }
        case 4u: { return normalize(vec3f(p.x, -p.y, 1.0)); }
        default: { return normalize(vec3f(-p.x, -p.y, -1.0)); }
    }
}

fn directionToEquirect(direction: vec3f) -> vec2f {
    let d = normalize(direction);
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2f(u, v);
}

fn srgbToLinear(color: vec3f) -> vec3f {
    let low = color / 12.92;
    let high = pow((color + vec3f(0.055)) / 1.055, vec3f(2.4));
    return select(high, low, color <= vec3f(0.04045));
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let normal = faceDirection(u32(params.faceRoughness.x), in.uv);
    let roughness = params.faceRoughness.y;
    let view = normal;
    var color = vec3f(0.0);
    var weight = 0.0;

    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let halfVector = importanceSampleGgx(hammersley(i, SAMPLE_COUNT), normal, roughness);
        let light = normalize(2.0 * dot(view, halfVector) * halfVector - view);
        let noL = max(dot(normal, light), 0.0);
        if (noL > 0.0) {
            let encoded = textureSampleLevel(
                sourceEnvironment,
                sourceSampler,
                directionToEquirect(light),
                0.0,
            ).rgb;
            color += srgbToLinear(encoded) * noL;
            weight += noL;
        }
    }
    return vec4f(color / max(weight, 0.0001), 1.0);
}
