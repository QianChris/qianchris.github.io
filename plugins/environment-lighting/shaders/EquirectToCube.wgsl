struct FaceParams {
    faceData: vec4f,
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
    out.uv = vec2f(
        positions[vertexIndex].x * 0.5 + 0.5,
        0.5 - positions[vertexIndex].y * 0.5,
    );
    return out;
}

const PI = 3.14159265359;

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
    return vec2f(
        atan2(d.z, d.x) / (2.0 * PI) + 0.5,
        acos(clamp(d.y, -1.0, 1.0)) / PI,
    );
}

fn srgbToLinear(color: vec3f) -> vec3f {
    let low = color / 12.92;
    let high = pow((color + vec3f(0.055)) / 1.055, vec3f(2.4));
    return select(high, low, color <= vec3f(0.04045));
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let direction = faceDirection(u32(params.faceData.x), in.uv);
    let encoded = textureSampleLevel(
        sourceEnvironment,
        sourceSampler,
        directionToEquirect(direction),
        0.0,
    ).rgb;
    return vec4f(srgbToLinear(encoded), 1.0);
}
