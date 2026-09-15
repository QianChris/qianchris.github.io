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
    out.uv = positions[vertexIndex] * 0.5 + 0.5;
    return out;
}

const PI = 3.14159265359;
const SAMPLE_COUNT = 128u;

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
    return vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

fn geometrySchlickGgx(noX: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let k = alpha * 0.5;
    return noX / (noX * (1.0 - k) + k);
}

fn geometrySmith(normal: vec3f, view: vec3f, light: vec3f, roughness: f32) -> f32 {
    return geometrySchlickGgx(max(dot(normal, view), 0.0), roughness)
        * geometrySchlickGgx(max(dot(normal, light), 0.0), roughness);
}

fn integrateBrdf(noV: f32, roughness: f32) -> vec2f {
    let view = vec3f(sqrt(max(1.0 - noV * noV, 0.0)), 0.0, noV);
    let normal = vec3f(0.0, 0.0, 1.0);
    var scale = 0.0;
    var bias = 0.0;
    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let halfVector = importanceSampleGgx(hammersley(i, SAMPLE_COUNT), normal, roughness);
        let light = normalize(2.0 * dot(view, halfVector) * halfVector - view);
        let noL = max(light.z, 0.0);
        let noH = max(halfVector.z, 0.0);
        let voH = max(dot(view, halfVector), 0.0);
        if (noL > 0.0) {
            let geometry = geometrySmith(normal, view, light, roughness);
            let visibility = geometry * voH / max(noH * noV, 0.0001);
            let fresnel = pow(1.0 - voH, 5.0);
            scale += (1.0 - fresnel) * visibility;
            bias += fresnel * visibility;
        }
    }
    return vec2f(scale, bias) / f32(SAMPLE_COUNT);
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    // Render-target Y is opposite sampled texture V, so flip while baking.
    let integrated = integrateBrdf(
        clamp(in.uv.x, 0.001, 0.999),
        clamp(1.0 - in.uv.y, 0.001, 0.999),
    );
    return vec4f(integrated, 0.0, 1.0);
}
