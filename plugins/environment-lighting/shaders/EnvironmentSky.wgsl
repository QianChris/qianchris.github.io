struct Camera {
    vp: mat4x4f,
    ivp: mat4x4f,
    camPos: vec4f,
};

struct EnvironmentData {
    sh: array<vec4f, 9>,
    params: vec4f,
    rotation: vec4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> environment: EnvironmentData;
@group(1) @binding(1) var environmentSampler: sampler;
@group(1) @binding(2) var environmentSkyRadiance: texture_cube<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) clipPosition: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var out: VertexOutput;
    out.position = vec4f(positions[vertexIndex], 1.0, 1.0);
    out.clipPosition = positions[vertexIndex];
    return out;
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

fn linearToSrgb(color: vec3f) -> vec3f {
    let safe = max(color, vec3f(0.0));
    let low = safe * 12.92;
    let high = 1.055 * pow(safe, vec3f(1.0 / 2.4)) - 0.055;
    return clamp(select(high, low, safe <= vec3f(0.0031308)), vec3f(0.0), vec3f(1.0));
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    if (environment.rotation.z <= 0.0) {
        discard;
    }
    let nearPoint = camera.ivp * vec4f(in.clipPosition, 0.0, 1.0);
    let farPoint = camera.ivp * vec4f(in.clipPosition, 1.0, 1.0);
    let direction = normalize(farPoint.xyz / farPoint.w - nearPoint.xyz / nearPoint.w);
    let radiance = textureSampleLevel(
        environmentSkyRadiance,
        environmentSampler,
        rotateEnvironment(direction),
        0.0,
    ).rgb * environment.rotation.z;
    // The current image provider is display-referred LDR. Do not tone-map it a
    // second time; only restore the output transfer for the unorm swapchain.
    return vec4f(linearToSrgb(radiance), 1.0);
}
