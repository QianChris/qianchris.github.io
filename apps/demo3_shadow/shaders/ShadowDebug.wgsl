// ShadowMap debug visualization: fullscreen pass sampling the 2D-array shadow depth.
// Binding 3 (texture_depth_2d_array) comes from the frame bind group. Visualizes
// layer 0 (the first directional shadow light) by default.

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

@group(0) @binding(3) var shadowMap: texture_depth_2d_array;

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
    let dims = textureDimensions(shadowMap, 0);
    let tc = clamp(vec2i(vec2f(dims) * in.uv), vec2i(0, 0), vec2i(dims) - 1);
    let depth = textureLoad(shadowMap, tc, 0, 0);  // layer 0, mip 0
    // Visualize: near=black(0), far=white(1). Scale for visibility.
    return vec4f(vec3f(depth), 1.0);
}
