struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    let xy = p[vi];
    var out: VOut;
    out.pos = vec4f(xy, 0.0, 1.0);
    out.uv = xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
    return out;
}

struct Tint { color: vec4f };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: Tint;

@fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
    let c = textureSample(tex, samp, uv);
    let tinted = c.rgb * u.color.rgb;
    return vec4f(mix(c.rgb, tinted, u.color.a), c.a);
}
