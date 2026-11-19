struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let pos = array<vec2f, 3>(
        vec2f( 0.0,  0.8),
        vec2f(-0.8, -0.8),
        vec2f( 0.8, -0.8),
    );
    let col = array<vec4f, 3>(
        vec4f(1.0, 0.2, 0.3, 1.0),
        vec4f(0.2, 0.8, 0.3, 1.0),
        vec4f(0.2, 0.4, 1.0, 1.0),
    );
    var out: VertexOutput;
    out.position = vec4f(pos[vi], 0.0, 1.0);
    out.color = col[vi];
    return out;
}

@fragment fn fs(@location(0) color: vec4f) -> @location(0) vec4f {
    return color;
}
