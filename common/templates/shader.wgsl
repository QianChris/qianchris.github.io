// WGSL shader template — vertex + fragment stage.
// bindLayout: declared in the referencing pipeline.json (frame/object/material).

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs(@location(0) pos: vec3f) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(pos, 1.0);
    out.color = vec4f(1.0, 0.4, 0.3, 1.0);
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
