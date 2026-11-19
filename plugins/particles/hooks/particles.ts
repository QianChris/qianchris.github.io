// GPU-driven particle system hooks.
//   script:particles.simulate — compute stage (emit + simulate)
//   script:particles.draw     — geometry stage (indirect instanced billboards)
// aux keys (declared in ParticlePipeline.json renderer.aux):
//   emit     — particles:pipelines/ParticleEmitPipeline.json
//   simulate — particles:pipelines/ParticleSimPipeline.json
import type { GeometryHookContext, ComputeHookContext } from '@shaderlab/api';
import type { ParticleManager } from '../ParticleManager.ts';

export function simulate(encoder: GPUCommandEncoder, ctx: ComputeHookContext): void {
    const particles = ctx.attachments.particles as ParticleManager | undefined;
    if (!particles) return;
    const emit = ctx.aux.emit ? ctx.computePipelines.get(ctx.aux.emit) : undefined;
    const sim = ctx.aux.simulate ? ctx.computePipelines.get(ctx.aux.simulate) : undefined;
    if (!emit || !sim) return;
    const emitTgs = ctx.getComputeMeta?.(ctx.aux.emit!)?.workgroupSize ?? 64;
    const simTgs = ctx.getComputeMeta?.(ctx.aux.simulate!)?.workgroupSize ?? 64;
    particles.simulate(encoder, ctx.scene, emit, sim, ctx.entities, ctx.dt, ctx.time, emitTgs, simTgs);
}

export function draw(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    const particles = ctx.attachments.particles as ParticleManager | undefined;
    if (!particles) return;
    pass.setPipeline(ctx.pipeline);
    particles.draw(pass, ctx.scene, ctx.pipeline, ctx.entities);
}
