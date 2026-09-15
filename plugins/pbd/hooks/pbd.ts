import type { ComputeHookContext, GeometryHookContext } from '@shaderlab/api';
import type { PbdManager } from '../PbdManager.ts';

export function simulate(encoder: GPUCommandEncoder, ctx: ComputeHookContext): void {
    const pbd = ctx.attachments.pbd as PbdManager | undefined;
    if (!pbd) return;
    const predict = ctx.aux.predict ? ctx.computePipelines.get(ctx.aux.predict) : undefined;
    const solve = ctx.aux.solve ? ctx.computePipelines.get(ctx.aux.solve) : undefined;
    const apply = ctx.aux.apply ? ctx.computePipelines.get(ctx.aux.apply) : undefined;
    const integrate = ctx.aux.integrate ? ctx.computePipelines.get(ctx.aux.integrate) : undefined;
    if (!predict || !solve || !apply || !integrate) return;
    const predictTgs = ctx.getComputeMeta?.(ctx.aux.predict!)?.workgroupSize ?? 64;
    const solveTgs = ctx.getComputeMeta?.(ctx.aux.solve!)?.workgroupSize ?? 64;
    const applyTgs = ctx.getComputeMeta?.(ctx.aux.apply!)?.workgroupSize ?? 256;
    const integrateTgs = ctx.getComputeMeta?.(ctx.aux.integrate!)?.workgroupSize ?? 256;
    pbd.simulate(
        encoder, ctx.scene,
        predict, solve, apply, integrate,
        ctx.entities, ctx.dt, ctx.time,
        predictTgs, solveTgs, applyTgs, integrateTgs,
    );
}

export function draw(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    const pbd = ctx.attachments.pbd as PbdManager | undefined;
    if (!pbd) return;
    pass.setPipeline(ctx.pipeline);
    pbd.draw(pass, ctx.scene, ctx.pipeline, ctx.entities);
}

export function debug(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    const pbd = ctx.attachments.pbd as PbdManager | undefined;
    if (!pbd) return;
    pbd.drawDebug(pass, ctx.pipeline, ctx.entities);
}

export function floor(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    pass.setPipeline(ctx.pipeline);
    pass.draw(6);
}
