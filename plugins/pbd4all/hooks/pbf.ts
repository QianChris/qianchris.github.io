import type { ComputeHookContext, GeometryHookContext } from '@shaderlab/api';
import type { PbfManager, PbfPipes } from '../PbfManager.ts';

/** Resolve the 12 aux compute pipelines + their workgroup sizes from the
 *  draw pipeline's aux block. Fail-loud when any ref is missing. */
function resolvePipes(ctx: ComputeHookContext): { pipes: PbfPipes; tgs: Record<keyof PbfPipes, number> } | null {
    const keys: (keyof PbfPipes)[] = [
        'predict', 'count', 'scanBlock', 'scanBlocks', 'scanAdd',
        'scatterSlot', 'scatterMove', 'lambda', 'delta',
        'velFromPos', 'xsph', 'finalize',
    ];
    const pipes = {} as Record<keyof PbfPipes, GPUComputePipeline>;
    const tgs = {} as Record<keyof PbfPipes, number>;
    for (const k of keys) {
        const ref = (ctx.aux as Record<string, string> | undefined)?.[k];
        if (!ref) throw new Error(`[pbd4all] aux pipeline '${k}' not declared in PbfDrawPipeline.json renderer.aux`);
        const pipe = ctx.computePipelines.get(ref);
        if (!pipe) throw new Error(`[pbd4all] aux pipeline '${k}' not loaded: ${ref}`);
        pipes[k] = pipe;
        tgs[k] = ctx.getComputeMeta?.(ref)?.workgroupSize ?? 256;
    }
    return { pipes, tgs };
}

export function simulate(encoder: GPUCommandEncoder, ctx: ComputeHookContext): void {
    const mgr = ctx.attachments.pbd4all as PbfManager | undefined;
    if (!mgr) return;
    const resolved = resolvePipes(ctx);
    if (!resolved) return;
    mgr.simulate(encoder, ctx.scene, resolved.pipes, resolved.tgs, ctx.entities, ctx.dt);
}

export function draw(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    const mgr = ctx.attachments.pbd4all as PbfManager | undefined;
    if (!mgr) return;
    mgr.draw(pass, ctx.pipeline, ctx.entities);
}

export function box(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    const mgr = ctx.attachments.pbd4all as PbfManager | undefined;
    if (!mgr) return;
    mgr.drawBox(pass, ctx.pipeline, ctx.entities);
}
