// Gaussian splat geometry hook (script:splat.draw).
// Iterates all loaded SplatInstances and draws each one's instanced quads.
// Group 0 (frame / camera UBO) is already bound by the render graph.
// Splats render into the shared "scene" color + "sceneDepth" depth target, so
// opaque meshes (Opaque phase) correctly occlude / are occluded by splats.
import type { GeometryHookContext } from '@shaderlab/api';
import type { GaussianSplatManager } from '../GaussianSplatManager.ts';

export function draw(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    const splats = ctx.attachments.splats as GaussianSplatManager | undefined;
    if (!splats || !splats.ready) return;
    pass.setPipeline(ctx.pipeline);
    splats.forEachReady(inst => {
        const bg = inst.bindGroup();
        if (!bg) return;
        pass.setBindGroup(1, bg);
        pass.draw(6, inst.count);
    });
}
