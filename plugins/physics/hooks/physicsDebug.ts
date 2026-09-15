// Physics debug geometry hook (script:physics.debug).
// Draws the Rapier debug line buffers straight from the physics system.
// Group 0 (frame) is already bound by the render graph.
import type { GeometryHookContext } from '@shaderlab/api';
import type { PhysicsSystem } from '../PhysicsSystem.ts';

export function debug(pass: GPURenderPassEncoder, ctx: GeometryHookContext): void {
    const physics = ctx.attachments.physics as PhysicsSystem | undefined;
    if (!physics) return;
    const count = physics.debugVertexCount;
    if (count === 0 || !physics.debugPosBuffer || !physics.debugColBuffer) return;
    pass.setPipeline(ctx.pipeline);
    pass.setVertexBuffer(0, physics.debugPosBuffer);
    pass.setVertexBuffer(1, physics.debugColBuffer);
    pass.draw(count);
}
