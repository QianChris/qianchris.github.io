import { defineQuery, schemaRegistry } from '@shaderlab/api';
import type { Scene, FrameContext, System } from '@shaderlab/api';

/**
 * Writes GlobalTransform (world matrix component) for every entity carrying a
 * Transform, composing parent * local via Scene.getModelMatrix (recursive,
 * per-frame cached). Runs after physics/script (which mutate Transform) and
 * before camera/render (which read GlobalTransform). The renderer reads the
 * GlobalTransform component instead of recomputing the parent chain each
 * frame, and the gizmo reads GlobalTransform.col3 for the world position.
 */
export class TransformSystem implements System {
    private scene!: Scene;
    private initialized = false;
    private query: (w: import('bitecs').World) => readonly number[] = () => [];
    private scratch = new Float32Array(16);

    attach(scene: Scene): void { this.scene = scene; }

    update(_ctx: FrameContext): void {
        if (!this.initialized) {
            this.initialized = true;
            const t = schemaRegistry.get('Transform');
            if (t) this.query = defineQuery([t]);
        }
        const scene = this.scene;
        for (const eid of this.query(scene.world)) {
            const m = scene.getModelMatrix(eid, this.scratch);
            scene.writeGlobalMatrix(eid, m);
        }
    }
}
