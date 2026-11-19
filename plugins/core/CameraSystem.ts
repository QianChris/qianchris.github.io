import { resourceManager, uniformLayouts } from '@shaderlab/api';
import type { Scene, FrameContext, System } from '@shaderlab/api';

const layout = uniformLayouts;
const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/**
 * Collects the active Camera entity each frame and uploads its view-projection
 * matrices to the shared camera UBO (@group(0) @binding(0)).
 *
 * UBO layout: declared in uniform-layouts.json as "camera" (vp, ivp, pos, view, proj).
 *
 * In multi-view mode (render.json `multiView: true`), RenderGraph handles the
 * per-camera UBO uploads itself via its own scratch buffer (one writeBuffer →
 * one submit per camera). This class still writes the primary camera's matrices
 * via `update` so off-pipeline consumers (splat sort, the editor gizmo) see a
 * valid view, and tracks `lastView`/`lastPos` from that primary camera.
 */
export class CameraSystem implements System {
    private scene!: Scene;
    private data: Float32Array = new Float32Array(0);
    /** Last frame's view matrix + position (for splat sort / off-pipeline use). */
    lastView: Float32Array | null = null;
    lastPos: Float32Array | null = null;

    attach(scene: Scene): void {
        this.scene = scene;
        this.data = layout.get('camera').createBuffer();
    }

    update(ctx: FrameContext): void {
        const cam = this.scene.getActiveCamera(ctx.aspect);
        const buf = this.data;
        const camLayout = layout.get('camera');
        if (cam) {
            camLayout.write(buf, 'vp', cam.vp);
            camLayout.write(buf, 'ivp', cam.ivp);
            camLayout.write(buf, 'pos', cam.pos);
            camLayout.write(buf, 'view', cam.view);
            camLayout.write(buf, 'proj', cam.proj);
            this.lastView = cam.view;
            this.lastPos = cam.pos;
        } else {
            buf.fill(0);
            camLayout.write(buf, 'vp', IDENTITY_MAT4);
            camLayout.write(buf, 'ivp', IDENTITY_MAT4);
            camLayout.write(buf, 'view', IDENTITY_MAT4);
            camLayout.write(buf, 'proj', IDENTITY_MAT4);
            this.lastView = null;
            this.lastPos = null;
        }
        const ubo = resourceManager.cameraUBO;
        resourceManager.device.queue.writeBuffer(ubo, 0, buf.buffer, buf.byteOffset, buf.byteLength);
    }
}
