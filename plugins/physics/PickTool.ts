import { mat4TransformVec4, EVENT_TYPES } from '@shaderlab/api';
import type { SceneTool, ToolConfig, ToolContext } from '@shaderlab/api';
import type { PhysicsSystem } from './PhysicsSystem.ts';

interface MouseEventPayload {
    button: number;
    x: number;
    y: number;
}

/**
 * Picks a collider by casting a ray from the mouse through the active camera.
 * On hit, emits `pick` { eid, key, point, distance } on the event bus.
 * Purely optional: only active when configured in tools.json.
 */
export class PickTool implements SceneTool {
    private ctx: ToolContext;
    private button: number;
    private selectEvent: string;
    private unsubscribe: (() => void) | null = null;

    constructor(config: ToolConfig, ctx: ToolContext) {
        this.ctx = ctx;
        this.button = typeof config.button === 'number' ? config.button : 0;
        this.selectEvent = typeof config.selectEvent === 'string' ? config.selectEvent : 'pick';
    }

    attach(): void {
        this.unsubscribe = this.ctx.bus.on(EVENT_TYPES.MOUSE_DOWN, payload => this.onMouseDown(payload as MouseEventPayload));
    }

    detach(): void {
        this.unsubscribe?.();
        this.unsubscribe = null;
    }

    private onMouseDown(payload: MouseEventPayload): void {
        if (payload.button !== this.button) return;

        const cam = this.ctx.scene.getActiveCamera(this.ctx.getAspect());
        if (!cam) return;

        // unproject NDC near/far points through inverse view-projection
        const near = mat4TransformVec4(cam.ivp, [payload.x, payload.y, 0, 1]);
        const far = mat4TransformVec4(cam.ivp, [payload.x, payload.y, 1, 1]);
        const n: [number, number, number] = [near[0] / near[3], near[1] / near[3], near[2] / near[3]];
        const f: [number, number, number] = [far[0] / far[3], far[1] / far[3], far[2] / far[3]];
        const dir: [number, number, number] = [f[0] - n[0], f[1] - n[1], f[2] - n[2]];

        const hit = this.ctx.getSystem<PhysicsSystem>('physics')?.castRay(n, dir);
        if (!hit) return;
        this.ctx.bus.emit(this.selectEvent, hit);
    }
}
