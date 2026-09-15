import { EVENT_TYPES } from '@shaderlab/api';
import type { EventBus, FrameContext, System } from '@shaderlab/api';
import type { Scene } from '@shaderlab/api';

/**
 * Mouse-driven orbit camera controller (replaces the 5 pasted orbit.js scripts).
 *
 * Each entity with an OrbitCameraComponent gets its own spherical state
 * (azimuth/elevation/distance/target) and input drag state (prevX/prevY/dragButton),
 * all stored in the component fields. The system subscribes to the shared EventBus
 * once and dispatches input to the matching entities (filtered by `viewport` for
 * multi-view split-screen).
 *
 * Lifecycle: constructed in the orbit plugin's setup(); disposed when the plugin
 * unloads (app switch). dispose() unsubscribes all event handlers.
 */
export class OrbitCameraSystem implements System {
    private bus: EventBus;
    private scene: Scene | null = null;
    private unsubs: (() => void)[] = [];

    constructor(bus: EventBus) {
        this.bus = bus;
        this.unsubs.push(bus.on(EVENT_TYPES.MOUSE_DOWN, this.onDown));
        this.unsubs.push(bus.on(EVENT_TYPES.MOUSE_UP, this.onUp));
        this.unsubs.push(bus.on(EVENT_TYPES.MOUSE_MOVE, this.onMove));
        this.unsubs.push(bus.on(EVENT_TYPES.WHEEL, this.onWheel));
    }

    private inViewport(viewport: number, x: number): boolean {
        if (viewport === 0) return true;
        if (viewport < 0) return x < 0;
        return x >= 0;
    }

    private onDown = (payload: unknown): void => {
        const e = payload as { button: number; x: number; y: number };
        const scene = this.scene;
        if (!scene) return;
        for (const [, eid] of scene.entityKeyMap) {
            if (!scene.hasComponent(eid, 'OrbitCameraComponent')) continue;
            const viewport = Number(scene.getField(eid, 'OrbitCameraComponent', 'viewport') ?? 0);
            if (!this.inViewport(viewport, e.x)) continue;
            scene.setField(eid, 'OrbitCameraComponent', 'dragButton', e.button);
            scene.setField(eid, 'OrbitCameraComponent', 'prevX', e.x);
            scene.setField(eid, 'OrbitCameraComponent', 'prevY', e.y);
        }
    };

    private onUp = (_payload: unknown): void => {
        const scene = this.scene;
        if (!scene) return;
        for (const [, eid] of scene.entityKeyMap) {
            if (!scene.hasComponent(eid, 'OrbitCameraComponent')) continue;
            scene.setField(eid, 'OrbitCameraComponent', 'dragButton', -1);
        }
    };

    private onMove = (payload: unknown): void => {
        const e = payload as { x: number; y: number };
        const scene = this.scene;
        if (!scene) return;
        const MIN_EL = -Math.PI / 2 + 0.05;
        const MAX_EL = Math.PI / 2 - 0.05;

        for (const [, eid] of scene.entityKeyMap) {
            if (!scene.hasComponent(eid, 'OrbitCameraComponent')) continue;
            const dragButton = Number(scene.getField(eid, 'OrbitCameraComponent', 'dragButton') ?? -1);
            if (dragButton < 0) continue;

            const prevX = Number(scene.getField(eid, 'OrbitCameraComponent', 'prevX') ?? 0);
            const prevY = Number(scene.getField(eid, 'OrbitCameraComponent', 'prevY') ?? 0);
            const dx = e.x - prevX;
            const dy = e.y - prevY;
            scene.setField(eid, 'OrbitCameraComponent', 'prevX', e.x);
            scene.setField(eid, 'OrbitCameraComponent', 'prevY', e.y);

            if (dragButton === 0) {
                const rotSpeed = Number(scene.getField(eid, 'OrbitCameraComponent', 'rotSpeed') ?? 3.0);
                let azimuth = Number(scene.getField(eid, 'OrbitCameraComponent', 'azimuth') ?? 0);
                let elevation = Number(scene.getField(eid, 'OrbitCameraComponent', 'elevation') ?? 0.25);
                azimuth -= dx * rotSpeed;
                elevation -= dy * rotSpeed;
                elevation = Math.max(MIN_EL, Math.min(MAX_EL, elevation));
                scene.setField(eid, 'OrbitCameraComponent', 'azimuth', azimuth);
                scene.setField(eid, 'OrbitCameraComponent', 'elevation', elevation);
            } else if (dragButton === 2) {
                const panSpeed = Number(scene.getField(eid, 'OrbitCameraComponent', 'panSpeed') ?? 1.5);
                const distance = Number(scene.getField(eid, 'OrbitCameraComponent', 'distance') ?? 5);
                const azimuth = Number(scene.getField(eid, 'OrbitCameraComponent', 'azimuth') ?? 0);
                const elevation = Number(scene.getField(eid, 'OrbitCameraComponent', 'elevation') ?? 0.25);
                const target = scene.getField(eid, 'OrbitCameraComponent', 'target') as number[] | null;
                let targetX = target?.[0] ?? 0;
                let targetY = target?.[1] ?? 0;
                let targetZ = target?.[2] ?? 0;

                const cel = Math.cos(elevation);
                const sel = Math.sin(elevation);
                const caz = Math.cos(azimuth);
                const saz = Math.sin(azimuth);
                const fwd = [-cel * saz, -sel, -cel * caz];
                const right = normalize(cross(fwd, [0, 1, 0]));
                const up = cross(right, fwd);
                const pan = distance * panSpeed * 0.3;
                targetX += (right[0] * (-dx) + up[0] * (-dy)) * pan;
                targetY += (right[1] * (-dx) + up[1] * (-dy)) * pan;
                targetZ += (right[2] * (-dx) + up[2] * (-dy)) * pan;
                scene.setField(eid, 'OrbitCameraComponent', 'target', [targetX, targetY, targetZ]);
            }
        }
    };

    private onWheel = (payload: unknown): void => {
        const e = payload as { delta: number; x: number };
        const scene = this.scene;
        if (!scene) return;
        for (const [, eid] of scene.entityKeyMap) {
            if (!scene.hasComponent(eid, 'OrbitCameraComponent')) continue;
            const viewport = Number(scene.getField(eid, 'OrbitCameraComponent', 'viewport') ?? 0);
            if (!this.inViewport(viewport, e.x)) continue;
            const zoomSpeed = Number(scene.getField(eid, 'OrbitCameraComponent', 'zoomSpeed') ?? 0.15);
            const minDist = Number(scene.getField(eid, 'OrbitCameraComponent', 'minDistance') ?? 1.0);
            const maxDist = Number(scene.getField(eid, 'OrbitCameraComponent', 'maxDistance') ?? 50.0);
            let distance = Number(scene.getField(eid, 'OrbitCameraComponent', 'distance') ?? 5);
            distance *= 1 + e.delta * zoomSpeed;
            distance = Math.max(minDist, Math.min(maxDist, distance));
            scene.setField(eid, 'OrbitCameraComponent', 'distance', distance);
        }
    };

    update(ctx: FrameContext): void {
        this.scene = ctx.scene;
        for (const [, eid] of ctx.scene.entityKeyMap) {
            if (!ctx.scene.hasComponent(eid, 'OrbitCameraComponent')) continue;

            const initialized = Number(ctx.scene.getField(eid, 'OrbitCameraComponent', 'initialized') ?? 0);
            if (!initialized) {
                const pos = ctx.scene.getField(eid, 'Transform', 'position') as number[] | null;
                if (pos && pos.length >= 3) {
                    const minDist = Number(ctx.scene.getField(eid, 'OrbitCameraComponent', 'minDistance') ?? 1.0);
                    const px = pos[0], py = pos[1], pz = pos[2];
                    const distance = Math.max(minDist, Math.hypot(px, py, pz));
                    const elevation = Math.asin(Math.max(-1, Math.min(1, py / distance)));
                    const azimuth = Math.atan2(px, pz);
                    ctx.scene.setField(eid, 'OrbitCameraComponent', 'distance', distance);
                    ctx.scene.setField(eid, 'OrbitCameraComponent', 'elevation', elevation);
                    ctx.scene.setField(eid, 'OrbitCameraComponent', 'azimuth', azimuth);
                }
                ctx.scene.setField(eid, 'OrbitCameraComponent', 'initialized', 1);
            }

            const azimuth = Number(ctx.scene.getField(eid, 'OrbitCameraComponent', 'azimuth') ?? 0);
            const elevation = Number(ctx.scene.getField(eid, 'OrbitCameraComponent', 'elevation') ?? 0.25);
            const distance = Number(ctx.scene.getField(eid, 'OrbitCameraComponent', 'distance') ?? 5);
            const target = ctx.scene.getField(eid, 'OrbitCameraComponent', 'target') as number[] | null;
            const targetX = target?.[0] ?? 0;
            const targetY = target?.[1] ?? 0;
            const targetZ = target?.[2] ?? 0;

            const cel = Math.cos(elevation);
            const sel = Math.sin(elevation);
            const cx = targetX + cel * Math.sin(azimuth) * distance;
            const cy = targetY + sel * distance;
            const cz = targetZ + cel * Math.cos(azimuth) * distance;

            const ha = azimuth * 0.5;
            const he = elevation * 0.5;
            const sa = Math.sin(ha), ca = Math.cos(ha);
            const se = Math.sin(he), ce = Math.cos(he);

            ctx.scene.setField(eid, 'Transform', 'position', [cx, cy, cz]);
            ctx.scene.setField(eid, 'Transform', 'rotation', [-ca * se, ce * sa, sa * se, ca * ce]);
        }
    }

    dispose(): void {
        for (const unsub of this.unsubs) unsub();
        this.unsubs = [];
        this.scene = null;
    }
}

function cross(a: number[], b: number[]): number[] {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

function normalize(v: number[]): number[] {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
}
