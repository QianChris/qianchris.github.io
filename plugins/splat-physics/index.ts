import { EnginePlugin, type PluginContext, type FrameContext, type System, RAPIER } from '@shaderlab/api';
import { generateSplatCollider, type SplatColliderResult } from './SplatCollider.ts';

interface PhysicsSystemLike {
    hasBodyRecord(eid: number): boolean;
    attachColliderToBody(eid: number, desc: import('@dimforge/rapier3d-compat').ColliderDesc): boolean;
}

/** Structural contract for the gaussianSplat system fields the splat-physics
 *  plugin consumes. Iterate instances (one per GsComponent entity) to read
 *  each entity's eid + splat center positions. */
interface SplatInstanceView {
    eid: number;
    count: number;
    getCenters(): Float32Array | null;
}
interface SplatManagerLike {
    forEachInstance(cb: (inst: SplatInstanceView) => void): void;
}

interface SplatPhysicsState {
    gsEid: number;
    hullDescs: SplatColliderResult | null;
    attached: boolean;
}

class SplatPhysicsSystem implements System {
    private states: SplatPhysicsState[] = [];

    setStates(states: SplatPhysicsState[]): void {
        this.states = states;
    }

    update(ctx: FrameContext): void {
        if (this.states.length === 0) return;
        const physics = ctx.attachments?.physics as PhysicsSystemLike | undefined;
        if (!physics) return;

        for (const state of this.states) {
            if (state.attached) continue;
            if (!physics.hasBodyRecord(state.gsEid)) continue;

            const result = state.hullDescs;
            if (!result || result.colliderDescs.length === 0) {
                state.attached = true;
                continue;
            }

            let attached = 0;
            for (const desc of result.colliderDescs) {
                desc.setDensity(8);
                desc.setFriction(0.9);
                desc.setRestitution(0.05);
                desc.setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS);
                if (physics.attachColliderToBody(state.gsEid, desc)) {
                    attached++;
                }
            }

            state.attached = true;
            state.hullDescs = null;
            console.log(`[splat-physics] attached ${attached} convex hulls to GsEntity ${state.gsEid}`);
        }
    }
}

export default class SplatPhysicsPlugin extends EnginePlugin {
    readonly meta = { id: 'splat-physics', dependencies: ['splat', 'physics'] };

    private system = new SplatPhysicsSystem();

    setup(ctx: PluginContext): void {
        ctx.registerSystem('splatPhysics', this.system);
    }

    async appLoaded(ctx: PluginContext): Promise<void> {
        const splatSys = ctx.getSystem<SplatManagerLike>('gaussianSplat');
        if (!splatSys) {
            console.warn('[splat-physics] gaussianSplat system not found');
            return;
        }

        const states: SplatPhysicsState[] = [];
        splatSys.forEachInstance(inst => {
            const centers = inst.getCenters();
            if (!centers || centers.length === 0) {
                console.warn(`[splat-physics] no splat centers for GsEntity ${inst.eid}`);
                return;
            }

            // PhysicsSystem reads Transform.position + rotation (not scale) into
            // the body pose, so collider vertices must be pre-scaled to match the
            // entity's Transform.scale. Otherwise a scaled-down bicycle would
            // collide with full-size (unscaled) hulls — visually disjoint.
            const scale = ctx.scene.getField(inst.eid, 'Transform', 'scale') as number[] | undefined;
            const sx = scale?.[0] ?? 1, sy = scale?.[1] ?? 1, sz = scale?.[2] ?? 1;
            const scaled = new Float32Array(centers.length);
            for (let i = 0; i < centers.length; i += 4) {
                scaled[i]     = centers[i]     * sx;
                scaled[i + 1] = centers[i + 1] * sy;
                scaled[i + 2] = centers[i + 2] * sz;
            }

            console.log(`[splat-physics] GsEntity ${inst.eid}: generating convex hulls from ${Math.floor(centers.length / 4)} splats (scale ${sx},${sy},${sz})...`);
            const start = performance.now();
            const result = generateSplatCollider(scaled);
            const elapsed = (performance.now() - start).toFixed(0);
            console.log(`[splat-physics] GsEntity ${inst.eid}: generated ${result.hullCount} convex hulls in ${elapsed}ms`);

            ctx.scene.toggleComponent(inst.eid, 'RigidBodyComponent', true);
            ctx.scene.setField(inst.eid, 'RigidBodyComponent', 'bodyType', 'dynamic');
            ctx.scene.setField(inst.eid, 'RigidBodyComponent', 'ccd', 1);
            ctx.scene.setField(inst.eid, 'RigidBodyComponent', 'linearDamping', 0.05);
            ctx.scene.setField(inst.eid, 'RigidBodyComponent', 'angularDamping', 0.3);
            ctx.scene.setField(inst.eid, 'RigidBodyComponent', 'gravityScale', 1);

            ctx.scene.toggleComponent(inst.eid, 'ColliderComponent', true);
            ctx.scene.setField(inst.eid, 'ColliderComponent', 'shape', 'cuboid');
            ctx.scene.setField(inst.eid, 'ColliderComponent', 'halfExtents', [0.01, 0.01, 0.01]);
            ctx.scene.setField(inst.eid, 'ColliderComponent', 'isSensor', 1);
            ctx.scene.setField(inst.eid, 'ColliderComponent', 'density', 0);

            states.push({ gsEid: inst.eid, hullDescs: result, attached: false });
        });

        if (states.length === 0) {
            console.warn('[splat-physics] no splat instances processed');
        }
        this.system.setStates(states);
    }

    appUnloading(): void {
        this.system.setStates([]);
    }
}
