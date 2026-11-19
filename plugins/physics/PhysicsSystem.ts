import { RAPIER, defineQuery, schemaRegistry, resourceManager, EVENT_TYPES } from '@shaderlab/api';
import type { Scene, EventBus, FrameContext, System } from '@shaderlab/api';
import type * as RapierNS from '@dimforge/rapier3d-compat';

interface ControllerParams {
    enabled: boolean;
    gravity: [number, number, number];
    timeStep: number;
    maxSubsteps: number;
    debugDraw: boolean;
    groundEnabled: boolean;
    groundHeight: number;
    groundHalfExtent: number;
    groundThickness: number;
}

interface BodyRecord {
    eid: number;
    body: RapierNS.RigidBody;
    collider: RapierNS.Collider;
    sig: string;
    bodyType: string;
    /** true = this collider is attached to a parent body (attachTo) and has no
     *  body of its own; it must not drive/be driven by this entity's Transform. */
    attached: boolean;
    /** Parent entity key for attached colliders ('' for roots). Used to force a
     *  child to rebuild when its parent root is rebuilt (colliders died with it). */
    attachKey: string;
}

/** RAPIER collider shape builder lookup (option name �?factory). */
type ColliderBuilder = (R: typeof RAPIER, he: number[], radius: number, halfHeight: number, extra?: Record<string, unknown>) => RapierNS.ColliderDesc;
const COLLIDER_BUILDERS: Record<string, ColliderBuilder> = {
    cuboid: (R, he) => R.ColliderDesc.cuboid(he[0], he[1], he[2]),
    ball: (R, _he, radius) => R.ColliderDesc.ball(radius),
    capsule: (R, _he, radius, halfHeight) => R.ColliderDesc.capsule(halfHeight, radius),
    convexHull: (R, _he, _radius, _halfHeight, extra) => {
        const verts = extra?.verts as Float32Array | undefined;
        if (!verts || verts.length < 12) throw new Error('convexHull requires verts (Float32Array, ≥4 points)');
        const desc = R.ColliderDesc.convexHull(verts);
        if (!desc) throw new Error('Rapier convexHull returned null');
        return desc;
    },
};

export interface RayHit {
    eid: number;
    key: string;
    point: [number, number, number];
    distance: number;
}

/** RAPIER rigid-body constructor lookup (option name �?descriptor factory). */
const BODY_DESC_BUILDERS: Record<string, (R: typeof RAPIER) => RapierNS.RigidBodyDesc> = {
    dynamic: (R) => R.RigidBodyDesc.dynamic(),
    fixed: (R) => R.RigidBodyDesc.fixed(),
    kinematicPosition: (R) => R.RigidBodyDesc.kinematicPositionBased(),
    kinematicVelocity: (R) => R.RigidBodyDesc.kinematicVelocityBased(),
};

/**
 * Rapier-backed physics. Reads Transform + ColliderComponent (+ optional
 * RigidBodyComponent), advances a fixed-step world, and writes poses back into
 * Transform. A PhysicsControllerComponent singleton toggles stepping, gravity,
 * an optional ground plane and debug drawing.
 */
export class PhysicsSystem implements System {
    debugVertexCount = 0;
    debugPosBuffer: GPUBuffer | null = null;
    debugColBuffer: GPUBuffer | null = null;

    private scene!: Scene;
    private bus: EventBus | null = null;
    private query!: (w: import('bitecs').World) => readonly number[];
    private world: RapierNS.World | null = null;
    private eventQueue: RapierNS.EventQueue | null = null;
    private records = new Map<number, BodyRecord>();
    private handleToEid = new Map<number, number>();
    private accumulator = 0;

    private groundBody: RapierNS.RigidBody | null = null;
    private groundSig = '';

    private extraColliderHandles = new Map<number, number[]>();

    buildConvexHullDesc(verts: Float32Array): RapierNS.ColliderDesc | null {
        if (verts.length < 12) return null;
        return RAPIER.ColliderDesc.convexHull(verts);
    }

    attachColliderToBody(eid: number, colliderDesc: RapierNS.ColliderDesc): boolean {
        const rec = this.records.get(eid);
        if (!rec || !this.world) return false;
        const collider = this.world.createCollider(colliderDesc, rec.body);
        this.handleToEid.set(collider.handle, eid);
        let handles = this.extraColliderHandles.get(eid);
        if (!handles) { handles = []; this.extraColliderHandles.set(eid, handles); }
        handles.push(collider.handle);
        return true;
    }

    hasBodyRecord(eid: number): boolean {
        return this.records.has(eid);
    }

    attach(scene: Scene, bus?: EventBus): void {
        this.scene = scene;
        this.bus = bus ?? null;
        this.query = defineQuery([schemaRegistry.get('ColliderComponent')!]);
        const g = (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'gravity') as number[]) ?? [0, -9.81, 0];
        this.world = new RAPIER.World({ x: g[0], y: g[1], z: g[2] });
        this.eventQueue = new RAPIER.EventQueue(true);
    }

    /** Remove all bodies from the Rapier world (call when the scene is reset). */
    reset(): void {
        if (!this.world) return;
        // Only remove root bodies: an "attached" collider record shares its parent
        // root's body object (see createRecord), and removeRigidBody(body) already
        // drops every collider attached to that body. Removing the parent and then
        // the attached child's body would double-free the same body �?WASM trap.
        for (const rec of this.records.values()) {
            if (rec.attached) continue;
            this.world.removeRigidBody(rec.body);
        }
        if (this.groundBody) {
            this.world.removeRigidBody(this.groundBody);
            this.groundBody = null;
            this.groundSig = '';
        }
        this.records.clear();
        this.handleToEid.clear();
        this.extraColliderHandles.clear();
        this.accumulator = 0;
        this.debugVertexCount = 0;
    }

    update(ctx: FrameContext): void {
        const dt = ctx.dt;
        const world = this.world;
        if (!world) return;

        const p = this.readController();
        world.gravity = { x: p.gravity[0], y: p.gravity[1], z: p.gravity[2] };

        this.syncGround(world, p);
        this.reconcile(world);
        this.writeTransformsToBodies();

        if (p.enabled) {
            world.timestep = p.timeStep;
            let steps = 0;
            this.accumulator += dt;
            while (this.accumulator >= p.timeStep && steps < p.maxSubsteps) {
                world.step(this.eventQueue ?? undefined);
                this.drainCollisionEvents();
                this.accumulator -= p.timeStep;
                steps++;
            }
            if (steps === p.maxSubsteps) this.accumulator = 0;
        }

        this.writeBodiesToTransforms();
        this.updateDebug(world, p.debugDraw);
    }

    /** Drain Rapier contact-start events and republish them on the EventBus. */
    private drainCollisionEvents(): void {
        const eq = this.eventQueue;
        const bus = this.bus;
        const world = this.world;
        if (!eq || !bus || !world) return;
        eq.drainCollisionEvents((h1, h2, started) => {
            if (!started) return;
            const a = this.handleToEid.get(h1);
            const b = this.handleToEid.get(h2);
            const point = this.contactPoint(world, h1, h2, a, b);
            bus.emit(EVENT_TYPES.COLLISION, {
                a, b,
                keyA: a !== undefined ? this.keyForEid(a) : '',
                keyB: b !== undefined ? this.keyForEid(b) : '',
                point,
            });
        });
    }

    /** Resolve a world-space contact point for a collider pair (fallback: body midpoint). */
    private contactPoint(
        world: RapierNS.World, h1: number, h2: number,
        a: number | undefined, b: number | undefined,
    ): [number, number, number] | null {
        const c1 = world.getCollider(h1);
        const c2 = world.getCollider(h2);
        let point: [number, number, number] | null = null;
        if (c1 && c2) {
            world.contactPair(c1, c2, (manifold) => {
                if (manifold.numSolverContacts() > 0) {
                    const p = manifold.solverContactPoint(0);
                    if (p) point = [p.x, p.y, p.z];
                }
            });
        }
        if (point) return point;
        // fallback: midpoint of the two body centres
        const ra = a !== undefined ? this.records.get(a) : undefined;
        const rb = b !== undefined ? this.records.get(b) : undefined;
        if (ra && rb) {
            const ta = ra.body.translation();
            const tb = rb.body.translation();
            return [(ta.x + tb.x) / 2, (ta.y + tb.y) / 2, (ta.z + tb.z) / 2];
        }
        const rec = ra ?? rb;
        if (rec) { const t = rec.body.translation(); return [t.x, t.y, t.z]; }
        return null;
    }

    /** Cast a world-space ray; returns the closest collider's entity or null. */
    castRay(origin: [number, number, number], dir: [number, number, number]): RayHit | null {
        if (!this.world) return null;
        const ray = new RAPIER.Ray(
            { x: origin[0], y: origin[1], z: origin[2] },
            { x: dir[0], y: dir[1], z: dir[2] },
        );
        const hit = this.world.castRay(ray, 1e6, true);
        if (!hit) return null;
        const eid = this.handleToEid.get(hit.collider.handle);
        if (eid === undefined) return null;
        const key = this.keyForEid(eid);
        const t = hit.timeOfImpact;
        return {
            eid,
            key,
            point: [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t],
            distance: t,
        };
    }

    /** Scalar fields are stored as single-element arrays; unwrap to a number. */
    private num(eid: number, comp: string, field: string, fallback = 0): number {
        const v = this.scene.getField(eid, comp, field);
        const raw = Array.isArray(v) ? v[0] : v;
        return raw == null ? fallback : Number(raw);
    }

    private readController(): ControllerParams {
        const comp = schemaRegistry.get('PhysicsControllerComponent');
        const scene = this.scene;
        if (!comp) return this.defaultParams();
        for (const [, eid] of scene.entityKeyMap) {
            if (!scene.hasComponent(eid, 'PhysicsControllerComponent')) continue;
            const g = scene.getField(eid, 'PhysicsControllerComponent', 'gravity') as number[];
            return {
                enabled: this.num(eid, 'PhysicsControllerComponent', 'enabled') === 1,
                gravity: [g?.[0] ?? 0, g?.[1] ?? -9.81, g?.[2] ?? 0],
                timeStep: this.num(eid, 'PhysicsControllerComponent', 'timeStep') || 1 / 60,
                maxSubsteps: Math.max(1, Math.floor(this.num(eid, 'PhysicsControllerComponent', 'maxSubsteps', 4))),
                debugDraw: this.num(eid, 'PhysicsControllerComponent', 'debugDraw') === 1,
                groundEnabled: this.num(eid, 'PhysicsControllerComponent', 'groundEnabled') === 1,
                groundHeight: this.num(eid, 'PhysicsControllerComponent', 'groundHeight'),
                groundHalfExtent: this.num(eid, 'PhysicsControllerComponent', 'groundHalfExtent', 500),
                groundThickness: this.num(eid, 'PhysicsControllerComponent', 'groundThickness', 0.05),
            };
        }
        return this.defaultParams();
    }

    private defaultParams(): ControllerParams {
        const g = (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'gravity') as number[]) ?? [0, -9.81, 0];
        return {
            enabled: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'enabled') as boolean) ?? true,
            gravity: [g[0], g[1], g[2]],
            timeStep: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'timeStep') as number) ?? 1 / 60,
            maxSubsteps: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'maxSubsteps') as number) ?? 4,
            debugDraw: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'debugDraw') as boolean) ?? false,
            groundEnabled: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'groundEnabled') as boolean) ?? false,
            groundHeight: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'groundHeight') as number) ?? 0,
            groundHalfExtent: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'groundHalfExtent') as number) ?? 500,
            groundThickness: (schemaRegistry.getFieldDefault('PhysicsControllerComponent', 'groundThickness') as number) ?? 0.05,
        };
    }

    private syncGround(world: RapierNS.World, p: ControllerParams): void {
        const sig = p.groundEnabled ? `on|${p.groundHeight}` : 'off';
        if (sig === this.groundSig) return;
        if (this.groundBody) {
            world.removeRigidBody(this.groundBody);
            this.groundBody = null;
        }
        if (p.groundEnabled) {
            const body = world.createRigidBody(
                RAPIER.RigidBodyDesc.fixed().setTranslation(0, p.groundHeight, 0),
            );
            world.createCollider(RAPIER.ColliderDesc.cuboid(p.groundHalfExtent, p.groundThickness, p.groundHalfExtent), body);
            this.groundBody = body;
        }
        this.groundSig = sig;
    }

    private reconcile(world: RapierNS.World): void {
        const scene = this.scene;
        const live = new Set<number>();
        const roots: Array<{ eid: number; sig: string }> = [];
        const attached: Array<{ eid: number; sig: string }> = [];
        const recreatedRootKeys = new Set<string>();

        // Collect changed/new records, split by whether they own a body or attach
        // to a parent. Roots must be (re)created before attached children so the
        // parent body exists when children look it up.
        for (const eid of this.query(scene.world)) {
            const enabled = this.num(eid, 'ColliderComponent', 'enabled') === 1;
            if (!enabled) continue;
            live.add(eid);

            const sig = this.signature(eid);
            const rec = this.records.get(eid);
            if (rec && rec.sig === sig && !recreatedRootKeys.has(rec.attachKey)) continue;
            const isRecreate = !!rec;
            const attachTo = (scene.getField(eid, 'ColliderComponent', 'attachTo') as string) ?? '';
            if (isRecreate && attachTo === '') {
                // A root being rebuilt invalidates all colliders attached to it
                // (Rapier removed them with the old body); mark for forced recreate.
                recreatedRootKeys.add(this.keyForEid(eid));
            }
            if (rec) this.destroyRecord(world, eid);
            if (attachTo !== '') attached.push({ eid, sig });
            else roots.push({ eid, sig });
        }

        for (const r of roots) this.createRecord(world, r.eid, r.sig, false);
        for (const a of attached) this.createRecord(world, a.eid, a.sig, true);

        for (const eid of [...this.records.keys()]) {
            if (!live.has(eid)) this.destroyRecord(world, eid);
        }
    }

    private createRecord(world: RapierNS.World, eid: number, sig: string, attached: boolean): void {
        const scene = this.scene;
        const attachTo = (scene.getField(eid, 'ColliderComponent', 'attachTo') as string) ?? '';

        if (attached) {
            // Attach this collider to a parent body (looked up by entity key).
            // The parent must already have been created as a root this frame.
            const parentEid = scene.entityKeyMap.get(attachTo);
            const parentRec = parentEid !== undefined ? this.records.get(parentEid) : undefined;
            if (!parentRec) return;  // parent missing �?skip until next frame
            const collider = world.createCollider(this.buildColliderDesc(eid), parentRec.body);
            this.records.set(eid, { eid, body: parentRec.body, collider, sig, bodyType: 'fixed', attached: true, attachKey: attachTo });
            this.handleToEid.set(collider.handle, eid);
            return;
        }

        const pos = scene.getField(eid, 'Transform', 'position') as number[] ?? [0, 0, 0];
        const rot = scene.getField(eid, 'Transform', 'rotation') as number[] ?? [0, 0, 0, 1];

        const bodyType = scene.hasComponent(eid, 'RigidBodyComponent')
            ? (scene.getField(eid, 'RigidBodyComponent', 'bodyType') as string) ?? 'dynamic'
            : 'fixed';

        const builder = BODY_DESC_BUILDERS[bodyType] ?? BODY_DESC_BUILDERS.dynamic;
        let desc = builder(RAPIER);
        desc.setTranslation(pos[0], pos[1], pos[2]);
        desc.setRotation({ x: rot[0], y: rot[1], z: rot[2], w: rot[3] });

        if (scene.hasComponent(eid, 'RigidBodyComponent')) {
            desc.setLinearDamping(this.num(eid, 'RigidBodyComponent', 'linearDamping', 0));
            desc.setAngularDamping(this.num(eid, 'RigidBodyComponent', 'angularDamping', 0));
            desc.setGravityScale(this.num(eid, 'RigidBodyComponent', 'gravityScale', 1));
            desc.setCcdEnabled(this.num(eid, 'RigidBodyComponent', 'ccd') === 1);
            if (this.num(eid, 'RigidBodyComponent', 'lockRotation') === 1) desc.lockRotations();
            const lv = scene.getField(eid, 'RigidBodyComponent', 'linearVelocity') as number[] ?? [0, 0, 0];
            const av = scene.getField(eid, 'RigidBodyComponent', 'angularVelocity') as number[] ?? [0, 0, 0];
            if (lv[0] || lv[1] || lv[2]) desc.setLinvel(lv[0], lv[1], lv[2]);
            if (av[0] || av[1] || av[2]) desc.setAngvel({ x: av[0], y: av[1], z: av[2] });
        }

        const body = world.createRigidBody(desc);
        const collider = world.createCollider(this.buildColliderDesc(eid), body);

        this.records.set(eid, { eid, body, collider, sig, bodyType, attached: false, attachKey: '' });
        this.handleToEid.set(collider.handle, eid);
    }

    private buildColliderDesc(eid: number): RapierNS.ColliderDesc {
        const scene = this.scene;
        const shape = (scene.getField(eid, 'ColliderComponent', 'shape') as string) ?? 'cuboid';
        const he = scene.getField(eid, 'ColliderComponent', 'halfExtents') as number[] ?? [0.5, 0.5, 0.5];
        const radius = this.num(eid, 'ColliderComponent', 'radius', 0.5);
        const halfHeight = this.num(eid, 'ColliderComponent', 'halfHeight', 0.5);

        const builder = COLLIDER_BUILDERS[shape] ?? COLLIDER_BUILDERS.cuboid;
        const desc = builder(RAPIER, he, radius, halfHeight);

        // Local transform relative to the parent body: for roots this is identity
        // (offset [0,0,0], rot [0,0,0,1]); for attached colliders it positions the
        // collider on the parent (e.g. a wall offset from the box center).
        const off = scene.getField(eid, 'ColliderComponent', 'offset') as number[] ?? [0, 0, 0];
        const crot = scene.getField(eid, 'ColliderComponent', 'colliderRot') as number[] ?? [0, 0, 0, 1];
        desc.setTranslation(off[0], off[1], off[2]);
        desc.setRotation({ x: crot[0], y: crot[1], z: crot[2], w: crot[3] });

        const layer = this.num(eid, 'ColliderComponent', 'layer', 1) & 0xffff;
        const mask = this.num(eid, 'ColliderComponent', 'mask', 0xffff) & 0xffff;
        desc.setDensity(this.num(eid, 'ColliderComponent', 'density', 1));
        desc.setFriction(this.num(eid, 'ColliderComponent', 'friction', 0.5));
        desc.setRestitution(this.num(eid, 'ColliderComponent', 'restitution', 0));
        desc.setSensor(this.num(eid, 'ColliderComponent', 'isSensor') === 1);
        desc.setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS);
        desc.setCollisionGroups(((layer << 16) | mask) >>> 0);
        return desc;
    }

    private destroyRecord(world: RapierNS.World, eid: number): void {
        const rec = this.records.get(eid);
        if (!rec) return;
        this.handleToEid.delete(rec.collider.handle);
        const extraHandles = this.extraColliderHandles.get(eid);
        if (extraHandles) {
            for (const h of extraHandles) this.handleToEid.delete(h);
            this.extraColliderHandles.delete(eid);
        }
        if (rec.attached) {
            // Remove just this collider; keep the parent body intact.
            world.removeCollider(rec.collider, true);
        } else {
            world.removeRigidBody(rec.body);
        }
        this.records.delete(eid);
    }

    private writeTransformsToBodies(): void {
        const scene = this.scene;
        for (const rec of this.records.values()) {
            if (rec.attached) continue;  // attached colliders follow the parent body
            if (rec.bodyType === 'dynamic' || rec.bodyType === 'kinematicVelocity') continue;
            const pos = scene.getField(rec.eid, 'Transform', 'position') as number[] ?? [0, 0, 0];
            const rot = scene.getField(rec.eid, 'Transform', 'rotation') as number[] ?? [0, 0, 0, 1];
            const t = { x: pos[0], y: pos[1], z: pos[2] };
            const r = { x: rot[0], y: rot[1], z: rot[2], w: rot[3] };
            if (rec.bodyType === 'kinematicPosition') {
                rec.body.setNextKinematicTranslation(t);
                rec.body.setNextKinematicRotation(r);
            } else {
                rec.body.setTranslation(t, true);
                rec.body.setRotation(r, true);
            }
        }
    }

    private writeBodiesToTransforms(): void {
        const scene = this.scene;
        for (const rec of this.records.values()) {
            if (rec.attached) continue;  // attached colliders have no Transform of their own
            if (rec.bodyType === 'fixed' || rec.bodyType === 'kinematicPosition') continue;
            const t = rec.body.translation();
            const r = rec.body.rotation();
            scene.setField(rec.eid, 'Transform', 'position', [t.x, t.y, t.z]);
            scene.setField(rec.eid, 'Transform', 'rotation', [r.x, r.y, r.z, r.w]);
        }
    }

    private updateDebug(world: RapierNS.World, enabled: boolean): void {
        if (!enabled) { this.debugVertexCount = 0; return; }
        const buffers = world.debugRender();
        const vertexCount = buffers.vertices.length / 3;
        this.debugVertexCount = vertexCount;
        if (vertexCount === 0) return;
        this.debugPosBuffer = resourceManager.getStorageBuffer('physicsDebugPos', buffers.vertices.byteLength);
        this.debugColBuffer = resourceManager.getStorageBuffer('physicsDebugCol', buffers.colors.byteLength);
        resourceManager.device.queue.writeBuffer(this.debugPosBuffer, 0, buffers.vertices.buffer, buffers.vertices.byteOffset, buffers.vertices.byteLength);
        resourceManager.device.queue.writeBuffer(this.debugColBuffer, 0, buffers.colors.buffer, buffers.colors.byteOffset, buffers.colors.byteLength);
    }

    private signature(eid: number): string {
        const scene = this.scene;
        const parts: unknown[] = [];
        for (const field of ['shape', 'halfExtents', 'radius', 'halfHeight', 'density', 'friction', 'restitution', 'isSensor', 'layer', 'mask', 'attachTo', 'offset', 'colliderRot']) {
            parts.push(scene.getField(eid, 'ColliderComponent', field));
        }
        if (scene.hasComponent(eid, 'RigidBodyComponent')) {
            for (const field of ['bodyType', 'linearDamping', 'angularDamping', 'gravityScale', 'ccd', 'lockRotation']) {
                parts.push(scene.getField(eid, 'RigidBodyComponent', field));
            }
        } else {
            parts.push('nobody');
        }
        return JSON.stringify(parts);
    }

    private keyForEid(eid: number): string {
        for (const [key, e] of this.scene.entityKeyMap) {
            if (e === eid) return key;
        }
        return '';
    }
}
