import { resourceManager, uniformLayouts, quatRotateVec3 } from '@shaderlab/api';
import type { UniformLayout, Scene } from '@shaderlab/api';

const SHAPE_ENUM: Record<string, number> = { point: 0, sphere: 1, box: 2 };
const FIELD_ENUM: Record<string, number> = { point: 0, vortex: 1, directional: 2, drag: 3 };

const MAX_EMITTERS = 8;

function maxFieldsFromLayout(): number {
    const layout = uniformLayouts.get('particleSim');
    let count = 0;
    while (layout.has(`field${count}_posType`)) count++;
    return count;
}

let MAX_FIELDS = 8;

interface SysGpu {
    capacity: number;
    pool: GPUBuffer;
    dead: GPUBuffer;
    alive: GPUBuffer;
    drawArgs: GPUBuffer;
    simUBO: GPUBuffer;
    emitUBO: GPUBuffer[];
    simBind: GPUBindGroup;
    drawBind: GPUBindGroup;
    emitBind: GPUBindGroup[];
}

/**
 * GPU-driven particle system: owns per-ParticleSystem pools and drives the
 * emit → simulate → indirect-draw pipeline. Emitters and force fields are
 * separate entities that reference their owning system by name.
 *
 * UBO layouts: declared in uniform-layouts.json as "particleEmit" and "particleSim".
 * Particle struct stride: declared as "particleLayout".
 */
export class ParticleManager {
    private systems = new Map<number, SysGpu>();
    private emitAccum = new Map<number, number>();

    private drawArgsReset = new Uint32Array([6, 0, 0, 0]);

    constructor() {
        MAX_FIELDS = maxFieldsFromLayout();
    }

    /** Release every system's GPU buffers and forget the systems (scene reset). */
    clear(): void {
        for (const sys of this.systems.values()) {
            sys.pool.destroy();
            sys.dead.destroy();
            sys.alive.destroy();
            sys.drawArgs.destroy();
            sys.simUBO.destroy();
            for (const ub of sys.emitUBO) ub.destroy();
        }
        this.systems.clear();
        this.emitAccum.clear();
    }

    /** Emit + simulate every system that this entry's query matches. */
    simulate(
        encoder: GPUCommandEncoder,
        scene: Scene,
        emitPipeline: GPUComputePipeline,
        simPipeline: GPUComputePipeline,
        entities: readonly number[],
        dt: number,
        time: number,
        emitTgs: number = 64,
        simTgs: number = 64,
    ): void {
        const device = resourceManager.device;
        for (const sysEid of entities) {
            const capacity = Math.max(1, Math.floor(scene.getField(sysEid, 'ParticleSystemComponent', 'maxParticles') as number ?? 1000));
            const sys = this.ensure(sysEid, capacity);
            const sysName = scene.getField(sysEid, 'NameComponent', 'name') as string ?? '';

            // reset the indirect instance count for this frame
            device.queue.writeBuffer(sys.drawArgs, 0, this.drawArgsReset);

            // ── Emit pass: one dispatch per attached emitter ──
            const emitters = this.collect(scene, sysName, 'EmitterComponent', MAX_EMITTERS);
            for (let k = 0; k < emitters.length; k++) {
                const eEid = emitters[k];
                const emitCount = this.stepEmitCount(scene, eEid, dt, capacity);
                if (emitCount <= 0) continue;

                this.writeEmitUBO(scene, sys.emitUBO[k], eEid, emitCount, time);
                const pass = encoder.beginComputePass();
                pass.setPipeline(emitPipeline);
                pass.setBindGroup(0, sys.emitBind[k]);
                pass.dispatchWorkgroups(Math.ceil(emitCount / emitTgs));
                pass.end();
            }

            // ── Simulate pass: integrate + recycle + build alive list ──
            this.writeSimUBO(scene, sys.simUBO, sysEid, sysName, capacity, dt, time);
            const pass = encoder.beginComputePass();
            pass.setPipeline(simPipeline);
            pass.setBindGroup(0, sys.simBind);
            pass.dispatchWorkgroups(Math.ceil(capacity / simTgs));
            pass.end();
        }
    }

    /** Indirect-draw the alive particles of every matching system. */
    draw(
        pass: GPURenderPassEncoder,
        scene: Scene,
        drawPipeline: GPURenderPipeline,
        entities: readonly number[],
    ): void {
        for (const sysEid of entities) {
            const sys = this.systems.get(sysEid);
            if (!sys) continue;
            pass.setPipeline(drawPipeline);
            pass.setBindGroup(1, sys.drawBind);
            pass.setVertexBuffer(0, resourceManager.getNamedVBO('quad') ?? resourceManager.quadVBO);
            pass.drawIndirect(sys.drawArgs, 0);
        }
    }

    /* ── per-system GPU resources ─────────────────── */

    private ensure(eid: number, capacity: number): SysGpu {
        const existing = this.systems.get(eid);
        if (existing && existing.capacity === capacity) return existing;

        const device = resourceManager.device;
        const particleStride = uniformLayouts.get('particleLayout').byteSize;
        const emitUboBytes = uniformLayouts.get('particleEmit').byteSize;
        const simUboBytes = uniformLayouts.get('particleSim').byteSize;
        existing?.pool.destroy();
        existing?.dead.destroy();
        existing?.alive.destroy();
        existing?.drawArgs.destroy();
        existing?.simUBO.destroy();
        for (const ub of existing?.emitUBO ?? []) ub.destroy();

        const pool = device.createBuffer({
            size: capacity * particleStride,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dead = device.createBuffer({
            size: 16 + capacity * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const alive = device.createBuffer({
            size: capacity * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const drawArgs = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        const simUBO = device.createBuffer({
            size: simUboBytes,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const emitUBO: GPUBuffer[] = [];
        for (let i = 0; i < MAX_EMITTERS; i++) {
            emitUBO.push(device.createBuffer({
                size: emitUboBytes,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }

        // seed the dead-list: count = capacity, indices = [0, 1, ... capacity-1].
        // WGSL layout: { count: u32 @0, indices: array<u32> @4 }
        const init = new Uint32Array(1 + capacity);
        init[0] = capacity;
        for (let i = 0; i < capacity; i++) init[1 + i] = i;
        device.queue.writeBuffer(dead, 0, init.buffer, 0, init.byteLength);

        const simBind = device.createBindGroup({
            layout: resourceManager.namedLayout('particleSim'),
            entries: [
                { binding: 0, resource: { buffer: pool } },
                { binding: 1, resource: { buffer: dead } },
                { binding: 2, resource: { buffer: alive } },
                { binding: 3, resource: { buffer: drawArgs } },
                { binding: 4, resource: { buffer: simUBO } },
            ],
        });
        const drawBind = device.createBindGroup({
            layout: resourceManager.namedLayout('particleDraw'),
            entries: [
                { binding: 0, resource: { buffer: pool } },
                { binding: 1, resource: { buffer: alive } },
            ],
        });
        const emitBind = emitUBO.map(ub => device.createBindGroup({
            layout: resourceManager.namedLayout('particleEmit'),
            entries: [
                { binding: 0, resource: { buffer: pool } },
                { binding: 1, resource: { buffer: dead } },
                { binding: 2, resource: { buffer: ub } },
            ],
        }));

        const sys: SysGpu = { capacity, pool, dead, alive, drawArgs, simUBO, emitUBO, simBind, drawBind, emitBind };
        this.systems.set(eid, sys);
        return sys;
    }

    /* ── uniform packing ──────────────────────────── */

    private stepEmitCount(scene: Scene, eEid: number, dt: number, capacity: number): number {
        const rate = scene.getField(eEid, 'EmitterComponent', 'rate') as number ?? 0;
        const acc = (this.emitAccum.get(eEid) ?? 0) + rate * dt;
        const n = Math.floor(acc);
        this.emitAccum.set(eEid, acc - n);
        return Math.min(n, capacity);
    }

    private writeEmitUBO(scene: Scene, buf: GPUBuffer, eEid: number, emitCount: number, time: number): void {
        const layout = uniformLayouts.get('particleEmit');
        const ab = new ArrayBuffer(layout.byteSize);
        const f = new Float32Array(ab);
        const u = new Uint32Array(ab);

        const rot = this.vec4(scene, eEid, 'rotation', 'Transform', [0, 0, 0, 1]);
        const pos = this.vec3(scene, eEid, 'position', 'Transform', [0, 0, 0]);
        const shape = SHAPE_ENUM[scene.getField(eEid, 'EmitterComponent', 'shape') as string] ?? 0;
        const radius = scene.getField(eEid, 'EmitterComponent', 'radius') as number ?? 0;
        const half = this.vec3(scene, eEid, 'halfExtents', 'EmitterComponent', [0.5, 0.5, 0.5]);
        const dirLocal = this.vec3(scene, eEid, 'direction', 'EmitterComponent', [0, 1, 0]);
        const dir = quatRotateVec3(rot as [number, number, number, number], dirLocal as [number, number, number]);
        const spread = scene.getField(eEid, 'EmitterComponent', 'spread') as number ?? 0;
        const speedMin = scene.getField(eEid, 'EmitterComponent', 'speedMin') as number ?? 1;
        const speedMax = scene.getField(eEid, 'EmitterComponent', 'speedMax') as number ?? 1;
        const lifeMin = scene.getField(eEid, 'EmitterComponent', 'lifeMin') as number ?? 1;
        const lifeMax = scene.getField(eEid, 'EmitterComponent', 'lifeMax') as number ?? 1;
        const startSize = scene.getField(eEid, 'EmitterComponent', 'startSize') as number ?? 0.1;
        const endSize = scene.getField(eEid, 'EmitterComponent', 'endSize') as number ?? 0;
        const gravityScale = scene.getField(eEid, 'EmitterComponent', 'gravityScale') as number ?? 1;
        const sc = this.vec4(scene, eEid, 'startColor', 'EmitterComponent', [1, 1, 1, 1]);
        const ec = this.vec4(scene, eEid, 'endColor', 'EmitterComponent', [1, 1, 1, 0]);

        layout.write(f, 'origin', [pos[0], pos[1], pos[2], 0]);
        layout.write(f, 'direction', [dir[0], dir[1], dir[2], spread]);
        layout.writeU32(u, 'shapeR', shape);
        f[layout.floatOffsetOf('shapeR') + 1] = radius;
        f[layout.floatOffsetOf('shapeR') + 2] = speedMin;
        f[layout.floatOffsetOf('shapeR') + 3] = speedMax;
        layout.write(f, 'half', [half[0], half[1], half[2], gravityScale]);
        layout.write(f, 'life', [lifeMin, lifeMax, startSize, endSize]);
        layout.write(f, 'startColor', sc);
        layout.write(f, 'endColor', ec);
        const infoBase = layout.floatOffsetOf('info');
        u[infoBase] = emitCount >>> 0;
        u[infoBase + 1] = (Math.random() * 0xffffffff) >>> 0;
        f[infoBase + 2] = time;

        resourceManager.device.queue.writeBuffer(buf, 0, ab);
    }

    private writeSimUBO(
        scene: Scene, buf: GPUBuffer, sysEid: number, sysName: string,
        capacity: number, dt: number, time: number,
    ): void {
        const layout = uniformLayouts.get('particleSim');
        const ab = new ArrayBuffer(layout.byteSize);
        const f = new Float32Array(ab);
        const u = new Uint32Array(ab);

        const gravity = this.vec3(scene, sysEid, 'gravity', 'ParticleSystemComponent', [0, -1.5, 0]);
        const drag = scene.getField(sysEid, 'ParticleSystemComponent', 'drag') as number ?? 0;

        const fields = this.collect(scene, sysName, 'ForceFieldComponent', MAX_FIELDS);
        layout.write(f, 'header', [dt, time, drag, 0]);
        u[layout.floatOffsetOf('header') + 3] = fields.length >>> 0;
        layout.write(f, 'gravity', [gravity[0], gravity[1], gravity[2], 0]);
        u[layout.floatOffsetOf('gravity') + 3] = capacity >>> 0;

        for (let i = 0; i < fields.length; i++) {
            const ffEid = fields[i];
            const prefix = `field${i}_`;
            const pos = this.vec3(scene, ffEid, 'position', 'Transform', [0, 0, 0]);
            const rot = this.vec4(scene, ffEid, 'rotation', 'Transform', [0, 0, 0, 1]);
            const type = FIELD_ENUM[scene.getField(ffEid, 'ForceFieldComponent', 'type') as string] ?? 0;
            const strength = scene.getField(ffEid, 'ForceFieldComponent', 'strength') as number ?? 0;
            const radius = scene.getField(ffEid, 'ForceFieldComponent', 'radius') as number ?? 1;
            const falloff = scene.getField(ffEid, 'ForceFieldComponent', 'falloff') as number ?? 1;
            const dirLocal = this.vec3(scene, ffEid, 'direction', 'ForceFieldComponent', [0, 1, 0]);
            const dir = quatRotateVec3(rot as [number, number, number, number], dirLocal as [number, number, number]);

            const posTypeBase = layout.floatOffsetOf(`${prefix}posType`);
            f[posTypeBase] = pos[0]; f[posTypeBase + 1] = pos[1]; f[posTypeBase + 2] = pos[2];
            u[posTypeBase + 3] = type;
            layout.write(f, `${prefix}dirStr`, [dir[0], dir[1], dir[2], strength]);
            layout.write(f, `${prefix}shape`, [radius, falloff, 0, 0]);
        }

        resourceManager.device.queue.writeBuffer(buf, 0, ab);
    }

    /* ── scene helpers ────────────────────────────── */

    private collect(scene: Scene, sysName: string, comp: string, max: number): number[] {
        const out: number[] = [];
        for (const { eid } of scene.getAllEntities()) {
            if (out.length >= max) break;
            if (!scene.hasComponent(eid, comp)) continue;
            if (Number(scene.getField(eid, comp, 'enabled') ?? 1) !== 1) continue;
            const owner = scene.getField(eid, comp, 'system') as string ?? '';
            if (owner !== sysName) continue;
            out.push(eid);
        }
        return out;
    }

    private vec3(scene: Scene, eid: number, field: string, comp: string, def: number[]): number[] {
        const v = scene.getField(eid, comp, field) as number[] | undefined;
        return v ?? def;
    }

    private vec4(scene: Scene, eid: number, field: string, comp: string, def: number[]): number[] {
        const v = scene.getField(eid, comp, field) as number[] | undefined;
        return v ?? def;
    }
}
