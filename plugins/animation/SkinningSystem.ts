import { defineQuery, schemaRegistry, resourceManager } from '@shaderlab/api';
import { mat4MulInto } from '@shaderlab/api';
import type { Scene, FrameContext, System, PluginContext, GpuResourceSet, GltfSkinData } from '@shaderlab/api';

const SET_NAME = 'animation';
const JOINT_MATRICES = 'animation.joint-matrices';
const MAX_JOINTS = 256;

/**
 * Builds the per-skeleton joint-matrix storage buffer each frame:
 *   jointMatrix[i] = jointWorld[i] × inverseBindMatrix[i]
 * where jointWorld comes from GlobalTransform (written by TransformSystem from
 * the Local TRS that AnimationSamplerSystem sampled). The buffer is published
 * via ctx.replaceGpuResourceSet so SkinnedPbrPipeline's @group(3) reads it
 * through `resource:animation.joint-matrix`.
 *
 * System name 'skinning', runs after 'transform' (GlobalTransform ready) and
 * before 'camera'/'render' (so the renderer sees this frame's pose).
 *
 * One buffer is shared across all skeletons in the scene (laid out
 * contiguously); CesiumMan has a single skeleton so this is the common case.
 */
export class SkinningSystem implements System {
    private scene!: Scene;
    private ctx!: PluginContext;
    private initialized = false;
    private query: (w: import('bitecs').World) => readonly number[] = () => [];
    private buffer: GPUBuffer | null = null;
    private staging: Float32Array = new Float32Array(MAX_JOINTS * 16);
    private published = false;
    /** Cached skin data per skeleton entity (resolved from skinAsset). */
    private skins = new Map<number, GltfSkinData>();
    /** Offset (in mat4 units) of each skeleton's joints in the shared buffer. */
    private offsets = new Map<number, number>();

    attach(scene: Scene): void { this.scene = scene; }
    setContext(ctx: PluginContext): void { this.ctx = ctx; }

    clear(): void {
        this.initialized = false;
        this.query = () => [];
        this.skins.clear();
        this.offsets.clear();
    }

    dispose(): void {
        this.buffer?.destroy();
        this.buffer = null;
        if (this.published && this.ctx) {
            try { this.ctx.unregisterGpuResourceSet(SET_NAME); } catch { /* already swept */ }
        }
        this.published = false;
    }

    /** Called from appLoaded: allocate the storage buffer sized to the sum of
     *  all skeleton joint counts, and publish a fallback (identity) set so
     *  SkinnedPbrPipeline can compile before the first frame. */
    ensureBuffer(device: GPUDevice): void {
        if (this.buffer) return;
        this.buffer = device.createBuffer({
            label: 'animation.joint-matrices',
            size: MAX_JOINTS * 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }

    publishFallback(ctx: PluginContext): void {
        if (this.published) return;
        if (!this.buffer) this.ensureBuffer(ctx.device);
        // Identity matrices (1 per joint) so the mesh renders in bind pose
        // before the first SkinningSystem.update writes real data.
        const id = new Float32Array(MAX_JOINTS * 16);
        for (let i = 0; i < MAX_JOINTS; i++) {
            id[i * 16] = 1; id[i * 16 + 5] = 1; id[i * 16 + 10] = 1; id[i * 16 + 15] = 1;
        }
        ctx.device.queue.writeBuffer(this.buffer!, 0, id);
        const set: GpuResourceSet = { [JOINT_MATRICES]: { kind: 'buffer', buffer: this.buffer!, owned: false } };
        ctx.registerGpuResourceSet(SET_NAME, set);
        this.published = true;
    }

    update(_ctx: FrameContext): void {
        if (!this.initialized) {
            this.initialized = true;
            const sc = schemaRegistry.get('SkeletonComponent');
            if (sc) this.query = defineQuery([sc]);
        }
        if (!this.buffer) return;
        const scene = this.scene;
        const buf = this.staging;
        let cursor = 0;
        let any = false;
        for (const eid of this.query(scene.world)) {
            let skin = this.skins.get(eid);
            if (!skin) {
                const skinAsset = scene.getField(eid, 'SkeletonComponent', 'skinAsset') as string;
                if (!skinAsset) continue;
                skin = resourceManager.getSkin(skinAsset) ?? undefined;
                if (!skin) continue;
                this.skins.set(eid, skin);
                this.offsets.set(eid, cursor);
            }
            const off = this.offsets.get(eid) ?? cursor;
            const jointWorld = new Float32Array(16);
            const jointMatrix = new Float32Array(16);
            for (let j = 0; j < skin.joints.length; j++) {
                const jointName = skin.jointNames[j];
                const jointEid = scene.entityKeyMap.get(jointName);
                if (jointEid == null) continue;
                scene.getGlobalMatrix(jointEid, jointWorld);
                // jointMatrix = jointWorld × IBM[j]
                const ibm = skin.inverseBindMatrices.subarray(j * 16, j * 16 + 16);
                mat4MulInto(jointWorld, ibm as unknown as Float32Array, jointMatrix);
                buf.set(jointMatrix, (off + j) * 16);
            }
            cursor = off + skin.joints.length;
            any = true;
        }
        if (any) {
            this.ctx.device.queue.writeBuffer(this.buffer, 0, buf.buffer, 0, buf.byteLength);
            // Replace with the same buffer object (identity-stable → no destroy).
            this.ctx.replaceGpuResourceSet(SET_NAME, {
                [JOINT_MATRICES]: { kind: 'buffer', buffer: this.buffer, owned: false },
            });
        }
    }
}
