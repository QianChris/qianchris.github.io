import { hasComponent, schemaRegistry, resourceManager, uniformLayouts, mat4LookAt, mat4Mul, mat4OrthographicSym, mat4Perspective, quatRotateVec3 } from '@shaderlab/api';
import type { UniformLayout, Scene, FrameContext, System } from '@shaderlab/api';

const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function maxLightsFromLayout(): number {
    const layout = uniformLayouts.get('light');
    let count = 0;
    while (layout.has(`light${count}_posOrDir`)) count++;
    return count;
}

let MAX_LIGHTS = 4;

/** Six face orientations for a point light's shadow: +X, -X, +Y, -Y, +Z, -Z.
 *  Each entry is [forward, up]; up is any non-parallel vector for that face.
 *  Point shadows use a 2d-array (6 layers per light) rather than a cube texture,
 *  so these orientations are only ever compared against our own PBR face
 *  selection — there is no WebGPU cube-face convention to match. */
const CUBE_FACE_AXES: ReadonlyArray<readonly [readonly [number, number, number], readonly [number, number, number]]> = [
    [[1, 0, 0], [0, 1, 0]],
    [[-1, 0, 0], [0, 1, 0]],
    [[0, 1, 0], [0, 0, -1]],
    [[0, -1, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0]],
    [[0, 0, -1], [0, 1, 0]],
];

/** One shadow render pass to be executed by the RenderGraph shadow phase. */
export interface ShadowPassDesc {
    /** Index of the light in the light UBO (light0..light3). */
    lightIdx: number;
    /** true = point light cube face; false = directional single face. */
    isPoint: boolean;
    /** Array layer: directional → 2D-array layer; point → cube index. */
    layer: number;
    /** Cube face 0..5 (point lights only; 0 for directional). */
    face: number;
    /** Depth attachment view for this pass (one layer/face of the shadow textures). */
    view: GPUTextureView;
}

/**
 * Collects LightComponent entities (+ EnvironmentComponent ambient) each frame
 * and uploads them to the shared light UBO (@group(0) @binding(1)).
 *
 * Multi-shadow support: every light with castShadow gets its own shadow map
 * slot. Directional lights render into a 2d-array layer (shadowDepth2D); point
 * lights render a 6-face cube into shadowCube. LightSystem exposes
 * `shadowPassList` each frame so the RenderGraph shadow phase can iterate the
 * per-face render passes.
 *
 * UBO `light` layout (uniform-layouts.json): per light
 *   posOrDir (xyz = dir/pos, w = 0 dir / 1 point), color (rgb, a = intensity),
 *   viewProj (directional shadow VP; identity for point/non-shadow), params
 *   (x = range, y = castShadow, z = shadowMapIndex, w = 0).
 * Point-light 6 face VPs are uploaded to the separate pointShadowFaceUBO
 * (`pointShadowFaces` layout), indexed by shadowMapIndex*6 + face.
 */
export class LightSystem implements System {
    private scene!: Scene;
    private data: Float32Array = new Float32Array(0);
    private faceData: Float32Array = new Float32Array(0);
    private faceLayout: UniformLayout | null = null;
    /** Shadow render passes for the current frame, ordered: directional first
     *  (one per light), then point (six per light). */
    shadowPassList: ShadowPassDesc[] = [];

    attach(scene: Scene): void {
        this.scene = scene;
        MAX_LIGHTS = maxLightsFromLayout();
        this.data = uniformLayouts.get('light').createBuffer();
        this.faceLayout = uniformLayouts.get('pointShadowFaces');
        this.faceData = this.faceLayout.createBuffer();
    }

    update(_ctx: FrameContext): void {
        const scene = this.scene;
        const buf = this.data;
        buf.fill(0);
        const layout = uniformLayouts.get('light');

        const ambient = scene.getEnvironmentAmbient();
        layout.write(buf, 'ambient', ambient);

        const comp = schemaRegistry.get('LightComponent');
        let count = 0;
        let dirShadowCount = 0;
        let pointShadowCount = 0;
        this.shadowPassList = [];

        if (comp) {
            for (const [, eid] of scene.entityKeyMap) {
                if (count >= MAX_LIGHTS) break;
                if (!hasComponent(scene.world, comp, eid)) continue;
                const r = this.writeLight(eid, count, buf, layout, dirShadowCount, pointShadowCount);
                if (r.isShadow) {
                    if (r.isPoint) {
                        for (let face = 0; face < 6; face++) {
                            this.shadowPassList.push({
                                lightIdx: count, isPoint: true,
                                layer: r.shadowMapIndex!, face,
                                view: resourceManager.shadowPoint2DFaceView(r.shadowMapIndex! * 6 + face),
                            });
                        }
                        pointShadowCount++;
                    } else {
                        this.shadowPassList.push({
                            lightIdx: count, isPoint: false,
                            layer: r.shadowMapIndex!, face: 0,
                            view: resourceManager.shadowDepth2DLayerView(r.shadowMapIndex!),
                        });
                        dirShadowCount++;
                    }
                }
                count++;
            }
        }

        // count = [lightCount, dirShadowCount, pointShadowCount, 0]; per-light
        // params.y gates shadow (no single shadowIdx anymore).
        layout.write(buf, 'count', [count, dirShadowCount, pointShadowCount, 0]);

        const device = resourceManager.device;
        device.queue.writeBuffer(resourceManager.lightUBO, 0, buf.buffer, buf.byteOffset, buf.byteLength);
        device.queue.writeBuffer(resourceManager.pointShadowFaceUBO, 0, this.faceData.buffer, this.faceData.byteOffset, this.faceData.byteLength);
    }

    /** Write one light into the UBO by index. Returns shadow metadata. */
    private writeLight(
        eid: number,
        idx: number,
        buf: Float32Array,
        layout: UniformLayout,
        dirShadowIdx: number,
        pointShadowIdx: number,
    ): { isShadow: boolean; isPoint: boolean; shadowMapIndex: number } {
        const scene = this.scene;
        const prefix = `light${idx}_`;

        const type = (scene.getField(eid, 'LightComponent', 'type') as string) ?? 'directional';
        const isPoint = type === 'point';

        const rot = (scene.getField(eid, 'Transform', 'rotation') as number[]) ?? [0, 0, 0, 1];
        const pos = (scene.getField(eid, 'Transform', 'position') as number[]) ?? [0, 0, 0];
        const rLen = Math.hypot(rot[0], rot[1], rot[2], rot[3]) || 1;
        const qn: [number, number, number, number] = [rot[0] / rLen, rot[1] / rLen, rot[2] / rLen, rot[3] / rLen];
        const forward = quatRotateVec3(qn, [0, 0, -1]);
        const fLen = Math.hypot(forward[0], forward[1], forward[2]) || 1;
        const dir: [number, number, number] = [forward[0] / fLen, forward[1] / fLen, forward[2] / fLen];

        const color = (scene.getField(eid, 'LightComponent', 'color') as number[]) ?? [1, 1, 1];
        const intensity = Number(scene.getField(eid, 'LightComponent', 'intensity') ?? 1);
        const range = Number(scene.getField(eid, 'LightComponent', 'range') ?? 20);
        const castShadow = Number(scene.getField(eid, 'LightComponent', 'castShadow') ?? 0) === 1;
        const shadowOrtho = Number(scene.getField(eid, 'LightComponent', 'shadowOrtho') ?? 10);
        const shadowNear = Number(scene.getField(eid, 'LightComponent', 'shadowNear') ?? 0.1);
        const shadowFar = Number(scene.getField(eid, 'LightComponent', 'shadowFar') ?? 50);
        const shadowStrategy = (scene.getField(eid, 'LightComponent', 'shadowStrategy') as string) ?? 'origin-look';

        if (isPoint) {
            layout.write(buf, `${prefix}posOrDir`, [pos[0], pos[1], pos[2], 1]);
        } else {
            layout.write(buf, `${prefix}posOrDir`, [dir[0], dir[1], dir[2], 0]);
        }

        layout.write(buf, `${prefix}color`, [color[0], color[1], color[2], intensity]);

        const shadowMapIndex = castShadow ? (isPoint ? pointShadowIdx : dirShadowIdx) : 0;

        if (castShadow) {
            if (isPoint) {
                // Point light: identity in the light UBO (PBR samples the cube by
                // direction, not by a single VP); the 6 face VPs go to pointShadowFaces.
                layout.write(buf, `${prefix}viewProj`, IDENTITY_MAT4);
                this.writePointShadowFaces([pos[0], pos[1], pos[2]], range, shadowNear, pointShadowIdx);
            } else {
                const vp = this.directionalViewProj(
                    [pos[0], pos[1], pos[2]], dir, shadowOrtho, shadowNear, shadowFar, shadowStrategy);
                layout.write(buf, `${prefix}viewProj`, vp);
            }
        } else {
            layout.write(buf, `${prefix}viewProj`, IDENTITY_MAT4);
        }

        // params = [range, castShadow, shadowMapIndex, 0]
        layout.write(buf, `${prefix}params`, [range, castShadow ? 1 : 0, shadowMapIndex, 0]);

        return { isShadow: castShadow, isPoint, shadowMapIndex };
    }

    /** Write 6 cube-face view-projections for one point light into pointShadowFaces. */
    private writePointShadowFaces(
        pos: [number, number, number],
        range: number,
        near: number,
        pointShadowIdx: number,
    ): void {
        const layout = this.faceLayout!;
        const proj = mat4Perspective(Math.PI / 2, 1, near, range);
        for (let face = 0; face < 6; face++) {
            const [fwd, up] = CUBE_FACE_AXES[face];
            const target: [number, number, number] = [pos[0] + fwd[0], pos[1] + fwd[1], pos[2] + fwd[2]];
            const view = mat4LookAt(pos, target, [up[0], up[1], up[2]]);
            layout.write(this.faceData, `face${pointShadowIdx * 6 + face}`, mat4Mul(proj, view));
        }
    }

    /** Directional light-space view-projection.
     *  - "origin-look": eye placed back along -dir, looks at world origin.
     *  - "follow-entity": eye = pos - dir*dist, target = pos (light follows its entity). */
    private directionalViewProj(
        pos: [number, number, number],
        dir: [number, number, number],
        halfExtent: number,
        near: number,
        far: number,
        strategy: string = 'origin-look',
    ): Float32Array {
        const dist = far * 0.5;
        const up: [number, number, number] = Math.abs(dir[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
        let eye: [number, number, number];
        let target: [number, number, number];
        if (strategy === 'follow-entity') {
            eye = [pos[0] - dir[0] * dist, pos[1] - dir[1] * dist, pos[2] - dir[2] * dist];
            target = pos;
        } else {
            eye = [-dir[0] * dist, -dir[1] * dist, -dir[2] * dist];
            target = [0, 0, 0];
        }
        const view = mat4LookAt(eye, target, up);
        const proj = mat4OrthographicSym(halfExtent, near, far);
        return mat4Mul(proj, view);
    }
}
