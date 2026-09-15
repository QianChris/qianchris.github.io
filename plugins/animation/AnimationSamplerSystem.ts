import { defineQuery, schemaRegistry, resourceManager } from '@shaderlab/api';
import type { Scene, FrameContext, System, GltfAnimationData, GltfAnimationSampler } from '@shaderlab/api';

/**
 * Advances AnimationPlayerComponent.time and samples the referenced clip's
 * channels, writing each channel's value into the target joint entity's Local
 * Transform (translation/rotation/scale). TransformSystem (core) then
 * recomputes GlobalTransform; SkinningSystem reads GlobalTransform × IBM to
 * build the joint matrix storage buffer.
 *
 * System name 'skeletal-animation', runs after 'input' (so it sees fresh
 * play/pause state) and before 'transform' (so the same frame's GlobalTransform
 * reflects the sampled pose).
 */
export class AnimationSamplerSystem implements System {
    private scene!: Scene;
    private initialized = false;
    private query: (w: import('bitecs').World) => readonly number[] = () => [];

    attach(scene: Scene): void { this.scene = scene; }

    clear(): void {
        this.initialized = false;
        this.query = () => [];
    }

    update(ctx: FrameContext): void {
        if (!this.initialized) {
            this.initialized = true;
            const apc = schemaRegistry.get('AnimationPlayerComponent');
            if (apc) this.query = defineQuery([apc]);
        }
        const scene = this.scene;
        for (const eid of this.query(scene.world)) {
            const clipName = scene.getField(eid, 'AnimationPlayerComponent', 'clip') as string;
            if (!clipName) continue;
            const anim = resourceManager.getAnimation(clipName);
            if (!anim) continue;
            const playing = Number(scene.getField(eid, 'AnimationPlayerComponent', 'playing') ?? 0);
            const loop = Number(scene.getField(eid, 'AnimationPlayerComponent', 'loop') ?? 1);
            const speed = Number(scene.getField(eid, 'AnimationPlayerComponent', 'speed') ?? 1);
            // Sample from the engine clock (ctx.time) so Timeline scrub
            // (Engine.setFrameTime + stepOnce) drives the pose directly: pause
            // freezes ctx.time → pose holds; play advances ctx.time → animates.
            let time = ctx.time * speed;
            if (anim.duration > 0) {
                if (loop) {
                    time = time % anim.duration;
                } else if (time > anim.duration) {
                    time = anim.duration;
                    if (playing) scene.setField(eid, 'AnimationPlayerComponent', 'playing', 0);
                }
            }
            if (playing) scene.setField(eid, 'AnimationPlayerComponent', 'time', time);
            this.sampleChannels(anim, time, scene);
        }
    }

    private sampleChannels(anim: GltfAnimationData, time: number, scene: Scene): void {
        for (const channel of anim.channels) {
            if (!channel.nodeName) continue;
            const jointEid = scene.entityKeyMap.get(channel.nodeName);
            if (jointEid == null) continue;
            const sampler = anim.samplers[channel.sampler];
            if (!sampler) continue;
            const value = this.sample(sampler, time);
            switch (channel.path) {
                case 'translation':
                    scene.setField(jointEid, 'Transform', 'position', [value[0], value[1], value[2]]);
                    break;
                case 'rotation':
                    scene.setField(jointEid, 'Transform', 'rotation', [value[0], value[1], value[2], value[3]]);
                    break;
                case 'scale':
                    scene.setField(jointEid, 'Transform', 'scale', [value[0], value[1], value[2]]);
                    break;
                // 'weights' (morph targets) not handled yet (Phase 2c).
            }
        }
    }

    /** Sample a sampler's output at `time`. Returns the component array
     *  (3 for translation/scale, 4 for rotation). LINEAR interpolation;
     *  STEP returns the last keyframe; CUBICSPLINE approximated as LINEAR. */
    private sample(sampler: GltfAnimationSampler, time: number): number[] {
        const input = sampler.input;
        const output = sampler.output;
        const c = sampler.components;
        const n = input.length;
        if (n === 0) return new Array(c).fill(0);
        if (time <= input[0]) return [output[0], output[1], output[2], output[3]].slice(0, c);
        if (time >= input[n - 1]) {
            const off = (n - 1) * c;
            return [output[off], output[off + 1], output[off + 2], output[off + 3]].slice(0, c);
        }
        // Binary search for the interval [i, i+1].
        let lo = 0;
        let hi = n - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (input[mid] <= time) lo = mid; else hi = mid;
        }
        const t0 = input[lo];
        const t1 = input[hi];
        const f = t1 > t0 ? (time - t0) / (t1 - t0) : 0;
        const a = lo * c;
        const b = hi * c;
        if (sampler.interpolation === 'STEP') {
            return [output[a], output[a + 1], output[a + 2], output[a + 3]].slice(0, c);
        }
        if (c === 4) {
            // SLERP for rotation.
            return slerpQuat(
                output[a], output[a + 1], output[a + 2], output[a + 3],
                output[b], output[b + 1], output[b + 2], output[b + 3],
                f,
            );
        }
        // LINEAR lerp for vec3.
        return [
            output[a] + (output[b] - output[a]) * f,
            output[a + 1] + (output[b + 1] - output[a + 1]) * f,
            output[a + 2] + (output[b + 2] - output[a + 2]) * f,
        ];
    }
}

function slerpQuat(
    ax: number, ay: number, az: number, aw: number,
    bx: number, by: number, bz: number, bw: number,
    t: number,
): number[] {
    // Normalize inputs (glTF quaternions should be unit, but be safe).
    let la = Math.hypot(ax, ay, az, aw) || 1;
    let lb = Math.hypot(bx, by, bz, bw) || 1;
    let dot = (ax * bx + ay * by + az * bz + aw * bw) / (la * lb);
    // Take the shorter arc.
    let sbx = bx, sby = by, sbz = bz, sbw = bw;
    if (dot < 0) {
        sbx = -bx; sby = -by; sbz = -bz; sbw = -bw;
        dot = -dot;
    }
    if (dot > 0.9995) {
        // Nearly parallel → linear interpolation + renormalize.
        const rx = ax + (sbx - ax) * t;
        const ry = ay + (sby - ay) * t;
        const rz = az + (sbz - az) * t;
        const rw = aw + (sbw - aw) * t;
        const l = Math.hypot(rx, ry, rz, rw) || 1;
        return [rx / l, ry / l, rz / l, rw / l];
    }
    const theta = Math.acos(Math.max(-1, Math.min(1, dot)));
    const sinTheta = Math.sin(theta);
    const sinT = Math.sin(theta * (1 - t)) / sinTheta;
    const sinU = Math.sin(theta * t) / sinTheta;
    return [
        ax / la * sinT + sbx / lb * sinU,
        ay / la * sinT + sby / lb * sinU,
        az / la * sinT + sbz / lb * sinU,
        aw / la * sinT + sbw / lb * sinU,
    ];
}
