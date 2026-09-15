import type { FrameContext, System } from '@shaderlab/api';

/**
 * Orbits an OrbitComponent-tagged entity around the world origin and spins it.
 * A TS class with full type checking — dispatched by the engine each frame in
 * the slot where systems.json places 'orbit'.
 */
export class OrbitSystem implements System {
    update(ctx: FrameContext): void {
        const scene = ctx.scene;
        for (const [, eid] of scene.entityKeyMap) {
            if (!scene.hasComponent(eid, 'OrbitComponent')) continue;
            const radius = Number(scene.getField(eid, 'OrbitComponent', 'radius') ?? 2.5);
            const speed = Number(scene.getField(eid, 'OrbitComponent', 'speed') ?? 0.8);
            const spin = Number(scene.getField(eid, 'OrbitComponent', 'spin') ?? 1.5);
            const t = ctx.time;
            const x = Math.cos(t * speed) * radius;
            const z = Math.sin(t * speed) * radius;
            scene.setField(eid, 'Transform', 'position', [x, 0, z]);

            const angle = t * spin;
            scene.setField(eid, 'Transform', 'rotation', [0, Math.sin(angle * 0.5), 0, Math.cos(angle * 0.5)]);

            // Script systems keep GPU buffer access: write the orbit state into
            // the 'orbitScratch' storage buffer declared in the plugin systemDefs.
            ctx.writeBuffer('orbitScratch', new Float32Array([t, eid, x, z]));
        }
    }
}
