import { EnginePlugin, type PluginContext, type ComponentDef } from '@shaderlab/api';
import { OrbitSystem } from './OrbitSystem.ts';
import { OrbitCameraSystem } from './OrbitCameraSystem.ts';

/**
 * Demo plugin: a self-contained capability (component schema + system behavior)
 * living entirely in public/plugins/orbit/. The engine has no compile-time
 * knowledge of it.
 *
 * Two capabilities:
 *  1. 'orbit' system + OrbitComponent — time-based auto-orbit for entities
 *     (demo8: spins an entity around the origin on a time-driven schedule).
 *  2. 'orbitCamera' system + OrbitCameraComponent — mouse-driven orbit camera
 *     controller (replaces the 5 pasted orbit.js scripts in demo3/5/6/7).
 *
 * Apps opt in via app.json `"plugins": ["orbit"]` and list the systems they
 * need in their systems.json order.
 */
export default class OrbitPlugin extends EnginePlugin {
    readonly meta = { id: 'orbit' };

    components: ComponentDef[] = [
        {
            name: 'OrbitComponent',
            fields: {
                radius: { type: 'f32', default: 2.5 },
                speed: { type: 'f32', default: 0.8 },
                spin: { type: 'f32', default: 1.5 },
            },
        },
        {
            name: 'OrbitCameraComponent',
            fields: {
                // Spherical state (derived from Transform.position on first frame
                // when initialized=0; then continuously updated by input + update).
                azimuth:     { type: 'f32', default: 0.0 },
                elevation:  { type: 'f32', default: 0.25 },
                distance:   { type: 'f32', default: 5.0 },
                target:     { type: 'vec3', default: [0, 0, 0] },
                // Input speeds.
                rotSpeed:   { type: 'f32', default: 3.0 },
                panSpeed:   { type: 'f32', default: 1.5 },
                zoomSpeed:  { type: 'f32', default: 0.15 },
                // Distance limits.
                minDistance:{ type: 'f32', default: 1.0 },
                maxDistance:{ type: 'f32', default: 50.0 },
                // Viewport filter: 0 = all, -1 = left half (NDC x<0), 1 = right half (x>=0).
                viewport:   { type: 'i32', default: 0 },
                // Runtime input state (managed by system, not set in scene.json).
                prevX:      { type: 'f32', default: 0.0 },
                prevY:      { type: 'f32', default: 0.0 },
                dragButton: { type: 'i32', default: -1 },
                initialized:{ type: 'bool', default: 0 },
            },
        },
    ];

    systemDefs = [
        {
            name: 'orbit',
            source: 'plugin:orbit',
            components: ['Transform', 'OrbitComponent'],
            ubos: [],
            buffers: [
                { name: 'orbitScratch', size: 64, usage: ['storage', 'copy_dst'] },
            ],
            needs: [],
            after: ['animation'],
        },
        {
            name: 'orbitCamera',
            source: 'plugin:orbit',
            components: ['Transform', 'OrbitCameraComponent'],
            ubos: [],
            buffers: [],
            needs: [],
            after: ['input'],
        },
    ];

    setup(ctx: PluginContext): void {
        ctx.registerSystem('orbit', new OrbitSystem());
        ctx.registerSystem('orbitCamera', new OrbitCameraSystem(ctx.eventBus));
    }
}
