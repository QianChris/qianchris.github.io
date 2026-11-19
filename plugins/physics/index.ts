import { EnginePlugin, RAPIER, type PluginContext, type ToolConfig, type ToolContext } from '@shaderlab/api';
import { PhysicsSystem } from './PhysicsSystem.ts';
import { PickTool } from './PickTool.ts';
import * as physicsHooks from './hooks/physicsDebug.ts';

/**
 * Rapier3D physics capability (engine-scoped via engine-config.json `plugins`).
 * Registers:
 *   - the 'physics' system (fixed-step world, Transform sync, collision events)
 *   - the 'physics' attachment (debug-draw buffers read by
 *     common/scripts/render/physics.js, gameplay ray casts via ScriptContext)
 *   - the 'pick' tool type (mouse ray-cast entity picking, tools.json opt-in)
 * The Rapier WASM module initializes in setup() — apps that don't load this
 * plugin never pay the WASM cost.
 */
export default class PhysicsPlugin extends EnginePlugin {
    readonly meta = { id: 'physics' };

    /** System metadata (replaces the old common/systems/physics.json def). */
    systemDefs = [
        {
            name: 'physics',
            source: 'plugin:physics',
            components: ['ColliderComponent', 'RigidBodyComponent', 'PhysicsControllerComponent', 'Transform'],
            ubos: [],
            buffers: [],
            needs: ['script'],
            requires: ['wasm:rapier'],
        },
    ];

    /** script:physics.debug — Rapier debug-line geometry hook. */
    renderHooks = {
        'physics.debug': physicsHooks.debug,
    };

    private system: PhysicsSystem | null = null;

    async setup(ctx: PluginContext): Promise<void> {
        await RAPIER.init();
        this.system = new PhysicsSystem();
        this.system.attach(ctx.scene, ctx.eventBus);
        ctx.registerSystem('physics', this.system);
        ctx.registerAttachment('physics', this.system);
        ctx.registerToolType('pick', (config: ToolConfig, tctx: ToolContext) => new PickTool(config, tctx));
    }

    appUnloading(): void {
        this.system?.reset();
    }

    teardown(): void {
        this.system?.reset();
        this.system = null;
    }
}
