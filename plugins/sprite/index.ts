import { EnginePlugin, type PluginContext, type ComponentDef } from '@shaderlab/api';
import { SpriteSystem } from './SpriteSystem.ts';

/**
 * Sprite capability plugin: owns SpriteSheetComponent + SpriteAnimationComponent
 * (schema), the 'animation' system (SpriteSystem — advances sprite-sheet frames
 * + lazily loads sheet JSON and textures), and the SpritePipeline + WGSL shader.
 *
 * Engine-scoped (listed in engine-config.json `plugins`) because the 'animation'
 * system name is in common/systems.json — every app's frame order references it,
 * so the system must always be registered. For apps without sprite entities the
 * system is a no-op (defineQuery returns empty).
 *
 * Migrated from core: previously core's AnimationSystem referenced
 * SpriteSheetComponent/SpriteAnimationComponent that core didn't declare — a
 * cross-layer coupling bug. Now the sprite plugin owns both the components and
 * the system that reads them.
 */
export default class SpritePlugin extends EnginePlugin {
    readonly meta = { id: 'sprite', dependencies: ['core'] };

    components: ComponentDef[] = [
        {
            name: 'SpriteSheetComponent',
            fields: {
                sheet:     { type: 'string', default: '' },
                texHandle: { type: 'u32', default: 0 },
                columns:   { type: 'u32', default: 8 },
                rows:      { type: 'u32', default: 9 },
            },
        },
        {
            name: 'SpriteAnimationComponent',
            fields: {
                animation: { type: 'u32', default: 0 },
                row:       { type: 'u32', default: 0 },
                frame:     { type: 'u32', default: 0 },
                elapsed:   { type: 'f32', default: 0.0 },
                playing:   { type: 'bool', default: 1 },
                direction: { type: 'i32', default: 1 },
            },
        },
    ];

    systemDefs = [
        {
            name: 'animation',
            source: 'plugin:sprite',
            components: ['SpriteSheetComponent', 'SpriteAnimationComponent'],
            ubos: [],
            buffers: [],
            needs: [],
        },
    ];

    private system: SpriteSystem | null = null;

    setup(ctx: PluginContext): void {
        this.system = new SpriteSystem();
        this.system.attach(ctx.scene);
        ctx.registerSystem('animation', this.system);
    }

    appLoaded(_ctx: PluginContext, appBase: string): void {
        this.system?.setBaseDir(appBase);
    }

    appUnloading(_ctx: PluginContext): void {
        this.system?.clear();
    }
}
