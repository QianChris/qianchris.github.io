import { EnginePlugin, type PluginContext } from '@shaderlab/api';
import { GaussianSplatManager } from './GaussianSplatManager.ts';
import * as splatHooks from './hooks/splatDraw.ts';

/**
 * 3D Gaussian Splatting capability (app-scoped: declared by an app's
 * `"plugins": ["splat"]`). Registers:
 *   - GsComponent schema (ply asset path + splat count)
 *   - the 'gaussianSplat' system (model UBO refresh + per-frame radix sort;
 *     apps place it in systems.json after 'camera')
 *   - the 'splats' attachment consumed by the render hook
 *     (common/scripts/render/splat.js, script:splat.draw)
 * On appLoaded it scans the scene for GsComponent entities and loads their PLY
 * data into app-scoped storage buffers (released on app switch).
 */
export default class SplatPlugin extends EnginePlugin {
    readonly meta = { id: 'splat' };

    components = [
        {
            name: 'GsComponent',
            fields: {
                ply: { type: 'string', default: '' },
                count: { type: 'u32', default: 0 },
            },
        },
    ];

    /** script:splat.draw - instanced splat quad geometry hook. */
    renderHooks = {
        'splat.draw': splatHooks.draw,
    };

    systemDefs = [
        {
            name: 'gaussianSplat',
            source: 'plugin:splat',
            components: ['GsComponent', 'Transform'],
            ubos: [],
            buffers: [],
            needs: [],
            before: ['camera'],
        },
    ];

    private manager: GaussianSplatManager | null = null;

    setup(ctx: PluginContext): void {
        this.manager = new GaussianSplatManager();
        ctx.registerSystem('gaussianSplat', this.manager);
        ctx.registerAttachment('splats', this.manager);
    }

    async appLoaded(ctx: PluginContext, appBase: string): Promise<void> {
        await this.manager?.loadFromScene(ctx.scene, appBase);
    }

    appUnloading(): void {
        this.manager?.dispose();
    }

    teardown(): void {
        this.manager?.dispose();
        this.manager = null;
    }
}
