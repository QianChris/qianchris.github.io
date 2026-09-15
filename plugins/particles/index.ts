import { EnginePlugin, type PluginContext } from '@shaderlab/api';
import { ParticleManager } from './ParticleManager.ts';
import * as particleHooks from './hooks/particles.ts';

/**
 * GPU particle capability. Publishes the ParticleManager as the 'particles'
 * attachment consumed by the render hooks (common/scripts/render/particles.js:
 * script:particles.simulate / script:particles.draw). Per-app GPU pools are
 * released on app unload.
 *
 * Engine-scoped (listed in engine-config.json `plugins`) — every app can use
 * ParticleSystemComponent entities without extra wiring.
 */
export default class ParticlesPlugin extends EnginePlugin {
    readonly meta = { id: 'particles' };

    /** script:particles.simulate (compute) + script:particles.draw (geometry). */
    renderHooks = {
        'particles.simulate': particleHooks.simulate,
        'particles.draw': particleHooks.draw,
    };

    private manager: ParticleManager | null = null;

    setup(ctx: PluginContext): void {
        this.manager = new ParticleManager();
        ctx.registerAttachment('particles', this.manager);
    }

    appUnloading(): void {
        this.manager?.clear();
    }

    teardown(): void {
        this.manager?.clear();
        this.manager = null;
    }
}
