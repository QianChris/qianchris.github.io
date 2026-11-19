import { EnginePlugin, type PluginContext, type ComponentDef } from '@shaderlab/api';
import { PbdManager } from './PbdManager.ts';
import * as pbdHooks from './hooks/pbd.ts';

/**
 * GPU PBD (Position-Based Dynamics) soft-body capability plugin.
 *
 * Uses **Shape Matching** constraints only — no distance constraints. Each
 * particle owns a cluster (itself + every neighbor within Chebyshev radius
 * `clusterRadius`), and clusters overlap so a particle typically belongs to
 * several. The solve pass computes goal positions per cluster via polar
 * decomposition of the covariance matrix, then atomicAdd's corrections into
 * per-particle accumulators (no graph coloring). The integrate pass averages
 * the corrections and applies them. See
 * public/apps/demo9_softBody/wgsl_shape_matching_pbd_atomic.md.
 *
 * Structural data (particle positions, surface indices, cluster topology) is
 * pre-baked into a JSON asset file by generate_softbody.py / .mjs and loaded
 * in appLoaded(). The runtime applies a fresh random rotation each re-seed.
 * gridN / cellSize / clusterRadius are NOT component fields — they're bake
 * parameters of the generator script. Only physics params (gravity, damping,
 * restitution, compliance, solverIterations, mass) are runtime-tweakable.
 *
 * Declares:
 *   - PbdSoftBodyComponent — asset path + physics params
 *   - pbdParams uniform layout, 5 bind layouts (pbdPredict / pbdSolve (7
 *     bindings) / pbdApply (4) / pbdIntegrate (4) / pbdDraw), 4 compute
 *     pipelines + 2 render pipelines
 *   - 4 render hooks:
 *       pbd.simulate (ComputeHook) — predict -> [solve+apply]×iters -> integrate
 *       pbd.draw     (GeometryHook) — surface mesh via storage-buffer vertex lookup
 *       pbd.floor    (GeometryHook) — checker-grid floor quad at y = 0
 *       pbd.debug    (GeometryHook) — cluster structure overlay (toggle: C, cycle: [ ])
 *
 * The 'pbd' attachment publishes a PbdManager so other plugins/scripts could
 * query particle positions via structural contract (not used by this demo).
 *
 * App-scoped: load via app.json "plugins": ["pbd"]. Depends on 'core' for the
 * 'frame' bind group (camera UBO) and the Opaque phase.
 */
export default class PbdPlugin extends EnginePlugin {
    readonly meta = { id: 'pbd', dependencies: ['core'] };

    components: ComponentDef[] = [
        {
            name: 'PbdSoftBodyComponent',
            fields: {
                asset:             { type: 'string', default: 'softbody_asset.json' },
                gravity:           { type: 'f32', default: -9.81 },
                damping:           { type: 'f32', default: 0.995 },
                solverIterations:  { type: 'u32', default: 12 },
                compliance:        { type: 'f32', default: 0.0 },
                restitution:       { type: 'f32', default: 0.3 },
                mass:              { type: 'f32', default: 1.0 },
            },
        },
    ];

    renderHooks = {
        'pbd.simulate': pbdHooks.simulate,
        'pbd.draw': pbdHooks.draw,
        'pbd.floor': pbdHooks.floor,
        'pbd.debug': pbdHooks.debug,
    };

    private manager: PbdManager | null = null;

    async init(ctx: PluginContext): Promise<void> {
        const load = async (file: string): Promise<never> => {
            const resp = await fetch(`${ctx.baseUrl}/${file}`);
            const contentType = resp.headers.get('content-type') ?? '';
            if (!resp.ok || contentType.includes('text/html')) {
                throw new Error(`[pbd] declaration file missing: ${ctx.baseUrl}/${file}`);
            }
            return await resp.json() as never;
        };
        this.uniformLayouts = await load('uniform-layouts.json');
        this.bindLayouts = await load('bind-layouts.json');
    }

    setup(ctx: PluginContext): void {
        this.manager = new PbdManager();
        ctx.registerAttachment('pbd', this.manager);
    }

    async appLoaded(ctx: PluginContext, appBase: string): Promise<void> {
        await this.manager!.loadAsset(ctx.scene, appBase);
    }

    appUnloading(): void { this.manager?.clear(); }
    teardown(): void { this.manager?.clear(); this.manager = null; }
}
