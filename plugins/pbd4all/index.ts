import { EnginePlugin, type PluginContext, type ComponentDef } from '@shaderlab/api';
import { PbfManager } from './PbfManager.ts';
import * as pbfHooks from './hooks/pbf.ts';

/**
 * PBF (Position Based Fluids) capability plugin — ported from the
 * Particles4All project (C:\qc\Projects\Particles4All), fluid scope.
 *
 * Model / approximation / units:
 *   - SPH density constraint C = rho/rho0 − 1 solved by position projection
 *     with Lagrange multipliers (Macklin & Müller, "Position Based Fluids",
 *     2013). poly6 kernel for density, spiky gradient for the constraint
 *     gradient. Constraint enforced only under compression (C > 0).
 *   - CFM regularization ε = cfmEpsilonRel · denomRest (denomRest calibrated
 *     on the CPU from the rest lattice), SOR over-relaxation ω.
 *   - Akinci tensile-instability fix (sCorr, k scaled by denomRest/iters).
 *   - Akinci psi boundary particles on the static box shell (two-way
 *     pressure, no friction) — CPU precomputed psi, CPU-sorted per seed.
 *   - XSPH viscosity. No surface tension, no rigid-body coupling (later
 *     phases of the port).
 *   - Neighbour search: uniform grid (cell = h = 2·spacing), GPU counting
 *     sort per substep; particle index == sorted slot after the scatter.
 *   - Units: SI (m, kg, s); rest density 1000 (water); fixed timestep
 *     (1/60)/substeps with a time-bank accumulator (work cap 8× substeps).
 *
 * Rendering (this phase): camera-facing billboard spheres with fragment
 * sphere-normal reconstruction + @builtin(frag_depth), speed→colour gradient
 * (deep blue → white), plus a container wireframe.
 *
 * Declares:
 *   - PbfFluidComponent — one fluid per entity (structural fields:
 *     particleCount/box/spacing/restDensity + Transform.position re-seed the
 *     fluid when changed; physics fields are live-tweakable).
 *   - pbd4allParams / pbd4allView / pbd4allBoxParams uniform layouts,
 *     14 bind layouts, 12 compute pipelines + 2 render pipelines.
 *   - 3 render hooks:
 *       pbd4all.simulate (ComputeHook) — fixed-substep PBF loop
 *       pbd4all.draw     (GeometryHook) — billboard particle spheres
 *       pbd4all.box      (GeometryHook) — container wireframe
 *
 * The 'pbd4all' attachment publishes the PbfManager (structural contract).
 *
 * App-scoped: load via app.json "plugins": ["pbd4all"]. Depends on 'core'
 * for the 'frame' bind group (camera UBO) and the Opaque phase.
 */
export default class Pbd4allPlugin extends EnginePlugin {
    readonly meta = { id: 'pbd4all', dependencies: ['core'] };

    components: ComponentDef[] = [
        {
            name: 'PbfFluidComponent',
            fields: {
                particleCount:  { type: 'u32', default: 30000 },
                box:            { type: 'vec3', default: [1.5, 1.0, 1.0] },
                spacing:        { type: 'f32', default: 0.02 },
                restDensity:    { type: 'f32', default: 1000 },
                gravity:        { type: 'f32', default: 9.81 },
                substeps:       { type: 'u32', default: 2 },
                iterations:     { type: 'u32', default: 4 },
                cfmEpsilonRel:  { type: 'f32', default: 0.01 },
                sCorrK:         { type: 'f32', default: 0.1 },
                sCorrDq:        { type: 'f32', default: 0.3 },
                xsphC:          { type: 'f32', default: 0.066 },
                omega:          { type: 'f32', default: 1.03 },
                timeScale:      { type: 'f32', default: 0.578 },
                drawRadiusScale:{ type: 'f32', default: 0.37 },
                speedMax:       { type: 'f32', default: 2.845 },
            },
        },
    ];

    renderHooks = {
        'pbd4all.simulate': pbfHooks.simulate,
        'pbd4all.draw': pbfHooks.draw,
        'pbd4all.box': pbfHooks.box,
    };

    private manager: PbfManager | null = null;

    async init(ctx: PluginContext): Promise<void> {
        const load = async (file: string): Promise<never> => {
            const resp = await fetch(`${ctx.baseUrl}/${file}`);
            const contentType = resp.headers.get('content-type') ?? '';
            if (!resp.ok || contentType.includes('text/html')) {
                throw new Error(`[pbd4all] declaration file missing: ${ctx.baseUrl}/${file}`);
            }
            return await resp.json() as never;
        };
        this.uniformLayouts = await load('uniform-layouts.json');
        this.bindLayouts = await load('bind-layouts.json');
    }

    setup(ctx: PluginContext): void {
        this.manager = new PbfManager();
        ctx.registerAttachment('pbd4all', this.manager);
    }

    appUnloading(): void { this.manager?.clear(); }
    teardown(): void { this.manager?.clear(); this.manager = null; }
}
