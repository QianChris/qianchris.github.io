import { EnginePlugin, type PluginContext, type FrameContext, type System } from '@shaderlab/api';
import { InputSystem } from './InputSystem.ts';
import { ScriptSystem } from './ScriptSystem.ts';
import { CameraSystem } from './CameraSystem.ts';
import { LightSystem } from './LightSystem.ts';
import { TransformSystem } from './TransformSystem.ts';
import * as paramsHooks from './hooks/params.ts';

/**
 * Core capability plugin: the baseline systems every stock app composes
 * (systems.json order): input / script / camera / light / render.
 * The 'render' system is a thin translation entry — one call into the engine's
 * renderer mechanism (Component → declarative drivers → UBO/SSBO → passes).
 * The 'animation' system is now owned by the 'sprite' plugin (engine-scoped).
 *
 * Engine-scoped: listed first in engine-config.json `plugins` (other plugins
 * may depend on 'core').
 */
export default class CorePlugin extends EnginePlugin {
    readonly meta = { id: 'core' };

    /** System metadata (ubos/buffers/needs) — consumed by BufferRegistry and
     *  order validation; replaces the old common/systems/<name>.json files. */
    systemDefs = [
        { name: 'input', source: 'plugin:core', components: [], ubos: ['timeInput'], buffers: [], needs: [] },
        { name: 'script', source: 'plugin:core', components: ['ScriptComponent'], ubos: [], buffers: [], needs: ['input'] },
        { name: 'transform', source: 'plugin:core', components: ['Transform', 'GlobalTransform'], ubos: [], buffers: [], needs: [] },
        { name: 'camera', source: 'plugin:core', components: ['Camera', 'Transform'], ubos: ['camera'], buffers: [], needs: ['transform'] },
        { name: 'light', source: 'plugin:core', components: ['LightComponent', 'EnvironmentComponent', 'Transform'], ubos: ['light', 'pointShadowFaces'], buffers: [], needs: ['transform'] },
        { name: 'render', source: 'plugin:core', components: [], ubos: [], buffers: [], needs: ['camera', 'light', 'animation', 'gaussianSplat'] },
    ];

    /** Escape-hatch hooks addressable from pipeline JSON as script:params.<fn>. */
    renderHooks = {
        'params.point': paramsHooks.point,
        'params.edge': paramsHooks.edge,
    };

    private script: ScriptSystem | null = null;

    /** Fetch the co-located declaration JSONs into the declaration fields.
     *  Runs before the engine applies declarations (PluginManager order:
     *  init → applyDeclarations → setup). All files fetch in parallel —
     *  over high-latency links a serial chain costs one full RTT per file. */
    async init(ctx: PluginContext): Promise<void> {
        const load = async (file: string): Promise<never> => {
            const resp = await fetch(`${ctx.baseUrl}/${file}`);
            const contentType = resp.headers.get('content-type') ?? '';
            if (!resp.ok || contentType.includes('text/html')) {
                throw new Error(`[core] declaration file missing: ${ctx.baseUrl}/${file}`);
            }
            return await resp.json() as never;
        };
        const [
            components, uniformLayouts, bindLayouts, vertexSlots, vertexInputs,
            samplers, blendPresets, fallbackTextures, vboPresets, meshes,
            renderTargets, phases,
        ] = await Promise.all([
            load('components.json'),
            load('uniform-layouts.json'),
            load('bind-layouts.json'),
            load('vertex-slots.json'),
            load('vertex-inputs.json'),
            load('samplers.json'),
            load('blend-presets.json'),
            load('fallback-textures.json'),
            load('vbo-presets.json'),
            load('meshes.json'),
            load('render-targets.json'),
            load('phases.json'),
        ]);
        this.components = components;
        this.uniformLayouts = uniformLayouts;
        this.bindLayouts = bindLayouts;
        this.vertexSlots = vertexSlots;
        this.vertexInputs = vertexInputs;
        this.samplers = samplers;
        this.blendPresets = blendPresets;
        this.fallbackTextures = fallbackTextures;
        this.vboPresets = vboPresets;
        this.meshes = meshes;
        this.renderTargets = renderTargets;
        this.phases = phases;
    }

    setup(ctx: PluginContext): void {
        const input = new InputSystem(ctx.canvas, ctx.eventBus);
        input.attach();

        this.script = new ScriptSystem(ctx.eventBus, '');
        this.script.attach(ctx.scene);
        this.script.setHooks(ctx.engineConfig.scriptHooks);
        this.script.provide(
            () => ctx.getSystem('physics'),
            () => ctx.canvas.width / Math.max(1, ctx.canvas.height),
        );

        const camera = new CameraSystem();
        camera.attach(ctx.scene);

        const light = new LightSystem();
        light.attach(ctx.scene);

        const transform = new TransformSystem();
        transform.attach(ctx.scene);

        /** Thin render entry: Component data has been translated by the earlier
         *  systems into UBO/attachment state; this hands the frame to the
         *  renderer mechanism (the built-in RenderGraph unless replaced). */
        const render: System = { update: (fctx: FrameContext) => ctx.renderer.update(fctx) };

        ctx.registerSystem('input', input);
        ctx.registerSystem('script', this.script);
        ctx.registerSystem('transform', transform);
        ctx.registerSystem('camera', camera);
        ctx.registerSystem('light', light);
        ctx.registerSystem('render', render);
    }

    appLoaded(_ctx: PluginContext, appBase: string): void {
        // Script assets resolve relative to the active app.
        this.script?.setBaseDir(appBase);
    }

    appUnloading(): void {
        this.script?.clear();
    }
}

