import {
    EnginePlugin,
    defineQuery,
    schemaRegistry,
    type BindLayoutDecls,
    type ComponentDef,
    type PluginContext,
    type SystemDef,
} from '@shaderlab/api';
import { buildEnvironment, runtimeSourceWidth } from './EnvironmentProcessor.ts';
import {
    StaticImageEnvironmentSourceProvider,
} from './EnvironmentSourceProvider.ts';
import {
    createNeutralEnvironmentResourceSet,
    destroyUnpublishedEnvironmentResourceSet,
    ENVIRONMENT_RESOURCE_SET_NAME,
    environmentResourceDescriptors,
    type EnvironmentResourceSet,
} from './EnvironmentResourceSet.ts';
import {
    EnvironmentLightingSystem,
    readEnvironmentSettings,
    type EnvironmentSettings,
} from './EnvironmentLightingSystem.ts';

export default class EnvironmentLightingPlugin extends EnginePlugin {
    readonly meta = { id: 'environment-lighting', dependencies: ['core'] };

    bindLayouts: BindLayoutDecls = {
        environmentLighting: {
            entries: [
                { binding: 0, visibility: ['fragment'], buffer: 'uniform' },
                { binding: 1, visibility: ['fragment'], sampler: 'filtering' },
                { binding: 2, visibility: ['fragment'], texture: 'float', viewDimension: 'cube' },
                { binding: 3, visibility: ['fragment'], texture: 'float', viewDimension: '2d' },
            ],
        },
        environmentSky: {
            entries: [
                { binding: 0, visibility: ['fragment'], buffer: 'uniform' },
                { binding: 1, visibility: ['fragment'], sampler: 'filtering' },
                { binding: 2, visibility: ['fragment'], texture: 'float', viewDimension: 'cube' },
            ],
        },
    };

    systemDefs: SystemDef[] = [{
        name: 'environmentLighting',
        source: 'plugin:environment-lighting',
        components: ['EnvironmentLightComponent'],
        ubos: [],
        buffers: [],
        needs: [],
        after: ['light'],
        before: ['render'],
    }];

    private system = new EnvironmentLightingSystem(
        settings => this.handleComponentChange(settings),
    );
    private sourceProvider = new StaticImageEnvironmentSourceProvider();
    private published = false;
    private appContext: PluginContext | null = null;
    private appBase = '';
    private appGeneration = 0;
    private resourceGeneration = 0;

    async init(ctx: PluginContext): Promise<void> {
        const url = `${ctx.baseUrl}/components.json`;
        const response = await fetch(url);
        const contentType = response.headers.get('content-type') ?? '';
        if (!response.ok || contentType.includes('text/html')) {
            throw new Error(`[environment-lighting] declaration file missing: ${url}`);
        }
        this.components = await response.json() as ComponentDef[];
    }

    setup(ctx: PluginContext): void {
        ctx.registerSystem('environmentLighting', this.system);
    }

    async appLoaded(ctx: PluginContext, appBase: string): Promise<void> {
        this.appContext = ctx;
        this.appBase = appBase;
        const generation = ++this.appGeneration;
        const resourceGeneration = ++this.resourceGeneration;
        this.publishFallback(ctx);
        const component = schemaRegistry.get('EnvironmentLightComponent');
        if (!component) throw new Error(`EnvironmentLightComponent is not registered`);
        const entities = defineQuery([component])(ctx.scene.world);
        if (entities.length > 1) {
            throw new Error(
                `environment-lighting supports at most one EnvironmentLightComponent, found ${entities.length}`,
            );
        }
        if (entities.length === 0) {
            this.system.setComponentEid(null);
            return;
        }

        const eid = entities[0];
        const settings = readEnvironmentSettings(eid);
        this.system.setComponentEid(eid);
        if (!settings.source.trim()) return;
        await this.loadEnvironmentResources(
            ctx,
            appBase,
            settings,
            generation,
            resourceGeneration,
        );
    }

    appUnloading(ctx: PluginContext): void {
        this.appGeneration++;
        this.resourceGeneration++;
        this.appContext = null;
        this.appBase = '';
        this.system.clear();
        if (!this.published) return;
        ctx.unregisterGpuResourceSet(ENVIRONMENT_RESOURCE_SET_NAME);
        this.published = false;
    }

    private handleComponentChange(settings: EnvironmentSettings | null): void {
        const ctx = this.appContext;
        if (!ctx) return;
        const generation = this.appGeneration;
        const resourceGeneration = ++this.resourceGeneration;

        if (!settings) {
            this.replaceWithFallback(ctx, generation, resourceGeneration);
            return;
        }

        void this.loadEnvironmentResources(
            ctx,
            this.appBase,
            settings,
            generation,
            resourceGeneration,
        ).catch((err: unknown) => {
            if (generation !== this.appGeneration
                || resourceGeneration !== this.resourceGeneration
                || ctx !== this.appContext) return;
            this.system.fail(err);
        });
    }

    private async loadEnvironmentResources(
        ctx: PluginContext,
        appBase: string,
        settings: EnvironmentSettings,
        generation: number,
        resourceGeneration: number,
    ): Promise<void> {
        let built: EnvironmentResourceSet;

        if (!settings.source.trim()) {
            built = createNeutralEnvironmentResourceSet(ctx.device);
        } else {
            const sourceUrl = settings.source.startsWith('/')
                ? settings.source
                : `${appBase}/${settings.source}`;
            const source = await this.sourceProvider.load({
                sourceUrl,
                maxWidth: runtimeSourceWidth(
                    settings.skyResolution,
                    ctx.device.limits.maxTextureDimension2D,
                ),
            });
            built = await buildEnvironment(
                ctx.device,
                source,
                settings,
                `${ctx.baseUrl}/shaders`,
            );
        }

        if (generation !== this.appGeneration
            || resourceGeneration !== this.resourceGeneration
            || ctx !== this.appContext
            || !this.published) {
            destroyUnpublishedEnvironmentResourceSet(built);
            return;
        }

        try {
            ctx.replaceGpuResourceSet(
                ENVIRONMENT_RESOURCE_SET_NAME,
                environmentResourceDescriptors(built),
            );
        } catch (err) {
            destroyUnpublishedEnvironmentResourceSet(built);
            throw err;
        }

        this.system.setRuntime(built.dataBuffer, built.specularMipCount);
    }

    private replaceWithFallback(
        ctx: PluginContext,
        generation: number,
        resourceGeneration: number,
    ): void {
        if (generation !== this.appGeneration
            || resourceGeneration !== this.resourceGeneration
            || ctx !== this.appContext
            || !this.published) return;
        const fallback = createNeutralEnvironmentResourceSet(ctx.device);
        try {
            ctx.replaceGpuResourceSet(
                ENVIRONMENT_RESOURCE_SET_NAME,
                environmentResourceDescriptors(fallback),
            );
        } catch (err) {
            destroyUnpublishedEnvironmentResourceSet(fallback);
            throw err;
        }
        this.system.setRuntime(fallback.dataBuffer, fallback.specularMipCount);
    }

    private publishFallback(ctx: PluginContext): void {
        if (this.published) throw new Error(`Environment resource set is already published`);
        const fallback = createNeutralEnvironmentResourceSet(ctx.device);
        try {
            ctx.registerGpuResourceSet(
                ENVIRONMENT_RESOURCE_SET_NAME,
                environmentResourceDescriptors(fallback),
            );
        } catch (err) {
            destroyUnpublishedEnvironmentResourceSet(fallback);
            throw err;
        }
        this.published = true;
        this.system.setRuntime(fallback.dataBuffer, fallback.specularMipCount);
    }
}
