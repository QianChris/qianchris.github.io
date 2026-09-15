import { EnginePlugin, type PluginContext, type ComponentDef, type BindLayoutDecls, type SystemDef } from '@shaderlab/api';
import { AnimationSamplerSystem } from './AnimationSamplerSystem.ts';
import { SkinningSystem } from './SkinningSystem.ts';

/**
 * Skeletal animation capability plugin. Owns:
 *   - SkeletonComponent (skin asset name → ResourceManager.getSkin)
 *   - AnimationPlayerComponent (clip name + playback state)
 *   - 'skeletal-animation' system (samples clip channels → joint Local TRS)
 *   - 'skinning' system (joint world × IBM → storage buffer, published as
 *     the `animation.joint-matrices` GPU resource set consumed by
 *     SkinnedPbrPipeline's @group(3))
 *   - SkinnedPbrPipeline + SkinnedPbr.wgsl (vertex skinning via JOINTS_0/
 *     WEIGHTS_0 + the storage buffer)
 *
 * The engine's glTF loader already parses skins + animations into
 * ResourceManager (GltfSkinData / GltfAnimationData with jointNames /
 * channel.nodeName resolved to entity keys), so this plugin only reads that
 * data — no glTF knowledge lives here.
 *
 * App-scoped: declared by an app's `"plugins": ["animation"]`. The two
 * systems auto-insert into the common system order via after/before.
 */
export default class AnimationPlugin extends EnginePlugin {
    readonly meta = { id: 'animation', dependencies: ['core'] };

    bindLayouts: BindLayoutDecls = {
        skinnedJoints: {
            entries: [
                { binding: 0, visibility: ['vertex'], buffer: 'read-only-storage' },
            ],
        },
    };

    systemDefs: SystemDef[] = [
        {
            name: 'skeletal-animation',
            source: 'plugin:animation',
            components: ['AnimationPlayerComponent'],
            ubos: [], buffers: [], needs: [],
            after: ['input'],
            before: ['transform'],
        },
        {
            name: 'skinning',
            source: 'plugin:animation',
            components: ['SkeletonComponent', 'GlobalTransform'],
            ubos: [], buffers: [], needs: ['transform'],
            before: ['camera'],
        },
    ];

    private sampler = new AnimationSamplerSystem();
    private skinning = new SkinningSystem();

    async init(ctx: PluginContext): Promise<void> {
        const url = `${ctx.baseUrl}/components.json`;
        const resp = await fetch(url);
        const ct = resp.headers.get('content-type') ?? '';
        if (!resp.ok || ct.includes('text/html')) {
            throw new Error(`[animation] declaration file missing: ${url}`);
        }
        this.components = await resp.json() as ComponentDef[];
    }

    setup(ctx: PluginContext): void {
        this.sampler.attach(ctx.scene);
        this.skinning.attach(ctx.scene);
        this.skinning.setContext(ctx);
        this.skinning.ensureBuffer(ctx.device);
        this.skinning.publishFallback(ctx);
        ctx.registerSystem('skeletal-animation', this.sampler);
        ctx.registerSystem('skinning', this.skinning);
    }

    appUnloading(_ctx: PluginContext): void {
        this.sampler.clear();
        this.skinning.clear();
    }

    teardown(ctx: PluginContext): void {
        this.skinning.dispose();
        this.sampler.clear();
    }
}
