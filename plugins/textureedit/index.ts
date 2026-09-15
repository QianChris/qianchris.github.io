import { EnginePlugin, type PluginContext, type ComponentDef, type BindLayoutDecls, type SamplerDecls } from '@shaderlab/api';
import { TextureEditSystem } from './TextureEditSystem.ts';

/**
 * Brush options accepted by the paint API. All fields optional — omitted
 * fields fall back to the target canvas entity's BrushComponent, then to the
 * system defaults (black, radius 14, opacity 1).
 */
export interface BrushOpts {
    /** RGBA 0..1. */
    color?: [number, number, number, number];
    /** Brush footprint radius in pixels. */
    radius?: number;
    /** 0..1 alpha multiplier applied on top of color.a. */
    opacity?: number;
}

/**
 * Programmatic paint surface exposed via the 'textureedit' attachment.
 * Callers (mouse handler, collision script, procedural timer) drive all
 * drawing through this; the system handles CPU blending + dirty-region flush.
 */
export interface TextureEditApi {
    /** Stamp a brush at a canvas UV (0..1). */
    paint(eid: number, uv: [number, number], brush?: BrushOpts): void;
    /** Paint a line between two UVs (interpolated to avoid gaps on fast moves). */
    paintLine(eid: number, a: [number, number], b: [number, number], brush?: BrushOpts): void;
    /** Paint at a world-space point — converted to UV via the entity's inverse model. */
    paintWorld(eid: number, world: [number, number, number], brush?: BrushOpts): void;
    /** Reset to white. Omit eid to reset every canvas. */
    reset(eid?: number): void;
}

/**
 * Texture editing capability: CPU-mirrored canvas texture(s) paintable by
 * mouse or by programmatic API, sampled by the TextureEditPipeline.
 *
 * Multi-canvas: one GPU texture per CanvasComponent entity, keyed
 * `textureedit:canvas#<eid>` in resourceManager. The system raycasts the
 * active camera against every canvas plane and picks the nearest hit.
 *
 * Brush config (color/radius/opacity) lives on a BrushComponent per canvas
 * entity (scene-configurable defaults) and can be overridden per paint call
 * or updated live via the `textureedit:setBrush` event (HUD-driven).
 *
 * App-scoped plugin (list in app.json `plugins`). Depends on 'core'.
 */
export default class TextureEditPlugin extends EnginePlugin {
    readonly meta = { id: 'textureedit', dependencies: ['core'] };

    components: ComponentDef[] = [
        {
            name: 'CanvasComponent',
            fields: {
                /** u32 GPU texture handle — written by the system in appLoaded. */
                texHandle: { type: 'u32', default: 0, role: 'texture' },
                /** Canvas dimensions in pixels. Defaults to 512; the system
                 *  allocates the GPU texture at this size. */
                width:  { type: 'u32', default: 512 },
                height: { type: 'u32', default: 512 },
            },
        },
        {
            name: 'BrushComponent',
            fields: {
                /** RGBA 0..1. */
                color:   { type: 'vec4', default: [0, 0, 0, 1] },
                radius:  { type: 'f32', default: 14 },
                opacity: { type: 'f32', default: 1 },
            },
        },
    ];

    bindLayouts: BindLayoutDecls = {
        textureEdit: {
            entries: [
                { binding: 0, visibility: ['fragment'], sampler: 'filtering' },
                { binding: 1, visibility: ['fragment'], texture: 'float' },
            ],
        },
    };

    samplers: SamplerDecls = {
        paint: {
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            magFilter: 'linear',
            minFilter: 'linear',
        },
    };

    systemDefs = [
        {
            name: 'textureEdit',
            source: 'plugin:textureedit',
            components: ['CanvasComponent', 'Transform'],
            ubos: [],
            buffers: [],
            needs: [],
            after: ['camera'],
            before: ['render'],
        },
    ];

    private system: TextureEditSystem | null = null;

    async setup(ctx: PluginContext): Promise<void> {
        this.system = new TextureEditSystem(ctx.eventBus);
        this.system.attach();
        ctx.registerSystem('textureEdit', this.system);
        ctx.registerAttachment('textureedit', this.system);
    }

    async appLoaded(ctx: PluginContext, _appBase: string): Promise<void> {
        await this.system?.bindToScene(ctx.scene);
    }

    appUnloading(): void {
        this.system?.clear();
    }

    teardown(): void {
        this.system?.clear();
        this.system = null;
    }
}
