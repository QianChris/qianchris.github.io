import { defineQuery, schemaRegistry } from '@shaderlab/api';
import type { Scene, EventBus, FrameContext, System } from '@shaderlab/api';

export interface ScriptContext {
    eid: number;
    scene: Scene;
    time: number;
    dt: number;
    /** Current viewport aspect ratio (width / height). */
    aspect: number;
    /** Physics system (ray casts, etc); null when the physics plugin is absent.
     *  Structural contract — gameplay scripts call e.g. physics.castRay(...). */
    physics: unknown | null;
    getField(compName: string, field: string): unknown;
    setField(compName: string, field: string, value: unknown): void;
    on(type: string, handler: (payload: unknown) => void): () => void;
    /** Publish an event on the shared bus. */
    emit(type: string, payload?: unknown): void;
}

interface ScriptModule {
    init?: (ctx: ScriptContext) => void;
    update?: (ctx: ScriptContext) => void;
    [key: string]: unknown;
}

export class ScriptSystem implements System {
    private scene!: Scene;
    private bus: EventBus;
    private baseDir: string;
    private getAspect: () => number = () => 1;
    /** Lazy physics lookup (the physics plugin registers after ScriptSystem exists). */
    private getPhysics: () => unknown | null = () => null;
    private hooks: string[] = ['init', 'update'];
    private query!: (w: import('bitecs').World) => readonly number[];
    private modules = new Map<string, ScriptModule>();
    private loading = new Set<string>();
    /** Paths whose hot-reload import is in flight. While a path is reloading,
     *  update() must NOT fall back to load() (which would re-fetch the OLD file
     *  from disk and clobber the just-edited in-memory source). */
    private reloading = new Set<string>();
    private initialized = new Set<string>();

    constructor(bus: EventBus, baseDir = '') {
        this.bus = bus;
        this.baseDir = baseDir;
    }

    /** Set the script lifecycle hooks to call (from engine-config.json). */
    setHooks(hooks: string[]): void {
        this.hooks = hooks;
    }

    /** Set the base directory for resolving relative script paths (e.g. '/apps/shadow'). */
    setBaseDir(dir: string): void {
        this.baseDir = dir;
    }

    attach(scene: Scene): void {
        this.scene = scene;
        this.query = defineQuery([schemaRegistry.get('ScriptComponent')!]);
    }

    /** Provide runtime services scripts can use (physics ray casts, viewport aspect).
     *  `getPhysics` is a lazy lookup — the physics system is plugin-provided and
     *  may register (or be absent) independently of this system's lifetime. */
    provide(getPhysics: () => unknown | null, getAspect: () => number): void {
        this.getPhysics = getPhysics;
        this.getAspect = getAspect;
    }

    /** Drop cached modules and init flags (call when the scene is reset). */
    clear(): void {
        this.modules.clear();
        this.initialized.clear();
        this.loading.clear();
        this.reloading.clear();
    }

    /** Hot-reload a gameplay script (ScriptComponent.script path) with in-memory
     *  source. Replaces the cached module so the next frame picks up the new
     *  implementation; per-entity init() is re-run lazily. Throws on bad import. */
    reloadScript(path: string, source: string): void {
        this.modules.delete(path);
        this.reloading.add(path);
        for (const key of [...this.initialized]) {
            if (key.startsWith(`${path}#`)) this.initialized.delete(key);
        }
        const blob = new Blob([source], { type: 'text/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        import(/* @vite-ignore */ blobUrl)
            .then(mod => {
                this.modules.set(path, (mod.default ?? mod) as ScriptModule);
            })
            .catch(err => {
                console.error(`[ScriptSystem] failed to hot-reload '${path}':`, err);
            })
            .finally(() => {
                this.reloading.delete(path);
                URL.revokeObjectURL(blobUrl);
            });
    }

    /** All currently-loaded gameplay script paths (ScriptComponent.script).
     *  Read-only enumeration for the editor's Scripts tab. */
    getScriptPaths(): string[] {
        return [...this.modules.keys()];
    }

    update(ctx: FrameContext): void {
        const time = ctx.time;
        const dt = ctx.dt;
        for (const eid of this.query(this.scene.world)) {
            const enabled = schemaRegistry.getScalar(schemaRegistry.get('ScriptComponent')!, eid, 'enabled');
            if (enabled !== 1) continue;

            const path = this.scene.getField(eid, 'ScriptComponent', 'script') as string;
            if (!path) continue;

            const mod = this.modules.get(path);
            if (!mod) {
                // During a hot-reload the new module's import is in flight —
                // don't re-fetch the old file from disk (would clobber the edit).
                if (this.reloading.has(path)) continue;
                this.load(path);
                continue;
            }

            const key = `${path}#${eid}`;
            const ctx = this.makeContext(eid, time, dt);
            for (const hook of this.hooks) {
                if (hook === 'init') {
                    if (!this.initialized.has(key)) {
                        this.initialized.add(key);
                        const fn = mod[hook];
                        if (typeof fn === 'function') fn(ctx);
                    }
                } else {
                    const fn = mod[hook];
                    if (typeof fn === 'function') fn(ctx);
                }
            }
        }
    }

    private makeContext(eid: number, time: number, dt: number): ScriptContext {
        const scene = this.scene;
        const bus = this.bus;
        return {
            eid,
            scene,
            time,
            dt,
            aspect: this.getAspect(),
            physics: this.getPhysics(),
            getField: (compName, field) => scene.getField(eid, compName, field),
            setField: (compName, field, value) => scene.setField(eid, compName, field, value),
            on: (type, handler) => bus.on(type, handler),
            emit: (type, payload) => bus.emit(type, payload),
        };
    }

    private load(path: string): void {
        if (this.loading.has(path)) return;
        this.loading.add(path);
        const url = (path.startsWith('/') ? path : `${this.baseDir}/${path}`) + `?t=${Date.now()}`;
        fetch(url)
            .then(r => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.text();
            })
            .then(src => {
                const blob = new Blob([src], { type: 'text/javascript' });
                const blobUrl = URL.createObjectURL(blob);
                return import(/* @vite-ignore */ blobUrl).finally(() => URL.revokeObjectURL(blobUrl));
            })
            .then(mod => {
                this.modules.set(path, (mod.default ?? mod) as ScriptModule);
            })
            .catch(err => {
                console.error(`[ScriptSystem] failed to load '${path}':`, err);
            })
            .finally(() => {
                this.loading.delete(path);
            });
    }
}
