import { defineQuery, schemaRegistry, resourceManager } from '@shaderlab/api';
import type { Scene, FrameContext, System } from '@shaderlab/api';

interface SheetAnimation {
    name: string;
    row: number;
    frames: number;
    fps: number;
    loop: 'pingpong' | 'loop' | 'once';
}

interface SheetData {
    texture: string;
    columns: number;
    rows: number;
    animations: SheetAnimation[];
}

/**
 * Advances sprite-sheet animations. Each entity needs a SpriteSheetComponent
 * (asset reference: path to a sheet JSON + texture handle + grid columns/rows)
 * and a SpriteAnimationComponent (playback state: animation index, frame, etc.).
 *
 * The sheet JSON (app-specific) declares the texture path, grid dimensions, and
 * per-animation metadata (frame count, fps, loop mode). It is fetched lazily on
 * first encounter and cached. The texture is also loaded lazily and its handle
 * stored in SpriteSheetComponent.texHandle for the renderer to bind.
 *
 * Loop modes:
 *   "pingpong" – forward then backward (0,1,…,n-1,n-2,…,1,0,1,…)
 *   "loop"     – wrap around (0,1,…,n-1,0,1,…)
 *   "once"     – play forward once then stop (sets playing=0)
 *
 * Migrated from core/AnimationSystem.ts — the sprite plugin now owns the
 * component declarations, the system, and the pipeline. The system name
 * remains 'animation' (systems.json order unchanged).
 */
export class SpriteSystem implements System {
    private scene!: Scene;
    private baseDir = '';
    private sheets = new Map<string, SheetData | null>();
    private texHandles = new Map<string, number>();
    private loading = new Set<string>();
    private texLoading = new Set<string>();
    private initialized = false;
    private query: (w: import('bitecs').World) => readonly number[] = () => [];

    attach(scene: Scene): void {
        this.scene = scene;
    }

    setBaseDir(dir: string): void {
        this.baseDir = dir;
    }

    clear(): void {
        this.sheets.clear();
        this.texHandles.clear();
        this.loading.clear();
        this.texLoading.clear();
        this.initialized = false;
        this.query = () => [];
    }

    update(ctx: FrameContext): void {
        const dt = ctx.dt;
        if (!this.initialized) {
            this.initialized = true;
            const sheetComp = schemaRegistry.get('SpriteSheetComponent');
            const animComp = schemaRegistry.get('SpriteAnimationComponent');
            if (sheetComp && animComp) {
                this.query = defineQuery([sheetComp, animComp]);
            }
        }

        const scene = this.scene;
        for (const eid of this.query(scene.world)) {
            const sheetPath = scene.getField(eid, 'SpriteSheetComponent', 'sheet') as string;
            if (!sheetPath) continue;

            const sheet = this.getSheet(sheetPath);
            if (!sheet) continue;

            scene.setField(eid, 'SpriteSheetComponent', 'columns', sheet.columns);
            scene.setField(eid, 'SpriteSheetComponent', 'rows', sheet.rows);

            if (!this.texHandles.has(sheetPath) && !this.texLoading.has(sheetPath)) {
                this.loadTexture(sheetPath, sheet);
            }

            const handle = this.texHandles.get(sheetPath) ?? 0;
            scene.setField(eid, 'SpriteSheetComponent', 'texHandle', handle);

            const animIndex = Number(scene.getField(eid, 'SpriteAnimationComponent', 'animation') ?? 0) | 0;
            const anim = sheet.animations[animIndex];
            if (!anim) continue;

            scene.setField(eid, 'SpriteAnimationComponent', 'row', anim.row);

            const playing = Number(scene.getField(eid, 'SpriteAnimationComponent', 'playing') ?? 0);
            if (playing !== 1) continue;

            const fps = anim.fps;
            if (fps <= 0) continue;

            const frames = anim.frames;
            if (frames <= 1) continue;

            let elapsed = Number(scene.getField(eid, 'SpriteAnimationComponent', 'elapsed') ?? 0) + dt;
            let frame = Number(scene.getField(eid, 'SpriteAnimationComponent', 'frame') ?? 0) | 0;
            let direction = Number(scene.getField(eid, 'SpriteAnimationComponent', 'direction') ?? 1) | 0;
            if (direction === 0) direction = 1;
            const frameDuration = 1 / fps;

            let steps = 0;
            while (elapsed >= frameDuration && steps < 100) {
                elapsed -= frameDuration;
                frame += direction;

                if (anim.loop === 'pingpong') {
                    if (frame >= frames) {
                        frame = frames - 2;
                        direction = -1;
                    } else if (frame < 0) {
                        frame = 1;
                        direction = 1;
                    }
                } else if (anim.loop === 'loop') {
                    if (frame >= frames) {
                        frame = 0;
                    } else if (frame < 0) {
                        frame = frames - 1;
                    }
                } else {
                    if (frame >= frames) {
                        frame = frames - 1;
                        scene.setField(eid, 'SpriteAnimationComponent', 'playing', 0);
                        break;
                    } else if (frame < 0) {
                        frame = 0;
                    }
                }

                if (frame < 0) frame = 0;
                if (frame >= frames) frame = frames - 1;
                steps++;
            }

            scene.setField(eid, 'SpriteAnimationComponent', 'frame', frame);
            scene.setField(eid, 'SpriteAnimationComponent', 'direction', direction);
            scene.setField(eid, 'SpriteAnimationComponent', 'elapsed', elapsed);
        }
    }

    private getSheet(path: string): SheetData | null {
        const cached = this.sheets.get(path);
        if (cached !== undefined) return cached;

        if (!this.loading.has(path)) {
            this.loading.add(path);
            const url = path.startsWith('/') ? path : `${this.baseDir}/${path}`;
            fetch(url)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    return r.json();
                })
                .then(data => {
                    this.sheets.set(path, data as SheetData);
                })
                .catch(err => {
                    console.error(`[SpriteSystem] failed to load sheet '${path}' (will retry):`, err);
                })
                .finally(() => { this.loading.delete(path); });
        }
        return null;
    }

    private loadTexture(sheetPath: string, data: SheetData): void {
        const texPath = data.texture;
        if (!texPath) {
            this.texHandles.set(sheetPath, 0);
            this.texLoading.delete(sheetPath);
            return;
        }
        const texUrl = texPath.startsWith('/') ? texPath : `${this.baseDir}/${texPath}`;
        this.texLoading.add(sheetPath);
        resourceManager.loadTexture(texUrl)
            .then(() => {
                this.texHandles.set(sheetPath, resourceManager.textureHandle(texUrl));
            })
            .catch(err => {
                console.error(`[SpriteSystem] failed to load texture '${texUrl}' (will retry):`, err);
            })
            .finally(() => { this.texLoading.delete(sheetPath); });
    }
}
