import {
    resourceManager,
    mat4Inverse,
    mat4TransformVec4,
    quatRotateVec3,
    EVENT_TYPES,
    type EventBus,
    type FrameContext,
    type System,
    type Scene,
} from '@shaderlab/api';
import type { BrushOpts, TextureEditApi } from './index.ts';

const DEFAULT_CANVAS_W = 512;
const DEFAULT_CANVAS_H = 512;
const DEFAULT_BRUSH: Required<BrushOpts> = {
    color: [0, 0, 0, 1],
    radius: 14,
    opacity: 1,
};

interface CanvasState {
    w: number;
    h: number;
    gpu: GPUTexture;
    handle: number;
    cpu: Uint8Array;
    dirty: DirtyRect;
    /** Last painted UV on this canvas — for stroke continuity. */
    lastUvx: number | null;
    lastUvy: number | null;
}

interface DirtyRect {
    y0: number;
    y1: number;
    has: boolean;
}

interface CameraView {
    ivp: Float32Array;
    pos: Float32Array;
}

interface PickResult {
    eid: number;
    uv: [number, number];
    t: number;
}

type Brush = Required<BrushOpts>;

/**
 * CPU-mirrored canvas painting. One CanvasState per CanvasComponent entity.
 * Drawing (alpha-blended circles) happens on the CPU mirror; only dirty rows
 * are flushed to the GPU each frame via queue.writeTexture (full-width rows
 * keep bytesPerRow a multiple of 256).
 *
 * pbr_plane mesh: local XZ plane, normal +Y, UV.x = localX + 0.5,
 * UV.y = 0.5 - localZ.
 */
export class TextureEditSystem implements System, TextureEditApi {
    private bus: EventBus;
    private scene: Scene | null = null;
    private canvases = new Map<number, CanvasState>();
    private unsubs: Array<() => void> = [];

    // Cached mouse state (events fire between updates; paint happens in update).
    private mouseNdcX = 0;
    private mouseNdcY = 0;
    private mouseDown = false;
    private lastPick: PickResult | null = null;

    constructor(bus: EventBus) {
        this.bus = bus;
    }

    /** Subscribe to mouse + control events. */
    attach(): void {
        this.unsubs.push(
            this.bus.on(EVENT_TYPES.MOUSE_DOWN, this.onDown),
            this.bus.on(EVENT_TYPES.MOUSE_MOVE, this.onMove),
            this.bus.on(EVENT_TYPES.MOUSE_UP, this.onUp),
            this.bus.on('textureedit:reset', this.onReset),
            this.bus.on('textureedit:setBrush', this.onSetBrush),
        );
    }

    /** Create one canvas per CanvasComponent entity + stamp the handle/size. */
    async bindToScene(scene: Scene): Promise<void> {
        this.scene = scene;
        // Drop old states (app reload).
        for (const st of this.canvases.values()) st.gpu.destroy();
        this.canvases.clear();

        for (const { eid } of scene.getAllEntities()) {
            if (!scene.hasComponent(eid, 'CanvasComponent')) continue;
            const w = Number(scene.getField(eid, 'CanvasComponent', 'width')) || DEFAULT_CANVAS_W;
            const h = Number(scene.getField(eid, 'CanvasComponent', 'height')) || DEFAULT_CANVAS_H;
            const state = await this.createCanvas(eid, w, h);
            scene.setField(eid, 'CanvasComponent', 'texHandle', state.handle);
            scene.setField(eid, 'CanvasComponent', 'width', w);
            scene.setField(eid, 'CanvasComponent', 'height', h);
            this.canvases.set(eid, state);
        }
    }

    update(ctx: FrameContext): void {
        if (!this.scene) return;
        if (this.mouseDown) {
            const cam = ctx.scene.getActiveCameras(ctx.aspect)[0] ?? null;
            if (cam) {
                const hit = this.pick(ctx.scene, cam, this.mouseNdcX, this.mouseNdcY);
                if (hit) {
                    const brush = this.resolveBrush(ctx.scene, hit.eid);
                    if (this.lastPick && this.lastPick.eid === hit.eid) {
                        this.paintLine(hit.eid, this.lastPick.uv, hit.uv, brush);
                    } else {
                        this.paint(hit.eid, hit.uv, brush);
                    }
                    this.lastPick = hit;
                } else {
                    this.lastPick = null;
                }
            }
        }
        this.flushAll();
    }

    clear(): void {
        for (const off of this.unsubs) off();
        this.unsubs = [];
        for (const st of this.canvases.values()) st.gpu.destroy();
        this.canvases.clear();
        this.scene = null;
        this.mouseDown = false;
        this.lastPick = null;
    }

    dispose(): void {
        this.clear();
    }

    /* ── public paint API (TextureEditApi) ─────── */

    paint(eid: number, uv: [number, number], brush?: BrushOpts): void {
        const st = this.canvases.get(eid);
        if (!st) return;
        const b = brush ? this.mergeBrush(brush)
            : (this.scene ? this.resolveBrush(this.scene, eid) : { ...DEFAULT_BRUSH });
        const px = uv[0] * st.w;
        const py = uv[1] * st.h;
        this.drawBrush(st, px, py, b);
    }

    paintLine(eid: number, a: [number, number], b: [number, number], brush?: BrushOpts): void {
        const st = this.canvases.get(eid);
        if (!st) return;
        const br = brush ? this.mergeBrush(brush)
            : (this.scene ? this.resolveBrush(this.scene, eid) : { ...DEFAULT_BRUSH });
        const ax = a[0] * st.w, ay = a[1] * st.h;
        const bx = b[0] * st.w, by = b[1] * st.h;
        const dist = Math.hypot(bx - ax, by - ay);
        const steps = Math.max(1, Math.ceil(dist));
        for (let i = 1; i <= steps; i++) {
            const t = i / steps;
            this.drawBrush(st, ax + (bx - ax) * t, ay + (by - ay) * t, br);
        }
    }

    paintWorld(eid: number, world: [number, number, number], brush?: BrushOpts): void {
        if (!this.scene) return;
        const uv = this.worldToUv(this.scene, eid, world);
        if (uv) this.paint(eid, uv, brush);
    }

    reset(eid?: number): void {
        if (eid !== undefined) {
            const st = this.canvases.get(eid);
            if (st) this.resetCanvas(st);
            return;
        }
        for (const st of this.canvases.values()) this.resetCanvas(st);
    }

    /* ── event handlers ─────────────────────────── */

    private onDown = (payload: unknown): void => {
        const e = payload as { button: number; x: number; y: number };
        if (e.button !== 0) return;
        this.mouseNdcX = e.x;
        this.mouseNdcY = e.y;
        this.mouseDown = true;
        this.lastPick = null;
    };

    private onMove = (payload: unknown): void => {
        const e = payload as { x: number; y: number; buttons: number };
        this.mouseNdcX = e.x;
        this.mouseNdcY = e.y;
        if ((e.buttons & 1) === 0) this.mouseDown = false;
    };

    private onUp = (payload: unknown): void => {
        const e = payload as { button: number };
        if (e.button === 0) {
            this.mouseDown = false;
            this.lastPick = null;
        }
    };

    private onReset = (payload: unknown): void => {
        const e = (payload ?? {}) as { eid?: number };
        this.reset(e.eid);
    };

    private onSetBrush = (payload: unknown): void => {
        if (!this.scene) return;
        const e = (payload ?? {}) as { color?: number[]; radius?: number; opacity?: number };
        for (const { eid } of this.scene.getAllEntities()) {
            if (!this.scene.hasComponent(eid, 'BrushComponent')) continue;
            if (e.color) this.scene.setField(eid, 'BrushComponent', 'color', e.color);
            if (e.radius !== undefined) this.scene.setField(eid, 'BrushComponent', 'radius', e.radius);
            if (e.opacity !== undefined) this.scene.setField(eid, 'BrushComponent', 'opacity', e.opacity);
        }
    };

    /* ── canvas creation ─────────────────────────── */

    private async createCanvas(eid: number, w: number, h: number): Promise<CanvasState> {
        const imageData = new ImageData(w, h);
        for (let i = 0; i < w * h; i++) {
            imageData.data[i * 4 + 0] = 255;
            imageData.data[i * 4 + 1] = 255;
            imageData.data[i * 4 + 2] = 255;
            imageData.data[i * 4 + 3] = 255;
        }
        const bitmap = await createImageBitmap(imageData);
        const key = `textureedit:canvas#${eid}`;
        const gpu = await resourceManager.uploadTextureFromImage(key, bitmap);
        const handle = resourceManager.textureHandle(key);
        bitmap.close();
        const cpu = new Uint8Array(w * h * 4);
        cpu.fill(255);
        return { w, h, gpu, handle, cpu, dirty: { y0: 0, y1: 0, has: false }, lastUvx: null, lastUvy: null };
    }

    private resetCanvas(st: CanvasState): void {
        st.cpu.fill(255);
        st.dirty = { y0: 0, y1: st.h, has: true };
        st.lastUvx = null;
        st.lastUvy = null;
    }

    /* ── brush blending ──────────────────────────── */

    private resolveBrush(scene: Scene, eid: number): Brush {
        if (!scene.hasComponent(eid, 'BrushComponent')) return { ...DEFAULT_BRUSH };
        const c = scene.getField(eid, 'BrushComponent', 'color') as number[] | undefined;
        const r = Number(scene.getField(eid, 'BrushComponent', 'radius') ?? DEFAULT_BRUSH.radius);
        const o = Number(scene.getField(eid, 'BrushComponent', 'opacity') ?? DEFAULT_BRUSH.opacity);
        return {
            color: c ? [c[0], c[1], c[2], c[3]] : DEFAULT_BRUSH.color,
            radius: r,
            opacity: o,
        };
    }

    private mergeBrush(brush: BrushOpts): Brush {
        return {
            color: brush.color ?? DEFAULT_BRUSH.color,
            radius: brush.radius ?? DEFAULT_BRUSH.radius,
            opacity: brush.opacity ?? DEFAULT_BRUSH.opacity,
        };
    }

    private drawBrush(st: CanvasState, cx: number, cy: number, b: Brush): void {
        const r = b.radius;
        const x0 = Math.max(0, Math.floor(cx - r));
        const y0 = Math.max(0, Math.floor(cy - r));
        const x1 = Math.min(st.w, Math.ceil(cx + r) + 1);
        const y1 = Math.min(st.h, Math.ceil(cy + r) + 1);
        const r2 = r * r;
        const cr = b.color[0] * 255;
        const cg = b.color[1] * 255;
        const cb = b.color[2] * 255;
        const a = b.color[3] * b.opacity;
        const inv = 1 - a;
        const buf = st.cpu;
        const rowBytes = st.w * 4;
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                const dx = x + 0.5 - cx;
                const dy = y + 0.5 - cy;
                if (dx * dx + dy * dy > r2) continue;
                const o = y * rowBytes + x * 4;
                buf[o]     = cr * a + buf[o]     * inv;
                buf[o + 1] = cg * a + buf[o + 1] * inv;
                buf[o + 2] = cb * a + buf[o + 2] * inv;
                buf[o + 3] = 255;
            }
        }
        if (!st.dirty.has) {
            st.dirty = { y0, y1, has: true };
        } else {
            st.dirty.y0 = Math.min(st.dirty.y0, y0);
            st.dirty.y1 = Math.max(st.dirty.y1, y1);
        }
    }

    private flushAll(): void {
        for (const st of this.canvases.values()) this.flush(st);
    }

    private flush(st: CanvasState): void {
        if (!st.dirty.has) return;
        const { y0, y1 } = st.dirty;
        const h = y1 - y0;
        if (h <= 0) {
            st.dirty.has = false;
            return;
        }
        const rowBytes = st.w * 4;
        const buf = new Uint8Array(h * rowBytes);
        for (let y = 0; y < h; y++) {
            const srcStart = (y + y0) * rowBytes;
            buf.set(st.cpu.subarray(srcStart, srcStart + rowBytes), y * rowBytes);
        }
        resourceManager.device.queue.writeTexture(
            { texture: st.gpu, origin: { x: 0, y: y0 } },
            buf,
            { bytesPerRow: rowBytes, rowsPerImage: h },
            { width: st.w, height: h },
        );
        st.dirty.has = false;
    }

    /* ── raycast / picking ──────────────────────── */

    private pick(scene: Scene, cam: CameraView, ndcX: number, ndcY: number): PickResult | null {
        let best: PickResult | null = null;
        for (const [eid] of this.canvases) {
            const hit = this.raycastPlane(scene, cam, eid, ndcX, ndcY);
            if (!hit) continue;
            if (!best || hit.t < best.t) best = hit;
        }
        return best;
    }

    private raycastPlane(scene: Scene, cam: CameraView, planeEid: number, ndcX: number, ndcY: number): PickResult | null {
        const near = mat4TransformVec4(cam.ivp, [ndcX, ndcY, 0, 1]);
        const invW = near[3] !== 0 ? 1 / near[3] : 1;
        const nearX = near[0] * invW;
        const nearY = near[1] * invW;
        const nearZ = near[2] * invW;

        const camX = cam.pos[0], camY = cam.pos[1], camZ = cam.pos[2];
        let dx = nearX - camX, dy = nearY - camY, dz = nearZ - camZ;
        const dl = Math.hypot(dx, dy, dz) || 1;
        dx /= dl; dy /= dl; dz /= dl;

        const rot = scene.getField(planeEid, 'Transform', 'rotation') as [number, number, number, number] | undefined;
        const q: [number, number, number, number] = rot ?? [0, 0, 0, 1];
        const n = quatRotateVec3(q, [0, 1, 0]);

        const pos = scene.getField(planeEid, 'Transform', 'position') as [number, number, number] | undefined;
        const pp: [number, number, number] = pos ?? [0, 0, 0];

        const denom = dx * n[0] + dy * n[1] + dz * n[2];
        if (Math.abs(denom) < 1e-6) return null;
        const t = ((pp[0] - camX) * n[0] + (pp[1] - camY) * n[1] + (pp[2] - camZ) * n[2]) / denom;
        if (t < 0) return null;

        const hitX = camX + dx * t, hitY = camY + dy * t, hitZ = camZ + dz * t;
        const uv = this.worldToUv(scene, planeEid, [hitX, hitY, hitZ]);
        if (!uv) return null;
        return { eid: planeEid, uv, t };
    }

    private worldToUv(scene: Scene, planeEid: number, world: [number, number, number]): [number, number] | null {
        const model = scene.getModelMatrix(planeEid);
        const inv = mat4Inverse(model);
        const local = mat4TransformVec4(inv, [world[0], world[1], world[2], 1]);
        const u = local[0] + 0.5;
        const v = 0.5 - local[2];
        if (u < 0 || u > 1 || v < 0 || v > 1) return null;
        return [u, v];
    }
}
