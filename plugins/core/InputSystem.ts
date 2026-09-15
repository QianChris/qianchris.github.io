import { resourceManager, uniformLayouts, EVENT_TYPES } from '@shaderlab/api';
import type { EventBus, FrameContext, System } from '@shaderlab/api';

/**
 * Mouse input tracker. Fills the per-frame TimeInput UBO.
 *
 * UBO layout: declared in uniform-layouts.json as "timeInput"
 * (time, dt, frame, _pad, mouse).
 */
export class InputSystem implements System {
    private canvas: HTMLCanvasElement;
    private bus: EventBus;
    private mouseX = 0;
    private mouseY = 0;
    private buttons = 0;
    private wheel = 0;
    private frameCount = 0;

    private data: Float32Array = new Float32Array(0);

    constructor(canvas: HTMLCanvasElement, bus: EventBus) {
        this.canvas = canvas;
        this.bus = bus;
        this.data = uniformLayouts.get('timeInput').createBuffer();
    }

    attach(): void {
        this.canvas.addEventListener('mousemove', this.onMove);
        this.canvas.addEventListener('mousedown', this.onDown);
        window.addEventListener('mouseup', this.onUp);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: true });
        this.canvas.addEventListener('contextmenu', this.onContextMenu);
    }

    detach(): void {
        this.canvas.removeEventListener('mousemove', this.onMove);
        this.canvas.removeEventListener('mousedown', this.onDown);
        window.removeEventListener('mouseup', this.onUp);
        this.canvas.removeEventListener('wheel', this.onWheel);
        this.canvas.removeEventListener('contextmenu', this.onContextMenu);
    }

    update(ctx: FrameContext): void {
        const layout = uniformLayouts.get('timeInput');
        const buf = this.data;
        buf.fill(0);
        layout.write(buf, 'time', ctx.time);
        layout.write(buf, 'dt', ctx.dt);
        layout.write(buf, 'frame', this.frameCount++);
        layout.write(buf, 'mouse', [this.mouseX, this.mouseY, this.buttons, this.wheel]);
        const ubo = resourceManager.timeInputUBO;
        resourceManager.device.queue.writeBuffer(ubo, 0, buf.buffer, buf.byteOffset, buf.byteLength);
    }

    private onMove = (e: MouseEvent): void => {
        const rect = this.canvas.getBoundingClientRect();
        const nx = (e.clientX - rect.left) / rect.width;
        const ny = (e.clientY - rect.top) / rect.height;
        this.mouseX = nx * 2 - 1;
        this.mouseY = 1 - ny * 2;
        this.bus.emit(EVENT_TYPES.MOUSE_MOVE, { x: this.mouseX, y: this.mouseY, buttons: this.buttons });
    };

    private onDown = (e: MouseEvent): void => {
        this.buttons |= 1 << e.button;
        this.bus.emit(EVENT_TYPES.MOUSE_DOWN, { button: e.button, x: this.mouseX, y: this.mouseY });
    };

    private onUp = (e: MouseEvent): void => {
        this.buttons &= ~(1 << e.button);
        this.bus.emit(EVENT_TYPES.MOUSE_UP, { button: e.button, x: this.mouseX, y: this.mouseY });
    };

    private onWheel = (e: WheelEvent): void => {
        this.wheel += Math.sign(e.deltaY);
        this.bus.emit(EVENT_TYPES.WHEEL, { delta: Math.sign(e.deltaY), x: this.mouseX, y: this.mouseY });
    };

    private onContextMenu = (e: MouseEvent): void => {
        e.preventDefault();
    };
}
