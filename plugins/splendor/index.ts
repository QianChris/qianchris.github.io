import { EnginePlugin, type PluginContext } from '@shaderlab/api';
import { SplendorSystem } from './SplendorSystem.ts';

/**
 * Splendor — a fully playable 2-player game running on the engine's host layer.
 *
 * Capability split (see docs/splendor-gap-analysis.md):
 *   - rules + AI: this plugin (pure TS, `npm test` covered)
 *   - board/UI:    public/apps/splendor/ui/board.js (app DOM UI; loaded by the
 *                  player runtime too, not just the editor)
 *   - engine:      scene/render are intentionally empty — the board is DOM. The
 *                  engine still owns the frame loop, app lifecycle and the
 *                  event bus, so the game gets plugin/owner cleanup for free.
 *
 * The system is auto-inserted before `render` (systemDefs.before) because the
 * app ships no systems.json of its own.
 */
export default class SplendorPlugin extends EnginePlugin {
    readonly meta = { id: 'splendor' };

    systemDefs = [
        {
            name: 'splendor',
            source: 'plugin:splendor',
            components: [],
            ubos: [],
            buffers: [],
            needs: [],
            before: ['render'],
        },
    ];

    private system: SplendorSystem | null = null;

    setup(ctx: PluginContext): void {
        this.system = new SplendorSystem(ctx.eventBus);
        ctx.registerSystem('splendor', this.system);
        // Published so other plugins (or an app UI) can inspect the match.
        ctx.registerAttachment('splendor', this.system);
    }

    appLoaded(_ctx: PluginContext, appBase: string): void {
        this.system?.start(appBase);
    }

    appUnloading(): void {
        this.system?.stop();
    }

    teardown(): void {
        this.system?.dispose();
        this.system = null;
    }
}
