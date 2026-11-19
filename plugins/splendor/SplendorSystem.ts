/**
 * Frame system that owns the Splendor match and bridges it to the DOM UI.
 *
 * The system is deliberately thin: all rules live in rules.ts (pure, unit
 * tested), the opponent in ai.ts. The system only does three things:
 *   1. keep the single `GameState` and apply actions arriving over the event bus,
 *   2. schedule AI turns with a human-visible delay (turn-based games should not
 *      feel instantaneous),
 *   3. publish a render-ready view (state + affordability + take options) so the
 *      UI never has to re-implement rules.
 *
 * It is registered as the `splendor` system (systems.json auto-inserts it before
 * `render`) and published as the `splendor` attachment for other plugins.
 */
import type { EventBus, FrameContext } from '@shaderlab/api';
import { chooseAction } from './ai.ts';
import { CARDS, NOBLES } from './cards.ts';
import {
    applyAction, canAfford, canTakeTwo, cardOf, createGame, legalTakes, scores,
    takeSize, COLORS, type Action, type CardDef, type GameState, type NobleDef, type ScoreLine,
} from './rules.ts';

export interface SplendorView {
    /** App base URL ('/apps/splendor'), used by the UI to resolve card art. */
    base: string;
    state: GameState;
    /** Read-only catalogues (passed by reference — the UI renders cost/points). */
    cards: readonly CardDef[];
    nobles: readonly NobleDef[];
    scores: ScoreLine[];
    /** Seats controlled by the AI. */
    aiSeats: number[];
    /** card id -> affordable now. */
    canBuy: Record<number, boolean>;
    canReserve: boolean;
    takeSize: number;
    /** per color: whether a "take two of the same" is legal. */
    canTakeTwo: boolean[];
    thinking: boolean;
    over: boolean;
    winner: number | null;
}

const AI_DELAY = 0.55;

export class SplendorSystem {
    private bus: EventBus;
    private state: GameState;
    private base = '';
    private aiSeats: number[] = [1];
    private timer = 0;
    private unsubs: Array<() => void> = [];
    private started = false;

    constructor(bus: EventBus) {
        this.bus = bus;
        this.state = createGame();
        this.unsubs.push(bus.on('splendor:request', () => this.emit()));
        this.unsubs.push(bus.on('splendor:action', payload => this.handleAction(payload)));
        this.unsubs.push(bus.on('splendor:new', payload => this.newGame(payload)));
        this.unsubs.push(bus.on('splendor:mode', payload => this.setMode(payload)));
    }

    /* ── lifecycle (driven by the plugin) ── */

    start(appBase: string): void {
        this.base = appBase;
        this.started = true;
        this.newGame(undefined);
    }

    stop(): void {
        this.started = false;
        this.timer = 0;
    }

    dispose(): void {
        for (const unsub of this.unsubs) unsub();
        this.unsubs = [];
        this.started = false;
    }

    /* ── system interface ── */

    update(ctx: FrameContext): void {
        if (!this.started || this.state.over) return;
        if (!this.aiSeats.includes(this.state.current)) {
            this.timer = 0;
            return;
        }
        // dt is unclamped by the engine (tab switches produce huge deltas).
        this.timer += Math.min(ctx.dt, 0.25);
        if (this.timer < AI_DELAY) return;
        this.timer = 0;
        const action = chooseAction(this.state);
        if (!action) return;
        const result = applyAction(this.state, action);
        if (!result.ok) {
            this.bus.emit('splendor:error', { message: `AI 动作被拒绝：${result.error}` });
            return;
        }
        this.state = result.state;
        this.emit();
    }

    /* ── event handlers ── */

    private handleAction(payload: unknown): void {
        if (!this.started) return;
        if (this.aiSeats.includes(this.state.current)) {
            this.bus.emit('splendor:error', { message: '现在是对手回合' });
            return;
        }
        const action = payload as Action;
        if (!action || typeof action !== 'object' || typeof action.type !== 'string') {
            this.bus.emit('splendor:error', { message: '无效的动作' });
            return;
        }
        const result = applyAction(this.state, action);
        if (!result.ok) {
            this.bus.emit('splendor:error', { message: result.error });
            return;
        }
        this.state = result.state;
        this.timer = 0;
        this.emit();
    }

    private newGame(payload: unknown): void {
        const seed = (payload as { seed?: number } | undefined)?.seed;
        this.state = createGame(typeof seed === 'number' ? seed : (Date.now() >>> 0));
        this.timer = 0;
        this.emit();
    }

    private setMode(payload: unknown): void {
        const ai = (payload as { ai?: boolean } | undefined)?.ai;
        this.aiSeats = ai === false ? [] : [1];
        this.timer = 0;
        this.emit();
    }

    /* ── view assembly ── */

    private emit(): void {
        this.bus.emit('splendor:state', this.view());
    }

    view(): SplendorView {
        const state = this.state;
        const player = state.players[state.current];
        const canBuy: Record<number, boolean> = {};
        for (const id of state.market.flat()) {
            if (id != null) canBuy[id] = canAfford(cardOf(id), player);
        }
        for (const id of player.reserved) canBuy[id] = canAfford(cardOf(id), player);
        return {
            base: this.base,
            state,
            cards: CARDS,
            nobles: NOBLES,
            scores: scores(state),
            aiSeats: this.aiSeats.slice(),
            canBuy,
            canReserve: player.reserved.length < 3,
            takeSize: takeSize(state),
            canTakeTwo: COLORS.map(c => canTakeTwo(state, c)),
            thinking: !state.over && this.aiSeats.includes(state.current),
            over: state.over,
            winner: state.winner,
        };
    }

    /** Legal take shapes (for tests / other plugins; the UI reads them off the view). */
    takeShapes(): number[][] {
        return legalTakes(this.state);
    }
}
