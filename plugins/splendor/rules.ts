/**
 * Splendor rules engine — pure functions, zero engine/DOM dependencies.
 *
 * Everything here is deterministic given (state, action): the UI, the AI and the
 * unit tests all drive the same transition function. `createGame(seed)` is the
 * only entry point that consumes randomness, and it keeps its RNG state in the
 * game state so a match can be replayed from its seed.
 *
 * Model (matches the reference implementation, with the "finish the round"
 * endgame rule restored — see ENDGAME below):
 *   - 2 players, 5 colors x 4 tokens + 5 gold.
 *   - 3 tiers (40/30/20 cards), 4 market slots per tier, refill from the deck.
 *   - 10 nobles, 3 dealt; 3 points each, visited automatically after a purchase.
 *   - One action per turn: take gems / reserve (+1 gold) / buy.
 *   - Hand limit 10 tokens (gold included), reserve limit 3.
 *
 * ENDGAME: reaching 15 points triggers the final round; the opponent still gets
 * their turn, then the higher score wins (tie -> fewer purchased cards). The
 * reference stops immediately at 15, which hands the first player an edge.
 */
import { CARD_BY_ID, CARDS, NOBLE_BY_ID, NOBLES } from './cards.ts';

/* ── types ─────────────────────────────────────────────────────────── */

export type Color = 0 | 1 | 2 | 3 | 4;
export type Tier = 0 | 1 | 2;

/** Color indices; index 5 is gold. Token vectors are always length 6. */
export const COLORS: readonly Color[] = [0, 1, 2, 3, 4];
export const GOLD = 5;
export const HAND_LIMIT = 10;
export const RESERVE_LIMIT = 3;
export const WIN_POINTS = 15;
export const MARKET_SIZE = 4;
/** Consecutive non-purchase actions before the game settles on points. */
export const DRY_LIMIT = 40;

export const COLOR_KEYS = ['white', 'blue', 'green', 'red', 'black', 'gold'] as const;
export const COLOR_LABELS_ZH = ['白', '蓝', '绿', '红', '黑', '金'] as const;

export interface CardDef {
    id: number;
    tier: Tier;
    bonus: Color;
    points: number;
    cost: readonly [number, number, number, number, number];
}

export interface NobleDef {
    id: number;
    points: number;
    req: readonly [number, number, number, number, number];
}

export interface PlayerState {
    /** length 6: 5 colors + gold. */
    gems: number[];
    /** length 5: permanent discounts by color. */
    bonuses: number[];
    points: number;
    /** card ids in the reserved zone (max 3). */
    reserved: number[];
    /** card ids owned, acquisition order (tie-break uses its length). */
    purchased: number[];
    /** noble ids claimed by this player. */
    claimedNobles: number[];
}

export type Action =
    | { type: 'take'; take: number[]; discard?: number[] }
    | { type: 'reserve'; tier: Tier; index: number; discard?: number[] }
    | { type: 'buy'; source: 'market'; tier: Tier; index: number }
    | { type: 'buy'; source: 'reserved'; cardId: number }
    | { type: 'pass' };

export interface GameState {
    version: 1;
    seed: number;
    rng: number;
    /** length 6. */
    bank: number[];
    /** [tier][slot] -> card id or null (slot empty once the deck runs out). */
    market: (number | null)[][];
    /** [tier] remaining card ids; index 0 is the top of the deck. */
    decks: number[][];
    nobles: (number | null)[];
    players: [PlayerState, PlayerState];
    current: 0 | 1;
    /** number of completed actions (a full round = 2). */
    turn: number;
    /** consecutive passes — two in a row end a deadlocked game (see stalemate). */
    passes: number;
    /** actions since the last purchase; a long drought ends a deadlocked game. */
    dryTurns: number;
    finalRound: boolean;
    triggeredBy: 0 | 1 | null;
    over: boolean;
    /** seat index, or null for a draw (only possible after `over`). */
    winner: number | null;
    log: string[];
}

export type ApplyResult =
    | { ok: true; state: GameState }
    | { ok: false; error: string };

/* ── helpers ───────────────────────────────────────────────────────── */

const zero = (n: number): number[] => new Array(n).fill(0);

function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) | 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function cloneState(state: GameState): GameState {
    return structuredClone(state) as GameState;
}

export function totalGems(gems: readonly number[]): number {
    let n = 0;
    for (let i = 0; i < gems.length; i++) n += gems[i];
    return n;
}

export function cardOf(id: number): CardDef {
    const card = CARD_BY_ID.get(id);
    if (!card) throw new Error(`unknown card id ${id}`);
    return card;
}

export function nobleOf(id: number): NobleDef {
    const noble = NOBLE_BY_ID.get(id);
    if (!noble) throw new Error(`unknown noble id ${id}`);
    return noble;
}

/** How many of `cost[c]` the player still lacks after bonuses (gold NOT counted). */
export function gemDeficit(card: CardDef, player: PlayerState): number[] {
    const need: number[] = zero(5);
    for (let c = 0; c < 5; c++) need[c] = Math.max(0, card.cost[c] - player.bonuses[c]);
    return need;
}

/** How many of `cost[c]` the player still lacks after bonuses AND tokens. */
export function cardDeficit(card: CardDef, player: PlayerState): number[] {
    const need: number[] = zero(5);
    for (let c = 0; c < 5; c++) {
        need[c] = Math.max(0, card.cost[c] - player.bonuses[c] - player.gems[c]);
    }
    return need;
}

/** Tokens required from the bank (gold included) to buy the card; null = cannot afford. */
export function paymentFor(card: CardDef, player: PlayerState): number[] | null {
    const pay: number[] = zero(6);
    let goldNeeded = 0;
    for (let c = 0; c < 5; c++) {
        const need = Math.max(0, card.cost[c] - player.bonuses[c]);
        const fromGems = Math.min(need, player.gems[c]);
        pay[c] = fromGems;
        goldNeeded += need - fromGems;
    }
    if (goldNeeded > player.gems[GOLD]) return null;
    pay[GOLD] = goldNeeded;
    return pay;
}

export function canAfford(card: CardDef, player: PlayerState): boolean {
    return paymentFor(card, player) !== null;
}

/** Colors with at least one token in the bank. */
export function availableColors(state: GameState): Color[] {
    return COLORS.filter(c => state.bank[c] > 0);
}

/** How many distinct tokens a player may take this turn (1..3). */
export function takeSize(state: GameState): number {
    return Math.min(3, availableColors(state).length);
}

/** Whether the bank holds 4+ of a color, i.e. a "take two of the same" is legal. */
export function canTakeTwo(state: GameState, color: Color): boolean {
    return state.bank[color] >= 4;
}

function combinations(colors: readonly Color[], size: number): Color[][] {
    if (size <= 0) return [];
    const out: Color[][] = [];
    const pick: Color[] = [];
    const walk = (start: number): void => {
        if (pick.length === size) { out.push(pick.slice()); return; }
        for (let i = start; i < colors.length; i++) {
            pick.push(colors[i]);
            walk(i + 1);
            pick.pop();
        }
    };
    walk(0);
    return out;
}

/* ── setup ─────────────────────────────────────────────────────────── */

function shuffled<T>(items: readonly T[], rand: () => number): T[] {
    const arr = items.slice();
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
}

export function createGame(seed: number = (Date.now() >>> 0)): GameState {
    const rand = mulberry32(seed >>> 0);
    const decks: number[][] = [0, 1, 2].map(tier =>
        shuffled(CARDS.filter(c => c.tier === tier).map(c => c.id), rand),
    );
    const market: (number | null)[][] = decks.map(deck => {
        const slots: (number | null)[] = [];
        for (let i = 0; i < MARKET_SIZE; i++) slots.push(deck.shift() ?? null);
        return slots;
    });
    const nobles = shuffled(NOBLES.map(n => n.id), rand).slice(0, 3);
    const newPlayer = (): PlayerState => ({
        gems: zero(6),
        bonuses: zero(5),
        points: 0,
        reserved: [],
        purchased: [],
        claimedNobles: [],
    });
    return {
        version: 1,
        seed: seed >>> 0,
        rng: seed >>> 0,
        bank: [4, 4, 4, 4, 4, 5],
        market,
        decks,
        nobles,
        players: [newPlayer(), newPlayer()],
        current: 0,
        turn: 0,
        passes: 0,
        dryTurns: 0,
        finalRound: false,
        triggeredBy: null,
        over: false,
        winner: null,
        log: [],
    };
}

/* ── legality ──────────────────────────────────────────────────────── */

/** Take shapes a player may choose right now, as token vectors (no discards). */
export function legalTakes(state: GameState): number[][] {
    const colors = availableColors(state);
    const size = Math.min(3, colors.length);
    const takes: number[][] = [];
    for (const combo of combinations(colors, size)) {
        const take = zero(6);
        for (const c of combo) take[c] = 1;
        takes.push(take);
    }
    for (const c of colors) {
        if (canTakeTwo(state, c)) {
            const take = zero(6);
            take[c] = 2;
            takes.push(take);
        }
    }
    return takes;
}

export function affordableCards(state: GameState): Array<{ source: 'market'; tier: Tier; index: number } | { source: 'reserved'; cardId: number }> {
    const player = state.players[state.current];
    const out: Array<{ source: 'market'; tier: Tier; index: number } | { source: 'reserved'; cardId: number }> = [];
    for (let tier = 0; tier < 3; tier++) {
        for (let index = 0; index < MARKET_SIZE; index++) {
            const id = state.market[tier][index];
            if (id == null) continue;
            if (canAfford(cardOf(id), player)) out.push({ source: 'market', tier: tier as Tier, index });
        }
    }
    for (const cardId of player.reserved) {
        if (canAfford(cardOf(cardId), player)) out.push({ source: 'reserved', cardId });
    }
    return out;
}

export function reservableCards(state: GameState): Array<{ tier: Tier; index: number }> {
    if (state.players[state.current].reserved.length >= RESERVE_LIMIT) return [];
    const out: Array<{ tier: Tier; index: number }> = [];
    for (let tier = 0; tier < 3; tier++) {
        for (let index = 0; index < MARKET_SIZE; index++) {
            if (state.market[tier][index] != null) out.push({ tier: tier as Tier, index });
        }
        if (state.decks[tier].length > 0) out.push({ tier: tier as Tier, index: -1 });
    }
    return out;
}

/** Complete, directly applicable actions (takes over the hand limit get a disc). */
export function legalActions(state: GameState): Action[] {
    if (state.over) return [];
    const player = state.players[state.current];
    const actions: Action[] = [];
    for (const card of affordableCards(state)) {
        actions.push(card.source === 'market'
            ? { type: 'buy', source: 'market', tier: card.tier, index: card.index }
            : { type: 'buy', source: 'reserved', cardId: card.cardId });
    }
    for (const r of reservableCards(state)) {
        const after = player.gems.slice();
        if (state.bank[GOLD] > 0) after[GOLD] += 1;
        const overflow = totalGems(after) - HAND_LIMIT;
        if (overflow > 0) {
            const discard = autoDiscard(state, after, player, overflow);
            if (!discard) continue;
            actions.push({ type: 'reserve', tier: r.tier, index: r.index, discard });
        } else {
            actions.push({ type: 'reserve', tier: r.tier, index: r.index });
        }
    }
    for (const take of legalTakes(state)) {
        const after = player.gems.map((n, i) => n + take[i]);
        const overflow = totalGems(after) - HAND_LIMIT;
        if (overflow > 0) {
            const discard = autoDiscard(state, after, player, overflow);
            if (!discard) continue;
            actions.push({ type: 'take', take, discard });
        } else {
            actions.push({ type: 'take', take });
        }
    }
    if (actions.length === 0) actions.push({ type: 'pass' });
    return actions;
}

/**
 * Discard heuristic: give back the colors that no visible card needs, then
 * whatever is most abundant. Returns null when even discarding everything
 * overflowed (impossible in practice) — callers then drop the action.
 */
export function autoDiscard(
    state: GameState,
    gemsAfterTake: readonly number[],
    player: PlayerState,
    count: number,
): number[] | null {
    const need: number[] = zero(6);
    const visible = [
        ...state.market.flat().filter((id): id is number => id != null).map(cardOf),
        ...player.reserved.map(cardOf),
    ];
    for (const card of visible) {
        for (let c = 0; c < 5; c++) {
            const short = card.cost[c] - player.bonuses[c] - gemsAfterTake[c];
            if (short > 0) need[c] += short;
        }
    }
    // Gold is a wildcard: give every other color back before touching it.
    need[GOLD] = 1e6;
    const order = [0, 1, 2, 3, 4, 5]
        .filter(c => gemsAfterTake[c] > 0)
        .sort((a, b) => (need[a] - need[b]) || (gemsAfterTake[b] - gemsAfterTake[a]) || (a - b));
    const discard = zero(6);
    let left = count;
    for (const c of order) {
        if (left <= 0) break;
        const n = Math.min(left, gemsAfterTake[c]);
        discard[c] = n;
        left -= n;
    }
    return left === 0 ? discard : null;
}

/* ── transitions ───────────────────────────────────────────────────── */

function describe(card: CardDef): string {
    return `T${card.tier + 1}#${card.id}(${card.points}分)`;
}

/** Push a log line, keeping the tail bounded. */
function log(state: GameState, line: string): void {
    state.log.push(line);
    if (state.log.length > 40) state.log.splice(0, state.log.length - 40);
}

function validVector(vec: unknown, length: number): number[] | null {
    if (!Array.isArray(vec) || vec.length !== length) return null;
    const out: number[] = [];
    for (const n of vec) {
        if (typeof n !== 'number' || !Number.isFinite(n) || n < 0 || !Number.isInteger(n)) return null;
        out.push(n);
    }
    return out;
}

function takeShapeError(state: GameState, take: number[]): string | null {
    if (take[GOLD] !== 0) return '不能直接拿金币（金币只能通过保留卡获得）';
    const picked = COLORS.filter(c => take[c] > 0);
    if (picked.length === 0) return '没有选择任何宝石';
    const doubled = picked.filter(c => take[c] >= 2);
    if (doubled.length > 0) {
        if (picked.length > 1 || take[doubled[0]] !== 2) return '拿取形状非法：同色 2 枚不能与其他颜色混拿';
        if (!canTakeTwo(state, doubled[0])) return `${COLOR_LABELS_ZH[doubled[0]]}色库存不足 4 枚，不能同色拿 2`;
        return null;
    }
    const size = takeSize(state);
    if (picked.length !== size) {
        return size === 3
            ? '必须拿 3 个不同颜色（或同色 2 枚，库存需 ≥4）'
            : `银行只剩 ${size} 种颜色，需拿 ${size} 个不同颜色`;
    }
    return null;
}

function applyTokenDelta(state: GameState, player: PlayerState, add: number[], sub: number[]): void {
    for (let c = 0; c < 6; c++) {
        player.gems[c] += add[c] - sub[c];
        state.bank[c] += sub[c] - add[c];
    }
}

function checkDiscard(player: PlayerState, afterGems: readonly number[], discard: number[] | undefined): string | null {
    const excess = totalGems(afterGems) - HAND_LIMIT;
    if (excess <= 0) {
        if (discard && totalGems(discard) > 0) return '无需弃牌';
        return null;
    }
    if (!discard) return `超过 ${HAND_LIMIT} 枚上限，必须弃回 ${excess} 枚`;
    if (totalGems(discard) !== excess) return `必须正好弃回 ${excess} 枚`;
    for (let c = 0; c < 6; c++) {
        if (discard[c] > afterGems[c]) return '弃牌数量超过持有量';
    }
    return null;
}

function advance(state: GameState): void {
    state.turn += 1;
    state.current = state.current === 0 ? 1 : 0;
    if (state.finalRound && state.triggeredBy !== null && state.current === state.triggeredBy) {
        finish(state);
        return;
    }
    // Safety valve: a 20-round buying drought means both players are shuffling
    // tokens (bank drained / nothing affordable). Settle on points instead of
    // looping forever — real games almost always buy within a few rounds.
    if (state.dryTurns >= DRY_LIMIT) {
        log(state, `连续 ${DRY_LIMIT} 手无人购牌，按当前分数结算`);
        finish(state);
    }
}

function finish(state: GameState): void {
    state.over = true;
    const [a, b] = state.players;
    if (a.points !== b.points) {
        state.winner = a.points > b.points ? 0 : 1;
    } else {
        // Official tie-break: fewest development cards wins.
        const ca = a.purchased.length;
        const cb = b.purchased.length;
        state.winner = ca === cb ? null : (ca < cb ? 0 : 1);
    }
    const score = `${a.points} : ${b.points}`;
    log(state, state.winner === null
        ? `平局！最终比分 ${score}`
        : `玩家 ${state.winner + 1} 获胜！最终比分 ${score}`);
}

/** Validate + apply one action. Never mutates the input state. */
export function applyAction(state: GameState, action: Action): ApplyResult {
    if (state.over) return { ok: false, error: '对局已结束' };
    const next = cloneState(state);
    const me = next.players[next.current];
    const seat = next.current;

    if (action.type === 'take') {
        const take = validVector(action.take, 6);
        if (!take) return { ok: false, error: '拿取向量非法' };
        const shapeError = takeShapeError(next, take);
        if (shapeError) return { ok: false, error: shapeError };
        for (let c = 0; c < 6; c++) {
            if (take[c] > next.bank[c]) return { ok: false, error: '银行库存不足' };
        }
        let discard: number[] | undefined;
        if (action.discard) {
            const parsed = validVector(action.discard, 6);
            if (!parsed) return { ok: false, error: '弃牌向量非法' };
            discard = parsed;
        }
        const after = me.gems.map((n, i) => n + take[i]);
        const discardError = checkDiscard(me, after, discard);
        if (discardError) return { ok: false, error: discardError };
        applyTokenDelta(next, me, take, discard ?? zero(6));
        const parts = COLORS.filter(c => take[c] > 0).map(c => `${COLOR_LABELS_ZH[c]}${take[c] > 1 ? '×2' : ''}`);
        log(next, `玩家 ${seat + 1} 拿取 ${parts.join(' + ')}${discard && totalGems(discard) > 0 ? '（并弃牌）' : ''}`);
        next.passes = 0;
        next.dryTurns += 1;
        advance(next);
        return { ok: true, state: next };
    }

    if (action.type === 'reserve') {
        const { tier, index } = action;
        if (tier !== 0 && tier !== 1 && tier !== 2) return { ok: false, error: '无效的牌堆层级' };
        if (me.reserved.length >= RESERVE_LIMIT) return { ok: false, error: `保留区已满（最多 ${RESERVE_LIMIT} 张）` };
        let cardId: number | null = null;
        if (index === -1) {
            cardId = next.decks[tier][0] ?? null;
            if (cardId == null) return { ok: false, error: '该层牌堆已空' };
            next.decks[tier].shift();
        } else {
            if (index < 0 || index >= MARKET_SIZE) return { ok: false, error: '无效的市场位置' };
            cardId = next.market[tier][index];
            if (cardId == null) return { ok: false, error: '该位置没有卡牌' };
            next.market[tier][index] = next.decks[tier].shift() ?? null;
        }
        me.reserved.push(cardId);
        // Gold bonus for reserving (only if the bank still has gold).
        const goldGained = next.bank[GOLD] > 0 ? 1 : 0;
        if (goldGained) {
            next.bank[GOLD] -= 1;
            me.gems[GOLD] += 1;
        }
        let discard: number[] | undefined;
        if (action.discard) {
            const parsed = validVector(action.discard, 6);
            if (!parsed) return { ok: false, error: '弃牌向量非法' };
            discard = parsed;
        }
        const discardError = checkDiscard(me, me.gems, discard);
        if (discardError) return { ok: false, error: discardError };
        if (discard) applyTokenDelta(next, me, zero(6), discard);
        log(next, `玩家 ${seat + 1} 保留 ${describe(cardOf(cardId))}${goldGained ? '（+1 金币）' : ''}`);
        next.passes = 0;
        next.dryTurns += 1;
        advance(next);
        return { ok: true, state: next };
    }

    if (action.type === 'buy') {
        let cardId: number | null = null;
        if (action.source === 'market') {
            const { tier, index } = action;
            if (tier !== 0 && tier !== 1 && tier !== 2) return { ok: false, error: '无效的牌堆层级' };
            if (index < 0 || index >= MARKET_SIZE) return { ok: false, error: '无效的市场位置' };
            cardId = next.market[tier][index];
            if (cardId == null) return { ok: false, error: '该位置没有卡牌' };
        } else {
            cardId = action.cardId;
            if (!me.reserved.includes(cardId)) return { ok: false, error: '该卡不在你的保留区' };
        }
        const card = cardOf(cardId);
        const pay = paymentFor(card, me);
        if (!pay) return { ok: false, error: '宝石不足，买不起这张卡' };
        for (let c = 0; c < 6; c++) {
            if (pay[c] > me.gems[c]) return { ok: false, error: '宝石不足' };
            me.gems[c] -= pay[c];
            next.bank[c] += pay[c];
        }
        if (action.source === 'market') {
            next.market[action.tier][action.index] = next.decks[action.tier].shift() ?? null;
        } else {
            me.reserved = me.reserved.filter(id => id !== cardId);
        }
        me.purchased.push(cardId);
        me.bonuses[card.bonus] += 1;
        me.points += card.points;
        log(next, `玩家 ${seat + 1} 购买 ${describe(card)}`);

        // Nobles visit after a purchase only; several may arrive at once.
        for (let i = 0; i < next.nobles.length; i++) {
            const nobleId = next.nobles[i];
            if (nobleId == null) continue;
            const noble = nobleOf(nobleId);
            const visited = noble.req.every((req, c) => me.bonuses[c] >= req);
            if (!visited) continue;
            next.nobles[i] = null;
            me.points += noble.points;
            me.claimedNobles.push(nobleId);
            log(next, `玩家 ${seat + 1} 获得贵族 #${nobleId}（+${noble.points} 分）`);
        }

        if (!next.finalRound && me.points >= WIN_POINTS) {
            next.finalRound = true;
            next.triggeredBy = seat;
            log(next, `玩家 ${seat + 1} 达到 ${me.points} 分，本轮结束后结算`);
        }
        next.passes = 0;
        next.dryTurns = 0;
        advance(next);
        return { ok: true, state: next };
    }

    if (action.type === 'pass') {
        // Both players out of options (bank drained, nothing affordable): two
        // consecutive passes settle the game on points instead of looping.
        log(next, `玩家 ${seat + 1} 无合法动作，跳过`);
        next.passes += 1;
        next.dryTurns += 1;
        advance(next);
        if (next.passes >= 2 && !next.over) {
            log(next, '双方连续跳过，按当前分数结算');
            finish(next);
        }
        return { ok: true, state: next };
    }

    return { ok: false, error: '未知动作类型' };
}

/* ── read-only views ───────────────────────────────────────────────── */

export interface ScoreLine {
    seat: number;
    points: number;
    cards: number;
    nobles: number;
}

export function scores(state: GameState): ScoreLine[] {
    return state.players.map((p, seat) => ({
        seat,
        points: p.points,
        cards: p.purchased.length,
        nobles: p.claimedNobles.length,
    }));
}

export function totalCards(): number {
    return CARDS.length;
}

export function totalNobles(): number {
    return NOBLES.length;
}
