/**
 * Rule-based Splendor opponent — no ML. Every legal action is simulated with the
 * same `applyAction` the UI uses, then the resulting position is scored by a
 * static evaluation: buy points when possible, otherwise close the gap to the
 * best reachable card, keep hand pressure low, and grab gold by reserving when
 * it is cheap. Swapping in an RL policy later only means replacing chooseAction.
 */
import {
    applyAction, canAfford, cardDeficit, cardOf, GOLD, legalActions, totalGems,
    type Action, type GameState, type PlayerState,
} from './rules.ts';

function distinctGems(player: PlayerState): number {
    let n = 0;
    for (let c = 0; c < 5; c++) if (player.gems[c] > 0) n++;
    return n;
}

function bonusTotal(player: PlayerState): number {
    let n = 0;
    for (let c = 0; c < 5; c++) n += player.bonuses[c];
    return n;
}

/** Static evaluation of `state` from `seat`'s point of view. */
export function evaluate(state: GameState, seat: number): number {
    const me = state.players[seat];
    const opp = state.players[1 - seat];
    let score = me.points * 120 - opp.points * 50;
    score += bonusTotal(me) * 9;
    score += distinctGems(me) * 2;
    score += me.gems[GOLD] * 5;
    score -= Math.max(0, totalGems(me.gems) - 8) * 3;

    const visible = [
        ...state.market.flat().filter((id): id is number => id != null).map(cardOf),
        ...me.reserved.map(cardOf),
    ];
    for (const card of visible) {
        if (canAfford(card, me)) {
            // Points now are worth real value; the accompanying bonus adds a little.
            score += card.points * 14 + 5;
        } else {
            const deficit = cardDeficit(card, me).reduce((a, b) => a + b, 0);
            if (deficit <= 4) score += card.points * 5 - deficit * 3;
        }
    }
    return score;
}

/**
 * Pick an action for the player to move. Candidates come from `legalActions`
 * (already complete, including forced discards), so AI moves can never be legal
 * in the engine's eyes.
 */
export function chooseAction(state: GameState): Action | null {
    if (state.over) return null;
    const seat = state.current;
    const candidates = legalActions(state);
    if (candidates.length === 0) return null;

    let best: Action | null = null;
    let bestScore = -Infinity;
    for (const action of candidates) {
        const result = applyAction(state, action);
        if (!result.ok) continue;
        // `advance` flips `current`, but the position still belongs to `seat`.
        const score = evaluate(result.state, seat);
        if (score > bestScore) {
            bestScore = score;
            best = action;
        }
    }
    return best;
}
