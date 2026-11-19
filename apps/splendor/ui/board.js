/**
 * Splendor board — app DOM UI (loaded by BOTH the editor and player runtimes).
 *
 * The UI is intentionally "dumb": it renders the view published by the splendor
 * plugin (`splendor:state`) and sends semantic actions back (`splendor:action`).
 * All legality checks live in the plugin, so nothing here can desync the rules.
 *
 * No imports: app UI scripts are fetched and Blob-imported raw (no bundler), so
 * this file must be self-contained plain ES2020 JavaScript.
 */
export function mount(container, host) {
    const bus = host.eventBus;

    /* ── styles ── */
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement('div');
    root.className = 'sp-root';
    container.appendChild(root);

    /* ── helpers (declared before the state that uses them) ── */
    const zero = n => new Array(n).fill(0);
    const sum = arr => arr.reduce((a, b) => a + b, 0);
    const base = () => (view && view.base) || '/apps/splendor';
    const asset = path => `${base()}/assets/${path}`;
    const me = () => view.state.players[view.state.current];
    const busy = () => !view || view.thinking || view.state.over;

    /* ── local state ── */
    let view = null;
    let picks = zero(6);
    let discards = zero(6);
    let selected = null;   // {source:'market'|'reserved'|'deck', tier?, index?, cardId?}
    let pending = null;    // {action, need} — waiting for a forced discard
    let note = '';
    let noteTimer = null;

    const CARD = new Map();
    const NOBLE = new Map();
    const COLORS = [0, 1, 2, 3, 4];
    const COLOR_ZH = ['白', '蓝', '绿', '红', '黑', '金'];
    const HAND_LIMIT = 10;

    function toast(text) {
        note = text;
        if (noteTimer) clearTimeout(noteTimer);
        noteTimer = setTimeout(() => { note = ''; noteTimer = null; render(); }, 2600);
        render();
    }

    function resetSelection() {
        picks = zero(6);
        discards = zero(6);
        selected = null;
        pending = null;
    }

    function canAddGem(color) {
        if (view.state.bank[color] <= 0) return false;
        if (picks[color] !== 0) return true;
        if (picks.some(n => n >= 2)) return false;
        const distinct = picks.filter(n => n > 0).length;
        return distinct < (view.takeSize || 3);
    }

    function canDouble(color) {
        const distinct = picks.filter(n => n > 0).length;
        return picks[color] === 1 && distinct === 1 && !!view.canTakeTwo[color] && view.state.bank[color] >= 2;
    }

    /* ── render ── */
    function render() {
        if (!view) {
            root.innerHTML = '<div class="sp-loading">载入 Splendor…</div>';
            return;
        }
        CARD.clear();
        for (const c of view.cards) CARD.set(c.id, c);
        NOBLE.clear();
        for (const n of view.nobles) NOBLE.set(n.id, n);
        root.style.setProperty('--table', `url("${asset('table.webp')}")`);

        const s = view.state;
        root.innerHTML = [
            headerHtml(s),
            `<div class="sp-main">`,
            bankColHtml(s),
            marketHtml(s),
            sideHtml(s),
            `</div>`,
            trayHtml(s),
            s.over ? overHtml(s) : '',
            note ? `<div class="sp-note">${note}</div>` : '',
        ].join('');
    }

    function headerHtml(s) {
        const turnNo = Math.floor(s.turn / 2) + 1;
        const turnText = view.thinking
            ? 'AI 思考中…'
            : (s.over ? '对局结束' : `轮到 玩家 ${s.current + 1}`);
        const sc = view.scores.map(x =>
            `<span class="sp-sc${x.seat === s.current && !s.over ? ' on' : ''}">P${x.seat + 1} <b>${x.points}</b></span>`).join('');
        return `<header class="sp-head">
            <div class="sp-brand">SPLENDOR<span>ShaderLab-WebGPU · 引擎上的第一个完整游戏</span></div>
            <div class="sp-turn">${turnText} · 第 ${turnNo} 回合</div>
            <div class="sp-scores">${sc}</div>
            <div class="sp-modes">
                <button data-act="mode" data-ai="0" class="${view.aiSeats.length === 0 ? 'on' : ''}">双人热座</button>
                <button data-act="mode" data-ai="1" class="${view.aiSeats.length > 0 ? 'on' : ''}">人机对战</button>
                <button data-act="new">新开局</button>
            </div>
        </header>`;
    }

    function marketHtml(s) {
        return `<section class="sp-market">
            ${noblesHtml(s)}
            ${[2, 1, 0].map(tier => tierHtml(s, tier)).join('')}
        </section>`;
    }

    function noblesHtml(s) {
        return `<div class="sp-nobles">
            <div class="sp-label">贵族</div>
            ${s.nobles.map(id => {
                if (id == null) return `<div class="sp-noble gone"><span>已被迎接</span></div>`;
                const noble = NOBLE.get(id);
                return `<div class="sp-noble" title="贵族 #${id} · ${noble.points} 分 · 需要 ${noble.req.map((n, c) => n ? COLOR_ZH[c] + n : '').filter(Boolean).join(' ')}">
                    <img src="${asset(`nobles/${id}.webp`)}" alt="noble ${id}" draggable="false">
                </div>`;
            }).join('')}
        </div>`;
    }

    function bankColHtml(s) {
        const gems = COLORS.map(c => {
            const n = s.bank[c];
            const picked = picks[c] || 0;
            return `<div class="sp-gem${n === 0 ? ' off' : ''}${picked ? ' picked' : ''}" data-act="gem" data-color="${c}" title="${COLOR_ZH[c]}：银行剩余 ${n}">
                <img src="${asset(`gems/${c}.png`)}" alt="" draggable="false">
                <span class="sp-gem-n">${n}</span>
                ${picked ? `<span class="sp-gem-p">${picked}</span>` : ''}
            </div>`;
        }).join('');
        return `<aside class="sp-bankcol">
            <div class="sp-label">银行</div>
            <div class="sp-gems">${gems}
                <div class="sp-gem gold" title="金币：只能通过保留卡获得"><img src="${asset('gems/5.png')}" alt="" draggable="false"><span class="sp-gem-n">${s.bank[5]}</span></div>
            </div>
        </aside>`;
    }

    function tierHtml(s, tier) {
        const remaining = s.decks[tier].length;
        const deck = `<div class="sp-deck${view.canReserve && remaining > 0 && !busy() ? ' live' : ''}" data-act="deck" data-tier="${tier}" title="保留牌堆顶（+1 金币）">
            <img src="${asset(`cards/${tier}/cover.webp`)}" alt="" draggable="false">
            <span class="sp-deck-n">${remaining}</span>
        </div>`;
        const cards = s.market[tier].map((id, index) => id == null
            ? `<div class="sp-empty"></div>`
            : cardHtml(id, tier, index)).join('');
        return `<div class="sp-tier">${deck}${cards}</div>`;
    }

    function costDots(cost) {
        return cost.map((n, c) => n > 0 ? `<i class="sp-dot c${c}">${n}</i>` : '').join('');
    }

    function cardHtml(id, tier, index) {
        const def = CARD.get(id);
        const isSel = selected && selected.source === 'market' && selected.cardId === id;
        const afford = !!view.canBuy[id];
        return `<div class="sp-card${isSel ? ' sel' : ''}${afford ? ' buy' : ''}"
                data-act="card" data-source="market" data-tier="${tier}" data-index="${index}" data-card="${id}"
                title="T${tier + 1} #${id} · ${def.points} 分 · ${def.cost.map((n, c) => n ? COLOR_ZH[c] + n : '').filter(Boolean).join(' ')}">
            <img src="${asset(`cards/${tier}/${id}.webp`)}" alt="card ${id}" draggable="false">
        </div>`;
    }

    function sideHtml(s) {
        return `<aside class="sp-side">
            ${playerHtml(s, 0)}
            ${playerHtml(s, 1)}
            <div class="sp-log"><div class="sp-label">对局记录</div>
                ${s.log.slice(-7).reverse().map(line => `<div class="sp-logline">${line}</div>`).join('') || '<div class="sp-logline dim">刚刚开始</div>'}
            </div>
        </aside>`;
    }

    function playerHtml(s, seat) {
        const p = s.players[seat];
        const sc = view.scores[seat];
        const active = s.current === seat && !s.over;
        const isAI = view.aiSeats.includes(seat);
        const canDiscardThis = !!pending && active;

        const gems = p.gems.map((n, c) => n === 0 ? '' : `<div class="sp-own${discards[c] ? ' disc' : ''}"
                ${canDiscardThis ? `data-act="own-gem" data-color="${c}"` : ''}
                title="${COLOR_ZH[c]} ×${n}">
                <img src="${asset(`gems/${c}.png`)}" alt="" draggable="false">
                <span>${n}</span>${discards[c] ? `<b>-${discards[c]}</b>` : ''}
            </div>`).join('') || '<span class="sp-none">无宝石</span>';

        const bonuses = p.bonuses.map((n, c) => n === 0 ? '' : `<i class="sp-chip c${c}">${COLOR_ZH[c]}${n}</i>`).join('')
            || '<span class="sp-none">无折扣</span>';

        const reserved = p.reserved.map(id => {
            const def = CARD.get(id);
            const afford = !!view.canBuy[id];
            const isSel = selected && selected.source === 'reserved' && selected.cardId === id;
            return `<div class="sp-mini${afford ? ' buy' : ''}${isSel ? ' sel' : ''}" data-act="reserved" data-card="${id}"
                    title="保留卡 T${def.tier + 1} #${id} · ${def.points} 分">
                <img src="${asset(`cards/${def.tier}/${id}.webp`)}" alt="" draggable="false">
            </div>`;
        }).join('') || '<span class="sp-none">无保留卡</span>';

        return `<section class="sp-player${active ? ' active' : ''}">
            <div class="sp-phead">
                <span class="sp-pname">玩家 ${seat + 1}${isAI ? ' · AI' : ''}</span>
                <span class="sp-ppt"><b>${sc.points}</b> 分</span>
                <span class="sp-pmeta">卡 ${sc.cards} · 贵族 ${sc.nobles}</span>
            </div>
            <div class="sp-prow"><span class="sp-plabel">宝石</span>${gems}</div>
            <div class="sp-prow"><span class="sp-plabel">折扣</span>${bonuses}</div>
            <div class="sp-prow"><span class="sp-plabel">保留</span>${reserved}</div>
        </section>`;
    }

    function trayHtml(s) {
        if (s.over) return '';
        if (view.thinking) {
            return `<div class="sp-tray"><div class="sp-hint">对手（AI）正在思考…</div></div>`;
        }
        if (pending) {
            const picked = discards.map((n, c) => n > 0 ? `<span class="sp-pick c${c}" data-act="undiscard" data-color="${c}">${COLOR_ZH[c]} ×${n}</span>` : '').join('');
            const ok = sum(discards) === pending.need;
            return `<div class="sp-tray sp-tray-discard">
                <div class="sp-hint">手牌上限 ${HAND_LIMIT} 枚：请点击自己面板中的宝石，弃回 <b>${pending.need}</b> 枚</div>
                <div class="sp-picks">${picked || '<span class="sp-none">未选择</span>'}</div>
                <div class="sp-btns">
                    <button class="sp-primary" data-act="confirm-discard" ${ok ? '' : 'disabled'}>确认弃牌</button>
                    <button data-act="clear">取消</button>
                </div>
            </div>`;
        }
        if (selected) return selectedTrayHtml();
        if (sum(picks) > 0) {
            const picked = picks.map((n, c) => n > 0 ? `<span class="sp-pick c${c}">${COLOR_ZH[c]} ×${n}</span>` : '').join('');
            return `<div class="sp-tray">
                <div class="sp-hint">已选宝石</div>
                <div class="sp-picks">${picked}</div>
                <div class="sp-btns">
                    <button class="sp-primary" data-act="confirm-take">确认拿取</button>
                    <button data-act="clear">清空</button>
                </div>
            </div>`;
        }
        return `<div class="sp-tray"><div class="sp-hint">
            点击银行宝石选择拿取（最多 ${view.takeSize} 种不同色；同色 2 枚需库存 ≥ 4）· 点击卡牌可购买 / 保留
        </div></div>`;
    }

    function selectedTrayHtml() {
        const s = view.state;
        const isDeck = selected.source === 'deck';
        const def = isDeck ? null : CARD.get(selected.cardId);
        const afford = !isDeck && !!view.canBuy[selected.cardId];
        const title = isDeck
            ? `保留 T${(selected.tier || 0) + 1} 牌堆顶`
            : `T${def.tier + 1} #${selected.cardId} · ${def.points} 分`;
        const cost = def ? `<span class="sp-cost">${costDots(def.cost)}</span>` : '';
        return `<div class="sp-tray">
            <div class="sp-selinfo"><span class="sp-seltitle">${title}</span>${cost}</div>
            <div class="sp-btns">
                ${isDeck ? '' : `<button class="sp-primary" data-act="buy" ${afford ? '' : 'disabled'}>购买</button>`}
                ${view.canReserve ? `<button data-act="reserve" ${isDeck ? 'class="sp-primary"' : ''}>保留（+1 金币）</button>` : ''}
                <button data-act="cancel">取消</button>
            </div>
        </div>`;
    }

    function overHtml(s) {
        const lines = view.scores.map(x => `<div class="sp-fin${s.winner === x.seat ? ' win' : ''}">
            <span>玩家 ${x.seat + 1}${view.aiSeats.includes(x.seat) ? '（AI）' : ''}</span>
            <b>${x.points}</b><span class="dim">分 · ${x.cards} 张卡 · ${x.nobles} 位贵族</span>
        </div>`).join('');
        const title = s.winner === null ? '平局' : (view.aiSeats.includes(s.winner) ? 'AI 获胜' : `玩家 ${s.winner + 1} 获胜`);
        return `<div class="sp-over"><div class="sp-overcard">
            <h2>${title}</h2>${lines}
            <button class="sp-primary" data-act="new">再来一局</button>
        </div></div>`;
    }

    /* ── interactions ── */
    function onGem(color) {
        if (busy()) return;
        if (pending) return;
        if (picks[color] === 0) {
            if (!canAddGem(color)) { toast(`最多选 ${view.takeSize} 种不同颜色的宝石`); return; }
            picks[color] = 1;
        } else if (canDouble(color)) {
            picks[color] = 2;
        } else {
            picks[color] = 0;
        }
        selected = null;
        render();
    }

    function onOwnGem(color) {
        if (!pending) return;
        const gained = pending.action.type === 'reserve' && view.state.bank[5] > 0 && color === 5 ? 1 : 0;
        const takeAdd = pending.action.take ? pending.action.take[color] : 0;
        const available = me().gems[color] + takeAdd + gained - discards[color];
        if (sum(discards) >= pending.need || available <= 0) return;
        discards[color] += 1;
        render();
    }

    function onUndiscard(color) {
        if (!pending || discards[color] <= 0) return;
        discards[color] -= 1;
        render();
    }

    function onConfirmTake() {
        if (busy() || sum(picks) === 0) return;
        const take = picks.slice();
        const after = me().gems.map((n, i) => n + take[i]);
        const overflow = sum(after) - HAND_LIMIT;
        if (overflow > 0) {
            pending = { action: { type: 'take', take }, need: overflow };
            picks = zero(6);
            discards = zero(6);
            render();
            return;
        }
        submit({ type: 'take', take });
    }

    function submit(action) {
        bus.emit('splendor:action', action);
        resetSelection();
        render();
    }

    function onBuy() {
        if (busy() || !selected) return;
        if (selected.source === 'reserved') {
            submit({ type: 'buy', source: 'reserved', cardId: selected.cardId });
        } else if (selected.source === 'market') {
            submit({ type: 'buy', source: 'market', tier: selected.tier, index: selected.index });
        }
    }

    function onReserve() {
        if (busy() || !selected) return;
        const action = { type: 'reserve', tier: selected.tier, index: selected.source === 'deck' ? -1 : selected.index };
        const goldGain = view.state.bank[5] > 0 ? 1 : 0;
        const after = me().gems.slice();
        after[5] += goldGain;
        const overflow = sum(after) - HAND_LIMIT;
        if (overflow > 0) {
            pending = { action, need: overflow };
            discards = zero(6);
            selected = null;
            render();
            return;
        }
        submit(action);
    }

    function onConfirmDiscard() {
        if (!pending || sum(discards) !== pending.need) return;
        submit({ ...pending.action, discard: discards.slice() });
    }

    function onClear() {
        resetSelection();
        render();
    }

    /* ── wiring ── */
    const offState = bus.on('splendor:state', v => { view = v; render(); });
    const offError = bus.on('splendor:error', e => toast((e && e.message) || '操作被拒绝'));

    function onClick(event) {
        const el = event.target.closest('[data-act]');
        if (!el || !view) return;
        const act = el.dataset.act;
        if (el.hasAttribute('disabled')) return;
        switch (act) {
            case 'gem': onGem(Number(el.dataset.color)); break;
            case 'card':
                if (busy()) return;
                selected = { source: 'market', tier: Number(el.dataset.tier), index: Number(el.dataset.index), cardId: Number(el.dataset.card) };
                picks = zero(6);
                render();
                break;
            case 'deck':
                if (busy()) return;
                selected = { source: 'deck', tier: Number(el.dataset.tier) };
                picks = zero(6);
                render();
                break;
            case 'reserved':
                if (busy()) return;
                selected = { source: 'reserved', cardId: Number(el.dataset.card) };
                picks = zero(6);
                render();
                break;
            case 'own-gem': onOwnGem(Number(el.dataset.color)); break;
            case 'undiscard': onUndiscard(Number(el.dataset.color)); break;
            case 'confirm-take': onConfirmTake(); break;
            case 'confirm-discard': onConfirmDiscard(); break;
            case 'buy': onBuy(); break;
            case 'reserve': onReserve(); break;
            case 'clear': onClear(); break;
            case 'cancel': selected = null; render(); break;
            case 'new': bus.emit('splendor:new', { seed: Date.now() >>> 0 }); resetSelection(); break;
            case 'mode':
                bus.emit('splendor:mode', { ai: el.dataset.ai === '1' });
                resetSelection();
                break;
            default: break;
        }
    }

    function onKey(event) {
        if (event.key === 'Escape') {
            if (sum(picks) > 0 || pending || selected) { onClear(); event.preventDefault(); }
        } else if (event.key === 'Enter') {
            if (pending && sum(discards) === pending.need) { onConfirmDiscard(); event.preventDefault(); }
            else if (!pending && !selected && sum(picks) > 0) { onConfirmTake(); event.preventDefault(); }
        }
    }

    root.addEventListener('click', onClick);
    document.addEventListener('keydown', onKey);
    bus.emit('splendor:request');
    render();

    return function unmount() {
        root.removeEventListener('click', onClick);
        document.removeEventListener('keydown', onKey);
        offState();
        offError();
        if (noteTimer) clearTimeout(noteTimer);
        style.remove();
        root.remove();
    };
}

const CSS = `
.sp-root{position:fixed;inset:0;display:flex;flex-direction:column;gap:8px;box-sizing:border-box;
    --cardH:clamp(104px,19vh,190px);
    padding:10px 14px 12px;color:#f6ecd8;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
    background:linear-gradient(rgba(16,10,6,.55),rgba(16,10,6,.72)), var(--table) center/cover no-repeat, #221710;
    overflow:hidden;pointer-events:auto;user-select:none;}
.sp-root *{pointer-events:auto;box-sizing:border-box;}
.sp-root img{display:block;}
.sp-label{font-size:11px;letter-spacing:.16em;color:#c8a86a;text-transform:uppercase;}
.sp-none{font-size:11px;color:#7d6a52;}
.sp-loading{position:absolute;inset:0;display:grid;place-items:center;font-size:15px;color:#c8a86a;}

/* the shell shows a small overlay badge in the same corner as our header; hide it
   in the player page only (the editor has #sidebar and its own info readout) */
html:not(:has(#sidebar)) #info{display:none !important;}

/* header */
.sp-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:2px 2px 6px;}
.sp-brand{font-weight:800;font-size:20px;letter-spacing:.22em;color:#f1d9a6;text-shadow:0 2px 6px #000;}
.sp-brand span{display:block;font-size:9px;letter-spacing:.14em;color:#9c7f56;font-weight:500;text-shadow:none;}
.sp-turn{padding:5px 12px;border-radius:999px;background:rgba(0,0,0,.42);border:1px solid #6b5127;font-size:13px;color:#f0dfb8;}
.sp-scores{display:flex;gap:8px;}
.sp-sc{padding:4px 10px;border-radius:8px;background:rgba(0,0,0,.35);border:1px solid #4b3a22;font-size:12px;color:#cbb894;}
.sp-sc b{font-size:15px;color:#f5e6c0;}
.sp-sc.on{border-color:#d8b25e;background:rgba(216,178,94,.18);}
.sp-modes{margin-left:auto;display:flex;gap:6px;}
.sp-modes button,.sp-btns button,.sp-overcard button{font:inherit;font-size:12px;padding:6px 12px;border-radius:8px;cursor:pointer;
    background:rgba(30,22,14,.85);color:#e8d7b3;border:1px solid #5c4626;}
.sp-modes button:hover,.sp-btns button:hover{background:rgba(70,52,28,.9);}
.sp-modes button.on{border-color:#d8b25e;color:#ffe9b6;background:rgba(216,178,94,.2);}
button:disabled{opacity:.4;cursor:not-allowed;}
button.sp-primary{background:linear-gradient(#c79a4b,#9c6f27);color:#231708;border-color:#e3c07a;font-weight:700;}
button.sp-primary:disabled{background:rgba(60,45,25,.8);color:#9d8767;}

/* layout */
.sp-main{flex:1 1 auto;display:grid;grid-template-columns:104px minmax(0,1fr) 322px;gap:12px;min-height:0;}
.sp-market{display:flex;flex-direction:column;justify-content:center;gap:12px;min-width:0;min-height:0;}
.sp-side{display:flex;flex-direction:column;gap:8px;min-height:0;overflow:auto;padding-right:2px;}
.sp-bankcol{display:flex;flex-direction:column;align-items:center;justify-content:center;align-self:center;gap:8px;padding:10px 4px;
    border-radius:12px;background:rgba(0,0,0,.32);border:1px solid rgba(120,90,45,.5);}
.sp-bankcol .sp-gems{flex-direction:column;align-items:center;gap:9px;}
@media (max-width:1180px){.sp-main{grid-template-columns:1fr;grid-template-rows:auto 1fr auto;overflow:auto;}
    .sp-bankcol{flex-direction:row;gap:14px;padding:6px 10px;}.sp-bankcol .sp-gems{flex-direction:row;}}

/* nobles + bank */
.sp-nobles{display:flex;align-items:center;justify-content:center;gap:10px;}
.sp-noble{position:relative;width:66px;height:66px;border-radius:9px;overflow:hidden;background:#191109;
    border:1px solid #6b5127;box-shadow:0 4px 10px rgba(0,0,0,.45);}
.sp-noble img{width:100%;height:100%;object-fit:cover;}
.sp-noble.gone{display:grid;place-items:center;font-size:9px;color:#6d5a40;border-style:dashed;}
.sp-gems{display:flex;gap:8px;flex-wrap:wrap;}
.sp-gem{position:relative;width:46px;height:46px;border-radius:50%;cursor:pointer;transition:transform .12s;
    box-shadow:0 3px 8px rgba(0,0,0,.5);}
.sp-gem img{width:100%;height:100%;object-fit:contain;}
.sp-gem:hover{transform:translateY(-2px) scale(1.04);}
.sp-gem.off{filter:grayscale(1) brightness(.55);cursor:not-allowed;}
.sp-gem.gold{cursor:default;}
.sp-gem-n{position:absolute;right:-3px;bottom:-3px;min-width:17px;height:17px;border-radius:9px;background:#16100a;
    border:1px solid #8a6a33;color:#ffe9b6;font-size:11px;display:grid;place-items:center;padding:0 3px;}
.sp-gem.picked{outline:2px solid #4fd67c;outline-offset:2px;}
.sp-gem-p{position:absolute;left:-3px;top:-3px;width:18px;height:18px;border-radius:9px;background:#4fd67c;color:#08210f;
    font-size:11px;font-weight:800;display:grid;place-items:center;}

/* tiers + cards */
.sp-tier{display:grid;grid-template-columns:calc(var(--cardH,124px)*.75) repeat(4,calc(var(--cardH,124px)*.75));
    gap:10px;justify-content:center;align-items:start;}
.sp-deck{position:relative;height:var(--cardH,124px);width:calc(var(--cardH,124px)*.75);border-radius:8px;overflow:hidden;
    border:1px solid #57401f;box-shadow:0 4px 10px rgba(0,0,0,.5);cursor:default;opacity:.92;}
.sp-deck.live{cursor:pointer;outline:1px dashed #8f6f38;}
.sp-deck.live:hover{outline-color:#d8b25e;}
.sp-deck img{width:100%;height:100%;object-fit:cover;}
.sp-deck-n{position:absolute;right:2px;top:2px;font-size:11px;background:#16100a;border:1px solid #8a6a33;border-radius:8px;padding:0 4px;color:#ffe9b6;}
.sp-card{position:relative;height:var(--cardH,124px);width:calc(var(--cardH,124px)*.75);justify-self:center;border-radius:8px;
    overflow:hidden;background:#140d07;border:1px solid #57401f;box-shadow:0 5px 12px rgba(0,0,0,.55);cursor:pointer;
    transition:transform .12s, box-shadow .12s, border-color .12s;}
.sp-card:hover{transform:translateY(-3px);box-shadow:0 10px 18px rgba(0,0,0,.6);}
.sp-card img{width:100%;height:100%;object-fit:cover;}
.sp-card.buy{border-color:#57c98a;box-shadow:0 0 0 1px #57c98a, 0 6px 14px rgba(0,0,0,.55);}
.sp-card.sel,.sp-mini.sel{outline:2px solid #ffd76a;outline-offset:1px;}
.sp-cost{position:absolute;left:3px;bottom:2px;display:flex;gap:2px;flex-wrap:wrap;max-width:calc(100% - 6px);}
.sp-dot{display:inline-grid;place-items:center;min-width:14px;height:14px;border-radius:7px;font-size:9px;font-weight:800;
    font-style:normal;color:#fff;text-shadow:0 1px 2px #000;border:1px solid rgba(0,0,0,.5);padding:0 2px;}
.sp-empty{height:var(--cardH,124px);width:calc(var(--cardH,124px)*.75);justify-self:center;border-radius:8px;
    border:1px dashed rgba(140,108,60,.35);}
.c0{background:linear-gradient(#fdfdfd,#c9c9c9);color:#222;}
.c1{background:linear-gradient(#6fb6ff,#2b6fd0);}
.c2{background:linear-gradient(#7fe08a,#2c9a45);}
.c3{background:linear-gradient(#ff8a8a,#c62f2f);}
.c4{background:linear-gradient(#9a9a9a,#2a2a2a);}
.c5{background:linear-gradient(#ffe08a,#c9a227);color:#231708;}

/* player panels */
.sp-player{border-radius:10px;padding:7px 9px;background:rgba(0,0,0,.34);border:1px solid #4a3a22;display:flex;
    flex-direction:column;gap:5px;}
.sp-player.active{border-color:#d8b25e;background:rgba(216,178,94,.12);box-shadow:0 0 0 1px rgba(216,178,94,.35);}
.sp-phead{display:flex;align-items:baseline;gap:8px;}
.sp-pname{font-size:13px;font-weight:600;color:#f2e2c0;}
.sp-ppt{font-size:12px;color:#cbb894;}
.sp-ppt b{font-size:17px;color:#ffdf9a;}
.sp-pmeta{margin-left:auto;font-size:11px;color:#9f8865;}
.sp-prow{display:flex;align-items:center;gap:6px;flex-wrap:wrap;min-height:26px;}
.sp-plabel{font-size:10px;color:#8d7553;width:28px;letter-spacing:.1em;}
.sp-own{position:relative;display:flex;align-items:center;gap:2px;padding:1px 5px 1px 1px;border-radius:999px;
    background:rgba(255,255,255,.06);font-size:11px;color:#e7d6b2;}
.sp-own img{width:20px;height:20px;object-fit:contain;}
.sp-own.disc{outline:1px solid #ff7b6b;}
.sp-own b{color:#ff9a88;font-size:11px;}
.sp-chip{font-style:normal;font-size:10px;padding:1px 6px;border-radius:999px;color:#fff;border:1px solid rgba(0,0,0,.4);}
.sp-mini{position:relative;width:34px;height:45px;border-radius:5px;overflow:hidden;border:1px solid #57401f;cursor:pointer;}
.sp-mini img{width:100%;height:100%;object-fit:cover;}
.sp-mini.buy{border-color:#57c98a;}

/* log */
.sp-log{border-radius:10px;padding:6px 9px;background:rgba(0,0,0,.28);border:1px solid #3f3120;overflow:auto;min-height:64px;}
.sp-logline{font-size:11px;color:#c3b191;line-height:1.55;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sp-logline.dim{color:#7d6a52;}

/* tray */
.sp-tray{flex:0 0 auto;display:flex;align-items:center;gap:12px;padding:9px 12px;border-radius:12px;
    background:rgba(12,8,4,.82);border:1px solid #5c4626;box-shadow:0 -4px 16px rgba(0,0,0,.4);min-height:56px;}
.sp-tray-discard{border-color:#b4553f;}
.sp-hint{font-size:12.5px;color:#d8c8a6;}
.sp-hint b{color:#ffd76a;}
.sp-picks{display:flex;gap:6px;flex-wrap:wrap;}
.sp-pick{font-size:11px;color:#fff;border-radius:999px;padding:2px 9px;border:1px solid rgba(0,0,0,.45);}
.sp-selinfo{display:flex;align-items:center;gap:8px;}
.sp-seltitle{font-size:13px;color:#f4e3bf;}
.sp-btns{margin-left:auto;display:flex;gap:8px;}

/* overlays */
.sp-note{position:fixed;left:50%;top:14px;transform:translateX(-50%);background:rgba(180,60,40,.95);color:#fff;
    padding:8px 16px;border-radius:10px;font-size:13px;box-shadow:0 6px 18px rgba(0,0,0,.5);z-index:20;}
.sp-over{position:fixed;inset:0;display:grid;place-items:center;background:rgba(6,4,2,.72);z-index:15;}
.sp-overcard{min-width:320px;padding:22px 26px;border-radius:16px;background:linear-gradient(#241a10,#180f08);
    border:1px solid #b1894a;box-shadow:0 20px 50px rgba(0,0,0,.7);text-align:center;}
.sp-overcard h2{margin:0 0 12px;font-size:24px;color:#ffdf9a;letter-spacing:.06em;}
.sp-fin{display:flex;align-items:baseline;gap:8px;justify-content:center;font-size:13px;color:#d8c8a6;margin:4px 0;}
.sp-fin b{font-size:20px;color:#ffd76a;}
.sp-fin.win b,.sp-fin.win span{color:#8ef0b0;}
.sp-overcard .dim{color:#9c8766;font-size:11px;}
.sp-overcard button{margin-top:14px;width:100%;}
`;
