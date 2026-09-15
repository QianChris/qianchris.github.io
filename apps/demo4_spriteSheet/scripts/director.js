const SPRITE_KEYS = ['SpriteBottom', 'SpriteCenter', 'SpriteTop'];

export function init(ctx) {
    ctx.on('mousedown', () => {
        for (const key of SPRITE_KEYS) {
            const eid = ctx.scene.entityKeyMap.get(key);
            if (eid === undefined) continue;
            const rows = Number(ctx.scene.getField(eid, 'SpriteSheetComponent', 'rows'));
            const animIndex = Math.floor(Math.random() * rows);
            ctx.scene.setField(eid, 'SpriteAnimationComponent', 'animation', animIndex);
            ctx.scene.setField(eid, 'SpriteAnimationComponent', 'frame', 0);
            ctx.scene.setField(eid, 'SpriteAnimationComponent', 'elapsed', 0);
            ctx.scene.setField(eid, 'SpriteAnimationComponent', 'direction', 1);
            ctx.scene.setField(eid, 'SpriteAnimationComponent', 'playing', 1);
        }
    });
}
