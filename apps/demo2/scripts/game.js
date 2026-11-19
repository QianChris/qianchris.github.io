// Demo2 game logic — attached to the GameState entity via ScriptComponent.
//
// - Left-click: spawn a dynamic rigidbody ball at the camera that flies toward
//   the point under the cursor, knocking over the cube stack.
// - Collision: pulse a burst particle emitter at the impact point (sparks).
// - Cleanup: recycle balls that fall out of the world.
//
// Uses only public engine APIs (ctx.scene, ctx.physics, ctx.on/emit) — no
// engine code is app-specific; this whole demo is data + this script.

let nextId = 0;
const balls = new Set();            // eid set of live balls
const ballKeys = new Map();         // eid -> entity key
let burstUntil = 0;                 // time the spark emitter stays on until
let sparkEmitterEid = -1;

function mat4Vec4(m, v) {
    return [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
        m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
    ];
}

function unproject(ivp, x, y, z) {
    const p = mat4Vec4(ivp, [x, y, z, 1]);
    return [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
}

function normalize(v) {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
}

export function init(ctx) {
    // find the spark emitter entity by key
    sparkEmitterEid = ctx.scene.entityKeyMap.get('SparkEmitter') ?? -1;

    ctx.on('mousedown', (e) => {
        if (e.button !== 0) return;
        spawnBall(ctx, e.x, e.y);
    });

    ctx.on('collision', (c) => {
        // only spark when a ball is involved and we have an impact point
        if (!c.point) return;
        const aBall = balls.has(c.a);
        const bBall = balls.has(c.b);
        if (!aBall && !bBall) return;
        // the non-ball entity decides the spark colour
        const otherKey = aBall ? c.keyB : c.keyA;
        triggerBurst(ctx, c.point, otherKey);
    });
}

function spawnBall(ctx, ndcX, ndcY) {
    const cam = ctx.scene.getActiveCamera(ctx.aspect);
    if (!cam) return;

    const near = unproject(cam.ivp, ndcX, ndcY, 0);
    const far = unproject(cam.ivp, ndcX, ndcY, 1);
    const dir = normalize([far[0] - near[0], far[1] - near[1], far[2] - near[2]]);

    const speed = Number(ctx.getField('GameStateComponent', 'ballSpeed') ?? 18);
    const radius = Number(ctx.getField('GameStateComponent', 'ballRadius') ?? 0.35);

    // start just in front of the near plane so the ball is visible immediately
    const origin = [near[0] + dir[0] * 0.5, near[1] + dir[1] * 0.5, near[2] + dir[2] * 0.5];
    const vel = [dir[0] * speed, dir[1] * speed, dir[2] * speed];

    const key = `Ball_${nextId++}`;
    const eid = ctx.scene.createEntity(key, {
        Transform: { position: origin, rotation: [0, 0, 0, 1], scale: [radius * 2, radius * 2, radius * 2] },
        MeshComponent: { mesh: 'icosphere' },
        TestMeshRender: { color: [0.95, 0.95, 1.0, 1] },
        RigidBodyComponent: { bodyType: 'dynamic', linearVelocity: vel, ccd: 1 },
        ColliderComponent: { shape: 'ball', radius: radius, restitution: 0.4, density: 6, friction: 0.5 },
    });
    balls.add(eid);
    ballKeys.set(eid, key);
}

function triggerBurst(ctx, point, otherKey) {
    if (sparkEmitterEid < 0) return;

    // ground → blue fire, cube (or anything else) → red/orange fire
    const onGround = otherKey === 'Ground';
    const startColor = onGround ? [0.4, 0.7, 1.0, 1.0] : [1.0, 0.85, 0.4, 1.0];
    const endColor = onGround ? [0.1, 0.2, 1.0, 0.0] : [1.0, 0.25, 0.05, 0.0];
    ctx.scene.setField(sparkEmitterEid, 'EmitterComponent', 'startColor', startColor);
    ctx.scene.setField(sparkEmitterEid, 'EmitterComponent', 'endColor', endColor);

    ctx.scene.setField(sparkEmitterEid, 'Transform', 'position', point);
    ctx.scene.setField(sparkEmitterEid, 'EmitterComponent', 'enabled', 1);
    burstUntil = ctx.time + 0.08;
    const score = Number(ctx.getField('GameStateComponent', 'score') ?? 0);
    ctx.setField('GameStateComponent', 'score', score + 1);
}

export function update(ctx) {
    // close the burst window
    if (sparkEmitterEid >= 0 && burstUntil > 0 && ctx.time >= burstUntil) {
        ctx.scene.setField(sparkEmitterEid, 'EmitterComponent', 'enabled', 0);
        ctx.scene.setField(sparkEmitterEid, 'Transform', 'position', [0, -50, 0]);
        burstUntil = 0;
    }

    // recycle balls that fell out of the world
    for (const eid of [...balls]) {
        const pos = ctx.scene.getField(eid, 'Transform', 'position');
        if (!pos) { balls.delete(eid); continue; }
        if (pos[1] < -10) {
            const key = ballKeys.get(eid);
            if (key) ctx.scene.removeEntity(key);
            balls.delete(eid);
            ballKeys.delete(eid);
        }
    }
}
