import { resourceManager, uniformLayouts } from '@shaderlab/api';
import type { Scene } from '@shaderlab/api';

/**
 * PBF (Position Based Fluids) fluid manager — ported from Particles4All
 * (src/sim.js + src/scene.js), fluid-only scope:
 *
 *   - Density constraint C = rho/rho0 − 1 solved by position projection with
 *     Lagrange multipliers (Macklin & Müller 2013). poly6 for density, spiky
 *     gradient for ∇C, CFM regularization ε (scaled by a CPU-calibrated rest
 *     denominator), Akinci tensile-instability fix (sCorr).
 *   - Static box boundary via Akinci psi boundary particles (CPU precomputed
 *     psi = rho0 / Σ poly6, CPU-sorted once per seed).
 *   - XSPH viscosity. (Surface tension / rigid coupling: later phases.)
 *   - Neighbour search: uniform grid (cell = h) rebuilt every substep by a
 *     GPU counting sort (count → block scan → block-sum scan → add → slot →
 *     move). After the scatter, the particle index IS the sorted slot, so all
 *     neighbour loops walk contiguous memory. pos/vel/pred are double
 *     buffered (A/B parity); pred additionally ping-pongs per Jacobi
 *     iteration and is re-synced by a copy when the iteration count is odd.
 *
 * Units: SI (metres, kg, s), rest density 1000 (water). The fluid box is
 * anchored at the entity's Transform.position (re-seeds if it moves).
 *
 * Unlike Particles4All's Sim.primeGrid, no grid-priming pass is needed here:
 * every substep builds cellStart from the current pred before any
 * cross-particle kernel runs, and the draw reads the live parity buffer.
 */

const WG = 256;

/** The 12 compute kernels, resolved from the render pipeline's aux refs. */
export interface PbfPipes {
    predict: GPUComputePipeline;
    count: GPUComputePipeline;
    scanBlock: GPUComputePipeline;
    scanBlocks: GPUComputePipeline;
    scanAdd: GPUComputePipeline;
    scatterSlot: GPUComputePipeline;
    scatterMove: GPUComputePipeline;
    lambda: GPUComputePipeline;
    delta: GPUComputePipeline;
    velFromPos: GPUComputePipeline;
    xsph: GPUComputePipeline;
    finalize: GPUComputePipeline;
}

interface FluidGpu {
    eid: number;
    n: number;
    /** World-space box origin (entity Transform.position). */
    boxMin: [number, number, number];
    box: [number, number, number];
    gridDim: [number, number, number];
    nCells: number;
    h: number;
    d: number;
    mass: number;
    rho0: number;
    denomRest: number;
    nBoundary: number;
    parity: number;
    predParity: number;
    timeBank: number;
    structuralKey: string;
    pos: GPUBuffer[];
    vel: GPUBuffer[];
    pred: GPUBuffer[];
    lambdaBuf: GPUBuffer;
    density: GPUBuffer;
    slot: GPUBuffer;
    corr: GPUBuffer;
    cellCount: GPUBuffer;
    cellStart: GPUBuffer;
    blockSum: GPUBuffer;
    cursor: GPUBuffer;
    bpos: GPUBuffer;
    bpsi: GPUBuffer;
    bcellStart: GPUBuffer;
    ubo: GPUBuffer;
    viewUbo: GPUBuffer;
    boxUbo: GPUBuffer;
    uniF: Float32Array;
    uniU: Uint32Array;
    uniBuf: ArrayBuffer;
    binds: {
        predict: GPUBindGroup[];
        count: GPUBindGroup[];
        scatterSlot: GPUBindGroup[];
        scatterMove: GPUBindGroup[];
        lambda: GPUBindGroup[];
        delta: GPUBindGroup[];
        velFromPos: GPUBindGroup[];
        xsph: GPUBindGroup[];
        finalize: GPUBindGroup[];
        scanBlock: GPUBindGroup;
        scanBlocks: GPUBindGroup;
        scanAdd: GPUBindGroup;
        draw: GPUBindGroup[];
        box: GPUBindGroup;
    };
}

/* ── CPU scene construction (ported from Particles4All scene.js) ── */

const poly6Coef = (h: number): number => 315 / (64 * Math.PI * Math.pow(h, 9));

/** Jittered fluid lattice occupying 35% of the box x-extent (dam break). */
function fluidBlock(count: number, box: [number, number, number], d: number): number[] {
    const margin = d;
    const nx = Math.max(1, Math.floor(box[0] * 0.35 / d));
    const nz = Math.max(1, Math.floor((box[2] - 2 * margin) / d));
    const maxLayers = Math.max(1, Math.floor((box[1] - 2 * margin) / d));
    const layers = Math.min(Math.ceil(count / (nx * nz)), maxLayers);
    const out: number[] = [];

    let seed = 12345;
    const rnd = (): number => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return (seed / 0x7fffffff - 0.5) * 0.002 * d;
    };
    for (let iy = 0; iy < layers && out.length / 3 < count; iy++) {
        for (let ix = 0; ix < nx && out.length / 3 < count; ix++) {
            for (let iz = 0; iz < nz && out.length / 3 < count; iz++) {
                out.push(
                    margin + (ix + 0.5) * d + rnd(),
                    margin + (iy + 0.5) * d + rnd(),
                    margin + (iz + 0.5) * d + rnd(),
                );
            }
        }
    }
    return out;
}

/** Static boundary particles on the box shell, with Akinci psi weights
 *  (psi = rho0 / Σ poly6 over boundary neighbours). Local coords [0, size]. */
function boundaryParticles(
    box: [number, number, number],
    d: number,
    h: number,
    rho0: number,
): { pos: Float32Array; psi: Float32Array; count: number } {
    const nb = box.map(s => Math.max(2, Math.ceil(s / d) + 1));
    const step = box.map((s, k) => s / (nb[k] - 1));
    const pts: number[] = [];
    for (let ix = 0; ix < nb[0]; ix++) {
        for (let iy = 0; iy < nb[1]; iy++) {
            for (let iz = 0; iz < nb[2]; iz++) {
                const shell = ix === 0 || ix === nb[0] - 1 || iy === 0 || iy === nb[1] - 1 ||
                    iz === 0 || iz === nb[2] - 1;
                if (shell) pts.push(ix * step[0], iy * step[1], iz * step[2]);
            }
        }
    }

    const n = pts.length / 3;
    const dim = box.map(s => Math.max(1, Math.ceil(s / h)));
    const cellOf = (x: number, y: number, z: number): number[] => [
        Math.min(dim[0] - 1, Math.max(0, Math.floor(x / h))),
        Math.min(dim[1] - 1, Math.max(0, Math.floor(y / h))),
        Math.min(dim[2] - 1, Math.max(0, Math.floor(z / h))),
    ];
    const key = (a: number, b: number, c: number): number => (c * dim[1] + b) * dim[0] + a;
    const buckets = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
        const c = cellOf(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
        const k = key(c[0], c[1], c[2]);
        let arr = buckets.get(k);
        if (!arr) { arr = []; buckets.set(k, arr); }
        arr.push(i);
    }
    const coef = poly6Coef(h);
    const h2 = h * h;
    const psi = new Float32Array(Math.max(1, n));
    for (let i = 0; i < n; i++) {
        const c = cellOf(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
        let sum = 0;
        for (let dc = -1; dc <= 1; dc++) {
            if (c[2] + dc < 0 || c[2] + dc >= dim[2]) continue;
            for (let db = -1; db <= 1; db++) {
                if (c[1] + db < 0 || c[1] + db >= dim[1]) continue;
                for (let da = -1; da <= 1; da++) {
                    if (c[0] + da < 0 || c[0] + da >= dim[0]) continue;
                    const arr = buckets.get(key(c[0] + da, c[1] + db, c[2] + dc));
                    if (!arr) continue;
                    for (const j of arr) {
                        const dx = pts[i * 3] - pts[j * 3];
                        const dy = pts[i * 3 + 1] - pts[j * 3 + 1];
                        const dz = pts[i * 3 + 2] - pts[j * 3 + 2];
                        const r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 >= h2) continue;
                        const t = h2 - r2;
                        sum += t * t * t;
                    }
                }
            }
        }
        sum *= coef;
        psi[i] = sum > 0 ? rho0 / sum : 0;
    }
    return { pos: new Float32Array(pts), psi, count: n };
}

/** CPU-side fluid scene build: lattice, boundary psi, mass calibration and
 *  rest-denominator calibration (scales CFM ε and sCorr k). */
function buildFluidScene(
    count: number,
    box: [number, number, number],
    spacing: number,
    rho0: number,
): { n: number; posLocal: Float32Array; mass: number; h: number; denomRest: number; boundary: { pos: Float32Array; psi: Float32Array; count: number } } {
    const d = spacing;
    const h = 2 * d;

    const fluid = fluidBlock(count, box, d);
    const nFluid = fluid.length / 3;
    const pos = new Float32Array(nFluid * 4);
    for (let i = 0; i < nFluid; i++) {
        pos[i * 4 + 0] = fluid[i * 3 + 0];
        pos[i * 4 + 1] = fluid[i * 3 + 1];
        pos[i * 4 + 2] = fluid[i * 3 + 2];
    }

    const boundary = boundaryParticles(box, d, h, rho0);

    // ── mass calibration: scale m so the densest lattice sample ≈ rho0 ──
    const coef = poly6Coef(h);
    const h2 = h * h;
    const spiky = -45 / (Math.PI * Math.pow(h, 6));
    const dim = [
        Math.max(1, Math.floor(box[0] / h)),
        Math.max(1, Math.floor(box[1] / h)),
        Math.max(1, Math.floor(box[2] / h)),
    ];
    const cellOf = (i: number): number[] => [
        Math.min(dim[0] - 1, Math.max(0, Math.floor(pos[i * 4 + 0] / h))),
        Math.min(dim[1] - 1, Math.max(0, Math.floor(pos[i * 4 + 1] / h))),
        Math.min(dim[2] - 1, Math.max(0, Math.floor(pos[i * 4 + 2] / h))),
    ];
    const key = (a: number, b: number, c: number): number => (c * dim[1] + b) * dim[0] + a;
    const buckets = new Map<number, number[]>();
    for (let i = 0; i < nFluid; i++) {
        const c = cellOf(i);
        const k = key(c[0], c[1], c[2]);
        let arr = buckets.get(k);
        if (!arr) { arr = []; buckets.set(k, arr); }
        arr.push(i);
    }
    const forEachNeighbour = (i: number, f: (j: number) => void): void => {
        const c = cellOf(i);
        for (let dc = -1; dc <= 1; dc++) {
            if (c[2] + dc < 0 || c[2] + dc >= dim[2]) continue;
            for (let db = -1; db <= 1; db++) {
                if (c[1] + db < 0 || c[1] + db >= dim[1]) continue;
                for (let da = -1; da <= 1; da++) {
                    if (c[0] + da < 0 || c[0] + da >= dim[0]) continue;
                    const arr = buckets.get(key(c[0] + da, c[1] + db, c[2] + dc));
                    if (arr) for (const j of arr) if (j !== i) f(j);
                }
            }
        }
    };
    const densityAt = (i: number, m: number): number => {
        let rho = m * coef * h2 * h2 * h2;
        forEachNeighbour(i, j => {
            const r2 = (pos[i * 4] - pos[j * 4]) ** 2 + (pos[i * 4 + 1] - pos[j * 4 + 1]) ** 2 +
                (pos[i * 4 + 2] - pos[j * 4 + 2]) ** 2;
            if (r2 >= h2) return;
            const t = h2 - r2;
            rho += m * coef * t * t * t;
        });
        return rho;
    };
    let mass = rho0 * d * d * d;
    {
        let maxRho = 0;
        for (let i = 0; i < nFluid; i++) maxRho = Math.max(maxRho, densityAt(i, mass));
        if (maxRho > 0.5 * rho0) {
            mass *= rho0 / maxRho;
        }
    }

    // ── rest denominator: max SPH constraint denominator over interior
    //    particles (early-bounded) — used to scale CFM ε and sCorr k ──
    let denomRest = 0;
    {
        const volume = mass / rho0;
        for (let i = 0; i < nFluid; i++) {
            if (densityAt(i, mass) < 0.99 * rho0) continue;
            let gx = 0, gy = 0, gz = 0, sumGrad2 = 0;
            forEachNeighbour(i, j => {
                const rx = pos[i * 4] - pos[j * 4];
                const ry = pos[i * 4 + 1] - pos[j * 4 + 1];
                const rz = pos[i * 4 + 2] - pos[j * 4 + 2];
                const r2 = rx * rx + ry * ry + rz * rz;
                if (r2 < 1e-12 || r2 >= h2) return;
                const r = Math.sqrt(r2);
                const hr = h - r;
                const s = volume * spiky * hr * hr / r;
                gx += s * rx; gy += s * ry; gz += s * rz;
                sumGrad2 += s * s * r2;
            });
            denomRest = Math.max(denomRest, gx * gx + gy * gy + gz * gz + sumGrad2);
            if (i > 2000 && denomRest > 0) break;
        }
    }

    return { n: nFluid, posLocal: pos, mass, h, denomRest, boundary };
}

/* ── manager ───────────────────────────────────────────────────────── */

export class PbfManager {
    private systems = new Map<number, FluidGpu>();
    private frameLog = 0;

    clear(): void {
        for (const sys of this.systems.values()) this.destroySys(sys);
        this.systems.clear();
        this.frameLog = 0;
    }

    /** Destroy all GPU buffers held by a FluidGpu entry. */
    private destroySys(sys: FluidGpu): void {
        for (const b of sys.pos) b.destroy();
        for (const b of sys.vel) b.destroy();
        for (const b of sys.pred) b.destroy();
        sys.lambdaBuf.destroy();
        sys.density.destroy();
        sys.slot.destroy();
        sys.corr.destroy();
        sys.cellCount.destroy();
        sys.cellStart.destroy();
        sys.blockSum.destroy();
        sys.cursor.destroy();
        sys.bpos.destroy();
        sys.bpsi.destroy();
        sys.bcellStart.destroy();
        sys.ubo.destroy();
        sys.viewUbo.destroy();
        sys.boxUbo.destroy();
    }

    simulate(
        encoder: GPUCommandEncoder,
        scene: Scene,
        pipes: PbfPipes,
        tgs: Record<keyof PbfPipes, number>,
        entities: readonly number[],
        frameDt: number,
    ): void {
        // Sweep SysGpu entries for entities that no longer exist (scene
        // reloads / entity removal without an app unload).
        if (this.systems.size > 0) {
            const valid = new Set(entities);
            for (const [eid, sys] of this.systems) {
                if (!valid.has(eid)) {
                    this.destroySys(sys);
                    this.systems.delete(eid);
                }
            }
        }

        for (const eid of entities) {
            let sys = this.systems.get(eid);
            const key = this.structuralKey(scene, eid);
            if (sys && sys.structuralKey !== key) {
                this.destroySys(sys);
                this.systems.delete(eid);
                sys = undefined;
            }
            if (!sys) sys = this.ensure(scene, eid, key);
            this.writeViewUbo(sys, scene, eid);
            this.step(encoder, scene, eid, sys, pipes, tgs, frameDt);
            if (this.frameLog < 5) {
                console.log(`[pbd4all] frame ${this.frameLog}: dt=${frameDt.toFixed(4)}s n=${sys.n} entities=${entities.length}`);
                this.frameLog++;
            }
        }
    }

    draw(pass: GPURenderPassEncoder, pipeline: GPURenderPipeline, entities: readonly number[]): void {
        for (const eid of entities) {
            const sys = this.systems.get(eid);
            if (!sys) continue;
            pass.setPipeline(pipeline);
            pass.setBindGroup(1, sys.binds.draw[sys.parity]);
            pass.draw(6, sys.n);
        }
    }

    drawBox(pass: GPURenderPassEncoder, pipeline: GPURenderPipeline, entities: readonly number[]): void {
        for (const eid of entities) {
            const sys = this.systems.get(eid);
            if (!sys) continue;
            pass.setPipeline(pipeline);
            pass.setBindGroup(1, sys.binds.box);
            pass.draw(24);
        }
    }

    /* ── per-substep orchestration (ported from Particles4All Sim.step) ── */

    private step(
        encoder: GPUCommandEncoder,
        scene: Scene,
        eid: number,
        sys: FluidGpu,
        pipes: PbfPipes,
        tgs: Record<keyof PbfPipes, number>,
        frameDt: number,
    ): void {
        const substeps = Math.max(1, Math.floor(this.numField(scene, eid, 'substeps') ?? 2));
        const iterations = Math.max(1, Math.floor(this.numField(scene, eid, 'iterations') ?? 4));
        const timeScale = this.numField(scene, eid, 'timeScale') ?? 0.578;

        const dtTarget = (1 / 60) / substeps;
        sys.timeBank += frameDt * timeScale;
        let sub = Math.floor(sys.timeBank / dtTarget + 1e-4);
        const workCap = substeps * 8;
        if (sub > workCap) {
            sub = workCap;
            sys.timeBank = 0;
        } else {
            sys.timeBank -= sub * dtTarget;
        }
        if (sub < 1) return;

        this.uploadParams(sys, scene, eid, dtTarget, iterations);

        // Thread counts per kernel: particles, grid cells, cells+1 (scanAdd
        // also writes the sentinel at cellStart[nCells]), 1 (block-sum scan).
        const nT = sys.n;
        const cellT = sys.nCells;
        const scanAddT = sys.nCells + 1;

        for (let s = 0; s < sub; s++) {
            const par = sys.parity;

            // ── predict ──
            const passP = encoder.beginComputePass();
            this.run(passP, pipes.predict, sys.binds.predict[par], nT, tgs.predict);
            passP.end();

            encoder.clearBuffer(sys.cellCount);
            encoder.clearBuffer(sys.cursor);

            // ── grid counting sort ──
            const passG = encoder.beginComputePass();
            this.run(passG, pipes.count, sys.binds.count[par], nT, tgs.count);
            this.run(passG, pipes.scanBlock, sys.binds.scanBlock, cellT, tgs.scanBlock);
            this.run(passG, pipes.scanBlocks, sys.binds.scanBlocks, 1, tgs.scanBlocks);
            this.run(passG, pipes.scanAdd, sys.binds.scanAdd, scanAddT, tgs.scanAdd);
            this.run(passG, pipes.scatterSlot, sys.binds.scatterSlot[par], nT, tgs.scatterSlot);
            this.run(passG, pipes.scatterMove, sys.binds.scatterMove[par], nT, tgs.scatterMove);
            passG.end();
            sys.parity ^= 1;
            sys.predParity = sys.parity;

            // ── density-constraint iterations (lambda + delta) ──
            const par2 = sys.parity;
            let pp = sys.predParity;
            for (let it = 0; it < iterations; it++) {
                const passI = encoder.beginComputePass();
                this.run(passI, pipes.lambda, sys.binds.lambda[pp], nT, tgs.lambda);
                this.run(passI, pipes.delta, sys.binds.delta[pp], nT, tgs.delta);
                passI.end();
                pp ^= 1;
            }
            sys.predParity = pp;

            // Re-align pred with pos/vel when the iteration count is odd.
            if (sys.predParity !== sys.parity) {
                encoder.copyBufferToBuffer(
                    sys.pred[sys.predParity], 0, sys.pred[sys.parity], 0, sys.n * 16);
                sys.predParity = sys.parity;
            }

            // ── finish: velocity from positions, XSPH, commit ──
            const passF = encoder.beginComputePass();
            this.run(passF, pipes.velFromPos, sys.binds.velFromPos[par2], nT, tgs.velFromPos);
            this.run(passF, pipes.xsph, sys.binds.xsph[par2], nT, tgs.xsph);
            this.run(passF, pipes.finalize, sys.binds.finalize[par2], nT, tgs.finalize);
            passF.end();
        }
    }

    /** Dispatch a kernel over `threads` logical threads (rounded up to whole
     *  workgroups). `threads` is a THREAD count (particles / cells), not a
     *  workgroup count. */
    private run(
        pass: GPUComputePassEncoder,
        pipe: GPUComputePipeline,
        bind: GPUBindGroup,
        threads: number,
        tgs: number,
    ): void {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(threads / tgs));
    }

    /* ── UBO packing ─────────────────────────────────────────────── */

    private uploadParams(sys: FluidGpu, scene: Scene, eid: number, dt: number, iterations: number): void {
        const L = uniformLayouts.get('pbd4allParams');
        const f = sys.uniF;
        const u = sys.uniU;
        const h = sys.h;
        const d = sys.d;
        const halfD = 0.5 * d;

        const gravity = this.numField(scene, eid, 'gravity') ?? 9.81;
        const cfmEpsilonRel = this.numField(scene, eid, 'cfmEpsilonRel') ?? 0.01;
        const sCorrK = this.numField(scene, eid, 'sCorrK') ?? 0.1;
        const sCorrDq = this.numField(scene, eid, 'sCorrDq') ?? 0.3;
        const xsphC = this.numField(scene, eid, 'xsphC') ?? 0.066;
        const omega = this.numField(scene, eid, 'omega') ?? 1.03;

        L.write(f, 'boxMin', sys.boxMin);
        L.write(f, 'dt', dt);
        L.write(f, 'boxMax', [sys.boxMin[0] + sys.box[0], sys.boxMin[1] + sys.box[1], sys.boxMin[2] + sys.box[2]]);
        L.write(f, 'h', h);
        L.writeU32(u, 'gridDimX', sys.gridDim[0]);
        L.writeU32(u, 'gridDimY', sys.gridDim[1]);
        L.writeU32(u, 'gridDimZ', sys.gridDim[2]);
        L.write(f, 'h2', h * h);
        L.write(f, 'clampMin', [sys.boxMin[0] + halfD, sys.boxMin[1] + halfD, sys.boxMin[2] + halfD]);
        L.write(f, 'poly6', poly6Coef(h));
        L.write(f, 'clampMax', [
            sys.boxMin[0] + sys.box[0] - halfD,
            sys.boxMin[1] + sys.box[1] - halfD,
            sys.boxMin[2] + sys.box[2] - halfD,
        ]);
        L.write(f, 'spikyGrad', -45 / (Math.PI * Math.pow(h, 6)));
        L.write(f, 'gravity', gravity);
        L.write(f, 'mass', sys.mass);
        L.write(f, 'rho0', sys.rho0);
        L.write(f, 'invRho0', 1 / sys.rho0);
        L.write(f, 'cfmEps', Math.max(1e-9, cfmEpsilonRel * sys.denomRest));
        L.write(f, 'sCorrK', sCorrK / (sys.denomRest * Math.max(1, iterations)));

        const rq = sCorrDq * h;
        const tq = h * h - rq * rq;
        const wq = poly6Coef(h) * tq * tq * tq;
        L.write(f, 'sCorrWq', wq > 0 ? 1 / wq : 0);
        L.write(f, 'xsphC', xsphC);
        L.writeU32(u, 'n', sys.n);
        L.writeU32(u, 'nCells', sys.nCells);
        L.writeU32(u, 'nBoundary', sys.nBoundary);
        L.write(f, 'omega', omega);
        L.write(f, 'invDt', dt > 0 ? 1 / dt : 0);
        L.write(f, 'volume', sys.mass / sys.rho0);

        resourceManager.device.queue.writeBuffer(sys.ubo, 0, sys.uniBuf);
    }

    private writeViewUbo(sys: FluidGpu, scene: Scene, eid: number): void {
        const L = uniformLayouts.get('pbd4allView');
        const ab = new ArrayBuffer(L.byteSize);
        const f = new Float32Array(ab);
        const scale = this.numField(scene, eid, 'drawRadiusScale') ?? 0.37;
        const speedMax = this.numField(scene, eid, 'speedMax') ?? 2.845;
        L.write(f, 'radius', scale * sys.d);
        L.write(f, 'speedMax', speedMax);
        resourceManager.device.queue.writeBuffer(sys.viewUbo, 0, ab);
    }

    /* ── per-entity GPU state ─────────────────────────────────────── */

    /** Structural fields (particleCount / box / spacing / restDensity /
     *  entity position) are baked at seed time; changing any of them
     *  re-seeds the fluid. Physics fields are read live every frame. */
    private structuralKey(scene: Scene, eid: number): string {
        const count = this.numField(scene, eid, 'particleCount') ?? 30000;
        const box = this.vec3Field(scene, eid, 'box') ?? [1.5, 1.0, 1.0];
        const spacing = this.numField(scene, eid, 'spacing') ?? 0.02;
        const rho0 = this.numField(scene, eid, 'restDensity') ?? 1000;
        const tr = this.vec3Field(scene, eid, 'position', 'Transform') ?? [0, 0, 0];
        return [count, box[0], box[1], box[2], spacing, rho0, tr[0], tr[1], tr[2]].join('|');
    }

    private ensure(scene: Scene, eid: number, structuralKey: string): FluidGpu {
        const dev = resourceManager.device;

        const count = Math.max(1, Math.floor(this.numField(scene, eid, 'particleCount') ?? 30000));
        const box = this.vec3Field(scene, eid, 'box') ?? [1.5, 1.0, 1.0] as [number, number, number];
        const spacing = Math.max(1e-4, this.numField(scene, eid, 'spacing') ?? 0.02);
        const rho0 = Math.max(1, this.numField(scene, eid, 'restDensity') ?? 1000);
        const tr = this.vec3Field(scene, eid, 'Transform', 'position') ?? [0, 0, 0];
        const boxMin: [number, number, number] = [tr[0], tr[1], tr[2]];

        const sc = buildFluidScene(count, box, spacing, rho0);
        const h = sc.h;
        const gridDim: [number, number, number] = [
            Math.max(1, Math.floor(box[0] / h)),
            Math.max(1, Math.floor(box[1] / h)),
            Math.max(1, Math.floor(box[2] / h)),
        ];
        const nCells = gridDim[0] * gridDim[1] * gridDim[2];
        const n = sc.n;
        const nb = sc.boundary.count;

        console.log(`[pbd4all] seeding fluid: n=${n} (target ${count}) box=${box.map(v => v.toFixed(2)).join('×')} ` +
            `spacing=${spacing} h=${h.toFixed(4)} mass=${sc.mass.toExponential(3)} boundary=${nb} cells=${nCells} ` +
            `denomRest=${sc.denomRest.toExponential(3)}`);

        const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        const alloc = (bytes: number): GPUBuffer =>
            dev.createBuffer({ size: Math.max(16, bytes), usage: ST });

        const pos: GPUBuffer[] = [alloc(n * 16), alloc(n * 16)];
        const vel: GPUBuffer[] = [alloc(n * 16), alloc(n * 16)];
        const pred: GPUBuffer[] = [alloc(n * 16), alloc(n * 16)];
        const lambdaBuf = alloc(n * 4);
        const density = alloc(n * 4);
        const slot = alloc(n * 4);
        const corr = alloc(n * 16);
        const cellCount = alloc((nCells + 1) * 4);
        const cellStart = alloc((nCells + 2) * 4);
        const blockSum = alloc((Math.ceil(nCells / WG) + 2) * 4);
        const cursor = alloc((nCells + 1) * 4);

        // ── initial fluid state (world coords = local + boxMin) ──
        const posInit = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
            posInit[i * 4 + 0] = sc.posLocal[i * 4 + 0] + boxMin[0];
            posInit[i * 4 + 1] = sc.posLocal[i * 4 + 1] + boxMin[1];
            posInit[i * 4 + 2] = sc.posLocal[i * 4 + 2] + boxMin[2];
        }
        const velInit = new Float32Array(n * 4);
        dev.queue.writeBuffer(pos[0], 0, posInit);
        dev.queue.writeBuffer(vel[0], 0, velInit);
        dev.queue.writeBuffer(pred[0], 0, posInit);

        // ── boundary particles, CPU-sorted into grid order ──
        const bposWorld = new Float32Array(Math.max(1, nb) * 4);
        for (let i = 0; i < nb; i++) {
            bposWorld[i * 4 + 0] = sc.boundary.pos[i * 3 + 0] + boxMin[0];
            bposWorld[i * 4 + 1] = sc.boundary.pos[i * 3 + 1] + boxMin[1];
            bposWorld[i * 4 + 2] = sc.boundary.pos[i * 3 + 2] + boxMin[2];
        }
        const sorted = this.sortBoundary(bposWorld, sc.boundary.psi, nb, gridDim, h, boxMin);
        const bpos = alloc(Math.max(1, nb) * 16);
        const bpsi = alloc(Math.max(1, nb) * 4);
        const bcellStart = alloc((nCells + 2) * 4);
        dev.queue.writeBuffer(bpos, 0, sorted.pos);
        dev.queue.writeBuffer(bpsi, 0, sorted.psi);
        dev.queue.writeBuffer(bcellStart, 0, sorted.cellStart);

        // ── UBOs ──
        const paramsLayout = uniformLayouts.get('pbd4allParams');
        const uniBuf = new ArrayBuffer(paramsLayout.byteSize);
        const ubo = dev.createBuffer({
            size: paramsLayout.byteSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const viewLayout = uniformLayouts.get('pbd4allView');
        const viewUbo = dev.createBuffer({
            size: viewLayout.byteSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const boxLayout = uniformLayouts.get('pbd4allBoxParams');
        const boxUbo = dev.createBuffer({
            size: boxLayout.byteSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        {
            const ab = new ArrayBuffer(boxLayout.byteSize);
            const f = new Float32Array(ab);
            boxLayout.write(f, 'boxMin', boxMin);
            boxLayout.write(f, 'boxMax', [
                boxMin[0] + box[0], boxMin[1] + box[1], boxMin[2] + box[2],
            ]);
            dev.queue.writeBuffer(boxUbo, 0, ab);
        }

        // ── bind groups ──
        const bg = (layout: string, buffers: GPUBuffer[]): GPUBindGroup =>
            dev.createBindGroup({
                layout: resourceManager.namedLayout(layout),
                entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
            });

        const binds: FluidGpu['binds'] = {
            predict: [], count: [], scatterSlot: [], scatterMove: [],
            lambda: [], delta: [], velFromPos: [], xsph: [], finalize: [],
            scanBlock: bg('pbd4allScanBlock', [cellCount, cellStart, blockSum, ubo]),
            scanBlocks: bg('pbd4allScanBlocks', [blockSum, ubo]),
            scanAdd: bg('pbd4allScanAdd', [cellStart, blockSum, ubo]),
            draw: [],
            box: bg('pbd4allBox', [boxUbo]),
        };
        for (let par = 0; par < 2; par++) {
            const o = 1 - par;
            binds.predict.push(bg('pbd4allPredict', [pos[par], vel[par], pred[par], ubo]));
            binds.count.push(bg('pbd4allCount', [pred[par], cellCount, ubo]));
            binds.scatterSlot.push(bg('pbd4allScatterSlot', [pred[par], cellStart, cursor, slot, ubo]));
            binds.scatterMove.push(bg('pbd4allScatterMove', [
                slot, pos[par], vel[par], pred[par], pos[o], vel[o], pred[o], ubo,
            ]));
            binds.velFromPos.push(bg('pbd4allVelFromPos', [pos[par], vel[par], pred[par], ubo]));
            binds.xsph.push(bg('pbd4allXsph', [pred[par], vel[par], density, corr, cellStart, ubo]));
            binds.finalize.push(bg('pbd4allFinalize', [pos[par], vel[par], pred[par], corr, ubo]));
            binds.draw.push(bg('pbd4allDraw', [pos[par], vel[par], viewUbo]));
        }
        binds.lambda.push(bg('pbd4allLambda', [pred[0], lambdaBuf, density, cellStart, bpos, bpsi, bcellStart, ubo]));
        binds.lambda.push(bg('pbd4allLambda', [pred[1], lambdaBuf, density, cellStart, bpos, bpsi, bcellStart, ubo]));
        binds.delta.push(bg('pbd4allDelta', [pred[0], pred[1], lambdaBuf, cellStart, bpos, bpsi, bcellStart, ubo]));
        binds.delta.push(bg('pbd4allDelta', [pred[1], pred[0], lambdaBuf, cellStart, bpos, bpsi, bcellStart, ubo]));

        const sys: FluidGpu = {
            eid, n,
            boxMin, box, gridDim, nCells,
            h, d: spacing, mass: sc.mass, rho0, denomRest: sc.denomRest, nBoundary: nb,
            parity: 0, predParity: 0, timeBank: 0,
            structuralKey,
            pos, vel, pred,
            lambdaBuf, density, slot, corr,
            cellCount, cellStart, blockSum, cursor,
            bpos, bpsi, bcellStart,
            ubo, viewUbo, boxUbo,
            uniF: new Float32Array(uniBuf), uniU: new Uint32Array(uniBuf), uniBuf,
            binds,
        };
        this.systems.set(eid, sys);
        return sys;
    }

    /** Counting-sort the (static) boundary particles into grid order on the
     *  CPU, once per seed. cellStart has nCells + 2 entries. */
    private sortBoundary(
        bpos: Float32Array,
        psi: Float32Array,
        nb: number,
        gridDim: [number, number, number],
        h: number,
        boxMin: [number, number, number],
    ): { pos: Float32Array<ArrayBuffer>; psi: Float32Array<ArrayBuffer>; cellStart: Uint32Array<ArrayBuffer> } {
        const nCells = gridDim[0] * gridDim[1] * gridDim[2];
        const counts = new Uint32Array(nCells + 2);
        const cellOf = (x: number, y: number, z: number): number => {
            const cx = Math.min(gridDim[0] - 1, Math.max(0, Math.floor((x - boxMin[0]) / h)));
            const cy = Math.min(gridDim[1] - 1, Math.max(0, Math.floor((y - boxMin[1]) / h)));
            const cz = Math.min(gridDim[2] - 1, Math.max(0, Math.floor((z - boxMin[2]) / h)));
            return (cz * gridDim[1] + cy) * gridDim[0] + cx;
        };
        const cell = new Uint32Array(Math.max(1, nb));
        for (let i = 0; i < nb; i++) {
            cell[i] = cellOf(bpos[i * 4], bpos[i * 4 + 1], bpos[i * 4 + 2]);
            counts[cell[i]]++;
        }
        const start = new Uint32Array(nCells + 2);
        let run = 0;
        for (let c = 0; c <= nCells; c++) { start[c] = run; run += counts[c] || 0; }
        start[nCells + 1] = run;
        const cur = start.slice();
        const sortedPos = new Float32Array(Math.max(1, nb) * 4);
        const sortedPsi = new Float32Array(Math.max(1, nb));
        for (let i = 0; i < nb; i++) {
            const s = cur[cell[i]]++;
            sortedPos[s * 4 + 0] = bpos[i * 4 + 0];
            sortedPos[s * 4 + 1] = bpos[i * 4 + 1];
            sortedPos[s * 4 + 2] = bpos[i * 4 + 2];
            sortedPsi[s] = psi[i];
        }
        return { pos: sortedPos, psi: sortedPsi, cellStart: start };
    }

    private numField(scene: Scene, eid: number, field: string, comp = 'PbfFluidComponent'): number | undefined {
        const v = scene.getField(eid, comp, field);
        if (typeof v === 'number') return v;
        const arr = v as number[] | undefined;
        return Array.isArray(arr) ? arr[0] : undefined;
    }

    private vec3Field(scene: Scene, eid: number, field: string, comp = 'PbfFluidComponent'): [number, number, number] | undefined {
        const v = scene.getField(eid, comp, field);
        if (Array.isArray(v) && v.length >= 3) return [v[0], v[1], v[2]];
        return undefined;
    }
}
