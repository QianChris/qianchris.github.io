#!/usr/bin/env node
/**
 * Generate a pre-baked soft-body asset file for demo9's PBD plugin.
 *
 * The asset contains all structural data that used to be generated at runtime:
 *   - particle rest positions (axis-aligned, centered at origin, no rotation)
 *   - surface triangle indices (cube shell)
 *   - shape-matching cluster data (sparse centers, Chebyshev membership)
 *
 * Cluster center selection: greedy, minimum Chebyshev spacing `clusterSpacing`
 * between centers. Each center's cluster = all particles within Chebyshev
 * radius `clusterRadius`. With clusterSpacing=clusterRadius, every particle is
 * covered by at least one cluster (full coverage, minimal overlap). With
 * clusterSpacing=0, every particle is a center (densest, maximum overlap).
 *
 * Usage:
 *   node generate_softbody.mjs [--gridN 5] [--cellSize 0.4]
 *                              [--clusterRadius 2] [--clusterSpacing 2]
 *                              [--mass 1.0] [--output softbody_asset.json]
 *
 * A Python equivalent (generate_softbody.py) is also provided.
 */

import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

function idxAt(i, j, k, n) {
    return i + j * n + k * n * n;
}

function buildSurfaceIndices(n) {
    const out = [];
    const at = (i, j, k) => idxAt(i, j, k, n);

    for (let j = 0; j < n - 1; j++) {
        for (let k = 0; k < n - 1; k++) {
            const a = at(0, j, k), b = at(0, j + 1, k), c = at(0, j + 1, k + 1), d = at(0, j, k + 1);
            out.push(a, b, c, a, c, d);
            const a2 = at(n - 1, j, k), b2 = at(n - 1, j + 1, k), c2 = at(n - 1, j + 1, k + 1), d2 = at(n - 1, j, k + 1);
            out.push(a2, c2, b2, a2, d2, c2);
        }
    }
    for (let i = 0; i < n - 1; i++) {
        for (let k = 0; k < n - 1; k++) {
            const a = at(i, 0, k), b = at(i + 1, 0, k), c = at(i + 1, 0, k + 1), d = at(i, 0, k + 1);
            out.push(a, c, b, a, d, c);
            const a2 = at(i, n - 1, k), b2 = at(i + 1, n - 1, k), c2 = at(i + 1, n - 1, k + 1), d2 = at(i, n - 1, k + 1);
            out.push(a2, b2, c2, a2, c2, d2);
        }
    }
    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - 1; j++) {
            const a = at(i, j, 0), b = at(i + 1, j, 0), c = at(i + 1, j + 1, 0), d = at(i, j + 1, 0);
            out.push(a, b, c, a, c, d);
            const a2 = at(i, j, n - 1), b2 = at(i + 1, j, n - 1), c2 = at(i + 1, j + 1, n - 1), d2 = at(i, j + 1, n - 1);
            out.push(a2, c2, b2, a2, d2, c2);
        }
    }
    return out;
}

/**
 * Seeded LCG random number generator (deterministic — same seed → same output).
 * Returns a function producing floats in [0, 1).
 */
function makeRng(seed) {
    let state = (seed >>> 0) || 1;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

function randInt(rng, min, max) {
    return Math.floor(rng() * (max - min + 1)) + min;
}

function shuffle(rng, arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
}

/**
 * Option C: jittered sublattice + greedy dedup + forced coverage.
 *
 * 1. Generate candidate centers on a regular sublattice (spacing = spacingCells),
 *    each displaced by ±jitter cells (clamped to grid).
 * 2. Shuffle candidates (seeded) and greedily select, skipping any within
 *    Chebyshev distance `spacingCells` of an already-selected center.
 * 3. Coverage check: every particle must be within `radiusCells` of some center.
 *    Uncovered particles are force-added as new centers (may violate spacing —
 *    coverage is mandatory, spacing is a soft constraint).
 *
 * With radiusCells=R and spacingCells=S, adjacent centers share (2R+1−S) layers
 * of particles. For 1–2 layers shared: S = 2R or S = 2R−1.
 */
function selectCenters(gridN, radiusCells, spacingCells, jitter, rng) {
    // 1. Sublattice candidates with jitter
    const candidates = [];
    for (let k = 0; k < gridN; k += spacingCells) {
        for (let j = 0; j < gridN; j += spacingCells) {
            for (let i = 0; i < gridN; i += spacingCells) {
                const ci = Math.max(0, Math.min(gridN - 1, i + randInt(rng, -jitter, jitter)));
                const cj = Math.max(0, Math.min(gridN - 1, j + randInt(rng, -jitter, jitter)));
                const ck = Math.max(0, Math.min(gridN - 1, k + randInt(rng, -jitter, jitter)));
                candidates.push([ci, cj, ck]);
            }
        }
    }
    // Ensure the far edge is represented even if gridN isn't a multiple of spacing.
    if ((gridN - 1) % spacingCells !== 0) {
        for (let j = 0; j < gridN; j += spacingCells) {
            for (let i = 0; i < gridN; i += spacingCells) {
                candidates.push([
                    Math.max(0, Math.min(gridN - 1, i + randInt(rng, -jitter, jitter))),
                    Math.max(0, Math.min(gridN - 1, j + randInt(rng, -jitter, jitter))),
                    gridN - 1,
                ]);
            }
        }
        for (let i = 0; i < gridN; i += spacingCells) {
            for (let k = 0; k < gridN; k += spacingCells) {
                candidates.push([
                    Math.max(0, Math.min(gridN - 1, i + randInt(rng, -jitter, jitter))),
                    gridN - 1,
                    Math.max(0, Math.min(gridN - 1, k + randInt(rng, -jitter, jitter))),
                ]);
            }
        }
        for (let j = 0; j < gridN; j += spacingCells) {
            for (let k = 0; k < gridN; k += spacingCells) {
                candidates.push([
                    gridN - 1,
                    Math.max(0, Math.min(gridN - 1, j + randInt(rng, -jitter, jitter))),
                    Math.max(0, Math.min(gridN - 1, k + randInt(rng, -jitter, jitter))),
                ]);
            }
        }
    }

    // 2. Shuffle + greedy select with min spacing
    shuffle(rng, candidates);
    const centers = [];
    for (const [ci, cj, ck] of candidates) {
        let tooClose = false;
        for (const [si, sj, sk] of centers) {
            if (Math.abs(ci - si) <= spacingCells &&
                Math.abs(cj - sj) <= spacingCells &&
                Math.abs(ck - sk) <= spacingCells) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) centers.push([ci, cj, ck]);
    }

    // 3. Coverage check: force-add centers for uncovered particles
    let added = 0;
    for (let k = 0; k < gridN; k++) {
        for (let j = 0; j < gridN; j++) {
            for (let i = 0; i < gridN; i++) {
                let covered = false;
                for (const [ci, cj, ck2] of centers) {
                    if (Math.abs(i - ci) <= radiusCells &&
                        Math.abs(j - cj) <= radiusCells &&
                        Math.abs(k - ck2) <= radiusCells) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) {
                    centers.push([i, j, k]);
                    added++;
                }
            }
        }
    }
    if (added > 0) {
        console.warn(`info: coverage check added ${added} extra center(s) for uncovered particles`);
    }

    return centers;
}

function generate(gridN, cellSize, clusterRadius, clusterSpacing, jitter, seed, mass) {
    const particleCount = gridN ** 3;
    const invMass = particleCount / mass;
    const half = (gridN - 1) / 2;

    // ── particle rest positions (axis-aligned, centered at origin) ───────
    const positions = [];
    for (let k = 0; k < gridN; k++) {
        for (let j = 0; j < gridN; j++) {
            for (let i = 0; i < gridN; i++) {
                positions.push([
                    (i - half) * cellSize,
                    (j - half) * cellSize,
                    (k - half) * cellSize,
                    invMass,
                ]);
            }
        }
    }

    // ── surface indices ─────────────────────────────────────────────────
    const surfaceIndices = buildSurfaceIndices(gridN);

    // ── select cluster centers (jittered sublattice + greedy + coverage) ─
    const rng = makeRng(seed);
    const centers = selectCenters(gridN, clusterRadius, clusterSpacing, jitter, rng);
    const clusterCount = centers.length;

    // First pass: count members per cluster
    const clusterSizes = new Uint32Array(clusterCount);
    for (let c = 0; c < clusterCount; c++) {
        const [ci, cj, ck] = centers[c];
        let cnt = 0;
        for (let di = -clusterRadius; di <= clusterRadius; di++) {
            const ni = ci + di;
            if (ni < 0 || ni >= gridN) continue;
            for (let dj = -clusterRadius; dj <= clusterRadius; dj++) {
                const nj = cj + dj;
                if (nj < 0 || nj >= gridN) continue;
                for (let dk = -clusterRadius; dk <= clusterRadius; dk++) {
                    const nk = ck + dk;
                    if (nk < 0 || nk >= gridN) continue;
                    cnt++;
                }
            }
        }
        clusterSizes[c] = cnt;
    }

    // Prefix-sum for offsets
    const offsets = new Uint32Array(clusterCount);
    let total = 0;
    for (let c = 0; c < clusterCount; c++) {
        offsets[c] = total;
        total += clusterSizes[c];
    }
    const clusterEntries = total;

    // Second pass: write cluster data + indices + rest offsets
    const clusters = [];
    const clusterIndices = new Uint32Array(clusterEntries);
    const restOffsets = [];

    for (let c = 0; c < clusterCount; c++) {
        const [ci, cj, ck] = centers[c];
        const cnt = clusterSizes[c];
        const offset = offsets[c];

        // Rest center of mass
        let comX = 0, comY = 0, comZ = 0;
        for (let di = -clusterRadius; di <= clusterRadius; di++) {
            const ni = ci + di;
            if (ni < 0 || ni >= gridN) continue;
            for (let dj = -clusterRadius; dj <= clusterRadius; dj++) {
                const nj = cj + dj;
                if (nj < 0 || nj >= gridN) continue;
                for (let dk = -clusterRadius; dk <= clusterRadius; dk++) {
                    const nk = ck + dk;
                    if (nk < 0 || nk >= gridN) continue;
                    const pid = idxAt(ni, nj, nk, gridN);
                    comX += positions[pid][0];
                    comY += positions[pid][1];
                    comZ += positions[pid][2];
                }
            }
        }
        comX /= cnt;
        comY /= cnt;
        comZ /= cnt;

        clusters.push({ restCom: [comX, comY, comZ], count: cnt, offset });

        let localIdx = 0;
        for (let di = -clusterRadius; di <= clusterRadius; di++) {
            const ni = ci + di;
            if (ni < 0 || ni >= gridN) continue;
            for (let dj = -clusterRadius; dj <= clusterRadius; dj++) {
                const nj = cj + dj;
                if (nj < 0 || nj >= gridN) continue;
                for (let dk = -clusterRadius; dk <= clusterRadius; dk++) {
                    const nk = ck + dk;
                    if (nk < 0 || nk >= gridN) continue;
                    const pid = idxAt(ni, nj, nk, gridN);
                    const slot = offset + localIdx;
                    clusterIndices[slot] = pid;
                    restOffsets[slot] = [
                        positions[pid][0] - comX,
                        positions[pid][1] - comY,
                        positions[pid][2] - comZ,
                    ];
                    localIdx++;
                }
            }
        }
    }

    return {
        meta: {
            gridN, cellSize, mass, seed, jitter,
            clusterRadius: clusterRadius * cellSize,
            clusterSpacing: clusterSpacing * cellSize,
            clusterRadiusCells: clusterRadius,
            clusterSpacingCells: clusterSpacing,
            particleCount, clusterCount, clusterEntries,
            surfaceVertexCount: surfaceIndices.length,
        },
        positions,
        surfaceIndices,
        clusters,
        clusterIndices: Array.from(clusterIndices),
        restOffsets,
    };
}

const { values } = parseArgs({
    options: {
        gridN:          { type: 'string', default: '5' },
        cellSize:       { type: 'string', default: '0.4' },
        clusterRadius:  { type: 'string', default: '0.8' },   // meters
        clusterSpacing: { type: 'string', default: '' },      // default: = clusterRadius
        jitter:         { type: 'string', default: '1' },     // cells
        seed:           { type: 'string', default: '42' },    // RNG seed
        mass:           { type: 'string', default: '1.0' },
        output:         { type: 'string', default: 'softbody_asset.json' },
    },
});

const gridN = parseInt(values.gridN, 10);
const cellSize = parseFloat(values.cellSize);
const clusterRadiusM = parseFloat(values.clusterRadius);
const clusterSpacingM = values.clusterSpacing ? parseFloat(values.clusterSpacing) : clusterRadiusM;
const jitter = parseInt(values.jitter, 10);
const seed = parseInt(values.seed, 10);
const mass = parseFloat(values.mass);

// Convert meter values to grid cells (quantized — actual physical size may differ).
const clusterRadius = Math.max(1, Math.round(clusterRadiusM / cellSize));
const clusterSpacing = Math.max(1, Math.round(clusterSpacingM / cellSize));

if (gridN < 2) { console.error(`error: --gridN must be >= 2, got ${gridN}`); process.exit(1); }
if (cellSize <= 0) { console.error(`error: --cellSize must be > 0, got ${cellSize}`); process.exit(1); }
if (clusterRadiusM <= 0) { console.error(`error: --clusterRadius must be > 0, got ${clusterRadiusM}`); process.exit(1); }
if (clusterRadius > gridN - 1) { console.error(`error: --clusterRadius ${clusterRadiusM}m = ${clusterRadius} cells, must be < gridN (${gridN}) = ${(gridN-1)*cellSize}m`); process.exit(1); }
if (clusterSpacingM < 0) { console.error(`error: --clusterSpacing must be >= 0, got ${clusterSpacingM}`); process.exit(1); }
if (jitter < 0) { console.error(`error: --jitter must be >= 0, got ${jitter}`); process.exit(1); }

// Warn if quantization changed the actual value.
if (clusterRadius * cellSize !== clusterRadiusM) {
    console.warn(`warning: --clusterRadius ${clusterRadiusM}m quantized to ${clusterRadius} cells = ${(clusterRadius * cellSize).toFixed(3)}m`);
}
if (clusterSpacing * cellSize !== clusterSpacingM) {
    console.warn(`warning: --clusterSpacing ${clusterSpacingM}m quantized to ${clusterSpacing} cells = ${(clusterSpacing * cellSize).toFixed(3)}m`);
}

const data = generate(gridN, cellSize, clusterRadius, clusterSpacing, jitter, seed, mass);
const outPath = resolve(values.output);
writeFileSync(outPath, JSON.stringify(data, null, 2), 'utf-8');

const m = data.meta;
const cubeEdge = (m.gridN - 1) * m.cellSize;
const maxCluster = (2 * m.clusterRadiusCells + 1) ** 3;
const sharedLayers = Math.max(0, 2 * m.clusterRadiusCells + 1 - m.clusterSpacingCells);
console.log(
    `Generated ${values.output}:\n` +
    `  gridN=${m.gridN}  cellSize=${m.cellSize}m  cubeEdge=${cubeEdge.toFixed(3)}m  cubeVolume=${(cubeEdge ** 3).toFixed(3)}m³\n` +
    `  clusterRadius=${m.clusterRadius}m (${m.clusterRadiusCells} cells)  clusterSpacing=${m.clusterSpacing}m (${m.clusterSpacingCells} cells)` +
    `  jitter=${m.jitter}  seed=${m.seed}  mass=${m.mass}\n` +
    `  particles=${m.particleCount}  clusters=${m.clusterCount}  clusterEntries=${m.clusterEntries}\n` +
    `  maxClusterMembers=${maxCluster}  sharedLayers≈${sharedLayers}  surfaceTriangles=${m.surfaceVertexCount / 3}\n` +
    `  perParticleMass=${(m.mass / m.particleCount).toFixed(6)} kg  invMass=${(m.particleCount / m.mass).toFixed(1)}`
);
