#!/usr/bin/env python3
"""Generate a pre-baked soft-body asset file for demo9's PBD plugin.

The asset contains all structural data that used to be generated at runtime:
  - particle rest positions (axis-aligned, centered at origin, no rotation)
  - surface triangle indices (cube shell)
  - shape-matching cluster data (sparse centers, Chebyshev membership)

Cluster center selection: greedy, minimum Chebyshev spacing `clusterSpacing`
between centers. Each center's cluster = all particles within Chebyshev
radius `clusterRadius`. With clusterSpacing=clusterRadius, every particle is
covered by at least one cluster (full coverage, minimal overlap). With
clusterSpacing=0, every particle is a center (densest, maximum overlap).

The runtime PbdManager loads this file, applies a random rotation + the
entity's Transform translation, and writes the result to GPU buffers.
Changing gridN / cellSize / clusterRadius / clusterSpacing requires re-running
this script — they are not runtime-tweakable component fields.

Usage:
    python generate_softbody.py [--gridN 5] [--cellSize 0.4]
                                [--clusterRadius 2] [--clusterSpacing 2]
                                [--mass 1.0] [--output softbody_asset.json]
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Tuple, Callable


def idx_at(i: int, j: int, k: int, n: int) -> int:
    return i + j * n + k * n * n


def build_surface_indices(n: int) -> List[int]:
    """Cube shell triangulation — 6 faces, 2 triangles per cell, consistent winding."""
    out: List[int] = []
    at = lambda i, j, k: idx_at(i, j, k, n)

    for j in range(n - 1):
        for k in range(n - 1):
            a, b, c, d = at(0, j, k), at(0, j+1, k), at(0, j+1, k+1), at(0, j, k+1)
            out.extend([a, b, c, a, c, d])
            a2, b2, c2, d2 = at(n-1, j, k), at(n-1, j+1, k), at(n-1, j+1, k+1), at(n-1, j, k+1)
            out.extend([a2, c2, b2, a2, d2, c2])

    for i in range(n - 1):
        for k in range(n - 1):
            a, b, c, d = at(i, 0, k), at(i+1, 0, k), at(i+1, 0, k+1), at(i, 0, k+1)
            out.extend([a, c, b, a, d, c])
            a2, b2, c2, d2 = at(i, n-1, k), at(i+1, n-1, k), at(i+1, n-1, k+1), at(i, n-1, k+1)
            out.extend([a2, b2, c2, a2, c2, d2])

    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = at(i, j, 0), at(i+1, j, 0), at(i+1, j+1, 0), at(i, j+1, 0)
            out.extend([a, b, c, a, c, d])
            a2, b2, c2, d2 = at(i, j, n-1), at(i+1, j, n-1), at(i+1, j+1, n-1), at(i, j+1, n-1)
            out.extend([a2, c2, b2, a2, d2, c2])

    return out


def make_rng(seed: int) -> Callable[[], float]:
    """Seeded LCG random number generator (deterministic). Returns floats in [0, 1)."""
    state = [seed & 0xFFFFFFFF or 1]
    def _rng() -> float:
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] / 0x100000000
    return _rng


def rand_int(rng: Callable[[], float], lo: int, hi: int) -> int:
    return int(rng() * (hi - lo + 1)) + lo


def shuffle(rng: Callable[[], float], arr: list) -> list:
    for i in range(len(arr) - 1, 0, -1):
        j = int(rng() * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def select_centers(grid_n: int, radius_cells: int, spacing_cells: int,
                   jitter: int, rng: Callable[[], float]) -> List[Tuple[int, int, int]]:
    """Option C: jittered sublattice + greedy dedup + forced coverage.

    1. Generate candidate centers on a regular sublattice (spacing = spacing_cells),
       each displaced by ±jitter cells (clamped to grid).
    2. Shuffle candidates (seeded) and greedily select, skipping any within
       Chebyshev distance spacing_cells of an already-selected center.
    3. Coverage check: every particle must be within radius_cells of some center.
       Uncovered particles are force-added as new centers (may violate spacing —
       coverage is mandatory, spacing is a soft constraint).

    With radius_cells=R and spacing_cells=S, adjacent centers share (2R+1−S) layers
    of particles. For 1–2 layers shared: S = 2R or S = 2R−1.
    """
    # 1. Sublattice candidates with jitter
    candidates: List[Tuple[int, int, int]] = []
    for k in range(0, grid_n, spacing_cells):
        for j in range(0, grid_n, spacing_cells):
            for i in range(0, grid_n, spacing_cells):
                ci = max(0, min(grid_n - 1, i + rand_int(rng, -jitter, jitter)))
                cj = max(0, min(grid_n - 1, j + rand_int(rng, -jitter, jitter)))
                ck = max(0, min(grid_n - 1, k + rand_int(rng, -jitter, jitter)))
                candidates.append((ci, cj, ck))

    # Ensure the far edge is represented even if gridN isn't a multiple of spacing.
    if (grid_n - 1) % spacing_cells != 0:
        for j in range(0, grid_n, spacing_cells):
            for i in range(0, grid_n, spacing_cells):
                candidates.append((
                    max(0, min(grid_n - 1, i + rand_int(rng, -jitter, jitter))),
                    max(0, min(grid_n - 1, j + rand_int(rng, -jitter, jitter))),
                    grid_n - 1,
                ))
        for i in range(0, grid_n, spacing_cells):
            for k in range(0, grid_n, spacing_cells):
                candidates.append((
                    max(0, min(grid_n - 1, i + rand_int(rng, -jitter, jitter))),
                    grid_n - 1,
                    max(0, min(grid_n - 1, k + rand_int(rng, -jitter, jitter))),
                ))
        for j in range(0, grid_n, spacing_cells):
            for k in range(0, grid_n, spacing_cells):
                candidates.append((
                    grid_n - 1,
                    max(0, min(grid_n - 1, j + rand_int(rng, -jitter, jitter))),
                    max(0, min(grid_n - 1, k + rand_int(rng, -jitter, jitter))),
                ))

    # 2. Shuffle + greedy select with min spacing
    shuffle(rng, candidates)
    centers: List[Tuple[int, int, int]] = []
    for (ci, cj, ck) in candidates:
        too_close = False
        for (si, sj, sk) in centers:
            if (abs(ci - si) <= spacing_cells and
                abs(cj - sj) <= spacing_cells and
                abs(ck - sk) <= spacing_cells):
                too_close = True
                break
        if not too_close:
            centers.append((ci, cj, ck))

    # 3. Coverage check: force-add centers for uncovered particles
    added = 0
    for k in range(grid_n):
        for j in range(grid_n):
            for i in range(grid_n):
                covered = False
                for (ci, cj, ck2) in centers:
                    if (abs(i - ci) <= radius_cells and
                        abs(j - cj) <= radius_cells and
                        abs(k - ck2) <= radius_cells):
                        covered = True
                        break
                if not covered:
                    centers.append((i, j, k))
                    added += 1
    if added > 0:
        print(f"info: coverage check added {added} extra center(s) for uncovered particles",
              file=sys.stderr)

    return centers


def generate(grid_n: int, cell_size: float, cluster_radius: int,
             cluster_spacing: int, jitter: int, seed: int,
             mass: float) -> Dict[str, Any]:
    particle_count = grid_n ** 3
    inv_mass = particle_count / mass
    half = (grid_n - 1) / 2.0

    # ── particle rest positions (axis-aligned, centered at origin) ───────
    positions: List[List[float]] = []
    for k in range(grid_n):
        for j in range(grid_n):
            for i in range(grid_n):
                x = (i - half) * cell_size
                y = (j - half) * cell_size
                z = (k - half) * cell_size
                positions.append([x, y, z, inv_mass])

    # ── surface indices ─────────────────────────────────────────────────
    surface_indices = build_surface_indices(grid_n)

    # ── select cluster centers (jittered sublattice + greedy + coverage) ─
    rng = make_rng(seed)
    centers = select_centers(grid_n, cluster_radius, cluster_spacing, jitter, rng)
    cluster_count = len(centers)

    # First pass: count members per cluster
    cluster_sizes = [0] * cluster_count
    for c, (ci, cj, ck) in enumerate(centers):
        cnt = 0
        for di in range(-cluster_radius, cluster_radius + 1):
            ni = ci + di
            if ni < 0 or ni >= grid_n:
                continue
            for dj in range(-cluster_radius, cluster_radius + 1):
                nj = cj + dj
                if nj < 0 or nj >= grid_n:
                    continue
                for dk in range(-cluster_radius, cluster_radius + 1):
                    nk = ck + dk
                    if nk < 0 or nk >= grid_n:
                        continue
                    cnt += 1
        cluster_sizes[c] = cnt

    # Prefix-sum for offsets
    offsets = [0] * cluster_count
    total = 0
    for c in range(cluster_count):
        offsets[c] = total
        total += cluster_sizes[c]

    cluster_entries = total

    # Second pass: write cluster data + indices + rest offsets
    clusters: List[Dict[str, Any]] = []
    cluster_indices: List[int] = [0] * cluster_entries
    rest_offsets: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(cluster_entries)]

    for c, (ci, cj, ck) in enumerate(centers):
        cnt = cluster_sizes[c]
        offset = offsets[c]

        # Rest center of mass
        com_x = com_y = com_z = 0.0
        for di in range(-cluster_radius, cluster_radius + 1):
            ni = ci + di
            if ni < 0 or ni >= grid_n:
                continue
            for dj in range(-cluster_radius, cluster_radius + 1):
                nj = cj + dj
                if nj < 0 or nj >= grid_n:
                    continue
                for dk in range(-cluster_radius, cluster_radius + 1):
                    nk = ck + dk
                    if nk < 0 or nk >= grid_n:
                        continue
                    pid = idx_at(ni, nj, nk, grid_n)
                    com_x += positions[pid][0]
                    com_y += positions[pid][1]
                    com_z += positions[pid][2]
        com_x /= cnt
        com_y /= cnt
        com_z /= cnt

        clusters.append({
            "restCom": [com_x, com_y, com_z],
            "count": cnt,
            "offset": offset,
        })

        local_idx = 0
        for di in range(-cluster_radius, cluster_radius + 1):
            ni = ci + di
            if ni < 0 or ni >= grid_n:
                continue
            for dj in range(-cluster_radius, cluster_radius + 1):
                nj = cj + dj
                if nj < 0 or nj >= grid_n:
                    continue
                for dk in range(-cluster_radius, cluster_radius + 1):
                    nk = ck + dk
                    if nk < 0 or nk >= grid_n:
                        continue
                    pid = idx_at(ni, nj, nk, grid_n)
                    slot = offset + local_idx
                    cluster_indices[slot] = pid
                    rest_offsets[slot] = [
                        positions[pid][0] - com_x,
                        positions[pid][1] - com_y,
                        positions[pid][2] - com_z,
                    ]
                    local_idx += 1

    return {
        "meta": {
            "gridN": grid_n,
            "cellSize": cell_size,
            "mass": mass,
            "seed": seed,
            "jitter": jitter,
            "clusterRadius": cluster_radius * cell_size,
            "clusterSpacing": cluster_spacing * cell_size,
            "clusterRadiusCells": cluster_radius,
            "clusterSpacingCells": cluster_spacing,
            "particleCount": particle_count,
            "clusterCount": cluster_count,
            "clusterEntries": cluster_entries,
            "surfaceVertexCount": len(surface_indices),
        },
        "positions": positions,
        "surfaceIndices": surface_indices,
        "clusters": clusters,
        "clusterIndices": cluster_indices,
        "restOffsets": rest_offsets,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pre-baked soft-body asset file for demo9 PBD.",
    )
    parser.add_argument("--gridN", type=int, default=5, help="grid resolution (gridN^3 particles)")
    parser.add_argument("--cellSize", type=float, default=0.4, help="particle spacing / cell edge length (m)")
    parser.add_argument("--clusterRadius", type=float, default=0.8, help="Chebyshev membership radius (meters)")
    parser.add_argument("--clusterSpacing", type=float, default=None,
                        help="min Chebyshev distance between cluster centers (meters, default: = clusterRadius)")
    parser.add_argument("--jitter", type=int, default=1, help="sublattice jitter in cells (0 = regular grid)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible jitter")
    parser.add_argument("--mass", type=float, default=1.0, help="total mass (kg)")
    parser.add_argument("--output", type=str, default="softbody_asset.json", help="output file path")
    args = parser.parse_args()

    if args.gridN < 2:
        print(f"error: --gridN must be >= 2, got {args.gridN}", file=sys.stderr)
        sys.exit(1)
    if args.cellSize <= 0:
        print(f"error: --cellSize must be > 0, got {args.cellSize}", file=sys.stderr)
        sys.exit(1)
    if args.clusterRadius <= 0:
        print(f"error: --clusterRadius must be > 0, got {args.clusterRadius}", file=sys.stderr)
        sys.exit(1)
    if args.jitter < 0:
        print(f"error: --jitter must be >= 0, got {args.jitter}", file=sys.stderr)
        sys.exit(1)

    cluster_spacing_m = args.clusterSpacing if args.clusterSpacing is not None else args.clusterRadius

    # Convert meter values to grid cells (quantized — actual physical size may differ).
    cluster_radius = max(1, round(args.clusterRadius / args.cellSize))
    cluster_spacing = max(1, round(cluster_spacing_m / args.cellSize))

    if cluster_radius > args.gridN - 1:
        print(f"error: --clusterRadius {args.clusterRadius}m = {cluster_radius} cells, "
              f"must be < gridN ({args.gridN}) = {(args.gridN-1)*args.cellSize}m", file=sys.stderr)
        sys.exit(1)

    # Warn if quantization changed the actual value.
    if cluster_radius * args.cellSize != args.clusterRadius:
        print(f"warning: --clusterRadius {args.clusterRadius}m quantized to {cluster_radius} cells "
              f"= {cluster_radius * args.cellSize:.3f}m", file=sys.stderr)
    if cluster_spacing * args.cellSize != cluster_spacing_m:
        print(f"warning: --clusterSpacing {cluster_spacing_m}m quantized to {cluster_spacing} cells "
              f"= {cluster_spacing * args.cellSize:.3f}m", file=sys.stderr)

    data = generate(args.gridN, args.cellSize, cluster_radius, cluster_spacing,
                    args.jitter, args.seed, args.mass)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    m = data["meta"]
    cube_edge = (m["gridN"] - 1) * m["cellSize"]
    max_cluster = (2 * m["clusterRadiusCells"] + 1) ** 3
    shared_layers = max(0, 2 * m["clusterRadiusCells"] + 1 - m["clusterSpacingCells"])
    print(
        f"Generated {args.output}:\n"
        f"  gridN={m['gridN']}  cellSize={m['cellSize']}m  cubeEdge={cube_edge:.3f}m  cubeVolume={cube_edge**3:.3f}m³\n"
        f"  clusterRadius={m['clusterRadius']}m ({m['clusterRadiusCells']} cells)"
        f"  clusterSpacing={m['clusterSpacing']}m ({m['clusterSpacingCells']} cells)"
        f"  jitter={m['jitter']}  seed={m['seed']}  mass={m['mass']}\n"
        f"  particles={m['particleCount']}  clusters={m['clusterCount']}  clusterEntries={m['clusterEntries']}\n"
        f"  maxClusterMembers={max_cluster}  sharedLayers≈{shared_layers}"
        f"  surfaceTriangles={m['surfaceVertexCount'] // 3}\n"
        f"  perParticleMass={m['mass'] / m['particleCount']:.6f} kg  invMass={m['particleCount'] / m['mass']:.1f}"
    )


if __name__ == "__main__":
    main()
