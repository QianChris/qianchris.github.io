import { RAPIER } from '@shaderlab/api';
import type * as RapierNS from '@dimforge/rapier3d-compat';

interface SplatColliderOptions {
    stride?: number;
    gridRes?: number;
    maxHulls?: number;
}

export interface SplatColliderResult {
    colliderDescs: RapierNS.ColliderDesc[];
    hullCount: number;
}

function downsample(centers: Float32Array, stride: number): Float32Array {
    const srcCount = centers.length / 4;
    const n = Math.ceil(srcCount / stride);
    const out = new Float32Array(n * 3);
    for (let i = 0, j = 0; i < srcCount && j < n; i += stride, j++) {
        const b = i * 4;
        out[j * 3] = centers[b];
        out[j * 3 + 1] = centers[b + 1];
        out[j * 3 + 2] = centers[b + 2];
    }
    return out;
}

function voxelCluster(points: Float32Array, gridRes: number): Float32Array[] {
    const n = points.length / 3;
    if (n < 4) return [points];

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < points.length; i += 3) {
        const x = points[i], y = points[i + 1], z = points[i + 2];
        if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
        if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
    }

    const pad = 0.001;
    const sizeX = (maxX - minX) + pad;
    const sizeY = (maxY - minY) + pad;
    const sizeZ = (maxZ - minZ) + pad;
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) return [points];

    const cells: Float32Array[] = new Array(gridRes * gridRes * gridRes);
    const counts: number[] = new Array(gridRes * gridRes * gridRes).fill(0);

    for (let i = 0; i < points.length; i += 3) {
        const x = points[i], y = points[i + 1], z = points[i + 2];
        const ix = Math.min(gridRes - 1, Math.floor((x - minX) / sizeX * gridRes));
        const iy = Math.min(gridRes - 1, Math.floor((y - minY) / sizeY * gridRes));
        const iz = Math.min(gridRes - 1, Math.floor((z - minZ) / sizeZ * gridRes));
        const ci = ix + iy * gridRes + iz * gridRes * gridRes;
        counts[ci]++;
    }

    for (let ci = 0; ci < cells.length; ci++) {
        if (counts[ci] > 0) {
            cells[ci] = new Float32Array(counts[ci] * 3);
        }
    }
    const cursors = new Uint32Array(cells.length);

    for (let i = 0; i < points.length; i += 3) {
        const x = points[i], y = points[i + 1], z = points[i + 2];
        const ix = Math.min(gridRes - 1, Math.floor((x - minX) / sizeX * gridRes));
        const iy = Math.min(gridRes - 1, Math.floor((y - minY) / sizeY * gridRes));
        const iz = Math.min(gridRes - 1, Math.floor((z - minZ) / sizeZ * gridRes));
        const ci = ix + iy * gridRes + iz * gridRes * gridRes;
        const cell = cells[ci];
        if (!cell) continue;
        const cursor = cursors[ci];
        cell[cursor] = x;
        cell[cursor + 1] = y;
        cell[cursor + 2] = z;
        cursors[ci] = cursor + 3;
    }

    const clusters: Float32Array[] = [];
    for (const cell of cells) {
        if (cell && cell.length >= 12) {
            clusters.push(cell);
        }
    }

    if (clusters.length === 0) {
        clusters.push(points);
    }

    return clusters;
}

function clusterAabbVolume(cluster: Float32Array): number {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < cluster.length; i += 3) {
        const x = cluster[i], y = cluster[i + 1], z = cluster[i + 2];
        if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
        if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
    }
    const vx = maxX - minX, vy = maxY - minY, vz = maxZ - minZ;
    if (vx <= 0 || vy <= 0 || vz <= 0) return 0;
    return vx * vy * vz;
}

export function generateSplatCollider(
    centers: Float32Array,
    options: SplatColliderOptions = {},
): SplatColliderResult {
    const stride = options.stride ?? 200;
    const gridRes = options.gridRes ?? 5;
    const maxHulls = options.maxHulls ?? 40;

    const sampled = downsample(centers, stride);
    const clusters = voxelCluster(sampled, gridRes);

    const hulls: { desc: RapierNS.ColliderDesc; volume: number }[] = [];

    for (const cluster of clusters) {
        const desc = RAPIER.ColliderDesc.convexHull(cluster);
        if (!desc) continue;
        const volume = clusterAabbVolume(cluster);
        hulls.push({ desc, volume });
    }

    hulls.sort((a, b) => b.volume - a.volume);
    const kept = hulls.slice(0, maxHulls);

    return {
        colliderDescs: kept.map(h => h.desc),
        hullCount: kept.length,
    };
}
