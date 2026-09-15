/**
 * Focused 3D-Gaussian-Splatting PLY loader.
 *
 * Parses a standard 3DGS PLY (binary little-endian; vertex properties
 * x/y/z, rot_0..3, scale_0..2, opacity, f_dc_0..2, optional f_rest_* SH)
 * and produces GPU-ready typed arrays with the same data semantics used by
 * @d5techs/d5-gaussian-splat-lib's SplatBuffer.fill* path:
 *   - scale    = exp(scale_k)              (log-space decoded)
 *   - opacity  = sigmoid(opacity)          -> [0,1]
 *   - color    = 0.5 + SH_C0 * f_dc_k      -> [0,1] linear RGB  (SH_C0 = 0.28209479)
 *   - rotation = normalized quaternion      (PLY order rot_0..3 = W,X,Y,Z -> X,Y,Z,W)
 *   - 3D covariance Sigma = R * diag(s^2) * R^T  (symmetric, 6 floats/splat)
 *
 * Output layout (GPU storage-buffer friendly):
 *   centers:     Float32Array  (N*4, xyz + pad)   -> array<vec4f>
 *   colors:      Float32Array  (N*4, rgba [0,1])  -> array<vec4f>
 *   covariances: Float32Array  (N*6, xx,xy,xz,yy,yz,zz) -> array<f32>
 *
 * The parsed arrays are the "SplatBuffer" a compute shader can subsequently
 * cache / rewrite in-place once uploaded to a WebGPU storage buffer.
 */

export interface SplatData {
    centers: Float32Array;
    colors: Float32Array;
    covariances: Float32Array;
    count: number;
}

const SH_C0 = 0.28209479177387814;

interface PlyProperty {
    name: string;
    type: string;
    isList: boolean;
    countType?: string;
}

interface ParsedHeader {
    format: 'ascii' | 'binary_little_endian' | 'binary_big_endian';
    vertexCount: number;
    properties: PlyProperty[];
    headerBytes: number;
}

function parseHeader(bytes: Uint8Array): ParsedHeader {
    // Header is ASCII up to "end_header\n".
    const decoder = new TextDecoder('ascii');
    // Find end_header marker.
    const marker = 'end_header';
    let lineEnd = -1;
    let i = 0;
    // Scan line by line to be safe with newlines.
    let start = 0;
    let format: ParsedHeader['format'] = 'ascii';
    let vertexCount = 0;
    const properties: PlyProperty[] = [];
    while (start < bytes.length) {
        let nl = bytes.indexOf(0x0a, start);
        if (nl === -1) { nl = bytes.length; }
        const line = decoder.decode(bytes.subarray(start, nl)).trim();
        start = nl + 1;
        if (line === marker) { lineEnd = start; break; }
        const lower = line.toLowerCase();
        if (lower.startsWith('format ')) {
            const parts = line.split(/\s+/);
            format = (parts[1] as ParsedHeader['format']) ?? 'ascii';
        } else if (lower.startsWith('element vertex')) {
            const parts = line.split(/\s+/);
            vertexCount = parseInt(parts[2], 10) || 0;
        } else if (lower.startsWith('property ')) {
            const parts = line.split(/\s+/);
            if (parts[1] === 'list') {
                properties.push({ name: parts[4], type: parts[3], isList: true, countType: parts[2] });
            } else {
                properties.push({ name: parts[2], type: parts[1], isList: false });
            }
        }
    }
    if (lineEnd === -1) throw new Error('PLY: end_header not found');
    return { format, vertexCount, properties, headerBytes: lineEnd };
}

/** Read a little-endian float32 at byte offset. */
function readF32(view: DataView, offset: number): number {
    return view.getFloat32(offset, true);
}

/** Index of a property by name (case-sensitive, exact). */
function indexOfProp(props: PlyProperty[], name: string): number {
    for (let i = 0; i < props.length; i++) if (props[i].name === name) return i;
    return -1;
}

/**
 * Load a 3DGS PLY file from `url` and return GPU-ready splat arrays.
 * Throws if the file is not binary_little_endian or lacks the core properties.
 */
export async function loadSplatPly(url: string): Promise<SplatData> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`SplatLoader: HTTP ${resp.status} for ${url}`);
    const ab = await resp.arrayBuffer();
    const bytes = new Uint8Array(ab);
    const header = parseHeader(bytes);

    if (header.format !== 'binary_little_endian') {
        throw new Error(`SplatLoader: only binary_little_endian PLY supported (got '${header.format}')`);
    }
    if (header.vertexCount <= 0) {
        throw new Error('SplatLoader: PLY has no vertices');
    }

    // Resolve property offsets (only float scalars are expected for 3DGS PLY).
    const props = header.properties;
    for (const p of props) {
        if (!p.isList && p.type !== 'float' && p.type !== 'double') {
            throw new Error(`SplatLoader: unsupported property type '${p.type}' for '${p.name}'`);
        }
        if (p.type === 'double') {
            throw new Error(`SplatLoader: double properties not supported ('${p.name}')`);
        }
    }
    if (props.some(p => p.isList)) {
        throw new Error('SplatLoader: list properties not supported');
    }

    const stride = props.length * 4; // all f32
    const ix = indexOfProp(props, 'x');
    const iy = indexOfProp(props, 'y');
    const iz = indexOfProp(props, 'z');
    const irot0 = indexOfProp(props, 'rot_0');
    const irot1 = indexOfProp(props, 'rot_1');
    const irot2 = indexOfProp(props, 'rot_2');
    const irot3 = indexOfProp(props, 'rot_3');
    const is0 = indexOfProp(props, 'scale_0');
    const is1 = indexOfProp(props, 'scale_1');
    const is2 = indexOfProp(props, 'scale_2');
    const iop = indexOfProp(props, 'opacity');
    const idc0 = indexOfProp(props, 'f_dc_0');
    const idc1 = indexOfProp(props, 'f_dc_1');
    const idc2 = indexOfProp(props, 'f_dc_2');

    if (ix < 0 || iy < 0 || iz < 0) throw new Error('SplatLoader: missing position (x/y/z)');
    if (irot0 < 0 || irot1 < 0 || irot2 < 0 || irot3 < 0) throw new Error('SplatLoader: missing rotation (rot_0..3)');
    if (is0 < 0 || is1 < 0 || is2 < 0) throw new Error('SplatLoader: missing scale (scale_0..2)');
    if (iop < 0) throw new Error('SplatLoader: missing opacity');
    // Color: prefer f_dc_*; fall back to red/green/blue if present.
    let useShColor = idc0 >= 0 && idc1 >= 0 && idc2 >= 0;
    const ir = indexOfProp(props, 'red');
    const ig = indexOfProp(props, 'green');
    const ib = indexOfProp(props, 'blue');
    if (!useShColor && (ir < 0 || ig < 0 || ib < 0)) {
        throw new Error('SplatLoader: missing color (f_dc_0..2 or red/green/blue)');
    }

    const n = header.vertexCount;
    const view = new DataView(ab, header.headerBytes, ab.byteLength - header.headerBytes);

    const centers = new Float32Array(n * 4);
    const colors = new Float32Array(n * 4);
    const covariances = new Float32Array(n * 6);

    for (let s = 0; s < n; s++) {
        const base = s * stride;

        const x = readF32(view, base + ix * 4);
        const y = readF32(view, base + iy * 4);
        const z = readF32(view, base + iz * 4);
        centers[s * 4] = x;
        centers[s * 4 + 1] = y;
        centers[s * 4 + 2] = z;
        centers[s * 4 + 3] = 0;

        // Quaternion: PLY order rot_0..3 = W,X,Y,Z -> internal X,Y,Z,W
        let qw = readF32(view, base + irot0 * 4);
        let qx = readF32(view, base + irot1 * 4);
        let qy = readF32(view, base + irot2 * 4);
        let qz = readF32(view, base + irot3 * 4);
        const ql = Math.hypot(qx, qy, qz, qw) || 1;
        qx /= ql; qy /= ql; qz /= ql; qw /= ql;
        // Guard: degenerate quaternion -> identity
        if (!isFinite(qx) || !isFinite(qy) || !isFinite(qz) || !isFinite(qw)) {
            qx = 0; qy = 0; qz = 0; qw = 1;
        }

        const sx = Math.exp(readF32(view, base + is0 * 4));
        const sy = Math.exp(readF32(view, base + is1 * 4));
        const sz = Math.exp(readF32(view, base + is2 * 4));

        const op = readF32(view, base + iop * 4);
        const alpha = 1 / (1 + Math.exp(-op));

        let r = 0.5, g = 0.5, b = 0.5;
        if (useShColor) {
            r = 0.5 + SH_C0 * readF32(view, base + idc0 * 4);
            g = 0.5 + SH_C0 * readF32(view, base + idc1 * 4);
            b = 0.5 + SH_C0 * readF32(view, base + idc2 * 4);
        } else {
            r = readF32(view, base + ir * 4) / 255;
            g = readF32(view, base + ig * 4) / 255;
            b = readF32(view, base + ib * 4) / 255;
        }
        colors[s * 4] = r;
        colors[s * 4 + 1] = g;
        colors[s * 4 + 2] = b;
        colors[s * 4 + 3] = alpha;

        // Rotation matrix R from quaternion (X,Y,Z,W).
        const xx = qx * qx, yy = qy * qy, zz = qz * qz;
        const xy = qx * qy, xz = qx * qz, yz = qy * qz;
        const wx = qw * qx, wy = qw * qy, wz = qw * qz;
        // Column-major R: R[col*3 + row]
        const r00 = 1 - 2 * (yy + zz), r10 = 2 * (xy + wz),     r20 = 2 * (xz - wy);
        const r01 = 2 * (xy - wz),     r11 = 1 - 2 * (xx + zz), r21 = 2 * (yz + wx);
        const r02 = 2 * (xz + wy),     r12 = 2 * (yz - wx),     r22 = 1 - 2 * (xx + yy);

        // Sigma = R * diag(sx^2, sy^2, sz^2) * R^T
        const sx2 = sx * sx, sy2 = sy * sy, sz2 = sz * sz;
        const c00 = sx2 * r00 * r00 + sy2 * r01 * r01 + sz2 * r02 * r02;
        const c01 = sx2 * r00 * r10 + sy2 * r01 * r11 + sz2 * r02 * r12;
        const c02 = sx2 * r00 * r20 + sy2 * r01 * r21 + sz2 * r02 * r22;
        const c11 = sx2 * r10 * r10 + sy2 * r11 * r11 + sz2 * r12 * r12;
        const c12 = sx2 * r10 * r20 + sy2 * r11 * r21 + sz2 * r12 * r22;
        const c22 = sx2 * r20 * r20 + sy2 * r21 * r21 + sz2 * r22 * r22;

        // Order: xx, xy, xz, yy, yz, zz (matches d5 fillSplatCovarianceArray)
        covariances[s * 6] = c00;
        covariances[s * 6 + 1] = c01;
        covariances[s * 6 + 2] = c02;
        covariances[s * 6 + 3] = c11;
        covariances[s * 6 + 4] = c12;
        covariances[s * 6 + 5] = c22;
    }

    return { centers, colors, covariances, count: n };
}
