# WGSL 实现无染色 Shape Matching PBD（基于原子操作）

## 目录

- [1. 核心思路](#1-核心思路)
- [2. 为什么需要原子操作](#2-为什么需要原子操作)
- [3. WGSL Fixed-Point 原子累加方案](#3-wgsl-fixed-point-原子累加方案)
- [4. 完整 WGSL 实现](#4-完整-wgsl-实现)
  - [4.1 数据布局](#41-数据布局)
  - [4.2 工具函数](#42-工具函数)
  - [4.3 Shape Matching 求解核](#43-shape-matching-求解核)
  - [4.4 修正量应用核](#44-修正量应用核)
- [5. CPU 端调度](#5-cpu-端调度)
- [6. 注意事项与优化](#6-注意事项与优化)
- [7. 总结](#7-总结)

---

## 1. 核心思路

**把 Shape Matching 的 correction 拆成"计算"和"累加"两步**：

1. 所有 Cluster 并行计算 goal position，每个线程只读当前粒子位置；
2. 通过 Fixed-Point 编码的 `atomicAdd` 把 correction 累加到每个粒子专属的共享缓冲区；
3. 最后开一个全局 Pass，把累加值除以权重计数，统一 Apply 到位置上。

这样**不需要预先对 Cluster 做 Graph Coloring**，一个 `dispatch` 就能处理所有约束，动态增删 Cluster 时维护成本极低。

---

## 2. 为什么需要原子操作

Shape Matching（区域级）中，一个粒子通常属于多个 Cluster（比如重叠的邻域）。每个 Cluster 独立计算该粒子的 goal position 并施加拉力。

如果不做染色（Graph Coloring），多个线程会同时写同一个粒子的位置，产生 Race Condition。传统做法是先把 Cluster 分成若干组（Color），保证同组内 Cluster 不共享粒子，然后按组顺序执行。但这需要预处理，且动态拓扑时维护成本高。

**原子操作方案**完全跳过染色：
- 每个线程只读当前粒子位置，计算完 correction 后不直接写位置；
- 把 `correction` 通过 `atomicAdd` 累加到每个粒子专属的累加缓冲区；
- 最后统一把累加值除以权重，加到位置上。

这相当于把 Gauss-Seidel 改成了带原子累加的 Jacobi 风格，实现简单，动态拓扑友好。

---

## 3. WGSL Fixed-Point 原子累加方案

WGSL 原生不支持 `atomic<f32>`，只支持 `atomic<i32>` / `atomic<u32>`。因此需要 **Fixed-Point Encoding**：

```wgsl
const FIXED_SCALE: f32 = 1024.0;  // 精度/范围权衡

fn atomicAddF32(addr: ptr<storage, atomic<i32>, read_write>, value: f32) {
    atomicAdd(addr, i32(round(value * FIXED_SCALE)));
}
```

- 累加前：`float → int32`，放大 1024 倍；
- 读取时：`int32 → float`，缩小 1024 倍；
- `i32` 上限约 2.1e9，对应 float 范围约 ±200万，对常规场景足够。

如果有多个 Cluster 对同一粒子施加了修正，缓冲区里存的是它们的**和**，最后除以权重（也是原子累加的计数）做平均。

---

## 4. 完整 WGSL 实现

假设：
- 每个 Cluster 预存了：粒子索引列表、Rest 质心、各粒子相对于 Rest 质心的偏移；
- 一个线程处理一个 Cluster（适合 Cluster 数量多的场景）；
- 使用 Polar Decomposition（迭代法）提取旋转矩阵。

### 4.1 数据布局

```wgsl
struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    invMass: f32,
};

struct Cluster {
    offset: u32,          // 在 clusterIndices / restOffsets 中的起始
    count: u32,           // 粒子数
    restCom: vec3<f32>,   // 预计算的 Rest 质心
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec3<f32>>; // 预测后的位置

@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(3) var<storage, read> clusterIndices: array<u32>;   // 粒子索引
@group(0) @binding(4) var<storage, read> restOffsets: array<vec3<f32>>; // 相对 Rest COM 的偏移

// 原子累加缓冲区（Fixed-Point）
@group(0) @binding(5) var<storage, read_write> accumX: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> accumY: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> accumZ: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> accumW: array<atomic<i32>>; // 权重计数

@group(0) @binding(9) var<uniform> params: SimParams;

struct SimParams {
    numClusters: u32,
    numParticles: u32,
    stiffness: f32,   // 0..1
    dt: f32,
};

const FIXED_SCALE: f32 = 1024.0;
```

### 4.2 工具函数

```wgsl
fn atomicAddF32(addr: ptr<storage, atomic<i32>, read_write>, value: f32) {
    atomicAdd(addr, i32(round(value * FIXED_SCALE)));
}

fn outerProduct(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
}

// 3x3 矩阵求逆
fn inverseMat3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0][0]; let b = m[0][1]; let c = m[0][2];
    let d = m[1][0]; let e = m[1][1]; let f = m[1][2];
    let g = m[2][0]; let h = m[2][1]; let i = m[2][2];

    let A = e*i - f*h; let B = c*h - b*i; let C = b*f - c*e;
    let D = f*g - d*i; let E = a*i - c*g; let F = c*d - a*f;
    let G = d*h - e*g; let H = b*g - a*h; let I = a*e - b*d;

    let det = a*A + b*D + c*G;
    let invDet = 1.0 / det;

    return mat3x3<f32>(
        vec3<f32>(A, B, C) * invDet,
        vec3<f32>(D, E, F) * invDet,
        vec3<f32>(G, H, I) * invDet
    );
}

// Polar Decomposition: A = R * S, 提取旋转矩阵 R
// 迭代公式: R_{k+1} = 0.5 * (R_k + transpose(inverse(R_k)))
fn polarDecompose(A: mat3x3<f32>) -> mat3x3<f32> {
    var R = A;
    // 正则化，防止奇异
    R = R + mat3x3<f32>(
        vec3<f32>(1e-6, 0.0, 0.0), 
        vec3<f32>(0.0, 1e-6, 0.0), 
        vec3<f32>(0.0, 0.0, 1e-6)
    );

    for (var iter = 0; iter < 8; iter++) {
        let invRt = inverseMat3(transpose(R));
        R = 0.5 * (R + invRt);
    }
    return R;
}
```

### 4.3 Shape Matching 求解核

**一个线程 = 一个 Cluster**

```wgsl
@compute @workgroup_size(64)
fn solveShapeMatching(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let cid = gid.x;
    if (cid >= params.numClusters) { return; }

    let cl = clusters[cid];

    // ---- 3.1 计算当前质心 ----
    var com = vec3<f32>(0.0);
    for (var i = 0u; i < cl.count; i++) {
        let pid = clusterIndices[cl.offset + i];
        com += positions[pid];
    }
    com /= f32(cl.count);

    // ---- 3.2 计算协方差矩阵 A = sum( (pi - com) * (qi)^T ) ----
    var A = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var i = 0u; i < cl.count; i++) {
        let pid = clusterIndices[cl.offset + i];
        let pi = positions[pid] - com;
        let qi = restOffsets[cl.offset + i]; // 已预计算，相对 restCom
        A = A + outerProduct(pi, qi);
    }

    // ---- 3.3 提取旋转 R ----
    let R = polarDecompose(A);

    // ---- 3.4 计算 Goal Position 并原子累加 Correction ----
    for (var i = 0u; i < cl.count; i++) {
        let pid = clusterIndices[cl.offset + i];
        let qi = restOffsets[cl.offset + i];
        let goal = com + R * qi;
        let corr = (goal - positions[pid]) * params.stiffness;

        atomicAddF32(&accumX[pid], corr.x);
        atomicAddF32(&accumY[pid], corr.y);
        atomicAddF32(&accumZ[pid], corr.z);
        atomicAdd(&accumW[pid], i32(FIXED_SCALE)); // weight += 1.0
    }
}
```

### 4.4 修正量应用核

```wgsl
@compute @workgroup_size(256)
fn applyCorrections(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let pid = gid.x;
    if (pid >= params.numParticles) { return; }

    let w = f32(atomicLoad(&accumW[pid])) / FIXED_SCALE;
    if (w > 0.0) {
        let dx = f32(atomicLoad(&accumX[pid])) / FIXED_SCALE;
        let dy = f32(atomicLoad(&accumY[pid])) / FIXED_SCALE;
        let dz = f32(atomicLoad(&accumZ[pid])) / FIXED_SCALE;

        positions[pid] += vec3<f32>(dx, dy, dz) / w;

        // 清零，为下一次迭代做准备
        atomicStore(&accumX[pid], 0);
        atomicStore(&accumY[pid], 0);
        atomicStore(&accumZ[pid], 0);
        atomicStore(&accumW[pid], 0);
    }
}
```

---

## 5. CPU 端调度

```cpp
// 每帧/每子步
for (int substep = 0; substep < numSubsteps; ++substep) {
    // 1. Predict: position += velocity * dt, 应用外力
    encoder.Dispatch(predictKernel, (numParticles + 255)/256);

    // 2. 可选：碰撞/其他约束...

    // 3. Shape Matching（多次迭代）
    for (int iter = 0; iter < solverIterations; ++iter) {
        // 3.1 所有 Cluster 并行计算并原子累加
        encoder.Dispatch(solveShapeMatching, (numClusters + 63)/64);
        encoder.Barrier(); // 内存屏障，确保原子操作完成

        // 3.2 统一 Apply
        encoder.Dispatch(applyCorrections, (numParticles + 255)/256);
        encoder.Barrier();
    }

    // 4. Update Velocity: v = (x - x_prev) / dt
    encoder.Dispatch(updateVelocityKernel, (numParticles + 255)/256);
}
```

**关键点**：`solveShapeMatching` 和 `applyCorrections` 之间必须有 **内存屏障**，确保原子累加全部完成后再读取。

---

## 6. 注意事项与优化

| 问题 | 方案 |
|------|------|
| **Fixed-Point 精度** | `FIXED_SCALE = 1024` 对一般场景够用；若模拟尺度极大，可降为 `256` 或改用 `atomic<u32>` 做 IEEE 754 位操作（更复杂）。 |
| **Polar Decomposition 发散** | 如果 Cluster 退化（如全部粒子共线），`A` 接近奇异，迭代会不稳定。建议给 `A` 加小量正则化（代码中已加 `1e-6 * I`）。 |
| **性能：原子竞争** | 如果大量 Cluster 共享同一粒子（如全局 Shape Matching + 局部 Region），原子竞争会很严重。此时可考虑 **Shared Memory 预归约**：一个 Workgroup 处理一个 Cluster，内部先用 `workgroup` 内存做求和，最后只对全局内存做一次原子写。 |
| **全局 vs 区域 Shape Matching** | 如果每个粒子**只属于一个 Cluster**（全局 Shape Matching），那根本不需要原子操作，直接写 `positions` 即可，性能更好。上面代码仍然正确，只是没有竞争。 |
| **内存屏障** | 同一 Compute Pass 内，两个 Dispatch 之间需要 `storageBarrier()` 或 CPU 端 Pipeline Barrier。在 WebGPU 中，通常放在不同 Pass 或显式插入 barrier。 |

---

## 7. 总结

不用染色的核心就是 **"计算分离 + 原子累加"**：

1. 把传统 PBD 中"计算 correction → 立即写位置"的串行依赖拆开；
2. 所有 Cluster 并行算出自己的 correction，通过 `atomic<i32>`（Fixed-Point 编码）累加到每个粒子的私有槽位；
3. 最后统一做 `position += accum / weight`。

这在 WGSL 里完全可行，虽然原子操作有一定开销，但避免了 Graph Coloring 的预处理和动态维护成本，代码也更简洁。如果 Cluster 划分比较规则（比如 Uniform Grid 上的邻域），还可以进一步结合 Workgroup Shared Memory 减少全局原子竞争。
