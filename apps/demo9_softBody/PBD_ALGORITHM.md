# GPU PBD 软体仿真 — 算法文档 (Shape Matching + Atomic)

## 1. 概述

Position-Based Dynamics (PBD)，使用 **Shape Matching 约束 + 原子累加** 的并行求解方案。**不再使用距离约束**——物体的刚度完全来自 cluster 级的形状匹配。

### 与旧方案 (距离约束 + Jacobi atomic) 的区别

| 项 | 旧方案 | 新方案 |
|----|--------|--------|
| 约束类型 | 结构性距离约束 (a, b, rest, compliance) | Shape Matching cluster (粒子索引列表 + rest COM + rest 偏移) |
| 求解核 | 每线程一约束：算 (goal - p) 修正量 | 每线程一 cluster：算质心、协方差、极分解、goal position |
| 并行安全 | 原子累加修正量到 corrections/counts | 同左（atomicAdd） |
| 旋转恢复 | 无（纯距离） | 极分解 8 次牛顿迭代求 R |
| 视觉效果 | 软体保持体积但易剪切变形 | 软体保持局部形状，旋转刚性更强 |
| 收敛 | Jacobi 平均，需 8+ 迭代 | Jacobi 平均，建议 12+ 迭代 |
| 代码复杂度 | 低 | 中（极分解 + 3x3 求逆） |

### 核心循环（每帧）

```
1. Predict   — v += g·dt;  p_pred = p + v·dt              (1 pass)
2. for iter in solverIterations:
     Solve    — 每 cluster 算 goal position，atomicAdd 修正     (1 pass)
     Apply    — 把修正写回 predicted，清零 accumulators          (1 pass)
3. Integrate — v = (p_pred - p) / dt，damping + 地板反弹        (1 pass，仅一次！)
```

**关键**：Integrate 只在所有 solve 迭代结束后跑一次。Predict 把重力 baked 进
`predicted`，Solve+Apply 只修改 `predicted`（不碰 `positions`/`velocities`），
最后 Integrate 从总位移 `(predicted - p₀) / dt` 推出速度——重力贡献完整保留。

> 早期版本 Integrate 每轮迭代都跑，导致迭代 2+ 时 `newVel = (corr_k - corr_{k-1})/dt`
> （重力项被吞），cube 下落速度与重力大小几乎无关。已修复。

参考算法说明：`wgsl_shape_matching_pbd_atomic.md`

## 2. 文件结构

```
public/plugins/pbd/
  index.ts                    插件入口
  PbdManager.ts               GPU 资源管理 (cluster 生成 + buffer 分配 + dispatch)
  hooks/pbd.ts                simulate / draw / floor hooks
  components.json             PbdSoftBodyComponent schema (含 clusterRadius)
  uniform-layouts.json        pbdParams UBO (48 bytes, 12 fields)
  bind-layouts.json           pbdPredict(4) / pbdSolve(7) / pbdApply(4) / pbdIntegrate(4) / pbdDraw(2)
  pipelines/                  4 compute + 2 render
  shaders/
    PbdPredict.wgsl           Step 1: 预测位置
    PbdSolve.wgsl             Step 2: Shape Matching + 原子累加
    PbdApply.wgsl             Step 3a: 应用修正到 predicted（每轮迭代）
    PbdIntegrate.wgsl         Step 3b: 速度积分（每帧一次）
    PbdDraw.wgsl              表面网格渲染
    PbdFloor.wgsl             地板棋盘格
```

## 3. Cluster 生成

每个粒子拥有一个 cluster，包含 **其 Chebyshev 半径 `clusterRadius` 内的所有邻居**（含自己）。clusters 互相重叠——一个粒子通常属于多个 cluster，这是 Shape Matching 工作正确的前提（每个 cluster 独立求 goal position，再通过原子累加 + 计数平均合并到粒子）。

**clusterRadius = 2 (默认, gridN=5)** → 每 cluster 含最多 5×5×5 = 125 粒子，125 个 cluster，总 cluster 条目 = 539。

每个 cluster 预计算：
- `restCom`：rest 状态下的质心（cluster 内粒子 rest 位置的平均）
- `count`：成员粒子数
- `offset`：在 `clusterIndices` / `restOffsets` 平坦数组中的起始索引

每个 cluster 条目预计算：
- `clusterIndices[offset + i]`：成员粒子在 `positions` 中的全局索引
- `restOffsets[offset + i]`：该粒子 rest 位置减去 cluster 的 rest COM（即 `q_i`）

## 4. GPU Buffer 布局

### 粒子数据 (vec4f × particleCount)

| Buffer       | .xyz        | .w         | Usage                           |
|--------------|-------------|------------|---------------------------------|
| `positions`  | 当前位置    | invMass    | STORAGE \| VERTEX \| COPY_DST   |
| `predicted`  | 预测位置    | invMass    | STORAGE \| COPY_DST            |
| `velocities` | 速度        | pinned标志 | STORAGE \| COPY_DST             |

### Cluster 数据

| Buffer           | 类型                | 说明                                         |
|------------------|---------------------|----------------------------------------------|
| `clusters`       | array<Cluster, 32B> | struct { restCom: vec3f, count: u32, offset: u32 } (32 字节含 padding) |
| `clusterIndices` | array<u32>          | cluster 成员的全局粒子索引                   |
| `restOffsets`    | array<vec3f>        | 每个 cluster 条目的 rest 偏移（相对 cluster rest COM），stride=16 |

### Jacobi 修正缓冲

| Buffer        | 类型                  | 说明                                  |
|---------------|-----------------------|---------------------------------------|
| `corrections` | array<AtomicCorr>    | 累加的修正量 (.xyz = i32 fixed-point, ._pad 对齐) |
| `counts`      | array<atomic<u32>>    | 每粒子被多少个 cluster 修正过         |

- Solve pass 写入 (atomicAdd)，Integrate pass 读取后清零
- corrections.xyz 存储 `i32(round(correction * atomFactor))`，integrate 时 `f32(load) / count / atomFactor` 还原

### 表面索引 (u32 × surfaceVertexCount)

6 面 × (N-1)² × 6 顶点。

### UBO: pbdParams (48 bytes)

| 偏移 | 字段              | 类型 | 说明                          |
|------|-------------------|------|-------------------------------|
| 0    | dt                | f32  | 帧时间步长 (钳制到 1/30)      |
| 4    | time              | f32  | 累计时间                      |
| 8    | gravityY          | f32  | Y 重力加速度                  |
| 12   | damping           | f32  | 每帧速度衰减                  |
| 16   | solverIterations  | u32  | solve+integrate 循环轮数      |
| 20   | particleCount     | u32  | 粒子总数                      |
| 24   | restitution      | f32  | 地板弹性系数                  |
| 28   | clusterCount      | u32  | cluster 总数（= particleCount）|
| 32   | atomFactor        | f32  | f32→i32 fixed-point 缩放 (1024) |
| 36   | stiffness         | f32  | 全局刚度 (1-compliance)       |
| 40   | _pad0             | f32  | 对齐                          |
| 44   | _pad1             | f32  | 对齐                          |

## 5. Bind Group 布局

### pbdPredict (group 0)

| binding | 类型           | 资源         |
|---------|----------------|--------------|
| 0       | storage (rw)   | positions    |
| 1       | storage (rw)   | predicted    |
| 2       | storage (rw)   | velocities   |
| 3       | uniform        | pbdParams    |

### pbdSolve (group 0) — 7 bindings

| binding | 类型               | 资源           |
|---------|--------------------|----------------|
| 0       | storage (rw)       | predicted      |
| 1       | read-only-storage  | clusters       |
| 2       | read-only-storage  | clusterIndices |
| 3       | read-only-storage  | restOffsets    |
| 4       | storage (rw)       | corrections    |
| 5       | storage (rw)       | counts         |
| 6       | uniform            | pbdParams      |

### pbdApply (group 0)

| binding | 类型           | 资源         |
|---------|----------------|--------------|
| 0       | storage (rw)   | predicted    |
| 1       | storage (rw)   | corrections  |
| 2       | storage (rw)   | counts       |
| 3       | uniform        | pbdParams    |

### pbdIntegrate (group 0)

| binding | 类型           | 资源         |
|---------|----------------|--------------|
| 0       | storage (rw)   | positions    |
| 1       | storage (rw)   | predicted    |
| 2       | storage (rw)   | velocities   |
| 3       | uniform        | pbdParams    |

### pbdDraw (group 1, group 0 = frame)

| binding | 类型               | 资源           |
|---------|--------------------|----------------|
| 0       | read-only-storage  | positions      |
| 1       | read-only-storage  | surfaceIndices |

## 6. 算法详解

### 6.1 Step 1 — Predict (`PbdPredict.wgsl`)

半隐式 Euler 预测：
```
accel = (0, gravityY, 0)
newVel = v + accel * dt
newPos = p + newVel * dt
predicted[i] = (newPos, invMass)
```
pinned 粒子 (v.w > 0.5): `predicted[i] = positions[i]`

### 6.2 Step 2 — Solve (`PbdSolve.wgsl`) — Shape Matching + Atomic

**核心思想**: 每 cluster 独立算出每个粒子应有的 goal position（通过极分解恢复 cluster 的 rest 形状并旋转到当前朝向），把 `(goal - p) * stiffness` 通过 `atomicAdd` 累加到粒子的 corrections/counts。不修改 `predicted`，无竞态。

**每线程一 cluster** 的处理流程：

1. **当前质心** `com = (1/N) * sum(predicted[pid])`
2. **协方差矩阵** `A = sum( (p_i - com) * q_i^T )`，其中 `q_i = restOffsets[i]`（已相对 rest COM）
3. **极分解** `A = R * S`，迭代 8 次：`R = 0.5 * (R + transpose(inverse(R)))`（先做 `R = A + 1e-6*I` 正则化防奇异）
4. **Goal position** `g_i = com + R * q_i`
5. **修正量** `corr_i = (g_i - p_i) * stiffness`
6. **原子累加**：
   - `atomicAdd(corrections[pid].x, i32(round(corr.x * atomFactor)))`
   - 同理 .y / .z
   - `atomicAdd(counts[pid], 1)`

### 6.3 Step 3a — Apply (`PbdApply.wgsl`)

每轮 solve 之后跑一次。读 corrections/counts，算 Jacobi 平均，把修正写回
`predicted`（不碰 `positions`/`velocities`）。清零 corrections/counts 为下一轮 solve
准备。地板夹紧（`pred.y >= 0`）在此 pass 做，防止 solve 阶段穿透地板。

### 6.4 Step 3b — Integrate (`PbdIntegrate.wgsl`)

**每帧仅跑一次**（所有 solve 迭代之后）。从总位移推速度：

```wgsl
newVel = (predicted[i] - positions[i]) / dt   // positions = 帧初位置
// 地板反弹
if pred.y < 0.001 && newVel.y < -0.3:
    newVel.y = -newVel.y * restitution
    newVel.xz *= 0.8
newVel *= damping
positions[i] = predicted[i]
velocities[i] = (newVel, pinned_flag)
```

- `positions[i]` 在 solve 循环期间不被修改 → 速度推算基于完整帧位移
- pinned 粒子接受 predicted 但不更新速度
- damping 只应用一次（不是 solverIterations 次）

### 6.5 每帧 pass 数量

```
1 (predict) + iterations × 2 (solve + apply) + 1 (integrate) = 2 + 2N
```
N=12 时: 26 passes。

## 7. f32 原子加法：Fixed-Point 方案

WebGPU 只支持 `atomic<u32>` 和 `atomic<i32>` 的 `atomicAdd`，不支持 `atomic<f32>`。

**Fixed-Point 编码** (atomFactor = 1024)：
1. 修正量乘以 atomFactor：`scaled = correction * 1024`
2. 四舍五入为 i32：`encoded = i32(round(scaled))`
3. `atomicAdd(&buffer[i], encoded)` 原子累加
4. 读取时：`f32(atomicLoad(&buffer[i])) / count / atomFactor` 还原

精度：`atomFactor = 1024` → 最小可分辨修正量 = `1 / 1024 ≈ 0.001` 米。对 cellSize=0.4 的网格精度足够。

> **重要**：早期版本的代码使用 `bitcast<i32>(correction * factor)` 而非 `i32(round(...))`。这是**错误**的——bitcast 重解释 IEEE 754 位模式，多个 atomicAdd 累加的是位模式而非数值，无法正确求和。Fixed-Point 是数值层面的整数加法，数学上正确。

## 8. 可调参数

| 参数              | 默认  | 作用                           |
|-------------------|-------|--------------------------------|
| gridN             | 5     | 网格分辨率 (gridN³ 粒子)       |
| cellSize          | 0.4   | 粒子间距                       |
| gravity           | -9.81 | Y 重力                         |
| damping           | 0.995 | 每帧速度衰减                   |
| solverIterations  | 12    | solve+integrate 循环轮数       |
| compliance        | 0.0   | 柔度 (0=刚性, >0=软)           |
| restitution       | 0.35  | 地板弹性                       |
| mass              | 1.0   | 总质量                          |
| clusterRadius     | 2     | Chebyshev 邻域半径 (cluster 大小 = (2r+1)³ 上限) |

`atomFactor` 和 `stiffness` 在 UBO 中，不由 PbdManager 写入，不由 scene.json 直接设置。`stiffness = 1 - clamp(compliance, 0, 0.95)`。

### 调参建议

- **Shape Matching Jacobi 收敛比距离约束慢** → 建议 `solverIterations=12-16`
- `clusterRadius` 增大 → 每 cluster 覆盖更广，刚度感更强，但 GPU 线程循环更长（性能线性下降）。gridN=5 时半径 2 → 每 cluster 最多 125 粒子；半径 1 → 最多 27 粒子。
- `compliance` 在 Shape Matching 下效果更明显（更软）→ 可适当降低或保持 0
- 若需更刚性，提 `solverIterations` 而非简单降 `compliance`

## 9. 已踩的坑

1. **WGSL swizzle 复合赋值**: `pa.xyz -= corr` 非法 → 用 `pa = vec4f(pa.xyz - corr, pa.w)` 重建

2. **`mappedAtCreation` 初始化**: 所有 buffer 必须用 `mappedAtCreation: true` 初始化，否则 `queue.writeBuffer` 可能与同帧 compute encoder 竞态

3. **`COPY_SRC` usage**: 如果要做 GPU readback (`copyBufferToBuffer`)，positions buffer 必须包含 `COPY_SRC`，否则整个 encoder invalid

4. **dt 钳制**: `dt` 钳制到 `[0, 1/30]`，防止 tab 切换/GC 停顿后首帧爆炸

5. **corrections 清零**: Integrate pass 必须在每轮结束时清零 corrections/counts，否则下一轮 solve 会累加到旧值上

6. **fixed-point 不是 bitcast**: 早期版本 `bitcast<i32>(value * factor)` 把 IEEE 754 位模式重解释为 i32，多 contributor 累加时数学错误。`i32(round(value * factor))` 才是真正的 fixed-point。

7. **Cluster struct 布局**: `struct Cluster { restCom: vec3f, count: u32, offset: u32 }` 因 vec3f align(16)，struct 实际占 32 字节（4 字节 padding 在尾部）。CPU 写入时按 32 字节 stride，每 cluster 8 个 u32/f32 槽位（restCom.xyz @ 0-2, count @ 3, offset @ 4, padding @ 5-7）。

8. **array<vec3f> stride**: WGSL 中 `array<vec3f>` 元素 stride 是 16（不是 12），因为 vec3f 在数组中会被对齐到 16 字节。CPU 写入 restOffsets 时每条目占 4 个 f32 槽位（.xyz 用前 3 个，第 4 个空）。

9. **Integrate 不能在 solve 循环内跑**: 早期版本每轮 solve 后都跑 Integrate，导致 `newVel = (predicted + corr_k - (predicted + corr_{k-1})) / dt = (corr_k - corr_{k-1}) / dt` —— 重力项被吞掉，cube 下落速度与 |g| 几乎无关（g 从 -10 到 -300 几乎没变化）。正确做法是把"应用修正"和"速度积分"拆成两个 shader：Apply 每轮跑（只改 predicted），Integrate 每帧跑一次（从总位移推速度）。

10. **damping 不能在循环内累积**: Integrate 每轮跑时 damping 也每轮乘，effective per-frame damping = `damping^solverIterations`（12 轮 0.995^12 ≈ 0.94），速度损失被放大。拆成 Apply/Integrate 后 damping 每帧只乘一次。
