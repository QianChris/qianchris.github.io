# Splendor — 引擎上的第一个完整游戏

一个可玩的两人 Splendor（宝石商人），跑在 ShaderLab-WebGPU 的宿主层上：
**规则引擎 + AI 是插件，棋盘是 app 级 DOM UI，引擎只负责帧循环、插件装载、事件总线与生命周期。**

![Splendor on ShaderLab-WebGPU](../../../docs/images/splendor-board.jpg)

```
运行（开发）：  npm run dev
                → https://localhost:5173/player.html?app=splendor
运行（构建后）：npm run build && npm run preview
                → https://localhost:4173/player.html?app=splendor
编辑器内打开：  https://localhost:5173/?app=splendor
```

## 玩法

- 默认 **人机对战**（玩家 1 对你，玩家 2 是规则 AI）；点「双人热座」可两人轮流。
- 拿宝石：点银行里的宝石（最多 3 种不同色；同色 2 枚需要该色库存 ≥ 4）→ 托盘「确认拿取」。
- 买卡：点市场卡 / 自己保留区的卡 → 托盘「购买」（买得起时卡片有绿色描边）。
- 保留：点市场卡 → 「保留（+1 金币）」，或点牌堆封面暗抽一张（同样 +1 金币，上限 3 张）。
- 手牌上限 10 枚（含金币）：超过时会进入弃牌模式，点自己面板里的宝石弃回，再确认。
- Esc 取消当前选择，Enter 确认。

## 结构

```
public/plugins/splendor/        ← 能力（与 demo 解耦，可整目录拷走）
  rules.ts                      纯函数规则引擎：createGame / legalTakes / legalActions /
                                applyAction / paymentFor / autoDiscard / scores
  cards.ts                      90 张卡 + 10 位贵族（由脚本从参考实现生成）
  ai.ts                         规则 AI：枚举合法动作 → 模拟 → 静态评估取最优
  SplendorSystem.ts             'splendor' 系统：持有唯一 GameState、收发事件、调度 AI 回合
  index.ts                      插件入口（systemDefs.before=['render']，app 无需自带 systems.json）
public/apps/splendor/           ← 组合（纯内容）
  app.json / scene.json         scene 故意为空：棋盘是 DOM，引擎侧没有 3D 内容
  render.json                   phases 为空（canvas 只清屏，界面在 #ui-container 里）
  ui-config.json → ui/board.js  整个棋盘（DOM），只渲染插件发来的 view、回报语义动作
  assets/                       90 张卡面 + 10 贵族 + 6 枚代币 + 桌面（WebP/PNG，3.2 MB）
tests/unit/splendorRules.test.ts 24 个测试：规则/边界/终局/死局裁决/随机自对弈不变量/AI 自对弈
```

数据流（单向）：

```
UI  --eventBus.emit('splendor:action', action)-->  SplendorSystem
    <--eventBus.emit('splendor:state',  view)----  (state + 可购性 + 拿取规格)
```

UI 不做任何规则判断（连"这张卡买不买得起"都来自 view），所以规则不可能与界面分叉；
`ui/board.js` 是无 import 的裸 ES 模块（app UI 不走打包器），改动无需重新构建。

## 规则实现说明（与参考实现的差异）

| 项 | 本实现 | Ref/Splendor-AI |
|---|---|---|
| 终局 | 到 15 分**打玩本轮**，再比分数；同分比「卡数更少」 | 一到 15 分立即结束（先手占优） |
| 拿取 | 库存够 3 色必须拿 3 色；同色 2 枚需库存 ≥ 4 | 允许随时拿 1 / 拿 2 色（更宽松） |
| 贵族 | 购买后自动到访，可一次来多张 | 相同 |
| 金币 | 只能靠保留获得，计入 10 枚上限 | 相同 |
| 死局 | 银行枯竭时双方连续跳过 / 连续 40 手无人购牌 → 按分数结算 | 无（会无限循环） |
| 对手 | 规则 AI（贪心估值） | DQN（**仓库里没有权重文件**） |

## 已知限制

- 只支持 2 人（与参考实现一致）；3–4 人需要改 `createGame` 的银行/贵族数量与座位布局。
- 没有音效、没有发牌动画（CSS 过渡留给后续）。
- 没有存档：`SplendorSystem` 的 `GameState` 已设计成纯 JSON（`structuredClone` 安全），
  接 `host.projectFS` 即可做存读档。
- 棋盘是 DOM，不是 GPU 渲染。要让卡牌进 WebGPU，需要 `docs/splendor-gap-analysis.md` 里路线 B 的
  `board2d` 插件（屏幕空间 quad + 自算 NDC + 自排序），引擎当前无正交相机、无 2D 文本。
