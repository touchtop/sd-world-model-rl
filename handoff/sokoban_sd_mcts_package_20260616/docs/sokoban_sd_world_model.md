# Sokoban SD World Model Design Note

## 当前定位

本研究线用于把“面向决策的系统动力学世界模型”扩展到 Sokoban/Boxoban。当前主线不再是强化 [src/sokoban/search.py](/home/data/round/code/sd_world_model_rl/src/sokoban/search.py) 中的 best-first 求解器，而是设计一个由存量、流量和反馈构成的可学习 SD 世界模型，并将其接入 `MBRL + SD-MCTS`。

完整设计以 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md) 为准。该设计把原始网格状态编码为六维核心存量状态 `x_t=[G_t,U_t,M_t,R_t,C_t,I_t]`，用非负流量、已知 incidence matrix 和反馈函数预测 `x_{t+1}`，并由 SD-MCTS 在学习到的抽象动力学上规划。

## 核心口径

Sokoban/Boxoban 在本项目中不是为了证明启发式搜索可以解题，而是用于研究抽象高层空间状态刻画下的 SD 世界模型设计方法。因此：

1. `SD` 的核心元素是存量、流量和反馈；
2. 初始模型应是结构已知、部分参数未知；
3. 首版核心存量为目标完成、未解决压力、机动性、可达容量、拥塞和不可逆风险；
4. goal-room pressure、plan progress 和 structural slack 暂作为派生特征、辅助头或第二阶段扩展；
5. Junghanns、Pereira、Lipovetzky、Guez、Orseau、Martin、Taufeeque 等文献中的机制用于提供结构先验、观测代理、风险标签或诊断项；
6. matching lower bound、dead-square、macro、relevance 和 path-channel 等机制不应被写成最终启发函数；
7. 规划基准是 SD-MCTS over learned abstract dynamics，而不是 best-first 或 IDA* 式启发式搜索。

## 文献依据

文献证据矩阵见 [sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md)。当前可直接支撑设计的结论包括：

1. Sokoban 的动作后果具有不可逆性，错误推箱可使状态进入死锁，因此 SD 状态需要包含不可逆风险、可恢复机动性和玩家可达容量。
2. 玩家可达域、push-level 动作、箱子-目标匹配、deadlock table、宏动作、goal/tunnel 结构和 relevance cuts 等文献机制适合作为结构门控、流量输入、风险标签或第二阶段扩展。
3. pattern database 与实例依赖抽象文献支持把多箱交互、目标分配自由度和局部 pattern 风险纳入 `IrreversibilityRiskStock`、`UnresolvedBoxStock` 和派生 slack 诊断。
4. traps、invariants 和 dead-ends 文献支持把不可恢复性作为风险存量与风险流量，而不是只作为硬剪枝。
5. Boxoban、DRC、policy tree search 和抽象 imagination planning 文献支持把 Sokoban 作为 MBRL、policy prior、额外 thinking time 和高层 MCTS 的研究场景。

Botea、Muller 与 Schaeffer 2002 年论文的本地 PDF 抽取文本存在严重编码异常。因此该文献当前只作为题录和后续引用链背景，不把乱码正文作为具体设计证据。

## 目标接口

目标 SD 世界模型首版应输出：

1. `stock_state`：六个核心存量 `goal_completion`、`unresolved_box_pressure`、`mobility`、`reachable_capacity`、`congestion`、`irreversibility_risk`；
2. `flow_state`：非负流量、incidence matrix 后的净变化和边界截断信息；
3. `feedback_state`：目标压力、拥塞-机动性-风险、不确定性三类反馈；
4. `predicted_next_stock`：下一抽象状态；
5. `reward_or_progress`：任务进展、奖励或成功概率；
6. `risk_and_uncertainty`：动作后果风险 `risk_t`、当前状态风险存量 `I_t`、模型不确定性及其校准信息；
7. `feasibility_state`：当前 raw-state handle 下 single-push option 的可执行性；
8. `diagnostic_features`：可达域、合法推箱、匹配残差、deadlock evidence、goal-room proxy、macro proxy、relevance proxy、path-channel probe 和 structural slack proxy。

其中 `diagnostic_features` 不构成 SD 的核心状态转移，只用于解释、训练辅助和审计。

## 当前诊断草稿

已存在代码路径如下：

1. [src/sokoban/env.py](/home/data/round/code/sd_world_model_rl/src/sokoban/env.py)
2. [src/sokoban/sd_model.py](/home/data/round/code/sd_world_model_rl/src/sokoban/sd_model.py)
3. [src/sokoban/search.py](/home/data/round/code/sd_world_model_rl/src/sokoban/search.py)
4. [evaluate_sokoban_sd_model.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py)

首轮评测命令为：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 50 \
  --max-expansions 2000 \
  --out results/sokoban_sd_world_model/eval_medium_train000_50.json
```

结果摘要为：transition adapter consistency `266 / 266`；无启发式 push-level 均匀代价搜索 `2 / 50` solved，平均扩展 `1938.18`；SD-guided best-first search `45 / 50` solved，平均扩展 `422.16`。

该结果只说明第一版结构代理足以产生有区分度的诊断信号。它没有学习参数化流量，没有反馈回路，也没有 SD-MCTS，因此不应作为 SD 世界模型或完整 Boxoban benchmark 结论。

## 替换路线

后续推进顺序为：

1. 以 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md) 固定六核心存量、incidence matrix 和三类反馈设计；
2. 将当前可达域、合法推箱、匹配下界和死锁证据改为 encoder 输入、结构门控、风险标签和诊断项；
3. 实现首版可学习流量模型，至少输出核心存量、非负流量、反馈、不确定性、可执行性和抽象下一状态；
4. 实现 SD-MCTS，使搜索节点保存抽象状态分布、uncertainty、raw-state handle 或 belief support，并在展开时调用 learned SD transition provider；
5. 用 `no-learned-parameter`、`no-feedback`、`black-box latent MBRL`、`heuristic-search diagnostic` 和 `learned-transition-provider audit` 消融检查是否退化；
6. 在固定 Boxoban split 上报告世界模型指标、规划指标、aliasing 诊断和失败类型，而不只报告 solved rate。
