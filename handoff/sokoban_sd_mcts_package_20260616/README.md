# Sokoban SD-MCTS 交接包

本目录是 Sokoban 相关成果的独立交接包，保留了最佳版本代码、最佳结果日志、关键 checkpoint、Boxoban/XSokoban 数据和研究文档。包内相对路径与原实验保持一致，可从本目录直接运行评估脚本。

## 最佳结果

最佳公开测试结果位于：

`results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json`

该结果在 Boxoban `unfiltered/test/000.txt` 的 1000 关上评估，解出 989 关，成功率为 98.9%。状态统计为 `solved=989`、`deadlocked=4`、`max_steps=7`，已解关卡平均 push 数为 12.3984。独立 gate 状态为 `grid_sd_gate=PASS`、`tva_independent_gate=PASS`，线上测试不调用 teacher。

对应的 TVA checkpoint 为：

`results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json`

## 方法口径

本方法属于基于先验知识的结构化显式建模。核心是 exact-grid / GFT stock-flow 状态表示、结构化 deadlock 与运输约束、SD-MCTS 搜索，以及小规模 TVA value 校正。TVA 训练使用 40 关、1378 个 raw 状态、834 个去重状态、114 个 value 训练状态；独立审计使用 120 个叶状态、78 个高置信 teacher 标签。teacher 不参与训练目标生成和在线测试，只用于独立审计。

## 目录说明

- `src/sokoban/`：Sokoban 环境、stock-flow 模型、GFT-SD 模型和基础搜索代码。
- `experiments/sokoban_sd_world_model/`：最佳评估入口、训练脚本、审计脚本和诊断脚本。
- `datasets/sokoban_boxoban/`：Boxoban levels，已排除上游 `.git` 元数据。
- `datasets/sokoban_xsokoban/`：XSokoban first-40 转换为 Boxoban 格式后的探索数据。
- `results/sokoban_sd_world_model/`：最佳 1000 题结果、20 个分片日志、TVA checkpoint、独立 gate/audit 结果和 XSokoban 探索结果。
- `docs/`：方法、决策、复现记录、论文草稿和 Sokoban 文献材料。
- `scripts/`：复现实验封装脚本。

## 环境

建议使用 Python 3.10 或更新版本。最佳评估路径至少需要 `numpy`；训练和部分审计脚本需要 `torch`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_sokoban.txt
```

## 快速验证

从本目录运行：

```bash
bash scripts/run_boxoban_best_smoke1.sh
```

该脚本只复跑 Boxoban 测试集第 1 关，用于确认代码、数据、checkpoint 和 gate 文件路径均可用。完整 1000 关复跑计算量较大，建议按 50 关分片运行：

```bash
bash scripts/run_boxoban_best_shard50.sh 0
bash scripts/run_boxoban_best_shard50.sh 50
```

其中参数为 `level_offset`。历史 20 个分片使用的 seed 规则为 `seed = level_offset * 1009`。

XSokoban 外推测试可运行：

```bash
bash scripts/run_xsokoban_l001_maxsteps1000.sh
```

已有记录显示第一关在 `max_steps=1000` 下未解出，该结果用于刻画当前方案在更难 OOD 关卡上的边界。

