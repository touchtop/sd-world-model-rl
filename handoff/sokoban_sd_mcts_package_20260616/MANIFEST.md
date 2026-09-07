# 文件清单

## 代码

- `src/sokoban/env.py`：Boxoban/XSokoban level 解析、状态表示和基础状态转移。
- `src/sokoban/sd_model.py`：Sokoban stock-flow SD 模型。
- `src/sokoban/gft_sd.py`：GFT stock-flow 表示、运输代价、软 deadlock 与结构化先验。
- `src/sokoban/search.py`：基础 best-first search 工具。
- `experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py`：当前最佳 GFT-SD-MCTS 评估入口。
- `experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py`：exact-grid SD-MCTS 基线与 TVA checkpoint 加载工具。
- `experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py`：当前最佳 GFT online TVA 训练入口。
- `experiments/sokoban_sd_world_model/*.py`：训练、审计、消融和诊断脚本。

## 数据

- `datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt`：最佳 1000 关评估数据。
- `datasets/sokoban_boxoban/boxoban-levels/medium/train/`：TVA 和相关审计所需训练关卡来源。
- `datasets/sokoban_boxoban/boxoban-levels/`：Boxoban 其余 level 文件及许可证。
- `datasets/sokoban_xsokoban/xsokoban_first40_boxoban_format.txt`：XSokoban first-40 的 Boxoban 格式转换版本。

## 最佳结果与关键产物

- `results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json`：Boxoban 1000 关最佳汇总结果。
- `results/sokoban_sd_world_model/test1000_shards/`：20 个 50 关分片的 JSON 结果和运行日志。
- `results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json`：最佳 TVA checkpoint。
- `results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json`：对应训练日志。
- `results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json`：grid SD 独立 gate 结果。
- `results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json`：TVA 独立审计结果。
- `results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json`：含完整 gate 路径记录的 holdout50 验证结果。
- `results/sokoban_sd_world_model/xsokoban_first40_bestconfig_exact_smoke1_seed0.json`：XSokoban first-40 第一关、`max_steps=80` 探索结果。
- `results/sokoban_sd_world_model/xsokoban_first40_bestconfig_exact_l001_maxsteps1000_seed0.json`：XSokoban first-40 第一关、`max_steps=1000` 探索结果。
- `results/sokoban_sd_world_model/repro_boxoban_best_smoke1_seed0.json`：打包后从包内复跑生成的单关 smoke 结果。

## 文档

- `docs/sokoban_sd_world_model.md`：Sokoban SD world model 主文档。
- `docs/methods.md`：方法、参数和文献对照口径。
- `docs/decisions.md`：关键决策记录。
- `docs/research_log.md`：实验和文档更新日志。
- `docs/reproducibility.md`：环境、数据、随机种子和复现依据。
- `docs/neurips/paper/full_draft_zh.tex`、`full_draft_zh.pdf`、`full_draft_zh.md`：论文草稿。
- `docs/sokoban_sd_literature/`：Sokoban 相关文献 PDF 与文本摘录。
- `README.md`、`RUNBOOK.md`、`PACKAGE_STATUS.md`：团队交接入口、复现说明和打包验证记录。
- `CHECKSUMS.sha256`：包内文件校验和。

## 未纳入内容

原仓库 `results/sokoban_sd_world_model/` 中有大量历史消融结果，本包仅纳入最佳结果、关键 gate/audit、checkpoint、XSokoban 探索和 1000 题分片日志。若后续需要追溯某个早期失败分支，可回到原仓库历史结果目录查找。
