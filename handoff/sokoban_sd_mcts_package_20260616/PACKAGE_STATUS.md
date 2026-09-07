# 包状态记录

生成日期：2026-06-16

## 打包范围

本包为 Sokoban SD-MCTS 研究成果交接包，包含最佳版本代码、Boxoban/XSokoban 数据、最佳结果日志、关键 checkpoint、独立 gate/audit 文件、论文草稿和研究文档。包内路径按照原实验相对路径保留，可独立于原仓库运行。

## 已验证事项

- `python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py --help` 可正常执行。
- `datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt` 可解析出 1000 关。
- `datasets/sokoban_xsokoban/xsokoban_first40_boxoban_format.txt` 可解析出 40 关。
- 最佳汇总结果中 `solved_count=989`，`grid_sd_gate=PASS`，`tva_independent_gate=PASS`。
- `bash scripts/run_boxoban_best_smoke1.sh` 已在包内复跑通过，第 1 关解出，输出为 `results/sokoban_sd_world_model/repro_boxoban_best_smoke1_seed0.json`。

## 最佳结果摘要

- 数据集：Boxoban `unfiltered/test/000.txt`，1000 关。
- 成功率：989/1000，98.9%。
- 失败类型：`deadlocked=4`，`max_steps=7`。
- 已解平均 push 数：12.3984。
- 训练和在线测试 teacher 调用量：0；teacher 仅用于独立审计。

## 注意事项

完整 1000 关复跑计算量较大，建议按 `RUNBOOK.md` 的 50 关分片方式运行。XSokoban 第一关 `max_steps=1000` 的既有记录为未解出，该测试用于标记当前最佳方案在更难 OOD 关卡上的边界。

