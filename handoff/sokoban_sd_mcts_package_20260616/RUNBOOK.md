# 复现说明

## 运行约定

所有命令均从本包根目录运行：

```bash
cd sokoban_sd_mcts_package_20260616
export PYTHONPATH="$PWD"
```

评估脚本会把包根目录和脚本目录加入 `sys.path`，因此通常不需要安装为 Python package。

## Boxoban 单关 smoke

```bash
bash scripts/run_boxoban_best_smoke1.sh
```

输出写入：

`results/sokoban_sd_world_model/repro_boxoban_best_smoke1_seed0.json`

该命令用于验证最小闭环：level 解析、GFT-SD-MCTS、TVA checkpoint、独立 gate 文件和结果落盘。

## Boxoban 1000 关复跑

完整测试建议按 20 个分片运行，每片 50 关。分片命令为：

```bash
bash scripts/run_boxoban_best_shard50.sh 0
bash scripts/run_boxoban_best_shard50.sh 50
bash scripts/run_boxoban_best_shard50.sh 100
```

依此递增到 `950`。脚本内部使用 `seed = level_offset * 1009`，与历史 20 个分片结果一致。输出文件形如：

`results/sokoban_sd_world_model/repro_shards/gft_sdmcts_boxoban_unfiltered_test1000_shard000_049_repro_seed0.json`

## XSokoban 第一关外推测试

```bash
bash scripts/run_xsokoban_l001_maxsteps1000.sh
```

输出写入：

`results/sokoban_sd_world_model/repro_xsokoban_l001_maxsteps1000_seed0.json`

历史记录为未解出，`status_counts={"max_steps": 1}`。该测试不代表 XSokoban first-40 的总体解题率，仅用于复查更难关卡上的单关边界。

## 独立 gate 文件

最佳配置运行时要求两个独立 gate：

- `--sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json`
- `--tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json`

这两个文件已包含在包内。若重新训练 TVA 或修改 SD provider，应重新生成 gate/audit 文件后再使用 `--require-independent-gates`。

