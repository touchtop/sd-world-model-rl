# legacy_v3b_restore

本目录用于在不改动当前主线调用链的前提下，尽可能恢复 `108c10b` 锁定的 Hopper `v3b` 训练路径。

## 目标

- 为 `v3b` Hopper SD 训练链保留一份独立 overlay；
- 优先加载本目录中的 `v3b` 相关文件；
- 对于 `external/tdmpc2` 中无法从历史 submodule 精确取回的部分，回落到当前主线实现。

## 目录说明

- `src/world_model/dmcontrol_hopper_sd_model.py`
  `108c10b` 可直接读取，已按锚点恢复。
- `scripts/run_hopper_mcts.py`
  `108c10b` 可直接读取，已按锚点恢复。
- `external/tdmpc2/tdmpc2/common/sd_world_model.py`
- `external/tdmpc2/tdmpc2/common/parser.py`
- `external/tdmpc2/tdmpc2/tdmpc2.py`
- `external/tdmpc2/tdmpc2/train.py`
- `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
- `external/tdmpc2/tdmpc2/config.yaml`
  以上文件用于构造 `v3b` 训练 overlay。由于 `108c10b` 时 `external/tdmpc2` 仍是 submodule，无法在当前仓库中直接取回其完整历史内容，因此这里采用当前主线中最接近 `v3b` 的版本作为 best-effort 恢复源。

## 使用方式

建议从仓库根目录执行，并先设置 overlay 优先级：

```bash
export PYTHONPATH=$PWD/legacy_v3b_restore/external/tdmpc2/tdmpc2:$PWD/legacy_v3b_restore:$PWD/external/tdmpc2/tdmpc2:$PWD:$PYTHONPATH
export MUJOCO_GL=glfw
export CUDA_VISIBLE_DEVICES=0
```

训练入口示例：

```bash
python -u legacy_v3b_restore/external/tdmpc2/tdmpc2/train.py \
  task=hopper-hop world_model_kind=sd_hopper model_size=null obs=state \
  steps=5000 seed=3 exp_name=legacy_v3b_restore_smoke \
  enable_wandb=false save_video=false save_agent=true save_csv=true \
  eval_episodes=10 eval_freq=1000 compile=false
```

MCTS 入口示例：

```bash
python -u legacy_v3b_restore/scripts/run_hopper_mcts.py --seed 3 --budget 5000
```

## 说明

- 本目录是“恢复快照”，不是对当前主线的替换。
- 若需要进一步逼近 `108c10b` 的完整外部子模块状态，需额外取得 `external/tdmpc2@df1b9d7` 的实际文件内容。
