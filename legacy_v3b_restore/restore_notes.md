# v3b 恢复说明

## 恢复基点

- 主锚点提交：`108c10b`
- 提交信息：`Lock SD-hopper v3b baseline: routeB dynamics + call-chain fixes`
- 该提交中 `external/tdmpc2` 指向 submodule commit：`df1b9d7ef919c00a8f5139d805f4fb3edfe507c3`

## 可精确恢复部分

以下文件已确认与 `108c10b` 工作树内容一致，因此在本目录中可视为精确恢复：

- `src/world_model/dmcontrol_hopper_sd_model.py`
- `scripts/run_hopper_mcts.py`

## 仅能 best-effort 恢复部分

由于当前仓库无法直接读取 `df1b9d7` 对应 submodule 内容，以下文件只能采用当前主线中最接近 `v3b` 的版本：

- `external/tdmpc2/tdmpc2/common/sd_world_model.py`
- `external/tdmpc2/tdmpc2/common/parser.py`
- `external/tdmpc2/tdmpc2/tdmpc2.py`
- `external/tdmpc2/tdmpc2/train.py`
- `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
- `external/tdmpc2/tdmpc2/config.yaml`

选择这些文件的理由如下：

1. 当前主线仍保留 `v3` 风格的 `15` 维结构态与 `sd_latent_dim=32` 的训练链；
2. `108c10b` 提交说明明确指出当时 submodule 已包含“matching SDWorldModel wrapper、resume/buffer fixes、reward-scale-aware planning path”，与当前主线保留的相关实现方向一致；
3. 在无法获得 submodule 原文的前提下，保留当前可运行链比手工猜测更稳妥。

## overlay 机制

本目录采用 overlay 方式工作：

- `legacy_v3b_restore` 中存在的模块优先加载；
- 未在本目录覆盖的模块，继续从主线仓库加载；
- `common`、`trainer`、`src`、`src.world_model` 的 `__init__.py` 已通过 `extend_path` 打开跨目录搜索路径。

## 当前已知边界

- 这不是 `108c10b + df1b9d7` 的位级复刻；
- 这是围绕 `108c10b` 叙事与当前可运行主线构造的恢复工作目录；
- 后续若拿到历史 submodule 内容，应优先替换本目录中对应的 `external/tdmpc2` 文件。
