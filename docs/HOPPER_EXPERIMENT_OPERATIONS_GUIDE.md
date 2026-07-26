# Hopper 实验操作总指南

本文件用于固定当前仓库中 Hopper 实验的稳定操作口径，避免重复试错。

适用目录：

- 仓库根目录：`/home/data/round/code/sd_world_model_rl`

适用环境：

- Python 环境：`/home/anaconda3/envs/sdmcts_p100/bin/python`
- MuJoCo 渲染口径：`MUJOCO_GL=glfw`
- 训练默认走 `external/tdmpc2/tdmpc2/train.py`

## 一、总原则

1. Hopper 单回合长度固定为 `500`。
2. TD-MPC2 的种子采样阶段固定为：
   `seed_steps = max(1000, 5 * episode_length) = 2500`
3. 因此：
   `2500` 之前是随机采样，不训练，不应在此之前插入官方评估。
4. 训练启动时必须使用新的 `exp_name`，不要复用旧目录。
5. 训练命令必须显式保留：
   `save_full_state=true`
6. 当前配置中已经默认开启：
   `save_eval_traces=true`
7. 因此只要发生官方评估，就会自动在
   `official_eval/step_XXXXXXXX/`
   下保存轨迹 csv 和汇总。
8. 自 `2026-04-30` 接口修复后，启用
   `sd_online_supervision_enabled=true` 时，`SD transition core`
   只由 `sd_optim` 管理，不再同时进入 TD-MPC2 主优化器。
9. 当前正在运行的旧进程不会自动吸收代码修复；验证修复效果必须使用新的
   `exp_name` 启动新实验。

## 二、标准启动脚本

以下脚本已经固化了稳定参数，不要再手写长命令。

### 1. 5k smoke

v5 当前推荐脚本：

- [run_hopper_v5_5k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_5k_smoke.sh)
- [run_hopper_v5_5k_pureenv_sdinit_online_sup.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_5k_pureenv_sdinit_online_sup.sh)

用法：

```bash
bash scripts/run_hopper_v5_5k_smoke.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_5k_smoke.sh 0 sd_hopper_5k_v5densepose_recover010_smoke_20260426_run1 3
```

固定口径：

- `steps=5000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`

纯环境奖励 / supervised SD init / online SD supervision 脚本：

用法：

```bash
bash scripts/run_hopper_v5_5k_pureenv_sdinit_online_sup.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_5k_pureenv_sdinit_online_sup.sh \
  1 sd_hopper_5k_pureenv_sdinit_online_sup_seed3_20260429_run1 3
```

固定口径：

- `steps=5000`
- `eval_freq=2500`
- `checkpoint_freq=2500`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_use_transition_residual=true`
- `sd_transition_residual_scale=4.0`
- `sd_reward_mode=learned`
- `sd_action_prior_enabled=false`
- `sd_height_loss_coef=0.0`
- `sd_online_supervision_enabled=true`
- `sd_online_supervision_one_step_weight=1.0`
- `sd_online_supervision_rollout_weight=1.0`
- `sd_online_supervision_omega_weight=0.25`
- `sd_online_supervision_contact_weight=0.25`
- `sd_online_supervision_phase_weight=2.0`
- `sd_latent_consistency_struct_weight=1.0`
- `sd_latent_consistency_resid_weight=1.0`

含义：

- 保持纯环境奖励闭环口径不变
- 继续使用当前最佳 supervised SD checkpoint 初始化
- 在共享 `SD transition core` 上追加独立在线监督更新
- 用于验证在线监督支路能否稳定接入、落盘并写出训练统计

planning-aware SD 5k smoke 脚本：

用法：

```bash
bash scripts/run_hopper_v5_5k_pureenv_sdinit_planning_sd.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_5k_pureenv_sdinit_planning_sd.sh \
  1 sd_hopper_5k_pureenv_sdinit_planning_sd_seed3_20260430_run1 3
```

相对 online supervision 版本，额外启用：

- `sd_online_planning_weight_enabled=true`
- `sd_online_planning_weight_mode=td_positive`
- `sd_online_planning_weight_coef=1.0`
- `sd_online_planning_weight_max=5.0`

含义：

- 保持纯环境奖励、监督初始化、共享 transition core 与在线 `SD` 监督口径不变
- 使用 TD 目标对在线 `SD` 监督样本做均值归一化重加权
- 用于检查 planning-aware 样本权重是否改善早期训练统计、关键相位轨迹与闭环得分

planning-aware SD 100k 三种子脚本：

脚本：

- [run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh)

用法：

```bash
bash scripts/run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh \
  1 sd_hopper_100k_pureenv_sdinit_planning_sd_seed1_20260430_run1 1
```

固定口径：

- `target_steps=100000`
- `steps=100500`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- 其余 `SD` 与 planning-aware 参数沿用 5k smoke

含义：

- 目标仍按 `100k` 训练预算比较；
- 实际多跑 `500` 步，用于等待 `100k` 评估触发到达 episode 边界，减少末端 `eval.csv` 和 `official_eval/step_0100000/` 漏写；
- 若需精确评估 `models/100000.pt`，仍应按第九节从 checkpoint 导出 eval 轨迹。

mechanism-oriented SD 100k 三种子脚本：

脚本：

- [run_hopper_v5_100k_pureenv_sdinit_mech_sd_budget.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit_mech_sd_budget.sh)

用法：

```bash
bash scripts/run_hopper_v5_100k_pureenv_sdinit_mech_sd_budget.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_100k_pureenv_sdinit_mech_sd_budget.sh \
  1 sd_hopper_100k_pureenv_sdinit_mech_sd_budget_seed1_20260501_run1 1
```

相对 planning-aware SD 100k 版本，固定修订为：

- `sd_online_planning_weight_enabled=false`
- `sd_action_prior_enabled=false`
- `sd_online_supervision_phase_weight=2.0`
- `sd_online_supervision_omega_weight=0.25`
- `sd_online_supervision_contact_weight=0.25`
- `num_samples=256`
- `num_elites=32`
- `num_pi_trajs=24`
- `iterations=6`

含义：

- 保留纯环境奖励、supervised `SD` 初始化、共享 `SD transition core` 与独立在线 `SD` 监督；
- 移除 critic-dependent 的 `td_positive` 样本重加权，使 `SD` 监督回退到机制关键相位口径；
- 将 planner 从原先 `128/16/16/3` 恢复到半默认以上的搜索预算，减少与 TD-MPC2 对比时由搜索预算过低引入的混杂；
- 仍按 `100k` 目标预算比较，实际 `steps=100500`，用于等待 `100k` 评估触发到达 episode 边界。

mechanism-oriented SD 200k 接续脚本：

脚本：

- [run_hopper_v5_200k_pureenv_sdinit_mech_sd_budget_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_mech_sd_budget_resume.sh)

用法：

```bash
bash scripts/run_hopper_v5_200k_pureenv_sdinit_mech_sd_budget_resume.sh \
  <gpu_id> <exp_name> <seed> <resume_from_full_pt>
```

示例：

```bash
bash scripts/run_hopper_v5_200k_pureenv_sdinit_mech_sd_budget_resume.sh \
  1 sd_hopper_200k_pureenv_sdinit_mech_sd_budget_seed1_resume100k_20260502_run1 \
  1 logs/hopper-hop/1/sd_hopper_100k_pureenv_sdinit_mech_sd_budget_seed1_20260501_run1/models/final.full.pt
```

固定口径：

- `target_steps=200000`
- `steps=200500`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- 从 `final.full.pt` 恢复模型、优化器、buffer、计数器与 `SD` checkpoint 状态
- 其余机制监督与 planner 预算沿用 `100k mechanism-oriented SD` 三种子脚本

含义：

- 目标按 `200k` 训练预算比较；
- 实际多跑 `500` 步，用于等待 `200k` 评估触发到达 episode 边界，减少末端 `eval.csv` 和 `official_eval/step_0200000/` 漏写；
- 接续目录必须使用新的 `exp_name`，不得复用 `100k` 目录；
- 继续保持 `sd_online_planning_weight_enabled=false` 与 `256/32/24/6` 的 fairness 预算口径。

mechanism-oriented SD 300k 接续脚本（TD-MPC2 官方默认规划预算）：

脚本：

- [run_hopper_v5_300k_pureenv_sdinit_mech_sd_official_budget_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_300k_pureenv_sdinit_mech_sd_official_budget_resume.sh)

用法：

```bash
bash scripts/run_hopper_v5_300k_pureenv_sdinit_mech_sd_official_budget_resume.sh \
  <gpu_id> <exp_name> <seed> <resume_from_full_pt>
```

示例：

```bash
bash scripts/run_hopper_v5_300k_pureenv_sdinit_mech_sd_official_budget_resume.sh \
  1 sd_hopper_300k_pureenv_sdinit_mech_sd_official_budget_seed1_resume200k_20260504_run1 \
  1 logs/hopper-hop/1/sd_hopper_200k_pureenv_sdinit_mech_sd_budget_seed1_resume100k_20260502_run1/models/final.full.pt
```

固定口径：

- `target_steps=300000`
- `steps=300500`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- 从 `final.full.pt` 恢复模型、优化器、buffer、计数器与 `SD` checkpoint 状态
- `num_samples=512`
- `num_elites=64`
- `num_pi_trajs=24`
- `horizon=3`
- `iterations=6`

含义：

- 目标按 `300k` 训练预算比较；
- 实际多跑 `500` 步，用于等待 `300k` 评估触发到达 episode 边界，减少末端 `eval.csv` 和 `official_eval/step_0300000/` 漏写；
- `SD` 监督、reward、residual 与 `sd_online_planning_weight_enabled=false` 口径保持不变；
- planner 搜索预算恢复为 TD-MPC2 官方默认 `512/64/24/6`，用于观察在更公平预算下该机制导向版本的长程收益。

planning-aware SD 200k 接续脚本：

脚本：

- [run_hopper_v5_200k_pureenv_sdinit_planning_sd_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_planning_sd_resume.sh)

用法：

```bash
bash scripts/run_hopper_v5_200k_pureenv_sdinit_planning_sd_resume.sh \
  <gpu_id> <exp_name> <seed> <resume_from_full_pt>
```

示例：

```bash
bash scripts/run_hopper_v5_200k_pureenv_sdinit_planning_sd_resume.sh \
  1 sd_hopper_200k_pureenv_sdinit_planning_sd_seed1_resume100k_20260501_run1 \
  1 logs/hopper-hop/1/sd_hopper_100k_pureenv_sdinit_planning_sd_seed1_20260430_run1/models/final.full.pt
```

固定口径：

- `target_steps=200000`
- `steps=200500`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- 从 `final.full.pt` 恢复模型、优化器、buffer、计数器与 SD checkpoint 状态
- 其余 `SD` 与 planning-aware 参数沿用 100k 三种子脚本

含义：

- 目标按 `200k` 训练预算比较；
- 实际多跑 `500` 步，用于等待 `200k` 评估触发到达 episode 边界，减少末端 `eval.csv` 和 `official_eval/step_0200000/` 漏写；
- 接续目录必须使用新的 `exp_name`，不得复用 100k 目录。

历史 v429 脚本：

脚本：

- [run_hopper_v429_5k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v429_5k_smoke.sh)

用法：

```bash
bash scripts/run_hopper_v429_5k_smoke.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v429_5k_smoke.sh 1 sd_hopper_5k_v438_actionprior_smoke_20260424_run1 3
```

固定口径：

- `steps=5000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`

### 2. 10k smoke

v5 当前推荐脚本：

- [run_hopper_v5_10k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_10k_smoke.sh)
- [run_hopper_v5_10k_pureenv_noresid.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_10k_pureenv_noresid.sh)
- [run_hopper_v5_10k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_10k_pureenv_sdinit.sh)

用法：

```bash
bash scripts/run_hopper_v5_10k_smoke.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_10k_smoke.sh 1 sd_hopper_10k_v5minimal_landingboost_smoke_20260425_run1 3
```

固定口径：

- `steps=10000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`

纯环境奖励 / 无 residual 基线脚本：

用法：

```bash
bash scripts/run_hopper_v5_10k_pureenv_noresid.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_10k_pureenv_noresid.sh \
  0 sd_hopper_10k_pureenv_noresid_seed3_20260428_run1 3
```

固定口径：

- `steps=10000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=17`
- `sd_reward_mode=learned`
- `sd_action_prior_enabled=false`
- `sd_height_loss_coef=0.0`

含义：

- 只保留 TD-MPC2 原有环境奖励监督
- 不做 shaping / curriculum 混入
- 不使用 residual latent
- 仅验证纯结构 SD dynamics 接入 TD-MPC2 后的基线表现

纯环境奖励 / supervised SD init 脚本：

用法：

```bash
bash scripts/run_hopper_v5_10k_pureenv_sdinit.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_10k_pureenv_sdinit.sh \
  0 sd_hopper_10k_pureenv_sdinit_seed3_20260428_run1 3
```

固定口径：

- `steps=10000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_use_transition_residual=true`
- `sd_transition_residual_scale=4.0`
- `sd_reward_mode=learned`
- `sd_action_prior_enabled=false`
- `sd_height_loss_coef=0.0`
- `sd_supervised_init_from=<best_supervised_short_horizon_ckpt>`

含义：

- 只保留 TD-MPC2 原有环境奖励监督
- 不做 shaping / curriculum 混入
- 不使用 residual latent
- 使用当前最佳 supervised SD checkpoint 初始化显式动力学
- 用于检验更新后的 `SD + internal transition residual` 是否能稳定服务闭环训练

### 2.1 独立 SD 拟合

脚本：

- [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py)

用法：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python scripts/fit_hopper_sd_supervised.py \
  --seed 0 \
  --episodes 32 \
  --epochs 200 \
  --device cpu \
  --out_dir results/hopper_sd_supervised/run1
```

输出：

- `summary.json`：一步 / 多步 rollout 误差与关键结构量误差
- `history.json`：训练与验证误差曲线
- `hopper_sd_supervised.pt`：监督拟合后的 SD 参数

实现说明：

- 训练 / 验证划分按 `episode` 进行，而不是按单步转移随机打散，目的是保证 `rollout_5_rmse`、`rollout_10_rmse` 真正对应连续时间序列误差。
- 若启用 `--rollout_horizon > 1` 且 `--rollout_weight > 0`，训练中的多步损失也只从同一 `episode` 内的连续起点构造。
- 当前机器上独立监督拟合脚本默认使用 `MUJOCO_GL=glfw`。虽然导入 `dm_control` 时会提示缺少 `DISPLAY` 的警告，但采样与步进可正常进行；`egl` / `osmesa` 在本环境下不可用。
- 当前默认导出 `rollout_1_rmse / rollout_5_rmse / rollout_10_rmse / rollout_20_rmse`，用于检查短多步到中多步的漂移增长情况。

### 2.2 SD 动作敏感性诊断

脚本：

- [analyze_hopper_sd_action_sensitivity.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_sd_action_sensitivity.py)

用途：

- 固定若干真实 `episode_csv` 状态；
- 对同一状态施加一组动作 bank；
- 分别用 DMControl 环境与 SD world model 做一步推进；
- 比较 `height / vx / vz / pitch / pitch_rate / support_contact / full_foot_support / body_contact_z / z_min` 的一步响应是否一致。

用法：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python scripts/analyze_hopper_sd_action_sensitivity.py \
  --trace-dir logs/hopper-hop/3/<exp_name>/official_eval/step_0010000/episode_csv \
  --samples-per-trace 2 \
  --amplitudes 0.25 0.50 0.75 \
  --out-dir results/hopper_sd_action_sensitivity/run1
```

输出：

- `action_sensitivity_details.csv`：逐状态、逐动作的环境 / SD 响应明细
- `summary.json`：各关键量的平均绝对误差与方向一致率

判读口径：

- `abs_err_mean` 越小越好，表示一步响应量纲接近；
- `dir_match_mean` 越接近 `1.0` 越好，表示环境与 SD 对动作影响方向判断一致；
- `dominant_state_match_mean` 用于粗看一步动作后高层状态是否一致，不应单独作为主要指标。

强结构化对照脚本：

- [run_hopper_v5_10k_structx10.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_10k_structx10.sh)

用法：

```bash
bash scripts/run_hopper_v5_10k_structx10.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_10k_structx10.sh 0 sd_hopper_10k_v5_structx10_20260425_run1 3
```

固定口径：

- `steps=10000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_shaping_scale=10.0`

脚本：

- [run_hopper_v437_10k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v437_10k_smoke.sh)

用法：

```bash
bash scripts/run_hopper_v437_10k_smoke.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v437_10k_smoke.sh 1 sd_hopper_10k_v437_recovery2prepared_smoke_20260424_run1 3
```

固定口径：

- `steps=10000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`

### 2.5 阶段化 5k smoke

用于验证单一阶段的 dense / milestone 设计是否本身具备引导性。

先构造阶段 reset 池：

- [build_hopper_stage_reset_pool.py](/home/data/round/code/sd_world_model_rl/scripts/build_hopper_stage_reset_pool.py)

用法：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python scripts/build_hopper_stage_reset_pool.py \
  --trace-dir logs/hopper-hop/3/<exp_name>/official_eval/step_0025000/episode_csv \
  --stage landing \
  --out-csv results/hopper_stage_diagnostics/landing_pool.csv \
  --out-json results/hopper_stage_diagnostics/landing_pool.json
```

阶段取值：

- `landing`
- `ground`
- `rise`
- `prone_low`
- `prone_lift`

当前默认筛选口径：

- `landing`：抽取 `falling_air` 且 `vz < -0.10` 的空中下落帧
- `ground`：抽取 `grounded_rear / grounded_prone / grounded_upright`
- `rise`：优先抽取 `grounded_prepared`，不足时也接纳高质量 `grounded_upright`
- `prone_low`：抽取清晰前伏、`nose_minus_anchor_x >= 0.10`、`torso_z <= 0.45`、`support_contact < 0.30`
- `prone_lift`：抽取前伏抬身尝试、`nose_minus_anchor_x >= 0.10`、`0.40 < torso_z <= 0.75`、`support_contact >= 0.30`

若要只从一到两条已确认轨迹构造 reset 池，可改用：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python scripts/build_hopper_stage_reset_pool.py \
  --trace-csv <episode_csv_a> \
  --trace-csv <episode_csv_b> \
  --stage prone_low \
  --out-csv results/hopper_stage_diagnostics/prone_low_pool.csv \
  --out-json results/hopper_stage_diagnostics/prone_low_pool.json
```

再启动阶段化训练：

- [run_hopper_v5_stage_5k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_stage_5k_smoke.sh)

用法：

```bash
bash scripts/run_hopper_v5_stage_5k_smoke.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>
```

示例：

```bash
bash scripts/run_hopper_v5_stage_5k_smoke.sh \
  0 sd_hopper_5k_stage_landing_smoke_20260427_run1 3 landing \
  results/hopper_stage_diagnostics/landing_pool.csv
```

固定口径：

- `steps=5000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_stage_mode=<landing|ground|rise|prone_recover>`
- `hopper_stage_reset_pool_csv=<pool_csv>`
- `sd_reward_mode=hybrid`
- `sd_env_reward_scale=0.0`
- `sd_reward_hybrid_alpha=1.0`

含义：

- 每个 episode 从指定阶段池重置
- 只保留该阶段对应的 dense / milestone shaping
- 训练 reward target 只监督 shaping，不混入环境奖励

### 2.6 阶段化 10k smoke

当 `5k` 已经跑通，但需要判断“只是训练不足”还是“奖励结构本身过严”时，统一使用阶段化 `10k` 脚本，不再手写长命令。

脚本：

- [run_hopper_v5_stage_10k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_stage_10k_smoke.sh)

用法：

```bash
bash scripts/run_hopper_v5_stage_10k_smoke.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>
```

示例：

```bash
bash scripts/run_hopper_v5_stage_10k_smoke.sh \
  1 sd_hopper_10k_pronerecover_pure_pronelow_v5_seed2_20260427_run1 2 prone_recover \
  results/hopper_stage_diagnostics/prone_low_pool.csv
```

固定口径：

- `steps=10000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_stage_mode=<landing|ground|rise|prone_recover>`
- `hopper_stage_reset_pool_csv=<pool_csv>`
- `sd_reward_mode=hybrid`
- `sd_env_reward_scale=0.0`
- `sd_reward_hybrid_alpha=1.0`

含义：

- 与阶段化 `5k` 使用完全相同的奖励语义
- 自动在 `5000 / 10000` 两个节点写出 checkpoint 与 `official_eval`
- 如需继续到 `20k`，必须从同一语义版本的 `10000.full.pt` 严格接续

### 3. 50k 三种子主实验

v5 当前推荐脚本：

- [run_hopper_v5_50k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_50k.sh)
- [run_hopper_v5_50k_alpha5_seed2.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_50k_alpha5_seed2.sh)
- [run_hopper_v5_50k_pureenv_sdinit_online_sup.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_50k_pureenv_sdinit_online_sup.sh)

用法：

```bash
bash scripts/run_hopper_v5_50k.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_50k.sh 0 sd_hopper_50000_v5minimal_seed1_20260425_run1 1
bash scripts/run_hopper_v5_50k.sh 1 sd_hopper_50000_v5minimal_seed2_20260425_run1 2
bash scripts/run_hopper_v5_50k.sh 2 sd_hopper_50000_v5minimal_seed3_20260425_run1 3
```

固定口径：

- `steps=50000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`

高结构化混合单种子变体：

```bash
bash scripts/run_hopper_v5_50k_alpha5_seed2.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_50k_alpha5_seed2.sh 0 sd_hopper_50000_v5densepose_alpha5to05_seed2_20260426_run1 2
```

固定口径：

- `steps=50000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_reward_curriculum_alpha=5.0`
- `sd_reward_curriculum_alpha_min=0.5`
- `sd_reward_curriculum_start=10000`
- `sd_reward_curriculum_end=30000`

纯环境奖励 / supervised SD init / online SD supervision 长训脚本：

用法：

```bash
bash scripts/run_hopper_v5_50k_pureenv_sdinit_online_sup.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_50k_pureenv_sdinit_online_sup.sh \
  0 sd_hopper_50000_pureenv_sdinit_online_sup_seed1_20260429_run1 1
```

固定口径：

- `steps=50000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_use_transition_residual=true`
- `sd_transition_residual_scale=4.0`
- `sd_reward_mode=learned`
- `sd_action_prior_enabled=false`
- `sd_height_loss_coef=0.0`
- `sd_online_supervision_enabled=true`
- `sd_online_supervision_one_step_weight=1.0`
- `sd_online_supervision_rollout_weight=1.0`
- `sd_online_supervision_omega_weight=0.25`
- `sd_online_supervision_contact_weight=0.25`
- `sd_online_supervision_phase_weight=2.0`
- `sd_latent_consistency_struct_weight=1.0`
- `sd_latent_consistency_resid_weight=1.0`
- `sd_supervised_init_from=<best_supervised_short_horizon_ckpt>`

含义：

- 保持纯环境奖励闭环口径不变
- 继续使用当前最佳 supervised SD checkpoint 初始化
- 在共享 `SD transition core` 上持续施加在线监督
- 用于与旧 `pureenv_sdinit` 长训做干净对照，判断在线监督是否改善关键相位误差与闭环行为

旧主线脚本：

- [run_hopper_v429_50k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v429_50k.sh)

其参数仍对应较早口径，仅在需要复现实验时使用。

用法：

```bash
bash scripts/run_hopper_v429_50k.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v429_50k.sh 0 sd_hopper_50000_v429_seed1_run1 1
bash scripts/run_hopper_v429_50k.sh 1 sd_hopper_50000_v429_seed2_run1 2
bash scripts/run_hopper_v429_50k.sh 2 sd_hopper_50000_v429_seed3_run1 3
```

固定口径：

- `steps=50000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- 官方评估自动导出轨迹 csv

### 4. 100k 三种子长测

v5 当前推荐脚本：

- [run_hopper_v5_100k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k.sh)
- [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)
- [run_hopper_v5_stage_100k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_stage_100k.sh)

用法：

```bash
bash scripts/run_hopper_v5_100k.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_100k.sh 1 sd_hopper_100000_v5densepose_seed1_20260426_run1 1
bash scripts/run_hopper_v5_100k.sh 2 sd_hopper_100000_v5densepose_seed2_20260426_run1 2
bash scripts/run_hopper_v5_100k.sh 3 sd_hopper_100000_v5densepose_seed3_20260426_run1 3
```

固定口径：

- `steps=100000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`

纯环境奖励、监督初始化 SD 版本：

```bash
bash scripts/run_hopper_v5_100k_pureenv_sdinit.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v5_100k_pureenv_sdinit.sh 1 sd_hopper_100k_pureenv_sdinit_seed1_20260428_run1 1
bash scripts/run_hopper_v5_100k_pureenv_sdinit.sh 2 sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1 2
bash scripts/run_hopper_v5_100k_pureenv_sdinit.sh 3 sd_hopper_100k_pureenv_sdinit_seed3_20260428_run1 3
```

固定口径：

- `steps=100000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_use_transition_residual=true`
- `sd_transition_residual_scale=4.0`
- `sd_reward_mode=learned`
- `sd_env_reward_scale=1.0`
- `sd_action_prior_enabled=false`
- `sd_stage_mode=full`
- `sd_supervised_init_from=<best_supervised_short_horizon_ckpt>`

阶段化长测脚本：

```bash
bash scripts/run_hopper_v5_stage_100k.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>
```

示例：

```bash
bash scripts/run_hopper_v5_stage_100k.sh \
  3 sd_hopper_100k_pronerecover_pure_pronelow_v7firstprinciples_seed3_20260427_run1 3 prone_recover \
  results/hopper_stage_diagnostics/prone_low_pool.csv
```

固定口径：

- `steps=100000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_stage_mode=<landing|ground|rise|prone_recover>`
- `hopper_stage_reset_pool_csv=<pool_csv>`
- `sd_reward_mode=hybrid`
- `sd_env_reward_scale=0.0`
- `sd_reward_hybrid_alpha=1.0`

含义：

- 与阶段化 `5k / 10k` 使用完全相同的奖励语义
- 用于判断阶段势函数在长训练预算下能否形成稳定恢复策略
- 不混入环境奖励，避免把阶段引导与原任务回报混在一起

### 5. 200k 长程实验

脚本：

- [run_hopper_v429_200k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v429_200k.sh)
- [run_hopper_v5_200k_pureenv_sdinit_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_resume.sh)

用法：

```bash
bash scripts/run_hopper_v429_200k.sh <gpu_id> <exp_name> <seed>
```

示例：

```bash
bash scripts/run_hopper_v429_200k.sh 0 sd_hopper_200000_v429_seed1_run1 1
```

固定口径：

- `steps=200000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`

纯环境奖励、监督初始化 SD、从既有 `10k probe` 续跑版本：

```bash
bash scripts/run_hopper_v5_200k_pureenv_sdinit_resume.sh <gpu_id> <exp_name> <seed> [resume_from]
```

示例：

```bash
bash scripts/run_hopper_v5_200k_pureenv_sdinit_resume.sh \
  0 sd_hopper_200k_pureenv_sdinit_resume_from10k_seed3_20260428_run1 3
```

固定口径：

- `steps=200000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `save_eval_traces=true`
- `sd_struct_dim=17`
- `sd_latent_dim=512`
- `sd_use_transition_residual=true`
- `sd_transition_residual_scale=4.0`
- `sd_reward_mode=learned`
- `sd_env_reward_scale=1.0`
- `sd_action_prior_enabled=false`
- `sd_stage_mode=full`
- `sd_supervised_init_from=<best_supervised_short_horizon_ckpt>`
- `resume_from=<10k_probe_final.full.pt>`

## 三、训练启动前检查

### 1. 查 GPU 占用

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

判读口径：

- `utilization.gpu=0` 基本可视为空闲
- 仅有少量显存缓存占用，但利用率为 `0`，通常也可开新实验

### 2. 确认不要手写长命令

优先使用：

- `bash scripts/run_hopper_v429_5k_smoke.sh ...`
- `bash scripts/run_hopper_v437_10k_smoke.sh ...`
- `bash scripts/run_hopper_v5_5k_pureenv_sdinit_online_sup.sh ...`
- `bash scripts/run_hopper_v5_50k.sh ...`
- `bash scripts/run_hopper_v5_100k.sh ...`
- `bash scripts/run_hopper_v429_50k.sh ...`
- `bash scripts/run_hopper_v429_200k.sh ...`

不要临时手输 `train.py ...` 全参数串，除非确实要新增新口径脚本。

## 四、训练后立即检查什么

### 0. 什么叫“实验已成功启动”

只有同时满足以下条件，才可汇报“实验已成功启动”：

1. 启动命令本身没有立即退出；
2. `train.csv` 已持续写出，且至少出现 `step=500,1000,...`；
3. 到 `2500` 后，`models/2500.full.pt` 已落盘；
4. 若使用 PTY 挂起训练，必须记录会话编号，后续进度检查以该会话和日志文件为准；
5. 若训练已到评估步数，还需继续核对 `eval.csv` 与 `official_eval/step_XXXXXXXX/` 是否真正写出。

以下信号单独出现时，都不得据此判断“已经成功启动”：

- 只看到脚本打印了开头几行；
- 只看到目录被创建；
- 只看到 `train.csv` 表头，没有后续 step 记录；
- 只看到 `2500.pt`，但没有 `2500.full.pt`；
- 只看到训练步数到了 `5000/10000/25000`，但没有继续核对 `eval.csv` 与 `official_eval`。

### 1. 看是否进入训练

```bash
tail -n 20 logs/hopper-hop/<seed>/<exp_name>/train.csv
```

重点看：

- 是否已经写出 `step=500,1000,...`
- 是否已经走到 `2500`

### 2. 看 `2500.full.pt` 是否落盘

```bash
find logs/hopper-hop/<seed>/<exp_name>/models -maxdepth 1 -type f | sort
```

至少应看到：

- `2500.pt`
- `2500.full.pt`

### 3. 看官方评估是否真正完成

```bash
cat logs/hopper-hop/<seed>/<exp_name>/eval.csv
find logs/hopper-hop/<seed>/<exp_name>/official_eval -maxdepth 3 -type f | sort | tail -n 30
```

注意：

- 训练步数达到 `5000/10000/...` 不等于评估文件已经写完
- 常见情况是训练先到步数，随后才慢慢把 `official_eval` 写出来

### 4. 固定检查顺序

启动任一 Hopper 实验后，固定按以下顺序检查，不得跳步：

1. 看 `train.csv` 是否持续写出；
2. 到 `2500` 后看 `models/2500.full.pt` 是否落盘；
3. 若使用 PTY，记录会话编号；
4. 到评估步数后看 `eval.csv`；
5. 最后看 `official_eval/step_XXXXXXXX/` 是否完整写出。

未完成以上检查前，不得汇报“启动正常”“评估完成”或“实验结束”。

## 五、常用进度检查命令

### 1. 看单条实验当前进展

```bash
d=logs/hopper-hop/3/<exp_name>
tail -n 20 "$d/train.csv"
cat "$d/eval.csv"
find "$d/official_eval" -maxdepth 3 -type f | sort | tail -n 30
```

### 2. 查训练进程

```bash
pgrep -af "train.py|<exp_name>"
```

### 3. 同时看三种子

```bash
for s in 1 2 3; do
  d=logs/hopper-hop/$s/<exp_name_prefix>$s
  echo "== seed$s train =="
  tail -n 8 "$d/train.csv"
  echo "== seed$s eval =="
  cat "$d/eval.csv"
done
```

## 六、评估轨迹默认保存规则

当前仓库中：

- [config.yaml](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/config.yaml)
  默认有：
  `save_full_state: true`
  `save_eval_traces: true`
- [online_trainer.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/trainer/online_trainer.py#L133)
  说明 `official_eval` 轨迹导出只对 `hopper-hop` 生效

因此 Hopper 官方评估后，默认会生成：

- `official_eval/step_0005000/episode_csv/episode_00.csv ...`
- `official_eval/step_0005000/eval_episode_summary.csv`

## 七、从已保存轨迹直接渲染 mp4

脚本：

- [render_hopper_trace_csv_mp4.py](/home/data/round/code/sd_world_model_rl/scripts/render_hopper_trace_csv_mp4.py)

用途：

- 直接把已保存的 `episode_XX.csv` 渲染成 mp4
- 不重新跑评估

稳定口径：

```bash
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  /home/anaconda3/envs/sdmcts_p100/bin/python \
  scripts/render_hopper_trace_csv_mp4.py \
  --trace_csv <trace_csv_path> \
  --out_mp4 <out_mp4_path>
```

示例：

```bash
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  /home/anaconda3/envs/sdmcts_p100/bin/python \
  scripts/render_hopper_trace_csv_mp4.py \
  --trace_csv logs/hopper-hop/3/sd_hopper_5k_v438_actionprior_smoke_20260424_run1/official_eval/step_0005000/episode_csv/episode_07.csv \
  --out_mp4 results/hopper_trace_renders/v438_seed3_step5000_episode07_prepared_peak.mp4
```

输出文件通常放在：

- `results/hopper_trace_renders/`

## 八、从 checkpoint 重新回放评估并导出视频

脚本：

- [make_hopper_checkpoint_eval_mp4.py](/home/data/round/code/sd_world_model_rl/scripts/make_hopper_checkpoint_eval_mp4.py)

用途：

- 对任意 checkpoint 重新跑 eval
- 每条 episode 导出 mp4

稳定口径：

```bash
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  /home/anaconda3/envs/sdmcts_p100/bin/python \
  scripts/make_hopper_checkpoint_eval_mp4.py \
  --checkpoint <checkpoint_path> \
  --out_dir <out_dir>
```

说明：

- 该脚本固定 `MUJOCO_GL=glfw`
- 这是“重跑 eval”的口径，不适用于只想回放已保存轨迹的情形

## 九、从 checkpoint 导出 eval 轨迹 csv

脚本：

- [export_hopper_eval_traces.py](/home/data/round/code/sd_world_model_rl/scripts/export_hopper_eval_traces.py)

用途：

- 对已有 checkpoint 重新跑评估
- 不一定导视频，但会导出 `episode_csv` 与 `eval_episode_summary.csv`

示例：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python scripts/export_hopper_eval_traces.py \
  --checkpoint <checkpoint_path> \
  --out_dir <out_dir> \
  --episodes 10
```

注意：

- 该脚本要求 CUDA
- 不适合在无 GPU 的沙箱内执行

## 十、状态/奖励/动作先验分析脚本

### 1. 轨迹状态语义分析

脚本：

- [analyze_hopper_state_reward_sd_validation.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_state_reward_sd_validation.py)

用途：

- 对保存的 trace csv 做离线状态、奖励、SD 一致性分析

### 2. 动作先验符号标定

脚本：

- [probe_hopper_action_priors.py](/home/data/round/code/sd_world_model_rl/scripts/probe_hopper_action_priors.py)

用途：

- 从失败轨迹中自动选出 `rear_lying / prone_lying` 代表状态
- 对 4 维动作做 one-step / short rollout probe
- 支持 `sd` 与 `env` 两种 backend

示例：

```bash
python scripts/probe_hopper_action_priors.py \
  --backend env \
  --trace-dir logs/hopper-hop/3/<exp_name>/official_eval/step_0010000/episode_csv \
  --rollout-steps 2 \
  --amplitudes 0.35 0.70 \
  --topk 8
```

## 十一、Codex / 沙箱下哪些命令通常需要提权

如果是在 Codex 沙箱中执行，而不是你自己登录终端执行，以下命令通常要申请提权：

### 1. 训练启动

原因：

- 沙箱内 `torch.cuda.is_available()` 常为假
- GPU 训练必须在沙箱外跑
- 在 Codex 中应优先使用“提权 + PTY 会话”方式保活训练进程

典型命令：

- `bash scripts/run_hopper_v429_5k_smoke.sh ...`
- `bash scripts/run_hopper_v437_10k_smoke.sh ...`
- `bash scripts/run_hopper_v5_5k_pureenv_sdinit_online_sup.sh ...`
- `bash scripts/run_hopper_v429_50k.sh ...`
- `bash scripts/run_hopper_v429_200k.sh ...`

### 2. GPU 查询

命令：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

### 3. 进程查询

命令：

```bash
pgrep -af "train.py|<exp_name>"
```

### 4. Hopper 视频渲染

原因：

- 需要 `X/GL` 上下文
- 沙箱内通常拿不到完整 OpenGL 上下文

典型命令：

- `xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' ...`

## 十二、反复出错点总表

1. 不要在 `2500` 前设置官方评估。
2. 不要每次手写 `train.py` 长命令，优先用脚本。
3. 不要忘记换新的 `exp_name`。
4. 不要把“训练步数到达某点”和“官方评估文件写完”混为一谈。
5. 不要把“从 trace csv 渲染 mp4”和“从 checkpoint 重跑 eval 导视频”混为一谈。
6. Hopper 渲染当前固定用：
   `glfw + xvfb-run`
   不要临时改成 `egl` 试错。
7. 在 Codex 中，GPU 训练、`nvidia-smi`、`pgrep`、Hopper 渲染通常都要提权。
8. 不要把“脚本已打印开头几行”当作实验已成功启动。
9. 不要漏记 PTY 会话编号；后续进度检查必须能追溯到具体会话。
10. 在 Codex 沙箱内，不要用 `nohup ... &` 这类后台方式替代正式 GPU 启动；优先提权后以 PTY 会话启动，再按 `train.csv -> 2500.full.pt -> eval.csv -> official_eval` 顺序核对。

## 十三、建议工作流

### 启动新实验

1. 查 GPU：
   `nvidia-smi ...`
2. 选脚本：
   `run_hopper_v429_5k_smoke.sh / run_hopper_v437_10k_smoke.sh / run_hopper_v5_5k_pureenv_sdinit_online_sup.sh / run_hopper_v5_50k.sh / run_hopper_v5_100k.sh / run_hopper_v429_50k.sh / run_hopper_v429_200k.sh`
3. 指定新的 `exp_name`
4. 若在 Codex 中执行，先提权并以 PTY 会话启动
5. 启动后立即检查：
   `train.csv`
   `2500.full.pt`
   若为 PTY 启动，还要记录会话编号
6. 到评估步数后再检查：
   `eval.csv`
   `official_eval`
### 检查评估

1. 看 `eval.csv`
2. 看 `official_eval/step_XXXXXXX/`
3. 看 `eval_episode_summary.csv`
4. 必要时逐条扫描 `episode_csv`

### 看视频

1. 如果已经有 `episode_csv`：
   直接用 `render_hopper_trace_csv_mp4.py`
2. 如果只有 checkpoint：
   用 `make_hopper_checkpoint_eval_mp4.py`

## 十四、文件位置总览

- 训练日志：
  `logs/hopper-hop/<seed>/<exp_name>/`
- 模型：
  `logs/hopper-hop/<seed>/<exp_name>/models/`
- 官方评估轨迹：
  `logs/hopper-hop/<seed>/<exp_name>/official_eval/step_XXXXXXXX/`
- 渲染视频：
  `results/hopper_trace_renders/`

## 十五、维护原则

以后只要有新的稳定口径，优先更新本文件，不要再让口令只存在于对话记忆里。
