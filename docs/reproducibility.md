# Reproducibility

## 当前 Hopper 主线环境

### 运行环境

1. Python 解释器路径：`/home/anaconda3/envs/sdmcts_p100/bin/python`
2. 主要训练入口：
   - [external/tdmpc2/tdmpc2/train.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/train.py)
3. 当前主要实验任务：`task=hopper-hop`

### 当前活跃 world model 口径

1. `world_model_kind=sd_hopper`
2. `sd_struct_dim=17`
3. `sd_latent_dim=512`
4. `sd_use_transition_residual=true`
5. `sd_transition_residual_scale=4.0`
6. `sd_reward_mode=learned`
7. `sd_action_prior_enabled=false`
8. `sd_height_loss_coef=0.0`
9. `sd_struct_loss_scale_mode=batch_std`
10. `sd_struct_loss_scale_floor=0.1`
11. `sd_latent_consistency_struct_weight=1.0`
12. `sd_latent_consistency_resid_weight=1.0`

### 当前在线 SD 监督支路默认配置

截至 `2026-04-29`，TD-MPC2 默认配置文件 [config.yaml](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/config.yaml) 已补入在线 `SD` 监督支路所需参数。默认值如下：

1. `sd_online_supervision_enabled=false`
2. `sd_online_supervision_lr=3e-4`
3. `sd_online_supervision_one_step_weight=1.0`
4. `sd_online_supervision_rollout_weight=1.0`
5. `sd_online_supervision_omega_weight=0.25`
6. `sd_online_supervision_contact_weight=0.25`
7. `sd_online_supervision_phase_weight=2.0`
8. `sd_struct_loss_scale_mode=batch_std`
9. `sd_struct_loss_scale_floor=0.1`
10. `sd_latent_consistency_struct_weight=1.0`
11. `sd_latent_consistency_resid_weight=1.0`

说明如下：

1. 默认仍关闭该支路，以保证旧实验脚本不会因默认行为改变而失真；
2. 一旦显式启用，将为共享 `SD transition core` 增加独立 `Adam` 监督更新；
3. 当前最小实现依赖 replay batch 的 `horizon+1` 采样窗口，因此实际监督视界为 `1-step + 3-step`。
4. 自 `2026-04-29` 修复版本起，主 `consistency_loss` 的 SD 结构态部分与在线 SD 监督损失均使用 batch 目标结构态标准差归一化。
5. 主 `consistency_loss` 中的结构态块与 residual latent 块默认按块等权，避免 495 维残差块仅因维度数量压过 17 维显式结构态。

### 下一版本 planning-aware SD 权重配置

截至 `2026-04-30`，配置文件新增以下默认关闭的 planning-aware 权重参数：

1. `sd_online_planning_weight_enabled=false`
2. `sd_online_planning_weight_mode=td_positive`
3. `sd_online_planning_weight_coef=1.0`
4. `sd_online_planning_weight_max=5.0`

默认关闭意味着既有 `pureenv_sdinit_online_sup` 实验不受影响。启用该开关后，在线 `SD` 监督损失会根据 TD 目标或奖励目标对样本加权，用于验证 planning signal 是否能缓解平均结构态监督与闭环规划目标之间的错配。权重在 batch 内按均值归一化，并用 `sd_online_planning_weight_max` 截断，以降低启用该机制后因有效学习率变化造成的混淆。

### 当前监督初始化 checkpoint

用于当前 `pure-env + sdinit` 主线的监督初始化 checkpoint 为：

- [hopper_sd_supervised.pt](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt)

该 checkpoint 的关键信息为：

1. `seed=0`
2. `episodes=12`
3. `max_steps=250`
4. `epochs=30`
5. `batch_size=256`
6. `short_rollout_weight=0.10`
7. `use_transition_residual=true`
8. `transition_residual_scale=4.0`

对应摘要文件：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/summary.json)

### 当前闭环实验脚本

当前主要使用以下脚本：

1. [run_hopper_v5_5k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_5k_pureenv_sdinit.sh)
2. [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)
3. [run_hopper_v5_200k_pureenv_sdinit_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_resume.sh)

### 当前讨论所依据的关键实验

1. `100k` 三种子 pure-env + sdinit 主实验：
   - seed1: [sd_hopper_100k_pureenv_sdinit_seed1_20260428_run1](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/1/sd_hopper_100k_pureenv_sdinit_seed1_20260428_run1)
   - seed2: [sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/2/sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1)
   - seed3: [sd_hopper_100k_pureenv_sdinit_seed3_20260428_run1](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_100k_pureenv_sdinit_seed3_20260428_run1)
2. `200k` seed3 接续实验：
   - [sd_hopper_200k_pureenv_sdinit_resume_from10k_seed3_20260428_probe](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_200k_pureenv_sdinit_resume_from10k_seed3_20260428_probe)

### 当前关键诊断文件

1. `75k replay` 误差诊断：
   - [diagnostics_20260429_seed3_resume75k_policy_error.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/diagnostics_20260429_seed3_resume75k_policy_error.json)
2. `50k replay` 误差诊断：
   - [diagnostics_20260429_seed3_resume50k_policy_error.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/diagnostics_20260429_seed3_resume50k_policy_error.json)

### 说明

当前研究结论明确依赖以下假设：

1. 评估重点是 TD-MPC2 当前实际使用的短视界（尤其 `1 / 3 / 5` 步），而不是长视界自由仿真；
2. 当前 `SD` 的用途是作为 TD-MPC2 的 transition core，而不是独立完成完整控制任务；
3. 当前闭环行为结果不应被解读为“跳跃任务已解决”，而应被视为 transition core 与 planner 相互作用的诊断样本。
