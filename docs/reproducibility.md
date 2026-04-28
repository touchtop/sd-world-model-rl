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
3. `sd_latent_dim=17`
4. `sd_use_transition_residual=true`
5. `sd_transition_residual_scale=4.0`
6. `sd_reward_mode=learned`
7. `sd_action_prior_enabled=false`
8. `sd_height_loss_coef=0.0`

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
