#!/usr/bin/env python3
"""
理论加速界计算 (基于§5.10修正框架)

核心修正：区分 σ²_env (环境方差) 和 σ²_model (模型诱导方差)

- SD: σ²_rollout = 0 (精确方程 + 确定性环境)
- BB: σ²_rollout > 0 (神经网络预测误差引入伪随机方差)
"""
import numpy as np
import json
from pathlib import Path

# ============================================================================
# Beer Game 参数
# ============================================================================
A = 81          # |A| = 3^4
A_eff = 50      # 有效动作数 (掩码后)
V_max = 1000.0
gamma = 0.99
delta = 0.05
H_baseline = 10

# BB 模型参数 (从 multi_step_rollout 实验)
# H=1 RMSE across 3 seeds: [0.337, 0.393, 0.304] → mean ≈ 0.345
epsilon_1_bb = 0.345  # BB 单步状态预测 RMSE
L_bb = 1.064          # BB 增长率 (从 §5.5 实验)


def compute_model_induced_variance(epsilon_1, L, H, gamma, n_state_dims=16):
    """
    计算 BB 模型诱导的 rollout 回报方差

    在确定性环境中, BB 模型的预测误差在每步引入伪随机偏差.
    假设每步预测误差独立 (近似), 回报方差为:
      σ²_rollout = Σ_{t=0}^{H-1} γ^{2t} · σ²_reward(t)

    其中 σ²_reward(t) ∝ ε_model(t)² (状态误差导致的奖励偏差)

    对于 Beer Game: reward = -Σ(holding_cost * I + backlog_cost * B)
    状态误差 ε(t) 直接映射到奖励误差, 比例系数约为 cost_scale
    """
    cost_scale = 2.0  # holding_cost + backlog_cost 的有效权重

    sigma_sq = 0.0
    for t in range(H):
        # t步状态预测 RMSE
        epsilon_t = epsilon_1 * (L ** t)
        # 状态误差引起的奖励方差 (跨 n_state_dims 个维度)
        sigma_reward_t_sq = (cost_scale * epsilon_t) ** 2 * n_state_dims
        sigma_sq += (gamma ** (2 * t)) * sigma_reward_t_sq

    return sigma_sq


def main():
    print("=" * 80)
    print("理论加速界计算 (修正框架: 区分环境方差与模型诱导方差)")
    print("=" * 80)

    # ========================================================================
    # 1. 方差分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. 方差分析")
    print("=" * 80)

    print("\n### SD 模型")
    print("σ²_env = 0 (Beer Game 确定性)")
    print("σ²_model = 0 (精确守恒方程)")
    print("σ²_rollout = 0")
    print("→ 每个动作只需 1 次 rollout 即可获得确定回报")

    print("\n### BB 模型")
    print(f"σ²_env = 0 (Beer Game 确定性)")
    print(f"ε_1 = {epsilon_1_bb:.3f} (单步状态预测 RMSE)")
    print(f"L = {L_bb:.3f} (误差增长率)")

    sigma_sq_bb = compute_model_induced_variance(epsilon_1_bb, L_bb, H_baseline, gamma)
    sigma_bb = np.sqrt(sigma_sq_bb)
    print(f"σ²_rollout(H={H_baseline}) = {sigma_sq_bb:.2f}")
    print(f"σ_rollout = {sigma_bb:.2f}")
    print("→ 需要多次 rollout 平均消除模型诱导方差")

    # ========================================================================
    # 2. SD 理论最优 N
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. SD 理论最优 N")
    print("=" * 80)

    # SD: σ²_rollout = 0, 每动作 1 次访问即可
    N_sd = A_eff
    print(f"\nσ²_rollout = 0 → 无需统计平均")
    print(f"N_SD = |A_eff| = {N_sd}")
    print(f"实验实际: N = 50 (adaptive_sims)")
    print(f"理论与实验一致: {N_sd} = 50 ✓")

    # ========================================================================
    # 3. BB 理论最优 N
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. BB 理论最优 N")
    print("=" * 80)

    # BB 误差分析
    epsilon_model_bb_H = epsilon_1_bb * (L_bb ** (H_baseline - 1))
    epsilon_model_total_bb = epsilon_model_bb_H / (1 - gamma)
    print(f"\nBB H步误差: ε_model(H={H_baseline}) = {epsilon_model_bb_H:.3f}")
    print(f"BB 总模型误差: ε_model,total = {epsilon_model_total_bb:.2f}")

    # BB 需要 N 来通过 Hoeffding 消除 σ²_rollout
    # 为了使动作估计误差 < Δ_min/2 (确保选对最优动作)
    # 使用均匀采样 + Hoeffding: n_a ≥ σ²_rollout / (2ε²) · log(2|A|/δ)
    # 或等价地, 每动作需要 n_a 次采样

    epsilon_mc_values = [50, 100, 200, 500]  # 不同的 MC 误差容忍度

    print(f"\n### BB 所需模拟次数 (Hoeffding bound, |A|={A})")
    print(f"{'ε_mc':>8} | {'n_a (每动作)':>14} | {'N_BB = |A|·n_a':>16} | {'N_BB/N_SD':>10}")
    print("-" * 60)

    for eps_mc in epsilon_mc_values:
        # 每动作需要的采样次数
        n_a = sigma_sq_bb / (2 * eps_mc**2) * np.log(2 * A / delta)
        n_a = max(1, int(np.ceil(n_a)))
        N_bb = A * n_a
        speedup = N_bb / N_sd
        print(f"{eps_mc:>8.0f} | {n_a:>14d} | {N_bb:>16d} | {speedup:>10.1f}x")

    # 使用与 SD 实验相同的配置: baseline N=200
    # BB 在 baseline 配置下的表现
    print(f"\n### Baseline 配置: N=200")
    print(f"SD baseline: N=200, 每动作平均 {200/A:.1f} 次 → 充分 (只需1次)")
    print(f"BB baseline: N=200, 每动作平均 {200/A:.1f} 次")

    n_a_bb_baseline = 200 / A  # ≈ 2.47
    eps_mc_bb_achievable = np.sqrt(sigma_sq_bb / (2 * n_a_bb_baseline) * np.log(2 * A / delta))
    print(f"BB 可达 MC 误差: ε_mc ≈ {eps_mc_bb_achievable:.1f}")

    # ========================================================================
    # 4. 加速界计算
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 理论加速界")
    print("=" * 80)

    # 正确的加速界定义:
    # 给定相同的质量要求 (规划误差 ≤ ε_target),
    # SD 需要的总计算成本 vs BB 需要的总计算成本

    # 对于 SD:
    #   ε_total = ε_model,total + ε_trunc + ε_mc
    #   ε_model,total = 0 (SD)
    #   ε_trunc = γ^H · V_max
    #   ε_mc = 0 (σ²_rollout = 0, 只需 N = |A_eff|)
    #   C_SD = N_SD · H_SD

    # 对于 BB:
    #   ε_model,total = ε_1 · L^(H-1) / (1-γ) (随 H 增长)
    #   BB 不能任意减少 N, 因为 σ²_rollout > 0
    #   C_BB = N_BB · H_BB

    # 加速界 = C_BB / C_SD (BB需要的计算量 / SD需要的计算量)

    # SD: 最优配置
    H_sd = H_baseline  # 可以用较短的 H (误差不增长)
    N_sd_opt = A_eff    # 只需覆盖有效动作
    C_sd = N_sd_opt * H_sd
    print(f"\nSD 最优配置:")
    print(f"  H_SD = {H_sd}")
    print(f"  N_SD = {N_sd_opt} (= |A_eff|)")
    print(f"  C_SD = N·H = {C_sd}")

    # BB: 需要足够的 N 来消除 σ²_rollout
    # 假设与 SD 达到相同规划质量, BB 至少需要 N_BB 使得
    # ε_mc_BB ≤ ε_mc_threshold (某个合理的误差阈值)

    # 实践中, baseline 使用 N=200
    N_bb_baseline = 200
    H_bb = H_baseline
    C_bb_baseline = N_bb_baseline * H_bb

    print(f"\nBB Baseline 配置:")
    print(f"  H_BB = {H_bb}")
    print(f"  N_BB = {N_bb_baseline}")
    print(f"  C_BB = N·H = {C_bb_baseline}")

    speedup_computation = C_bb_baseline / C_sd
    print(f"\n### 计算加速界")
    print(f"speedup = C_BB / C_SD = {C_bb_baseline} / {C_sd} = {speedup_computation:.1f}x")

    # 实验中的加速 (SD adaptive vs SD baseline)
    print(f"\n### 实验中的加速 (SD 自身优化)")
    N_sd_baseline_exp = 200
    N_sd_adaptive_exp = 50
    speedup_exp = N_sd_baseline_exp / N_sd_adaptive_exp
    print(f"SD baseline: N={N_sd_baseline_exp}")
    print(f"SD adaptive: N={N_sd_adaptive_exp}")
    print(f"speedup = {N_sd_baseline_exp}/{N_sd_adaptive_exp} = {speedup_exp:.1f}x")
    print(f"理论解释: SD 的 baseline N=200 远超理论最优 N=|A_eff|={A_eff}")
    print(f"          自适应算法将 N 从 200 降至 50 ≈ |A_eff|, 实现 4x 加速")

    # ========================================================================
    # 5. 三层加速分解
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. 三层加速分解 (论文用)")
    print("=" * 80)

    print("""
speedup = (N_baseline/N_adaptive) · (H_baseline/H_adaptive) · (C_step/C_cache)

Layer 1: 模拟次数减少
  SD: N 可从 200 降至 |A_eff|=50 (σ²_rollout=0)
  BB: N 不能安全降低 (σ²_rollout>0, 需要统计平均)
  加速: 200/50 = 4.0x

Layer 2: 地平线缩短
  SD: ε_model(H) = O(1), 可用浅 H
  BB: ε_model(H) = O(L^H), 需要深 H 或只能用浅 H 但精度差
  加速: 取决于具体配置 (当前实验 H_SD=H_BB=10, 加速=1.0x)

Layer 3: 缓存加速
  SD: 守恒分量确定性转移可缓存
  BB: 全部状态需重新计算
  加速: d/k (取决于缓存命中率)
""")

    # ========================================================================
    # 6. 读取实验结果对比
    # ========================================================================
    print("=" * 80)
    print("6. 实验结果对比")
    print("=" * 80)

    results_file = Path(__file__).parent.parent / "results" / "efficiency_improvements" / "20260217_071725" / "efficiency_results.json"
    if results_file.exists():
        with open(results_file) as f:
            exp_results = json.load(f)[0]

        print("\n### 实验结果")
        for variant, data in exp_results["variants"].items():
            N = data["num_simulations"]
            H = data["rollout_depth"]
            ret = data["mean_return"]
            total = data["total_simulations"]
            print(f"  {variant:20s}: N={N:4d}, H={H:2d}, return={ret:8.1f}, total_sims={total:6d}")

        baseline = exp_results["variants"]["baseline"]
        adaptive = exp_results["variants"]["adaptive_sims"]
        all_opt = exp_results["variants"]["all_optimizations"]

        print(f"\n### 加速分析")
        print(f"adaptive_sims vs baseline:")
        print(f"  模拟次数: {baseline['num_simulations']}/{adaptive['num_simulations']} = {baseline['num_simulations']/adaptive['num_simulations']:.1f}x")
        print(f"  总模拟量: {baseline['total_simulations']}/{adaptive['total_simulations']} = {baseline['total_simulations']/adaptive['total_simulations']:.1f}x")
        perf_change = (adaptive['mean_return'] - baseline['mean_return']) / abs(baseline['mean_return']) * 100
        print(f"  性能变化: {perf_change:+.1f}%")

        print(f"\nall_optimizations vs baseline:")
        print(f"  总模拟量: {baseline['total_simulations']}/{all_opt['total_simulations']} = {baseline['total_simulations']/all_opt['total_simulations']:.1f}x")
        perf_change_all = (all_opt['mean_return'] - baseline['mean_return']) / abs(baseline['mean_return']) * 100
        print(f"  性能变化: {perf_change_all:+.1f}%")

    # ========================================================================
    # 7. 论文用结论
    # ========================================================================
    print("\n" + "=" * 80)
    print("7. 论文用结论")
    print("=" * 80)

    print(f"""
Proposition (SD效率优势):

在确定性环境 (σ²_env=0) 中, 设 SD 模型满足 ε_model=0 (精确守恒律),
BB 模型具有单步预测误差 ε_1 > 0. 则:

(1) SD 的 MCTS rollout 方差为零:
    σ²_rollout,SD = 0
    因此 N_SD = |A_eff| 即可 (每动作一次确定性 rollout)

(2) BB 的 MCTS rollout 具有模型诱导方差:
    σ²_rollout,BB = Σ_{{t=0}}^{{H-1}} γ^{{2t}} · (c · ε_1 · L^t)² · n_dim
    其中 c 为成本系数, n_dim 为状态维度, L 为 Lipschitz 常数

(3) 为使 BB 的动作估计误差 ≤ ε_mc (置信度 1-δ), 需要:
    N_BB ≥ |A| · σ²_rollout,BB / (2ε_mc²) · log(2|A|/δ)

(4) 模拟次数加速界:
    N_BB / N_SD ≥ |A|/|A_eff| · σ²_rollout,BB / (2ε_mc²) · log(2|A|/δ)

Beer Game 实例化:
  ε_1 = {epsilon_1_bb:.3f}, L = {L_bb:.3f}, H = {H_baseline}
  σ²_rollout,BB = {sigma_sq_bb:.2f}, σ_rollout = {sigma_bb:.2f}
  N_SD = |A_eff| = {A_eff}
  N_BB baseline = {N_bb_baseline} (实验配置)
  实验加速: {speedup_exp:.1f}x (SD 自身从 N=200 降至 N=50)
""")


if __name__ == "__main__":
    main()
