# SD Hopper 作为 TD-MPC2 World Model Core 的能力审计

## 2026-04-28 补充结论

本文件主体完成时，`sd_hopper` 仍存在明显的根部动作作用链缺口。2026-04-28 已完成两轮修补，并新增修复后结论如下。

### 接口能力结论更新

当前 `sd_hopper` 在接口层面已满足如下最低条件：

1. `encode(obs) -> z` 仍保持 17 维显式结构状态；
2. `next(z, a) -> z'` 中，动作已能在同一步显式传入 grounded 与 airborne 两类根部动力学通道；
3. 动作敏感性不再呈现“每个样本内部对 24 个动作给出完全相同根部响应”的结构退化。

对应新增证据见：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000_after_fix2/summary.json)
- [action_sensitivity_details.csv](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000_after_fix2/action_sensitivity_details.csv)

### 但仍不能宣称“已经可靠替代黑盒 world model”

修复后需要把“是否具备替代条件”拆成两层判断：

第一层，结构层：

1. 旧问题中的 `vx / pitch_rate` 全动作同值塌缩已经解除；
2. 这一层已经通过。

第二层，定量层：

1. `vx.dir_match_mean` 虽由 `0.16875` 提升到 `0.44583`，但仍不高；
2. `pitch_rate.dir_match_mean` 仅由 `0.33125` 提升到 `0.36667`；
3. 修复后从头监督拟合的多步误差没有同步改善。

修复后监督拟合结果见：

- 修复前基线：
  [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1/summary.json)
- 修复后重跑：
  [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_after_fix2/summary.json)

关键指标对比如下：

1. `one_step_struct_rmse`：`0.6318 -> 0.6409`
2. `rollout_5_rmse`：`1.1015 -> 1.1402`
3. `rollout_10_rmse`：`1.1860 -> 1.2461`
4. `vx_mae`：`0.0727 -> 0.0944`
5. `pitch_mae`：`0.0229 -> 0.0258`

因此当前最准确的判断是：

> `sd_hopper` 已经从“接口结构不成立”推进到“接口结构成立但数值质量不足”的阶段。

这也解释了为何旧 GPU 上那轮监督拟合结果现在只能保留为“修复前基线”，而不能继续作为当前模型可靠性的代表证据。

### 关于第一轮监督校准的补充判断

2026-04-28 随后进行了第一轮针对 `vx / omega` 的监督口径校准，结果位于：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_after_fix2_calib1/summary.json)

该轮结果表明：

1. `vx_mae` 与 `pitch_rate_mae` 继续下降；
2. `rollout_10/20_rmse` 明显下降；
3. 但 `one_step_struct_rmse` 没有同步变好，`vz_mae` 还有所恶化。

因此，对“当前 SD 是否已具备替代条件”的判断仍应保持保守：

> 当前模型已满足“接口上可以接入并产生有区分度的动作响应”，但尚未满足“数值质量已足以无保留替代黑盒 world model”。

更具体地说，下一步优化不应再泛化为“继续整体拟合”，而应围绕：

1. 根部关键量加权；
2. 多步 rollout 稳定性；
3. `vz` 与 `omega` 之间的权衡；
4. 监督训练效率；

这四点做针对性迭代。

### 2026-04-28 进一步更新：当前主问题已收缩到 grounded 接触支撑

在完成 `3.0x / 4.0x residual` 监督对照与责任分析后，当前判断需要进一步收紧。

新增证据：

1. `4.0x residual` 监督结果：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid4p0/summary.json)
2. `4.0x residual` 责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid4p0/residual_usage_analysis.json)

其中最关键的现象是：

1. grounded 样本上，`vz` 的 `mean_abs_base_delta ≈ 0.389`，接近 `g * dt ≈ 0.392`；
2. 与此同时，`vz` 的 residual 又以几乎等量、反向的方式补回；
3. 这说明主干在大量 grounded 帧里，仍近似把根部竖直更新当成“自由落体减重力”，而没有形成足够的地面法向支撑。

因此，本文件的总体判断应进一步更新为：

> 当前 Hopper SD 已不再主要卡在“动作作用链缺失”，而是主要卡在 grounded 接触支撑刻画过弱。更具体地说，现有主干尚未把“脚已着地时，地面如何为根部提供法向支撑、切向约束与力矩”这件事刻画正确。

据此，后续优化的优先级不再是继续放大 residual，而是：

1. 先回填 grounded 法向支撑；
2. 再回填 grounded 切向滑移 / 摩擦；
3. 最后再看 `omega` 的力矩链与等效惯量校准。

在上述基础上，2026-04-28 又进一步验证了 grounded 支撑门的作用。将 grounded 支撑基线从“线性乘 `total_gate`”改为“对 `total_gate` 施加更陡的软门”后：

1. 无残差同口径 smoke：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_noresid_cpu/summary.json)
2. `4.0x residual` 同口径 smoke：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_resid4_cpu/summary.json)
3. 新责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_resid4_cpu/residual_usage_analysis.json)

最重要的变化是：

1. 无残差 `vz_mae`：`0.3365 -> 0.3144`
2. 无残差 `rollout_20_rmse`：`2.1232 -> 2.0515`
3. residual 责任分析中，`vz.mean_abs_base_delta`：`0.3239 -> 0.2941`

因此，这一轮已经可以进一步确认：

> 当前主干竖直支撑不足的直接来源之一，确实是 grounded 支撑门过软。把支撑门做陡之后，主干已能继续回收一部分 `vz` 责任。

但同时也要保留约束性结论：

1. `4.0x residual` 的 `vz_mae` 仅基本持平，说明这轮修补还未把 residual 从 grounded `vz` 上彻底赶回局部修正；
2. 当前仍处在“方向正确但 grounded 支撑尚未充分”的阶段。

### 2026-04-28 再追加：仿真器先验有用，但不能直接原值硬塞进当前归一化主干

又进一步核对了 `dm_control` 的 Hopper 任务实现与 XML 模型，关键事实包括：

1. 任务控制步长 `control_timestep = 0.02`，而当前 TD-MPC2 包装器每次 agent 动作会执行环境两次，因此 `SD` 当前使用的 `step_dt = 0.04` 并未设错；
2. 真实总质量约为 `12.44`，按 reset 姿态估出的俯仰等效惯量约为 `2.86`；
3. 关节 `damping = 0.05`、`armature = 0.2`，执行器 gear 为 `30 / 40 / 30 / 10`；
4. 接触摩擦与软接触参数在 XML 中也有明确定义。

这些信息本身是有价值的，但直接将 `mass / inertia / gear / damping` 原值硬塞进当前 `SD` 主干后，快速监督 smoke 表现反而变差：

1. 无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_simprior1_noresid_cpu/summary.json)
2. `4.0x residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_simprior1_resid4_cpu/summary.json)

相对当时最佳 grounded 主干版本，这两条对照的共同特征是：

1. `vx / vz` 一步误差没有明显恶化；
2. 但 `rollout_10 / rollout_20` 明显变差；
3. `pitch_rate` 也没有得到改善。

因此，当前应把这一轮仿真器审计的结论写成：

> 仿真器代码确实提供了非常有用的定量先验，但这些先验不能在不重标当前归一化接触/驱动链的前提下，直接作为主干量纲替换项使用。它们更适合作为下一轮“系统重标”的依据，而不是作为局部补丁直接插入。

随后又尝试了更严格的一版“系统重标”：不仅把 `mass / inertia / gear / damping` 作为先验引入，还把 grounded 法向支撑、空中耦合等所有显式力/力矩项一起按质量或惯量同步重标。对应结果：

1. 无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_simprior2_noresid_cpu/summary.json)

结果比 `simprior1` 更差，尤其是：

1. `rollout_10_rmse = 2.88`
2. `rollout_20_rmse = 3.22`
3. `pitch_rate_mae = 1.03`

这说明一个更强的结论：

> 当前问题不是“sim prior 加得不够彻底”，而是“当前 17 维主干与这套显式接触近似，本身还承接不了物理量纲更真的系统重标”。因此，仿真器先验在本阶段更适合做审计与设计约束，而不是直接进入活跃主线。

## 0. 目的

本文只回答一个问题：

> 当前 `sd_hopper` 设计，是否具备为 TD-MPC2 提供 `encode(obs) -> z` 与 `next(z, a) -> z'` 这两条核心 world-model 接口所需的能力。

本文不讨论结构化奖励设计，不讨论 shaping 是否有效，也不讨论行为好坏；只讨论 world model 能力本身。

---

## 1. 接口替换事实

### 1.1 替换发生在哪里

当配置中：

```text
world_model_kind = sd_hopper
```

时，TD-MPC2 agent 不再实例化原始黑盒 `WorldModel`，而是实例化 `SDWorldModel`：

- [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:23)
- [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:25)

### 1.2 替换了什么，没替换什么

当前替换范围是：

1. `encode(obs) -> z`
2. `next(z, a) -> z'`

对应实现位于：

- [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py:161)
- [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py:171)

没有被显式 SD 取代的部分仍然是 learned heads：

1. `reward(z, a, z')`
2. `Q(z, a)`
3. `pi(z)`

对应实现位于：

- [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py:287)
- [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py:339)
- [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py:358)

因此，当前更准确的说法不是“整个 TD-MPC2 world model 已被 SD 替换”，而是：

> TD-MPC2 的状态表示接口与状态转移接口已被 SD 接管，而 reward / value / policy 头仍是原算法的 learned heads。

### 1.3 TD-MPC2 实际在什么地方调用这两条接口

规划时：

1. `act()` 中先 `encode(obs)`，见 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:137)
2. `_plan()` 中 rollout 反复调用 `next(...)`，见 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:175)、[tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:189)
3. `_estimate_value()` 中每一步也依赖 `next(...)` 和 `reward(...)`，见 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:149)、[tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:150)

训练时：

1. replay 中的 `obs_t`、`obs_{t+1}` 会先各自 `encode(...)`，见 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:314)、[tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:315)
2. latent rollout 通过 `next(...)` 展开并计算 consistency loss，见 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:329)、[tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py:335)

因此，`encode` 与 `next` 一旦有系统性偏差，既会污染规划，也会污染训练目标。

---

## 2. TD-MPC2 对这两条接口真正需要什么

对 TD-MPC2 而言，`encode` 与 `next` 至少要满足以下能力：

1. 一步预测要足够准确；
2. 多步 rollout 不能快速漂移；
3. 对动作变化要有正确且可分辨的响应；
4. 输出的 latent 几何必须能被 learned `reward / Q / pi` 头稳定利用；
5. 在 planner 实际采样到的状态动作分布上也要成立，而不只是随机动作数据上成立。

仅仅“一步监督误差不高”并不足以证明它满足 TD-MPC2 所需能力。

---

## 3. 当前 17 维结构状态的语义

当前 `sd_hopper` 在 `sd_struct_dim=17` 时，结构 latent 为：

```text
H,
vx, vz,
theta, omega,
A,
waist, hip, knee, ankle,
waist_dot, hip_dot, knee_dot, ankle_dot,
touch_toe, touch_heel,
z_min
```

见：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:684)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:1624)

与环境原始状态观测相比：

1. 原始观测是 `15` 维：`position(6) + velocity(7) + touch(2)`，见 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:308)
2. 结构状态额外引入了两个派生量：
   `A` 与 `z_min`

### 3.1 正面评价

这 17 维里，大部分量都具有明确物理语义：

1. `H, vx, vz, theta, omega` 对应 root 竖直位置、平动速度与躯干俯仰；
2. `waist, hip, knee, ankle` 及其速度对应四个关节自由度；
3. `touch_toe, touch_heel` 提供接触强度信息；
4. `z_min` 提供几何近地信息。

省略 `rootx` 在 Hopper-Hop 中通常是合理的，因为：

1. 环境奖励依赖站立高度与前向速度，而不依赖绝对水平位置；
2. 几何计算也只需相对形态，不需世界系绝对 `x`。

这一点说明该状态并非显然缺失“全局平移”信息。

### 3.2 结构上最值得警惕的量：`A`

`A` 不是环境真实观测量，而是从当前观测启发式估计出来的 activation proxy：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:645)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:682)

然后在 forward 时，`A` 又按一阶动力系统随动作推进：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:790)

这带来一个重要语义风险：

1. `encode(obs)` 生成的 `A` 是观测函数；
2. rollout 中的 `A` 是内部状态；
3. 这两者不一定完全一致。

也就是说，当前 `A` 同时承担了“由当前观测反推的代理变量”和“需要时间递推的内部记忆变量”两种角色。

这不是致命错误，但它是当前 17 维状态中最明显的闭合性风险点。若未来出现：

1. 一步误差尚可；
2. 多步 rollout 很快漂移；
3. 或 planner rollout 与 encode 重置后的状态语义不一致；

则 `A` 是第一优先级排查对象。

### 3.3 `z_min` 不是独立自由状态，而是冗余派生量

`z_min` 由当前几何姿态直接计算：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:503)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:735)

并且在 `project_state()` 中会被重新计算，而不是信任输入值：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:708)

这说明：

1. `z_min` 对 planner 是可用特征；
2. 但它不是独立自由变量；
3. 它更像是“为了让状态判断与接触近似更方便而显式缓存的几何派生量”。

这点本身没有问题，反而有助于保持几何自洽。

### 3.4 当前状态是否“足够 Markov”

结论不能简单回答“是”或“否”，更准确的表述是：

> 对环境显式观测子空间而言，这 17 维状态大体覆盖了 Hopper-Hop 的主要运动学、速度与接触信息；但对真正驱动多步 rollout 稳定性的内部闭合而言，`A` 与接触演化仍然带有近似性质，因此当前状态更像“近似 Markov state”，而非严格证明闭合的 state。

---

## 4. 当前 `next(z, a)` 动力学是否具备 TD-MPC2 需要的表达能力

### 4.1 正面一致性

模型时间尺度与环境步长是一致的：

1. DMControl wrapper 每个 agent step 实际执行环境 `2` 次 step，见 [dmcontrol.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/envs/dmcontrol.py:56)
2. `DMControlHopperSDModel` 里也显式设定 `ENV_ACTION_REPEAT = 2`，并使用 `dt * ENV_ACTION_REPEAT` 推进，见 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:49)、[dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:765)

这一点是一个明确优点，至少没有出现时间积分口径不一致的问题。

### 4.2 当前 forward 的主要近似

当前 `forward()` 的主要结构是：

1. 关节加速度由 `action_scale * action - damping * qdot - centering * (q - rest)` 给出；
2. 地面对身体的主作用被压缩为沿腿轴向的单一 `F_leg`；
3. 躯干俯仰由 `tau_bal = -k_theta * theta - k_omega * omega` 平衡项控制；
4. 接触强度由 toe/heel 的几何近地程度与垂向速度近似生成；
5. touch 通过 carry 系数平滑推进。

对应实现位于：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:794)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:802)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:810)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:826)

这说明它确实能表达“动作改变 -> 关节变化 -> 姿态变化 -> 接触变化 -> 根部动力学变化”这条链；因此从接口角度说，它不是一个退化模型。

### 4.3 当前 forward 的结构性风险

尽管它能表达上述链条，但仍存在三类显著近似风险。

#### 风险 1：接触动力学过于简化

当前 touch 的推进只依赖：

1. toe/heel 高度；
2. `vz`；
3. carry 平滑。

这忽略了：

1. 水平速度对触地/滑移的影响；
2. 更复杂的几何穿透与反作用变化；
3. 接触法向与切向的更真实耦合。

若 planner 反复利用接触边界附近的状态，这种简化很容易放大成多步 rollout 偏差。

#### 风险 2：根部动力学对关节动作的耦合较粗

当前 root 线速度与俯仰变化主要靠：

1. `F_leg`
2. `tau_bal`

而不是更完整的多连杆动力学耦合。

这意味着：

1. 模型对“动作如何改变根部姿态”的表达是有的；
2. 但很可能只在粗粒度上成立；
3. 对 planner 而言，最危险的情况不是“完全没响应”，而是“响应方向对，但量纲和耦合不准”。

#### 风险 3：`project_state()` 的硬裁剪可能在规划边界制造非物理吸附

`project_state()` 会对速度、角速度、接触等量做硬裁剪：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:711)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:725)

这会提升数值稳定性，但也可能在 planner 采样到极端动作时产生：

1. 大片状态被压到边界；
2. rollout 局部不可逆；
3. 价值评估对高能动作失真。

因此，这类裁剪对“训练不炸”有帮助，但对“planner 是否获得真实局部梯度结构”仍是风险。

---

## 5. 当前设计对 downstream `reward / Q / pi` 头的可用性

当前 `reward / Q / pi` 头仍是 learned MLP，但它们消费的 latent 已不是黑盒 latent，而是：

1. 纯结构 `17` 维；
2. 或结构 `17` 维 + residual latent。

当设置 `sd_latent_dim=17` 时：

1. `latent_dim` 被直接改成 `17`，见 [parser.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/parser.py:68)
2. `reward / Q / pi` 都只能工作在这 `17` 维结构状态上，见 [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py:76)

这有一个重要含义：

> 一旦选择 `sd_latent_dim=17`，当前实验就在做一个真正更干净也更严格的命题：后续头部不再有 residual latent 可以“偷偷补足”结构状态里遗漏的信息。

这正是当前纯基线实验想要的，但也意味着：

1. 若结构状态设计稍有缺口；
2. 或几何分布不利于 MLP 学习价值函数；

则性能会直接掉下来，而不会被 residual latent 掩盖。

因此，`sd_latent_dim=17` 是一个合适的 clean test，但它不是“更容易成功”的设置，而是“更有诊断价值”的设置。

---

## 6. 当前独立监督拟合脚本实际上验证了什么

独立脚本位于：

- [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py:1)

### 6.1 它已经能验证的内容

当前脚本已经能验证：

1. 在随机动作分布下，`encode(obs)` 与 `forward(state, action)` 的一步拟合是否有效；
2. 在同一 episode 内的真实连续片段上，`1 / 5 / 10` 步 rollout 误差；
3. 若启用 rollout loss，也是在真实连续起点上训练，而不是伪时间序列；
4. 关键结构量误差，如 `height / vx / vz / pitch / support_contact / full_foot_support / body_contact_z`。

对应实现位于：

- [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py:122)
- [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py:148)
- [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py:171)
- [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py:233)

### 6.2 它当前还不能验证的内容

当前脚本仍然不能回答以下问题：

1. 在 planner 真实采样分布上，rollout 是否仍稳定；
2. `20+` 步 horizon 上是否会快速漂移；
3. 动作敏感性是否与环境一致，例如对同一状态做动作 sweep 后，`vx / vz / pitch / support` 响应方向与幅度是否匹配；
4. 仅用结构 `17` 维时，downstream `reward / Q / pi` 是否容易学习；
5. 与原始黑盒 `WorldModel` 相比，哪一类误差更大。

因此，当前脚本是必要的，但远远不是充分的。

---

## 7. 审计结论

### 7.1 已经可以明确成立的结论

1. 当前 `sd_hopper` 已真实接管 TD-MPC2 的 `encode` 与 `next` 两条核心接口，而不是只在诊断分支里存在。
2. 17 维结构状态并非任意拼凑；其大部分维度具有明确物理语义，并覆盖了 Hopper-Hop 的主要运动学、速度与接触信息。
3. `rootx` 的省略在当前任务中是合理的，不构成明显缺陷。
4. `z_min` 作为冗余几何派生量是自洽的，不是主要风险。
5. 当前独立监督拟合链已经足以作为“一步 / 短多步结构拟合”的基础验证工具。

### 7.2 当前最主要的结构性风险

1. `A` 同时被当作观测代理量和递推内部状态，存在语义不一致风险。
2. 接触演化主要由高度与竖直速度近似驱动，缺少更真实的接触-滑移-切向动力学。
3. 根部动力学对关节动作的耦合较粗，可能只在趋势上正确，而不够支持 planner 依赖的精细 rollout。
4. `project_state()` 的硬裁剪虽能稳训练，但可能扭曲高能动作附近的规划几何。
5. 当 `sd_latent_dim=17` 时，downstream heads 没有 residual latent 兜底，任何状态设计缺口都会直接暴露。

### 7.3 当前总判断

综合判断如下：

> 当前 Hopper SD 设计已经具备“作为 TD-MPC2 world-model 接口候选实现”的基本形式条件，但尚没有足够证据证明它已经具备“稳定替代黑盒 world model”的能力条件。

换言之：

1. 现在已经可以合理地做“纯环境奖励、无 residual”的干净实验；
2. 但不能仅凭当前代码结构，就断言它已满足 TD-MPC2 规划所需能力；
3. 真正的决定因素将是后续对多步 rollout、动作敏感性、planner 分布与 downstream 头兼容性的定量验证。

---

## 8. 下一步最小验证建议

建议按以下顺序推进：

1. 先跑独立监督拟合正式实验，主看 `1 / 3 / 5` 步 rollout 指标，并保留 `10 / 20` 步作为漂移诊断。
2. 新增动作敏感性测试：
   固定若干典型状态，对动作做小范围 sweep，比对环境与 SD 在 `height / vx / vz / pitch / support_contact` 上的一步响应。
3. 在 `10k pureenv_noresid` 基线上读取训练日志中的 `struct_consistency_loss`，观察其是否随训练显著下降。
4. 若独立拟合良好而闭环 RL 仍差，再重点检查 `reward / Q / pi` 头在 `17` 维结构 latent 上的适配性，而不是回头继续改奖励。

对应工具入口：

1. 多步 rollout 监督拟合：
   [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py)
2. 一步动作敏感性诊断：
   [analyze_hopper_sd_action_sensitivity.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_sd_action_sensitivity.py)

关于 `vx / pitch_rate` 的专项静态与实证审查，另见：

- [sd_hopper_vx_pitchrate_model_audit.md](/home/data/round/code/sd_world_model_rl/docs/sd_hopper_vx_pitchrate_model_audit.md)

这四步做完，才能对“SD 是否真正具备替代条件”给出更强结论。

## 2026-04-28 再追加：增强 residual 输入后，一步与关键根部量继续改善，但长时 rollout 尚未同步改善

在确认 `contactfix5` 已是当前 grounded 主干最佳口径之后，又进一步参考了既有 `SD + residual` 文档里的边界约束：残差可以读取结构化当前量，但应只吸收主干未解释的局部余量，而不能退化为第二条自由 transition law。

据此，对当前 Hopper transition residual 做了一次低风险增强：

1. 保持输出位置不变，仍只修 `vx / vz / omega` 的一步增量；
2. 保持零初始化、`tanh` 限幅与同一组 `base_scale`；
3. 不再只输入裸 `state + action`，而是额外输入当前结构特征与主干提议量，包括：`joint_acc`、`next_activation`、`toe/heel` 高度、`hip` 几何、`c_leg / d_c_leg / e_leg`、`contact gate / support gate / toe_share / heel_share`、法向/切向接触量、滑移速度，以及主干自己的 `base_delta_vx / vz / omega`。

对应同口径 smoke 结果：

1. 旧版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_cpu/summary.json)
2. 新版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/summary.json)
3. 新版责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/residual_usage_analysis.json)

关键变化如下：

1. `one_step_struct_rmse`：`0.72395 -> 0.71918`
2. `vx_mae`：`0.13342 -> 0.08705`
3. `vz_mae`：`0.17875 -> 0.15467`
4. `pitch_rate_mae`：`0.77846 -> 0.74535`
5. `rollout_10_rmse`：`1.72812 -> 1.73113`
6. `rollout_20_rmse`：`1.64983 -> 1.67611`

因此，这轮结果的最准确解读不是“residual 整体更强了”，而是：

> 当前 residual 的主要缺口之一，确实是输入上下文过贫。把当前接触与主干提议量显式喂给 residual 后，它对一步关键根部量的局部修正能力明显增强，尤其是 `vx`。但这还没有自然转化为更好的长时 rollout 稳定性。

责任分析也支持这一点：

1. `vx` 的 `mean_ratio_resid_over_base` 已收敛到约 `1.52`，说明 residual 对 `vx` 已更接近“围绕主干做补偿”，而不是另起一条大旁路；
2. `vz` 的 `mean_abs_base_delta` 进一步降到约 `0.2719`，说明 grounded 主干本身也仍在回收一部分竖直责任；
3. 但 `omega` 仍表现为 `mean_abs_resid_delta ≈ 0.56`、`mean_ratio_resid_over_base ≈ 7.86`，说明俯仰角速度的长时误差并未因这轮上下文增强而被根治。

因此，当前关于“SD 是否具备替代条件”的判断应再细化一层：

1. `contactfix5 + feature-context residual` 已比旧 residual 版本更适合作为当前活跃基线；
2. 但它仍只能说明“一步局部修正足够有前景”，还不能说明“planner 依赖的短视界 rollout 已经稳”；
3. 对当前 TD-MPC2 Hopper 配置，更合适的主判据应是 `1 / 3 / 5` 步，因为规划与训练 rollout 的真实 `horizon = 3`，而 `10 / 20` 步只保留为漂移诊断；
4. 后续若继续推进，应优先围绕 residual 的 `1 / 3 / 5` 步一致性与 `omega` 通道做针对性约束，而不是再回到主干量纲硬替换。

### 2026-04-28 再追加：只改训练目标的短视界关键通道损失，确实优先改善了 `1 / 3 / 5` 步

随后又做了一条更保守的验证：不改 `SD` 主干、不改 residual 结构，只在监督训练目标中加入短视界关键通道损失。具体地说：

1. 保留原有一步 `structured_state_loss`；
2. 保留原有 `rollout_horizon=5` 的结构化 rollout loss；
3. 额外对 `3 / 5` 步 rollout 的 `vx / vz / omega` 加一个小的加权 `smooth_l1` 附加损失；
4. 其中 `omega` 通道权重略高于 `vx / vz`，因为它仍是当前最顽固的短视界瓶颈。

对应结果：

1. 旧版 `feature-context residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/summary.json)
2. 短视界损失版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss_cpu/summary.json)

关键对比如下：

1. `rollout_1_rmse`：`0.67576 -> 0.67496`
2. `rollout_3_rmse`：`1.18643 -> 1.18081`
3. `rollout_5_rmse`：`1.45345 -> 1.44230`
4. `rollout_10_rmse`：`1.73113 -> 1.71276`
5. `pitch_rate_mae`：`0.74535 -> 0.73885`

但同时也有一个清楚的代价：

1. `vx_mae`：`0.08705 -> 0.09226`
2. `vz_mae`：`0.15467 -> 0.16231`
3. `rollout_20_rmse`：`1.67611 -> 1.68251`

因此，这轮结果的最准确判断是：

> 当前这类短视界附加损失确实会把模型往 TD-MPC2 真正使用的 `1 / 3 / 5` 步区间推，但它不是无代价改进，而是明确地在“短视界一致性”与“单步 `vx / vz` 精度”之间做了再分配。

这说明方向是对的，但权重还不应直接定版。更合理的下一步不是继续加结构，而是先在这一条 loss 线上做小范围权重扫描，找一个既保住 `rollout_3 / 5`，又不过度牺牲 `vx / vz` 的点。

随后又继续做了 `short_rollout_weight = 0.10 / 0.15 / 0.25` 的同口径扫描。结果如下：

1. `0.10`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/summary.json)
2. `0.15`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss015_cpu/summary.json)
3. `0.25`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss_cpu/summary.json)

相对当前无短视界附加损失的 `feature-context residual` 基线，最值得看的量是：

1. `0.10`：`rollout_3 = 1.18643 -> 1.17881`，`rollout_5 = 1.45345 -> 1.43687`，`vx_mae = 0.08705 -> 0.08640`，`vz_mae = 0.15467 -> 0.15638`，`pitch_rate_mae = 0.74535 -> 0.73936`
2. `0.15`：`rollout_3 = 1.18057`，`rollout_5 = 1.43813`，`vx_mae = 0.08542`，`vz_mae = 0.15850`，`pitch_rate_mae = 0.73660`
3. `0.25`：`rollout_3 = 1.18081`，`rollout_5 = 1.44230`，`vx_mae = 0.09226`，`vz_mae = 0.16231`，`pitch_rate_mae = 0.73885`

因此，本轮扫描的结论可以进一步收紧为：

> 若以 TD-MPC2 当前真实使用的 `1 / 3 / 5` 步区间为主判据，`0.10` 是目前最平衡的短视界损失权重点。`0.15` 更偏向继续压 `omega`，但已经开始更明显地伤到 `vz`；`0.25` 则过强。

基于这一点，当前更合适的新监督基线应是：

1. `contactfix5 + feature-context residual`
2. `short_rollout_horizons = [3, 5]`
3. `short_rollout_weight = 0.10`

## 2026-04-28 追加：受限内部残差后的判断更新

在纯结构主干已经完成诊断使命之后，又新增了一条严格受限的 Hopper internal residual 对照：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid1/summary.json)

该对照并未回到“自由 latent residual”或“读出残差”，而是只在 `DMControlHopperSDModel` 内部，对 `vx / vz / omega` 的一步增量加入零初始化、小幅、限幅的 transition residual。结果表明，它相对当前无残差最佳口径 `calib2` 带来了全面改善：

1. 一步整体误差下降；
2. 5/10/20 步 rollout 误差下降；
3. `vx / vz / pitch_rate` 三个关键根部量全部明显下降；
4. 训练耗时未显著劣化。

因此，当前关于“SD 是否具备替代条件”的判断应更新为：

> 对 Hopper 当前这套 17 维状态与显式方程而言，最有前景的路线不再是继续强行维持无残差，而是转向“结构主干 + 受限内部残差”的第二阶段。

这并不等于重新回到黑盒 world model。更准确地说，是：

1. 显式 SD 主干继续承担主要动力学语义；
2. 内部残差只补根部增量上的结构外余量；
3. 后续若接回 TD-MPC2，也应优先沿这条受约束接口接入，而不是恢复自由 struct residual 旁路。

### 关于残差强度的当前证据

在固定同一训练数据规模、训练轮数与监督口径后，已完成 `0.5x / 1.0x / 2.0x` 三个 residual scale 的首轮扫描。结果表明，关键指标在该区间内呈清晰单调改善，且 `2.0x residual` 当前最好：

1. `one_step_struct_rmse = 0.60854`
2. `rollout_20_rmse = 1.00090`
3. `vx_mae = 0.06182`
4. `vz_mae = 0.13085`
5. `pitch_rate_mae = 0.63742`

因此，当前阶段已可以排除一种常见反对意见，即“受限残差只能做微弱补丁，稍微放大就会破坏主干语义”。至少在目前这版 Hopper transition residual 上，`0.5x -> 2.0x` 的放大并未表现出过冲迹象。
