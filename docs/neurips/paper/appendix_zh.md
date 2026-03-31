# 附录 A: 实验细节与复现指引 (Appendix: Experimental Details and Reproducibility)

为了确保本研究所述结论的有效性和可复现性（Reproducibility），本附录详细列出了各基准环境的拓扑参数、SD-MCTS 及基线算法的网络架构与超参数配置，以及由于主文篇幅限制而引用的扩展实验分析及其图表说明。

## A.1 实验环境设计细节

**(1) 四级啤酒游戏供应链 (Beer Game)**
*   **物理拓扑**：供应链包含依次相连的工厂 (Factory)、分销商 (Distributor)、批发商 (Wholesaler) 和零售商 (Retailer)。终端顾客的需求仅在零售商处可被观测。
*   **守恒定律**：每个节点 $i$ 包含库存状态 $q_i$ 与积压状态 $b_i$。满足典型的存量-流量（Stock-Flow）物料守恒等式。发货受限于本级实际库存。
*   **状态向量**：$s \in \mathbb{R}^{12}$，包含了四要素节点的库存量、积压量与在途订单量。
*   **延迟参数 $\tau$**：上下游节点的订单传递延迟固定为 2 步，物流交付延迟固定为 2 步。这意味着任意一项生产改变需要至少纵贯 8~10 个时间步才能送达终端需求。
*   **OOD 突变设定**：在第 $t=50$ 步时，将“分销商”到“批发商”的通信运输容量硬性截断 50%（Capacity Shock），或完全瘫痪某连接链路（Comm Severed），此图结构突变在训练期内完全不可见。

**(2) 延迟崩溃场景 (Delayed Collapse)**
*   **核心逻辑**：基于典型的设备退化或人力疲劳调度模型。除了可见的工作站排队数，存在一个不完全可观测的隐藏态 `Fatigue`（疲劳度）。
*   **动态激活**：当总派发任务 $a_t$ 长期处于高位时，隐藏状态的积累呈指数级激增，致使 $T+20$ 步后的产能缩水 90%。该环境检验了单纯缩减视野会导致“寅吃卯粮”的算法短视缺陷。

**(3) 流行病控制图 (SEIR Epidemic)**
*   **连续图结构**：具有非线性接触率的状态传递图 $S \to E \to I \to R$。代理负责输出隔离与干预因子（动作 $a$ 为一个实数分配向量）。环境内在蕴含着接触感染乘数的非线性激增与医疗挤兑拐点。

## A.2 Lipschitz 收缩参数的实测验证

为呼应正文（定理 2 命题 2）提出的“收敛映射使得多步展开误差约束至 $O(1)$ 常量”的论断，我们独立统计了在各个环境中所习得的系统动力学方差的有效经验 Lipschitz 常数 $\hat{L}_f$：
*   **Beer Game**: $\hat{L}_f \approx 0.96$ （主要由于商品交付与需求消耗带来的收敛阻尼）。
*   **Delayed Collapse**: $\hat{L}_f \approx 0.88$ （疲劳指数具备指数递减的自然衰退特性）。

上述测算在物理层面验证了环境差分图具有 $L_f < 1$ 的理论前提条件，进而在实际的多步前瞻中，保证了轨迹轨线偏差不被无限放大，从而为极度确定的控制奠定了信心。

## A.3 时序预测残差增长曲线 (Supplementary Figures Analysis)

由于正文排版限制，各模型在预测 1~15 步时的绝对积累误差（RMSE）时序演变图在此作补充阐述。这也是支撑第 6.3 节核心结论的基石：
**图 2（补充时序方差展宽图）**分析：
*   **x 轴**：模型执行自我回归展开的预测视界（Prediction Horizon $H \in [1, 15]$）。
*   **y 轴**：模型内隐状态与真实世界引擎在此阶段后的绝对 RMSE 差距。
*   **机制结论**：这个实验图表打破了长期困扰 MBRL 的复合不确定性（Compounding Uncertainty）刻板印象，有力地实证了使用微小白盒约束去压制大型黑盒推断的投资回报是决定性的。

## A.4 实验复现指引 (Reproducibility Statement)

为了支持本文所有实证主张的独立公开验证，我们在随附的补充材料目录中提供了完整的源代码仓库架构。
所有的核心结果均可以通过执行预设代码流水线产生（即产生文中所有评估表格及可视化曲线图表）：
* 核心实验命令入口涵盖：
  * **主结果（样本效率与表 1）**：`python scripts/run_sample_efficiency.py --env beer_game`
  * **误差增长率分析（图 2）**：`python scripts/run_multi_step_error.py --weights pretrained_models/`
  * **分布外因果参数阻断响应（表 3与图 3）**：`python scripts/run_causal_intervention.py --shock_type capacity`

源码库内包含了预构建好的模型架构超参数配置字典，并固化了用于控制变量测试的所有随机种子池，以原汁原味地还原所有定量比较的准确环境。

## A.5 Q 估计的 Bias-Variance 分解推导 (Bias-Variance Decomposition of Q-Estimation)

本节补充 §5.4 命题 4 的完整推导。

**设定**：MCTS 对动作 $a$ 进行 $n$ 次独立 rollout，每次从状态 $s$ 出发，利用世界模型 $\hat{T}$ 展开 $H$ 步，累积折扣奖励后加终端估值：

$$\hat{R}_i(s,a) = \sum_{t=0}^{H-1} \gamma^t r(\hat{s}_t, \hat{a}_t) + \gamma^H V_\psi(\hat{s}_H)$$

Q 估计量为 $n$ 次 rollout 的均值：$\hat{Q}_n(s,a) = \frac{1}{n} \sum_{i=1}^n \hat{R}_i(s,a)$。

设真实 Q 值为 $Q^\pi_H(s,a)$，误差分解为：

$$\hat{Q}_n(s,a) - Q^\pi_H(s,a) = \underbrace{\mathbb{E}[\hat{R}] - Q^\pi_H}_{\text{bias（系统性误差）}} + \underbrace{\frac{1}{n}\sum_i (\hat{R}_i - \mathbb{E}[\hat{R}])}_{\text{noise（随机方差，}\sim 1/\sqrt{n}\text{）}}$$

**命题 4 的关键推导**：

**（一）BlackBox 模型的 bias 来自状态误差的级联放大。** 由 §5.2 命题 2，BB 模型的 $t$ 步状态预测误差满足：

$$\|\hat{s}_t^{\text{BB}} - s_t\| \leq \epsilon_1 \cdot L^{t-1}, \quad L > 1$$

设奖励函数和终值函数的 Lipschitz 常数分别为 $c_r, c_V$，则：

$$|\text{bias}_{\text{BB}}| \leq \sum_{t=1}^{H-1} \gamma^t \cdot c_r \cdot \epsilon_1 L^{t-1} + \gamma^H \cdot c_V \cdot \epsilon_1 L^{H-1} = \mathcal{O}\bigl((\gamma L)^H\bigr)$$

当 $\gamma L > 1$ 时，偏差随 $H$ 指数增长。**注意**：bias 是世界模型预测误差的确定性函数，与 rollout 次数 $n$ 完全无关——增加 $n$ 只能消除 noise 项，无法减小 bias。

**（二）SD 模型的 bias 在守恒律精确时趋于零。** 由于 SD 模型的守恒分量（库存-流量平衡）被精确硬编码，$\hat{s}_t^{\text{SD}}$ 的守恒分量与真实状态一致，状态误差仅来自外生输入（如需求预测），且不会级联放大（Lemma: $L_f \leq 1$）：

$$|\text{bias}_{\text{SD}}| \leq \frac{\epsilon_{\text{exogenous}}}{1 - \gamma} = \mathcal{O}(1)$$

在确定性环境且守恒律精确时，$\epsilon_{\text{exogenous}} \approx 0$，即 $\text{bias}_{\text{SD}} \approx 0$。

**推论**：给定相同的计算预算（$n$ 次模拟），SD-MCTS 获得接近无偏的 Q 估计，而 BB-MCTS 的 Q 估计含有不可约偏差 $\mathcal{O}((\gamma L)^H)$，这直接导致动作选择质量的系统性差距，与 Table 1 和 Table 3 的实验结果一致。
