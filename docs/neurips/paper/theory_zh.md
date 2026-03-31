# 5 理论结果 (Theoretical Analysis)

为了深入解释结构化因果流形（Structured Causal Manifold）如何赋予世界模型显著的数据降维与跨分布鲁棒性，本节提供了统一的理论分析依据。由于真实强化学习场景常偏离 i.i.d. 等理想条件，本节的核心结论将以解释性命题（Propositions）和误差上界（Bounds）的形式呈现。详细的机制推导见补充附录 A.5。

## 5.1 结构化推断的样本复杂度 (Sample Complexity of Structured Inference)

通用黑盒 MBRL 模型的失效从本质上源于需要从海量试错（Trial and Error）中无约束地逼近未知的联合状态转移矩阵。相比之下，SD-MCTS 利用系统动力学偏导矩阵的物理屏蔽截断了无效特征空间，从而实现了极低样本体制下的参数拟合。

**命题 1：先验结构下采样的非正式样本复杂度界**（Sample Complexity Gap strictly under Structural Prior）
假设非线性转移受控于隐含的差分约束系统 $s' = \hat{\mathcal{P}}(s, a ; \theta^*) + \xi$，其中 $\hat{\mathcal{P}}$ 为拥有确切拓扑的偏微分算子，物理参数 $\theta^* \in \mathbb{R}^d$（$d$ 为低维状态量），且观测噪声遵循 $\sigma$-次高斯分布 ($\|\xi\|_{\psi_2} \leq \sigma$)，优化目标为强凸二次损失。

在理想的独立同分布 (i.i.d.) 采样及特征矩阵持续激励（Persistent Excitation）假设下，使得参数预测误差收敛至 $\epsilon$ 所需的样本集大小下界具有紧确界度：
$$ N_{\text{SD}} = \Omega\left( \frac{d}{\epsilon^2} \right) $$

相比之下，若对物理约束不作任何先验假定（即辨识整个联合状态-动作体积 $|\mathcal{S}| \times |\mathcal{A}|$ 的无偏估计），所需的次优解样本下界膨胀至 $\Omega\big( \frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^2} \big)$。此结论从模型维度解析了 **C1（样本效率）** 中观察到的百倍级效率差距：结构辨识将随状态空间呈组合爆炸的学习任务，严格退化为了与状态空间几何大小无关的低维标量标定过程。

## 5.2 抑制 OOD 散逸：从指数级复合误差到守恒界度 (Bounding Compounding Errors via Conservation)

在离散时间点进行多步展开是 Model-based 规划器的传统灾区，在缺乏物理不变量的自回归生成中，由于单步雅可比（Jacobian）放大效应，多步累积会呈现指数级的发散（Compounding Errors）。

**命题 2：质量守恒条件下的预测误差线性/恒定边界** (Error Bounds under Conservation Constraints)
考虑多步推演残差的期望 $\mathbb{E}[\|s_H - \hat{s}_H\|]$。对于无结构约束的全连接网络 $f_{\text{MLP}}$，假设局部 Lipschitz 常数 $L_f > 1$ 且单步估计存在扰动 $\epsilon_{\text{model}}$，其 $H$ 步展开误差遵循指数爆炸：
$$ \text{Error}_{\text{MLP}} \le \frac{L_f^H - 1}{L_f - 1} \cdot \epsilon_{\text{model}} = \mathcal{O}(L_f^H) $$

而在具有闭集（Closed-set）性质的系统动力学约束下，如果状态流形通过守恒流动（Mass Balance）及非负包络收缩算子（例如库存 $q \ge 0$ 的物理投影截断算子 $\max(0, \cdot)$）严格维持在不变集内。这使得结构化模型的流形梯度被截断（即在多数流形边界 $L_f \le 1$），则系统预测差异随时间的累加被刚性钳制在线性范畴甚至上确界极限：
$$ \text{Error}_{\text{SD}} \le H \cdot \epsilon_{\text{model}} = \mathcal{O}(H) $$

当系统自带强阻尼项或具备稳定吸引子（如老化、自然衰退），其结构算子具备收敛映射属性。以我们在第 6 节实验中所观测的系统动力学环境为例，其经验 Lipschitz 常数被测定为满足 $L_f < 1$，这使得多步绝对展开误差直接被限制在了 $\mathcal{O}(1)$ 的常量界度内。这一理论推演不仅解释了我们在 **C2 多步预测实验** 中为何能将长期预测偏差控制在极低的常量（例如 SD 误差未见显著增长），还深刻对比了 BlackBox 由于网络缺乏结构约束而遭遇的持续累加（约 6.35%/步 的误差叠加跃升）。

## 5.3 长视野截断补偿原理 (Addressing Horizon Truncation via Value Alignment)

如果推演过程是稳定可控的，为何我们不利用 SD 模型无限规划下去？这是由于树搜索算法的复杂度呈 $\mathcal{O}(|\mathcal{A}|^H)$ 随深度剧增。我们进而分析如何通过学习离线的近似值函数 $V_\psi$ 去抵消极深视野所带来的延迟变量惩罚。

**命题 3：截断偏置与参数精确度的视界折中** (The Horizon-Accuracy Trade-off)
设立一个触发延迟时效为 $\tau$ 步的控制任务。作为对比标杆，我们设立 **① Oracle MPC**：拥有真实参数 $\theta^*$ 但无终端价值（$V=0$）探索深度至 $H$。由于其强制末端截断，必然背负隐去偏见 (Truncation Bias) $\approx \frac{\gamma^H R_{\max}}{1-\gamma}$。

设 $\hat{\theta}$ 为基于有限观测数据产生的系统参数估计量，且满足参数估计误差边界 $\|\hat{\theta} - \theta^*\| \leq \delta_\theta$。
相对地，我们引入 **② SD-MCTS**：它使用带偏推断参数 $\hat{\theta}$，虽然规划（展开）深度同样被截断限制为 $H$，但凭借离线价值评估网络的末端补偿，获得了精度为 $\|V_\psi - V^*\|_\infty \leq \epsilon_V$ 的前瞻终端势能。

代理 **(2) SD-MCTS** 相比 **(1) 完美参数 Oracle** 产生的状态决策总体回报差（即 Value Prediction Error）满足更优上确界，当且仅当：
$$
\underbrace{L_\theta \delta_\theta}_{\text{SD 单步模型传递偏置}} + \underbrace{\gamma^H \epsilon_V}_{\text{折扣后的终端价值偏置}} < \underbrace{\frac{\gamma^H R_{\max}}{1-\gamma}}_{\text{Oracle 的绝对截断盲区}}
$$
*(其中 $L_\theta$ 是累积奖励关于环境参数的映射 Lipschitz 常数).*

**直觉启示**：这一界度不等式明确了"参数 + 短可视窗 + 终端价值"设计对于解决复杂系统长线规划具备极大优势——即便离线拟合的网络 $V_\psi$ 和经验推估量 $\hat{\theta}$ 均带有不完美残差（等式左边），但只要问题的核心诱发时长 $\tau$ 大于搜索窗口 $H$，等式右方由于绝对视距截断而引入的盲区界 $\frac{\gamma^H R_{\max}}{1-\gamma}$ 就依然占主导。在这种情况下，不追求极深的探索树（放弃深达 $\tau$ 步的指数级计算），转而将计算资源向单步结构推演出让，并由价值对齐器作 $H=1$ 的末端全景补偿，是缓解极高延迟控制痛点的可行策略。

## 5.4 规划质量的根本差异：Q 估计的不可约偏差 (Irreducible Bias in Q-Estimation)

三个命题共同揭示了 SD-MCTS 相对于黑盒 MCTS 在规划质量上的一个更深层次的差异：

**命题 4：黑盒世界模型的 Q 估计含有不可约偏差**

设 $\hat{Q}_{n}(s, a)$ 为 MCTS 通过 $n$ 次 rollout 对动作 $a$ 构建的 Q 估计量，$Q^{\pi}_H(s, a)$ 为真实 Q 值。Q 估计误差分解为：
$$\hat{Q}_{n}(s, a) - Q^{\pi}_H(s, a) = \underbrace{\text{bias}(s, a)}_{\text{模型结构误差（确定性）}} + \underbrace{\text{noise}/\sqrt{n}}_{\text{随机方差（可消除）}}$$

对于黑盒 MLP 模型（Lipschitz 常数 $L > 1$），其内部预测状态的偏差随规划深度 $H$ 指数增长（见命题 2），导致：
$$|\text{bias}_{\text{BB}}(s, a)| = \mathcal{O}\bigl((\gamma L)^H\bigr)$$

**此偏差与模拟次数 $n$ 无关**——即使增加模拟次数趋于无穷，$\text{bias}_{\text{BB}}$ 也不会收缩。这是一个不可约的（irreducible）系统性误差。

相比之下，SD-MCTS 的守恒律约束保证了其世界模型的预测状态物理一致，在确定性环境下：
$$|\text{bias}_{\text{SD}}(s, a)| \approx 0$$

**推论（效率差距的根因）**：SD-MCTS 样本效率优势的本质并非运气或超参数调节，而是 $\text{bias}_{\text{SD}} = 0$ 的结构保证使其 MCTS 能够构建精确的 Q 估计，从而做出更优的动作选择。BB-MCTS 即便在足够计算预算下，也受限于不可约偏差 $\mathcal{O}((\gamma L)^H)$，这直接体现为 Table 1 中 10K 步后 BB 的规划回报仍逊于 1K 步时的 SD-MCTS 的现象。完整的 Bias-Variance 分解推导见附录 A.5。
