# 4 方法 (Method)

为了解决泛用型隐空间世界模型（如 Dreamer 等模型使用的 RSSM）在面临复杂物理系统规划时所遭遇的数据低效与分布外 (OOD) 泛化盲区，我们提出了 **SD-MCTS (System Dynamics Monte Carlo Tree Search)**。本节将详细阐述其核心组件：首先，我们通过系统动力学方程（System Dynamics Equations）构建具有硬性质量/守恒律的多步转移模型，代理只需在此基础上在线辨识低维的前向动态系数；其次，在决策阶段，我们依靠截断的 MCTS 展开，配合离线学习的终端价值网络以实现长时间跨度的梯阶效应。

## 4.1 符号系统与框架概览

考虑一个离散时间的马尔可夫决策过程（MDP）$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$，其中状态空间为连续向量 $\mathcal{S} \subseteq \mathbb{R}^{d_s}$，动作空间取决于具体应用（如离散调度选择 $\mathcal{A}$ 或受限的连续流派调配），$\mathcal{P}(s'|s,a)$ 为具有延迟时效或隐含缓冲区的环境推演动态，我们的目标在于找到策略以最大化长期折扣回报 $\mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$，其中 $r_t = \mathcal{R}(s_t, a_t)$。

SD-MCTS 的体系架构主要由紧密耦合的三个核心流程构成：
1. **世界结构先验与微分辨识 (Structural Prior and Differentiable Identification)**：放弃让神经网络从白板（Tabula Rasa）内隐地学习未知的流形变换 $\mathcal{P}$，我们引入了一个可导差分方程组模型（基于守恒流量定律）$\hat{\mathcal{P}}_{\theta}$。代理通过在线体验历史数据，利用系统辨识（System Identification）技术通过反向传播来拟合标量参数集合 $\theta$。
2. **离线价值逼近 (Offline Value Approximation)**：通过时序差分（TD）更新，维护一个 off-policy 的神经网络价值评估器 $V_\psi(s) \approx V^\pi(s)$。
3. **基于守恒状态流形的树搜索 (Conservation-Constrained Tree Search)**：在线决策时，智能体运行 MCTS。其所有的模拟状态展开均被 $\hat{\mathcal{P}}_{\theta}$ 锁死在符合物理规则（如库存非负、流量连通）的拓扑中，并在探索深度 $H$ 的叶节点利用 $V_\psi(s_{t+H})$ 进行无穷视野代偿。

## 4.2 具有物理一致性的结构化推演模型 (Physics-Informed Structural World Model)

现有的转移核近似（如通过自回归前馈以逼近多步联合分布）极易因雅可比矩阵的连乘而诱发预测方差的长视距爆发。作为对比，结构化推演迫使模型的隐状态更迭受限于显式的物理保序设计。
我们将真实环境天然地解耦为两部分：**已知的微分流量结构（Hard Guardrails/Constraints）** 与 **受隐状态/随机误差驱动的未知转移率（Learnable Variables）**。

### 4.2.1 守恒方程结构注入 (Stock-Flow Conservation Injection)

以包含隐藏疲劳衰减属性的任务调度 MDP 举例：
令连续状态 $s_t = (q_t, b_{1,t}...b_{n,t}, f_t)$ 分别表征总排队队列、各分支执行缓冲、累积隐藏疲劳。控制输入动作 $a_{i,t}$ 表征向分支 $i$ 的任务分流率。
在物理空间中其一步转移（离散欧拉更新）被严格固定为质量守恒的 Stock-flow（库存-流量）形式：
$$
q_{t+1} = \max\left(0, q_t + \lambda_t - \sum_i a_{i,t}\right) \quad \text{(流量入口平衡)} 
$$
$$
b_{i,t+1} = \max\left(0, b_{i,t} + a_{i,t} - \eta(f_t) \mu_i\right) \quad \text{(缓冲流释放)}
$$
$$
f_{t+1} = (1-\rho) f_t + \kappa \sum_i (\eta(f_t) \mu_i) \quad \text{(隐含退化堆积)}
$$

上述一阶差分计算图中，没有任何黑盒神经网络介于 $s_t$ 与 $s_{t+1}$ 的转换。取而代之的是，系统只对参数构成的**低维隐可微系集 $\theta$** 进行梯度对齐，例如未知的环境抵达生成率 $\lambda_t$、退化乘数偏置 $\kappa,\rho$、工作站的基础产能上限 $\mu_i$。而非线性退化激活函数则由参数化曲线如 $ \eta(f; \alpha, \tau) = \frac{1}{1 + \alpha \cdot \max(0, f - \tau)^2} $ （包含可学习的容忍度标量 $\alpha, \tau$）构成。

### 4.2.2 微分图上的参数推断与误差约束 (Differentiable System Identification)

由于整个物理转移操作图 $\hat{\mathcal{P}}_{\theta}$ 完全由 PyTorch 等深度学习框架的张量运算原生构建（连续可微），我们将世界模型的学习彻底退化为参数的重构寻优：
$$ \mathcal{L}_{\text{model}} = \mathbb{E}_{(s,a,s') \in \mathcal{D}} \left[ \|s' - \hat{\mathcal{P}}_{\theta}(s, a)\|^2 \right] $$
这种降维解构为 6.3 节所述的样本效率提升提供了底层逻辑。更重要的是，哪怕在极低数据区间训练不充分（$\theta$ 存在较大偏差引发流速错估），结构上的硬编码保证了所有的微积分流转符合物理世界的底线前提（如非负算子 $\max(0,\cdot)$ 直接保证了不产生"负库存用于虚假交货"等破坏 MDP 定义的状态）。这从结构上截断了轨迹展开时的累加飞散效应（Divergence），是限制多步联合展开预测呈现 $O(1)$ 误差上界的理论根基。

## 4.3 长视野规划的有效代偿 (Horizon Compensation via Latent Value Alignment)

在面临延迟回报（如因超载导致 20 步后的系统崩溃）的全景任务中，MCTS 极易掉入深度指数爆炸（$\mathcal{O}(|\mathcal{A}|^H)$）的运算陷阱，这也是纯粹的 MPC（模型预测控制）长久以来的痛点。受近似动态规划（ADP）算法体系启发，我们向具有短视偏置（Truncation Bias）的 SD-MCTS 接入深度价值对齐模块。

**在线神经网络前瞻融合**
当 MCTS 搜索到达人为选定的短视野叶节点 $H$ 时，我们截断模型展开动作，并依赖离线学习得到的近似势能 $V_\psi$ 进行注能平替：
$$ G \leftarrow \sum_{k=0}^{H-1} \gamma^k r_{t+k} + \gamma^H V_\psi(s_{t+H}) $$
该价值预估网络在后端随着经验池异步迭代：$\mathcal{L}_{\text{val}} = \mathbb{E}_{\mathcal{D}} \left[ \|V_\psi(s_t) - (r_t + \gamma \bar{V}_\psi(s_{t+1}))\|^2 \right]$（采用 Target Network $\bar{V}_\psi$ 以稳定 TD 目标）。
在价值估计充分准确的极限条件下，这种组合相当于令 MCTS 的展开深度实现了 $H \to \infty$ 的行为等效。结合物理图结构限制短期轨迹飞离的功能，该范式真正做到了短平快在线搜索和稳健的长尾风险对齐。

## 4.4 因果图上的显式零样本干预 (Explicit Graph Editing for Zero-Shot Formulation)

对于通用多层感知器（MLP）或注意力体系所组装的世界模型，如果发生环境内在属性（或拓扑网络）的突发性更替，因为无法直接干预网络权重矩阵中的蕴含法则，智能体将对突变事件产生灾难性盲区。
而系统动力学模型具有与真实控制图的高度结构同构性（Structural Isomorphism）。

假设在时间 $t=T_{\text{shock}}$ 给定一外部**可观测的图级干预指标**，例如图节点编辑 $do(\mu_2 \leftarrow 0)$ 表征某特定产线的瘫痪，或者路径流向斩断。
对于该情形，SD-MCTS **无需在线数据收集、也无须触发网络重训 (Zero-shot)**。代理只需同步在其转移模型 $\hat{\mathcal{P}}_{\theta}$ 的图节点属性上覆盖此干预映射。在随后的线上决策生成周期里，这一定点物理变动将通过图架构的前馈，自发且精确无误地传导（Propagation）至系统的所有宏观关联变量中。
> *注释说明：尽管依赖离线历史经验的价值网络 $V_\psi(s)$ 在图干预后会发生评估失准（因旧策略价值失效），但我们实验观测到：只需借助 MCTS 放任干预模型在线试错探索的结构矫正力（通过短暂的 $H$ 视野搜索去覆盖长期偏见），无需特化抵御即可在此阶段极大削减策略溃败风险。* 
