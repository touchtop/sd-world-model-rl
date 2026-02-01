根据您提供的来源（包括G-Sim论文和RL综述论文），以下是关于所有提及的“啤酒游戏（Beer Game）”实验设计的详细梳理说明文档。

---

# 啤酒游戏（Beer Game）实验设计详细说明文档

本文档基于提供的来源资料，梳理了不同研究中关于啤酒游戏的实验设计、系统架构、参数设置及研究目标。啤酒游戏是供应链管理中的经典模拟实验，主要用于展示需求波动在多级供应链中被放大的现象（即“牛鞭效应”）。

## 1. 基于 G-Sim 的单阶段零售商实验设计
**来源**：Source 1 (G-Sim: Generative Simulations...)

该实验将啤酒游戏简化为一个**单阶段零售商（Single-stage Retailer）**模型，用于评估 G-Sim 框架在生成具有部分可观测性和随机性模拟器方面的能力。

### 1.1 系统架构与核心要素
*   **供应链层级**：单级（仅关注零售商与上游供应商及下游客户的交互）。
*   **核心机制**：基于 Sterman (1989) 的经典设定，包含库存管理、积压订单（Backlog）和运输在途（Pipeline）的延迟机制,。
*   **部分可观测性（Partial Observability）**：零售商只能看到当前库存和积压订单，无法直接观测到管道中未来的到货情况或上游的具体状态,。

### 1.2 状态空间 (State Variables)
在每个离散时间步 $t$，系统状态包含以下变量：
1.  **库存 (Inventory)**：当前手头的库存数量（非负整数）。
2.  **在途管道 (Pipeline)**：运输中的货物列表，每个条目包含 `(数量, 剩余到达时间)`。
3.  **积压订单 (Backlog)**：前期未满足的需求累积量。
4.  **时间 (t)**：当前的时间索引。

### 1.3 动作空间 (Actions)
*   **订货量 ($a_t$)**：代理（零售商）在时间 $t$ 决定向上游订购的数量（$a_t \ge 0$）。
*   **延迟机制**：下达的订单不会立即到达，而是进入管道，经过固定的提前期（Lead Time）后才会变成库存。

### 1.4 环境参数与动力学
*   **参数设置**:
    *   $\lambda_{demand}$：泊松分布的平均每日需求。
    *   $c_{holding}$：库存持有成本。
    *   $c_{backlog}$：缺货积压成本。
    *   $L_{lead}$：确定性的提前期（Lead Time）。
*   **状态更新逻辑**,:
    1.  **需求采样**：从泊松分布 $Poisson(\lambda_{demand})$ 中抽取当日需求 $D_t$。
    2.  **更新管道**：所有在途货物的剩余时间减1；剩余时间为0的货物计入库存。
    3.  **处理积压**：优先用现有库存满足之前的积压订单。
    4.  **满足当日需求**：用剩余库存满足 $D_t$；未满足部分加入积压。
    5.  **下达新订单**：代理采取动作 $a_t$，订单进入管道，剩余时间设为 $L_{lead}$。

### 1.5 特定实验场景
G-Sim 研究中设计了两个具体的干预实验来测试模拟器的泛化能力：
1.  **资源优化（Resource Optimization）**：通过改变额外仓库容量（$\Delta C$）和提前期（$\ell$），观察总成本的变化热力图。结果显示 G-Sim 生成的模拟器能准确捕捉容量扩张与运输延迟之间的权衡。
2.  **分布外泛化（OOD Generalization）**：在训练时使用较短的提前期（如 $\ell=1,2,3$），在测试时引入未见过的较长提前期（如 $\ell=4,5,6$）。结果表明，随着提前期增加，积压订单会出现剧烈波动，G-Sim 能够复现这种未在训练数据中出现的动态特性,,。

---

## 2. 多级供应链协调实验设计（基于强化学习综述）
**来源**：Source 2 (Reinforcement Learning for Logistics...)

该来源回顾了多项将强化学习应用于多级供应链协调的研究，这些研究通常使用更复杂的四级或五级啤酒游戏模型。

### 2.1 四级供应链库存管理 (Chaharsooghi et al., 2008)
*   **供应链层级**：四级（Four-tier SCM）,。
*   **方法**：表格型 Q-Learning (Tabular QL)。
*   **实验设计**：
    *   **代理设置**：每层级有一个决策者。
    *   **策略规则**：应用了 $(X+Y)$ 订货规则，其中 $X$ 是下游当前的订货量，$Y$ 是代理通过学习决定的动作。
    *   **目标**：优化供应链的订货管理应用。

### 2.2 五级啤酒分销游戏 (Fuji et al., 2018)
*   **供应链层级**：五级结构，明确包含：工厂 (Factory)、分销商 (Distributor)、批发商 (Wholesaler)、零售商 (Retailer) 和客户 (Customer)。
*   **方法**：多智能体强化学习 (Multi-Agent RL)，使用深度神经网络权重进化 (DNN-weight evolution),。
*   **目标**：通过优化各层级的订货策略，提升整个供应链的绩效。

### 2.3 深度Q网络库存优化 (Oroojlooyjadid et al., 2021)
*   **供应链层级**：四级啤酒游戏 (Four-tier beer game)。
*   **方法**：塑形奖励深度 Q 网络 (Shaped Reward Deep Q Network, SRDQN)。
*   **实验设计**：
    *   **混合策略环境**：由于多智能体系统的复杂性，实验设计为**每次只设定其中一个参与者为学习型智能体 (Learning Agent)**。
    *   **对手策略**：其他的参与者（Co-players）采用基本库存策略 (Base-stock) 或 **Sterman 策略**。Sterman 策略用于模拟人类在啤酒游戏中的非理性行为（即模拟真实人类决策者）。
    *   **状态输入**：库存水平、在途订单等信息通过重放缓冲区 (Replay Buffer) 进行学习。
    *   **目标**：最小化自身及整个系统的成本（包含缺货成本和存储成本）。结果显示该方法能显著降低系统总成本，即使在与模拟人类行为的对手博弈时依然有效,。

---

## 3. 实验设计对比总结

| 特征 | G-Sim (Source 1) | Chaharsooghi et al. (2008) | Fuji et al. (2018) | Oroojlooyjadid et al. (2021) |
| :--- | :--- | :--- | :--- | :--- |
| **层级结构** | **单级** (零售商) | **四级** | **五级** (含客户节点) | **四级** |
| **主要目标** | 模拟器生成与泛化测试 | 供应链订货管理 | 供应链整体绩效优化 | 库存优化与成本降低 |
| **核心算法** | G-Sim (结合LLM与进化策略/SBI) | Tabular Q-Learning | Multi-Agent RL (DNN进化) | SRDQN (Shaped Reward DQN) |
| **特殊设定** | 引入未知提前期($\ell$)进行压力测试 | 使用 $(X+Y)$ 订货规则 | 包含工厂到客户的全链路 | 混合策略：AI vs. Sterman(模拟人类) |
| **关注现象** | 牛鞭效应、OOD泛化能力 | 订货策略有效性 | 多智能体协作 | 在人类行为环境下的鲁棒性 |

**注**：
*   **OOD (Out-of-Distribution)**：分布外数据，指训练时未见过的情境（如更长的运输延迟）。
*   **Sterman 策略**：指 Sterman (1989) 研究中总结出的、导致牛鞭效应的典型人类次优决策模式。
*   
这是一个更新后的文档部分，包含了上述啤酒游戏实验设计中涉及的所有相关文献的详细书目信息。这些信息提取自您提供的来源文档的参考文献列表。

---

# 啤酒游戏实验设计相关文献详细信息列表

以下文献对应于前述“啤酒游戏实验设计说明文档”中提到的各个研究案例。

## 1. G-Sim 框架与单阶段零售商模型
**来源文档**：Source 1
该文献提出了 G-Sim 框架，并使用单阶段啤酒游戏（零售商模型）作为评估模拟器生成能力和分布外（OOD）泛化能力的基准环境。

*   **文献引用**：
    Holt, S., Luyten, M. R., Berthon, A., & van der Schaar, M. (2025). **G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration**. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, Vancouver, Canada. PMLR 267.

## 2. 供应链强化学习综述
**来源文档**：Source 2
该文献回顾了强化学习在物流和供应链管理中的应用，其中包括对啤酒游戏相关研究的分类讨论。

*   **文献引用**：
    Yan, Y., Chow, A. H., Ho, C. P., Kuo, Y. H., Wu, Q., & Ying, C. (2022). **Reinforcement Learning for Logistics and Supply Chain Management: Methodologies, State of the Art, and Future Opportunities**. *Transportation Research Part E: Logistics and Transportation Review*, 162, 102712.,

## 3. 具体啤酒游戏实验研究文献
以下文献在 Source 2 的综述部分被详细讨论，分别代表了不同类型的啤酒游戏实验设计。

### 3.1 四级供应链与 Q-Learning (Chaharsooghi et al., 2008)
该研究在四级供应链中使用 Q-Learning，并结合 $(X+Y)$ 订货规则进行优化。

*   **文献引用**：
    Chaharsooghi, S. K., Heydari, J., & Zegordi, S. H. (2008). **A Reinforcement Learning Model for Supply Chain Ordering Management: An Application to the Beer Game**. *Decision Support Systems*, 45(4), 949–959.,

### 3.2 五级供应链与多智能体 DNN 进化 (Fuji et al., 2018)
该研究构建了包含工厂、分销商、批发商、零售商和客户的五级模型，使用深度神经网络（DNN）权重进化方法。

*   **文献引用**：
    Fuji, T., Ito, K., Matsumoto, K., & Yano, K. (2018). **Deep Multi-Agent Reinforcement Learning using DNN-Weight Evolution to Optimize Supply Chain Performance**. *Proceedings of the 51st Hawaii International Conference on System Sciences*.,

### 3.3 深度 Q 网络与库存优化 (Oroojlooyjadid et al., 2021)
该研究提出了 SRDQN（Shaped Reward Deep Q Network），在四级啤酒游戏中训练智能体与模拟人类行为（Sterman 策略）的对手进行博弈。

*   **文献引用**：
    Oroojlooyjadid, A., Nazari, M., Snyder, L. V., & Takáč, M. (2021). **A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization**. *Manufacturing & Service Operations Management*.,

### 3.4 基础理论文献 (Sterman, 1989)
这是啤酒游戏及其产生的“牛鞭效应”的经典行为学研究，被上述大多数研究所引用作为环境动力学的基础。

*   **文献引用**：
    Sterman, J. D. (1989). **Modeling managerial behavior: Misperceptions of feedback in a dynamic decision making experiment**. *Management Science*, 35(3), 321–339.,