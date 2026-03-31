# NeurIPS 与 IJOC 叙事转换指南 (Narrative Shift)

为了确保 NeurIPS 版本的论文精准打击机器学习（Machine Learning）社区的核心痛点，我们需要与面向运筹学（Operations Research）社区的 IJOC 旧版草稿在核心创新点和叙事侧重上进行彻底的分离。

## 1. 核心受众痛点差异

| 维度 | IJOC (Operations Research) | NeurIPS (Machine Learning & AI) |
|------|---------------------------|---------------------------------|
| **核心痛点** | 传统求解器与 MPC 在面对**强延迟效应**时面临视界截断（Horizon Truncation Bias）。 | MBRL 模型（如 Dreamer）学习环境动力学时极度**数据饥渴**，泛化能力差，不满足**物理一致性**。 |
| **主要假想敌** | 完美环境模型下的 **Oracle MPC**。 | 大规模网络下的 **DreamerV3, TD-MPC2** 及其无结构的 MLP BlackBox。 |
| **期望范式** | "我们借鉴强化学习的价值网络来帮助 MPC 克服短视问题。" | "我们将系统动力学的物理因果图作为归纳偏置注入世界模型，解决端到端学习的缺陷。" |

## 2. 核心创新点 (Claims) 转换

### IJOC 的三大贡献（弃用作为主线，作为次要佐证）
1. Hybrid Planning Framework：将 OR 的系统动力学和 RL 结合。
2. Learned Terminal Cost：用价值网络充当终端损失，延展 MPC 有效前瞻深度。
3. Decision Boundary：发现"当延迟 > 2H 时，规划深度主导参数调优精度"。

### NeurIPS 的三大核心贡献（★ 必须全篇贯彻）
1. **C1 (Sample Efficiency & Physical Consistency)**：通过**结构化世界模型**（物理法则硬编码 + 残差参数学习），在 1K 量级（低 10 倍）样本下击败 DreamerV3，并实现了 100% 物理合规。**（解决 MBRL 痛点）**
2. **C2 (O(1) Error Bounding for Generalization)**：证明并验证了先验截断了分布外状态（OOD）的多步预测误差，从无结构的指数散逸 $O(\lambda^H)$ 收敛到了 $O(1)$ 常数界。**（解决多步 Rollout 缺陷）**
3. **C3 (Zero-Shot Robustness via Causal Intervention)**：赋予了世界模型的"可编辑性"，在结构化图发生突变时允许实施 $do()$ 操作符，获得了零样本分布外干预适配与方差归零的泛化力。**（提升与因果表征领域的联系）**

## 3. 语言与基调调整 (Tone for NeurIPS)

- **少提甚至不提**："我们改进了单纯的 MPC" (MPC 往往被 ML 社区视为古典方法)。
- **多提**："Tabula Rasa（白板学习）的极限与 Inductive Biases（归纳偏置）的力量"。
- **定性术语转化**：把 "Stock-flow Equations" 升级为 "Structural Causal Models (SCMs)" 或者 "Physics-informed Structural Priors"；将 "Terminal Value" 定性为 "Value Alignment in Latent Trees"。

## 4. 论文撰写行动要求

在后续 §4（Method）、§5（Theory）、§1（Introduction）的撰写中：
- 任何证明和架构图必须强调 $\theta$ 参数的学习，而非环境是已知的。
- 强调 MCTS 在这种参数不确定但结构绝对确定的世界中所展现的计算降维。
- 引言故事从强化学习在现实物理系统（如工业、机器人排产）等**结构已知的场景下强行从零学习**的荒谬性起手。
