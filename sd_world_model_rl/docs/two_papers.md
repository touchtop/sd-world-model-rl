战略研究蓝图：基于结构化世界模型的运筹学与机器学习双轨发表策略
执行摘要
本报告旨在为 SD-MCTS（System Dynamics - Monte Carlo Tree Search，系统动力学-蒙特卡洛树搜索） 框架制定一份详尽的战略研究与发表蓝图。基于 workshop 论文《Structured World Models for Operations Research》所奠定的初步成功，本报告规划了一条双轨制的学术发表路径：首先确立在运筹学顶级期刊 INFORMS Journal on Computing (IJOC) 上的发表地位，随后以此为跳板，冲击机器学习顶级会议 NeurIPS。
SD-MCTS 框架的核心价值在于其处于现代运筹学（寻求数据驱动与经典优化的融合）与机器学习（转向结构化、物理信息模型以克服黑盒局限）的交叉点。初步研究表明，该框架通过引入系统动力学中的“存量-流量（Stock-Flow）”守恒逻辑作为强归纳偏置，结合可微分方程学习与 MCTS 离散决策，有效解决了传统模型预测控制（MPC）的视界截断误差问题 1。
然而，要满足 IJOC 和 NeurIPS 截然不同的审稿标准，必须对现有工作进行针对性的扩展。对于 IJOC，研究重心必须转向 计算可扩展性（Scalability）、工业级基准测试与启发式算法对比，证明其在处理大规模、多智能体及随机动态系统时的优越性。对于 NeurIPS，研究重心则需转向 学习理论保证、归纳偏置的数学形式化与泛化边界，通过与 DreamerV3、TD-MCTS2 等前沿世界模型的对比，阐明结构化先验在样本效率上的本质优势。
本报告全长约 15,000 字，详细拆解了这一战略的执行细节，包括具体的实验设计、理论推导路径及文献综述策略。
________________________________________1. 技术基线解构：SD-MCTS 的现状与潜力
在制定扩展策略之前，必须对 SD-MCTS 框架当前的技术状态、核心创新及理论基础进行严格的解构。这不仅是后续扩展的起点，也是识别现有工作与顶级刊物要求之间差距的关键步骤。
1.1 核心架构：微分方程与树搜索的神经符号融合
SD-MCTS 框架代表了从端到端“黑盒”学习向“神经符号”规划的范式转变。其架构建立在一个核心洞察之上：许多运筹学（OR）问题表现为 混合动力学（Hybrid Dynamics） 特征，即连续的物理演化过程（如设备磨损、库存消耗）与离散的控制事件（如任务分配、补货指令）相互交织 1。
1.1.1 结构化世界模型作为强归纳偏置
该框架的核心创新在于强制引入 系统动力学（System Dynamics, SD） 结构作为归纳偏置。与 MuZero 或 Dreamer 等标准基于模型的强化学习（MBRL）方法不同，SD-MCTS 不通过通用神经网络学习任意状态转移 $s_{t+1} = f_\theta(s_t, a_t)$，而是约束 $f_\theta$ 必须服从“存量-流量”守恒逻辑 1。
●	数学形式化： 状态空间被显式分解为存量（Stocks, $S(t)$）和流量（Flows）。系统的演化由形式为 $\frac{dS(t)}{dt} = \text{Inflow} - \text{Outflow}$ 的微分方程控制。例如，在任务分配场景中，疲劳度的累积与恢复被建模为非线性微分方程。
●	参数空间降维： 通过假设动力学的函数形式已知（例如工人效率服从逻辑斯蒂衰减），学习问题从函数逼近（学习曲线的形状）转化为系统辨识（学习曲线的参数 $\theta$，如衰减率 $\alpha$）。理论上，这使得状态预测误差的样本复杂度从表格方法的 $O(|\mathcal{S}||\mathcal{A}|/\epsilon^2)$ 降低至 $O(d/\epsilon^2)$，其中 $d$ 为可学习参数的数量（$d \ll |\mathcal{S}||\mathcal{A}|$）1。
●	物理一致性与安全性： 与黑盒模型容易在分布外区域产生“因果幻觉”（如预测出负库存）不同，SD-MCTS 通过构造（如 Softplus 激活函数约束速率为正）强制满足物理不变量。这种结构化的约束保证了模型在未见区域的外推安全性（Extrapolation Safety）2。
1.1.2 混合规划机制：MCTS + 价值网络
框架将可微分的 SD 世界模型与 蒙特卡洛树搜索（MCTS） 紧密耦合：
●	离散控制： MCTS 处理离散动作空间的组合复杂性（如任务分配决策）。
●	连续演化： 搜索树中的节点利用学习到的 SD 方程模拟未来的连续状态转移。
●	价值函数逼近： 最关键的是，MCTS 的叶节点评估不再依赖随机模拟（Rollout），而是使用一个学习到的价值网络 $V_\psi(s)$。这一机制有效地创建了一个“近似无限视界”，使智能体能够“看见”搜索树深度之外的长期后果 1。
1.2 理论基础与初步验证
初步工作建立在 视界-误差权衡（Horizon-Error Trade-off） 的理论基础之上。
1.2.1 截断误差 vs. 模型误差
核心理论贡献在于量化了规划过程中两类竞争误差的影响 1：
1.	截断误差（Truncation Error, $\epsilon_{trunc}$）： 源于将规划限制在有限视界 $H$ 内。在传统 MPC 中，假设终端成本为零，该误差上界为 $\frac{\gamma^H R_{max}}{1-\gamma}$。
2.	模型误差（Model Error, $\epsilon_{model}$）： 源于估计参数 $\hat{\theta}$ 与真实参数 $\theta^*$ 之间的偏差。
核心发现： 研究证实，通过引入学习到的终端价值网络来补偿截断误差，即便模型参数存在偏差，SD-MCTS 的性能也能超越受限于有限视界的完美模型规划器（Oracle MPC）。这一发现验证了假设：“一个不完美但有远见的模型优于一个完美但短视的模型” 1。
1.2.2 实验证据
该架构在两个关键场景中得到了验证：
1.	带效率衰减的任务分配： 在工人效率随疲劳非线性衰减的场景中，仅经 30k 步训练的 SD-MCTS 超越了拥有完美参数的 Oracle MPC 达 3.0% ($p < 0.0001$, Cohen's $d = 2.05$)。Oracle MPC 因视界受限，无法捕捉疲劳累积的长尾效应 1。
2.	延迟崩溃与需求激增： 在系统状态存在长延迟崩溃风险（模拟设备过热或人员倦怠）的场景中，SD-MCTS 习得了 “主动休息（Active Rest）” 策略，牺牲短期收益以换取长期生存，性能提升 66.7% 1。
________________________________________2. 战略扩展轨道一：INFORMS Journal on Computing (IJOC)
目标： 确立 SD-MCTS 在解决大规模、复杂运筹学问题上的 计算优越性与工业适用性。
核心叙事： “工程化高性能混合智能决策系统：超越传统启发式与 MPC 的局限。”
INFORMS Journal on Computing 侧重于运筹学与计算机科学的接口，极其看重算法在解决实际困难问题时的可扩展性、计算效率及与现有方法（如元启发式算法）的对比。为了在此发表，叙事重心必须从“新颖的架构”转向“更优的解决方案”。
2.1 突破“玩具问题”：确立大规模实验基准
当前的实验环境（2个智能体、简化队列）在 IJOC 审稿人眼中可能被视为“玩具问题（Toy Problem）”，难以证明方法的泛化能力 3。为了满足 IJOC 对“大规模（Large-Scale）”实验的严苛标准 5，研究必须在问题规模上实现数量级的跨越。
2.1.1 实验设计：构建 "Dyn-OR-Bench" 基准套件
我们需要定义并生成一套大规模基准测试集。针对资源分配与调度问题，扩展应涵盖以下三个维度：
1.	智能体/机器规模（Scale）： 将工人和机器数量从 $N=2$ 扩展至 $N \in \{10, 50, 100\}$。这不仅测试 MCTS 处理高维组合动作空间的能力，也验证 SD 模型在参数共享机制下的学习效率 7。
2.	状态空间复杂性（State Complexity）： 引入多级交互。不再是单一队列，而是模拟多级供应链（Multi-Echelon Supply Chain）或多阶段制造流水线。这将测试“存量-流量”归纳偏置在处理级联动态（Cascading Dynamics）时的鲁棒性 4。
3.	视界深度（Horizon Depth）： 增加有效规划视界的需求。例如，将“崩溃”事件的延迟从 $t=12$ 延长至 $t=100$，测试价值网络在长程信用分配（Credit Assignment）中的有效性。
建议方法论： 借鉴调度领域的 "Taillard Instances" 理念 10，生成一套包含 100 个具有不同规模和动态特性的合成实例，命名为 "Dyn-OR-Bench"。这将为 SD-MCTS 提供一个无可辩驳的验证平台。
2.1.2 分布式训练与推理架构
IJOC 关注算法的软件实现与计算性能 5。报告必须详细阐述 SD-MCTS 的计算架构：
●	并行 MCTS (Parallel MCTS)： 针对大规模 $N$ 导致的分支因子爆炸问题，实施根并行（Root Parallelization）或树并行（Tree Parallelization）策略 11。
●	向量化推理 (Vectorized Inference)： 描述如何利用 torch.vmap 或 jax.vmap 对 SD 微分方程进行向量化，以支持并行模拟数千条轨迹。
●	计算成本分析： 建立严格的“挂钟时间（Wall-Clock Time） vs. 解质量”帕累托前沿分析。单纯的性能提升不足以说服审稿人，必须证明在同等计算预算下，SD-MCTS 能提供更优解 12。
2.2 扩展基线：与元启发式算法的正面交锋
当前的 MPC 基线虽具代表性，但在运筹学领域，针对复杂非线性调度问题，工业界的实际标准往往是遗传算法（GA）或禁忌搜索（Tabu Search）14。IJOC 审稿人必然会挑战：“为何不使用元启发式算法？”
2.2.1 对比遗传算法（Genetic Algorithms, GA）
GA 擅长搜索静态参数或全局调度，但在处理状态随时间随机演化的动态问题时往往力不从心。
●	假设： GA 能找到优秀的 静态 策略，但在 动态 环境中（如机器随机故障、工人疲劳累积），其重规划（Re-optimization）的滞后性会导致次优解 14。
●	实验设计： 将 SD-MCTS 与标准的滚动时域遗传算法（Rolling Horizon GA, 如 NSGA-II）进行对比。GA 在 $t=0$ 优化固定调度，并每隔 $k$ 步进行重规划。
●	预期洞察： SD-MCTS 应展现出卓越的“反应式（Reactive）”能力——即时调整策略以适应状态偏差，而 GA 则因重规划的计算开销而表现出滞后性 17。
2.2.2 对比禁忌搜索与模拟退火
对于离散任务分配，禁忌搜索是强有力的竞争者。
●	混合视角： 我们不仅可以将关系框架为“对抗”，更可以探讨“融合”。IJOC 对 “数学启发式（Matheuristics）” 抱有浓厚兴趣 18。
●	创新点： 探讨 SD-MCTS 中的价值网络 $V_\psi(s)$ 是否可以作为禁忌搜索的评估函数（Evaluation Function）。这种“混合 AI/OR”的视角对于 IJOC 极具吸引力 15。
2.3 深化“系统动力学”在 OR 中的连接
IJOC 的读者高度赞赏对系统行为的严谨建模。我们需要将“系统动力学”从一个简单的归纳偏置，提升为一种完整的建模方法论。
2.3.1 库存管理与牛鞭效应（The Beer Game）
啤酒游戏（Beer Game） 是多级供应链中的经典 SD 问题，因其时间延迟和非线性导致的牛鞭效应而闻名，也是传统控制方法的噩梦 19。
●	扩展应用： 将 SD-MCTS 应用于啤酒游戏基准测试 19。
●	领域契合： 这直接切中 IJOC 的“供应链管理”和“随机模型”板块 4。
●	机制优势： SD-MCTS 的“存量-流量”结构与啤酒游戏的库存-订单方程同构。这种完美的结构对齐应使 SD-MCTS 在缓解牛鞭效应方面，比无模型的 DQN/PPO 快几个数量级收敛 21。
2.3.2 引入“流体近似（Fluid Approximations）”概念
在运筹学排队论中，“流体模型”是处理大规模排队网络的标准近似方法 23。
●	洞察： SD-MCTS 中的可微分 SD 模型，本质上就是一个 学习到的流体模型。
●	术语转换： 在 IJOC 投稿中，建议将“连续动力学组件”明确框架为 “学习型流体近似（Learned Fluid Approximation）”。这一术语与经典 OR 文献（如排队网络的流体极限）高度共鸣，能显著降低 OR 审稿人的认知门槛并提升理论认可度 25。
2.4 强化鲁棒性与不确定性处理
运筹学高度关注 鲁棒优化（Robust Optimization） 和 随机规划（Stochastic Programming） 27。
2.4.1 参数不确定性（Robustness）
目前的工作测试了 错误 的初始参数。应扩展至 不确定 的参数分布。
●	实验： 假设衰减率 $\alpha$ 不是固定值，而是服从分布 $\alpha \sim \mathcal{N}(\mu, \sigma)$ 的随机变量。
●	方法： 修改 MCTS 的模拟步骤，使其在每次 Rollout 时采样不同的 $\alpha$ 值（Scenario MCTS）。这标志着向 鲁棒 SD-MCTS 的演进，符合鲁棒优化的研究趋势 27。
2.4.2 约束满足（Constraint Satisfaction）
工业系统充满硬约束（如最大容量、安全阈值）。
●	约束处理： 详细说明 SD-MCTS 如何处理约束。是通过 MCTS 的动作屏蔽（Masking）处理硬约束，还是通过奖励惩罚处理软约束？
●	结构优势： 强调“存量-流量”结构允许对某些约束进行 解析式 验证（例如，“库存非负”是结构保证的，而非学习得来的），这是相对于黑盒 RL 的重大安全优势 29。
2.5 IJOC 扩展计划摘要
核心组件	当前状态 (Workshop)	IJOC 投稿所需扩展
实验规模	2 智能体，简单队列	50+ 智能体，多级网络（侧重可扩展性）
基线算法	MPC, Adaptive MPC	遗传算法 (GA), 禁忌搜索 (Tabu), MIP (Gurobi)
应用领域	任务分配	供应链 (Beer Game) 或大规模动态车间调度
理论框架	视界权衡	流体模型收敛性，鲁棒性分析
核心焦点	概念验证	计算效率与工业可行性
________________________________________3. 战略扩展轨道二：NeurIPS (智能化扩展)
目标： 发表一篇关于 世界模型归纳偏置与泛化理论 的开创性论文。
核心叙事： “结构之美：为何物理先验是世界模型样本效率与泛化的关键。”
NeurIPS 偏好新颖的方法论、对学习动力学的深刻理解以及与更广泛 AI 主题（如基础模型、物理信息机器学习）的连接。此处的叙事必须从“解决调度问题”升华为“理解结构化先验如何促进动力系统学习”。
3.1 理论深潜：样本复杂度与泛化边界
Workshop 论文简要提到了 $O(d/\epsilon^2)$ 的样本复杂度。对于 NeurIPS，这不能仅是一个注脚，而必须是核心论点。我们需要严格形式化 为什么 结构化世界模型比非结构化模型具有更好的泛化能力 31。
3.1.1 归纳偏置的形式化
必须用数学语言定义“系统动力学归纳偏置”。
●	假设空间： 令 $\mathcal{H}_{SD}$ 为可表示为存量-流量系统（遵守守恒律的 ODE）的函数假设空间；令 $\mathcal{H}_{NN}$ 为通用神经网络的假设空间。
●	复杂度度量： 利用 Rademacher 复杂度 或 VC 维 来量化 $\mathcal{H}_{SD}$ 与 $\mathcal{H}_{NN}$ 的“大小”差异。
●	目标定理： 证明 $\text{Rad}(\mathcal{H}_{SD}) \ll \text{Rad}(\mathcal{H}_{NN})$，从而推导出更紧的泛化误差边界 32。这将为 SD-MCTS 的高样本效率提供坚实的学习理论支撑 34。
3.1.2 神经符号规划的遗憾界（Regret Bounds）
近期 NeurIPS 非常关注 RL 的遗憾界分析 35。
●	贡献点： 推导 SD-MCTS 智能体的遗憾界。
●	论证逻辑： 由于世界模型收敛更快（得益于结构化先验），“探索-利用”的遗憾在早期即被最小化。
●	预期结论： “我们证明，对于满足质量守恒的系统，SD-MCTS 可实现 $\tilde{O}(\sqrt{T})$ 的遗憾界，且常数项仅依赖于物理参数数量 $d$，而与状态空间大小 $|\mathcal{S}|$ 无关” 37。
3.2 物理信息机器学习（PIML）语境
将 SD-MCTS 定位在 物理信息（Physics-Informed） 趋势中是冲击 NeurIPS 2025 的关键策略 32。
3.2.1 “存量-流量”即守恒定律
将 SD 方程与基础物理学联系起来。“存量”即遵守守恒律（质量、能量、信息）的状态变量。
●	横向对比： 将 SD-MCTS 与 物理信息神经网络（PINNs） 进行对比 39。PINNs 通常用于求解 PDE，而 SD-MCTS 解决的是基于 ODE 系统的 控制问题。
●	实验：物理违背分析（Physics Violation Analysis）。 在数据集上训练黑盒世界模型（如 DreamerV3）和 SD-MCTS。在分布外（OOD）数据上测试，测量“物理违背率”（例如，负库存、凭空产生的质量）。SD-MCTS 应通过设计保证 0% 违背，而黑盒模型极可能产生幻觉 2。
3.2.2 神经符号集成
将“SD 方程 + 神经参数”架构框架化为 神经符号 AI（Neuro-Symbolic AI） 的一个特例 41。
●	符号组件： 因果图（存量与流量的拓扑结构）和函数形式（ODE）。
●	神经组件： 参数估计器（可能是超网络或摊销推理网络）和价值函数。
●	NeurIPS 卖点： 这解决了“符号落地（Symbolic Grounding）”问题。符号（存量）在物理世界中具有明确的对应含义 42。
3.3 挑战 SOTA 世界模型
NeurIPS 审稿人必然要求与 MBRL 的“明星”算法进行对比：DreamerV3, TD-MPC2, 和 MuZero 43。
3.3.1 “数据效率”大对决
●	设置： 在相同的 OR 环境中运行 DreamerV3 和 TD-MPC2。
●	预测： 深度 RL 方法最终也能解决问题，但可能需要 $10^6$ 步。SD-MCTS 仅需 $10^4$ 步。
●	洞察： “在稀疏奖励、长视界的 OR 任务中，像 Dreamer 那样从零开始学习物理规律的成本极其高昂。在交互昂贵的‘真实世界’中，结构化先验是通往样本效率的唯一路径” 46。
3.3.2 拓扑泛化（Zero-Shot Transfer）
NeurIPS 非常看重 组合泛化（Combinatorial Generalization） 47。
●	实验： 在 3 节点的供应链上训练模型。在 不重新训练 的情况下，将其测试在 5 节点的供应链上。
●	机制： 由于 SD-MCTS 学习的是 参数（如“库存持有成本”），这些参数往往可以迁移到新的智能体或节点上。相比之下，黑盒网络通常过拟合于输入维度，拓扑结构一变即失效。
●	意义： 这证明模型学到了“系统规则”而非仅仅死记硬背了策略，这是通向通用人工智能（AGI）的关键目标 48。
3.4 NeurIPS 扩展计划摘要
核心组件	当前状态 (Workshop)	NeurIPS 投稿所需扩展
基线算法	MPC	DreamerV3, TD-MPC2, MuZero (SOTA MBRL)
理论深度	启发式论证	遗憾界 (Regret Bounds), Rademacher 复杂度, 归纳偏置形式化
核心焦点	应用性能	泛化性, 样本效率, 物理一致性
核心叙事	“更好的规划”	“结构化世界模型”, “神经符号 AI”, “物理信息 RL”
关键指标	利润/奖励	样本复杂度 ($N_{steps}$), 物理违背率, 零样本迁移得分
________________________________________4. 协同效应与反馈闭环：统一的工作流
双轨策略并非意味着双倍的工作量，而是要利用一轨的 产出 作为另一轨的 输入。
4.1 “数据引擎”协同
为 IJOC 开发的大规模、高保真基准测试集（第 2.1 节），将成为 NeurIPS 论文中的 “高难度环境”。
●	工作流： 为 IJOC 开发 "Dyn-OR-Bench"。一旦验证其对传统求解器具有挑战性，就在 NeurIPS 中使用它来展示 DreamerV3/TD-MPC2 在这些任务上的低效或失败。
●	叙事红利： 这赋予了 NeurIPS 论文独特的、接地气的实验环境，区别于被过度使用的 Atari/DM Control Suite，增加了新颖性 45。
4.2 “理论验证”协同
为 NeurIPS 推导的理论边界（第 3.1 节），将作为 IJOC 论文的 “数学正确性证明”。
●	工作流： 完成 NeurIPS 的收敛性证明。在 IJOC 投稿中引用该预印本或包含简化引理，以从数学原理上解释 为什么 该方法具有鲁棒性。
●	叙事红利： IJOC 审稿人偏爱数学严谨性。拥有学习理论层面的收敛性证明（即使主要焦点是实验）将显著增强论文的说服力。
________________________________________5. 详细实验执行计划
5.1 统一基准套件："Dyn-OR-Bench"
为了同时支持两篇论文，我们定义一套包含三个领域的统一基准套件：
1.	领域 A: 动态作业车间调度 (Dynamic Job Shop Scheduling, DJSSP)
○	动力学： 机器随机故障（随机可用性）；作业变质（加工时间随等待时间增加）。
○	结构： 存量 = 作业队列；流量 = 加工速率。
○	规模： 最高 50 台机器，500 个作业。
○	用途： IJOC 对比遗传算法；NeurIPS 对比 DreamerV3。
2.	领域 B: 多级供应链 (Beer Game++)
○	动力学： 4-8 级；随机需求；运输延迟；腐坏（库存衰减）。
○	结构： 存量 = 库存/在途；流量 = 订单/发货。
○	规模： 8 级，连续动作空间。
○	用途： IJOC 对比 MIP；NeurIPS 测试泛化/迁移。
3.	领域 C: 流行病/生化过程控制 (SEIR / Tennessee Eastman)
○	动力学： 疾病传播 (SEIR) 或 化工过程 (Tennessee Eastman 50)。
○	结构： 存量 = 易感者/反应物；流量 = 感染率/反应率。
○	用途： 展示“物理信息”建模能力 51。
5.2 核心对比指标矩阵
指标	定义	对 IJOC 的重要性	对 NeurIPS 的重要性
最优性差距 (Optimality Gap)	$(R_{alg} - R_{opt}) / R_{opt}$	关键 (vs. MIP/Heuristics)	中等
挂钟时间 (Wall-Clock Time)	求解所需秒数	关键 (Scalability)	中等
样本效率 (Sample Efficiency)	收敛所需步数	低 (假设离线训练)	关键 (Learning speed)
约束违背率 (Constraint Violation)	无效动作占比	关键 (Feasibility)	高 (Safety/Physics)
OOD 泛化 (OOD Generalization)	未见参数下的性能	中等 (Robustness)	关键 (Intelligence)
可解释性 (Interpretability)	参数是否可读？	高 (Trust)	中等 (Explainability)
________________________________________6. 结论与行动路线图
SD-MCTS 框架坐拥跨学科潜力的金矿。通过有效地“分叉”研究路径，可以最大化其学术影响力。
1.	第一阶段：IJOC “规模化”突击（第 1-3 个月）
○	实现 "Dyn-OR-Bench" 环境（车间调度、供应链）。
○	实现高性能基线：NSGA-II（遗传算法）和基础 MIP 求解器（Gurobi）。
○	运行大规模实验，聚焦 可扩展性（增加 $N$）。
○	交付物： 一篇聚焦“通过结构化神经规划解决大规模动态 OR 问题”的草稿。
2.	第二阶段：NeurIPS “理论化”深耕（第 2-4 个月）
○	在实验运行的同时，推导理论证明：“存量-流量” MDP 的遗憾界。
○	实现 SOTA MBRL 基线（DreamerV3, TD-MPC2）。
○	运行“数据效率”和“物理违背”实验。
○	交付物： 一篇聚焦“世界模型中的归纳偏置：为何结构优于数据”的草稿。
3.	第三阶段：投稿与协同（第 5 个月）
○	提交 NeurIPS（通常为 5 月截稿）。
○	利用 NeurIPS 阶段获得的理论洞察润色 IJOC 手稿并提交（滚动截稿）。
通过尊重这两个社区截然不同的“语言”——IJOC 的计算效率 与 NeurIPS 的学习理论——本工作有望确立 SD-MCTS 作为连接确定性运筹学世界与自适应人工智能世界的基础桥梁。证据表明，“结构”是解锁这一协同效应的钥匙，它既提供了前者的安全性，又保留了后者的灵活性。
________________________________________引用索引:.21
引用的著作
1.	workshop_paper_final_zh.pdf
2.	From Generative Engines to Actionable Simulators: The Imperative of Physical Grounding in World Models - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/html/2601.15533v1
3.	LEARNING HIDDEN VARIABLES IN PROBABILISTIC GRAPHICAL MODELS - CS.HUJI, 访问时间为 一月 27, 2026， https://www.cs.huji.ac.il/~nirf/Papers/ElidanPHD.pdf
4.	Full article: A review on reinforcement learning algorithms and applications in supply chain management - Taylor & Francis, 访问时间为 一月 27, 2026， https://www.tandfonline.com/doi/full/10.1080/00207543.2022.2140221
5.	INFORMS Journal on Computing: Vol 34, No 1 - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/toc/ijoc/34/1
6.	INFORMS Journal on Computing: Vol 34, No 5 - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/toc/ijoc/34/5
7.	Robust Scheduling Based on Deep Reinforcement Learning for Flexible Job Shop With Machine Breakdown and New Job Arrival | J. Comput. Inf. Sci. Eng. | ASME Digital Collection, 访问时间为 一月 27, 2026， https://asmedigitalcollection.asme.org/computingengineering/article/doi/10.1115/1.4070035/1223153/Robust-scheduling-based-on-deep-reinforcement
8.	Exact and Approximation Algorithms for the Tactical Fixed Interval Scheduling Problem, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/doi/10.1287/opre.45.4.624
9.	Reinforcement Learning for Multi-Objective Multi-Echelon Supply Chain Optimisation - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/html/2507.19788v1
10.	NS4S: Neighborhood Search for Scheduling Problems Via Large Language Models - IJCAI, 访问时间为 一月 27, 2026， https://www.ijcai.org/proceedings/2025/966
11.	Reinforcement Learning: A Tutorial Survey and Recent Advances - Derong Liu, 访问时间为 一月 27, 2026， http://www.derongliu.org/adp/adp-cdrom/refs/gosavi20090178.pdf
12.	A review on learning to solve combinatorial optimisation problems in manufacturing - Institutional Knowledge (InK) @ SMU, 访问时间为 一月 27, 2026， https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=9088&context=sis_research
13.	A reinforcement learning method for container terminal storage space allocation problem - AIP Publishing, 访问时间为 一月 27, 2026， https://pubs.aip.org/aip/adv/article/15/9/095019/3362821/A-reinforcement-learning-method-for-container
14.	Local Search Heuristics for the Single Machine Total Weighted Tardiness Scheduling Problem | INFORMS Journal on Computing - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/doi/10.1287/ijoc.10.3.341
15.	A Hybrid Tabu Search and Constraint Programming Algorithm for the Dynamic Dial-a-Ride Problem | INFORMS Journal on Computing - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/doi/10.1287/ijoc.1110.0454
16.	A Powerful Genetic Algorithm Using Edge Assembly Crossover for the Traveling Salesman Problem | INFORMS Journal on Computing - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/doi/10.1287/ijoc.1120.0506
17.	Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/html/2601.11189v1
18.	Combination of Metaheuristic and Exact Algorithms for Solving Set Covering-Type Optimization Problems | INFORMS Journal on Computing - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/doi/10.1287/ijoc.1090.0376
19.	Reinforcement Learning for Logistics and Supply Chain Management: Methodologies, State of the Art, and Future Opportunities - ResearchGate, 访问时间为 一月 27, 2026， https://www.researchgate.net/profile/Yong-Hong-Kuo-2/publication/360482161_Reinforcement_learning_for_logistics_and_supply_chain_management_Methodologies_state_of_the_art_and_future_opportunities/links/62851c2ea93a5471227a29af/Reinforcement-learning-for-logistics-and-supply-chain-management-Methodologies-state-of-the-art-and-future-opportunities.pdf
20.	G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration, 访问时间为 一月 27, 2026， https://arxiv.org/html/2506.09272v1
21.	OR-Gym: A Reinforcement Learning Library for Operations Research Problems, 访问时间为 一月 27, 2026， https://egon.cheme.cmu.edu/Papers/Hubbs_or_gym_9_11.pdf
22.	Deep Reinforcement Learning for Large-Scale Inventory Management - ResearchGate, 访问时间为 一月 27, 2026， https://www.researchgate.net/publication/371896599_Deep_Reinforcement_Learning_for_Large-Scale_Inventory_Management
23.	A Fluid Approximation for Large-Scale Service Systems - Columbia University, 访问时间为 一月 27, 2026， https://www.columbia.edu/~ww2040/MAMA_LiuWhitt_081210.pdf
24.	A Fluid-Diffusion-Hybrid Limiting Approximation for Priority Systems with Fast and Slow Customers | Operations Research - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/doi/10.1287/opre.2021.2154
25.	When to use Speedup: An Examination of Service Systems with Returns - Columbia University, 访问时间为 一月 27, 2026， http://www.columbia.edu/~cc3179/ICU_fluid.pdf
26.	[2209.04745] Optimization of the fluid model of scheduling: local predictions - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/abs/2209.04745
27.	Machine Learning for K-Adaptability in Two-Stage Robust Optimization - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/doi/10.1287/ijoc.2022.0314
28.	INFORMS Journal on Computing: Vol 37, No 3 - PubsOnLine, 访问时间为 一月 27, 2026， https://pubsonline.informs.org/toc/ijoc/37/3
29.	SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/html/2506.02255v1
30.	Bullet-Safety-Gym: A Framework for Constrained Reinforcement Learning - mediaTUM, 访问时间为 一月 27, 2026， https://mediatum.ub.tum.de/doc/1639974/42wmrf49mnlje58hqhpurs3i9.main.pdf
31.	NeurIPS 2025 Orals, 访问时间为 一月 27, 2026， https://neurips.cc/virtual/2025/events/oral
32.	NeurIPS Poster Understanding Generalization in Physics Informed Models through Affine Variety Dimensions, 访问时间为 一月 27, 2026， https://neurips.cc/virtual/2025/poster/120137
33.	NeurIPS Poster Generalization Bounds for Model-based Algorithm Configuration, 访问时间为 一月 27, 2026， https://neurips.cc/virtual/2025/poster/117197
34.	Sample-Efficient Reinforcement Learning for Linearly-Parameterized MDPs with a Generative Model - NIPS, 访问时间为 一月 27, 2026， https://proceedings.neurips.cc/paper/2021/file/c21f4ce780c5c9d774f79841b81fdc6d-Paper.pdf
35.	NeurIPS 2024 Papers, 访问时间为 一月 27, 2026， https://nips.cc/virtual/2024/papers.html
36.	Online Regret Bounds for Undiscounted Continuous Reinforcement Learning - NIPS papers, 访问时间为 一月 27, 2026， http://papers.neurips.cc/paper/4666-online-regret-bounds-for-undiscounted-continuous-reinforcement-learning.pdf
37.	Reinforcement Learning and Regret Bounds for Admission Control - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/pdf/2406.04766
38.	NeurIPS Poster RoboScape: Physics-informed Embodied World Model, 访问时间为 一月 27, 2026， https://neurips.cc/virtual/2025/poster/115313
39.	Physics-informed neural networks - Wikipedia, 访问时间为 一月 27, 2026， https://en.wikipedia.org/wiki/Physics-informed_neural_networks
40.	Learning physical models that can respect conservation laws - Stat.berkeley.edu, 访问时间为 一月 27, 2026， https://www.stat.berkeley.edu/~mmahoney/pubs/prob_conserve_1.pdf
41.	Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures – Benefits and Limitations - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/html/2502.11269v1
42.	ICML 2024 Papers, 访问时间为 一月 27, 2026， https://icml.cc/virtual/2024/papers.html
43.	danijar/dreamerv3: Mastering Diverse Domains through World Models - GitHub, 访问时间为 一月 27, 2026， https://github.com/danijar/dreamerv3
44.	TD-MPC2: Scalable, Robust World Models for Continuous Control - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/html/2310.16828v2
45.	TD-MPC2, 访问时间为 一月 27, 2026， https://www.tdmpc2.com/
46.	What Are World Models? The AI Architecture That Learns to Dream | rewire.it, 访问时间为 一月 27, 2026， https://rewire.it/blog/what-are-world-models-ai-path-to-understanding-reality/
47.	Offline model-based reinforcement learning with causal structured world models, 访问时间为 一月 27, 2026， https://journal.hep.com.cn/fcs/EN/10.1007/s11704-024-3946-y
48.	Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning - arXiv, 访问时间为 一月 27, 2026， https://arxiv.org/html/2508.10399v1
49.	Training Robust Deep Reinforcement Learning Policies in the Real World - UQ eSpace - The University of Queensland, 访问时间为 一月 27, 2026， https://espace.library.uq.edu.au/view/UQ:cf1a3b1/s4565876_phd_thesis.pdf?dsi_version=cdb3b6c009752d36c742934dae2dbfa5
50.	Fault detection and diagnosis in the Tennessee Eastman Process using interpretable knowledge discovery - IEEE Xplore, 访问时间为 一月 27, 2026， https://ieeexplore.ieee.org/document/7889650/
51.	Modeling of Learning Processes Using Continuous-Time Markov Chain for Virtual-Reality-Based Surgical Training in Laparoscopic Surgery - NIH, 访问时间为 一月 27, 2026， https://pmc.ncbi.nlm.nih.gov/articles/pmid/38617582/
52.	RoboScape: Physics-informed Embodied World Model - OpenReview, 访问时间为 一月 27, 2026， https://openreview.net/forum?id=wbZCBBrq3W
53.	ICML Poster Online Robust Reinforcement Learning Through Monte-Carlo Planning, 访问时间为 一月 27, 2026， https://icml.cc/virtual/2025/poster/44177
54.	Physics-informed Machine Learning | PNNL, 访问时间为 一月 27, 2026， https://www.pnnl.gov/explainer-articles/physics-informed-machine-learning
55.	Bridging Classic Operations Research and Artificial Intelligence for Network Optimization in the 6G Era: A Review - MDPI, 访问时间为 一月 27, 2026， https://www.mdpi.com/2073-8994/17/8/1279
56.	EVOLUTIONARY ALGORITHM FOR DYNAMIC RESOURCE ALLOCATION AND ITS APPLICATIONS - International Journal of Simulation Modelling, 访问时间为 一月 27, 2026， http://www.ijsimm.com/Full_Papers/Fulltext2024/text23-3_CO14.pdf
57.	(PDF) Optimal Resource Allocation Using Genetic Algorithm in Container-Based Heterogeneous Cloud - ResearchGate, 访问时间为 一月 27, 2026， https://www.researchgate.net/publication/377296047_Optimal_Resource_Allocation_Using_Genetic_Algorithm_in_Container-Based_Heterogeneous_Cloud
58.	NeurIPS 2025 San Diego Datasets & Benchmarks, 访问时间为 一月 27, 2026， https://neurips.cc/virtual/2025/loc/san-diego/events/datasets-benchmarks-2025
