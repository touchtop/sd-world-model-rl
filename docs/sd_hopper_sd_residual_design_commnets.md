我仔细看了这版文档。我的总体判断是：

这版工作已经可以进入 TD-MPC2 框架做系统性闭环训练验证，但还不适合直接作为“已经证明有效”的最终模型。
当前最合理的下一步不是继续设计自定义奖励，而是把这版 SD transition core 接入 TD-MPC2，配合一组严格 ablation 与诊断实验，检验它是否真的提升 model-based RL 的样本效率和规划质量。

你们把研究重心从“结构化奖励塑形”转回“显式 SD dynamics 的刻画与状态推演能力”，这个方向是对的。该段评论对应早先的 no-residual 诊断口径：当时闭环推荐口径中设置 `sd_latent_dim=17`，关闭外层 latent residual，只保留 `DMControlHopperSDModel` 内部受限的 transition residual。此后主线已恢复为 `sd_latent_dim=512`，即保留 17 维显式结构态与 learned residual latent。

1. 总体评价

我认为当前成果有明显进步，主要体现在四点。

第一，模型边界更清楚了。现在不再试图靠结构化奖励引导行为，而是把 SD 作为 transition core 嵌入 TD-MPC2。这能减少实验解释中的混淆：如果后续有提升，更容易归因到 dynamics inductive bias，而不是奖励塑形。

第二，残差边界比之前更合理。当前 internal residual 只修正 vx / vz / omega 三个最难拟合的根部运动通道，不直接改 H / theta / q / qdot / touch / z_min，这保留了显式几何闭环与接触语义。相比早先外层 17 + 495 高维 residual latent 的方案，这更有利于解释 SD 主干的贡献。

第三，监督初始化流程比较干净。load_supervised_dynamics() 只加载 _dynamics，不加载 reward head、Q head、policy head，也不加载外层 residual。这意味着后续 TD-MPC2 训练仍然要自己学习 reward/Q/pi，下游性能不会被“预训练了一整套策略”混淆。

第四，你们已经意识到短视界是当前模型的真实适用范围。文档里明确把验收重点转向 1 / 3 / 5 步，而不是把它包装成长视界自由仿真的替代器；这和 TD-MPC2 Hopper 当前 horizon=3 的使用方式是匹配的。

所以，方向上我认可：这版已经具备接入 TD-MPC2 做闭环验证的价值。

2. 但仍存在几个明显问题
问题 1：当前证据主要还是监督拟合，不是闭环 RL 证据

文档里最强的结果来自独立监督拟合：one_step_struct_rmse = 0.7184、rollout_3_rmse = 1.1788、rollout_5_rmse = 1.4369，以及若干关键量的 MAE，例如 vx_mae = 0.08640、vz_mae = 0.15638、pitch_mae = 0.03091、pitch_rate_mae = 0.73936、support_contact_mae = 0.23627。这些结果说明模型比早期版本更可用，但它们还不能直接证明能提升 TD-MPC2 的闭环学习效果。

尤其是当前 5k pure-env smoke 结果显示：训练稳定，但 2500 和 5000 步都没有明显行为突破，5000 步只有极小非零分，状态仍主要停留在失败地面姿态。这个结果不是失败，但也说明当前证据还停留在“没有把训练弄坏”和“短视界更可信”的阶段。

所以现在不能下结论说：

SD world model 已经能显著提升 Hopper Hop。

更准确的结论是：

SD dynamics 的短视界预测能力已经比之前可信，值得进入 TD-MPC2 闭环实验。

问题 2：监督数据来自随机动作，可能和 TD-MPC2 训练分布不匹配

监督脚本是在 dm_control.suite.hopper.hop 中用随机动作采样数据，然后拟合 encode(obs_t), action_t -> encode(obs_{t+1})。这对模型初始校准有价值，但随机动作数据和 TD-MPC2 后续策略产生的数据分布可能差异很大。

Hopper 的关键高价值状态往往不是随机动作频繁覆盖的区域，例如：

接近站稳
准备起跳
高速前进跳跃
稳定落地
连续 hop cycle

如果监督数据主要覆盖倒地、低速、随机接触、失败姿态，那么模型在“真正需要规划成功行为”的区域仍可能不准。

这不是当前设计的致命问题，但后续必须补一个实验：

用 TD-MPC2 当前 policy / planner 生成的 replay 数据，再评估 SD model 的 1/3/5 步误差。

如果 policy-induced 数据上的误差明显高于随机数据，就说明监督初始化存在 distribution shift。

问题 3：support_contact_mae 仍然偏大，接触相位可能是闭环瓶颈

当前 support_contact_mae = 0.23627，而 full_foot_support_mae = 0.04425。这说明“双足同时支撑”这个较粗量拟合得还可以，但“任一足部支撑强度”这种更细粒度接触质量仍然较难。

对 Hopper 来说，接触误差不是普通误差。它会影响：

下一步垂直速度
水平推进
躯干角速度
落地/离地判断
reward 预测
planner 对动作后果的评估

即使 vx、vz 的平均误差看起来不大，如果接触切换时刻错了，planner 仍可能选择真实环境中不可行的动作。

因此我建议不要只看整体 MAE，而要增加 phase-sliced 接触诊断：

grounded -> airborne
airborne -> grounded
toe-only contact
heel-only contact
full-foot contact
body collision / near-ground

分别统计：

contact classification accuracy
contact onset timing error
contact release timing error
vx/vz/omega error near contact events

如果接触转换附近的误差明显高于普通区间，那后续 TD-MPC2 训练可能仍会被 model exploit 干扰。

问题 4：omega 仍是高风险通道

文档也指出 omega 是最顽固通道。当前 pitch_rate_mae = 0.73936，显著高于 height_mae、vx_mae、pitch_mae 等量。

这对 Hopper 非常关键。因为 Hopper 失败通常不是高度误差一点点，而是：

躯干旋转失控
触地力矩错误
落地后角速度过大
起跳时姿态翻转

你们现在让 internal residual 修正 omega 是合理的，但也说明主干里的接触力矩、空中动作耦合、关节对子身体姿态的影响还没有完全刻画。

在接入 TD-MPC2 前，我建议专门做一个 omega 诊断：

按 contact / airborne / landing / takeoff 分段统计 omega 误差；
统计 residual 对 omega 的贡献比例；
检查 omega residual 是否长期饱和；
检查高 |omega| 状态下预测是否系统性低估。

如果模型系统性低估角速度，planner 会倾向于选择“看起来能站住、真实会翻倒”的动作。

问题 5：internal residual 可能已经承担较大动力学责任，需要量化

当前 residual 的设计边界是好的：只修 vx / vz / omega，零初始化，tanh 限幅，并乘 transition_residual_base_scale = [0.10, 0.18, 0.30] 与全局 transition_residual_scale=4.0。

但注意：全局 scale 为 4.0 后，单步 residual 理论最大幅度大约是：

vx residual:    ±0.40
vz residual:    ±0.72
omega residual: ±1.20

如果这是直接加到下一步速度/角速度上的量，那它并不小。它仍然是“受限 residual”，但可能已经足以主导关键通道。

这不是说它错。Hopper 接触动力学复杂，有 residual 很正常。问题是，如果研究目标是证明“显式结构化知识有价值”，就必须量化：

base SD 主干贡献多少？
internal residual 贡献多少？

建议记录：

|delta_base_vx| vs |delta_resid_vx|
|delta_base_vz| vs |delta_resid_vz|
|delta_base_omega| vs |delta_resid_omega|
residual tanh saturation rate
residual sign distribution
residual contribution by phase

如果 residual 经常饱和，或者 residual 幅度和 base dynamics 同阶甚至更大，那么结论就要谨慎表述为：

显式 SD 提供了结构化特征与几何上下文，residual 完成关键短视界校正。

而不是：

显式 SD 主干本身已经足够准确。

问题 6：H 的语义需要彻底统一

文档中说 H 是“躯干高度，代码中同时视作 rootz / torso_z / height”。这句话本身有风险。

在 Hopper 中，不同“高度”概念可能对应：

rootz
torso world z
torso 相对 foot 的高度
官方 reward 用的 standing height
几何链里 torso base 的高度

如果这些在代码中被混用，短期可能不影响 supervised loss，但会影响：

reward approximation
termination
state diagnostics
contact geometry
height-based viability

你们当前主线用 learned reward，不再直接用显式 reward 作为规划奖励，这是安全的。但如果后面要使用 official reward approximation 或 viability reward 作为诊断，就必须确认：

H 是否严格对应官方 Hopper reward 中使用的 height？

建议把变量命名改清楚，例如：

root_z
torso_world_z
official_height_proxy
body_clearance

不要再让一个 H 同时承担多个语义。

问题 7：监督指标改善幅度较小，需要多 seed 确认

文档中显示 short_rollout_weight=0.10 相比上一基线改善了：

rollout_3_rmse: 1.1864 -> 1.1788
rollout_5_rmse: 1.4534 -> 1.4369
vx_mae:         0.08705 -> 0.08640
pitch_rate_mae: 0.74535 -> 0.73936

这些方向是好的，但幅度并不大。

如果只有一个 seed 或一个数据切分，这些差异可能落在训练噪声范围内。建议至少跑：

3 个 supervised seeds
3 个 data split seeds

不一定要大规模，只要确认 shortloss010 的优势稳定存在即可。

否则当前“最佳 checkpoint”的判断可能只是局部偶然。

3. 是否应该接入 TD-MPC2？

我的建议是：

应该接入 TD-MPC2，但要作为受控验证实验接入，而不是直接大规模押注。

也就是说，不需要回到自定义 reward shaping，也不建议继续无限细调 SD 公式。当前版本已经足够做下一阶段闭环验证。

但在接入时，要明确实验目标：

不是证明 Hopper 已经解决；
而是检验改进后的 SD transition core 是否能提高 TD-MPC2 的样本效率、模型预测质量和规划动作质量。
4. 接入前建议补充的最小检查

我建议接入前做 5 个补充检查。它们不需要很久，但能避免大量闭环实验后才发现问题。

检查 1：phase-sliced 预测误差

不要只看整体 RMSE。按以下相位拆分：

airborne
grounded
landing
takeoff
toe-only
heel-only
full-foot
high-omega
high-vz-impact

每个相位分别统计：

H_mae
vx_mae
vz_mae
theta_mae
omega_mae
touch_toe_mae
touch_heel_mae
z_min_mae

如果整体好、landing/takeoff 很差，TD-MPC2 仍可能失败。

检查 2：residual 贡献与饱和率

记录每一步：

base_delta_vx, residual_delta_vx
base_delta_vz, residual_delta_vz
base_delta_omega, residual_delta_omega

计算：

residual_ratio = |residual| / (|base_delta| + eps)
tanh_saturation_rate = mean(|raw_tanh_output| > 0.95)

如果 residual_ratio 长期过高，说明 SD 主干仍弱；如果 residual 饱和率高，说明 residual scale 或输入特征仍不足。

检查 3：policy-distribution validation

用当前 5k pure-env SD-init smoke 产生的 replay，重新评估监督模型误差。

也就是说，不只在随机动作数据上评估，而是在：

TD-MPC2 实际访问过的 states/actions

上评估：

1-step error
3-step rollout error
reward prediction error
contact error

如果 policy-distribution 误差显著变差，后续应考虑 iterative data aggregation，而不是继续用随机数据预训练。

检查 4：model-based action ranking

这是我最推荐加的一个诊断。

从真实 replay 中采样状态 s
t
	​

，对每个状态采样若干候选动作 a
i
	​

。然后比较：

model 预测的 next-state / reward / value 排名
真实环境 rollout 1~3 步后的排名

你不一定要让预测数值完全准，但 planner 需要模型把“更好的动作”和“更差的动作”排对。

可以统计：

Spearman rank correlation
top-k action hit rate
model-chosen action true return
random action true return

如果 RMSE 不错但 action ranking 很差，TD-MPC2 仍然不会受益。

检查 5：explicit reward approximation 对照

你们现在使用 reward_mode=learned 是合理的，因为要避免结构化 shaping 混入。但 Hopper 官方 reward 本质上并不是人为 shaping，而是任务环境本身的 reward。

因此我建议做一个对照实验：

A. learned reward head
B. explicit official reward approximation from z
C. learned reward + explicit reward auxiliary loss

这不是重新引入自定义结构化奖励，而是检验：

dynamics 明显改善后，reward head 是否仍然是瓶颈。

如果 learned reward 在 sparse early data 下学不到有效信号，而 explicit official reward 可以让 planner 更早利用准确的 vx/H，那说明问题不在 SD transition，而在 reward learning。

5. TD-MPC2 接入实验建议

我建议按以下顺序推进。

阶段 1：5k / 10k smoke，不看最终分数，只看稳定性

目的不是追求 return，而是确认：

不会数值爆炸
不会 action 崩掉
不会 model loss 崩掉
不会 reward/Q loss 异常

至少比较：

vanilla TD-MPC2
SD dynamics random init
SD dynamics supervised init
SD dynamics supervised init + frozen dynamics 前 N 步

记录：

episode return
model loss
reward loss
Q loss
policy entropy / action scale
residual magnitude
contact prediction error

5k 不出高分不要紧，因为文档里当前 pure-env 5k 已经显示 Hopper 仍然很难。

阶段 2：50k / 100k 样本效率对比

这个阶段才开始看 return。

至少做：

3 seeds
same replay budget
same evaluation schedule
same action repeat
same planning horizon

比较：

A. vanilla TD-MPC2 baseline
B. SD no supervised init
C. SD supervised init
D. SD supervised init + residual disabled
E. SD supervised init + internal residual enabled

这样才能回答：

是 SD 结构有用？
是 supervised init 有用？
是 residual 有用？
还是只是训练噪声？
阶段 3：冻结 / 微调策略对照

建议比较：

1. supervised SD dynamics 全程微调
2. 前 5k / 10k 冻结 dynamics，只训练 reward/Q/pi
3. dynamics 使用较小 learning rate 微调

原因是：监督初始化后，如果 RL 早期 replay 很差，dynamics 可能被劣质数据带偏。冻结一小段时间可以测试监督 init 是否本身有价值。

阶段 4：policy-distribution 迭代拟合

如果发现 TD-MPC2 训练过程中 model error 上升，可以采用轻量 DAgger 式流程：

随机数据预训练
→ 5k/10k policy replay
→ 重新 supervised fine-tune SD dynamics
→ 再接回 TD-MPC2

这比继续手改公式更有可能解决 distribution shift。

6. 对研究叙事的建议

如果你们的研究目标是体现“结构化知识显式建模的价值”，现在的叙事应该避免说：

我们用 SD 完全替代了神经网络 world model。

更稳妥、更准确的说法是：

我们将 Hopper dynamics 拆解为显式结构态、几何接触近似、弹簧-阻尼腿部主干和受限 transition residual。显式结构提供可解释状态空间与接触/姿态 inductive bias，residual 只校正最难的根部速度与角速度通道，从而提升短视界规划所需的 transition accuracy。

这和当前实现事实是一致的：SD 主干保留 q/qdot/activation/contact/z_min 的显式递推，internal residual 只补 vx/vz/omega。

后续论文或报告里，关键 ablation 应该是：

1. black-box TD-MPC2
2. SD transition, no residual
3. SD transition + internal residual
4. SD transition + supervised init
5. SD transition + supervised init + internal residual

如果第 5 个最好，而第 2/3/4 也能提供部分提升，就能支撑“结构化建模有价值”的论点。

如果只有第 5 个好，而第 2/3/4 都不好，则结论要更保守：

显式结构提供了 residual 学习的有用坐标系和物理特征，而不是单独完成了高精度建模。

这仍然是有价值的结果。

7. 我认为当前不需要继续做的事
不建议现在恢复自定义结构化 reward

你们已经发现结构化 reward 容易和 TD-MPC2 的 reward/Q 学习产生不协调，而且会让实验解释变复杂。取消自定义 shaping 是合理的。

后续如果要利用结构知识，可以优先通过：

transition model
auxiliary prediction losses
state feature diagnostics
explicit official reward approximation

而不是重新引入人工恢复奖励。

不建议继续大幅增加外层 latent residual

早先 v5 方案里外层 residual latent 是 495 维，高维 residual 如果按维度直接参与统一均方误差，容易掩盖 SD 结构贡献。因此当时将 `sd_latent_dim=17` 作为更干净的诊断口径。当前主线恢复 `sd_latent_dim=512` 后，需通过结构态块与 residual 块按块加权来避免同一问题。

除非你们后续明确转向“性能优先、不强调解释”，否则我建议继续保持当前 17 维结构态主线。

不建议仅凭 5k pure-env 结果判断成败

Hopper Hop 在纯环境奖励下本来就难，尤其 reward head 和 Q head 初期几乎没有高质量正样本。当前 5k 没有突破，只能说明：

监督 SD init 没有立即解决 sparse exploration / reward learning 问题。

不能说明 SD dynamics 没用。

8. 最终建议

我的最终建议是：

进入 TD-MPC2 闭环训练阶段，但要先补充短平快的诊断实验，并把后续训练设计成 ablation-driven，而不是单一路线跑分。

具体执行顺序：

1. 固定当前 contactfix5 + internal residual + shortloss010 checkpoint；
2. 做 phase-sliced error、residual audit、policy-distribution validation、action-ranking diagnostic；
3. 同步接入 TD-MPC2 做 5k/10k smoke；
4. 若数值稳定，推进 50k/100k 多 seed；
5. 做 supervised init / no init、residual on/off、dynamics frozen/fine-tuned 的 ablation；
6. 暂时不要恢复自定义结构化 reward shaping。

我对当前工作的判断是：

没有明显方向性错误；
模型边界比之前清楚很多；
已经具备接入 TD-MPC2 的条件；
但当前证据仍主要证明“短视界 SD transition 更可信”，还没有证明“闭环 RL 明显受益”。

最关键的一句话：

现在应该从“继续手工改 SD 公式”转向“用严格 ablation 检验 SD transition core 是否真的改善 TD-MPC2”。
