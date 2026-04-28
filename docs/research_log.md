# Research Log

## 2026-04-29

针对 Hopper `SD transition core` 当前闭环表现，补充完成一轮高层方法论收敛与文档整理。核心判断是：`SD dynamics` 的拟合目标与 TD-MPC2 的规划目标不应再被视为天然一致。当前证据已经显示，`consistency_loss` 在中期以后反弹，`75k` replay 上的 `landing / takeoff / high_omega` 误差仍显著，而 `100k` 闭环行为改善主要表现为“后仰失败态 -> 前倾推进态”的恢复子程序，而非 `hop` 行为链形成。

据此，后续研究口径正式拆分为两条线：

1. `model fidelity`：评估共享 `SD transition core` 在当前 replay 分布上的一步与短视界动力学保真度；
2. `planning usefulness`：评估该 core 是否改善动作排序、规划稳定性与闭环样本效率。

同时决定，若继续改训练方式，应优先采用“共享 transition core + 独立在线 SD 监督支路”，而不是引入两个相互竞争的 dynamics 头。详细分析见：

- [learnable_sd_v5_objective_decoupling_2026_04_29.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_objective_decoupling_2026_04_29.md)

相关方法文档与决策文档同步建立：

- [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)
- [decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md)
- [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)
