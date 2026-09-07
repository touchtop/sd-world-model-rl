# Sokoban SD Replacement Design

## 目标

替换路线的目标是把旧的诊断性 `best_first_search()` 路线替换为可学习的 stock-flow-feedback SD 世界模型，并将规划基准固定为 `MBRL + SD-MCTS`。旧代码中的玩家可达域、合法推箱、匹配下界和死锁证据只保留为结构观测代理、流量门控、风险标签或诊断项。

完整设计见 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md)。

## 核心对象

建议后续实现拆成以下概念层：

1. `SokobanStructureEncoder`：从原始状态和静态关卡中提取可达域、合法 single-push option、目标区结构、匹配残差、deadlock evidence 和局部拥塞代理；
2. `SokobanSDStateEncoder`：把结构代理投影为六个核心存量 `G_t, U_t, M_t, R_t, C_t, I_t`，并保留有界残差与单独隐藏上下文；
3. `SokobanFlowModel`：预测非负流量 `f_t`，并通过已知 incidence matrix `B` 得到存量净变化；
4. `SokobanFeedbackModel`：预测目标压力、拥塞-机动性-风险和不确定性反馈；
5. `SokobanSDWorldModel`：联合输出 `x_{t+1}`、progress/reward、risk、uncertainty、feasibility 和 diagnostics；
6. `SokobanSDMCTSAdapter`：在学习到的抽象 SD 动力学上展开 MCTS，并把选中的 single-push option 映射回低层推箱动作。

现有 `SokobanGoalDirectedSDModel` 只能作为结构代理原型拆分来源，不应继续承担世界模型、动作打分和搜索策略三种职责。

## 输出 Schema

每个状态建议输出：

```text
state_id
stock_state = {
  goal_completion,
  unresolved_box_pressure,
  mobility,
  reachable_capacity,
  congestion,
  irreversibility_risk
}
bounded_residual_state
hidden_context_state
deterministic_proxies = {
  reachable_region_mask,
  legal_single_pushes,
  box_goal_cost_matrix,
  matching_residual,
  dead_square_evidence,
  static_corner_evidence,
  local_pattern_evidence,
  goal_room_proxy,
  relevance_proxy,
  path_channel_probe,
  structural_slack_proxy
}
uncertainty_state
diagnostic_flags
```

每个 single-push option 建议输出：

```text
action_id
option_type = single_push
precondition_mask
predicted_nonnegative_flows
incidence_delta
predicted_next_stock
predicted_progress
predicted_branch_risk
predicted_uncertainty
predicted_feasibility
diagnostic_delta_features
```

`diagnostic_delta_features` 可包含匹配残差变化、目标占用变化、可达域变化和死锁证据变化，但不得被用作唯一的动作价值定义。`predicted_duration` 在首版 single-push 设置下为常量或省略，待引入 macro option 后再作为正式字段。

## 分阶段实现

第一阶段保留当前精确 push-level 转移、玩家可达域和结构代理，新增诊断 JSON，使每个状态和动作都能输出六核心存量代理、派生诊断项和结构门控。该阶段不声称实现了完整 SD 世界模型。

第二阶段实现最小可学习 stock-flow 模型。首版使用六个核心存量：目标完成、未解决压力、机动性、可达容量、拥塞和不可逆风险。模型必须输出非负流量、incidence matrix 后的净变化、边界截断信息和下一抽象状态。

第三阶段加入反馈模型。首版只实现目标压力反馈、拥塞-机动性-风险反馈和不确定性反馈，避免一次性引入过多不可辨识耦合。

第四阶段接入 SD-MCTS。MCTS 节点保存抽象存量状态分布、uncertainty、raw-state handle 或 belief support；扩展时调用学习到的 SD 世界模型预测后继状态、流量、进展、风险和不确定性。旧 best-first search 只作为 `heuristic-search diagnostic` 消融。

第五阶段扩展 option 和文献机制。goal-room、tunnel macro、relevance、局部 PDB 和 path-channel 机制先作为结构代理、派生特征、辅助头或诊断项进入，待过剪枝、aliasing 和流量辨识审计完成后再纳入更强约束。

## 当前草稿的保留边界

现有 `eval_medium_train000_50.json` 可作为 smoke/diagnostic baseline 保留，原因是其展示了结构代理对搜索排序的影响。但该结果没有可学习参数化流量、没有反馈回路，也没有 SD-MCTS，不应作为论文式性能主张。后续任何正式结果至少需要：

1. 固定 Boxoban split；
2. 明确难度层；
3. 多文件、多随机抽样关卡；
4. 同时报告世界模型误差、流量守恒误差、风险校准、不确定性校准、state aliasing、规划成功率、预算耗尽和失败类型；
5. 与 `no-learned-parameter`、`no-feedback`、`black-box latent MBRL`、`heuristic-search diagnostic` 和 `learned-transition-provider audit` 做消融。

## 风险

最大风险是把 SD 世界模型重新写成启发式搜索。实现时应坚持四条约束：必须学习参数化流量，必须通过已知 incidence matrix 更新存量，必须显式记录反馈，必须由 SD-MCTS 消费学习到的抽象动力学。任何只输出 `policy_score`、`heuristic_score` 或 `risk_score` 的版本只能作为诊断模型，不能作为主线 SD 世界模型。
