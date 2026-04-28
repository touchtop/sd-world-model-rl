# Hopper Reward Regression

该目录用于保存 Hopper 状态判定与结构化奖励的回归测试脚本及其结果。

当前覆盖重点：

1. 假 `rear` 回归：
   前伏抬身但整体仍前倾的片段，不应再被判为 `grounded_rear`。
2. 真 `rear` 保真：
   真正整体后仰的失败片段，仍应保持为 `grounded_rear`。
3. 悬空腿部的后仰保真：
   后仰躺地但腿脚悬空时，不应因脚部参照失真被误判为 `grounded_prone`。
4. 正向迁移奖励：
   `grounded_prone -> grounded_upright` 应触发正向 shaping。
5. 非目标迁移抑制：
   `falling_air -> grounded_prone` 不应触发正向里程碑奖励。

文件说明：

- `reward_regression_suite.py`
  样例定义、state 重建与检查逻辑。
- `test_reward_regression.py`
  `pytest` 回归测试入口。
- `run_reward_regression.py`
  直接执行并写出 JSON 结果。
- `results/`
  存放最近一次执行的结果。

运行方式：

```bash
python experiments/hopper_reward_regression/run_reward_regression.py
```

或：

```bash
pytest -q experiments/hopper_reward_regression/test_reward_regression.py
```

如果当前环境未安装 `pytest`，可直接使用标准库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m unittest -q \
  experiments.hopper_reward_regression.test_reward_regression
```
