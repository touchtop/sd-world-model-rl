#!/usr/bin/env python3
"""Evaluate a saved Hopper checkpoint with optional retrospective rule prior."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TD_ROOT = REPO_ROOT / "external" / "tdmpc2" / "tdmpc2"
for path in (str(REPO_ROOT), str(TD_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")

import hydra
import torch
from termcolor import colored

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from online_rule_prior_v6 import install_rule_prior
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _load_checkpoint(trainer: OnlineTrainer, agent: TDMPC2, checkpoint: Path) -> str:
    if checkpoint.name.endswith(".full.pt"):
        trainer.load_full_state(checkpoint)
        agent.model.eval()
        return "full_state"
    agent.load(checkpoint)
    agent.model.eval()
    return "agent"


def _force_train_like_eval_actions(agent: TDMPC2) -> None:
    """Use train-time planner sampling during eval-only diagnostics.

    This does not update the model or replay buffer. It only checks whether the
    train/eval gap is explained by the planner action mode.
    """
    original_act = agent.act

    def act_train_like(obs, t0=False, eval_mode=False, task=None):
        return original_act(obs, t0=t0, eval_mode=False, task=task)

    agent.act = act_train_like


@hydra.main(config_name="config", config_path="../../external/tdmpc2/tdmpc2")
def evaluate(cfg: dict) -> None:
    """Run eval-only comparison using the same planner patch as training."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    checkpoint = Path(str(cfg.get("checkpoint", "")))
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"

    report_path = cfg.get("eval_report", None)
    report_path = Path(str(report_path)) if report_path else Path(cfg.work_dir) / "eval_rule_prior_report.json"
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(colored("Eval checkpoint:", "yellow", attrs=["bold"]), checkpoint)
    print(colored("Rule prior enabled:", "yellow", attrs=["bold"]), bool(cfg.get("rule_prior_enabled", False)))
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
    action_mode = str(cfg.get("eval_action_mode", "eval"))
    if action_mode not in {"eval", "train_like"}:
        raise ValueError(f"Unknown eval_action_mode: {action_mode}")
    print(colored("Eval action mode:", "yellow", attrs=["bold"]), action_mode)

    env = make_env(cfg)
    agent = TDMPC2(cfg)
    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    checkpoint_kind = _load_checkpoint(trainer, agent, checkpoint)
    prior = install_rule_prior(agent, cfg)
    if action_mode == "train_like":
        _force_train_like_eval_actions(agent)

    metrics = trainer.eval()
    metrics.update(
        {
            "eval_action_mode": action_mode,
            "checkpoint": str(checkpoint),
            "checkpoint_kind": checkpoint_kind,
            "rule_prior_enabled": bool(cfg.get("rule_prior_enabled", False)),
            "rule_prior_library": str(cfg.get("rule_prior_library", "")),
            "rule_prior_allowed_statuses": str(cfg.get("rule_prior_allowed_statuses", "")),
            "rule_prior_mode": str(cfg.get("rule_prior_mode", "")),
            "rule_prior_num_rules": len(prior.rules) if prior is not None else 0,
            "rule_prior_sd_checkpoint": str(cfg.get("rule_prior_sd_checkpoint", "")),
            "rule_prior_sd_checkpoint_loaded": bool(prior.feature_model_checkpoint) if prior is not None else False,
            "work_dir": str(cfg.work_dir),
        }
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(colored("Report:", "yellow", attrs=["bold"]), report_path)


if __name__ == "__main__":
    evaluate()
