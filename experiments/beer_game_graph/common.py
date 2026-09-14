"""Shared components for the §8 Beer Game topology protocol.

Sequential per-node decisions on the generic graph layout; SD and BB arms
differ only in the joint world model. Planner RNG is seeded per
(seed, period, substep) so paired arms consume identical random streams.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.networks import ValueNetwork
from src.env.graph_beer_game import GraphBeerGameEnv, SupplyGraph, load_supply_graph, parse_generic_obs
from src.planning.period_aware_mcts import (
    SequentialDecisionWorldModel,
    make_sequential_reward_fn,
    pack_sequential_state,
    period_aware_mcts_search,
)
from src.world_model.graph_beer_game_sd_model import build_graph_sd_model

TOPOLOGY_DIR = Path(__file__).parent / "topologies"
TOPOLOGY_FILES = {"T1": "t1_serial4.yaml", "T2": "t2_serial8.yaml", "T3": "t3_fork.yaml",
                  "T4": "t4_dag.yaml", "T5": "t5_scale.yaml"}
ARMS = ("sd", "sd_node", "bb", "sd_oracle", "basestock", "sd_mpc")
# Paper protocol constants (run_internal_physics_consistency.py / run_causal_intervention.py).
GAMMA = 0.99
EPSILON = 0.3
MODEL_UPDATES_PER_EPISODE = 5
BATCH = 32
REPLAY = 5000
SD_LR, BB_LR, VALUE_LR = 1e-2, 1e-3, 1e-3


def load_topology(name: str) -> SupplyGraph:
    return load_supply_graph(TOPOLOGY_DIR / TOPOLOGY_FILES[name])


def random_steps_for(budget: int) -> int:
    return min(2000, max(500, budget // 3))


def planner_seed(seed: int, period: int, substep: int, episode: int = 0) -> int:
    key = f"{seed}:{episode}:{period}:{substep}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


class BlackBoxWorldModel(nn.Module):
    """Residual MLP on the generic layout; per-node one-hot actions (paper BB-MCTS family)."""

    def __init__(self, state_dim: int, n_nodes: int, order_levels: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim, self.n_nodes, self.order_levels = state_dim, n_nodes, order_levels
        in_dim = state_dim + n_nodes * order_levels
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, state_dim))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = action.view(state.shape[0], self.n_nodes).long()
        onehot = F.one_hot(action, self.order_levels).flatten(1).to(state.dtype)
        return state + self.mlp(torch.cat([state, onehot], 1))


class NormalizedValueNet(nn.Module):
    """Denormalised continuation value of the joint state; substep/pending slots are ignored."""

    def __init__(self, value_net: ValueNetwork, stats: dict, joint_dim: int):
        super().__init__()
        self.value_net, self.stats, self.joint_dim = value_net, stats, joint_dim

    def forward(self, states):
        return self.value_net(states[..., : self.joint_dim]) * self.stats["std"] + self.stats["mean"]


def update_return_stats(stats: dict, returns: List[float]):
    for r in returns:
        stats["count"] += 1
        delta = r - stats["mean"]
        stats["mean"] += delta / stats["count"]
        if stats["count"] > 1:
            stats["std"] = max(1.0, float(np.sqrt(((stats["count"] - 1) * stats["std"] ** 2
                                                   + delta * (r - stats["mean"])) / stats["count"])))


def physical_mask(g: SupplyGraph) -> np.ndarray:
    """Dims that must be non-negative (everything except t and demand slots)."""
    dim = GraphBeerGameEnv(g, serial_layout=False).state_dim
    mask = np.ones(dim, dtype=bool)
    mask[dim - 1 - len(g.retail_nodes):] = False
    return mask


def build_world_model(arm: str, g: SupplyGraph, state_dim: int):
    if arm == "bb":
        return BlackBoxWorldModel(state_dim, g.n_nodes, g.order_levels)
    if arm == "sd_oracle":
        return build_graph_sd_model(g, learn_demand=False)
    return build_graph_sd_model(g, shared_demand=(arm != "sd_node"), prior_belief=True)


class SequentialPlanner:
    """Per-node MCTS over ``SequentialDecisionWorldModel`` with seeded streams."""

    def __init__(self, env: GraphBeerGameEnv, joint_model: nn.Module, num_simulations: int,
                 leaf: str = "rollout", value_net: Optional[nn.Module] = None, record_imagined: bool = False):
        g = env.graph
        self.env, self.n = env, g.n_nodes
        self.joint_dim = env.state_dim
        self.model = SequentialDecisionWorldModel(joint_model, self.joint_dim, self.n)
        self.num_simulations = num_simulations
        self.leaf, self.value_net = leaf, value_net
        self.reward_fn = make_sequential_reward_fn(lambda s, a=None: env.period_reward(s), self.n, self.joint_dim)
        self.rollout_depth = 4 * self.n  # τ+L periods
        self.imagined: List[np.ndarray] = []
        self.recorder = self.imagined.append if record_imagined else None
        self.calls = 0

    def act(self, joint_state: np.ndarray, seed: int, period: int, episode: int = 0,
            return_details: bool = False):
        pending = np.zeros(self.n, dtype=np.float32)
        details = []
        for sub in range(self.n):
            np.random.seed(planner_seed(seed, period, sub, episode))
            root = pack_sequential_state(joint_state, sub, pending)
            res = period_aware_mcts_search(
                root, self.model, n_actions=self.env.graph.order_levels,
                num_simulations=self.num_simulations, gamma=GAMMA, reward_fn=self.reward_fn,
                value_net=self.value_net if self.leaf != "rollout" else None,
                rollout_depth=self.rollout_depth, period_length=self.n, substep_index=self.joint_dim,
                value_rollout_depth=self.rollout_depth if self.leaf == "hybrid" else 0,
                imagined_state_recorder=self.recorder, return_details=True)
            pending[sub] = res.action
            details.append(res)
        self.calls += self.n * self.num_simulations
        return (pending.astype(int), details) if return_details else pending.astype(int)


class BaseStockPolicy:
    """Order-up-to per node: S_v = (L + τ + 1.5)·D_v on the generic layout."""

    def __init__(self, g: SupplyGraph):
        self.g = g
        self.grid = g.order_quantities()
        self.target = np.array([(2 + 2 + 1.5) * n.demand for n in g.nodes])

    def act(self, obs: np.ndarray) -> np.ndarray:
        g, p = self.g, parse_generic_obs(self.g, np.asarray(obs))
        nr = len(g.retail_nodes)
        ip = p["inventory"].astype(float).copy()
        for i, n in enumerate(g.nodes):
            ip[i] -= sum(p["backlog"][nr + e.index] for e in g.edges if e.u == n.id)
            if n.kind == "retail":
                ip[i] -= p["backlog"][[r.id for r in g.retail_nodes].index(n.id)]
            for e in g.edges:
                if e.v == n.id:
                    ip[i] += p["ship"][e.index].sum() + p["order"][e.index].sum() + p["backlog"][nr + e.index]
            if n.kind == "source":
                ip[i] += p["source"][[s.id for s in g.source_nodes].index(n.id)].sum()
        need = np.clip(self.target - ip, 0, None)
        return np.array([int(np.abs(self.grid[i] - need[i]).argmin()) for i in range(g.n_nodes)])


def rollout_rmse(model: nn.Module, states: np.ndarray, actions: np.ndarray, mask: np.ndarray,
                 horizons=range(1, 9)) -> Dict[str, float]:
    """Open-loop multi-step RMSE on physical dims along an executed trajectory."""
    out = {}
    with torch.no_grad():
        for h in horizons:
            errs = []
            for t in range(len(states) - h):
                s = torch.tensor(states[t])[None]
                for k in range(h):
                    s = model(s, torch.tensor(actions[t + k])[None])
                errs.append(((s[0].numpy() - states[t + h])[mask]) ** 2)
            out[str(h)] = float(np.sqrt(np.mean(errs))) if errs else float("nan")
    return out


def violation_rate(states: List[np.ndarray], mask: np.ndarray) -> Dict[str, float]:
    if not states:
        return {"total": 0, "rate": 0.0}
    arr = np.stack(states)[:, :mask.shape[0]]
    bad = np.any(arr[:, mask] < -1e-6, axis=1)
    return {"total": int(len(arr)), "rate": float(bad.mean())}


def provenance(paths: List[Path]) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {"git_commit": commit,
            "source_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "versions": {"torch": torch.__version__, "numpy": np.__version__}}


def job_name(topology: str, arm: str, budget: int, seed: int, leaf: str = "rollout") -> str:
    tag = "" if leaf == "rollout" else f"_{leaf}"
    return f"{topology}_{arm}{tag}_b{budget}_s{seed}"


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)
