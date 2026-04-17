#!/usr/bin/env python3
"""Hopper-Hop via single-step discrete control modes + SD-MCTS.

Route B is implemented here as *single-step* discrete decision-making:
each label (compress/drive/launch/tuck/absorb) denotes the control mode for
the current decision step only. No explicit duration modeling is used.

The fixed torque table is a temporary low-level placeholder. The research
claim being tested in this script is the decision abstraction itself:
    MCTS over discrete control modes -> SD dynamics -> planning reward.
"""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "glfw")
from dm_control import suite

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


# ---------------------------------------------------------------------------
# 1. Discrete motor intent space
# ---------------------------------------------------------------------------

INTENT_NAMES = ["compress", "drive", "launch", "tuck", "absorb"]
N_INTENTS = len(INTENT_NAMES)
INTENT_TO_IDX = {name: idx for idx, name in enumerate(INTENT_NAMES)}
PHASE_NAMES = ("compression", "takeoff", "flight", "landing")

# Temporary fixed torque profiles: (waist, hip, knee, ankle), each in [-1, 1].
# These are placeholders for the eventual state-dependent low-level mapping.
INTENT_TORQUES = torch.tensor(
    [
        [0.10,  0.70, -0.85, -0.50],  # compress: crouch and load the leg spring
        [0.20, -0.55,  0.78,  0.55],  # drive: committed forward release, still weaker than launch
        [0.00, -0.85,  1.00,  0.90],  # launch: aggressive release for takeoff
        [-0.20, -0.45, -0.75, -0.30], # tuck: retract the leg in flight
        [0.05,  0.20, -0.45, -0.30],  # absorb: compliant landing / shock absorption
    ],
    dtype=torch.float32,
)

assert INTENT_TORQUES.shape == (N_INTENTS, 4)

PHASE_INTENT_PRIORS = {
    "compression": np.array([0.56, 0.20, 0.04, 0.08, 0.12], dtype=np.float32),
    "takeoff": np.array([0.05, 0.49, 0.30, 0.04, 0.12], dtype=np.float32),
    "flight": np.array([0.04, 0.10, 0.04, 0.68, 0.14], dtype=np.float32),
    "landing": np.array([0.12, 0.10, 0.02, 0.06, 0.70], dtype=np.float32),
}

SUBPHASE_INTENT_PRIORS = {
    "stance_load": np.array([0.74, 0.22, 0.04, 0.00, 0.00], dtype=np.float32),
    "stance_release": np.array([0.00, 0.58, 0.42, 0.00, 0.00], dtype=np.float32),
    "takeoff": np.array([0.00, 0.60, 0.40, 0.00, 0.00], dtype=np.float32),
    "flight_rise": np.array([0.00, 0.00, 0.00, 1.00, 0.00], dtype=np.float32),
    "flight_fall": np.array([0.00, 0.00, 0.00, 0.78, 0.22], dtype=np.float32),
    "landing_impact": np.array([0.28, 0.00, 0.00, 0.00, 0.72], dtype=np.float32),
    "landing_recover": np.array([0.62, 0.22, 0.00, 0.00, 0.16], dtype=np.float32),
}


def intent_to_action(intent_idx: int) -> torch.Tensor:
    """Map discrete intent index to continuous 4-dim action in [-1, 1]."""
    return INTENT_TORQUES[intent_idx].clone()


def infer_phase(state: torch.Tensor) -> str:
    """Infer a coarse locomotion phase from structured Hopper state."""
    return infer_phase_context(state)[0]


def infer_phase_context(state: torch.Tensor) -> Tuple[str, str]:
    """Infer both a coarse phase and a control-relevant subphase."""
    support = float(state[DMControlHopperSDModel.SUPPORT_IDX].item())
    vz = float(state[DMControlHopperSDModel.VZ_IDX].item())
    clearance = float(state[DMControlHopperSDModel.CLEARANCE_IDX].item())
    compression = float(state[DMControlHopperSDModel.COMP_IDX].item())
    impact = float(state[DMControlHopperSDModel.IMPACT_IDX].item())

    if support < 0.10 and clearance > -0.02:
        if vz >= -0.02:
            return "flight", "flight_rise"
        return "flight", "flight_fall"

    if support < 0.20 and vz > 0.12 and clearance > -0.05:
        return "takeoff", "takeoff"

    if support >= 0.18 and (vz < -0.16 or impact > 0.10):
        return "landing", "landing_impact"

    if support >= 0.28 and compression > 0.12 and vz > -0.04:
        return "compression", "stance_release"

    if support >= 0.15:
        if vz < -0.06 or compression > 0.10:
            return "compression", "stance_load"
        return "landing", "landing_recover"

    return "compression", "stance_load"


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, a_min=1e-6, a_max=None)
    return probs / probs.sum()


def get_valid_intents(state: torch.Tensor, phase: Optional[str] = None) -> List[int]:
    """Return the semantically admissible intents for the current phase."""
    if phase is None:
        phase, subphase = infer_phase_context(state)
    else:
        _, subphase = infer_phase_context(state)

    if subphase == "stance_release":
        return [INTENT_TO_IDX["drive"], INTENT_TO_IDX["launch"]]

    if subphase == "takeoff":
        return [INTENT_TO_IDX["drive"], INTENT_TO_IDX["launch"]]

    if subphase == "flight_rise":
        return [INTENT_TO_IDX["tuck"]]

    if subphase == "flight_fall":
        return [INTENT_TO_IDX["tuck"], INTENT_TO_IDX["absorb"]]

    if subphase == "landing_impact":
        return [INTENT_TO_IDX["absorb"], INTENT_TO_IDX["compress"]]

    if subphase == "landing_recover":
        return [INTENT_TO_IDX["compress"], INTENT_TO_IDX["drive"], INTENT_TO_IDX["absorb"]]

    if phase == "compression":
        return [INTENT_TO_IDX["compress"], INTENT_TO_IDX["drive"]]

    return list(range(N_INTENTS))


def get_intent_priors(state: torch.Tensor, prev_intent: int = -1) -> Tuple[str, np.ndarray]:
    """Subphase-aware prior over intents with hard semantic support."""
    phase, subphase = infer_phase_context(state)
    priors = SUBPHASE_INTENT_PRIORS.get(subphase, PHASE_INTENT_PRIORS[phase]).copy()
    valid_intents = get_valid_intents(state, phase)

    masked_priors = np.zeros_like(priors)
    masked_priors[valid_intents] = priors[valid_intents]
    priors = masked_priors

    if prev_intent >= 0:
        repeat_penalty = 0.35 if prev_intent == INTENT_TO_IDX["compress"] else 0.60
        if prev_intent in valid_intents:
            priors[prev_intent] *= repeat_penalty

    if priors.sum() <= 0:
        priors[valid_intents] = 1.0

    return phase, _normalize_probs(priors)


# ---------------------------------------------------------------------------
# 2. MCTS node and search (adapted from CartPole runtime)
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ["state", "parent", "intent", "edge_reward", "children", "N", "W", "Q", "P"]

    def __init__(
        self,
        state: torch.Tensor,
        parent=None,
        intent: int = -1,
        edge_reward: float = 0.0,
        prior: float = 1.0 / N_INTENTS,
    ):
        self.state = state
        self.parent = parent
        self.intent = intent
        self.edge_reward = edge_reward
        self.children: Dict[int, "MCTSNode"] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior


class HopperValueNet(nn.Module):
    def __init__(self, state_dim: int = 15, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _model_step(
    model: DMControlHopperSDModel,
    state: torch.Tensor,
    intent_idx: int,
    gamma: float,
) -> Tuple[torch.Tensor, float]:
    """Advance the SD model by one decision step.

    Planning uses the model-defined shaping reward, while environment-side
    training still uses the true reward observed from dm_control.
    """
    action = intent_to_action(intent_idx).unsqueeze(0)
    s = state.unsqueeze(0) if state.dim() == 1 else state
    action = action.to(s.device)
    with torch.no_grad():
        ns = model(s, action)
        reward = model.predict_plan_reward(s, action, ns).item()
    return ns.squeeze(0), reward


def _rollout_value(
    model: DMControlHopperSDModel,
    state: torch.Tensor,
    rollout_depth: int,
    gamma: float,
) -> float:
    """Phase-aware rollout for leaf evaluation."""
    s = state
    ret = 0.0
    disc = 1.0
    prev_intent = -1
    for _ in range(rollout_depth):
        _, priors = get_intent_priors(s, prev_intent)
        intent = int(np.random.choice(N_INTENTS, p=priors))
        s, step_r = _model_step(model, s, intent, gamma)
        ret += disc * step_r
        disc *= gamma
        prev_intent = intent
    return ret


def mcts_search(
    root_state: torch.Tensor,
    model: DMControlHopperSDModel,
    num_simulations: int = 100,
    c: float = 1.5,
    gamma: float = 0.99,
    rollout_depth: int = 10,
    value_net: Optional[HopperValueNet] = None,
    vnet_mix: float = 0.5,
) -> Tuple[int, Dict]:
    """MCTS over single-step discrete control modes."""
    root = MCTSNode(root_state)
    root_phase, root_priors = get_intent_priors(root_state)
    _, root_subphase = infer_phase_context(root_state)
    root_valid_intents = get_valid_intents(root_state, root_phase)

    for _ in range(num_simulations):
        node = root
        path = [node]

        # Selection
        while node.children and len(node.children) == len(get_valid_intents(node.state)):
            best, best_score = None, -float("inf")
            for child in node.children.values():
                if child.N == 0:
                    score = float("inf")
                else:
                    exploration = c * child.P * math.sqrt(node.N + 1) / (child.N + 1)
                    score = child.Q + exploration
                if score > best_score:
                    best_score = score
                    best = child
            node = best
            path.append(node)

        # Expansion: add one unexpanded intent
        valid_intents = get_valid_intents(node.state)
        if len(node.children) < len(valid_intents):
            _, priors = get_intent_priors(node.state, node.intent)
            unexpanded = [idx for idx in valid_intents if idx not in node.children]
            unexpanded_priors = _normalize_probs(np.array([priors[idx] for idx in unexpanded], dtype=np.float32))
            intent_idx = int(np.random.choice(unexpanded, p=unexpanded_priors))
            ns, er = _model_step(model, node.state, intent_idx, gamma)
            child = MCTSNode(
                ns,
                parent=node,
                intent=intent_idx,
                edge_reward=er,
                prior=float(priors[intent_idx]),
            )
            node.children[intent_idx] = child
            node = child
            path.append(node)

        # Leaf evaluation
        rollout_val = _rollout_value(model, node.state, rollout_depth, gamma)
        if value_net is not None:
            with torch.no_grad():
                vnet_val = value_net(node.state.unsqueeze(0)).item()
            leaf_value = vnet_mix * vnet_val + (1.0 - vnet_mix) * rollout_val
        else:
            leaf_value = rollout_val

        # Backpropagation
        g = leaf_value
        for bp_node in reversed(path):
            if bp_node.parent is not None:
                g = bp_node.edge_reward + gamma * g
            bp_node.N += 1
            bp_node.W += g
            bp_node.Q = bp_node.W / bp_node.N

    if not root.children:
        return 0, {"q_values": {}, "visits": {}}

    # Select most-visited action
    visits = {i: ch.N for i, ch in root.children.items()}
    q_vals = {i: ch.Q for i, ch in root.children.items()}
    best_intent = max(visits, key=visits.get)
    return best_intent, {
        "q_values": q_vals,
        "visits": visits,
        "root_value": root.Q,
        "root_phase": root_phase,
        "root_subphase": root_subphase,
        "root_valid_intents": [INTENT_NAMES[i] for i in root_valid_intents],
        "root_priors": {INTENT_NAMES[i]: float(root_priors[i]) for i in range(N_INTENTS)},
    }


# ---------------------------------------------------------------------------
# 3. Environment interface
# ---------------------------------------------------------------------------

def make_env(seed: int = 0):
    return suite.load("hopper", "hop", task_kwargs={"random": seed})


def flatten_obs(obs) -> np.ndarray:
    return np.concatenate([
        np.asarray(obs["position"], dtype=np.float32).reshape(-1),
        np.asarray(obs["velocity"], dtype=np.float32).reshape(-1),
        np.asarray(obs["touch"], dtype=np.float32).reshape(-1),
    ])


def obs_to_struct(model: DMControlHopperSDModel, obs_flat: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        dev = next(model.parameters()).device
        return model.encode_observation(
            torch.tensor(obs_flat, dtype=torch.float32, device=dev).unsqueeze(0)
        ).squeeze(0)


# ---------------------------------------------------------------------------
# 4. Replay buffer and SD model training
# ---------------------------------------------------------------------------

class SimpleReplay:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.next_states: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def add(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: float):
        if len(self.states) >= self.capacity:
            idx = np.random.randint(self.capacity)
            self.states[idx] = state
            self.actions[idx] = action
            self.next_states[idx] = next_state
            self.rewards[idx] = reward
        else:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
            self.rewards.append(reward)

    def __len__(self):
        return len(self.states)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.randint(len(self), size=batch_size)
        return (
            torch.stack([self.states[i] for i in idx]),
            torch.stack([self.actions[i] for i in idx]),
            torch.stack([self.next_states[i] for i in idx]),
            torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32),
        )


def train_sd_model(
    model: DMControlHopperSDModel,
    replay: SimpleReplay,
    optimizer: torch.optim.Optimizer,
    num_updates: int = 64,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, float]:
    if len(replay) < batch_size:
        return {}
    model.train()
    total_state_loss = 0.0
    total_reward_loss = 0.0
    for _ in range(num_updates):
        states, actions, next_states, rewards = replay.sample(batch_size)
        states, actions = states.to(device), actions.to(device)
        next_states, rewards = next_states.to(device), rewards.to(device)
        pred_next = model(states, actions)
        state_loss = F.mse_loss(pred_next, next_states)
        pred_reward = model.predict_reward(states, actions, pred_next)
        reward_loss = F.mse_loss(pred_reward, rewards)
        loss = state_loss + 0.1 * reward_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_state_loss += state_loss.item()
        total_reward_loss += reward_loss.item()
    model.eval()
    return {
        "state_loss": total_state_loss / num_updates,
        "reward_loss": total_reward_loss / num_updates,
    }


def train_value_net(
    vnet: HopperValueNet,
    replay: SimpleReplay,
    model: DMControlHopperSDModel,
    optimizer: torch.optim.Optimizer,
    num_updates: int = 32,
    batch_size: int = 64,
    gamma: float = 0.99,
    device: str = "cpu",
) -> float:
    if len(replay) < batch_size:
        return 0.0
    vnet.train()
    total_loss = 0.0
    for _ in range(num_updates):
        states, actions, next_states, rewards = replay.sample(batch_size)
        states, actions = states.to(device), actions.to(device)
        next_states = next_states.to(device)
        with torch.no_grad():
            plan_rewards = model.predict_plan_reward(states, actions, next_states)
            next_v = vnet(next_states)
        target = plan_rewards + gamma * next_v
        pred_v = vnet(states)
        loss = F.mse_loss(pred_v, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    vnet.eval()
    return total_loss / num_updates


# ---------------------------------------------------------------------------
# 5. Main loop
# ---------------------------------------------------------------------------

def run_episode(
    env,
    model: DMControlHopperSDModel,
    replay: SimpleReplay,
    num_simulations: int,
    rollout_depth: int,
    gamma: float,
    value_net: Optional[HopperValueNet] = None,
    vnet_mix: float = 0.5,
    eval_mode: bool = False,
    max_steps: int = 500,
    device: str = "cpu",
) -> Tuple[float, int, Dict]:
    ts = env.reset()
    obs_flat = flatten_obs(ts.observation)
    struct_state = obs_to_struct(model, obs_flat)

    ep_reward = 0.0
    ep_steps = 0
    intent_counts = np.zeros(N_INTENTS, dtype=int)
    while ep_steps < max_steps:
        best_intent, info = mcts_search(
            struct_state,
            model,
            num_simulations=num_simulations,
            gamma=gamma,
            rollout_depth=rollout_depth,
            value_net=value_net,
            vnet_mix=vnet_mix,
        )
        intent_counts[best_intent] += 1
        action_4d = intent_to_action(best_intent).numpy()

        ts = env.step(action_4d)
        step_reward = ts.reward
        ep_steps += 1

        ep_reward += step_reward
        next_obs_flat = flatten_obs(ts.observation)
        next_struct = obs_to_struct(model, next_obs_flat)

        if not eval_mode:
            replay.add(
                struct_state.clone(),
                torch.tensor(action_4d, dtype=torch.float32),
                next_struct.clone(),
                torch.tensor(step_reward, dtype=torch.float32),
            )

        struct_state = next_struct

    intent_dist = {INTENT_NAMES[i]: int(intent_counts[i]) for i in range(N_INTENTS) if intent_counts[i] > 0}
    return ep_reward, ep_steps, {"intents": intent_dist, "last_mcts": info}


def evaluate(
    env,
    model: DMControlHopperSDModel,
    num_episodes: int,
    num_simulations: int,
    rollout_depth: int,
    gamma: float,
    value_net: Optional[HopperValueNet] = None,
    vnet_mix: float = 0.5,
    device: str = "cpu",
) -> Tuple[float, float]:
    rewards = []
    dummy_replay = SimpleReplay(1)
    for _ in range(num_episodes):
        r, _, _ = run_episode(
            env, model, dummy_replay,
            num_simulations=num_simulations,
            rollout_depth=rollout_depth,
            gamma=gamma,
            value_net=value_net,
            vnet_mix=vnet_mix,
            eval_mode=True,
            max_steps=500,
            device=device,
        )
        rewards.append(r)
    return float(np.mean(rewards)), float(np.std(rewards))


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Hopper-Hop: Single-Step Discrete Intent SD-MCTS")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--budget", type=int, default=50000, help="Total env steps")
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--rollout_depth", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--eval_freq", type=int, default=5000, help="Eval every N env steps")
    parser.add_argument("--train_updates", type=int, default=64)
    parser.add_argument("--train_batch", type=int, default=64)
    parser.add_argument("--seed_steps", type=int, default=2000, help="Random exploration before MCTS")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vnet_mix", type=float, default=0.0, help="Value net mix (0=pure rollout)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.seed)
    model = DMControlHopperSDModel(dt=0.02, hidden=256, hop_speed=2.0).to(args.device)
    model.eval()
    vnet = HopperValueNet().to(args.device) if args.vnet_mix > 0 else None
    replay = SimpleReplay(capacity=args.budget)

    sd_optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    vnet_optim = torch.optim.Adam(vnet.parameters(), lr=args.lr) if vnet else None

    total_steps = 0
    episode_idx = 0
    eval_results = []

    log(f"Starting Hopper-Hop Single-Step Discrete Intent SD-MCTS (seed={args.seed}, budget={args.budget})")
    log(f"Intents: {INTENT_NAMES}")
    log(f"Simulations={args.num_simulations}, rollout_depth={args.rollout_depth}")

    # Initial eval
    eval_mean, eval_std = evaluate(
        env, model, args.eval_episodes, args.num_simulations,
        args.rollout_depth, args.gamma,
        vnet, args.vnet_mix, args.device,
    )
    eval_results.append((0, eval_mean, eval_std))
    log(f"eval step=0: {eval_mean:.4f} ± {eval_std:.4f}")

    while total_steps < args.budget:
        remaining_steps = max(args.budget - total_steps, 0)
        episode_cap = min(500, remaining_steps)
        if episode_cap == 0:
            break

        # Collect one episode
        if total_steps < args.seed_steps:
            # Phase-constrained seed exploration keeps replay within the intent semantics.
            ts = env.reset()
            obs_flat = flatten_obs(ts.observation)
            struct_state = obs_to_struct(model, obs_flat)
            ep_reward = 0.0
            ep_steps = 0
            prev_intent = -1
            while ep_steps < episode_cap:
                _, priors = get_intent_priors(struct_state, prev_intent)
                intent = int(np.random.choice(N_INTENTS, p=priors))
                action_4d = intent_to_action(intent).numpy()
                ts = env.step(action_4d)
                step_r = ts.reward
                ep_steps += 1
                ep_reward += step_r
                next_obs = flatten_obs(ts.observation)
                next_struct = obs_to_struct(model, next_obs)
                replay.add(
                    struct_state.clone(),
                    torch.tensor(action_4d, dtype=torch.float32),
                    next_struct.clone(),
                    torch.tensor(step_r, dtype=torch.float32),
                )
                struct_state = next_struct
                prev_intent = intent
            total_steps += ep_steps
            log(f"seed ep={episode_idx} steps={total_steps} ep_reward={ep_reward:.4f} replay={len(replay)}")
        else:
            # MCTS-guided episode
            ep_reward, ep_steps, ep_info = run_episode(
                env, model, replay,
                num_simulations=args.num_simulations,
                rollout_depth=args.rollout_depth,
                gamma=args.gamma,
                value_net=vnet,
                vnet_mix=args.vnet_mix,
                eval_mode=False,
                max_steps=episode_cap,
                device=args.device,
            )
            total_steps += ep_steps
            log(f"train ep={episode_idx} steps={total_steps} ep_reward={ep_reward:.4f} intents={ep_info['intents']}")

        episode_idx += 1

        # Train SD model
        if total_steps >= args.seed_steps and len(replay) >= args.train_batch:
            train_info = train_sd_model(
                model, replay, sd_optim,
                num_updates=args.train_updates,
                batch_size=args.train_batch,
                device=args.device,
            )
            if train_info:
                log(f"  sd_train: state_loss={train_info['state_loss']:.6f} reward_loss={train_info['reward_loss']:.6f}")

            if vnet and vnet_optim:
                vnet_loss = train_value_net(
                    vnet, replay, model, vnet_optim,
                    num_updates=32, batch_size=args.train_batch,
                    gamma=args.gamma, device=args.device,
                )
                log(f"  vnet_loss={vnet_loss:.6f}")

        # Eval
        if total_steps >= args.seed_steps and (
            episode_idx == 1
            or total_steps % args.eval_freq < 500
            or total_steps >= args.budget
        ):
            eval_mean, eval_std = evaluate(
                env, model, args.eval_episodes, args.num_simulations,
                args.rollout_depth, args.gamma,
                vnet, args.vnet_mix, args.device,
            )
            eval_results.append((total_steps, eval_mean, eval_std))
            log(f"eval step={total_steps}: {eval_mean:.4f} ± {eval_std:.4f}")

    # Final save
    if args.save_path:
        save_dir = Path(args.save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model.state_dict(),
            "eval_results": eval_results,
            "args": vars(args),
        }
        if vnet:
            payload["vnet"] = vnet.state_dict()
        torch.save(payload, args.save_path)
        log(f"Saved to {args.save_path}")

    log("=== Final eval results ===")
    for step, mean, std in eval_results:
        log(f"  step={step:>8}  reward={mean:.4f} ± {std:.4f}")
    log("Done.")


if __name__ == "__main__":
    main()
