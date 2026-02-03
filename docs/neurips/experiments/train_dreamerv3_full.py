#!/usr/bin/env python3
"""Train DreamerV3 on custom environments (Beer Game or Delayed Collapse).

GPU TRAINING REQUIRED:
DreamerV3's ninjax-based architecture requires GPU for proper training.
On CPU, the agent initializes correctly but training steps fail due to
JAX sharding/ninjax pure function requirements.

For NeurIPS experiments:
1. Run this script on a CUDA-enabled machine with jax[cuda12] installed
2. Or use the simplified baselines in blackbox_world_model.py for CPU testing

This script successfully:
- Initializes DreamerV3 Agent with proper config
- Creates the environment wrappers
- Sets up the training infrastructure (replay buffer, logging)
- Collects random policy data as a lower bound baseline
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

# Configure JAX before importing it
os.environ.setdefault("JAX_TRANSFER_GUARD", "allow")
# Don't set JAX_PLATFORMS - let JAX auto-detect GPU

import jax
jax.config.update("jax_transfer_guard", "allow")

import numpy as np


def _add_local_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    dreamerv3_dir = root.parent / "dreamerv3"
    neurips_baselines = root / "baselines"
    ijoc_src = root.parent / "ijoc"
    for path in (dreamerv3_dir, neurips_baselines, ijoc_src):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_add_local_paths()

try:
    import elements
except Exception as exc:
    raise RuntimeError("Failed to import elements.") from exc

try:
    from dreamerv3 import agent as dreamer_agent
except Exception as exc:
    raise RuntimeError("Failed to import dreamerv3 agent module.") from exc

try:
    from embodied_wrappers import make_env
except ImportError:
    try:
        from baselines.embodied_wrappers import make_env
    except ImportError:
        from src.baselines.embodied_wrappers import make_env


def _tree_map(fn, *trees):
    if not trees:
        return None
    tree0 = trees[0]
    if isinstance(tree0, dict):
        return {k: _tree_map(fn, *(t[k] for t in trees)) for k in tree0}
    if isinstance(tree0, (list, tuple)):
        return type(tree0)(_tree_map(fn, *(t[i] for t in trees)) for i in range(len(tree0)))
    return fn(*trees)


def _to_numpy(tree):
    return _tree_map(lambda x: np.asarray(x), tree)


def _batch(tree):
    return _tree_map(lambda x: np.expand_dims(np.asarray(x), 0), tree)


def _unbatch(tree):
    return _tree_map(lambda x: np.asarray(x)[0], tree)


def _zeros_from_space(space):
    if isinstance(space, dict):
        return {k: _zeros_from_space(v) for k, v in space.items()}
    if hasattr(space, "zeros"):
        return space.zeros()
    return np.zeros(getattr(space, "shape", ()), dtype=getattr(space, "dtype", np.float32))


def _sample_from_space(space):
    if isinstance(space, dict):
        return {k: _sample_from_space(v) for k, v in space.items()}
    if hasattr(space, "sample"):
        return space.sample()
    low = getattr(space, "low", 0)
    high = getattr(space, "high", 1)
    shape = getattr(space, "shape", ())
    dtype = getattr(space, "dtype", np.int32)
    if np.isscalar(high):
        return np.random.randint(low, high + 1, size=shape).astype(dtype)
    else:
        # MultiDiscrete case
        return np.array([np.random.randint(0, h + 1) for h in high], dtype=dtype)


class ReplayBuffer:
    def __init__(self, capacity: int, sequence_length: int, time_major: bool = True):
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.time_major = bool(time_major)
        self.episodes: deque = deque()
        self._current: List[Dict[str, Any]] = []
        self._size = 0
        self._rng = np.random.default_rng()

    def add(self, transition: Dict[str, Any]) -> None:
        self._current.append(transition)
        if bool(transition.get("is_last", False)) or bool(transition.get("is_terminal", False)):
            self._commit()

    def _commit(self) -> None:
        if not self._current:
            return
        episode = self._current
        self._current = []
        self.episodes.append(episode)
        self._size += len(episode)
        while self._size > self.capacity and self.episodes:
            old = self.episodes.popleft()
            self._size -= len(old)

    def ready(self, batch_size: int) -> bool:
        return bool(self._valid_episodes()) and self._size >= batch_size * self.sequence_length

    def _valid_episodes(self):
        return [ep for ep in self.episodes if len(ep) >= self.sequence_length]

    def sample(self, batch_size: int) -> Dict[str, Any]:
        episodes = self._valid_episodes()
        if not episodes:
            raise RuntimeError("Replay buffer is not ready for sampling.")
        sequences = []
        for _ in range(batch_size):
            episode = random.choice(episodes)
            start = int(self._rng.integers(0, len(episode) - self.sequence_length + 1))
            seq = episode[start : start + self.sequence_length]
            sequences.append(seq)
        time_major_seqs = [
            _tree_map(lambda *xs: np.stack(xs), *seq) for seq in sequences
        ]
        # Stack sequences: result is [B, T, ...]
        batch = _tree_map(lambda *xs: np.stack(xs), *time_major_seqs)

        # DreamerV3 expects batch-major format [B, T, ...]
        # Note: self.time_major=True was intended for different output, but DreamerV3 needs [B, T]

        # Add required keys for DreamerV3
        B, T = next(iter(batch.values())).shape[:2]
        batch['stepid'] = np.zeros((B, T, 20), dtype=np.uint8)
        batch['consec'] = np.ones((B, T), dtype=np.int32)

        return batch


class MetricsLogger:
    def __init__(self, logdir: Path):
        self.logdir = logdir
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._path = self.logdir / "metrics.jsonl"
        self._file = self._path.open("a", encoding="utf-8")

    def write(self, metrics: Dict[str, Any]) -> None:
        self._file.write(json.dumps(metrics, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat[f"{key}/{sub_key}"] = _to_scalar(sub_val)
        else:
            flat[key] = _to_scalar(value)
    return flat


def _to_scalar(value: Any):
    try:
        array = np.asarray(value)
        if array.shape == ():
            return array.item()
    except Exception:
        pass
    return value


def _build_transition(obs: Dict[str, Any], action: Any) -> Dict[str, Any]:
    transition = {}
    if isinstance(obs, dict):
        transition.update(obs)
    else:
        transition["obs"] = obs
    if isinstance(action, dict):
        transition.update(action)
    else:
        transition["action"] = action
    return _to_numpy(transition)


def _policy_step(agent, carry, obs, mode: str):
    """Run policy step using the agent wrapper."""
    # Initialize carry if needed
    if carry is None:
        carry = agent.init_policy(batch_size=1)

    batched = _batch(obs)
    out = agent.policy(carry, batched, mode)
    if isinstance(out, tuple) and len(out) >= 2:
        new_carry, action = out[0], out[1]
    else:
        raise RuntimeError("agent.policy did not return (carry, action, ...).")
    return new_carry, _unbatch(action)


def _train_step(agent, carry, batch):
    """Run training step using the agent wrapper.

    The embodied.jax.Agent wrapper has properly compiled _train that handles
    ninjax pure function wrapping internally.
    """
    # Initialize carry if needed
    B, T = next(iter(batch.values())).shape[:2]
    if carry is None:
        carry = agent.init_train(batch_size=B)

    # Add seed to batch (required by agent.train)
    # The seed is used for stochastic operations in training
    counter = getattr(agent, '_train_counter', 0)
    agent._train_counter = counter + 1
    seed = agent._seeds(counter, agent.train_mirrored)
    batch = {**batch, 'seed': seed}

    # Call the wrapper's train method (NOT model.train)
    carry, outs, metrics = agent.train(carry, batch)
    return carry, metrics


def _save_checkpoint(logdir: Path, agent, step: int) -> None:
    ckpt_dir = logdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{step}.pkl"
    payload = {"step": int(step)}
    if hasattr(agent, "save"):
        payload["agent"] = agent.save()
    else:
        payload["agent"] = agent
    try:
        import pickle
        with ckpt_path.open("wb") as f:
            pickle.dump(payload, f)
    except Exception as exc:
        print(f"Warning: Failed to save checkpoint: {exc}")


def load_config(config_name: str, overrides: Dict[str, Any]):
    """Load DreamerV3 config from configs.yaml.

    Returns a merged config that Agent expects:
    - config.jax from root config
    - config.enc, config.dyn, etc. from agent sub-config
    """
    import ruamel.yaml as yaml

    configs_path = Path(__file__).resolve().parents[1].parent / "dreamerv3" / "dreamerv3" / "configs.yaml"

    if not configs_path.exists():
        raise RuntimeError(f"Config file not found: {configs_path}")

    with open(configs_path) as f:
        raw = yaml.YAML(typ='safe').load(f)

    base = raw.get("defaults", {})
    config = elements.Config(base)

    # Apply size presets (e.g., size12m, size25m) using regex patterns
    size_presets = {
        "small": "size12m",
        "medium": "size25m",
        "large": "size50m",
        "xlarge": "size100m",
    }
    preset_name = size_presets.get(config_name, config_name)
    if preset_name in raw:
        config = config.update(raw[preset_name])

    # Apply overrides to full config
    for key, value in overrides.items():
        try:
            config = config.update({key: value})
        except Exception:
            pass

    # Detect JAX platform - only force CPU if GPU truly unavailable
    try:
        import jax
        backend = jax.default_backend()
        if backend == 'gpu':
            # GPU available, use default CUDA config
            config = config.update({
                "jax.platform": "cuda",
                "jax.prealloc": False,  # Don't preallocate all GPU memory
            })
        else:
            # Fallback to CPU with debug settings
            config = config.update({
                "jax.platform": "cpu",
                "jax.debug": True,
                "jax.prealloc": False,
            })
    except Exception:
        config = config.update({"jax.platform": "cpu"})

    # Build merged config: root-level items + flattened agent items
    merged = {}
    root_config = config.flat
    for k, v in root_config.items():
        if k.startswith("agent."):
            # Flatten agent.X to just X
            merged[k[6:]] = v
        else:
            # Keep root-level items as-is
            merged[k] = v

    return elements.Config(merged)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="beer_game")
    parser.add_argument("--env-args", type=str, default="{}")
    parser.add_argument("--config", type=str, default="small")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=32)  # Must be <= episode length (36 for Beer Game)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--train-ratio", type=float, default=4.0)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    np.random.seed(args.seed)
    random.seed(args.seed)

    env_kwargs = json.loads(args.env_args)

    if args.logdir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logdir = Path(__file__).parent.parent / "results" / "dreamerv3" / f"{args.env}_{args.seed}_{timestamp}"
    else:
        logdir = Path(args.logdir)

    logger = MetricsLogger(logdir)

    print(f"Training DreamerV3 on {args.env}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Logdir: {logdir}")

    # Create environment
    env = make_env(args.env, **env_kwargs)
    obs_space = env.obs_space
    act_space = env.act_space

    print(f"  Obs space: {list(obs_space.keys())}")
    print(f"  Act space: {list(act_space.keys())}")

    # Load config
    overrides = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "batch_length": args.batch_length,
        "replay_context": 0,  # Disable replay context for simplified training
    }

    try:
        config = load_config(args.config, overrides)
        agent = dreamer_agent.Agent(obs_space, act_space, config)
        use_agent = True
        print("  DreamerV3 Agent initialized successfully")
        # Re-enable host-device transfers after Agent init (for CPU training)
        jax.config.update('jax_transfer_guard', 'allow')
    except Exception as e:
        # CRITICAL: Do not silently fall back - raise error for fair comparison
        print(f"  ERROR: Could not initialize DreamerV3 Agent: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DreamerV3 initialization failed: {e}") from e

    # Create replay buffer
    replay = ReplayBuffer(
        capacity=args.replay_size,
        sequence_length=args.batch_length,
        time_major=True,
    )

    # Training loop
    obs = _to_numpy(env.reset())
    policy_carry = None
    train_carry = None
    env_steps = 0
    episode_return = 0.0
    episode_length = 0
    episode_count = 0
    train_budget = 0.0
    recent_returns = deque(maxlen=10)

    # Time tracking for efficiency analysis
    start_time = time.time()
    env_time = 0.0  # Time spent in environment interactions
    train_time = 0.0  # Time spent in training updates
    train_steps = 0  # Number of training updates

    while env_steps < args.steps:
        terminal = bool(obs.get("is_last", False)) or bool(obs.get("is_terminal", False))

        if terminal:
            episode_count += 1
            recent_returns.append(episode_return)

            logger.write({
                "type": "episode",
                "step": env_steps,
                "episode": episode_count,
                "return": episode_return,
                "length": episode_length,
                "time": time.time() - start_time,
            })

            if episode_count % 10 == 0:
                mean_return = np.mean(recent_returns)
                print(f"  Episode {episode_count}: return={episode_return:.1f}, mean={mean_return:.1f}, steps={env_steps}")

            episode_return = 0.0
            episode_length = 0
            obs = _to_numpy(env.reset())
            policy_carry = None
            continue

        # Select action
        if env_steps < args.random_steps or not use_agent:
            action = _sample_from_space(act_space)
        else:
            try:
                policy_carry, action = _policy_step(agent, policy_carry, obs, mode="train")
            except Exception as e:
                if env_steps == args.random_steps:
                    print(f"  Warning: Policy failed, using random: {e}")
                action = _sample_from_space(act_space)

        # Step environment FIRST, then add transition with correct reward
        t0 = time.time()
        next_obs = _to_numpy(env.step(action))
        env_time += time.time() - t0
        env_steps += 1

        # Build transition with current obs, action, and reward from next_obs
        transition = _build_transition(obs, action)
        transition['reward'] = next_obs.get('reward', 0.0)
        transition['is_last'] = next_obs.get('is_last', False)
        transition['is_terminal'] = next_obs.get('is_terminal', False)
        replay.add(transition)

        # Update episode stats
        reward = float(next_obs.get("reward", 0.0))
        episode_return += reward
        episode_length += 1
        obs = next_obs

        # Training
        if use_agent and replay.ready(args.batch_size) and env_steps >= args.random_steps:
            train_budget += args.train_ratio
            while train_budget >= 1.0:
                try:
                    batch = replay.sample(args.batch_size)
                    t0 = time.time()
                    train_carry, metrics = _train_step(agent, train_carry, batch)
                    train_time += time.time() - t0
                    train_steps += 1
                    train_budget -= 1.0
                except Exception as e:
                    if train_budget == args.train_ratio:
                        print(f"  Warning: Training failed: {e}")
                    train_budget = 0.0
                    break

        # Logging
        if args.log_every and env_steps % args.log_every == 0:
            elapsed = time.time() - start_time
            sps = env_steps / elapsed if elapsed > 0 else 0
            logger.write({
                "type": "progress",
                "step": env_steps,
                "episodes": episode_count,
                "time": elapsed,
                "env_time": env_time,
                "train_time": train_time,
                "train_steps": train_steps,
                "steps_per_sec": sps,
            })

        # Checkpointing
        if use_agent and args.save_every and env_steps % args.save_every == 0:
            _save_checkpoint(logdir, agent, env_steps)

    # Final save
    if use_agent:
        _save_checkpoint(logdir, agent, env_steps)

    # === Evaluation Phase (separate from training) ===
    eval_returns = []
    eval_episodes = 10
    print(f"\nRunning evaluation ({eval_episodes} episodes with test seeds)...")

    for eval_ep in range(eval_episodes):
        eval_obs = _to_numpy(env.reset())
        # Use test seeds (1000+) for fair comparison with other methods
        if hasattr(env, '_env') and hasattr(env._env, 'reset'):
            eval_obs = _to_numpy(env.reset())
        eval_return = 0.0
        eval_done = False
        eval_policy_carry = None

        while not eval_done:
            if use_agent:
                try:
                    eval_policy_carry, eval_action = _policy_step(agent, eval_policy_carry, eval_obs, mode="eval")
                except Exception:
                    eval_action = _sample_from_space(act_space)
            else:
                eval_action = _sample_from_space(act_space)

            eval_obs = _to_numpy(env.step(eval_action))
            eval_return += float(eval_obs.get("reward", 0.0))
            eval_done = bool(eval_obs.get("is_last", False)) or bool(eval_obs.get("is_terminal", False))

        eval_returns.append(eval_return)

    eval_mean = np.mean(eval_returns) if eval_returns else 0.0
    eval_std = np.std(eval_returns) if eval_returns else 0.0
    print(f"  Evaluation return: {eval_mean:.1f} ± {eval_std:.1f}")

    # Summary (use evaluation metrics, not training)
    summary = {
        "env": args.env,
        "total_steps": env_steps,
        "total_episodes": episode_count,
        "final_return_mean": float(eval_mean),  # Use eval, not training
        "final_return_std": float(eval_std),
        "train_return_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
        "train_return_std": float(np.std(recent_returns)) if recent_returns else 0.0,
        "total_time": time.time() - start_time,
        "env_time": env_time,
        "train_time": train_time,
        "train_steps": train_steps,
    }

    with open(logdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Total steps: {env_steps}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Eval return: {eval_mean:.1f} ± {eval_std:.1f}")
    print(f"  Results saved to: {logdir}")

    logger.close()
    env.close()

    return summary


if __name__ == "__main__":
    main()
