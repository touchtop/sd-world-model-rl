import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dm_control import suite
except ImportError:
    suite = None


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / 'checkpoints'
SUMMARY_PATH = CHECKPOINT_DIR / 'paperformal_summary.json'


def get_discrete_actions(n_bins=11):
    return np.linspace(-1.0, 1.0, n_bins).astype(np.float32)


def flatten_obs(obs):
    return np.concatenate([obs['position'], obs['velocity']]).astype(np.float32)


class DMControlCartPoleSDModel(nn.Module):
    def __init__(
        self,
        gear=10.0,
        dt=0.01,
        gravity_init=9.81,
        masscart_init=1.0,
        masspole_init=0.1,
        length_init=0.5,
        slider_damping=0.0005,
        hinge_damping=2e-6,
        learn_params=True,
        prior_noise=0.0,
        mp_prior_noise=None,
        mp_clamp_min=0.01,
    ):
        super().__init__()
        self.gear = gear
        self.dt = dt
        self.state_dim = 4
        self.register_buffer('slider_damping', torch.tensor(slider_damping, dtype=torch.float32))
        self.register_buffer('hinge_damping', torch.tensor(hinge_damping, dtype=torch.float32))
        self.register_buffer('true_gravity', torch.tensor(gravity_init))
        self.register_buffer('true_masscart', torch.tensor(masscart_init))
        self.register_buffer('true_masspole', torch.tensor(masspole_init))
        self.register_buffer('true_length', torch.tensor(length_init))

        if learn_params:
            g_init = gravity_init * (1 + prior_noise * np.random.uniform(-0.3, 0.3))
            mc_init = masscart_init * (1 + prior_noise * np.random.uniform(-0.3, 0.3))
            mp_noise = mp_prior_noise if mp_prior_noise is not None else prior_noise
            mp_init = masspole_init * (1 + mp_noise * np.random.uniform(-0.3, 0.3))
            l_init = length_init * (1 + prior_noise * np.random.uniform(-0.3, 0.3))
            self._mp_clamp_min = mp_clamp_min
            self._gravity_raw = nn.Parameter(torch.tensor(self._inv_softplus(max(g_init, 0.1)), dtype=torch.float32))
            self._masscart_raw = nn.Parameter(torch.tensor(self._inv_softplus(max(mc_init, 0.05)), dtype=torch.float32))
            self._masspole_raw = nn.Parameter(torch.tensor(self._inv_softplus(max(mp_init, mp_clamp_min)), dtype=torch.float32))
            self._length_raw = nn.Parameter(torch.tensor(self._inv_softplus(max(l_init, 0.01)), dtype=torch.float32))
        else:
            self.register_buffer('_g_fixed', torch.tensor(gravity_init, dtype=torch.float32))
            self.register_buffer('_mc_fixed', torch.tensor(masscart_init, dtype=torch.float32))
            self.register_buffer('_mp_fixed', torch.tensor(masspole_init, dtype=torch.float32))
            self.register_buffer('_l_fixed', torch.tensor(length_init, dtype=torch.float32))

        self.learn_params = learn_params
        self._mp_clamp_min_val = mp_clamp_min

    @staticmethod
    def _inv_softplus(x):
        return float(np.log(np.exp(x) - 1))

    @property
    def g(self):
        return F.softplus(self._gravity_raw).clamp_min(0.1) if self.learn_params else self._g_fixed

    @property
    def mc(self):
        return F.softplus(self._masscart_raw).clamp_min(0.05) if self.learn_params else self._mc_fixed

    @property
    def mp(self):
        return F.softplus(self._masspole_raw).clamp_min(self._mp_clamp_min_val) if self.learn_params else self._mp_fixed

    @property
    def l(self):
        return F.softplus(self._length_raw).clamp_min(0.05) if self.learn_params else self._l_fixed

    def _accelerations(self, x, theta, x_dot, theta_dot, force):
        gravity = self.g
        masscart = self.mc
        masspole = self.mp
        length = self.l
        total_mass = masscart + masspole
        polemass_length = masspole * length
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        force_with_damping = force - self.slider_damping * x_dot
        theta_dot_damped = theta_dot
        temp = (force_with_damping + polemass_length * theta_dot_damped ** 2 * sintheta) / total_mass
        denom = length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        denom = torch.clamp(denom, min=1e-4)
        theta_acc = (gravity * sintheta - costheta * temp) / denom
        theta_acc = theta_acc - self.hinge_damping * theta_dot / (masspole * length ** 2)
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass
        return x_acc, theta_acc

    def _rk4_step(self, state, force):
        x, theta, x_dot, theta_dot = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        h = self.dt
        ax1, ath1 = self._accelerations(x, theta, x_dot, theta_dot, force)
        k1_x, k1_th, k1_xd, k1_thd = x_dot, theta_dot, ax1, ath1

        x2 = x + 0.5 * h * k1_x
        th2 = theta + 0.5 * h * k1_th
        xd2 = x_dot + 0.5 * h * k1_xd
        thd2 = theta_dot + 0.5 * h * k1_thd
        ax2, ath2 = self._accelerations(x2, th2, xd2, thd2, force)
        k2_x, k2_th, k2_xd, k2_thd = xd2, thd2, ax2, ath2

        x3 = x + 0.5 * h * k2_x
        th3 = theta + 0.5 * h * k2_th
        xd3 = x_dot + 0.5 * h * k2_xd
        thd3 = theta_dot + 0.5 * h * k2_thd
        ax3, ath3 = self._accelerations(x3, th3, xd3, thd3, force)
        k3_x, k3_th, k3_xd, k3_thd = xd3, thd3, ax3, ath3

        x4 = x + h * k3_x
        th4 = theta + h * k3_th
        xd4 = x_dot + h * k3_xd
        thd4 = theta_dot + h * k3_thd
        ax4, ath4 = self._accelerations(x4, th4, xd4, thd4, force)
        k4_x, k4_th, k4_xd, k4_thd = xd4, thd4, ax4, ath4

        new_x = x + (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        new_theta = theta + (h / 6) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        new_x_dot = x_dot + (h / 6) * (k1_xd + 2 * k2_xd + 2 * k3_xd + k4_xd)
        new_theta_dot = theta_dot + (h / 6) * (k1_thd + 2 * k2_thd + 2 * k3_thd + k4_thd)
        return torch.stack([new_x, new_theta, new_x_dot, new_theta_dot], dim=-1)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        force = action[..., 0] * self.gear
        return self._rk4_step(state, force)

    @staticmethod
    def obs_to_state(obs):
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                x = obs[0]
                theta = np.arctan2(obs[2], obs[1])
                return np.array([x, theta, obs[3], obs[4]], dtype=np.float32)
            x = obs[:, 0]
            theta = np.arctan2(obs[:, 2], obs[:, 1])
            return np.stack([x, theta, obs[:, 3], obs[:, 4]], axis=1).astype(np.float32)
        raise TypeError(f'Expected numpy array, got {type(obs)}')

    def predict_reward(self, state, action, next_state=None):
        if next_state is None:
            next_state = self.forward(state, action)
        theta = next_state[..., 1]
        x = next_state[..., 0]
        theta_dot = next_state[..., 3]
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        ctrl = action[..., 0]
        upright = (torch.cos(theta) + 1) / 2
        centered_raw = torch.exp(-0.5 * (x / 0.9320) ** 2)
        centered = (1 + centered_raw) / 2
        sc_raw = torch.clamp(1 - (ctrl / 1.0) ** 2, min=0)
        small_control = (4 + sc_raw) / 5
        sv_raw = torch.exp(-0.5 * (theta_dot / 2.3300) ** 2)
        small_velocity = (1 + sv_raw) / 2
        return upright * centered * small_control * small_velocity

    def get_learned_params(self):
        return {
            'gravity': self.g.item(),
            'masscart': self.mc.item(),
            'masspole': self.mp.item(),
            'length': self.l.item(),
            'true_gravity': self.true_gravity.item(),
            'true_masscart': self.true_masscart.item(),
            'true_masspole': self.true_masspole.item(),
            'true_length': self.true_length.item(),
        }

    def set_learned_params(self, gravity=None, masscart=None, masspole=None, length=None):
        mapping = {
            '_gravity_raw': (gravity, 0.1),
            '_masscart_raw': (masscart, 0.05),
            '_masspole_raw': (masspole, self._mp_clamp_min_val),
            '_length_raw': (length, 0.05),
        }
        with torch.no_grad():
            for attr, (value, min_value) in mapping.items():
                if value is None:
                    continue
                raw = torch.tensor(self._inv_softplus(max(float(value), min_value)), dtype=torch.float32)
                getattr(self, attr).data.copy_(raw)


class DMControlCartPoleWrapper:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def __call__(self, state, action):
        return self.model(state, action)

    def predict_next_state(self, state_np, action_val):
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            a = torch.tensor([action_val], dtype=torch.float32, device=self.device)
            return self.model.forward(s, a).squeeze(0).cpu().numpy()

    def predict_reward_np(self, state_np, action_val):
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            a = torch.tensor([action_val], dtype=torch.float32, device=self.device)
            return self.model.predict_reward(s, a).squeeze().cpu().item()


class _FallbackTimeStep:
    def __init__(self, observation, reward=0.0, done=False):
        self.observation = observation
        self.reward = reward
        self._done = done

    def last(self):
        return self._done


class FallbackCartpoleEnv:
    def __init__(self, model, seed=0, max_steps=1000):
        self.model = model
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(self.seed)
        self.state = None
        self.steps = 0

    def _state_to_observation(self, state):
        return {
            'position': np.array([state[0], math.cos(state[1]), math.sin(state[1])], dtype=np.float32),
            'velocity': np.array([state[2], state[3]], dtype=np.float32),
        }

    def reset(self):
        theta = self.rng.uniform(-math.pi, math.pi)
        theta_dot = self.rng.uniform(-0.5, 0.5)
        x = self.rng.uniform(-0.05, 0.05)
        x_dot = self.rng.uniform(-0.05, 0.05)
        self.state = np.array([x, theta, x_dot, theta_dot], dtype=np.float32)
        self.steps = 0
        return _FallbackTimeStep(self._state_to_observation(self.state), reward=0.0, done=False)

    def step(self, action):
        action_val = float(np.asarray(action).reshape(-1)[0])
        s = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor([[action_val]], dtype=torch.float32)
        with torch.no_grad():
            next_state = self.model(s, a)[0].cpu().numpy().astype(np.float32)
            reward = float(self.model.predict_reward(s, a, torch.tensor(next_state).unsqueeze(0)).item())
        self.state = next_state
        self.steps += 1
        done = bool(self.steps >= self.max_steps or abs(float(self.state[0])) > 3.0)
        return _FallbackTimeStep(self._state_to_observation(self.state), reward=reward, done=done)

    def close(self):
        return None


class SwingupValueNet(nn.Module):
    def __init__(self, state_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MCTSNode:
    __slots__ = ['state', 'parent', 'action_val', 'edge_reward', 'children', 'N', 'W', 'Q']

    def __init__(self, state, parent=None, action_val=None, edge_reward=0.0):
        self.state = state
        self.parent = parent
        self.action_val = action_val
        self.edge_reward = edge_reward
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0


@dataclass
class ControllerConfig:
    num_simulations: int = 100
    c: float = 1.5
    gamma: float = 0.99
    rollout_depth: int = 20
    action_repeat: int = 2
    leaf_vnet_mix: float = 1.0
    leaf_vnet_gate_cos: Optional[float] = None
    cart_drift_penalty: float = 0.0


def _swingup_action(x, theta, x_dot, theta_dot, g_over_l=19.6):
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)
    if cos_th > 0.85:
        action = -(3.0 * sin_th + 1.0 * theta_dot + 0.4 * x + 0.3 * x_dot)
    else:
        e_norm = 0.5 * theta_dot * theta_dot / g_over_l + cos_th
        e_err = e_norm - 1.0
        pump_dir = 1.0 if theta_dot * cos_th >= 0 else -1.0
        action = -1.5 * e_err * pump_dir - 0.15 * x - 0.1 * x_dot
    return float(np.clip(action, -1.0, 1.0))


def model_forward_k(wrapper, state, action_val, k, gamma=0.99):
    s = state
    cum_reward = 0.0
    discount = 1.0
    action_t = torch.tensor([[action_val]], dtype=torch.float32)
    for _ in range(k):
        with torch.no_grad():
            next_s = wrapper(s.unsqueeze(0), action_t)[0]
            reward = wrapper.model.predict_reward(s.unsqueeze(0), action_t, next_s.unsqueeze(0)).item()
        cum_reward += discount * reward
        discount *= gamma
        s = next_s
    return s, cum_reward, discount


def _heuristic_rollout(wrapper, state, rollout_depth, action_repeat, gamma, g_over_l=19.6):
    gamma_k = gamma ** action_repeat
    rs = state
    rollout_return = 0.0
    disc = 1.0
    for _ in range(rollout_depth):
        h_action = _swingup_action(rs[0].item(), rs[1].item(), rs[2].item(), rs[3].item(), g_over_l=g_over_l)
        new_state, step_reward, _ = model_forward_k(wrapper, rs, h_action, action_repeat, gamma)
        rollout_return += disc * step_reward
        disc *= gamma_k
        rs = new_state
    return rollout_return


def _cart_drift_leaf_bias(state, alpha, cos_threshold=0.5):
    if alpha <= 0.0:
        return 0.0
    cos_th = math.cos(float(state[1].item()))
    if cos_th <= cos_threshold:
        return 0.0
    upright_w = (cos_th - cos_threshold) / (1.0 - cos_threshold)
    x_abs = abs(float(state[0].item()))
    drift = max(x_abs - 1.0, 0.0)
    x_penalty = drift * drift
    th = float(state[1].item())
    th_dot = float(state[3].item())
    th = (th + math.pi) % (2 * math.pi) - math.pi
    theta_penalty = 10.0 * (th * th) + 1.0 * (th_dot * th_dot)
    return upright_w * alpha * (x_penalty + theta_penalty)


def _value_of_state(value_net, state_tensor):
    if value_net is None:
        return 0.0
    with torch.no_grad():
        return float(value_net(state_tensor.unsqueeze(0)).item())


def mcts_action_with_trace(state_np, world_model_wrapper, discrete_actions, value_net, config, g_over_l):
    n_actions = len(discrete_actions)
    gamma_k = config.gamma ** config.action_repeat
    root = MCTSNode(torch.tensor(state_np, dtype=torch.float32))
    last_leaf = {}

    for _ in range(config.num_simulations):
        node = root
        path = [node]
        while node.children and len(node.children) == n_actions:
            best_child, best_score = None, -float('inf')
            for child in node.children.values():
                if child.N == 0:
                    score = float('inf')
                else:
                    score = child.Q + config.c * math.sqrt(math.log(node.N + 1) / (child.N + 1))
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            path.append(node)

        if len(node.children) < n_actions:
            for action_idx in range(n_actions):
                if action_idx in node.children:
                    continue
                action_val = discrete_actions[action_idx]
                final_state, edge_reward, _ = model_forward_k(
                    world_model_wrapper, node.state, action_val, config.action_repeat, config.gamma
                )
                child = MCTSNode(final_state, parent=node, action_val=action_val, edge_reward=edge_reward)
                node.children[action_idx] = child
                node = child
                path.append(node)
                break

        rollout_val = None
        vnet_value = _value_of_state(value_net, node.state)
        mix = float(np.clip(config.leaf_vnet_mix, 0.0, 1.0))
        cos_th = math.cos(float(node.state[1].item()))
        needs_rollout = mix < 1.0 or (config.leaf_vnet_gate_cos is not None and cos_th <= config.leaf_vnet_gate_cos)
        if value_net is None:
            needs_rollout = True
        if needs_rollout:
            rollout_val = _heuristic_rollout(
                world_model_wrapper, node.state, config.rollout_depth, config.action_repeat, config.gamma, g_over_l=g_over_l
            )
        use_rollout_only = config.leaf_vnet_gate_cos is not None and cos_th <= config.leaf_vnet_gate_cos
        if value_net is None:
            leaf_value = rollout_val
        elif use_rollout_only:
            leaf_value = rollout_val
        elif mix >= 1.0:
            leaf_value = vnet_value
        else:
            leaf_value = mix * vnet_value + (1.0 - mix) * rollout_val
        leaf_value -= _cart_drift_leaf_bias(node.state, config.cart_drift_penalty)
        last_leaf = {
            'theta': float(node.state[1].item()),
            'x': float(node.state[0].item()),
            'cos_theta': cos_th,
            'vnet_value': vnet_value,
            'rollout_value': rollout_val,
            'leaf_value': leaf_value,
        }

        g_value = leaf_value
        for node_bp in reversed(path):
            if node_bp.parent is not None:
                g_value = node_bp.edge_reward + gamma_k * g_value
            node_bp.N += 1
            node_bp.W += g_value
            node_bp.Q = node_bp.W / node_bp.N

    if not root.children:
        return 0.0, {'mode': 'sd_mcts', 'candidates': [], 'leaf': last_leaf}

    ranked = []
    for action_idx, child in root.children.items():
        ranked.append({
            'action': float(discrete_actions[action_idx]),
            'visits': int(child.N),
            'q': float(child.Q),
            'edge_reward': float(child.edge_reward),
            'next_x': float(child.state[0].item()),
            'next_theta': float(child.state[1].item()),
            'next_cos_theta': float(math.cos(float(child.state[1].item()))),
            'next_vnet': _value_of_state(value_net, child.state),
        })
    ranked.sort(key=lambda row: (-row['visits'], -row['q']))
    chosen = ranked[0]
    trace = {
        'mode': 'sd_mcts',
        'num_simulations': int(config.num_simulations),
        'action_repeat': int(config.action_repeat),
        'chosen_action': chosen['action'],
        'current_vnet': _value_of_state(value_net, root.state),
        'candidates': ranked,
        'leaf': last_leaf,
    }
    return chosen['action'], trace


def fast_action_with_trace(state_np, world_model_wrapper, discrete_actions, value_net, config):
    root = torch.tensor(state_np, dtype=torch.float32)
    gamma_k = config.gamma ** config.action_repeat
    candidates = []
    for action in discrete_actions:
        next_state, edge_reward, _ = model_forward_k(world_model_wrapper, root, float(action), config.action_repeat, config.gamma)
        next_vnet = _value_of_state(value_net, next_state)
        penalty = _cart_drift_leaf_bias(next_state, config.cart_drift_penalty)
        score = edge_reward + gamma_k * next_vnet - penalty
        candidates.append({
            'action': float(action),
            'score': float(score),
            'edge_reward': float(edge_reward),
            'next_vnet': float(next_vnet),
            'penalty': float(penalty),
            'next_x': float(next_state[0].item()),
            'next_theta': float(next_state[1].item()),
            'next_cos_theta': float(math.cos(float(next_state[1].item()))),
        })
    candidates.sort(key=lambda row: (-row['score'], -row['edge_reward']))
    chosen = candidates[0]
    trace = {
        'mode': 'fast',
        'action_repeat': int(config.action_repeat),
        'chosen_action': chosen['action'],
        'current_vnet': _value_of_state(value_net, root),
        'candidates': candidates,
    }
    return chosen['action'], trace


def heuristic_action_with_trace(state_np, world_model_wrapper, discrete_actions, value_net, config, g_over_l):
    root = torch.tensor(state_np, dtype=torch.float32)
    candidates = []
    rollout_depth = max(6, config.rollout_depth)
    gamma_k = config.gamma ** config.action_repeat
    for action in discrete_actions:
        next_state, edge_reward, _ = model_forward_k(world_model_wrapper, root, float(action), config.action_repeat, config.gamma)
        rollout_return = _heuristic_rollout(
            world_model_wrapper,
            next_state,
            rollout_depth=rollout_depth,
            action_repeat=config.action_repeat,
            gamma=config.gamma,
            g_over_l=g_over_l,
        )
        next_x = float(next_state[0].item())
        next_theta = float(next_state[1].item())
        next_cos = float(math.cos(next_theta))
        next_thdot = float(next_state[3].item())
        next_theta_abs = abs(((next_theta + math.pi) % (2 * math.pi)) - math.pi)
        score = edge_reward + gamma_k * rollout_return - 0.08 * abs(next_x) - 0.04 * abs(next_thdot)
        candidates.append({
            'action': float(action),
            'score': float(score),
            'edge_reward': float(edge_reward),
            'rollout_return': float(rollout_return),
            'next_x': next_x,
            'next_theta': next_theta,
            'next_cos_theta': next_cos,
            'next_theta_dot': next_thdot,
            'next_theta_abs': float(next_theta_abs),
        })
    candidates.sort(key=lambda row: (-row['score'], -row['rollout_return'], -row['next_cos_theta']))
    chosen = candidates[0]
    trace = {
        'mode': 'heuristic',
        'action_repeat': int(config.action_repeat),
        'rollout_depth': int(rollout_depth),
        'chosen_action': chosen['action'],
        'current_vnet': 0.0,
        'candidates': candidates,
    }
    return chosen['action'], trace


def available_checkpoints():
    items = {}
    for path in sorted(CHECKPOINT_DIR.glob('seed*.pt')):
        try:
            seed = int(path.stem.replace('seed', ''))
        except ValueError:
            continue
        items[seed] = path
    return items


def load_summary():
    if not SUMMARY_PATH.exists():
        return {}
    with SUMMARY_PATH.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    result = {}
    for row in payload.get('per_seed', []):
        result[int(row['seed'])] = {
            'mean_return': float(row['eval']['mean']),
            'std_return': float(row['eval']['std']),
            'val_loss': float(row['val_loss']),
            'mp_err_percent': float(row['mp_err_percent']),
        }
    return result


class DemoRuntime:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints = available_checkpoints()
        if not self.checkpoints:
            raise FileNotFoundError(f'No demo checkpoints found in {self.checkpoint_dir}')
        self.summary = load_summary()
        self.control_config = ControllerConfig()
        self.fast_config = ControllerConfig(num_simulations=1, rollout_depth=0)
        self.heuristic_config = ControllerConfig(num_simulations=1, rollout_depth=0)
        self.mode = 'sd_mcts'
        self.seed = sorted(self.checkpoints)[0]
        self.env = None
        self.base_model_state = None
        self.model = None
        self.vnet = None
        self.wrapper = None
        self.actions = get_discrete_actions(11)
        self.selected_cfg = {}
        self.initial_params = {}
        self.episode_return = 0.0
        self.env_step = 0
        self.decision_count = 0
        self.done = False
        self.last_trace = {}
        self.last_action = 0.0
        self.state = np.zeros(4, dtype=np.float32)
        self.history = {'x': [], 'cos_theta': [], 'return': [], 'action': []}
        self.backend = 'dm_control' if suite is not None else 'fallback'
        self.load_checkpoint(self.seed)

    def load_checkpoint(self, seed):
        seed = int(seed)
        path = self.checkpoints[seed]
        payload = torch.load(path, map_location='cpu', weights_only=False)
        self.model = DMControlCartPoleSDModel(learn_params=True, prior_noise=0.3, mp_prior_noise=0.15, mp_clamp_min=0.03)
        self.model.load_state_dict(payload['model'])
        self.vnet = SwingupValueNet()
        self.vnet.load_state_dict(payload['vnet'])
        self.vnet.eval()
        self.wrapper = DMControlCartPoleWrapper(self.model)
        self.base_model_state = copy.deepcopy(payload['model'])
        self.selected_cfg = copy.deepcopy(payload.get('selected_cfg') or {})
        self.initial_params = self.model.get_learned_params()
        self.seed = seed
        self.reset_episode()

    def _close_env(self):
        if self.env is None:
            return
        close_fn = getattr(self.env, 'close', None)
        if callable(close_fn):
            close_fn()
        self.env = None

    def reset_episode(self):
        self._close_env()
        if suite is not None:
            self.backend = 'dm_control'
            self.env = suite.load('cartpole', 'swingup', task_kwargs={'random': self.seed + 1000})
        else:
            self.backend = 'fallback'
            self.env = FallbackCartpoleEnv(self.model, seed=self.seed + 1000)
        ts = self.env.reset()
        self.state = DMControlCartPoleSDModel.obs_to_state(flatten_obs(ts.observation))
        self.episode_return = 0.0
        self.env_step = 0
        self.decision_count = 0
        self.done = False
        self.last_trace = {}
        self.last_action = 0.0
        self.history = {'x': [float(self.state[0])], 'cos_theta': [float(math.cos(float(self.state[1])))], 'return': [0.0], 'action': []}
        return self.get_snapshot()

    def restore_checkpoint_params(self):
        self.model.load_state_dict(self.base_model_state)
        self.wrapper = DMControlCartPoleWrapper(self.model)
        return self.get_snapshot()

    def set_mode(self, mode):
        if mode not in ('sd_mcts', 'fast', 'heuristic'):
            raise ValueError(f'Unsupported mode: {mode}')
        self.mode = mode

    def set_num_simulations(self, num_simulations):
        self.control_config.num_simulations = max(1, int(num_simulations))

    def apply_counterfactual(self, gravity, masscart, masspole, length):
        self.model.set_learned_params(gravity=gravity, masscart=masscart, masspole=masspole, length=length)
        self.wrapper = DMControlCartPoleWrapper(self.model)
        return self.get_snapshot()

    def _current_g_over_l(self):
        params = self.model.get_learned_params()
        return params['gravity'] / max(params['length'], 1e-3)

    def step_decision(self):
        if self.done:
            return self.get_snapshot()

        if self.mode == 'sd_mcts':
            cfg = self.control_config
            action, trace = mcts_action_with_trace(self.state, self.wrapper, self.actions, self.vnet, cfg, self._current_g_over_l())
        elif self.mode == 'fast':
            cfg = self.fast_config
            action, trace = fast_action_with_trace(self.state, self.wrapper, self.actions, self.vnet, cfg)
        else:
            cfg = self.heuristic_config
            action, trace = heuristic_action_with_trace(self.state, self.wrapper, self.actions, self.vnet, cfg, self._current_g_over_l())
        self.last_action = float(action)
        self.last_trace = trace

        for _ in range(cfg.action_repeat):
            ts = self.env.step(np.array([action], dtype=np.float32))
            self.episode_return += float(ts.reward if ts.reward is not None else 0.0)
            self.env_step += 1
            if ts.last():
                self.done = True
                break

        if not self.done:
            self.state = DMControlCartPoleSDModel.obs_to_state(flatten_obs(ts.observation))
        self.decision_count += 1
        self.history['action'].append(float(action))
        self.history['x'].append(float(self.state[0]))
        self.history['cos_theta'].append(float(math.cos(float(self.state[1]))))
        self.history['return'].append(float(self.episode_return))
        return self.get_snapshot()

    def get_snapshot(self):
        return {
            'seed': self.seed,
            'mode': self.mode,
            'state': self.state.copy(),
            'params': self.model.get_learned_params(),
            'selected_cfg': copy.deepcopy(self.selected_cfg),
            'summary': copy.deepcopy(self.summary.get(self.seed, {})),
            'episode_return': float(self.episode_return),
            'env_step': int(self.env_step),
            'decision_count': int(self.decision_count),
            'done': bool(self.done),
            'last_trace': copy.deepcopy(self.last_trace),
            'last_action': float(self.last_action),
            'num_simulations': int(self.control_config.num_simulations),
            'history': copy.deepcopy(self.history),
            'initial_params': copy.deepcopy(self.initial_params),
            'backend': self.backend,
        }


def run_headless_smoke(seed=42, decisions=3):
    runtime = DemoRuntime()
    runtime.load_checkpoint(seed)
    snapshots = []
    for mode in ('sd_mcts', 'fast', 'heuristic'):
        runtime.reset_episode()
        runtime.set_mode(mode)
        for _ in range(decisions):
            snapshots.append(runtime.step_decision())
            if snapshots[-1]['done']:
                break
    runtime._close_env()
    return snapshots
