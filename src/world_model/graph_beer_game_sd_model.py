"""Graph-generic Beer Game SD model.

``GenericGraphBeerGameSDModel`` is the single model family for T1--T5 under
the design §8 protocol (generic layout, per-node order grid, STE step gate,
shared or node-specific demand profile). ``GraphBeerGameSDModel`` keeps the
classic 6-slot serial layout for the historical T1 entry points (gate G-BT2).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.graph_beer_game import SupplyGraph, generic_state_dim, parse_generic_obs, serial_topology
from src.world_model.beer_game_sd_model import BeerGameSDModel


def theta_dimension(n_retail: int, shared: bool = True) -> int:
    """Shared demand profile is 3 scalars regardless of |V|."""
    return 3 if shared else 3 * int(n_retail)


class GraphBeerGameSDModel(BeerGameSDModel):
    """Serial graph SD model; identical to BeerGameSDModel on T1."""

    def __init__(self, graph: Optional[SupplyGraph] = None, **kwargs):
        if graph is None:
            graph = serial_topology(4)
        source = graph.source_nodes[0]
        kwargs.setdefault("n_echelons", graph.n_nodes)
        kwargs.setdefault("order_levels", graph.order_levels)
        kwargs.setdefault("O_max", graph.O_max)
        kwargs.setdefault("demand_low", float(graph.demand_low))
        kwargs.setdefault("demand_high", float(graph.demand_high))
        kwargs.setdefault("demand_step_time", graph.demand_step_time)
        kwargs.setdefault("production_capacity", int(source.capacity))
        kwargs.setdefault("learn_demand", True)
        super().__init__(**kwargs)
        self.graph_id = graph.id
        self.shared_demand = True
        self.n_retail = len(graph.retail_nodes)

    def theta_dim(self) -> int:
        return theta_dimension(self.n_retail, shared=True)

    def get_learned_params(self) -> dict:
        params = super().get_learned_params()
        params["theta_dim"] = self.theta_dim()
        params["graph_id"] = self.graph_id
        params["shared_demand"] = self.shared_demand
        return params


class GenericGraphBeerGameSDModel(nn.Module):
    """Differentiable DAG stock-flow model on the generic observation layout.

    Semantics follow ``BeerGameSDModel``: this period's retail demand is read
    from the observation; the learned profile only generates next period's
    demand slots. ``shared_demand=False`` gives each retail node its own
    (low, high, step_time) triple.
    """

    def __init__(self, graph: SupplyGraph, shared_demand: bool = True, learn_demand: bool = True,
                 prior_belief: bool = False):
        super().__init__()
        self.graph = graph
        self.shared_demand = bool(shared_demand)
        self.learn_demand = bool(learn_demand)
        self.n_nodes = graph.n_nodes
        self.n_edges = len(graph.edges)
        self.n_retail = len(graph.retail_nodes)
        self.order_levels = graph.order_levels
        self.state_dim = generic_state_dim(graph)
        self.register_buffer("order_quantities", torch.tensor(graph.order_quantities()))
        self.register_buffer("capacity", torch.tensor([n.capacity for n in graph.source_nodes], dtype=torch.float32))
        self.register_buffer("holding", torch.tensor([n.holding for n in graph.nodes], dtype=torch.float32))
        self.register_buffer("backlog_cost", torch.tensor(
            [n.backlog_cost for n in graph.retail_nodes] + [e.backlog_cost for e in graph.edges], dtype=torch.float32))
        self.register_buffer("true_demand_step_time", torch.tensor(float(graph.demand_step_time)))
        self._build_index_buffers()
        self.intervention: Optional[dict] = None  # structural edit mirroring GraphBeerGameEnv.intervention

        k = 1 if self.shared_demand else self.n_retail
        step = np.full(k, float(graph.demand_step_time))
        high = np.full(k, float(graph.demand_high))
        if prior_belief:  # same perturbation as BeerGameSDModel, per parameter slot
            step = step + np.random.uniform(-3.0, 3.0, size=k)
            high = high * np.random.uniform(0.7, 1.3, size=k)
        low_raw = torch.tensor(np.log(np.expm1(np.full(k, float(graph.demand_low)))), dtype=torch.float32)
        high_raw = torch.tensor(np.log(np.expm1(high)), dtype=torch.float32)
        step_t = torch.tensor(step, dtype=torch.float32)
        if learn_demand:
            self._demand_low_raw = nn.Parameter(low_raw)
            self._demand_high_raw = nn.Parameter(high_raw)
            self._demand_step_time = nn.Parameter(step_t)
        else:
            self.register_buffer("_demand_low_raw", low_raw)
            self.register_buffer("_demand_high_raw", high_raw)
            self.register_buffer("_demand_step_time", step_t)

    def theta_dim(self) -> int:
        return theta_dimension(self.n_retail, shared=self.shared_demand)

    def get_demand(self, t: torch.Tensor) -> torch.Tensor:
        """(batch, n_retail) demand at period t; STE hard gate as in BeerGameSDModel."""
        low = F.softplus(self._demand_low_raw)[None]
        high = F.softplus(self._demand_high_raw)[None]
        step = self._demand_step_time[None]
        tt = t[:, None]
        gate_soft = torch.sigmoid((tt + 0.5 - step) * 2.0)
        gate_hard = (tt >= step).to(gate_soft.dtype)
        gate = gate_hard.detach() - gate_soft.detach() + gate_soft
        return (low + (high - low) * gate).expand(t.shape[0], self.n_retail)

    def _build_index_buffers(self):
        g = self.graph
        node_idx = g.node_index()
        assert all(n.kind != "retail" or not any(e.u == n.id for e in g.edges) for n in g.nodes)
        L, tau = {e.L for e in g.edges}, {e.tau for e in g.edges}
        assert len(L) == 1 and len(tau) == 1, "vectorised forward needs uniform L and tau"
        self.L, self.tau = L.pop(), tau.pop()
        a_in = torch.zeros(self.n_nodes, self.n_edges)   # edge -> customer node
        a_out = torch.zeros(self.n_nodes, self.n_edges)  # edge -> supplier node
        for ei, e in enumerate(g.edges):
            a_in[node_idx[e.v], ei] = 1.0
            a_out[node_idx[e.u], ei] = 1.0
        s_in = torch.zeros(self.n_nodes, len(g.source_nodes))
        r_in = torch.zeros(self.n_nodes, self.n_retail)
        for si, n in enumerate(g.source_nodes):
            s_in[node_idx[n.id], si] = 1.0
        for ri, n in enumerate(g.retail_nodes):
            r_in[node_idx[n.id], ri] = 1.0
        indeg = a_in.sum(1)
        self.register_buffer("a_in", a_in)
        self.register_buffer("a_out", a_out)
        self.register_buffer("s_in", s_in)
        self.register_buffer("r_in", r_in)
        self.register_buffer("edge_target", torch.tensor([node_idx[e.v] for e in g.edges], dtype=torch.long))
        self.register_buffer("edge_share", torch.tensor([1.0 / indeg[node_idx[e.v]] for e in g.edges]))
        self.register_buffer("source_idx", torch.tensor([node_idx[n.id] for n in g.source_nodes], dtype=torch.long))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        b, n, E, R, S = state.shape[0], self.n_nodes, self.n_edges, self.n_retail, len(self.graph.source_nodes)
        action = action.view(b, n).long()
        k = 0
        inv = state[:, k:k + n]; k += n
        back = state[:, k:k + R + E]; k += R + E
        ship = state[:, k:k + E * self.L].view(b, E, self.L); k += E * self.L
        order = state[:, k:k + E * self.tau].view(b, E, self.tau); k += E * self.tau
        src = state[:, k:k + 2 * S].view(b, S, 2); k += 2 * S
        t = state[:, k]; k += 1
        demand = state[:, k:k + R]
        zero_e = torch.zeros((b, E, 1), device=state.device, dtype=state.dtype)
        # 1. arrivals and pipeline shifts
        inv = inv + ship[:, :, 0] @ self.a_in.T + src[:, :, 0] @ self.s_in.T
        ship = torch.cat([ship[:, :, 1:], zero_e], 2)
        src = torch.cat([src[:, :, 1:], torch.zeros((b, S, 1), device=state.device, dtype=state.dtype)], 2)
        # 2. incoming orders
        back = back + torch.cat([demand, order[:, :, 0]], 1)
        order = torch.cat([order[:, :, 1:], zero_e], 2)
        # 3. shipping: retail fulfils customer backlog; suppliers allocate by backlog share
        inv_r = inv @ self.r_in
        q_r = torch.minimum(inv_r, back[:, :R])
        bs = back[:, R:]
        total = bs @ self.a_out.T                       # (b, n) backlog facing each supplier
        supply = torch.minimum(inv, total)
        share = bs / (total @ self.a_out).clamp(min=1e-6)
        q_e = torch.where((total @ self.a_out) > 0, (supply @ self.a_out) * share, torch.zeros_like(bs))
        ship = torch.cat([ship[:, :, :-1], ship[:, :, -1:] + q_e.unsqueeze(2)], 2)
        back = torch.clamp(torch.cat([back[:, :R] - q_r, bs - q_e], 1), min=0.0)
        inv = torch.clamp(inv - q_r @ self.r_in.T - q_e @ self.a_out.T, min=0.0)
        # 4. new orders and production
        oq = self.order_quantities[torch.arange(n, device=state.device), action]
        capacity, new_orders = self._edited(t, oq)
        q_src = torch.minimum(oq[:, self.source_idx], capacity)
        src = torch.cat([src[:, :, :-1], q_src.unsqueeze(2)], 2)
        order = torch.cat([order[:, :, :-1], new_orders.unsqueeze(2)], 2)
        next_t = t + 1
        return torch.cat([inv, back, ship.reshape(b, -1), order.reshape(b, -1), src.reshape(b, -1),
                          next_t.unsqueeze(1), self.get_demand(next_t)], 1)

    def _edited(self, t: torch.Tensor, oq: torch.Tensor):
        """Apply the structural intervention (if any) to capacity and edge orders."""
        g, iv = self.graph, self.intervention
        capacity = self.capacity.expand(t.shape[0], -1)
        new_orders = oq[:, self.edge_target] * self.edge_share
        if iv is None:
            return capacity, new_orders
        if iv["type"] == "capacity_shock":
            si = [n.id for n in g.source_nodes].index(iv["node"])
            mask = F.one_hot(torch.tensor(si), len(g.source_nodes)).to(oq.dtype)[None]
            factor = torch.where((t >= iv["t_start"])[:, None], torch.full_like(mask, iv["factor"]), torch.ones_like(mask))
            capacity = capacity * (1 - mask + mask * factor)
        elif iv["type"] == "comm_severed":
            ei = [(e.u, e.v) for e in g.edges].index(tuple(iv["edge"]))
            active = ((t >= iv["t_start"]) & (t <= iv["t_end"]))[:, None]
            mask = F.one_hot(torch.tensor(ei), self.n_edges).to(oq.dtype)[None]
            new_orders = torch.where(active, new_orders * (1 - mask), new_orders)
        return capacity, new_orders

    def predict_reward(self, state: torch.Tensor, action: torch.Tensor,
                       next_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Same arithmetic as ``GraphBeerGameEnv.period_reward``."""
        if next_state is None:
            next_state = self.forward(state, action)
        p = parse_generic_obs(self.graph, next_state)
        holding = (self.holding * torch.clamp(p["inventory"], min=0)).sum(1)
        return -(holding + (self.backlog_cost * p["backlog"]).sum(1))

    def get_learned_params(self) -> dict:
        with torch.no_grad():
            low = F.softplus(self._demand_low_raw).cpu().numpy()
            high = F.softplus(self._demand_high_raw).cpu().numpy()
            step = self._demand_step_time.cpu().numpy()
        squeeze = (lambda a: float(a[0])) if self.shared_demand else (lambda a: a.tolist())
        return {
            "demand_low": squeeze(low),
            "demand_high": squeeze(high),
            "demand_step_time": squeeze(step),
            "true_step_time": float(self.true_demand_step_time),
            "theta_dim": self.theta_dim(),
            "graph_id": self.graph.id,
            "shared_demand": self.shared_demand,
        }


def build_graph_sd_model(graph: SupplyGraph, **kwargs) -> nn.Module:
    """§8.3: every topology uses the generic family."""
    return GenericGraphBeerGameSDModel(graph, **kwargs)
