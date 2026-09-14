"""Graph-generic Beer Game environment.

T1 (serial 4) reproduces ``BeerGameEnv`` state and step bit-for-bit on a
fixed action sequence (gate G-BT1). Larger topologies T2--T5 are defined by
YAML and are not used to overwrite any published T1 result.

Supply/demand pressure rule (design §8.2): every node quantity is scaled by
its routed high-phase downstream demand ``D_v`` with the T1 ratios
``O_max = 2 D_v``, ``capacity = 2.5 D_v``, ``I0 = 1.5 D_v``, pipelines
``0.5 F_e``. On T1 this reproduces the classic parameters exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from gymnasium import spaces


@dataclass
class GraphNode:
    id: str
    kind: str  # retail | warehouse | source
    holding: float = 0.5
    backlog_cost: float = 1.0
    index: int = 0
    # Derived by the pressure rule; not read from YAML.
    demand: float = field(default=0.0, init=False)
    O_max: float = field(default=0.0, init=False)
    capacity: float = field(default=0.0, init=False)
    initial_inventory: float = field(default=0.0, init=False)
    initial_pipeline: float = field(default=0.0, init=False)


@dataclass
class GraphEdge:
    u: str  # supplier
    v: str  # customer
    L: int = 2
    tau: int = 2
    backlog_cost: float = 1.0
    index: int = 0
    flow: float = field(default=0.0, init=False)
    initial_pipeline: float = field(default=0.0, init=False)


@dataclass
class SupplyGraph:
    id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    demand_low: int = 4
    demand_high: int = 8
    demand_step_time: int = 5
    T: int = 36
    # T1 reference scale: ratios to demand_high define the pressure rule.
    initial_inventory: int = 12
    initial_pipeline: int = 4
    O_max: int = 16
    source_capacity: int = 20
    order_levels: int = 9
    shared_demand: bool = True

    def __post_init__(self):
        self._apply_pressure_rule()

    def _apply_pressure_rule(self):
        """Route high-phase demand upstream with equal split over suppliers."""
        indeg = {n.id: 0 for n in self.nodes}
        for e in self.edges:
            indeg[e.v] += 1
        demand = {n.id: float(self.demand_high) if n.kind == "retail" else 0.0 for n in self.nodes}
        for _ in range(len(self.nodes)):
            new = {n.id: float(self.demand_high) if n.kind == "retail" else 0.0 for n in self.nodes}
            for e in self.edges:
                new[e.u] += demand[e.v] / indeg[e.v]
            demand = new
        ref = float(self.demand_high)
        for n in self.nodes:
            n.demand = demand[n.id]
            n.O_max = self.O_max / ref * n.demand
            n.capacity = self.source_capacity / ref * n.demand if n.kind == "source" else float("inf")
            n.initial_inventory = self.initial_inventory / ref * n.demand
            n.initial_pipeline = self.initial_pipeline / ref * n.demand
        for e in self.edges:
            e.flow = demand[e.v] / indeg[e.v]
            e.initial_pipeline = self.initial_pipeline / ref * e.flow

    def node_index(self) -> Dict[str, int]:
        return {n.id: n.index for n in self.nodes}

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def retail_nodes(self):
        return [n for n in self.nodes if n.kind == "retail"]

    @property
    def source_nodes(self):
        return [n for n in self.nodes if n.kind == "source"]

    def order_quantities(self) -> np.ndarray:
        """(n_nodes, order_levels) per-node order grid."""
        return np.stack([np.linspace(0, n.O_max, self.order_levels) for n in self.nodes]).astype(np.float32)

    def is_serial_uniform(self) -> bool:
        if len(self.edges) != self.n_nodes - 1:
            return False
        return all(e.L == 2 and e.tau == 2 for e in self.edges) and len(self.retail_nodes) == 1 and len(self.source_nodes) == 1


def load_supply_graph(path) -> SupplyGraph:
    raw = yaml.safe_load(Path(path).read_text())
    nodes = [GraphNode(index=i, **n) for i, n in enumerate(raw["nodes"])]
    edges = [GraphEdge(index=i, **e) for i, e in enumerate(raw["edges"])]
    meta = raw.get("meta", {})
    return SupplyGraph(
        id=raw["id"],
        nodes=nodes,
        edges=edges,
        demand_low=int(meta.get("demand_low", 4)),
        demand_high=int(meta.get("demand_high", 8)),
        demand_step_time=int(meta.get("demand_step_time", 5)),
        T=int(meta.get("T", 36)),
        initial_inventory=int(meta.get("initial_inventory", 12)),
        initial_pipeline=int(meta.get("initial_pipeline", 4)),
        O_max=int(meta.get("O_max", 16)),
        source_capacity=int(meta.get("source_capacity", 20)),
        order_levels=int(meta.get("order_levels", 9)),
        shared_demand=bool(meta.get("shared_demand", True)),
    )


def serial_topology(n: int, **meta) -> SupplyGraph:
    names = []
    if n == 4:
        names = ["Retailer", "Wholesaler", "Distributor", "Factory"]
        kinds = ["retail", "warehouse", "warehouse", "source"]
    else:
        names = [f"n{i}" for i in range(n)]
        kinds = ["retail"] + ["warehouse"] * (n - 2) + ["source"]
    nodes = [GraphNode(id=names[i], kind=kinds[i], holding=0.5, index=i) for i in range(n)]
    edges = [
        GraphEdge(u=names[i + 1], v=names[i], L=2, tau=2, backlog_cost=1.0, index=i)
        for i in range(n - 1)
    ]
    return SupplyGraph(id=f"serial_{n}", nodes=nodes, edges=edges, **meta)


class GraphBeerGameEnv:
    """Stock-flow environment on an arbitrary supply DAG.

    Generic layout: ``[I_v | B_slots | ship pipes | order pipes | source pipes | t | D_r...]``
    with one backlog slot per retail node followed by one per edge (at the
    supplier), and one demand slot per retail node. ``serial_layout`` selects
    the classic 6-slot echelon layout (auto for uniform serial chains).
    """

    def __init__(self, graph: SupplyGraph, seed: Optional[int] = None,
                 allocation: str = "backlog_proportional", serial_layout: Optional[bool] = None):
        self.graph = graph
        self.allocation = allocation
        self.rng = np.random.default_rng(seed)
        self.serial_layout = graph.is_serial_uniform() if serial_layout is None else bool(serial_layout)
        assert not self.serial_layout or graph.is_serial_uniform()
        # Test-time structural intervention (§8.6 item 3); see ``apply_intervention``.
        self.intervention: Optional[dict] = None
        self.order_quantities = graph.order_quantities()
        self.action_space = spaces.MultiDiscrete([graph.order_levels] * graph.n_nodes)
        self._init_state()
        self.observation_space = spaces.Box(
            low=np.zeros(self.state_dim, dtype=np.float32),
            high=np.full(self.state_dim, 50.0 * max(n.demand for n in graph.nodes), dtype=np.float32),
            dtype=np.float32,
        )

    def _init_state(self):
        g = self.graph
        self.inventory = np.array([n.initial_inventory for n in g.nodes], dtype=np.float32)
        self.backlog = np.zeros(len(g.edges) + len(g.retail_nodes), dtype=np.float32)
        self.ship = [np.full(e.L, e.initial_pipeline, dtype=np.float32) for e in g.edges]
        self.order_pipe = [np.full(e.tau, e.initial_pipeline, dtype=np.float32) for e in g.edges]
        self.source_ship = {
            n.id: np.full(2, n.initial_pipeline, dtype=np.float32) for n in g.source_nodes
        }
        self.t = 0
        self.current_demand = float(g.demand_low)
        self.total_cost = 0.0
        self.order_history = [[] for _ in range(g.n_nodes)]
        self.inventory_history = [[] for _ in range(g.n_nodes)]
        self._factory_orders = np.full(2, g.source_nodes[0].initial_pipeline, dtype=np.float32)
        if self.serial_layout:
            self.backlog = np.zeros(g.n_nodes, dtype=np.float32)

    @property
    def state_dim(self):
        if self.serial_layout:
            return self.graph.n_nodes * 6 + 2
        return generic_state_dim(self.graph)

    def _is_serial_uniform(self):
        return self.serial_layout

    def _demand(self):
        g = self.graph
        base = g.demand_low if self.t < g.demand_step_time else g.demand_high
        return float(base)

    def _get_obs(self) -> np.ndarray:
        if self.serial_layout:
            return self._serial_obs()
        g = self.graph
        parts = [self.inventory, self.backlog, *self.ship, *self.order_pipe]
        parts += [self.source_ship[n.id] for n in g.source_nodes]
        parts.append(np.array([self.t], dtype=np.float32))
        parts.append(np.full(len(g.retail_nodes), self.current_demand, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def load_obs(self, obs: np.ndarray):
        """Restore internals from an observation."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        g = self.graph
        if self.serial_layout:
            n = g.n_nodes
            for i in range(n):
                base = i * 6
                self.inventory[i] = obs[base]
                self.backlog[i] = obs[base + 1]
                if i < n - 1:
                    self.ship[i][0] = obs[base + 2]
                    self.ship[i][1] = obs[base + 3]
                    self.order_pipe[i][0] = obs[base + 4]
                    self.order_pipe[i][1] = obs[base + 5]
                else:
                    node = g.nodes[-1]
                    self.source_ship[node.id][0] = obs[base + 2]
                    self.source_ship[node.id][1] = obs[base + 3]
                    self._factory_orders[0] = obs[base + 4]
                    self._factory_orders[1] = obs[base + 5]
            self.t = int(round(float(obs[-2])))
            self.current_demand = float(obs[-1])
            return
        p = parse_generic_obs(g, obs)
        self.inventory = p["inventory"].copy()
        self.backlog = p["backlog"].copy()
        self.ship = [x.copy() for x in p["ship"]]
        self.order_pipe = [x.copy() for x in p["order"]]
        for n, pipe in zip(g.source_nodes, p["source"]):
            self.source_ship[n.id] = pipe.copy()
        self.t = int(round(float(p["t"])))
        self.current_demand = float(p["demand"][0])

    def _serial_obs(self) -> np.ndarray:
        """Match BeerGameEnv layout: per echelon I,B,ship0,ship1,order0,order1 + t,D."""
        g = self.graph
        obs = []
        n = g.n_nodes
        for i, node in enumerate(g.nodes):
            I = self.inventory[i]
            if i < n - 1:
                B = self.backlog[i]
                ship = self.ship[i]
                order = self.order_pipe[i]
            else:
                B = self.backlog[i] if i < len(self.backlog) else 0.0
                ship = self.source_ship[node.id]
                order = self._factory_orders
            obs.extend([I, B, float(ship[0]), float(ship[1]), float(order[0]), float(order[1])])
        obs.extend([float(self.t), float(self.current_demand)])
        return np.array(obs, dtype=np.float32)

    def _period_cost(self, inventory, backlog) -> float:
        g = self.graph
        holding = sum(n.holding * max(0.0, float(inventory[n.index])) for n in g.nodes)
        if self.serial_layout:
            return holding + float(np.sum(backlog))
        costs = [n.backlog_cost for n in g.retail_nodes] + [e.backlog_cost for e in g.edges]
        return holding + float(sum(c * float(b) for c, b in zip(costs, backlog)))

    def period_reward(self, obs: np.ndarray) -> float:
        """Reward implied by an observation; identical arithmetic to ``step``."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.serial_layout:
            return -self._period_cost(obs[0::6][: self.graph.n_nodes], obs[1::6][: self.graph.n_nodes])
        p = parse_generic_obs(self.graph, obs)
        return -self._period_cost(p["inventory"], p["backlog"])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        g = self.graph
        action = np.asarray(action).reshape(-1).astype(int)
        orders = self.order_quantities[np.arange(g.n_nodes), action]
        demand = self._demand()
        self.current_demand = demand

        if self.serial_layout:
            return self._step_serial(orders, demand)

        arrivals = {}
        for e, pipe in zip(g.edges, self.ship):
            arrivals[e.v] = arrivals.get(e.v, 0.0) + float(pipe[0])
            pipe[:-1] = pipe[1:]
            pipe[-1] = 0.0
        for n in g.source_nodes:
            pipe = self.source_ship[n.id]
            arrivals[n.id] = arrivals.get(n.id, 0.0) + float(pipe[0])
            pipe[:-1] = pipe[1:]
            pipe[-1] = 0.0
        for i, n in enumerate(g.nodes):
            self.inventory[i] += arrivals.get(n.id, 0.0)

        incoming = np.zeros_like(self.backlog)
        for r_i, retail in enumerate(g.retail_nodes):
            incoming[r_i] += demand
        for e, pipe in zip(g.edges, self.order_pipe):
            incoming[len(g.retail_nodes) + e.index] += float(pipe[0])
            pipe[:-1] = pipe[1:]
            pipe[-1] = 0.0
        self.backlog += incoming

        idx = g.node_index()
        shipments = np.zeros(g.n_nodes, dtype=np.float32)
        for i, n in enumerate(g.nodes):
            out_edges = [e for e in g.edges if e.u == n.id]
            if not out_edges:
                retail_index = next((j for j, retail in enumerate(g.retail_nodes) if retail.id == n.id), None)
                if retail_index is None:
                    continue
                need = float(self.backlog[retail_index])
                ship = min(self.inventory[i], need)
                self.inventory[i] -= ship
                self.backlog[retail_index] -= ship
                shipments[i] = ship
                continue
            total_b = 0.0
            shares = []
            for e in out_edges:
                b = float(self.backlog[len(g.retail_nodes) + e.index])
                shares.append((e, b))
                total_b += b
            available = float(self.inventory[i])
            proportional_supply = min(available, total_b)
            for e, b in shares:
                # Allocate against a common pre-shipment inventory. No demand
                # means no shipment; edge order must not define priority here.
                if self.allocation == "backlog_proportional":
                    ship = proportional_supply * b / total_b if total_b > 0 else 0.0
                else:
                    ship = min(available, b)
                ship = max(0.0, ship)
                self.inventory[i] -= ship
                available -= ship
                bi = len(g.retail_nodes) + e.index
                self.backlog[bi] = max(0.0, self.backlog[bi] - ship)
                self.ship[e.index][-1] += ship
                shipments[i] += ship

        for n in g.source_nodes:
            self.source_ship[n.id][-1] = min(orders[idx[n.id]], self._capacity(n))

        for i, n in enumerate(g.nodes):
            out = [e for e in g.edges if e.v == n.id]
            if not out:
                continue
            share = orders[i] / len(out)
            for e in out:
                self.order_pipe[e.index][-1] = 0.0 if self._severed(e) else share

        period_cost = self._period_cost(self.inventory, self.backlog)
        self.total_cost += period_cost
        self.t += 1
        self.current_demand = self._demand()
        for i in range(g.n_nodes):
            self.order_history[i].append(orders[i])
            self.inventory_history[i].append(self.inventory[i])
        terminated = self.t >= g.T
        info = {"period_cost": period_cost, "total_cost": self.total_cost, "customer_demand": demand, "orders": orders}
        return self._get_obs(), -period_cost, terminated, False, info

    def _capacity(self, node) -> float:
        iv = self.intervention
        if iv and iv["type"] == "capacity_shock" and iv["node"] == node.id and self.t >= iv["t_start"]:
            return node.capacity * iv["factor"]
        return node.capacity

    def _severed(self, edge) -> bool:
        iv = self.intervention
        return bool(iv and iv["type"] == "comm_severed" and (edge.u, edge.v) == tuple(iv["edge"])
                    and iv["t_start"] <= self.t <= iv["t_end"])

    def _step_serial(self, orders, demand):
        """Bit-compatible with BeerGameEnv.step for serial L=tau=2 chains."""
        n = self.graph.n_nodes
        arrivals = np.zeros(n, dtype=np.float32)
        for i in range(n - 1):
            arrivals[i] = self.ship[i][0]
            self.ship[i] = np.roll(self.ship[i], -1)
            self.ship[i][-1] = 0
        factory = self.graph.nodes[-1]
        arrivals[-1] = self.source_ship[factory.id][0]
        self.source_ship[factory.id] = np.roll(self.source_ship[factory.id], -1)
        self.source_ship[factory.id][-1] = 0
        self.inventory += arrivals

        incoming = np.zeros(n, dtype=np.float32)
        incoming[0] = demand
        for i in range(1, n):
            incoming[i] = self.order_pipe[i - 1][0]
        for i in range(n - 1):
            self.order_pipe[i] = np.roll(self.order_pipe[i], -1)
            self.order_pipe[i][-1] = 0
        self._factory_orders = np.roll(self._factory_orders, -1)
        self._factory_orders[-1] = 0
        self.backlog += incoming

        shipments = np.minimum(self.inventory, self.backlog)
        self.inventory -= shipments
        self.backlog -= shipments
        for i in range(n - 1):
            self.ship[i][-1] = shipments[i + 1]
        self.source_ship[factory.id][-1] = min(orders[-1], factory.capacity)
        for i in range(n - 1):
            self.order_pipe[i][-1] = orders[i]
        self._factory_orders[-1] = orders[-1]

        for i in range(n):
            self.order_history[i].append(orders[i])
            self.inventory_history[i].append(self.inventory[i])
        period_cost = self._period_cost(self.inventory, self.backlog)
        self.total_cost += period_cost
        self.t += 1
        self.current_demand = self._demand()
        terminated = self.t >= self.graph.T
        info = {
            "inventory": self.inventory.copy(),
            "backlog": self.backlog.copy(),
            "orders": orders.copy(),
            "shipments": shipments.copy(),
            "arrivals": arrivals.copy(),
            "period_cost": period_cost,
            "total_cost": self.total_cost,
            "customer_demand": demand,
        }
        return self._get_obs(), -period_cost, terminated, False, info


def generic_state_dim(g: SupplyGraph) -> int:
    dim = g.n_nodes + len(g.retail_nodes) + len(g.edges)
    dim += sum(e.L + e.tau for e in g.edges)
    dim += 2 * len(g.source_nodes) + 1 + len(g.retail_nodes)
    return dim


def parse_generic_obs(g: SupplyGraph, obs) -> dict:
    """Slice a generic-layout observation (numpy or torch, batched or not)."""
    batched = obs.ndim == 2
    take = (lambda a, b: obs[:, a:b]) if batched else (lambda a, b: obs[a:b])
    k = 0
    out = {}
    out["inventory"] = take(k, k + g.n_nodes); k += g.n_nodes
    nb = len(g.retail_nodes) + len(g.edges)
    out["backlog"] = take(k, k + nb); k += nb
    out["ship"] = []
    for e in g.edges:
        out["ship"].append(take(k, k + e.L)); k += e.L
    out["order"] = []
    for e in g.edges:
        out["order"].append(take(k, k + e.tau)); k += e.tau
    out["source"] = []
    for _ in g.source_nodes:
        out["source"].append(take(k, k + 2)); k += 2
    out["t"] = obs[:, k] if batched else obs[k]; k += 1
    out["demand"] = take(k, k + len(g.retail_nodes)); k += len(g.retail_nodes)
    assert k == generic_state_dim(g)
    return out


def classic_to_generic_obs(g: SupplyGraph, obs: np.ndarray) -> np.ndarray:
    """Map the 6-slot serial layout to the generic layout (T1/T2 only)."""
    assert g.is_serial_uniform()
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n = g.n_nodes
    inv = obs[0::6][:n]
    back = obs[1::6][:n]  # slot 0 retail, slot 1+i = edge i backlog at node i+1
    ship = [obs[6 * i + 2: 6 * i + 4] for i in range(n - 1)]
    order = [obs[6 * i + 4: 6 * i + 6] for i in range(n - 1)]
    source = obs[6 * (n - 1) + 2: 6 * (n - 1) + 4]
    return np.concatenate([inv, back, *ship, *order, source, obs[-2:-1], obs[-1:]]).astype(np.float32)


class GraphBeerGameEnvSequential(GraphBeerGameEnv):
    """Sequential per-node decisions; reward and discount only at period end."""

    def __init__(self, graph: SupplyGraph, **kwargs):
        super().__init__(graph, **kwargs)
        self.action_space = spaces.Discrete(graph.order_levels)
        base = self.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([base.low, [0.0]]),
            high=np.concatenate([base.high, [float(graph.n_nodes - 1)]]),
            dtype=np.float32,
        )
        self.current_node = 0
        self.pending = np.zeros(graph.n_nodes, dtype=np.int64)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_node = 0
        self.pending = np.zeros(self.graph.n_nodes, dtype=np.int64)
        return np.concatenate([obs, [0.0]]).astype(np.float32), info

    def step(self, action: int):
        self.pending[self.current_node] = int(action)
        self.current_node += 1
        if self.current_node < self.graph.n_nodes:
            obs = np.concatenate([self._get_obs(), [float(self.current_node)]]).astype(np.float32)
            return obs, 0.0, False, False, {"substep": True}
        self.current_node = 0
        obs, reward, term, trunc, info = super().step(self.pending)
        self.pending = np.zeros(self.graph.n_nodes, dtype=np.int64)
        info["substep"] = False
        return np.concatenate([obs, [0.0]]).astype(np.float32), reward, term, trunc, info


def t1_matches_beer_game(seed=0, steps=8) -> bool:
    from src.env.beer_game_env import BeerGameEnv

    classic = BeerGameEnv(seed=seed)
    graph = serial_topology(4, T=classic.T, demand_low=classic.demand_low, demand_high=classic.demand_high, demand_step_time=classic.demand_step_time)
    gen = GraphBeerGameEnv(graph, seed=seed)
    o1, _ = classic.reset(seed=seed)
    o2, _ = gen.reset(seed=seed)
    if not np.allclose(o1, o2):
        return False
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        a = rng.integers(0, classic.order_levels, size=classic.n_echelons)
        s1, r1, t1, _, _ = classic.step(a)
        s2, r2, t2, _, _ = gen.step(a)
        if not (np.allclose(s1, s2) and np.isclose(r1, r2) and t1 == t2):
            return False
    return True
