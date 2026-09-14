"""Design §8.7 gates G-BT1′, G-BT2′, G-BT4, G-BT7, G-BT8 plus legacy T1 checks."""
import numpy as np
import pytest
import torch

from src.env.beer_game_env import BeerGameEnv, BeerGameEnvSequential
from src.env.graph_beer_game import (
    GraphBeerGameEnv,
    GraphBeerGameEnvSequential,
    classic_to_generic_obs,
    load_supply_graph,
    serial_topology,
    t1_matches_beer_game,
)
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.world_model.graph_beer_game_sd_model import (
    GenericGraphBeerGameSDModel,
    GraphBeerGameSDModel,
    build_graph_sd_model,
    theta_dimension,
)

TOPOLOGIES = ["t1_serial4", "t2_serial8", "t3_fork", "t4_dag", "t5_scale"]


def graph(name):
    return load_supply_graph(f"experiments/beer_game_graph/topologies/{name}.yaml")


def test_bt0_sequential_instance():
    env = BeerGameEnvSequential(seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape[0] == 27


def test_bt1_t1_matches_classic():
    assert t1_matches_beer_game(seed=2, steps=10)


def test_yaml_t1_matches_classic():
    classic = BeerGameEnv(seed=5)
    gen = GraphBeerGameEnv(graph("t1_serial4"), seed=5)
    o1, _ = classic.reset(seed=5)
    o2, _ = gen.reset(seed=5)
    assert np.allclose(o1, o2)
    a = np.array([2, 2, 2, 2])
    s1, r1, t1, _, _ = classic.step(a)
    s2, r2, t2, _, _ = gen.step(a)
    assert np.allclose(s1, s2)
    assert np.isclose(r1, r2)
    assert t1 == t2


def test_bt2_sd_forward():
    classic = BeerGameSDModel(learn_demand=False)
    generic = GraphBeerGameSDModel(serial_topology(4), learn_demand=False)
    state = torch.zeros(1, 26)
    state[0, 0::6] = 12
    state[0, 2::6] = 4
    state[0, 3::6] = 4
    state[0, 4::6] = 4
    state[0, 5::6] = 4
    state[0, -1] = 4
    action = torch.tensor([[2, 2, 2, 2]])
    with torch.no_grad():
        assert torch.allclose(classic(state, action), generic(state, action), atol=1e-5)


def test_sequential_graph_period_reward():
    env = GraphBeerGameEnvSequential(serial_topology(4), seed=0)
    env.reset(seed=0)
    for a in (1, 1, 1):
        _, r, _, _, info = env.step(a)
        assert r == 0.0 and info["substep"]
    _, r, _, _, info = env.step(1)
    assert info["substep"] is False
    assert r < 0


def test_shared_theta_independent_of_nodes():
    assert theta_dimension(1, True) == theta_dimension(9, True) == 3
    assert theta_dimension(9, False) == 27
    model = build_graph_sd_model(graph("t5_scale"), shared_demand=True)
    assert model.theta_dim() == 3
    assert sum(p.numel() for p in model.parameters()) == 3
    node = build_graph_sd_model(graph("t5_scale"), shared_demand=False)
    assert node.theta_dim() == sum(p.numel() for p in node.parameters()) == 27


# G-BT7: pressure rule reproduces T1 and fixes relative pressure everywhere.
def test_bt7_pressure_rule():
    classic = BeerGameEnv()
    for n in graph("t1_serial4").nodes:
        assert (n.O_max, n.initial_inventory, n.initial_pipeline) == (classic.O_max, classic.initial_inventory, classic.initial_pipeline)
        assert n.kind != "source" or n.capacity == classic.production_capacity
    for name in TOPOLOGIES:
        g = graph(name)
        for n in g.nodes:
            assert n.demand > 0 and np.isclose(n.demand / n.O_max, 0.5)
            assert n.kind != "source" or np.isclose(n.capacity / n.demand, 2.5)
        assert np.isclose(sum(n.demand for n in g.source_nodes), len(g.retail_nodes) * g.demand_high)


# G-BT1′: generic env path equals serial path / classic env on T1 and T2 for 36 steps.
@pytest.mark.parametrize("name", ["t1_serial4", "t2_serial8"])
def test_bt1_prime_generic_path_matches_serial(name):
    g = graph(name)
    for seed in range(3):
        serial = GraphBeerGameEnv(g, seed=seed, serial_layout=True)
        generic = GraphBeerGameEnv(g, seed=seed, serial_layout=False)
        classic = BeerGameEnv(n_echelons=g.n_nodes, seed=seed) if g.n_nodes == 4 else None
        s_obs, _ = serial.reset(seed=seed)
        g_obs, _ = generic.reset(seed=seed)
        np.testing.assert_array_equal(classic_to_generic_obs(g, s_obs), g_obs)
        if classic:
            classic.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(g.T):
            a = rng.integers(0, g.order_levels, g.n_nodes)
            s_obs, r_s, d_s, _, _ = serial.step(a)
            g_obs, r_g, d_g, _, _ = generic.step(a)
            np.testing.assert_array_equal(classic_to_generic_obs(g, s_obs), g_obs)
            assert r_s == r_g and d_s == d_g
            if classic:
                c_obs, r_c, *_ = classic.step(a)
                np.testing.assert_array_equal(c_obs, s_obs)
                assert r_c == r_s


# G-BT2′: generic model forward equals BeerGameSDModel on T1 (all 81 joint actions, random states).
def test_bt2_prime_generic_model_matches_classic_on_t1():
    g = graph("t1_serial4")
    classic = BeerGameSDModel(learn_demand=False, order_levels=3)
    g3 = load_supply_graph("experiments/beer_game_graph/topologies/t1_serial4.yaml")
    g3.order_levels = 3
    generic = GenericGraphBeerGameSDModel(g3, learn_demand=False)
    rng = np.random.default_rng(0)
    actions = torch.tensor(np.array(np.meshgrid(*[range(3)] * 4, indexing="ij")).reshape(4, -1).T)
    for _ in range(5):
        state = rng.uniform(0, 20, size=26).astype(np.float32)
        state[-2] = rng.integers(0, 36)
        state[-1] = 4.0
        states = torch.tensor(np.repeat(state[None], 81, 0))
        gstates = torch.tensor(np.stack([classic_to_generic_obs(g, s) for s in states.numpy()]))
        with torch.no_grad():
            expect = torch.tensor(np.stack([classic_to_generic_obs(g, s) for s in classic(states, actions).numpy()]))
            got = generic(gstates, actions)
        assert torch.allclose(expect, got, atol=1e-5)


# Generic model equals generic env on every topology for 36 random steps.
@pytest.mark.parametrize("name", TOPOLOGIES)
def test_generic_model_matches_env(name):
    g = graph(name)
    env = GraphBeerGameEnv(g, seed=0, serial_layout=False)
    model = build_graph_sd_model(g, learn_demand=False)
    s, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(g.T):
        a = rng.integers(0, g.order_levels, g.n_nodes)
        with torch.no_grad():
            pred = model(torch.tensor(s)[None], torch.tensor(a)[None])[0].numpy()
        s, r, *_ = env.step(a)
        assert np.abs(pred - s).max() < 1e-4
        assert np.isclose(float(model.predict_reward(None, None, torch.tensor(s)[None])), r, atol=1e-4)


# G-BT4: planning reward equals environment reward on 36 random steps.
@pytest.mark.parametrize("name", TOPOLOGIES)
def test_bt4_period_reward_matches_env(name):
    g = graph(name)
    for serial in {g.is_serial_uniform(), False}:
        env = GraphBeerGameEnv(g, seed=1, serial_layout=serial)
        env.reset(seed=1)
        rng = np.random.default_rng(1)
        for _ in range(g.T):
            obs, r, *_ = env.step(rng.integers(0, g.order_levels, g.n_nodes))
            assert env.period_reward(obs) == r


# G-BT8: node-specific parameters enter the forward pass with individual gradients.
def test_bt8_node_specific_gradients():
    g = graph("t3_fork")
    model = build_graph_sd_model(g, shared_demand=False)
    env = GraphBeerGameEnv(g, seed=0, serial_layout=False)
    s, _ = env.reset(seed=0)
    a = np.full(g.n_nodes, 4)
    for _ in range(g.demand_step_time + 1):
        s0, s = s, env.step(a)[0]
    with torch.no_grad():
        model._demand_high_raw[1] += 0.5  # perturb one retail node only
        model._demand_step_time[2] += 2.0
    loss = ((model(torch.tensor(s0)[None], torch.tensor(a)[None]) - torch.tensor(s)[None]) ** 2).sum()
    loss.backward()
    assert model._demand_high_raw.grad.shape == (4,)
    assert model._demand_high_raw.grad[1] != 0 and model._demand_high_raw.grad[0] == 0
    assert model._demand_step_time.grad[2] != 0


def test_prior_belief_perturbs_each_slot():
    np.random.seed(0)
    model = build_graph_sd_model(graph("t4_dag"), shared_demand=False, prior_belief=True)
    p = model.get_learned_params()
    assert len(set(np.round(p["demand_high"], 4))) == 5 and len(set(np.round(p["demand_step_time"], 4))) == 5


# Interventions (§8.6 item 3): env and edited SD model stay step-consistent on T4.
@pytest.mark.parametrize("iv", [
    {"type": "capacity_shock", "node": "s0", "factor": 0.5, "t_start": 15},
    {"type": "comm_severed", "edge": ["w1", "r1"], "t_start": 10, "t_end": 20},
])
def test_intervention_env_model_consistent(iv):
    g = graph("t4_dag")
    env = GraphBeerGameEnv(g, seed=0, serial_layout=False)
    plain = GraphBeerGameEnv(g, seed=0, serial_layout=False)
    env.intervention = iv
    model = build_graph_sd_model(g, learn_demand=False)
    model.intervention = iv
    s, _ = env.reset(seed=0); plain.reset(seed=0)
    rng = np.random.default_rng(0)
    differed = False
    for _ in range(g.T):
        a = rng.integers(0, g.order_levels, g.n_nodes)
        with torch.no_grad():
            pred = model(torch.tensor(s)[None], torch.tensor(a)[None])[0].numpy()
        s, r, *_ = env.step(a)
        sp, *_ = plain.step(a)
        assert np.abs(pred - s).max() < 1e-4
        differed |= bool(np.abs(sp - s).max() > 1e-6)
    assert differed
