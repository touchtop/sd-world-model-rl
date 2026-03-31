import sys
import os
import torch
import numpy as np
from dm_control import suite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_dmcontrol_cartpole import (
    collect_trajectories, train_sd_model, SwingupValueNet, train_value_net, 
    generate_imagination_data, evaluate_agent
)
from src.world_model.dmcontrol_cartpole_sd_model import DMControlCartPoleSDModel, DMControlCartPoleWrapper


def eval_single():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})
    
    # 1. SI
    model = DMControlCartPoleSDModel()
    print("Collecting data...")
    data = collect_trajectories(env, 5000, policy='random')
    print("Training SI...")
    train_sd_model(model, data, epochs=100, lr=1e-2)
    
    # 2. Imagination
    print("Generating imagination data...")
    wrapper = DMControlCartPoleWrapper(model)
    train_s, train_r = generate_imagination_data(
        wrapper, action_repeat=2, gamma=0.99, n_trajectories=20000, horizon=60, real_states=data['states']
    )
    
    # 3. VNet Training
    print("Training VNet...")
    vnet = SwingupValueNet()
    train_value_net(vnet, train_s, train_r, epochs=500, lr=1e-3, min_loss=0.05)
    
    discrete_actions = torch.linspace(-1.0, 1.0, 11)
    
    # 4. Evaluation
    print("\n--- Evaluating without LQR Prior (alpha=0.0) ---")
    ret_0 = evaluate_agent(
        env, model, discrete_actions, n_episodes=5, num_simulations=50,
        value_net=vnet, action_repeat=2, verbose=True, cart_drift_penalty=0.0
    )
    print(f"Mean alpha=0.0 Return: {np.mean(ret_0):.1f}")
    
    print("\n--- Evaluating with LQR Prior (alpha=5.0) ---")
    ret_5 = evaluate_agent(
        env, model, discrete_actions, n_episodes=5, num_simulations=50,
        value_net=vnet, action_repeat=2, verbose=True, cart_drift_penalty=5.0
    )
    print(f"Mean alpha=5.0 Return: {np.mean(ret_5):.1f}")


if __name__ == "__main__":
    eval_single()
