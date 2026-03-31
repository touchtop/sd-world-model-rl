import sys
import os
import torch
import numpy as np
import copy
from dm_control import suite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world_model.dmcontrol_cartpole_sd_model import DMControlCartPoleSDModel, DMControlCartPoleWrapper
from scripts.run_dmcontrol_cartpole import collect_trajectories, train_sd_model, model_forward_k

def compare_controllers():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})

    model = DMControlCartPoleSDModel()
    
    print("Collecting 1000 steps for SI...")
    data = collect_trajectories(env, 1000, policy='random')
    print("Training SI...")
    train_sd_model(model, data, epochs=100, lr=1e-2)
    
    params = model.get_learned_params()
    wrapper = DMControlCartPoleWrapper(model)
    g_over_l = params['gravity'] / max(params['length'], 1e-3)
    
    real_states = data['states']
    
    def old_heuristic(x, theta, x_dot, theta_dot):
        import math
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        if cos_th > 0.85:
            # The ORIGINAL BACKWARD PD controller
            a = -(3.0 * sin_th + 1.0 * theta_dot + 0.4 * x + 0.3 * x_dot)
        else:
            # The ORIGINAL BACKWARD energy pump
            E_norm = 0.5 * theta_dot ** 2 / g_over_l + cos_th
            E_err = E_norm - 1.0
            pump_dir = 1.0 if theta_dot * cos_th >= 0 else -1.0
            a = -1.5 * E_err * pump_dir - 0.15 * x - 0.1 * x_dot

        return float(np.clip(a, -1.0, 1.0))

    def new_heuristic(x, theta, x_dot, theta_dot):
        import math
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        E_norm = 0.5 * theta_dot * theta_dot / g_over_l + cos_th
        
        near_upright = cos_th > 0.85
        catch_region = cos_th > 0.4

        if near_upright and abs(theta_dot) < 1.0:
            a = 4.0 * sin_th + 1.5 * theta_dot - 0.5 * x - 0.4 * x_dot
        elif catch_region:
            a = 3.0 * sin_th + 2.5 * theta_dot - 0.4 * x - 0.3 * x_dot
        else:
            pump_dir = 1.0 if theta_dot * cos_th >= 0 else -1.0
            a = 1.5 * (E_norm - 0.95) * pump_dir - 0.15 * x - 0.1 * x_dot
        return float(np.clip(a, -1.0, 1.0))

    print("\nStarting comparison...")
    
    # Eval both on the SAME random selection of 50 bottom/transition anchors
    np.random.seed(123)
    anchors = []
    for _ in range(50):
        # We want states that challenge the swingup
        while True:
            idx = np.random.randint(len(real_states))
            rs = real_states[idx].clone() + torch.randn_like(real_states[idx]) * 0.05
            if torch.cos(rs[1]) < 0.5: # bottom or transition
                anchors.append(rs)
                break

    for name, heuristic_fn in [("OLD BACKWARD", old_heuristic), ("NEW CORRECTED", new_heuristic)]:
        max_returns = []
        balanced_steps = 0
        
        for rs in anchors:
            traj_r = []
            curr_rs = rs.clone()
            b_count = 0
            for t in range(60):
                x, th, x_dot, th_dot = curr_rs[0].item(), curr_rs[1].item(), curr_rs[2].item(), curr_rs[3].item()
                if abs(th) < 0.25 and abs(th_dot) < 1.0:
                    b_count += 1
                
                act = heuristic_fn(x, th, x_dot, th_dot)
                new_state, step_r, _ = model_forward_k(wrapper, curr_rs, act, 2, 0.99)
                traj_r.append(step_r)
                curr_rs = new_state
                
            returns = np.zeros(60, dtype=np.float32)
            G = 0.0
            for t in reversed(range(60)):
                G = traj_r[t] + (0.99**2) * G
                returns[t] = G
                
            max_returns.append(returns.max())
            balanced_steps += b_count
            
        print(f"{name}: MeanMaxRet={np.mean(max_returns):.1f}, MaxMaxRet={np.max(max_returns):.1f}, BalSteps={balanced_steps}")

if __name__ == "__main__":
    compare_controllers()
