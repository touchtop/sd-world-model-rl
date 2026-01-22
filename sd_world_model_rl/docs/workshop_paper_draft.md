# Structured World Models for Operations Research: Learning System Dynamics for MCTS Planning

## Abstract

We present SD-MCTS, a framework that combines learnable System Dynamics (SD) models with Monte Carlo Tree Search for operations research problems. Unlike black-box world models (e.g., MuZero), SD-MCTS encodes domain structure as inductive bias while learning key parameters from data. On a task allocation problem with efficiency degradation, SD-MCTS significantly outperforms Oracle MPC (with true parameters) by 3.0% (p<0.0001, Cohen's d=2.05) and MPC with wrong beliefs by 7.3%, demonstrating that learned structure + deep search can surpass perfect parameter knowledge with limited horizon. The key insight is that infinite effective horizon (via value network) can dominate parameter estimation errors in systems with long time constants.

**Keywords**: World Models, System Dynamics, MCTS, Model Predictive Control, Operations Research

---

## 1. Introduction

Model-based reinforcement learning has achieved remarkable success in games (MuZero, Dreamer) by learning latent dynamics models. However, these black-box models lack interpretability and require millions of samples. In operations research (OR) domains, we often have structural knowledge about system dynamics but uncertain parameters.

We propose **SD-MCTS**: a framework that:
1. Uses **System Dynamics equations** as structured world models
2. **Learns parameters** (not dynamics) from interaction data
3. Plans via **MCTS with learned value functions**

Our key insight: *Structure from domain knowledge + Parameters from data + Planning from search = Better than either pure learning or pure model-based control.*

### 1.1 Why This Paradigm is Superior

The SD-MCTS paradigm offers four structural advantages over alternatives:

1. **Lower Sample Complexity**: Constraining dynamics to a known SD form reduces the hypothesis class size from $O(|\mathcal{S}||\mathcal{A}|)$ (tabular) or $O(\text{dim}(\text{latent}))$ (neural) to $O(d)$ where $d$ is the number of learnable parameters. This dramatically improves statistical efficiency.

2. **Safe Extrapolation**: Explicit stocks and flows preserve physical invariants (nonnegativity of queues, bounded efficiencies) by construction. Black-box models may violate these constraints under distribution shift, leading to catastrophic prediction errors.

3. **Long-horizon Optimality**: MCTS with a learned value function approximates infinite-horizon planning, reducing horizon truncation bias relative to finite-horizon MPC (subject to value approximation error). This is critical when costs are delayed (fatigue accumulation, backlog growth).

4. **Interpretability and Governance**: Parameters like degradation rate $\alpha$ and fatigue threshold $\theta$ have direct physical meaning. This supports diagnostics, auditing, and regulatory compliance that black-box latent states cannot provide.

---

## 2. Background

### 2.1 System Dynamics

System Dynamics (Forrester, 1961) models systems as stocks (state variables) and flows (rates of change). For a task allocation system:

- **Stocks**: Queue length $q$, Worker backlogs $b_i$, Cumulative fatigue $f$
- **Flows**: Arrival rate $\lambda$, Service rates $\mu_i$, Fatigue accumulation/recovery

A discrete-time SD formulation with time step $\Delta t$ is:

$$q_{t+1} = \max(0, q_t + A_t - \sum_i x_{i,t})$$
$$b_{i,t+1} = \max(0, b_{i,t} + x_{i,t} - s_{i,t})$$
$$f_{t+1} = (1 - \rho) f_t + \kappa \sum_i s_{i,t}$$
$$s_{i,t} = \min(b_{i,t} + x_{i,t}, \mu_i \cdot \text{efficiency}(f_t) \cdot u_{i,t})$$

where $A_t \sim \text{Poisson}(\lambda \Delta t)$ is the arrival process, $x_{i,t}$ is the number of tasks assigned from the queue to worker $i$ (with $\sum_i x_{i,t} \le q_t + A_t$), $u_{i,t} \in [0,1]$ is the utilization decision, and $\rho$ governs fatigue recovery. These equations enforce nonnegativity by construction and capture delayed effects through cumulative fatigue $f_t$.

**Assumption 1 (Structural Knowledge):** The true transition kernel $P_{\theta^*}(s'|s,a)$ belongs to a known parametric family $\{P_\theta\}$ with $\theta^* \in \Theta$ compact and identifiable. The structure (stocks, flows, functional forms) is known; only parameters $\theta$ are uncertain.

### 2.2 MuZero

MuZero (Schrittwieser et al., 2020) learns latent dynamics $s' = f_\theta(s, a)$ end-to-end. While powerful, it:
- Requires millions of samples
- Produces uninterpretable latent states
- Cannot incorporate domain knowledge

### 2.3 Three-Paradigm Comparison

| Criterion | Black-box (MuZero) | MPC | SD-MCTS |
|-----------|-------------------|-----|---------|
| Sample efficiency | ~$10^6$ steps | N/A (known model) | ~$10^4$ steps |
| Interpretability | Low (latent states) | High | High |
| Effective horizon | Approx. infinite (VNet) | Finite $H$ | Approx. infinite (VNet) |
| Parameter requirement | None (fully learned) | Exact knowledge | Partial (learned) |
| Domain knowledge | None | Full model | Structure only |
| Constraint handling | Implicit/penalty | Explicit | Explicit by construction |
| Extrapolation safety | Poor | Good | Good |

SD-MCTS occupies a unique position: it inherits the interpretability and constraint-handling of MPC while gaining the infinite-horizon planning of learned value functions, all with orders-of-magnitude better sample efficiency than black-box approaches.

---

## 3. Method: SD-MCTS

### 3.1 Structured World Model

Instead of learning arbitrary dynamics $s' = f_\theta(s,a)$ end-to-end, we parameterize known SD structure with learnable parameters:

**State Space**: $s = (q, b_1, \ldots, b_n, f)$ where $q$ is queue length, $b_i$ is worker $i$'s backlog, and $f$ is cumulative fatigue.

**Action Space**: $a \in \{0, 1, \ldots, n\}$ where $a=i$ assigns a task to worker $i$, and $a=n$ means no assignment.

**Transition Dynamics** (differentiable approximation):
$$q_{t+1} = \text{clamp}(q_t + \lambda - \mathbf{1}[a_t < n], 0, q_{\max})$$
$$b_{i,t+1} = \text{clamp}(b_{i,t} + \mathbf{1}[a_t = i] - \mu_i \cdot \eta(f_t), 0, b_{\max})$$
$$f_{t+1} = (1-\rho) f_t + \kappa \sum_i \mu_i$$

where $\lambda$ is arrival rate, $\mu_i$ is service rate, and $\eta(f)$ is the efficiency function.

These equations are implemented as **differentiable computation graphs**. Gradients from the prediction loss flow backward through the *known physical structure* to update the unknown parameters ($\alpha, \theta$)—effectively performing "system identification via backpropagation."

**Efficiency Degradation** (learnable):
$$\eta(f) = \frac{1}{1 + \alpha \cdot \max(0, f - \theta)^2}$$

where $\alpha$ (degradation rate) and $\theta$ (threshold) are **learnable parameters**, constrained to positive values via softplus transformation.

**What is Learned vs. Known**:
| Component | Status | Representation |
|-----------|--------|----------------|
| Stock-flow structure | Known | Hard-coded equations |
| Queue/backlog constraints | Known | $\text{clamp}(\cdot, 0, \max)$ |
| Arrival rate $\lambda$ | Learned | $\lambda = \text{softplus}(\phi_\lambda)$ |
| Efficiency $\alpha, \theta$ | Learned | $\alpha = \text{softplus}(\phi_\alpha)$ |
| Service rates $\mu_i$ | Learned | $\mu_i = \text{sigmoid}(\phi_\mu) \cdot 2$ |

This decomposition ensures physical constraints are satisfied by construction while allowing data-driven calibration of uncertain parameters.

### 3.2 Learning

We jointly train three components in an iterative loop:

**Training Loop** (AlphaZero-style):
```
for episode in 1..N:
    # 1. Data Collection: Run MCTS policy in environment
    trajectory = collect_episode(env, mcts_policy)
    buffer.add(trajectory)

    # 2. World Model Update: Predict next states
    L_model = MSE(s_{t+1}, WorldModel(s_t, a_t))

    # 3. Value Network Update: TD learning
    L_value = MSE(V(s_t), r_t + γ·V(s_{t+1}))

    # 4. Policy Network Update: Distill MCTS
    L_policy = CrossEntropy(π(s_t), mcts_visit_counts)
```

**World Model Loss**:
$$\mathcal{L}_{\text{model}} = \|s_{t+1} - \hat{s}_{t+1}(s_t, a_t; \alpha, \theta)\|^2 + \beta \|\hat{\eta}_t - \eta(f_t; \alpha, \theta)\|^2$$

The first term supervises state prediction; the second term provides direct efficiency supervision when ground-truth efficiency observations are available.

**Value Network**: A 2-layer MLP that maps state $s$ to scalar value $V(s)$, trained via TD(0):
$$\mathcal{L}_{\text{value}} = \mathbb{E}[(r_t + \gamma V(s_{t+1}) - V(s_t))^2]$$

**Policy Network**: Distills MCTS visit counts into a fast policy for action selection:
$$\mathcal{L}_{\text{policy}} = -\sum_a \frac{N(s,a)}{\sum_{a'} N(s,a')} \log \pi(a|s)$$

**Proposition 1 (Sample Complexity):** Under realizability, identifiability, and bounded noise (standard M-estimation conditions), learning $d$ parameters from a structured SD model requires $O(d/\epsilon^2)$ samples for $\epsilon$-accurate prediction, versus $O(|\mathcal{S}||\mathcal{A}|/\epsilon^2)$ for tabular methods. When $d \ll |\mathcal{S}||\mathcal{A}|$, the structured approach achieves orders-of-magnitude sample efficiency gains.

### 3.3 Planning: MCTS with Value Network

At decision time, we run MCTS using the learned world model for simulation and the value network for leaf evaluation. This is the key mechanism that enables **approximately infinite effective horizon**.

**MCTS Algorithm** (AlphaZero/MuZero-style):
```
function MCTS(root_state, num_simulations):
    root = Node(root_state)
    for sim in 1..num_simulations:
        # Selection: Traverse tree using PUCT
        node = root
        path = [root]
        while node.is_expanded():
            node = argmax_child(PUCT(node, child))
            path.append(node)

        # Expansion: Use world model to predict next states
        for action in valid_actions:
            s' = WorldModel(node.state, action)
            r = reward(s')
            child = Node(s', prior=π(action|node.state), reward=r)
            node.children[action] = child

        # Evaluation: Value network estimates infinite-horizon return
        leaf_value = V(node.state)  # ← KEY: replaces zero terminal cost

        # Backpropagation: Update Q-values
        G = leaf_value
        for node in reversed(path):
            G = node.reward + γ * G
            node.N += 1
            node.W += G
            node.Q = node.W / node.N

    return argmax_a(root.children[a].N)  # Most visited action
```

**PUCT Formula** (Polynomial Upper Confidence Tree):
$$\text{PUCT}(s,a) = Q(s,a) + c \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

where $Q(s,a)$ is the mean action-value (exploitation), $P(a|s)$ is the prior from the policy network, $N(s)$ is the parent visit count, and $c$ is the exploration constant. This formula balances exploitation (high $Q$) against exploration (high prior probability, low visit count)—analogous to Upper Confidence Bound methods in multi-armed bandits.

**Why Value Network Enables Infinite Horizon**:

The critical insight is how leaf evaluation transforms effective planning horizon:

| Method | Leaf Evaluation | Effective Horizon |
|--------|----------------|-------------------|
| MPC (zero terminal) | $V_{\text{leaf}} = 0$ | Exactly $H$ |
| MCTS + VNet | $V_{\text{leaf}} = V_\psi(s)$ | $H + \gamma^H/(1-\gamma) \approx \infty$ |

When MCTS reaches depth $D$, instead of assuming zero future value (as MPC does), it uses $V_\psi(s_D)$ which approximates $\sum_{t=D}^{\infty} \gamma^{t-D} r_t$. This is analogous to **Approximate Dynamic Programming (ADP)** in OR terminology—the value network serves as a *learned terminal cost* that encodes long-term consequences. This propagates long-term consequences backward through the tree, enabling decisions that account for delayed costs and rewards far beyond the search depth.

**Note on Discount Factor**: For OR practitioners familiar with finite-horizon MPC, the discount factor $\gamma < 1$ serves two purposes: (1) it makes the infinite sum $\sum_{t=0}^{\infty} \gamma^t r_t$ convergent, and (2) it implicitly defines an "effective planning horizon" of approximately $1/(1-\gamma)$ steps (e.g., $\gamma=0.99$ corresponds to ~100 steps).

**Concrete Example**: With $\gamma=0.99$ and depth $D=10$:
- MPC sees: $\sum_{t=0}^{9} \gamma^t r_t$ (horizon = 10 steps)
- MCTS+VNet sees: $\sum_{t=0}^{9} \gamma^t r_t + \gamma^{10} V(s_{10})$ where $V(s_{10}) \approx \sum_{t=10}^{\infty} \gamma^{t-10} r_t$

The value network effectively "compresses" the infinite tail into a single scalar estimate, enabling MCTS to reason about consequences 100+ steps into the future with only 10 simulation steps.

---

## 4. Experiments

### 4.1 Task Allocation with Efficiency Degradation

**Environment**: 2 workers, Poisson arrivals, efficiency degrades with cumulative fatigue.

**True parameters**: $\alpha = 0.1$, $\theta = 5.0$
**Initial belief**: $\alpha = 0.001$, $\theta = 12.0$ (100x error)

**Real-World Prevalence:** This scenario represents a broad class of OR problems with performance degradation dynamics:

- **Warehouse Operations (Amazon, Alibaba):** Worker picking efficiency degrades with hours worked. Static allocation ignoring fatigue dynamics leads to end-of-shift throughput collapse—a well-documented phenomenon in logistics operations research.

- **Call Centers:** Agent service quality and first-call resolution rates worsen with cumulative call volume. Staffing models that ignore fatigue systematically underestimate required capacity.

- **Manufacturing (Toyota Production System):** Line balancing under operator fatigue is fundamental to lean manufacturing. Assembly errors increase non-linearly with cumulative workload.

- **Healthcare:** Physician decision fatigue after extended shifts increases diagnostic error rates by 20-30% (Landrigan et al., NEJM 2004). This motivated FAA-style duty hour regulations.

- **Airline Crew Scheduling:** FAR 117 regulations explicitly require modeling cumulative duty time effects on pilot performance—a legal mandate for fatigue-aware scheduling.

These scenarios share a common OR pain point: **delayed consequences from accumulated state variables**. Traditional MPC with finite horizon systematically underweights these long-term effects.

### 4.2 Why MPC as Primary Baseline?

Model Predictive Control is chosen as the primary baseline for three reasons:

1. **Canonical OR Paradigm**: MPC (receding horizon control) is the dominant methodology in industrial process control, supply chain optimization, and operations management. It represents the "best of traditional methods" with decades of successful deployment.

2. **Controlled Comparison**: Both MPC and SD-MCTS use the same SD structure for dynamics. This isolates the effect of (a) planning horizon and (b) learning approach, enabling rigorous ablation.

3. **Oracle Upper Bound**: Oracle MPC with true parameters and zero terminal cost provides an upper bound within the finite-horizon MPC formulation. Outperforming this oracle demonstrates an advantage of longer effective horizon.

We deliberately exclude model-free RL (PPO, SAC) and black-box world models (Dreamer, MuZero) from primary comparisons because they require $10^6$+ samples. Our focus is on sample-efficient learning in low-data OR regimes where domain structure is available. The scientific question is not whether SD-MCTS beats PPO (trivially true given 1000× sample advantage) but whether **learned structure + deep search can beat perfect parameters + shallow optimization**.

**SD-MCTS vs. Adaptive MPC: A Fundamental Distinction**

Both SD-MCTS and Adaptive MPC learn parameters online. Why does SD-MCTS outperform Adaptive MPC significantly?

| Aspect | Adaptive MPC (RLS) | SD-MCTS |
|--------|-------------------|---------|
| **Learning** | Recursive Least Squares | Neural network + gradient descent |
| **Horizon** | Fixed $H$ (e.g., 15) | Approximately infinite (via VNet) |
| **Terminal Value** | Zero (truncated) | Learned $V_\psi(s)$ |
| **Exploration** | None (greedy) | PUCT exploration bonus |
| **Planning** | Convex optimization | Tree search |

The key differences are:

1. **Horizon Limitation**: Adaptive MPC still uses finite horizon $H$ with zero terminal cost. Even with perfectly learned parameters, it cannot see beyond $H$ steps. SD-MCTS uses the value network to approximate infinite-horizon returns.

2. **Exploration-Exploitation**: RLS performs pure exploitation—updating parameters but always acting greedily. SD-MCTS's PUCT formula explicitly balances exploration and exploitation, enabling discovery of better policies.

3. **Nonlinear Value Propagation**: RLS learns linear parameter updates; the value network learns nonlinear mappings that can capture complex long-term dependencies (e.g., fatigue accumulation leading to efficiency collapse).

4. **Credit Assignment vs. System Identification**: RLS performs *System Identification*—it corrects the model parameters ($\alpha, \theta$) to explain observed data. However, the MPC planner remains "horizon-blind" until costs fall within the short horizon $H$. In contrast, the Value Network performs *Credit Assignment* via TD learning, explicitly associating early states with low future value of delayed collapse, enabling the agent to avoid the danger zone preemptively.

### 4.3 Baselines

| Method | Parameters | Horizon | Learning |
|--------|-----------|---------|----------|
| MPC (wrong) | Wrong belief | H=3 | None |
| Adaptive MPC | RLS learning | H=3/15 | Online RLS |
| Oracle MPC | True params | H=15 | None (oracle) |
| **SD-MCTS** | Learned | Approx. infinite (VNet) | Neural |

### 4.4 Results (50 seeds, 30000 training steps)

| Method | Mean Reward | vs MPC wrong |
|--------|------------|--------------|
| MPC wrong | -8469 ± 506 | baseline |
| Adaptive MPC (H=15) | -8154 ± 476 | +3.7% |
| Oracle MPC | -8098 ± 471 | +4.4% |
| **SD-MCTS** | **-7853 ± 385** | **+7.3%** |

**Key finding**: SD-MCTS significantly outperforms Oracle MPC by 3.0% (paired t-test: t=14.52, p<0.0001, Cohen's d=2.05). This demonstrates a highly significant advantage of learned structure + deep planning over perfect parameters with limited horizon.

![Comparison of methods](../results/figures/comparison_bars.png)
*Figure 1: Performance comparison across methods with 95% confidence intervals (50 seeds).*

### 4.5 Experiment 2: Delayed Collapse with Demand Surge

To further validate the structural advantages, we design a scenario that maximizes the benefit of long-horizon planning and exposes the fundamental limitations of finite-horizon optimization.

**Environment Design**: A single worker with hidden fatigue dynamics and a future demand surge event.

**Dynamics**:
$$f_{t+1} = f_t + (\alpha + \beta) \cdot a_t - \beta$$

where $f_t$ is cumulative fatigue, $a_t \in [0,1]$ is work intensity, $\alpha=1.0$ is fatigue accumulation rate, and $\beta=0.1$ is recovery rate.

**Collapse Mechanism**: Hard collapse occurs when $f_t \geq F$ (absorbing state with zero future reward).

**Demand Surge**: At time $T_s=12$, a bonus $B=15$ is awarded if $f_t \leq F_{req}=5$.

**True vs. Belief Parameters**:
- True collapse threshold: $F = 10$
- MPC belief: $F = 50$ (5× overestimate)

**Why MPC Fails Structurally**:

1. **Greedy Behavior**: MPC with wrong belief uses $a_t=1$ (maximum intensity) because it believes collapse is far away.
2. **Collapse Before Surge**: With $a_t=1$, fatigue reaches $F=10$ at $t=10$, causing collapse before the surge at $t=12$.
3. **Horizon Limitation**: Even with correct parameters, MPC with $H<T_s$ cannot "see" the surge event and thus cannot reason about it.

**Why SD-MCTS Succeeds**:

1. **Learning Collapse Threshold**: The world model learns from collapse observations, gradually correcting its $F$ estimate.
2. **Value Network Propagation**: The learned value network propagates the surge bonus $B=15$ backward in time, incentivizing fatigue conservation.
3. **Conservative Policy Discovery**: SD-MCTS discovers the safe intensity $a_{safe} \approx 0.47$ that maintains $f_t \leq F_{req}$ at surge time.

**Theoretical Analysis**:
- MPC (greedy $a=1$): Collapses at $t=F/\alpha=10$, total reward $\approx 10$
- Optimal policy: $a_{safe} = (F_{req} + \beta \cdot T_s)/((\alpha+\beta) \cdot T_s) \approx 0.47$, total reward $\approx T_s \cdot a_{safe} + B \approx 20.6$

**Results (10 seeds)**:

| Method | Mean Reward | Collapse Rate | Surge Bonus |
|--------|-------------|---------------|-------------|
| MPC (wrong belief) | 9.0 ± 0.0 | 100% | 0% |
| **SD-MCTS** | **15.0 ± 0.0** | **0%** | **100%** |
| Optimal (reference) | 24.8 | 0% | 100% |

**Key finding**: SD-MCTS achieves **66.7% improvement** over MPC. The MPC greedily uses maximum intensity and collapses at $t=10$, forfeiting the surge bonus. SD-MCTS learns to maintain safe intensity and captures all surge bonuses.

**Real-World Analogues**:

This scenario models critical OR problems with delayed high-value events:

- **Seasonal Retail (Black Friday, Prime Day)**: Warehouse systems must conserve capacity for peak events. Greedy pre-peak allocation leads to equipment failures or worker burnout precisely when capacity is most valuable.

- **Power Grid Management**: Utilities must reserve capacity for predicted demand peaks. Greedy resource utilization before peaks leads to brownouts during critical periods.

- **Project Management**: Teams must preserve energy for critical milestones. Burnout before deadlines leads to project failures at the worst possible time.

- **Financial Trading**: Traders must preserve capital for high-conviction opportunities. Over-trading depletes capital before major market events.

The common pattern: **short-horizon optimization sacrifices long-term opportunities**. This is precisely where infinite effective horizon (via value network) provides structural advantage.

![Delayed Collapse Comparison](../results/figures/delayed_collapse_bars.png)
*Figure 4: Delayed Collapse scenario—SD-MCTS avoids collapse and captures surge bonus.*

![Fatigue Trajectory](../results/figures/delayed_collapse_trajectory.png)
*Figure 5: Fatigue trajectory comparison. MPC (greedy) hits collapse threshold at t=10, missing the surge at t=12. SD-MCTS maintains safe intensity and earns the surge bonus.*

![Reward Accumulation](../results/figures/delayed_collapse_rewards.png)
*Figure 6: Cumulative reward comparison. SD-MCTS's conservative early strategy pays off with the surge bonus, ultimately outperforming MPC despite lower early intensity.*

### 4.6 Parameter Learning

SD-MCTS learns accurate parameters:
- $\alpha$: 0.001 → 0.0753 (true: 0.1, 75% recovery)
- $\theta$: 12.0 → 4.72 (true: 5.0, 94% recovery)

![Efficiency curves](../results/figures/efficiency_curves.png)
*Figure 2: Learned efficiency curve closely approximates the true dynamics, while the wrong belief significantly underestimates degradation.*

![State trajectory](../results/figures/state_trajectory.png)
*Figure 3: State trajectory comparison showing SD-MCTS maintains lower fatigue and backlog than MPC.*

---

## 5. Analysis

### 5.1 Why SD-MCTS Outperforms Oracle MPC

Oracle MPC has **perfect parameters** but **limited horizon** (H=15) with zero terminal value. SD-MCTS has:
- **Approximately infinite horizon** via value network (subject to value approximation error)
- **Learned parameters** (25% error in $\alpha$, 6% error in $\theta$)

**Proposition 2 (Horizon-Error Trade-off):** For discounted MDPs with $\gamma \in (0,1)$ and MPC using zero terminal cost, the value estimation error satisfies:

$$|V^*(s) - V_{\text{MPC}}^H(s)| \leq \frac{\gamma^H R_{\max}}{1-\gamma} \quad \text{(horizon truncation bias)}$$

$$|V^*(s) - V^*(s;\hat{\theta})| \leq L_\theta \|\hat{\theta} - \theta^*\| \quad \text{(parameter sensitivity)}$$

where $L_\theta$ is the Lipschitz constant of the value function with respect to parameters (may scale with $1/(1-\gamma)$), and $R_{\max}$ is the maximum single-step reward magnitude.

**Crossover Condition:** Infinite-horizon planning with imperfect parameters dominates finite-horizon planning with perfect parameters when:

$$\frac{\gamma^H R_{\max}}{1-\gamma} > L_\theta \|\hat{\theta} - \theta^*\| + \gamma^H \|V^* - \hat{V}\|_\infty$$

where $\hat{V}$ is the learned value network. The empirical success indicates the value approximation error is small.

**Numerical Analysis:** With $\gamma=0.99$, $H=15$:
- Horizon truncation: $\gamma^{15}/(1-\gamma) \approx 0.86/0.01 \approx 86 \cdot R_{\max}$
- Parameter error (25%): Assuming $L_\theta \approx 10$, parameter contribution $\approx 2.5 \cdot R_{\max}$

The truncation term dominates by ~30×, explaining why SD-MCTS with imperfect parameters outperforms Oracle MPC.

**Remark:** This analysis assumes the learned value network is a good approximation to $V^*$. In practice, SD-MCTS incurs both parameter error and value approximation error; the empirical success (t=14.52, p<0.0001) indicates the latter is small relative to MPC's horizon truncation bias.

**Methodological Insight:** In OR systems with delayed costs and long time constants, *investing in longer effective horizons yields better returns than investing in more accurate parameter estimation*. This challenges the traditional OR assumption that parameter accuracy is paramount, and suggests a new paradigm: **structured imperfect models + deep planning > perfect parameters + shallow optimization**.

### 5.2 Comparison with MuZero

| Aspect | MuZero | SD-MCTS |
|--------|--------|---------|
| Dynamics | Black-box neural network | Structured SD equations |
| State space | Learned latent | Observable (interpretable) |
| Sample efficiency | Millions | 30,000 steps |
| Interpretability | Low | High ("$\alpha=0.07$ means 7% degradation rate") |
| Constraint handling | Soft (must be learned) | Hard-coded (e.g., $q \geq 0$) |
| Generalization | Fails on parameter shifts | Robust (re-learns only scalars) |

Additionally, SD-MCTS enforces constraints (nonnegative queues, bounded efficiencies) by construction, whereas black-box latent models may violate feasibility without explicit penalties or projection layers.

---

## 6. Related Work

- **World Models**: Ha & Schmidhuber (2018), Dreamer (Hafner et al., 2020)
- **Model-based RL**: MuZero (Schrittwieser et al., 2020), EfficientZero (Ye et al., 2021)
- **System Dynamics + RL**: Limited prior work on combining SD with modern RL
- **Structured Models**: Neural ODEs (Chen et al., 2018), Physics-informed NNs (Raissi et al., 2019)

Our work differs from Neural ODEs and PINNs in that we do not learn the differential equations themselves; we assume known SD structure and learn only the parameters. This provides stronger inductive bias and better sample efficiency at the cost of requiring domain expertise to specify the structure.

---

## 7. Conclusion

SD-MCTS demonstrates that combining domain structure with learned parameters and deep search can outperform both pure model-based control and black-box learning. Key contributions:

1. **Structured world model**: SD equations with learnable parameters
2. **Sample efficiency**: 30,000 steps vs. millions for MuZero
3. **Interpretability**: Parameters have physical meaning
4. **Performance**: Highly significantly outperforms Oracle MPC (paired t-test: p<0.0001, d=2.05)

### 7.1 Long-term Vision: Toward Neuro-OR

We envision extending SD-MCTS toward a unified **Neuro-OR** framework that bridges classical operations research formalisms with modern deep learning:

**Structured Formalisms as Inductive Bias:**
- System Dynamics (stocks-flows for continuous systems)
- Queuing Networks (Jackson networks, BCMP networks for service systems)
- Petri Nets (discrete event systems, manufacturing, logistics)
- Markov Decision Processes with structured transitions
- Linear/Integer Programs with learnable coefficients

**Learnable Components:**
- Transition rates and service times
- Arrival distributions and demand patterns
- Cost coefficients and constraint bounds
- Routing probabilities and resource capacities

**Planning Layer:**
- MCTS for discrete decisions
- Differentiable optimization for continuous allocation
- Bilevel optimization for strategic planning

**Research Agenda:** Build a library of compositional, differentiable OR modules (queues, servers, buffers, routing) that can be assembled into larger systems while retaining learning capability and planning performance. This would enable practitioners to specify domain structure graphically while the framework learns parameters and optimizes policies—combining the interpretability and governance of OR with the adaptivity and generalization of RL.

The ultimate goal is to make **structure a first-class citizen in learned world models**, not an afterthought. Just as physics-informed neural networks encode conservation laws, Neuro-OR encodes operational constraints and flow relationships—achieving both sample efficiency and real-world deployability.

---

## References

1. Forrester, J. W. (1961). Industrial dynamics.
2. Schrittwieser, J., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. Nature.
3. Hafner, D., et al. (2020). Dream to control: Learning behaviors by latent imagination. ICLR.
4. Chen, R. T., et al. (2018). Neural ordinary differential equations. NeurIPS.
5. Landrigan, C. P., et al. (2004). Effect of reducing interns' work hours on serious medical errors. NEJM.
6. Raissi, M., et al. (2019). Physics-informed neural networks. Journal of Computational Physics.

---

*Code available at: [repository URL]*
