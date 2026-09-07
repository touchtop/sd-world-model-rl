# Response to Reviewer GNGG

Thank you for recognizing the motivation, data efficiency, clarity, and reproducibility of our work. Your comments identify places where the compressed presentation obscured the grey-box scope, training/inference workflow, and role of each experiment.

## Scope and required prior information

SD-MCTS targets low-data grey-box planning in which a credible mechanistic skeleton is available, while task-relevant dynamical quantities and continuation values remain data-dependent. The supplied information consists of state semantics, macroscopic topology $T$, an analytical update skeleton $f_T$, and feasibility constraints $\mathcal C$. Learnable transition degrees of freedom are restricted to a low-dimensional parameter subspace, while a separate terminal estimator $V_\psi$ is learned from task data; tree statistics are computed online. The methodological contribution is the closed-loop integration of structured transition modeling and low-dimensional parameter estimation with feasibility-preserving planning and continuation-value evaluation at the explored frontier.

Beer Game fixes the four-echelon stock-flow topology, separate two-slot pipelines for orders and shipments, and feasibility rules, while parameterizing the step-demand profile with three trainable scalars. A separate terminal value network supplies continuation estimates. Cartpole fixes the damped Newton–Euler equations and RK4 integration while estimating cart mass, pole mass, and pole half-length from 5,000 real transitions per target seed.

The controlled Beer Game comparison keeps the environment, planner, terminal-value component, training protocol, and evaluation setting fixed while replacing the structured transition with an unrestricted MLP. It therefore isolates the effect of supplying the stock-flow structure on imagined-state feasibility and multi-step rollout fidelity.

When topology, conservation relations, delays, and feasible-state geometry are available, relearning the complete next-state mapping spends scarce data on known mechanisms and permits physically inconsistent imagined trajectories. Empirically, Cartpole evaluates low-dimensional parameter uncertainty, while Beer Game evaluates explicit structural editing under known capacity and communication interventions. Unknown topology misspecification, incorrect constraints, and unmodelled residual mechanisms remain outside the current empirical guarantee.

## Training and decision-time workflow

We will add an explicit training/inference workflow distinguishing component learning from decision-time planning. In Beer Game and Cartpole, trainable transition parameters are optimized through task-specific reconstruction objectives within the supplied mechanistic family, and separate terminal estimators are trained from task-specific return targets. Depending on the protocol, these components are trained before evaluation or updated periodically from interaction data; planning uses the corresponding current or frozen components. At each decision, a finite-budget planner evaluates continuation value at the explored frontier and executes the selected action.

Sokoban uses its task-specific structured planner with a pretrained frozen terminal estimator. Hopper retains the TD-MPC2 online learning loop and sampling-based solver while integrating the structured model through the solver interface.

## Long-horizon evidence and experimental roles

The submitted tasks test complementary properties: Beer Game evaluates delayed stock-flow planning over 36 periods; Cartpole evaluates low-data estimation in parameter-sensitive continuous dynamics; and Hopper evaluates structured-model integration in sparse-reward, contact-rich control.

In a post-submission supplementary evaluation, we evaluated SD-MCTS on the `Sokoban-v0` backend used by Thinker. Sokoban requires extended action sequences under sparse success signals and irreversible deadlocks. Across five seeds, the frozen-estimator system reaches a rolling last-200 completed-episode solve rate of $0.960\pm0.015$ at 20k online primitive frames per seed. Training the frozen terminal estimator required a separately audited collection cost of 4,289 primitive frames. Under the reported per-endpoint accounting, 20,000 online frames plus this cost are counted as 24,289 external frames. This is supplementary long-horizon evidence under a frozen-estimator endpoint protocol, not a Thinker-identical continuously trained learning curve.

## Horizons, active parameters, and sensitivity

We distinguish task horizon from online planning control. Beer Game and Cartpole use budget-limited tree expansion with frontier value evaluation: neither has a separately configured fixed tree-depth cap, and each uses 200 simulations per root. Hopper and Sokoban use explicit planning horizons of 3 and 20. Active settings are:

- **Beer Game:** 36 periods; 200 simulations/root; 81 joint actions; PUCT $c=1.4$; $\gamma=0.99$.
- **Cartpole:** 1,000-step cap; 200 simulations/root; 11 action branches; action repeat 2; UCB $c=1.5$; $\gamma=0.99$; 35k development transitions for the shared value prior plus 5k real transitions per target seed.
- **Hopper:** 500-step episodes; horizon 3; 256 sampled sequences; 6 refinement iterations; 32 elites; 24 policy proposals.
- **Sokoban:** at most 80 macro decisions; horizon 20 push macros; 4,096 simulations/root; frozen terminal estimator; 20k online frames per seed, with the separate 4,289-frame collection cost included in the endpoint accounting.

These settings are fixed across evaluation seeds within each task.

The reviewer cites $\lambda_{\mathrm{td}}$ as an example of an important tuning parameter. The coefficient in Eq. (5) is not active in the reported execution paths: Beer Game and Cartpole train terminal estimators from task-specific return targets; Hopper retains the native TD-MPC2 critic objective, whose value coefficient is distinct from $\lambda_{\mathrm{td}}$; and Sokoban evaluates a pretrained frozen terminal estimator. We will therefore report the parameters that govern each pipeline.

Multi-seed results characterize seed stability under a fixed protocol; budget and checkpoint results characterize sample efficiency and training progress; component comparisons characterize attribution under fixed settings. These are not sensitivity studies, and we do not claim universal insensitivity. A systematic sweep would address tuning robustness and compute–performance trade-offs; the current results support seed stability, sample efficiency, and component attribution under the disclosed protocols.

## Relation to MPC baselines

The Beer Game control addresses the model-level question central to our claim. Under a fixed MCTS, terminal-value component, training protocol, and evaluation setting, the supplied stock-flow structure enforces feasibility by construction and improves the reported multi-step rollout fidelity relative to an unrestricted MLP.

A matched receding-horizon MPC controller using the same structured model would address a complementary planner-level question by changing the planner while holding model information fixed. It would sharpen attribution between MCTS-based search and MPC while complementing, rather than replacing, the current model-level evidence. TD-MPC2 in Hopper demonstrates compatibility with a strong model-predictive solver, although it is not a matched classical MPC baseline for Beer Game or Cartpole. We do not claim that MCTS universally outperforms MPC.

## Projection and theoretical assumptions

Equation (2) uses $\Pi_{\mathcal C}$ as a unified feasibility-enforcement interface. For a closed convex feasible set, it may be instantiated as the non-expansive nearest-point projection

$$
\Pi_{\mathcal C}(x)
=
\arg\min_{z\in\mathcal C}\lVert z-x\rVert_2.
$$

The realization is task-specific. In Beer Game, conservation is encoded in the stock-flow equations, while bounded and clamped flows enforce non-negativity and capacity limits. Feasibility enforcement is evaluated using the physical violation rate of imagined states within the MCTS search tree, while predictive accuracy is evaluated using multi-step state RMSE.

The assumptions serve proposition-specific roles rather than forming one universal set of prerequisites. The known structural family and low-dimensional parameterization define the grey-box setting. Proposition 1 additionally uses sub-Gaussian noise, persistent excitation, and a generative-oracle abstraction; online trajectories introduce additional mixing-time factors. Proposition 2 uses positive invariance and a local contraction bound. The horizon–accuracy comparison is stated for $H\ll\tau$ in terms of parameter bias $L_\theta\delta_\theta$ and terminal-value estimation error $\epsilon_V$. The bias analysis uses a fixed rollout policy, concentration conditions, reward/value Lipschitz assumptions, and a uniform terminal-value approximation bound to isolate model-error propagation; it is not a convergence analysis of the full adaptive MCTS tree policy.

These are sufficient conditions for their corresponding guarantees. When a condition fails, the associated bound should not be invoked; the procedure may still be applied whenever the task-level model and feasibility operations remain well defined. We will map each condition to its conclusion and distinguish problem-setting assumptions from analytical simplifications.

We will revise the paper to make the scope, workflow, experimental roles, active parameters, projection implementation, and claim boundaries explicit, and to include Sokoban as supplementary long-horizon evidence. We hope these clarifications and additional evidence address the concerns underlying the current rating, and we respectfully ask you to reconsider it.