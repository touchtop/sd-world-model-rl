# Response to Reviewer BVBw

Thank you for the careful and candid review. Your questions highlight that the compressed presentation can obscure the distinction between the assumed grey-box information, the learned quantities, and the decision-time procedure. We clarify these points using the Beer Game as a running example.

## Scope and Concrete Model Construction

SD-MCTS targets low-data grey-box planning: a credible mechanistic skeleton is available, while task-relevant dynamical parameters must be identified and continuation values estimated from limited interaction. The topology, state semantics, analytical update skeleton, and feasible constraints are designer-specified; SD-MCTS identifies uncertainty within this transition family rather than discovering arbitrary dynamics from raw observations. The central contribution is the closed loop of low-dimensional structured identification, feasibility-constrained planning, and terminal-value compensation at the explicitly explored search frontier. Known-intervention editing and the Hopper extension are capabilities enabled by the explicit representation.

In the four-echelon Beer Game, retailer, wholesaler, distributor, and factory form a specified serial graph. Inventory and backlog are stocks; orders, shipments, demand fulfillment, and factory production are flows; order and shipment delays are represented by two-slot pipelines. A one-step transition receives arriving shipments, advances the pipelines, injects demand and delayed upstream orders, dispatches shipments subject to inventory and backlog, and generates production subject to capacity. Conservation is encoded in these difference equations, while non-negativity and capacity limits are enforced during each transition. The topology, delays, update sequence, cost definition, and baseline capacities are fixed.

The submitted Beer Game transition learns only three demand-process parameters: `demand_low`, `demand_high`, and `demand_step_time`. It contains no learned residual dynamics or learned reward model; all remaining state updates and immediate rewards are fixed analytical computations. The low-data claim therefore concerns concentrating scarce data on genuinely uncertain quantities rather than relearning known topology and conservation relations.

The same representation supports intervention. For the capacity shock, the protocol specifies that factory capacity changes from 20 to 2 for $t \geq 15$, so that local capacity term is edited. For link disruption, the wholesaler order-flow term is set to zero for $10 \leq t \leq 20$. Unaffected equations, delay pipelines, and mechanisms are retained, and planning is immediately rerun. The setting studies replanning after a known intervention supplied by the test protocol; it does not claim to detect or localize an unknown failure from observations. “Zero-shot” means that no post-intervention parameter fitting or value-network retraining is performed before replanning.

Imperfect parameter estimates may still cause quantitative prediction error. Structural editing removes the additional mismatch caused by retaining a capacity or edge known to have changed; it does not remove ordinary parameter error. In the evaluated interventions, the learned demand mechanism remains unchanged, so its estimates retain the same semantics after the local edit. Interventions that alter learned mechanisms are outside the present evaluation scope.

## Decision-Time Search Procedure

At each decision, the current state initializes a root and a fixed simulation budget is used. Each simulation traverses the partial tree using an upper-confidence rule, expands one untried action at the current frontier, predicts the successor and edge reward through the structured model, evaluates the resulting frontier with $V_\psi$, and backs up the finite-path return to update $N$, $W$, and $Q$. Later simulations restart from the root and selectively widen or deepen the partial tree; UCB/PUCT allocates the limited simulation budget without exhaustively enumerating action sequences.

For a frontier reached at depth $d$, the backed-up return is

$$
G_d
===

\sum_{k=0}^{d-1}\gamma^k r_{t+k}
+
\gamma^d V_\psi(s_{t+d}).
$$

In the reported value-guided Beer Game and Cartpole paths, $V_\psi$ is applied at the newly expanded frontier; no separate post-expansion rollout policy is used. The finite-lookahead formulation uses $H$ as the common horizon notation; in the reported value-guided path, each simulation bootstraps at its realized frontier depth $d$, which may vary as repeated simulations selectively widen or deepen different branches. We will make this implementation-level distinction explicit in Appendix B.1.

Hopper retains the TD-MPC2 online interaction/replay loop and MPPI-style action-sequence solver, while integrating a mechanism-aligned state representation, structured transition/projection, residual adaptation, and online structured-transition supervision behind the retained interfaces. It does not use the Beer Game/Cartpole MCTS execution loop. The Hopper experiment therefore tests whether the structured representation can operate within an established continuous-action sampling solver, rather than isolating a transition-only effect.

## Interpretation of the Theory

The propositions address four distinct levels: low-dimensional identification under a known transition family, persistent excitation, and generative sampling (Proposition 1); multi-step error propagation, with the horizon-independent bound requiring local $L_f < 1$ (Proposition 2); the condition under which terminal-value compensation offsets finite-horizon truncation (Proposition 3); and the separation of finite-sample variation from systematic model/value bias under a fixed rollout policy (Proposition 4). The last result isolates model mismatch and is not a convergence theorem for the adaptive MCTS tree policy.

Here, $L_f$ is the effective Lipschitz constant of the controlled one-step mapping over the relevant operating region. Stock-flow structure or projection does not automatically imply $L_f < 1$. Projection onto a closed convex feasible set is non-expansive, so it does not add distance amplification; contractivity of the complete mapping remains task- and region-dependent. The $O(1)$ horizon-independent bound is conditional on local $L_f < 1$; $L_f = 1$ yields at most linear growth, and $L_f > 1$ may yield geometric growth. In the Beer Game operating distribution, finite-difference analysis estimates $\hat{L}_f \approx 0.96$. This is local empirical evidence, not a global analytical guarantee.

## Experimental Evidence and Baseline Choice

The central empirical claim concerns the interaction threshold for entering a usable regime, rather than universally higher asymptotic return. In Beer Game, SD-MCTS reaches $-710 \pm 45$ at 1k interactions, while DreamerV3 reaches a comparable $-713 \pm 157$ at 10k. Figure 3 plots the same pre-specified 1k, 3k, and 10k budgets as Table 4; intermediate budgets were not evaluated under that protocol, so it is a fixed-budget threshold comparison rather than a densely sampled learning curve. These observed multi-seed checkpoints support the narrower conclusion that structure lowers the interaction threshold in this task.

The three tasks serve distinct evidential roles: Beer Game examines delayed stock-flow planning and known structural edits, Cartpole examines low-dimensional identification in parameter-sensitive dynamics, and Hopper tests integration with a continuous-action solver. Our empirical scope is limited accordingly.

The baseline design also separates complementary questions. DreamerV3 represents generic neural world-model learning, TD-MPC2 a strong continuous-action model-predictive solver, and BlackBox-MCTS keeps the Beer Game planner and value-estimation role fixed while replacing the structured transition with an MLP. For model-structure attribution, BlackBox-MCTS is the more controlled comparison: the structured model yields 0.00% imagined-state violations versus 93.66% for BlackBox-MCTS, together with substantially lower multi-step error.

MuZero and EfficientZero are relevant complementary baselines because they combine learned models with search, but they also jointly change representation, learning targets, and search. Such a comparison would address the broader question of planner-family competitiveness. Accordingly, the submitted evidence supports the model-structure claim under controlled planning conditions; we do not make a planner-family superiority claim over MuZero or EfficientZero.

We will add the Beer Game worked example and decision-time loop, distinguish the unified horizon notation $H$ from realized frontier depth $d$, define the theoretical conditions directly, and make the contribution and baseline hierarchy explicit.
