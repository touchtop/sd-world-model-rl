# Response to Reviewer 6uVV

Thank you for recognizing the motivation for exploiting known structure in low-data control and for asking us to clarify the information regime, evidence attribution, and theoretical scope.

## Scope and contribution

SD-MCTS targets **low-data grey-box planning**. In the native Beer Game and Cartpole implementations, state semantics, macroscopic topology $\mathcal{T}$, analytical update skeleton $f_{\mathcal{T}}$, and feasibility rules $\mathcal{C}$ are supplied. Data-driven transition estimation is restricted to task-relevant quantities within this family; continuation values are learned separately and planning statistics are computed online.

Beer Game fixes the four-echelon stock-flow topology, separate two-slot order and shipment pipelines, balance equations, and feasibility rules, while representing the step-demand profile with three trainable scalars. Cartpole fixes damped Newton–Euler equations and RK4 integration while estimating cart mass, pole mass, and pole half-length from 5,000 real transitions per target seed. Hopper is a hybrid cross-solver extension that embeds a structured mechanics core within retained learned TD-MPC2 components.

We therefore use “structured world model” in a grey-box sense: the predictive model is constrained by supplied mechanism knowledge while task-dependent quantities and continuation estimates remain data-dependent. The contribution goes beyond system identification because the resulting model directly supports feasibility-preserving decision-time planning; it also differs from fully known constrained planning because relevant dynamics and values remain learned. We will apply this scope and the designer-specified/data-learned distinction consistently in the revision.

## Incorrect or incomplete structural knowledge

Missing edges, incorrect conservation relations, and unmodeled residual dynamics constitute **model-family misspecification**, distinct from the well-specified grey-box regime studied here. Numerical uncertainty within a fixed analytical family can be handled through parameter estimation; Cartpole provides the direct example. Known structural changes can instead be incorporated through the intervention interface. When the true dynamics lie outside the prescribed family, adjusting the existing parameter vector need not remove systematic planning bias.

The Beer Game intervention experiment addresses supplied structural change: the edit is provided at test time and written into the model. SD versus $\mathrm{SD}_{\mathrm{no\_update}}$ measures the effect of applying that known edit and does not rely on detecting an unknown structural error. The learned outer adapter in Hopper was likewise not evaluated as a mechanism for discovering or correcting arbitrary missing edges or wrong constraints. The present claim is therefore conditional on a credible mechanistic skeleton.

## Attribution of the Hopper result

The Hopper experiment evaluates **cross-solver compatibility and system-level performance of the active hybrid SD adapter**, not the isolated effect of a single transition component.

The submitted path retains the TD-MPC2-style online interaction, replay, and update loop and the MPPI-style action-sequence solver. It maps observations into a mechanism-aligned state, advances that state through an explicit mechanics-based transition core, projects predicted structured states into fixed bounds, and applies non-zero online one-step and rollout supervision to the transition core. The surrounding hybrid interface also contains an outer neural residual adapter and retains learned latent, reward, value, and policy components.

Planning uses the learned reward head. Curriculum-based structural reward shaping was inactive: the logged reward-mixing coefficient and shaping term are zero. Height-head auxiliary supervision was also inactive: the submitted structured model has no active height head and the logged height loss is identically zero.

The result therefore tests whether this active hybrid adapter can operate effectively inside an established continuous-control planning pipeline under a low interaction budget. It does not identify the marginal contribution of each active component, and we do not attribute the full gain to the mechanics core alone.

Beer Game provides the most direct model-class evidence. In the controlled physics-consistency comparison, SD-MCTS and BB-MCTS share the environment, state/action and reward interfaces, interaction budget, decision-time MCTS function and settings, value-network architecture, and evaluation protocol. They differ in world-model parameterization and model-learning rate, and each learns its own model and value weights from its own trajectories. The comparison therefore focuses on supplied stock-flow structure versus an unrestricted MLP next-state model; it is not an experiment on recovering an initially unknown or deliberately misspecified demand process.

## Projection, predictive accuracy, and rollout-error theory

Projection and prediction accuracy have different roles:

$$
P_\theta(s,a)
=
\Pi_{\mathcal{C}}\left(f_{\mathcal{T}}(s,a;\theta)\right).
$$

Here, $f_{\mathcal{T}}$ produces a mechanism-based candidate update and $\Pi_{\mathcal{C}}$ enforces feasibility. For exact projection onto a closed convex set, if $s^\star \in \mathcal{C}$,

$$
\left\|
\Pi_{\mathcal{C}}(\widetilde{s})-s^\star
\right\|
=
\left\|
\Pi_{\mathcal{C}}(\widetilde{s})-\Pi_{\mathcal{C}}(s^\star)
\right\|
\leq
\left\|
\widetilde{s}-s^\star
\right\|.
$$

Thus, exact projection cannot increase Euclidean error to a feasible ground-truth state, but it can map a substantially inaccurate candidate to a feasible state. Feasibility therefore does not establish dynamical correctness or reveal pre-projection error. In the task implementations, feasibility is enforced through analytical constraints, closed-form bounds, and task-specific projection operators; the non-expansiveness statement applies specifically to exact closed-convex projection.

This is why physical-violation rate and multi-step rollout RMSE are reported separately. The former measures constraint satisfaction; the latter measures the complete constrained transition against the true trajectory. Beer Game’s zero rollout RMSE follows from the structured update matching the deterministic environment rules and is not attributed to projection alone.

The rollout-error result is also conditional. From

$$
e_{t+1}
\leq
L_f e_t+\epsilon_{\mathrm{model}},
$$

a horizon-independent bound requires local contraction, $L_f<1$; $L_f=1$ gives at most linear growth, while $L_f>1$ permits geometric amplification. Non-expansive projection alone does not establish $L_f<1$. The Beer Game appendix reports a fixed-action local finite-difference diagnostic consistent with a locally contractive regime; we treat it only as a local diagnostic, not as a verified maximum, global Lipschitz bound, or contraction proof. Contact-rich Hopper may cross multiple local regimes and is not claimed to satisfy a global contraction condition. The algorithm remains executable there, but the horizon-independent $O(1)$ guarantee should not be invoked.

## Test-time information for structural interventions

The experiment evaluates immediate replanning after a test-time-supplied structural edit. Factory production capacity is reduced from $20$ to $2$ on transitions with current-state time $t\geq15$. For $10\leq t\leq20$, the wholesaler’s placed-order quantity is forced to zero.

SD receives the intervention label and schedule and applies the corresponding model edit. $\mathrm{SD}_{\mathrm{no\_update}}$ retains the same trained structured-model and value-network weights but does not apply the edit. BB receives only observations from the changed environment and no explicit structural-edit input. No world-model or value-network optimizer step occurs after the intervention is selected.

“Zero-shot” therefore means that the intervened configuration was not activated during training and no intervention-specific fitting occurs before replanning; it does not mean autonomous detection or localization from observations. SD versus $\mathrm{SD}_{\mathrm{no\_update}}$ measures the value of immediately applying supplied structural information within the same trained model family. BB answers a different representational question under the changed environment without an edit input or post-intervention retraining. A neural model with matched intervention metadata would test intervention-conditioned learned adaptation; the current results are not presented as an information-matched comparison against such a baseline.

Overall, the evidence supports a conditional, layered claim: Beer Game compares supplied stock-flow structure with an unrestricted next-state model; Cartpole demonstrates low-dimensional parameter estimation from limited real transitions; Hopper evaluates hybrid integration with a retained continuous-control solver; trajectory analyses separate feasibility from rollout accuracy; Proposition 2 characterizes conditional local error propagation; and the intervention study evaluates immediate replanning after a supplied edit.