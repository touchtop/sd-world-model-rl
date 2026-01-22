# M1 Mismatch Experiment Findings

> **Date**: 2026-01-15
> **Status**: Experimental design flaw identified, requires revision

---

## 1. Original Hypothesis

Parameter mismatch between MPC's belief and true environment should cause MPC to make suboptimal decisions, while SD-MCTS (with learned model) should adapt.

## 2. Experimental Results

| Scenario | True Params | MPC Belief | MPC Result | SD-MCTS Result | Gap |
|----------|-------------|------------|------------|----------------|-----|
| mismatch_optimistic | μ=(0.5,0.5), λ=1.5 | μ=(1.2,1.2) | 22.74 | 22.77 | ~0 |
| mismatch_pessimistic | μ=(1.0,1.0) | μ=(0.5,0.5) | 3.52 | 3.52 | 0 |
| mismatch_arrival | λ=1.8 | λ=0.7 | 35.02 | 35.42 | ~0 |

**Observation**: MPC ≈ SD-MCTS despite 50-140% parameter mismatch.

## 3. Root Cause Analysis (Codex + Gemini Consensus)

### 3.1 Symmetric Mismatch Preserves Ordering

In discrete action space (choose worker A or B):
- MPC decision: `action = argmin(expected_delay)`
- With symmetric workers and symmetric mismatch, **relative ordering is preserved**
- Example: True μ=(0.5,0.5) vs Belief μ=(1.2,1.2) → both workers equally good in both models

### 3.2 Scale Invariance

```
J(action) = f(μ, state)
action* = argmin J(action)
```

Scaling μ uniformly doesn't change argmin result.

### 3.3 System Instability Issue (Gemini)

**Critical flaw in `mismatch_optimistic`**:
- True: λ=1.5, μ_total=1.0 → λ > μ (unstable system)
- Queue grows to `max_queue` regardless of policy
- Results reflect saturated system, not policy quality

## 4. Recommended Experimental Redesign

### Priority 1: Asymmetric Mismatch (Ranking Inversion)

| Config | True | Belief | Expected MPC Failure |
|--------|------|--------|---------------------|
| Blind Spot | μ_A=1.5, μ_B=0.5 | μ_A=1.0, μ_B=1.0 | Routes 50/50, overloads B |
| Inverted | μ_A=1.5, μ_B=0.6 | μ_A=0.6, μ_B=1.5 | Routes to B (wrong choice) |

### Priority 2: Fatigue Threshold Mismatch

| Config | True Threshold | Belief Threshold | Expected MPC Failure |
|--------|----------------|------------------|---------------------|
| Cliff | θ=8 | θ=15 | Pushes past 8, triggers collapse |

### Priority 3: Dynamic Drift

- Parameters change at t=T/2
- MPC uses static belief, cannot adapt
- SD-MCTS should update model online

## 5. Action Items

1. [ ] Implement asymmetric mismatch scenarios
2. [ ] Implement fatigue threshold mismatch
3. [ ] Ensure system stability (λ < μ_total) in all scenarios
4. [ ] Consider if SD-MCTS actually updates model online (currently may not)

## 6. Key Insight for Paper

**Claim Refinement Needed**:
- Original claim: "SD-MCTS is robust to parameter uncertainty"
- Issue: Current SD-MCTS may use fixed belief like MPC
- Required: SD-MCTS must have **online model adaptation** to show advantage

---

## References

- Codex analysis session: 019bbead-a1d1-7151-8807-e6d395a526d9
- Gemini analysis session: 950af78e-7e43-4983-8575-ae23d0a999c8
