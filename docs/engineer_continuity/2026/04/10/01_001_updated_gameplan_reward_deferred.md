# Updated Gameplan With Reward Deferred

Date: 2026-04-10  
Role: first engineer continuity artifact for today's session  
Authority: additive gameplan update reflecting the Project Owner decision to defer further reward development until the end

## Purpose

This document supersedes the active priority ordering from:

- `docs/engineer_continuity/2026/04/09/01_002_updated_gameplan_after_reward_boundary.md`

for one specific reason:

- further reward development is now intentionally deferred

The Project Owner decision is:

```text
Treat the current reward system as good enough for now.
Leave further reward development until the very end.
Do not keep driving project sequencing through TC21M reward expansion.
```

This does not mean reward is "finished."

It means reward is no longer the active frontier.

## Ground-Truth Status At This Update

### Reward state now

The reward boundary is implemented and usable:

- `rl_counterpoint/reward/protocol.py`
- `rl_counterpoint/reward/black_box.py`

The current repo already contains:

- a stable `RewardContext`
- `RewardResult`
- multiple reward implementations sufficient for placeholder/training/plumbing work
- direct reward smoke visibility via `scripts/smoke_reward.py`

Therefore further reward-family implementation is intentionally removed from the immediate gameplan.

### Smoke visibility now

Both previously missing smoke scripts are now implemented:

- `scripts/smoke_graph.py`
- `scripts/smoke_reward.py`

So the earlier Phase 5 smoke-script actions are complete and should not remain in the active queue.

### Still-missing work that remains relevant

The following items are still missing and still relevant after removing reward from the active frontier:

- a decision on whether to establish a stable `music/` helper boundary
- a decision on whether the near-term project actually needs a value-model / critic path
- reduction of drift between the design-side sparsity counting script and the live graph spec

## Updated Gameplan

### Phase 6

#### Stage 6.1

##### Action 6.1.1

Purpose:
Decide whether to create a real stable `music/` helper boundary or continue keeping musical primitives distributed across graph, env observation, policy, and reward code.

Ground-truth files:

- `rl_counterpoint/music/__init__.py`
- `rl_counterpoint/graph/state_space.py`
- `rl_counterpoint/models/policy.py`
- `rl_counterpoint/reward/black_box.py`

Machine operation:
Inspect where pitch-class, interval, consonance, and symbolic chord logic currently lives and decide whether a refactor is warranted now.

Associated tests:

- none unless a refactor is approved

Failure hypotheses:

- a premature `music/` refactor creates churn without giving a cleaner contract
- leaving the boundary implicit too long causes duplicated musical helper logic across modules
- symbolic encoding and musical interval logic remain coupled accidentally

### Phase 7

#### Stage 7.1

##### Action 7.1.1

Purpose:
Decide whether the project still needs a value-model / critic path in the near term.

Ground-truth files:

- `rl_counterpoint/models/value.py`
- `rl_counterpoint/algos/reinforce.py`
- `scripts/train_reinforce.py`

Machine operation:
Assess whether current policy-only REINFORCE remains the intended near-term learner path or whether value-model scaffolding should become active work.

Associated tests:

- none unless the value path is approved for implementation

Failure hypotheses:

- empty `value.py` is mistaken for an active architecture commitment
- training complexity rises before model/reward maturity justifies it
- a value path is added before the current explicit trainer has been fully exploited

### Phase 8

#### Stage 8.1

##### Action 8.1.1

Purpose:
Reduce drift between the design-side sparsity counting script and the live graph spec contract.

Ground-truth files:

- `docs/design/count_gn_sparsity.py`
- `docs/design/graph_spec_001.md`
- `rl_counterpoint/graph/graph_spec.py`

Machine operation:
Compare the count script constants and trim logic against the live graph spec and decide whether to unify them or explicitly document any remaining intentional differences.

Associated tests:

- none unless code unification is approved

Failure hypotheses:

- the design script and live graph evolve separately long enough that one no longer validates the other
- future graph changes become harder to trust because design counts no longer track runtime assumptions

## Deferred Work Register

The following work is intentionally deferred and should not be treated as the active next queue:

- further TC21M reward family classification
- further cadence-sensitive reward implementation
- further history-sensitive reward implementation
- broader reward hyperparameter development

These remain valid future work, but they are endgame items under the current project decision.

## Current Priority Order

If work continues immediately from current repo reality under the reward-deferred decision, the recommended order is:

1. `Phase 6.Stage 6.1.Action 6.1.1`
2. `Phase 7.Stage 7.1.Action 7.1.1`
3. `Phase 8.Stage 8.1.Action 8.1.1`

This ordering reflects the current intent:

- reward is no longer the active sequencing driver
- smoke visibility gaps have already been closed
- the main remaining non-reward questions are architecture-boundary questions, not more evaluator details
