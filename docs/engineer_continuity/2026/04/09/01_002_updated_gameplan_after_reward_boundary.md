# Updated Gameplan After Reward Boundary Work

Date: 2026-04-09  
Role: updated engineer continuity gameplan after today's reward-boundary implementation work  
Authority: additive continuity artifact reflecting current repo reality after the first TC21M-facing reward slices

## Purpose

This document updates today's gameplan after the following work was completed in-repo:

- reward contract extension
- beat-role reward diagnostics
- a first local static consonance rule family
- a first beat-sensitive consonance rule family

This document is not a full project rewrite.

It answers:

```text
What Phase.Stage.Action work is still missing and still relevant now?
```

## Ground-Truth Status Update

### Completed today

The following previously-live work is now implemented:

- `Phase 2.Stage 2.1.Action 2.1.1`
  - reward protocol sufficiency decision
- `Phase 2.Stage 2.2.Action 2.2.1`
  - reward contract extension was specified and implemented
- `Phase 3.Stage 3.1.Action 3.1.1`
  - first slice: beat-role diagnostics only
  - second slice: one local rule family with diagnostics
  - third slice: one beat-sensitive rule family with placeholder weights

Concrete current repo consequences:

- `RewardContext` now carries:
  - `step_delta`
  - `key_pitch_class`
  - `timed_chord_window`
- `rl_counterpoint/reward/black_box.py` now contains:
  - `ConstantReward`
  - `BeatRoleDiagnosticReward`
  - `StaticConsonanceReward`
  - `StrongBeatConsonanceReward`

### Still missing but still relevant

The following work remains live:

- TC21M reward formalization beyond the first local / beat-sensitive slices
- cadence-sensitive and history-sensitive evaluator logic
- a cleaner stable musical helper boundary
- graph and reward smoke scripts
- value-model / critic path
- unification of design-side sparsity counting with the live graph spec contract

## Updated Gameplan

### Phase 4

#### Stage 4.1

##### Action 4.1.1

Purpose:
Classify the remaining TC21M rule families into local, beat-sensitive, cadence-sensitive, and history-sensitive bands using the current reward implementations as reference checkpoints.

Ground-truth files:

- `assets/rules/tc21m_rules.md`
- `rl_counterpoint/reward/black_box.py`
- `rl_counterpoint/reward/protocol.py`

Machine operation:
Read the current rules asset and map each still-relevant family against what is already implemented versus what remains absent.

Associated tests:

- none at classification time

Failure hypotheses:

- current rule notes are still too editorial to bind directly into implementation order
- some “beat-sensitive” rules are actually cadence- or phrase-sensitive
- overlapping rule families may be accidentally counted twice

#### Stage 4.2

##### Action 4.2.1

Purpose:
Implement the first cadence- or history-sensitive reward family only after the classification in Stage 4.1 is explicit.

Ground-truth files:

- `assets/rules/tc21m_rules.md`
- `rl_counterpoint/reward/black_box.py`
- `tests/reward/test_black_box.py`

Machine operation:
Choose one narrowly scoped next reward family that genuinely uses longer-horizon or ending-position context and implement it with explicit diagnostics.

Associated tests:

- add focused reward tests for the selected family

Failure hypotheses:

- cadence logic gets implemented before the cadence signal is sufficiently specified
- a longer-horizon rule accidentally duplicates transition-local logic already present
- diagnostics become too opaque to support reality checks

### Phase 5

#### Stage 5.1

##### Action 5.1.1

Purpose:
Create a reward smoke script that exposes reward diagnostics directly without requiring rollout or training.

Ground-truth files:

- `scripts/smoke_reward.py`
- `rl_counterpoint/reward/black_box.py`
- `rl_counterpoint/reward/protocol.py`

Machine operation:
Populate `scripts/smoke_reward.py` with a deterministic local smoke entrypoint that prints reward diagnostics for one or more hand-picked chord examples.

Associated tests:

- add or revise a smoke-script test if needed

Failure hypotheses:

- reward debugging remains unnecessarily coupled to env or trainer execution
- the smoke path silently diverges from the live reward contract

#### Stage 5.2

##### Action 5.2.1

Purpose:
Create a graph smoke script that exposes graph legality and action-mask behavior directly.

Ground-truth files:

- `scripts/smoke_graph.py`
- `rl_counterpoint/graph/actions.py`
- `rl_counterpoint/graph/non_crossing.py`
- `rl_counterpoint/graph/state_space.py`

Machine operation:
Populate `scripts/smoke_graph.py` with a direct graph/action inspection path using a fixed small example state.

Associated tests:

- add or revise a smoke-script test if needed

Failure hypotheses:

- graph debugging stays unnecessarily coupled to environment behavior
- smoke output omits the actual legality boundary that reward and policy depend on

### Phase 6

#### Stage 6.1

##### Action 6.1.1

Purpose:
Decide whether to create a real stable `music/` helper boundary or continue keeping musical primitives distributed across graph, env observation, and policy code.

Ground-truth files:

- `rl_counterpoint/music/__init__.py`
- `rl_counterpoint/graph/state_space.py`
- `rl_counterpoint/models/policy.py`
- `rl_counterpoint/reward/black_box.py`

Machine operation:
Inspect where pitch-class, interval, and symbolic chord logic currently lives and decide whether a refactor is warranted yet.

Associated tests:

- none until a refactor is approved

Failure hypotheses:

- a premature `music/` refactor creates churn without reducing ambiguity
- leaving the boundary implicit too long causes duplicate interval logic across modules

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

- empty `value.py` is mistaken for an active module boundary
- training complexity increases before reward/evaluator maturity justifies it

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
Compare the count script constants and trim logic against the live graph spec and decide whether to unify or explicitly document the remaining differences.

Associated tests:

- none unless code unification is approved

Failure hypotheses:

- the design script and live graph continue to evolve separately
- future graph changes become harder to trust because design counts no longer match code assumptions

## Current Priority Order

If work continues immediately from current repo reality, the recommended order is:

1. `Phase 4.Stage 4.1.Action 4.1.1`
2. `Phase 5.Stage 5.1.Action 5.1.1`
3. `Phase 5.Stage 5.2.Action 5.2.1`
4. `Phase 4.Stage 4.2.Action 4.2.1`
5. `Phase 6.Stage 6.1.Action 6.1.1`
6. `Phase 7.Stage 7.1.Action 7.1.1`
7. `Phase 8.Stage 8.1.Action 8.1.1`

This ordering reflects present reality:

- reward formalization is still the main live frontier
- smoke visibility is still missing in two important places
- `music/`, `value.py`, and graph-spec unification are relevant but not yet the highest-leverage next moves
