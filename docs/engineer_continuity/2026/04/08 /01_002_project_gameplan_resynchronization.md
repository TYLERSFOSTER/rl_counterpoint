# RL Counterpoint Project Gameplan Resynchronization

Date: 2026-04-08  
Scope: reconcile the original gameplan with the repository as it actually exists now  
Authority: this file is a continuity-side correction artifact and does not replace `01_001_project_gameplan_001.md`

## Purpose

This document exists because the original gameplan in this folder remains valuable as a planning artifact, but it is no longer a globally accurate record of actual repo progress.

The purpose of this file is:

- to mark which planned areas are now implemented
- to mark which planned areas remain absent
- to identify where the current real checkpoint is
- to prevent future conversation from treating the original gameplan as a linear execution ledger

This is a resynchronization artifact, not a silent overwrite of the original plan.

## Current Implemented Reality

The repository now contains the following working, tested infrastructure:

### Graph layer

- `rl_counterpoint/graph/graph_spec.py`
- `rl_counterpoint/graph/state_space.py`
- `rl_counterpoint/graph/non_crossing.py`
- `rl_counterpoint/graph/actions.py`

Implemented behavior:

- graph spec contract
- node validation and enumeration
- edge validation
- `StepDelta` action contract
- fixed bounded nonzero step-delta lattice
- step-delta legality mask

### Reward layer

- `rl_counterpoint/reward/protocol.py`
- `rl_counterpoint/reward/black_box.py`

Implemented behavior:

- `RewardContext`
- `RewardResult`
- `RewardFn`
- `ConstantReward` placeholder

### Environment layer

- `rl_counterpoint/envs/counterpoint_env.py`
- `rl_counterpoint/envs/observation.py`
- `rl_counterpoint/envs/termination.py`

Implemented behavior:

- validated constructor-provided initial state
- `reset() -> (obs, info)`
- `step(step_delta) -> (obs, reward, terminated, truncated, info)`
- invalid action as no-op plus explicit penalty/diagnostics
- max-step truncation
- `action_space` and `action_mask` in `info`

### Rollout and smoke artifacts

- `rl_counterpoint/algos/rollout.py`
- `scripts/smoke_env.py`
- `scripts/smoke_rollout.py`
- `scripts/train_reinforce.py`

Implemented behavior:

- one-episode rollout collector
- masked random action chooser
- environment smoke script
- rollout smoke script
- explicit training wait-state in `train_reinforce.py`

### Placeholder policy layer

- `rl_counterpoint/models/policy.py`

Implemented behavior:

- raw `ChordState` to tensor helper
- placeholder policy module emitting logits over fixed `StepDelta` action space

## Still Absent Or Intentionally Incomplete

The following remain not implemented or not yet committed as stable architecture:

- `music/` layer beyond `__init__.py`
- real learner update code in `rl_counterpoint/algos/reinforce.py`
- real value model in `rl_counterpoint/models/value.py`
- TC21M-derived evaluator replacing `reward/black_box.py`
- final training-facing observation contract

## How This Maps To The Original Gameplan

### Original Phase 0

Partially historical / scaffold-level only.

The repo shape and import contract were effectively established, but the original Phase 0 should not be treated as an actively tracked next-work band anymore.

### Original Phase 1

Mostly not implemented as written.

The dedicated `music/` modules planned there do not exist yet. Some of that intended responsibility is currently being carried by the graph layer instead.

This means Phase 1 is not "done"; it is partially bypassed and should be revisited explicitly rather than assumed complete.

### Original Phase 2

Substantially implemented, but with an important evolution:

- the project did not remain centered on direct next-state actions
- the learning-facing action contract is now `StepDelta`

So Phase 2 is partly completed and partly superseded by newer design decisions.

### Original Phase 3

Implemented in its early form:

- reward protocol exists
- black-box placeholder exists

The later TC21M replacement path remains future work.

### Original Phase 4

Implemented in its first real form:

- environment exists
- reset/step semantics exist
- masking/info contract exists
- truncation exists

### Original Phase 5

Partially implemented:

- rollout collection exists
- placeholder policy exists
- training smoke wait-state exists

Not implemented:

- real policy-driven rollout
- real trainer

### Original Phase 6

Not implemented.

### Original Phase 7

Partially implemented through smoke scripts and continuity practice, but not in the full sense described in the original plan.

## Current Critical Checkpoint

The most important open issue is no longer "can we scaffold the early stack at all?"

That work is already partly done.

The critical current checkpoint is:

### Observation And Policy Architecture

Before real policy-driven training code begins, the Project Owner needs an explicit checkpoint discussion about whether the true training-facing policy should consume:

- only the current chord state, or
- a bounded sequence/history of chord states

This matters because it changes:

- environment observation contract
- rollout record shape
- model input contract
- trainer assumptions

This must be discussed before more serious learning code hardens the wrong interface.

## Practical Interpretation

The original `01_001_project_gameplan_001.md` remains the earlier planning document.

This file should be read as the correction layer answering:

```text
What is actually true in the repo now?
```

not:

```text
What was originally imagined in sequence?
```

## Standing Warning

Future "next stage/action" answers must distinguish between:

1. the next item in the original planning text
2. the next item in the project as actually implemented

Those are no longer the same thing.
