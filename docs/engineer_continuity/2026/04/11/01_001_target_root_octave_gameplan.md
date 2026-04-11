# Target Root Octave Episode Gameplan

Date: 2026-04-11
Role: engineer continuity gameplan for target-conditioned episode design
Authority: additive planning artifact for implementing random starts, sampled target root octaves, target-conditioned reward, and transformer conditioning

## Purpose

This document captures the implementation gameplan for the newly bound episode/task design direction:

- initial chord should be random at reset
- each episode should sample a `target_root_octave`
- chord root is defined as `m_0` for a chord `(m_0, ..., m_{n-1})`
- reward should include an inverted-distance-to-target-octave signal
- the target must be part of transformer input

This is a planning artifact, not an implementation report.

## Bound Design Decisions

The following decisions are treated as bound for this gameplan:

1. The root of a chord is the lowest MIDI note `m_0`.
2. `target_root_octave` is sampled randomly at `reset()`.
3. Episode reward should include some inverted-distance-to-target shaping.
4. The target must be present in the transformer-facing policy input.
5. Episode runtime remains fixed-length rather than waiting for a fully solved cadence termination rule.

## Phase 9

### Stage 9.1

#### Action 9.1.1

Purpose:
Bind the exact octave semantics before code changes.

Ground-truth files:

- `docs/design/graph_spec_001.md`
- `rl_counterpoint/graph/state_space.py`
- `rl_counterpoint/reward/protocol.py`

Machine operation:

- define the canonical MIDI-to-octave mapping
- define `root_octave(chord) := octave(m_0)`
- define the allowed sampled `target_root_octave` range

Associated tests:

- unit tests for octave mapping and root-octave extraction

Failure hypotheses:

- octave numbering drifts from the Project Owner's intended MIDI convention
- reward code and diagnostics compute octave differently
- sampled target range includes unreachable or nonsensical bands

### Stage 9.2

#### Action 9.2.1

Purpose:
Add episode-goal state to the environment contract.

Ground-truth files:

- `rl_counterpoint/envs/counterpoint_env.py`
- `rl_counterpoint/envs/termination.py`
- `tests/envs/test_counterpoint_env.py`

Machine operation:

- extend env state with `target_root_octave`
- sample `target_root_octave` randomly at `reset()`
- expose it in `info`
- ensure it persists across the episode

Associated tests:

- reset returns a sampled valid target octave
- target octave stays stable through episode steps
- target octave is present in `info`

Failure hypotheses:

- target is resampled accidentally during `step()`
- env exposes target in some paths but not others
- reset semantics become nondeterministic in tests without explicit seeding

## Phase 10

### Stage 10.1

#### Action 10.1.1

Purpose:
Replace fixed initial chord with random valid start-state sampling.

Ground-truth files:

- `rl_counterpoint/envs/counterpoint_env.py`
- `rl_counterpoint/graph/state_space.py`
- `tests/envs/test_counterpoint_env.py`

Machine operation:

- implement random valid initial chord sampling from the existing graph/node space
- bind any sampling constraints needed for tractable and meaningful starts
- wire sampled start into `reset()`

Associated tests:

- sampled starts are valid nodes
- repeated resets can produce different starts
- seeded resets are reproducible if seeding is already supported

Failure hypotheses:

- unconstrained start sampling creates unreachable target-octave tasks
- sampling becomes too expensive if node enumeration is naive
- random starts break assumptions in existing env tests

## Phase 11

### Stage 11.1

#### Action 11.1.1

Purpose:
Extend reward context for target-conditioned register reward.

Ground-truth files:

- `rl_counterpoint/reward/protocol.py`
- `tests/reward/test_protocol.py`
- `rl_counterpoint/envs/counterpoint_env.py`

Machine operation:

- add the minimum new `RewardContext` fields for:
  - current root octave
  - target root octave
  - final-step or terminal-step awareness
- populate those fields from the environment

Associated tests:

- protocol construction covers new fields
- env passes correct reward context values
- final-step flag is correct at the episode boundary

Failure hypotheses:

- too much env internals leak into reward context
- final-step semantics and truncation semantics diverge
- root octave is recomputed inconsistently between env and reward

### Stage 11.2

#### Action 11.2.1

Purpose:
Implement the first target-octave reward family.

Ground-truth files:

- `rl_counterpoint/reward/black_box.py`
- `tests/reward/test_black_box.py`

Machine operation:

- add an inverted-distance-to-target reward component
- add optional larger end-of-episode bonus for exact target-octave arrival
- preserve diagnostics so raw distance and scored reward are inspectable

Associated tests:

- exact-hit reward case
- near-target reward case
- far-target reward case
- final-step bonus behavior

Failure hypotheses:

- shaping reward overwhelms all other future musical signals
- terminal bonus is too sparse to help learning
- reward becomes opaque because diagnostics only expose final scalar score

## Phase 12

### Stage 12.1

#### Action 12.1.1

Purpose:
Inject target-root-octave into the canonical transformer input path.

Ground-truth files:

- `rl_counterpoint/envs/observation.py`
- `rl_counterpoint/models/policy.py`
- `tests/models/test_policy.py`
- `tests/envs/test_observation.py`

Machine operation:

- extend the observation/context representation so target octave is part of the policy-facing input
- update symbolic/context rendering to include target-octave conditioning
- keep one canonical ownership path for this context encoding

Associated tests:

- rendered policy context includes target octave
- encoded windows remain shape-consistent
- different targets produce different context strings or embeddings

Failure hypotheses:

- target is added in policy code but not observation code, creating split authority
- target encoding is too weak or ambiguous for conditioning
- context-shape changes silently break rollout or trainer assumptions

## Phase 13

### Stage 13.1

#### Action 13.1.1

Purpose:
Align rollout and training harness with the new target-conditioned episode contract.

Ground-truth files:

- `rl_counterpoint/algos/rollout.py`
- `rl_counterpoint/algos/reinforce.py`
- `scripts/train_reinforce.py`
- `tests/algos/test_rollout.py`
- `tests/algos/test_reinforce.py`
- `tests/test_train_reinforce.py`

Machine operation:

- ensure rollout records preserve the target-conditioned observation path
- verify trainer recomputation uses the same target-conditioned input
- update training script config/logging so target-conditioned episodes are visible in metrics or diagnostics

Associated tests:

- rollout records remain internally consistent
- reinforce episode stats still compute correctly
- training script still runs and logs meaningful episode summaries

Failure hypotheses:

- rollout stores windows that no longer match recomputed training inputs
- the target is visible during rollout but lost during loss recomputation
- script metrics remain too thin to diagnose target-conditioning failures

## Phase 14

### Stage 14.1

#### Action 14.1.1

Purpose:
Add direct smoke visibility for the new episode/task structure.

Ground-truth files:

- `scripts/smoke_env.py`
- `scripts/smoke_reward.py`
- `tests/test_smoke_env.py`
- `tests/test_smoke_reward.py`

Machine operation:

- update env smoke to print sampled start chord, target root octave, and ending outcome
- update reward smoke to show raw octave distance and scored reward on hand-picked examples

Associated tests:

- smoke scripts execute successfully
- smoke output exposes target-conditioned diagnostics

Failure hypotheses:

- reward debugging stays trapped inside training runs
- smoke artifacts omit the actual target-conditioning signal
- no-data or edge cases are not surfaced clearly

## Phase 15

### Stage 15.1

#### Action 15.1.1

Purpose:
Run a first target-conditioned end-to-end validation pass.

Ground-truth files:

- `scripts/train_reinforce.py`
- `artifacts/train_reinforce/`
- relevant updated tests above

Machine operation:

- run focused test subsets for env, observation, reward, policy, rollout, and trainer
- run one direct training harness execution
- inspect whether episode diagnostics show variation by target and distance

Associated tests:

- targeted pytest slices for touched modules
- one CLI training run

Failure hypotheses:

- rewards remain effectively flat despite the new target
- random starts plus random targets create unreachable tasks too often
- policy conditioning path compiles but is not behaviorally reflected in metrics
