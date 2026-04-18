# Tower Build Plan

This document is the Phase 6 / Stage 17 deliverable for the tower redesign.

The purpose is to turn the accepted tower design into machine-implementable actions.

This is still a plan, not implementation. It is written so a coding agent can execute each action without inventing new theory.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 6: Implementation Readiness Review |
| Stage | Stage 17: Produce machine-implementable build plan |
| Action | Break implementation into concrete file/function/test actions |

## Global Guardrails

These rules apply to every implementation action.

| Guardrail | Requirement |
| --- | --- |
| old system boundary | do not modify files under `rl_counterpoint/` |
| legacy dependency | do not import `rl_counterpoint/` from `tower/` at runtime |
| utility reuse | copy useful old code into `tower/`; copied code becomes tower-owned |
| canonical state | use tuple/list-like state representation first |
| canonical action | use tuple/list-like action representation first |
| tensors | tensor conversion belongs downstream or inside optional optimized batch helpers |
| no canonical parent field | parent is computed by projection, not stored as required state field |
| projection filename | use `tower/graph/projection.py`, singular |
| early slices | do not introduce neural training before Slice 7 |
| reward grammar | do not overbuild TC21M reward terms in early slices |

## Representation Contract

The first implementation slices use simple tuple-based contracts.

```python
MidiPitch = int
TowerState = tuple[int, ...]
TowerAction = tuple[int, ...]
```

Rank is normally:

```python
rank = len(state)
```

Public graph/projection/legality APIs should accept and return tuple-like objects.

Tensor use is allowed later as an internal optimization, especially for batched pruning, but tensor representation is not the core graph contract.

## Test Root

Tower tests should live under:

```text
tests/tower/
```

Use focused test files by module:

```text
tests/tower/test_state_action.py
tests/tower/test_window.py
tests/tower/graph/test_projection.py
tests/tower/action/test_assembly.py
tests/tower/graph/test_actions.py
```

## Slice 1: Rank-1 Core Contracts

### Action 1.1: Create Package Markers

Create files:

```text
tower/__init__.py
tower/reward/__init__.py
```

Content:

```python
"""Tower counterpoint package."""
```

Tests:

No dedicated tests required beyond import tests in later files.

### Action 1.2: Add Tuple-Based State/Action Helpers

Create:

```text
tower/state_action.py
```

Add:

```python
from typing import TypeAlias

MidiPitch: TypeAlias = int
TowerState: TypeAlias = tuple[int, ...]
TowerAction: TypeAlias = tuple[int, ...]

def rank_of_state(state: TowerState) -> int: ...
def validate_rank(rank: int) -> None: ...
def validate_state(state: TowerState, *, rank: int | None = None) -> None: ...
def validate_action(action: TowerAction, *, rank: int) -> None: ...
def apply_action(state: TowerState, action: TowerAction) -> TowerState: ...
```

Behavior:

| Function | Required behavior |
| --- | --- |
| `rank_of_state` | returns `len(state)` and rejects empty state |
| `validate_rank` | requires rank \(\ge 1\) |
| `validate_state` | requires nonempty tuple of ints, optional length equals rank, MIDI values in `[0, 127]`, strictly increasing if rank > 1 |
| `validate_action` | requires tuple of ints with length equal rank |
| `apply_action` | coordinatewise addition, validates action length, returns tuple |

Tests:

Create:

```text
tests/tower/test_state_action.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| rank of `(60,)` | returns `1` |
| rank of `(60, 64)` | returns `2` |
| empty state rejected | raises `ValueError` |
| non-int state entry rejected | raises `TypeError` or `ValueError` |
| MIDI out of range rejected | raises `ValueError` |
| non-increasing rank-2 state rejected | raises `ValueError` |
| valid rank-1 action | accepted |
| wrong action length | raises `ValueError` |
| apply `(60,) + (2,)` | returns `(62,)` |
| apply `(60, 67) + (1, -1)` | returns `(61, 66)` |

Run:

```bash
uv run pytest tests/tower/test_state_action.py
```

### Action 1.3: Add Tower Window

Create:

```text
tower/window.py
```

Add:

```python
from dataclasses import dataclass

PAD_BAR_POSITION = -1

@dataclass(frozen=True)
class TowerWindow:
    states: tuple[TowerState, ...]
    bar_positions: tuple[int, ...]
    valid_mask: tuple[bool, ...]

def pad_state(*, rank: int) -> TowerState: ...
def bar_position(*, step_index: int, measure_size: int) -> int: ...
def is_downbeat(*, step_index: int) -> bool: ...
def is_ending_beat(*, step_index: int, measure_size: int) -> bool: ...
def build_window(
    *,
    history: tuple[TowerState, ...],
    step_index: int,
    measure_size: int,
    context_measures: int,
) -> TowerWindow: ...
```

Behavior:

| Function | Required behavior |
| --- | --- |
| `pad_state` | returns `(0,) * rank` |
| `bar_position` | returns `step_index % measure_size` |
| `is_downbeat` | true iff `step_index % 2 == 0` |
| `is_ending_beat` | true iff bar position is `measure_size - 1` |
| `build_window` | fixed-length left-padded window of `context_measures * measure_size` |

Validation:

| Case | Required behavior |
| --- | --- |
| empty history | reject |
| nonpositive measure size | reject |
| nonpositive context measures | reject |
| mixed-rank history | reject |

Tests:

Create:

```text
tests/tower/test_window.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| pad rank 1 | `(0,)` |
| pad rank 2 | `(0, 0)` |
| bar position | `bar_position(step_index=5, measure_size=4) == 1` |
| downbeat | even steps true, odd steps false |
| ending beat | step 3 in 4/4 true |
| left padding | early history pads with PAD states |
| valid mask | padding false, real history true |
| bar positions | padding uses `-1`; real positions match step indices |
| long history | truncates to recent fixed-length suffix |
| mixed-rank history | raises |

Run:

```bash
uv run pytest tests/tower/test_window.py
```

### Action 1.4: Add Reward Result Shell

Create:

```text
tower/reward/result.py
```

Add:

```python
from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True)
class TowerRewardResult:
    reward: float
    hard_violation: bool = False
    is_terminal_success: bool = False
    diagnostics: Mapping[str, object] = field(default_factory=dict)
```

Tests:

Create or extend:

```text
tests/tower/reward/test_result.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| default flags | hard violation false, terminal success false |
| diagnostics accepted | mapping preserved |

Run:

```bash
uv run pytest tests/tower/reward/test_result.py
```

### Action 1.5: Add Reward Context Shell

Create:

```text
tower/reward/context.py
```

Add:

```python
from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True)
class NewFacts:
    new_voice_index: int | None = None
    new_action: int | None = None
    new_vertical_facts: tuple[int, ...] = ()
    full_sonority_used: bool = False

@dataclass(frozen=True)
class TowerRewardContext:
    rank: int
    step_index: int
    source: TowerState
    target: TowerState
    action: TowerAction
    window: TowerWindow
    measure_size: int | None = None
    max_steps: int | None = None
    max_step_size: int | None = None
    key_pitch_class: int | None = None
    target_root_octave: int | None = None
    is_final_step: bool = False
    new_facts: NewFacts = field(default_factory=NewFacts)
    diagnostics: Mapping[str, object] = field(default_factory=dict)
```

Behavior:

| Requirement | Detail |
| --- | --- |
| rank matches source/action/target/window | validate in `__post_init__` |
| window rank matches context rank | all valid states in window have same rank |
| no parent context required | projection will compute parent on demand later |

Tests:

Create:

```text
tests/tower/reward/test_context.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| context accepts rank-1 fields | constructed successfully |
| mismatched source rank | raises |
| mismatched action rank | raises |
| mismatched target rank | raises |
| new facts defaults | empty/default fields |
| metadata fields accepted | tonic/meter/goal/final-step fields preserved |

Run:

```bash
uv run pytest tests/tower/reward/test_context.py
```

### Action 1.6: Add Rank-1 Success Skeleton

Create:

```text
tower/reward/success.py
```

Add minimal shape:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SuccessResult:
    success: bool
    diagnostics: Mapping[str, object]

def rank1_projected_cadence_success(context: TowerRewardContext) -> SuccessResult: ...
```

Slice 1 behavior:

Keep this intentionally minimal. It may return `False` with diagnostics unless a very simple root-motion detector is agreed during implementation.

Required diagnostic keys:

| Key | Meaning |
| --- | --- |
| `rank` | active rank |
| `kind` | `"rank1_projected_cadence_success"` |
| `implemented` | false if skeleton only |

Tests:

Create:

```text
tests/tower/reward/test_success.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| success result shape | returns `SuccessResult` |
| diagnostics include rank/kind | true |
| non-rank-1 context rejected | raises or returns failure diagnostic |

Run:

```bash
uv run pytest tests/tower/reward/test_success.py
```

### Slice 1 Verification

Run:

```bash
uv run pytest tests/tower/test_state_action.py tests/tower/test_window.py tests/tower/reward/test_result.py tests/tower/reward/test_context.py tests/tower/reward/test_success.py
```

Stop if:

| Stop condition |
| --- |
| any tower module imports `rl_counterpoint` |
| state/action implementation introduces canonical parent fields |
| graph/projection/rollout behavior is added beyond Slice 1 |

## Slice 2: Rank-2 Projection And Action Assembly

### Action 2.1: Add Projection Module

Create:

```text
tower/graph/__init__.py
tower/graph/projection.py
```

Add:

```python
def project_tuple(values: tuple[int, ...]) -> tuple[int, ...]: ...
def project_state(state: TowerState) -> TowerState: ...
def project_action(action: TowerAction) -> TowerAction: ...
def project_window(window: TowerWindow) -> TowerWindow: ...
```

Behavior:

| Rank | Projection |
| --- | --- |
| 1 | reject; no lower rank |
| 2 | `(x0, x1) -> (x0,)` |
| 3 | `(x0, x1, x2) -> (x0, x2)` |
| 4 | `(x0, x1, x2, x3) -> (x0, x1, x3)` |
| \(k\ge3\) | remove second-from-top coordinate |

Tests:

Create:

```text
tests/tower/graph/test_projection.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| project rank 1 rejected | raises |
| project rank 2 state | `(60, 67) -> (60,)` |
| project rank 3 state | `(60, 64, 67) -> (60, 67)` |
| project rank 4 state | `(60, 64, 67, 72) -> (60, 64, 72)` |
| action projection mirrors state projection | same coordinate rule |
| window projection projects states | bar positions and valid mask unchanged |

Run:

```bash
uv run pytest tests/tower/graph/test_projection.py
```

### Action 2.2: Add Action Assembly

Create:

```text
tower/action/__init__.py
tower/action/assembly.py
```

Add:

```python
def new_voice_index(*, rank: int) -> int: ...
def assemble_action(
    *,
    rank: int,
    parent_action: TowerAction | None,
    new_action: int,
) -> TowerAction: ...
def validate_action_lift(
    *,
    action: TowerAction,
    parent_action: TowerAction,
) -> None: ...
```

Behavior:

| Rank | New voice index |
| --- | --- |
| 1 | `0` |
| 2 | `1` |
| \(k\ge3\) | `k - 2` |

Assembly:

| Rank | Behavior |
| --- | --- |
| 1 | parent must be `None`; action is `(new_action,)` |
| 2 | parent `(d0,)`, new `d1`; action `(d0, d1)` |
| 3 | parent `(d0, d2)`, new `d1`; action `(d0, d1, d2)` |
| 4 | parent `(d0, d1, d3)`, new `d2`; action `(d0, d1, d2, d3)` |

Tests:

Create:

```text
tests/tower/action/test_assembly.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| rank-1 assembly | `new=2 -> (2,)` |
| rank-2 assembly | parent `(1,)`, new `-1 -> (1, -1)` |
| rank-3 assembly | parent `(1, 3)`, new `2 -> (1, 2, 3)` |
| rank-4 assembly | parent `(1, 2, 4)`, new `3 -> (1, 2, 3, 4)` |
| assembled action projects to parent | `project_action(action) == parent` |
| invalid parent length rejected | raises |

Run:

```bash
uv run pytest tests/tower/action/test_assembly.py
```

### Action 2.3: Add Minimal Graph Spec

Create:

```text
tower/graph/spec.py
```

Add minimal dataclass:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TowerGraphSpec:
    rank: int
    pitch_min: int = 0
    pitch_max: int = 127
    max_step_size: int = 4
```

Behavior:

Validate:

| Field | Requirement |
| --- | --- |
| rank | \(\ge 1\) |
| pitch range | `0 <= pitch_min <= pitch_max <= 127` |
| max step size | \(\ge 1\) |

Do not yet implement full node/edge pruning parameters in Slice 2 unless needed for tests.

Tests:

Create:

```text
tests/tower/graph/test_spec.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| default spec rank 1 | accepted |
| invalid rank | raises |
| invalid pitch range | raises |
| invalid max step | raises |

Run:

```bash
uv run pytest tests/tower/graph/test_spec.py
```

### Action 2.4: Add Minimal Legality

Create:

```text
tower/graph/legality.py
```

Add:

```python
def is_valid_state(state: TowerState, spec: TowerGraphSpec) -> bool: ...
def is_valid_transition(
    source: TowerState,
    action: TowerAction,
    spec: TowerGraphSpec,
) -> bool: ...
```

Slice 2 behavior:

| Function | Minimal requirement |
| --- | --- |
| `is_valid_state` | rank matches spec, MIDI range, strict increase for rank > 1 |
| `is_valid_transition` | source valid, action rank matches, target via `apply_action` valid, no self-loop |

Do not yet implement full pruning or projection-compatible edge recursion in Slice 2 unless needed for commuting tests.

Tests:

Create:

```text
tests/tower/graph/test_legality.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| valid rank-1 state | true |
| valid rank-2 increasing state | true |
| non-increasing rank-2 state | false |
| transition to out-of-range target | false |
| self-loop action `(0,)` | false |
| valid action to valid target | true |

Run:

```bash
uv run pytest tests/tower/graph/test_legality.py
```

### Action 2.5: Add Projection/Update Commuting Test

Extend:

```text
tests/tower/graph/test_projection.py
```

Add test:

```python
source = (60, 67)
action = (2, -1)
target = apply_action(source, action)

assert project_state(target) == apply_action(
    project_state(source),
    project_action(action),
)
```

Also test rank 3:

```python
source = (60, 64, 67)
action = (1, -1, 2)
```

Run:

```bash
uv run pytest tests/tower/graph/test_projection.py
```

### Slice 2 Verification

Run:

```bash
uv run pytest tests/tower/test_state_action.py tests/tower/test_window.py tests/tower/action/test_assembly.py tests/tower/graph/test_projection.py tests/tower/graph/test_spec.py tests/tower/graph/test_legality.py
```

Stop if:

| Stop condition |
| --- |
| projection requires parent fields |
| assembly fails to project back to parent action |
| graph code imports `rl_counterpoint` |
| tensor dependencies are introduced in core graph modules |

## Slice 3: Rank-2 Lift-Fiber Masks

### Action 3.1: Add Candidate Action Generation

Create:

```text
tower/graph/actions.py
```

Add:

```python
def action_space(*, rank: int, max_step_size: int) -> tuple[TowerAction, ...]: ...
```

Behavior:

| Requirement | Detail |
| --- | --- |
| candidate deltas | Cartesian product over `[-max_step_size, max_step_size]` |
| zero action | exclude all-zero action |
| validation | rank and max step positive |

Tests:

Create:

```text
tests/tower/graph/test_actions.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| rank-1 max-step 1 | actions `(-1,)`, `(1,)` |
| rank-2 max-step 1 | 8 actions |
| all-zero excluded | true |
| invalid rank/max step rejected | raises |

Run:

```bash
uv run pytest tests/tower/graph/test_actions.py
```

### Action 3.2: Add Lift Fiber

Extend:

```text
tower/graph/actions.py
```

Add:

```python
def lift_fiber_actions(
    *,
    state: TowerState,
    parent_action: TowerAction,
    spec: TowerGraphSpec,
) -> tuple[TowerAction, ...]: ...
```

Behavior:

| Requirement | Detail |
| --- | --- |
| rank | state rank is active rank \(k\) |
| parent rank | parent action rank is \(k-1\) |
| projection filter | keep only actions with `project_action(action) == parent_action` |
| legality filter | keep only actions where `is_valid_transition(state, action, spec)` |

Tests:

Extend:

```text
tests/tower/graph/test_actions.py
```

Test cases:

| Test | Assertion |
| --- | --- |
| rank-2 fiber over `(1,)` | every returned action has first coordinate `1` |
| rank-2 fiber excludes other parent coords | no action projects to different parent |
| legality filter applies | target states are valid |
| parent rank mismatch rejected | raises |

Run:

```bash
uv run pytest tests/tower/graph/test_actions.py
```

### Action 3.3: Add Active Choices View

Extend:

```text
tower/graph/actions.py
```

Add:

```python
def active_lift_choices(
    *,
    state: TowerState,
    parent_action: TowerAction,
    spec: TowerGraphSpec,
) -> tuple[int, ...]: ...
```

Behavior:

Return the possible active new-coordinate values corresponding to the full lift-fiber actions.

For rank 2:

\[
\Delta s^2=(\Delta\lambda_0,\Delta\lambda_1)
\]

so active choices are the \(\Delta\lambda_1\) values.

Tests:

| Test | Assertion |
| --- | --- |
| active choices align with fiber | every choice reconstructs a fiber action via assembly |
| choices are unique | no duplicates |
| empty fiber gives empty choices | true |

Run:

```bash
uv run pytest tests/tower/graph/test_actions.py
```

### Action 3.4: Add Empty Fiber Diagnostic Helper

Extend:

```text
tower/graph/actions.py
```

Add:

```python
def has_empty_lift_fiber(
    *,
    state: TowerState,
    parent_action: TowerAction,
    spec: TowerGraphSpec,
) -> bool: ...
```

Behavior:

Returns true iff `lift_fiber_actions(...)` is empty.

Tests:

| Test | Assertion |
| --- | --- |
| nonempty ordinary fiber | false |
| constructed impossible boundary case | true |

If a natural impossible boundary case is awkward with minimal legality, create a tight `TowerGraphSpec` pitch range that forces all lifted targets out of range.

Run:

```bash
uv run pytest tests/tower/graph/test_actions.py
```

### Slice 3 Verification

Run:

```bash
uv run pytest tests/tower
```

Stop if:

| Stop condition |
| --- |
| lift fiber returns actions not projecting to parent |
| active choices cannot reconstruct full actions |
| empty fiber is silently ignored |
| graph action code imports old `rl_counterpoint` |

## Slice 4 Outline: Rank-2 Rollout Without Neural Policy

Do not implement until Slices 1-3 pass.

Planned files:

| File | Purpose |
| --- | --- |
| `tower/train/trajectory.py` | Option C step records |
| `tower/policy/samplers.py` | scripted/trivial parent and active samplers |
| `tower/train/rollout.py` | parent-first rollout choreography |

Key tests:

| Test | Proves |
| --- | --- |
| scripted rank-2 rollout advances | choreography works |
| invalid extension records diagnostic | Stage 11 semantics |
| empty fiber records `empty_lift_fiber` | exceptional case visible |
| active logprob separate from parent logprob | gradient ownership |

Decision needed before implementation:

| Decision | Needed |
| --- | --- |
| hard violation terminates immediately? | before rollout supports hard violations |

## Slice 5 Outline: Reward Context And Success Predicates

Do not implement full TC21M reward grammar here.

Planned files:

| File | Purpose |
| --- | --- |
| `tower/reward/success.py` | real rank-1/rank-2 success predicates |
| `tower/reward/terms.py` | basic term protocol/composite |

Key tests:

| Test | Proves |
| --- | --- |
| rank-1 projected cadence success | base terminal predicate |
| rank-2 lifted success requires parent | projection-based success |
| rank-2 lifted success requires outer-third condition | new terminal condition |

Deferred:

| Deferred |
| --- |
| exact full cadence-template vocabulary |
| full chord-template reward suite |
| six-four logic |

## Slice 6 Outline: Artifact/Checkpoint Skeleton

Planned files:

| File | Purpose |
| --- | --- |
| `tower/train/config.py` | config dataclasses |
| `tower/train/checkpoint.py` | lineage dirs, config, metrics, manifest, latest checkpoint path |

Key tests:

| Test | Proves |
| --- | --- |
| lineage paths deterministic | artifact contract |
| config round trip | config persistence |
| metrics JSONL append | old behavior carried over |
| manifest parent lookup | rank \(k+1\) can find rank \(k\) |

## Slice 7 Outline: Rank-1 Learning Loop

Planned files:

| File | Purpose |
| --- | --- |
| `tower/policy/base.py` | policy protocol |
| `tower/train/losses.py` | policy-gradient loss |
| `tower/train/protocol.py` | rank-1 training loop |

Key tests:

| Test | Proves |
| --- | --- |
| one rank-1 training episode runs | minimal training path |
| optimizer step changes active params | learning connected |
| checkpoint latest writes | artifact integration |

## Slice 8 Outline: Rank-2 Learning Over Frozen Rank 1

Planned files:

| File | Purpose |
| --- | --- |
| `tower/train/protocol.py` | rank-2 training lifecycle |
| `tower/train/checkpoint.py` | parent loading and lineage |
| `tower/policy/samplers.py` | parent top-\(m\) sampler |
| `tower/train/rollout.py` | frozen parent and active child rollout |

Key tests:

| Test | Proves |
| --- | --- |
| rank-2 training finds parent checkpoint | lineage lookup |
| parent params remain unchanged | freeze rule |
| child params update | active tier trains |
| rank-2 checkpoint records parent | dependency tracking |

## Stage 17 Completion Checklist

Stage 17 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| build plan starts with tuple-based state/action contracts | yes |
| tensor use is deferred or internal only | yes |
| first fully specified slices are Slice 1-3 | yes |
| rollout/training slices are outlined but not over-specified | yes |
| no runtime import from `rl_counterpoint` is allowed | yes |
| first verification command is focused pytest per slice | yes |
| full `tests/tower` run verifies Slice 3 | yes |

Once accepted, the project is ready for implementation of Slice 1.
