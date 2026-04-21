# Reward Expansion Slice A Contract

This document is the Post-Slice-8 Phase 7 / Stage 7.2 / Action 7.2.1
deliverable.

The purpose is to turn the accepted reward expansion plan into an explicit,
implementation-ready contract for the first narrow reward slice.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.2: Reward Slice A Contract |
| Action | Action 7.2.1: Specify Reward Expansion Slice A Contract |

## Source Authority

This contract is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/post_slice_8_reward_expansion_plan.md` | accepted Slice A plan |
| `assets/rules/tc21m_rules.md` | TC21M-derived melodic/cadence rules |
| `docs/design/tower/post_slice_8_training_evidence.md` | evidence that runner path exists |
| `tower/reward/context.py` | reward context contract |
| `tower/reward/result.py` | reward result contract |
| `tower/reward/success.py` | existing cadence predicates |
| `tower/reward/terms.py` | existing composable reward terms |

## Slice Name

```text
Reward Expansion Slice A:
Rank-1 Cadence And Melodic Shape Reward
```

## Goal

Add the first real rank-1 musical reward signal that can be used by tower
training without implementing broad harmonic-template machinery.

The slice should produce:

```text
rank-1 reward context
-> cadence reward term
-> recent melodic range term
-> large leap recovery term
-> optional step-leap penalty deferred by default
-> reward factory usable by runner/tests
```

## Non-Goals

Do not include:

| Excluded work | Reason |
| --- | --- |
| rank-2 vertical interval rewards | Slice B after rank-1 reward works |
| harmonic progression templates | requires chord-template detector |
| non-harmonic tone taxonomy | requires chord-tone inference |
| graph pruning | requires separate graph-boundary action |
| suspensions | deferred to later style update such as `beta.1` |
| six-four chord logic | explicitly deferred |
| checkpoint promotion by musical quality | acceptance remains episode-count based |

## Files

Expected source files:

| File | Work |
| --- | --- |
| `tower/reward/melody.py` | add rank-1 melodic shape reward terms |
| `tower/reward/factory.py` | build reward function from rank/runner config |

Expected test files:

| File | Work |
| --- | --- |
| `tests/tower/reward/test_melody.py` | focused melodic reward term tests |
| `tests/tower/reward/test_factory.py` | reward factory/config tests |
| `tests/tower/train/test_runner.py` | optional small runner integration test |

No existing reward files should need destructive refactors. Existing
`success.py` and `terms.py` should be reused.

## Term A1: Terminal Cadence Reward Factory

Use existing predicate:

```text
rank1_projected_cadence_success
```

Use existing adapter:

```text
SuccessRewardTerm
```

Default config:

| Field | Default |
| --- | --- |
| key_pitch_class | `0` |
| terminal_cadence_reward | `10.0` |
| cadence_failure_reward | `0.0` |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| rank | rank 1 only |
| final step | only succeeds when `context.is_final_step` is true |
| meter | requires `context.measure_size` |
| key | requires `context.key_pitch_class` |
| V-I | previous valid root pitch class is dominant and final is tonic |
| metrical arrival | final bar position is `measure_size - 1` |
| diagnostics | preserve predicate reason |

Implementation stance:

Do not rewrite cadence detection in Slice A. Reuse the tested predicate.

## Term A2: Recent Melodic Range Penalty

TC21M source:

Avoid jumps that give the recent melody a range of more than an octave.

Default config:

| Field | Default |
| --- | --- |
| max_recent_range | `12` |
| range_penalty | `-1.0` |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| rank | rank 1 only |
| input states | valid states in `context.window` |
| pitch extraction | use the only pitch coordinate from each rank-1 state |
| empty/short windows | no penalty if fewer than two valid states |
| penalty condition | `max(valid_pitches) - min(valid_pitches) > max_recent_range` |
| no-penalty condition | range is less than or equal to threshold |
| diagnostics | include observed range, threshold, valid pitch count |

Suggested object:

```text
RecentMelodicRangePenalty
```

Suggested call shape:

```python
RecentMelodicRangePenalty(
    max_recent_range=12,
    penalty=-1.0,
)(context)
```

## Term A3: Large Leap Recovery Reward/Penalty

TC21M source:

Follow large leaps with a small step in the opposite direction.

Default config:

| Field | Default |
| --- | --- |
| large_leap_threshold | `6` |
| recovery_step_threshold | `3` |
| recovery_reward | `0.5` |
| failure_penalty | `-0.5` |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| rank | rank 1 only |
| previous interval source | final two valid states in `context.window` |
| current interval source | `context.action[0]` |
| trigger condition | absolute previous interval is at least `large_leap_threshold` |
| no-trigger result | reward `0.0` with diagnostic reason |
| success condition | current action is nonzero, opposite direction, and absolute value <= `recovery_step_threshold` |
| failure condition | triggered but success condition fails |
| diagnostics | include previous interval, current action, direction check, threshold values |

Suggested object:

```text
LargeLeapRecoveryTerm
```

Suggested call shape:

```python
LargeLeapRecoveryTerm(
    large_leap_threshold=6,
    recovery_step_threshold=3,
    recovery_reward=0.5,
    failure_penalty=-0.5,
)(context)
```

## Term A4: Step-Leap Penalty

Status:

Deferred by default.

Reason:

Terms A1-A3 are enough for the first implementation slice. Step-leap penalty can
be added if the owner explicitly asks to include it before Slice A starts.

Default if later included:

| Field | Default |
| --- | --- |
| previous_step_threshold | `3` |
| current_leap_threshold | `6` |
| penalty | mild negative reward, suggested `-0.25` |

## Reward Factory

Purpose:

Create a small factory that builds a rank-1 reward function usable by runner
tests and later script/config wiring.

Suggested function:

```text
build_rank1_reward_fn(...)
```

Suggested file:

```text
tower/reward/factory.py
```

Required inputs:

| Input | Default |
| --- | --- |
| key_pitch_class | `0` |
| terminal_cadence_reward | `10.0` |
| cadence_failure_reward | `0.0` |
| max_recent_range | `12` |
| range_penalty | `-1.0` |
| large_leap_threshold | `6` |
| recovery_step_threshold | `3` |
| recovery_reward | `0.5` |
| failure_penalty | `-0.5` |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| output | callable accepting `TowerRewardContext` and returning `TowerRewardResult` |
| composition | use `CompositeRewardTerm` |
| cadence | use `SuccessRewardTerm(rank1_projected_cadence_success, ...)` |
| context key | inject or require `key_pitch_class=0` for first tests/runs |
| diagnostics | include term-level diagnostics |

Open implementation detail:

`TowerRewardContext` is frozen, so the factory cannot mutate context in place.
It should either:

| Option | Status |
| --- | --- |
| require caller/runner to provide `key_pitch_class` | acceptable |
| wrap context with `dataclasses.replace(context, key_pitch_class=...)` before term evaluation | acceptable |

Recommended first choice:

Use `dataclasses.replace` inside the factory so tests can call the reward
function without modifying runner plumbing immediately.

## Runner Integration

Slice A should not require runner refactor.

Allowed integration:

| Integration | Status |
| --- | --- |
| focused unit tests of reward factory | required |
| one tiny runner test using reward factory | optional |
| script argument expansion for reward config | defer |
| training evidence run with new reward | later evidence action |

If a runner test is added, it should be tiny and use existing `run_rank1_training`
with `episode_count=1`, `max_steps=1`, and a deterministic reward factory.

## Tests

Focused tests for `tests/tower/reward/test_melody.py`:

| Test | Proves |
| --- | --- |
| range penalty fires over threshold | A2 penalty path |
| range penalty does not fire at threshold | A2 boundary path |
| range term ignores short valid history | A2 no-op path |
| range term rejects non-rank-1 context | rank ownership |
| leap recovery rewards opposite small step | A3 success path |
| leap recovery penalizes wrong direction | A3 failure path |
| leap recovery no-ops without large previous leap | A3 no-trigger path |
| leap recovery rejects non-rank-1 context | rank ownership |

Focused tests for `tests/tower/reward/test_factory.py`:

| Test | Proves |
| --- | --- |
| factory returns `TowerRewardResult` | output contract |
| factory combines cadence and melodic terms | composition |
| factory injects configured key if chosen | key handling |
| factory preserves cadence diagnostics | terminal evidence |
| factory validates config values | bad config rejected |

Optional runner test:

| Test | Proves |
| --- | --- |
| rank-1 runner can train with factory reward | runner path accepts real reward function |

Focused verification command:

```bash
uv run pytest tests/tower/reward tests/tower/train/test_runner.py tests/tower/test_import_boundaries.py
```

## Stop Conditions

Pause and resynchronize if:

| Stop condition | Why |
| --- | --- |
| reward implementation requires graph pruning | separate owner approval needed |
| rank-1 reward terms need rank-2 interval facts | scope leak |
| harmonic template detector becomes necessary | Slice A too broad |
| runner must be refactored to pass reward context keys | contract should be revisited |
| owner wants A4 included immediately | contract should be amended before code |

## Proposed Implementation Actions

Recommended implementation sequence:

```text
Post-Slice-8 Phase 7.Stage 7.3.Action 7.3.1:
Implement Rank-1 Melody Reward Terms
```

Expected files:

```text
tower/reward/melody.py
tests/tower/reward/test_melody.py
```

Then:

```text
Post-Slice-8 Phase 7.Stage 7.4.Action 7.4.1:
Implement Rank-1 Reward Factory
```

Expected files:

```text
tower/reward/factory.py
tests/tower/reward/test_factory.py
tests/tower/train/test_runner.py
```

Then:

```text
Post-Slice-8 Phase 7.Stage 7.5.Action 7.5.1:
Run Tiny Rank-1 Reward Evidence
```

Expected output:

```text
new tiny run artifact or evidence report update
```
