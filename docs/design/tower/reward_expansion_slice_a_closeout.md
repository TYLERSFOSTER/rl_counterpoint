# Reward Expansion Slice A Closeout

This document is the Post-Slice-8 Phase 7 / Stage 7.12 / Action 7.12.1
deliverable.

The purpose is to close out Reward Expansion Slice A by recording what was
implemented, what evidence exists, what remains limited, and the recommended
next build direction.

This is a closeout document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.12: Reward Expansion Slice A Closeout |
| Action | Action 7.12.1: Write Reward Expansion Slice A Closeout |

## Source Authority

This closeout summarizes:

| Source | Role |
| --- | --- |
| `docs/design/tower/reward_expansion_slice_a_contract.md` | accepted Slice A contract |
| `docs/design/tower/post_slice_8_reward_evidence.md` | first artifact-backed reward run |
| `docs/design/tower/reward_diagnostics_evidence.md` | diagnostics artifact evidence |
| `docs/design/tower/slice_a_reward_probe_contract.md` | deterministic probe contract |
| `docs/design/tower/slice_a_reward_probe_evidence.md` | deterministic probe evidence |

## Slice A Goal

Slice A was defined as:

```text
Reward Expansion Slice A:
Rank-1 Cadence And Melodic Shape Reward
```

Goal:

Add the first real rank-1 musical reward signal usable by tower training without
building broad harmonic-template machinery.

## Implemented Source Surface

Implemented reward files:

| File | Status |
| --- | --- |
| `tower/reward/melody.py` | added rank-1 melodic range and large-leap recovery terms |
| `tower/reward/factory.py` | added composed rank-1 Slice A reward factory |
| `tower/reward/probe.py` | added deterministic Slice A reward probe |

Implemented training/artifact files:

| File | Status |
| --- | --- |
| `tower/train/diagnostics.py` | added per-step reward diagnostics serialization |
| `tower/train/checkpoint.py` | added `reward_diagnostics.jsonl` paths/read-write helpers and manifest field |
| `tower/train/runner.py` | writes reward diagnostics for training and final inference episodes |

Implemented scripts:

| File | Status |
| --- | --- |
| `scripts/tower_train.py` | wired rank-1 training script to `build_rank1_reward_fn(...)` |
| `scripts/tower_reward_probe.py` | writes deterministic Slice A probe JSONL artifact |

Implemented tests:

| File | Status |
| --- | --- |
| `tests/tower/reward/test_melody.py` | focused melodic term tests |
| `tests/tower/reward/test_factory.py` | composed factory/config tests |
| `tests/tower/reward/test_probe.py` | deterministic probe row/tests |
| `tests/tower/reward/test_probe_script.py` | probe script direct-execution tests |
| `tests/tower/train/test_diagnostics.py` | diagnostics serialization tests |
| `tests/tower/train/test_checkpoint.py` | diagnostics artifact helper and manifest tests |
| `tests/tower/train/test_runner.py` | runner diagnostics and factory integration tests |
| `tests/tower/train/test_runner_script.py` | script reward config and diagnostics artifact tests |

## Implemented Reward Terms

### A1: Terminal Cadence Reward

Status:

Implemented through the existing predicate and adapter:

```text
rank1_projected_cadence_success
SuccessRewardTerm
```

Factory defaults:

| Field | Value |
| --- | --- |
| key_pitch_class | `0` |
| terminal_cadence_reward | `10.0` |
| cadence_failure_reward | `0.0` |

Evidence:

| Evidence | Result |
| --- | --- |
| factory tests | cadence success and failure diagnostics preserved |
| probe evidence | `terminal_cadence_success` row records `reason = success` and `reward = 10.0` |

### A2: Recent Melodic Range Penalty

Status:

Implemented as:

```text
RecentMelodicRangePenalty
```

Defaults:

| Field | Value |
| --- | --- |
| max_recent_range | `12` |
| range_penalty | `-1.0` |

Evidence:

| Evidence | Result |
| --- | --- |
| melody tests | fires above threshold, no-ops at threshold and with short history |
| probe evidence | `recent_range_penalty` row records `observed_range = 13`, `penalty_applied = true`, `reason = range_exceeded` |

### A3: Large Leap Recovery Reward/Penalty

Status:

Implemented as:

```text
LargeLeapRecoveryTerm
```

Defaults:

| Field | Value |
| --- | --- |
| large_leap_threshold | `6` |
| recovery_step_threshold | `3` |
| recovery_reward | `0.5` |
| failure_penalty | `-0.5` |

Evidence:

| Evidence | Result |
| --- | --- |
| melody tests | success, wrong direction, oversized recovery, no-trigger, short-history paths covered |
| probe evidence | success row records `reason = recovered`; failure rows record `reason = failed_recovery` |

### A4: Step-Leap Penalty

Status:

Deferred by contract.

Reason:

Terms A1-A3 were enough for the first narrow Slice A implementation. A4 remains
a possible later rank-1 melodic shaping addition.

## Factory Wiring

The rank-1 reward factory is:

```text
build_rank1_reward_fn(...)
```

It returns a callable reward function built from:

```text
CompositeRewardTerm(
  SuccessRewardTerm(rank1_projected_cadence_success),
  RecentMelodicRangePenalty,
  LargeLeapRecoveryTerm,
)
```

The factory injects the configured key into the frozen reward context with
`dataclasses.replace(...)`.

Reason:

`TowerRewardContext` is frozen, and the existing rollout path does not currently
own key configuration. The factory-level replacement avoids changing rollout
plumbing while keeping cadence evaluation configured.

## Script And Runner Wiring

The rank-1 training script now uses Slice A reward:

```text
scripts/tower_train.py -> build_rank1_reward_fn(...)
```

The script records reward config in `config.json`:

```text
reward_config.kind = rank1_slice_a
```

The runner now writes:

```text
reward_diagnostics.jsonl
```

for:

| Episode kind | Status |
| --- | --- |
| training episodes | written |
| final no-train inference | written |

## Evidence Artifacts

### First Reward Evidence

Document:

```text
docs/design/tower/post_slice_8_reward_evidence.md
```

Command:

```bash
uv run python scripts/tower_train.py \
  --rank 1 \
  --episodes 1 \
  --lineage-id reward-slice-a-tiny \
  --artifact-root artifacts/tower \
  --seed 123 \
  --max-steps 1 \
  --max-step-size 1
```

Result:

| Evidence | Result |
| --- | --- |
| script reward | `reward: rank1_slice_a` |
| final return | `0.0` |
| terminal success | `false` |
| config | reward config persisted |
| artifacts | config/metrics/checkpoint/manifest/MIDI written |

Interpretation:

The Slice A reward factory ran in the artifact-backed training path. The return
was `0.0`, as expected for a one-step non-cadential run with no range/leap
condition triggered.

### Reward Diagnostics Evidence

Document:

```text
docs/design/tower/reward_diagnostics_evidence.md
```

Command:

```bash
uv run python scripts/tower_train.py \
  --rank 1 \
  --episodes 1 \
  --lineage-id reward-diagnostics-tiny \
  --artifact-root artifacts/tower \
  --seed 123 \
  --max-steps 1 \
  --max-step-size 1
```

Result:

| Evidence | Result |
| --- | --- |
| diagnostics artifact | `reward_diagnostics.jsonl` written |
| manifest | includes `reward_diagnostics` path |
| rows | training row and final inference row |
| reward kind | `rank1_reward` |
| nested diagnostics | cadence/range/leap diagnostics preserved |

Interpretation:

The artifact path now preserves reward explanations, not only scalar return.
The one-step training run only exercises insufficient-history paths.

### Slice A Reward Probe Evidence

Document:

```text
docs/design/tower/slice_a_reward_probe_evidence.md
```

Command:

```bash
uv run python scripts/tower_reward_probe.py \
  --lineage-id slice-a-reward-probe \
  --artifact-root artifacts/tower
```

Result:

| Case | Reward | Key diagnostic |
| --- | --- | --- |
| `terminal_cadence_success` | `9.5` | cadence `reason = success` |
| `recent_range_penalty` | `-1.5` | range `reason = range_exceeded` |
| `large_leap_recovery_success` | `0.5` | leap `reason = recovered` |
| `large_leap_recovery_failure` | `-0.5` | leap `reason = failed_recovery` |

Interpretation:

The composed Slice A reward factory can produce and serialize meaningful
term-level reward diagnostics in deterministic contexts.

## Important Correction Captured

The Slice A reward probe implementation found one sign issue in the probe
contract sketch.

The original cadence probe idea used:

```text
history = ((67,), (60,))
action = (+1,)
```

But the previous interval is:

```text
60 - 67 = -7
```

Therefore `+1` is opposite-direction recovery and would score:

```text
10.5
```

The implemented probe uses:

```text
action = (-1,)
```

This preserves the intended composed interaction:

```text
cadence success +10.0
large leap recovery failure -0.5
total = 9.5
```

## Verification Summary

Latest focused and broad verification from Slice A work:

| Command | Result |
| --- | --- |
| `uv run pytest tests/tower/reward/test_melody.py tests/tower/test_import_boundaries.py` | passed during A2/A3 implementation |
| `uv run pytest tests/tower/reward/test_factory.py tests/tower/train/test_runner.py tests/tower/test_import_boundaries.py` | passed during factory implementation |
| `uv run pytest tests/tower/train/test_diagnostics.py tests/tower/train/test_checkpoint.py tests/tower/train/test_runner.py tests/tower/train/test_runner_script.py tests/tower/test_import_boundaries.py` | `93 passed` during diagnostics evidence |
| `uv run pytest tests/tower/reward/test_probe.py tests/tower/reward/test_probe_script.py tests/tower/reward tests/tower/test_import_boundaries.py` | `69 passed` during probe implementation |
| `uv run pytest tests/tower` | `396 passed` after probe implementation |
| `uv run pytest` | `551 passed` after probe implementation |

## What Slice A Now Proves

Slice A now proves:

| Claim | Evidence |
| --- | --- |
| rank-1 reward terms exist | melody/factory source and tests |
| reward factory is runner-usable | runner acceptance test and script evidence |
| script uses real reward | `reward: rank1_slice_a` and config evidence |
| per-step diagnostics persist | `reward_diagnostics.jsonl` evidence |
| cadence success can be diagnosed | probe row with `reason = success` |
| range penalty can be diagnosed | probe row with `range_exceeded` |
| leap recovery success can be diagnosed | probe row with `recovered` |
| leap recovery failure can be diagnosed | probe rows with `failed_recovery` |

## Remaining Limitations

Slice A does not yet prove:

| Limitation | Meaning |
| --- | --- |
| musical learning | no longer-horizon training evidence yet |
| positive cadence in training | cadence success has probe evidence, not training-run evidence |
| reward shaping improves trajectories | no comparative training run yet |
| rank-2 reward behavior | Slice B remains future work |
| graph-boundary rules | no pruning or hard TC21M constraints added |
| harmonic template handling | still deferred |
| A4 step-leap penalty | still deferred |
| checkpoint promotion by quality | acceptance remains episode-count based |

## Recommended Next Direction

Recommendation:

Run a small longer rank-1 Slice A training/evaluation evidence pass before
starting Slice B rank-2 reward planning.

Reason:

Slice A is now:

```text
implemented
factory-wired
script-wired
diagnostics-backed
probe-backed
```

The remaining uncertainty is not whether the terms work in isolation. The
remaining uncertainty is how the composed reward behaves over a slightly longer
actual rank-1 rollout.

## Proposed Next Phase.Stage.Action

Recommended next action:

```text
Post-Slice-8 Phase 8.Stage 8.1.Action 8.1.1:
Specify Longer Rank-1 Slice A Training Evidence Run
```

Purpose:

Define one small but nontrivial rank-1 training/evaluation run that uses Slice A
reward and diagnostics, likely with:

| Setting | Initial recommendation |
| --- | --- |
| rank | `1` |
| episodes | small, such as `5` or `10` |
| max_steps | enough to permit history-sensitive terms, such as `8` or `16` |
| max_step_size | likely `2` or `4` |
| artifact lineage | new explicit evidence lineage |
| evidence | metrics, diagnostics, MIDI, and short evidence report |

This should still be treated as evidence, not a claim of musical quality.
