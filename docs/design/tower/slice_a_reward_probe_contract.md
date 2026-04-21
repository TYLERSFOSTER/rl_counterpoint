# Slice A Reward Term Evidence Probe Contract

This document is the Post-Slice-8 Phase 7 / Stage 7.9 / Action 7.9.1
deliverable.

The purpose is to specify a tiny deterministic probe that produces
artifact-style evidence for the meaningful Reward Expansion Slice A term paths.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.9: Slice A Reward Term Evidence Probe Contract |
| Action | Action 7.9.1: Specify Slice A Reward Term Evidence Probe |

## Source Authority

This contract is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/reward_expansion_slice_a_contract.md` | Slice A reward term contract |
| `docs/design/tower/reward_diagnostics_evidence.md` | evidence gap: diagnostics artifact only shows insufficient-history paths |
| `tower/reward/factory.py` | rank-1 reward factory |
| `tower/reward/melody.py` | melodic range and leap-recovery terms |
| `tower/train/diagnostics.py` | JSON-compatible diagnostics row serializer |

## Problem

The tiny reward diagnostics evidence run proves:

```text
reward_diagnostics.jsonl exists
training and final_inference rows are written
rank1_reward diagnostics survive JSONL serialization
```

However, because that run has only one step and a one-state window, it only
shows these term reasons:

```text
cadence: insufficient_valid_history
recent_melodic_range: insufficient_valid_history
large_leap_recovery: insufficient_valid_history
```

It does not yet prove, in saved artifact form, that the meaningful Slice A paths
are inspectable:

```text
cadence success
recent range penalty
large leap recovery success
large leap recovery failure
```

## Design Decision

Build a runner-free deterministic probe.

Do not involve:

| Excluded dependency | Reason |
| --- | --- |
| policy/model sampling | would make evidence dependent on stochastic action selection |
| optimizer/training loop | not needed for reward-term evidence |
| MIDI rendering | reward diagnostics only |
| graph pruning | out of Slice A scope |

The probe should:

| Behavior | Requirement |
| --- | --- |
| construct hand-built rank-1 reward contexts | force specific reward paths |
| use `build_rank1_reward_fn(...)` | prove the composed factory, not isolated terms only |
| serialize artifact-style rows | use the same row shape as reward diagnostics evidence |
| write JSONL under artifacts | keep evidence inspectable outside pytest |
| have focused tests | prove cases and row shape deterministically |

## Artifact

Recommended artifact filename:

```text
reward_term_probe.jsonl
```

Recommended location:

```text
artifacts/tower/<lineage_id>/rank_1/reward_term_probe.jsonl
```

Recommended default lineage id:

```text
slice-a-reward-probe
```

This artifact is a probe artifact, not a training artifact. It does not need to
appear in the lineage manifest for the first implementation. A later artifact
indexing pass can decide whether non-training probe artifacts belong in the
manifest.

## Probe Cases

The first probe should include four cases.

### Case 1: Terminal Cadence Success

Purpose:

Prove the cadence reward path can succeed through the factory.

Context:

| Field | Value |
| --- | --- |
| rank | `1` |
| history | `((67,), (60,))` |
| step_index | `3` |
| measure_size | `4` |
| context_measures | `1` or larger |
| source | `(60,)` |
| action | `(1,)` |
| target | `(61,)` |
| key_pitch_class | injected by factory as `0` |
| is_final_step | `true` |

Expected diagnostics:

| Term | Expected |
| --- | --- |
| cadence | `reason = success`, `reward = 10.0`, `is_terminal_success = true` |
| recent range | no penalty because range is `7 <= 12` |
| large leap recovery | failure by default if previous interval is large and current action is not opposite |

Expected total reward with defaults:

```text
9.5
```

Explanation:

```text
cadence success +10.0
large leap recovery failure -0.5
range penalty 0.0
```

This interaction is acceptable evidence because it proves the composed factory
terms are all active. The probe row should record the term-level components so
the interaction is transparent.

### Case 2: Recent Range Penalty

Purpose:

Prove the recent melodic range penalty fires above one octave.

Context:

| Field | Value |
| --- | --- |
| rank | `1` |
| history | `((60,), (73,))` |
| step_index | `1` |
| measure_size | `4` |
| context_measures | `1` or larger |
| source | `(73,)` |
| action | `(1,)` |
| target | `(74,)` |
| is_final_step | `false` |

Expected diagnostics:

| Term | Expected |
| --- | --- |
| cadence | `reason = not_final_step` |
| recent range | `observed_range = 13`, `penalty_applied = true`, `reward = -1.0` |
| large leap recovery | triggered by prior interval but fails because current action is same direction |

Expected total reward with defaults:

```text
-1.5
```

Explanation:

```text
range penalty -1.0
large leap recovery failure -0.5
cadence 0.0
```

### Case 3: Large Leap Recovery Success

Purpose:

Prove the large leap recovery reward fires when the current action is a small
opposite-direction step.

Context:

| Field | Value |
| --- | --- |
| rank | `1` |
| history | `((60,), (67,))` |
| step_index | `1` |
| measure_size | `4` |
| context_measures | `1` or larger |
| source | `(67,)` |
| action | `(-2,)` |
| target | `(65,)` |
| is_final_step | `false` |

Expected diagnostics:

| Term | Expected |
| --- | --- |
| cadence | `reason = not_final_step` |
| recent range | no penalty because range is `7 <= 12` |
| large leap recovery | `previous_interval = 7`, `current_action = -2`, `success = true`, `reward = 0.5` |

Expected total reward with defaults:

```text
0.5
```

### Case 4: Large Leap Recovery Failure

Purpose:

Prove the large leap recovery penalty fires when recovery is required but the
current action fails the recovery condition.

Context:

| Field | Value |
| --- | --- |
| rank | `1` |
| history | `((60,), (67,))` |
| step_index | `1` |
| measure_size | `4` |
| context_measures | `1` or larger |
| source | `(67,)` |
| action | `(2,)` |
| target | `(69,)` |
| is_final_step | `false` |

Expected diagnostics:

| Term | Expected |
| --- | --- |
| cadence | `reason = not_final_step` |
| recent range | no penalty because range is `7 <= 12` |
| large leap recovery | `previous_interval = 7`, `current_action = 2`, `opposite_direction = false`, `success = false`, `reward = -0.5` |

Expected total reward with defaults:

```text
-0.5
```

## Probe Row Shape

The probe artifact should use a row shape compatible with
`reward_diagnostics.jsonl`, with a small probe-specific extension.

Required fields:

| Field | Requirement |
| --- | --- |
| `artifact_schema_version` | `1` |
| `lineage_id` | probe lineage id |
| `rank` | `1` |
| `episode_index` | `0` |
| `episode_kind` | `probe` |
| `step_index` | case index, starting at `0` |
| `case_name` | stable case name |
| `source_state` | list form of source |
| `assembled_action` | list form of action |
| `attempted_target_state` | list form of target |
| `realized_next_state` | list form of target |
| `reward` | total reward |
| `hard_violation` | reward flag |
| `is_terminal_success` | reward flag |
| `reward_diagnostics` | full factory diagnostics |
| `terminated` | same as `is_terminal_success` |
| `truncated` | `false` |
| `outcome` | `valid` |

Do not pretend these are training episodes. The `episode_kind = probe` label is
intentional.

## Suggested Implementation Files

Expected source files:

| File | Work |
| --- | --- |
| `tower/reward/probe.py` | build probe cases and rows |
| `scripts/tower_reward_probe.py` | CLI wrapper to write probe artifact |

Expected tests:

| File | Work |
| --- | --- |
| `tests/tower/reward/test_probe.py` | probe case/reward/diagnostics tests |
| `tests/tower/train/test_diagnostics.py` | extend only if serializer needs probe support |
| `tests/tower/train/test_runner_script.py` | no change unless script discovery changes |

## Minimal CLI

Suggested command:

```bash
uv run python scripts/tower_reward_probe.py \
  --lineage-id slice-a-reward-probe \
  --artifact-root artifacts/tower
```

Expected stdout:

```text
probe_path: artifacts/tower/slice-a-reward-probe/rank_1/reward_term_probe.jsonl
case_count: 4
cases: terminal_cadence_success,recent_range_penalty,large_leap_recovery_success,large_leap_recovery_failure
```

## Tests

Focused tests should prove:

| Test | Proves |
| --- | --- |
| probe builds four cases | stable case inventory |
| terminal cadence row succeeds | cadence reward and terminal flag preserved |
| recent range row penalizes | observed range and penalty flag preserved |
| leap recovery success row rewards | success diagnostics preserved |
| leap recovery failure row penalizes | failure diagnostics preserved |
| probe writer emits JSONL | artifact can be read outside pytest |
| script runs by file path | direct script execution import path works |

## Non-Goals

Do not include:

| Non-goal | Reason |
| --- | --- |
| policy inference | reward evidence only |
| training | reward evidence only |
| MIDI export | not relevant to reward-term diagnostics |
| rank-2 reward terms | Slice B later |
| graph-boundary changes | separate action |
| manifest integration | optional later artifact-indexing pass |

## Verification Command

Recommended focused verification after implementation:

```bash
uv run pytest tests/tower/reward/test_probe.py \
  tests/tower/reward/test_probe_script.py \
  tests/tower/reward \
  tests/tower/test_import_boundaries.py
```

If script coverage is placed under `tests/tower/train` instead, use:

```bash
uv run pytest tests/tower/reward/test_probe.py \
  tests/tower/train/test_reward_probe_script.py \
  tests/tower/reward \
  tests/tower/test_import_boundaries.py
```

## Proposed Implementation Action

Recommended next action:

```text
Post-Slice-8 Phase 7.Stage 7.10.Action 7.10.1:
Implement Slice A Reward Term Evidence Probe
```

Expected files:

```text
tower/reward/probe.py
scripts/tower_reward_probe.py
tests/tower/reward/test_probe.py
tests/tower/reward/test_probe_script.py
```
