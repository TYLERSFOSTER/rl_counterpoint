# Rank-1 Slice A Training Evidence Contract

This document is the Post-Slice-8 Phase 8 / Stage 8.1 / Action 8.1.1
deliverable.

The purpose is to specify a small but nontrivial rank-1 training/evaluation run
using Reward Expansion Slice A.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 8: Longer Rank-1 Slice A Evidence |
| Stage | Stage 8.1: Evidence Run Contract |
| Action | Action 8.1.1: Specify Longer Rank-1 Slice A Training Evidence Run |

## Source Authority

This contract is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/reward_expansion_slice_a_closeout.md` | recommends longer rank-1 evidence before rank-2 work |
| `docs/design/tower/training_runner_contract.md` | rank training lifecycle and final inference requirement |
| `docs/design/tower/reward_diagnostics_evidence.md` | diagnostics artifact evidence |
| `docs/design/tower/slice_a_reward_probe_evidence.md` | Slice A term-path evidence |
| `scripts/tower_train.py` | current rank-1 training CLI |

## Purpose

The purpose is not to claim musical quality.

The purpose is to answer:

| Question | Evidence needed |
| --- | --- |
| can Slice A run beyond a one-step smoke test? | training metrics and final inference complete |
| do history-sensitive terms appear in real rollouts? | `reward_diagnostics.jsonl` contains range/leap rows beyond insufficient history |
| does the reward produce nonzero returns in practice? | metrics and diagnostics show reward contributions |
| does final MIDI still write on a longer run? | `example_episode.mid` exists and is inspectable |
| are artifacts still coherent? | config, metrics, checkpoint, manifest, diagnostics, MIDI all present |

## Recommended Run

Use this exact command unless reopened before execution:

```bash
uv run python scripts/tower_train.py \
  --rank 1 \
  --episodes 10 \
  --lineage-id rank1-slice-a-evidence-10x16 \
  --artifact-root artifacts/tower \
  --seed 123 \
  --max-steps 16 \
  --max-step-size 2
```

## Run Settings

| Setting | Value | Reason |
| --- | --- | --- |
| rank | `1` | evaluate Slice A before rank-2 expansion |
| episodes | `10` | still quick, but more than smoke |
| max_steps | `16` | enough history for range/leap terms to occur |
| max_step_size | `2` | modest action space; avoids huge random jumps dominating |
| seed | `123` | reproducible with prior evidence style |
| lineage_id | `rank1-slice-a-evidence-10x16` | explicit evidence lineage |
| artifact_root | `artifacts/tower` | existing tower artifact root |
| reward | `rank1_slice_a` defaults | test current Slice A, not tuning |
| policy | current script transformer defaults | keep policy architecture constant |

Current script policy defaults:

| Field | Value |
| --- | --- |
| d_model | `8` |
| num_layers | `1` |
| num_heads | `2` |
| ff_dim | `16` |
| dropout | `0.0` |

Current Slice A reward defaults:

| Field | Value |
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

## Expected Artifacts

The run should produce:

| Artifact | Path |
| --- | --- |
| lineage manifest | `artifacts/tower/rank1-slice-a-evidence-10x16/manifest.json` |
| rank config | `artifacts/tower/rank1-slice-a-evidence-10x16/rank_1/config.json` |
| metrics JSONL | `artifacts/tower/rank1-slice-a-evidence-10x16/rank_1/metrics.jsonl` |
| reward diagnostics JSONL | `artifacts/tower/rank1-slice-a-evidence-10x16/rank_1/reward_diagnostics.jsonl` |
| latest checkpoint | `artifacts/tower/rank1-slice-a-evidence-10x16/rank_1/checkpoint_latest.pt` |
| final MIDI | `artifacts/tower/rank1-slice-a-evidence-10x16/rank_1/example_episode.mid` |

Expected row counts:

| File | Expected |
| --- | --- |
| `metrics.jsonl` | `11` rows: 10 training, 1 final inference |
| `reward_diagnostics.jsonl` | at least `11` rows; likely up to `176` rows if every episode reaches 16 steps |

The diagnostics row count may be lower if terminal success occurs before
`max_steps`.

## Evidence To Inspect

### Script Output

Record stdout, including:

| Field | Expected |
| --- | --- |
| run_dir | rank artifact directory |
| lineage_dir | lineage artifact directory |
| rank | `1` |
| episodes | `10` |
| max_steps | `16` |
| reward | `rank1_slice_a` |
| latest checkpoint | checkpoint path |
| final midi | MIDI path |
| final episode return | observed value |
| final terminal_success | observed value |

### Config

Confirm:

| Config area | Requirement |
| --- | --- |
| reward_config.kind | `rank1_slice_a` |
| episode_budget | `10` |
| training max_steps | `16` |
| max_step_size | `2` |
| seed | `123` |
| policy config | current script defaults |

### Metrics

Summarize:

| Metric | Need |
| --- | --- |
| per-episode returns | list all 10 training returns |
| final inference return | record final row return |
| episode lengths | min/max and final |
| terminal_success count | count training and final successes |
| truncated count | count truncations |
| loss values | first, last, and any obvious anomaly |

### Reward Diagnostics

Inspect the diagnostics file for:

| Signal | Interpretation |
| --- | --- |
| cadence `success` rows | terminal cadence occurred in a real rollout |
| cadence `wrong_root_motion` rows | cadence checked but failed root motion |
| cadence `not_final_step` rows | expected common nonterminal path |
| range `range_exceeded` rows | range penalty fired in real rollout |
| range `within_range` rows | range stayed bounded |
| leap `recovered` rows | large leap recovery reward fired |
| leap `failed_recovery` rows | large leap recovery penalty fired |
| leap `no_large_leap` rows | expected common path |

Record counts of these reasons, not every row.

### MIDI

Confirm:

| Evidence | Requirement |
| --- | --- |
| file exists | yes |
| header magic | begins with `MThd` |
| final state | record from metrics |

This action does not require human musical evaluation of the MIDI yet. Listening
can be a follow-up if the run looks structurally sane.

## Evidence Report

After the run, write:

```text
docs/design/tower/rank1_slice_a_training_evidence.md
```

Required sections:

| Section | Content |
| --- | --- |
| command run | exact command |
| stdout | observed script output |
| artifacts | paths and file sizes |
| config evidence | key reward/training/policy fields |
| metrics evidence | returns, lengths, success/truncation, loss summary |
| diagnostics evidence | reason counts for cadence/range/leap terms |
| checkpoint evidence | checkpoint loads and stats match metrics |
| MIDI evidence | file exists and starts with `MThd` |
| interpretation | what the run suggests and what it does not prove |
| next recommendation | likely rank-2 script contract or repeat/tune rank-1 evidence |

## Stop Conditions

Pause and investigate if:

| Stop condition | Why |
| --- | --- |
| script errors | training path regression |
| any expected artifact missing | artifact contract regression |
| metrics row count is not 11 and no terminal successes explain it | runner evidence mismatch |
| diagnostics file missing or malformed | diagnostics artifact regression |
| checkpoint fails to load | checkpoint artifact regression |
| full run takes unexpectedly long | run size may be too large for current policy path |

## Verification

Before or after the run, keep the focused implementation path green:

```bash
uv run pytest tests/tower/train/test_runner_script.py \
  tests/tower/train/test_runner.py \
  tests/tower/train/test_diagnostics.py \
  tests/tower/reward \
  tests/tower/test_import_boundaries.py
```

Full repo verification is optional after this evidence-only run unless code is
changed.

## Proposed Execution Action

Recommended next action:

```text
Post-Slice-8 Phase 8.Stage 8.2.Action 8.2.1:
Run Longer Rank-1 Slice A Training Evidence Command
```

Expected command:

```bash
uv run python scripts/tower_train.py \
  --rank 1 \
  --episodes 10 \
  --lineage-id rank1-slice-a-evidence-10x16 \
  --artifact-root artifacts/tower \
  --seed 123 \
  --max-steps 16 \
  --max-step-size 2
```
