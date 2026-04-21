# Slice A Reward Probe Evidence

This document is the Post-Slice-8 Phase 7 / Stage 7.11 / Action 7.11.1
deliverable.

The purpose is to record deterministic artifact-style evidence for the
meaningful Reward Expansion Slice A term paths.

This is an evidence document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.11: Slice A Reward Probe Evidence |
| Action | Action 7.11.1: Write Slice A Reward Probe Evidence Report |

## Source Authority

This evidence follows:

| Source | Role |
| --- | --- |
| `docs/design/tower/slice_a_reward_probe_contract.md` | accepted probe contract |
| `tower/reward/probe.py` | deterministic probe case implementation |
| `scripts/tower_reward_probe.py` | probe artifact command |
| `tower/reward/factory.py` | composed Slice A rank-1 reward factory |
| `tower/reward/melody.py` | rank-1 melodic shape terms |

## Command Run

Executed from repo root:

```bash
uv run python scripts/tower_reward_probe.py \
  --lineage-id slice-a-reward-probe \
  --artifact-root artifacts/tower
```

## Script Output

Observed stdout:

```text
probe_path: artifacts/tower/slice-a-reward-probe/rank_1/reward_term_probe.jsonl
case_count: 4
cases: terminal_cadence_success,recent_range_penalty,large_leap_recovery_success,large_leap_recovery_failure
```

## Artifact Paths

The run produced:

| Artifact | Path |
| --- | --- |
| probe JSONL | `artifacts/tower/slice-a-reward-probe/rank_1/reward_term_probe.jsonl` |

Observed file size at run time:

| Artifact | Size |
| --- | --- |
| probe JSONL | 5641 bytes |

This is intentionally not a training artifact. It does not write config,
metrics, checkpoint, manifest, or MIDI.

## Probe Rows

The probe artifact contained four rows:

| Row | Case | Reward | Terminal success |
| --- | --- | --- | --- |
| 0 | `terminal_cadence_success` | `9.5` | `true` |
| 1 | `recent_range_penalty` | `-1.5` | `false` |
| 2 | `large_leap_recovery_success` | `0.5` | `false` |
| 3 | `large_leap_recovery_failure` | `-0.5` | `false` |

Every row recorded:

| Field | Status |
| --- | --- |
| artifact_schema_version | present |
| lineage_id | `slice-a-reward-probe` |
| rank | `1` |
| episode_index | `0` |
| episode_kind | `probe` |
| case_name | present |
| source_state | present |
| assembled_action | present |
| attempted_target_state | present |
| realized_next_state | present |
| reward | present |
| reward_diagnostics | present |
| outcome | `valid` |

Top-level reward diagnostics for every row used:

```text
kind = rank1_reward
key_pitch_class = 0
terms = <cadence, recent range, large leap recovery>
```

## Term Diagnostics Summary

Observed cadence reasons:

| Case | Cadence reason |
| --- | --- |
| `terminal_cadence_success` | `success` |
| `recent_range_penalty` | `not_final_step` |
| `large_leap_recovery_success` | `not_final_step` |
| `large_leap_recovery_failure` | `not_final_step` |

Observed recent melodic range reasons:

| Case | Range reason |
| --- | --- |
| `terminal_cadence_success` | `within_range` |
| `recent_range_penalty` | `range_exceeded` |
| `large_leap_recovery_success` | `within_range` |
| `large_leap_recovery_failure` | `within_range` |

Observed large leap recovery reasons:

| Case | Leap recovery reason |
| --- | --- |
| `terminal_cadence_success` | `failed_recovery` |
| `recent_range_penalty` | `failed_recovery` |
| `large_leap_recovery_success` | `recovered` |
| `large_leap_recovery_failure` | `failed_recovery` |

## Case Evidence

### Terminal Cadence Success

Observed row:

| Field | Value |
| --- | --- |
| source_state | `[60]` |
| assembled_action | `[-1]` |
| attempted_target_state | `[59]` |
| realized_next_state | `[59]` |
| reward | `9.5` |
| is_terminal_success | `true` |

Term contributions:

| Term | Reward | Diagnostics |
| --- | --- | --- |
| cadence | `10.0` | `reason = success`, `previous_pitch_class = 7`, `final_pitch_class = 0`, `dominant_pitch_class = 7`, `tonic_pitch_class = 0` |
| recent range | `0.0` | `observed_range = 7`, `reason = within_range` |
| large leap recovery | `-0.5` | `previous_interval = -7`, `current_action = -1`, `opposite_direction = false`, `reason = failed_recovery` |

Total:

```text
10.0 + 0.0 - 0.5 = 9.5
```

### Recent Range Penalty

Observed row:

| Field | Value |
| --- | --- |
| source_state | `[73]` |
| assembled_action | `[1]` |
| attempted_target_state | `[74]` |
| realized_next_state | `[74]` |
| reward | `-1.5` |
| is_terminal_success | `false` |

Term contributions:

| Term | Reward | Diagnostics |
| --- | --- | --- |
| cadence | `0.0` | `reason = not_final_step` |
| recent range | `-1.0` | `observed_range = 13`, `penalty_applied = true`, `reason = range_exceeded` |
| large leap recovery | `-0.5` | `previous_interval = 13`, `current_action = 1`, `opposite_direction = false`, `reason = failed_recovery` |

Total:

```text
0.0 - 1.0 - 0.5 = -1.5
```

### Large Leap Recovery Success

Observed row:

| Field | Value |
| --- | --- |
| source_state | `[67]` |
| assembled_action | `[-2]` |
| attempted_target_state | `[65]` |
| realized_next_state | `[65]` |
| reward | `0.5` |
| is_terminal_success | `false` |

Term contributions:

| Term | Reward | Diagnostics |
| --- | --- | --- |
| cadence | `0.0` | `reason = not_final_step` |
| recent range | `0.0` | `observed_range = 7`, `reason = within_range` |
| large leap recovery | `0.5` | `previous_interval = 7`, `current_action = -2`, `opposite_direction = true`, `small_step = true`, `success = true`, `reason = recovered` |

Total:

```text
0.0 + 0.0 + 0.5 = 0.5
```

### Large Leap Recovery Failure

Observed row:

| Field | Value |
| --- | --- |
| source_state | `[67]` |
| assembled_action | `[2]` |
| attempted_target_state | `[69]` |
| realized_next_state | `[69]` |
| reward | `-0.5` |
| is_terminal_success | `false` |

Term contributions:

| Term | Reward | Diagnostics |
| --- | --- | --- |
| cadence | `0.0` | `reason = not_final_step` |
| recent range | `0.0` | `observed_range = 7`, `reason = within_range` |
| large leap recovery | `-0.5` | `previous_interval = 7`, `current_action = 2`, `opposite_direction = false`, `success = false`, `reason = failed_recovery` |

Total:

```text
0.0 + 0.0 - 0.5 = -0.5
```

## Sign Correction

During implementation, a small sign issue in the probe contract was found.

The original cadence probe sketch used:

```text
history = ((67,), (60,))
action = (+1,)
```

But the prior interval is:

```text
60 - 67 = -7
```

So action `+1` is opposite-direction recovery and would produce:

```text
cadence success +10.0
large leap recovery success +0.5
total = 10.5
```

The intended probe case was cadence success plus failed leap recovery, with
total `9.5`. The implemented probe therefore uses:

```text
action = (-1,)
```

That preserves:

```text
cadence success +10.0
large leap recovery failure -0.5
total = 9.5
```

This correction is now encoded in tests and in the generated artifact row.

## Verification

Focused verification during implementation:

```bash
uv run pytest tests/tower/reward/test_probe.py \
  tests/tower/reward/test_probe_script.py \
  tests/tower/reward \
  tests/tower/test_import_boundaries.py
```

Observed result:

```text
69 passed
```

Broader verification:

```bash
uv run pytest tests/tower
uv run pytest
```

Observed results:

```text
396 passed
551 passed
```

## What This Proves

This probe proves:

| Claim | Evidence |
| --- | --- |
| composed Slice A reward is probeable without training | runner-free JSONL artifact exists |
| cadence success is inspectable | terminal cadence row has `reason = success` |
| recent range penalty is inspectable | range row has `observed_range = 13`, `penalty_applied = true` |
| large leap recovery success is inspectable | recovery success row has `reason = recovered` |
| large leap recovery failure is inspectable | failure rows have `reason = failed_recovery` |
| reward interactions are visible | each row preserves all three term records |

## Limitations

This probe is deterministic reward evidence only.

| Limitation | Meaning |
| --- | --- |
| no policy | does not prove model can learn these cases |
| no training loop | does not prove reward improves training |
| no graph legality check | contexts are hand-built reward contexts |
| rank 1 only | rank-2 reward evidence remains future work |
| no manifest | probe artifact is intentionally not manifest-indexed yet |

## Next Recommendation

The next useful move is to close Phase 7 with a reward-expansion status note and
choose the next build direction.

Recommended next action:

```text
Post-Slice-8 Phase 7.Stage 7.12.Action 7.12.1:
Write Reward Expansion Slice A Closeout
```

That closeout should summarize implemented terms, factory wiring, diagnostics
artifacts, probe evidence, limitations, and whether the next phase should be
longer rank-1 training evidence or Slice B rank-2 reward planning.
