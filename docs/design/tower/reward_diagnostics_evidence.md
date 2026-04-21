# Reward Diagnostics Evidence

This document is the Post-Slice-8 Phase 7 / Stage 7.8 / Action 7.8.1
deliverable.

The purpose is to record the first tiny artifact-backed run proving that
per-step reward diagnostics are persisted to `reward_diagnostics.jsonl`.

This is an evidence document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.8: Reward Diagnostics Evidence |
| Action | Action 7.8.1: Run Tiny Reward Diagnostics Evidence |

## Source Authority

This evidence follows:

| Source | Role |
| --- | --- |
| `docs/design/tower/reward_diagnostics_evidence_contract.md` | accepted diagnostics artifact contract |
| `tower/train/diagnostics.py` | diagnostics row serializer |
| `tower/train/checkpoint.py` | diagnostics artifact path/read/write helpers |
| `tower/train/runner.py` | runner writes training and final inference diagnostics rows |
| `scripts/tower_train.py` | artifact-backed training command |

## Command Run

Executed from repo root:

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

## Script Output

Observed stdout:

```text
run_dir: artifacts/tower/reward-diagnostics-tiny/rank_1
lineage_dir: artifacts/tower/reward-diagnostics-tiny
rank: 1
episodes: 1
max_steps: 1
reward: rank1_slice_a
latest checkpoint: artifacts/tower/reward-diagnostics-tiny/rank_1/checkpoint_latest.pt
final midi: artifacts/tower/reward-diagnostics-tiny/rank_1/example_episode.mid
final episode return: 0.0
final terminal_success: False
```

## Artifact Paths

The run produced:

| Artifact | Path |
| --- | --- |
| lineage manifest | `artifacts/tower/reward-diagnostics-tiny/manifest.json` |
| rank config | `artifacts/tower/reward-diagnostics-tiny/rank_1/config.json` |
| metrics JSONL | `artifacts/tower/reward-diagnostics-tiny/rank_1/metrics.jsonl` |
| reward diagnostics JSONL | `artifacts/tower/reward-diagnostics-tiny/rank_1/reward_diagnostics.jsonl` |
| latest checkpoint | `artifacts/tower/reward-diagnostics-tiny/rank_1/checkpoint_latest.pt` |
| final MIDI | `artifacts/tower/reward-diagnostics-tiny/rank_1/example_episode.mid` |

Observed file sizes at run time:

| Artifact | Size |
| --- | --- |
| manifest | 333 bytes |
| config | 854 bytes |
| metrics | 638 bytes |
| reward diagnostics | 3261 bytes |
| checkpoint | 32463 bytes |
| MIDI | 44 bytes |

## Manifest Evidence

The lineage manifest recorded:

| Field | Value |
| --- | --- |
| lineage_id | `reward-diagnostics-tiny` |
| artifact_schema_version | `1` |
| rank 1 status | `accepted` |
| rank 1 checkpoint | `rank_1/checkpoint_latest.pt` |
| rank 1 config | `rank_1/config.json` |
| rank 1 metrics | `rank_1/metrics.jsonl` |
| rank 1 reward diagnostics | `rank_1/reward_diagnostics.jsonl` |

This proves the diagnostics artifact is now discoverable from the lineage
manifest.

## Config Evidence

The rank config recorded:

| Field | Value |
| --- | --- |
| reward kind | `rank1_slice_a` |
| key_pitch_class | `0` |
| terminal_cadence_reward | `10.0` |
| cadence_failure_reward | `0.0` |
| max_recent_range | `12` |
| range_penalty | `-1.0` |
| large_leap_threshold | `6` |
| recovery_step_threshold | `3` |
| recovery_reward | `0.5` |
| failure_penalty | `-0.5` |

Training settings:

| Field | Value |
| --- | --- |
| rank | `1` |
| lineage_id | `reward-diagnostics-tiny` |
| episode_budget | `1` |
| measure_size | `4` |
| context_measures | `2` |
| max_step_size | `1` |
| training max_steps | `1` |
| seed | `123` |

## Metrics Evidence

The metrics file still contains scalar episode summaries only.

Rows:

| Row | Meaning |
| --- | --- |
| 0 | training episode |
| 1 | final no-train inference episode |

Both rows recorded:

| Metric | Value |
| --- | --- |
| rank | `1` |
| episode_length | `1` |
| episode_return | `0.0` |
| mean_step_reward | `0.0` |
| terminal_success | `false` |
| terminated | `false` |
| truncated | `true` |

The final inference row additionally recorded:

| Metric | Value |
| --- | --- |
| kind | `final_inference` |
| final_inference | `true` |
| final_state | `[61]` |
| midi_path | `rank_1/example_episode.mid` |

This confirms that scalar metrics remain compact while the new diagnostics
artifact carries per-step explanations.

## Reward Diagnostics Evidence

The diagnostics file contained two JSONL rows:

| Row | episode_index | episode_kind | reward | reward diagnostics kind |
| --- | --- | --- | --- | --- |
| 0 | `0` | `training` | `0.0` | `rank1_reward` |
| 1 | `1` | `final_inference` | `0.0` | `rank1_reward` |

Both rows included:

| Field | Example value |
| --- | --- |
| source_state | `[60]` |
| assembled_action | `[1]` |
| attempted_target_state | `[61]` |
| realized_next_state | `[61]` |
| active_choice | `1` |
| outcome | `valid` |
| terminated | `false` |
| truncated | `true` |
| hard_violation | `false` |
| is_terminal_success | `false` |

Top-level reward diagnostics:

| Field | Value |
| --- | --- |
| kind | `rank1_reward` |
| key_pitch_class | `0` |
| terms | three term records |

The first training row preserved the expected Slice A term diagnostics:

| Term | Reward | Reason |
| --- | --- | --- |
| cadence | `0.0` | `insufficient_valid_history` |
| recent melodic range | `0.0` | `insufficient_valid_history` |
| large leap recovery | `0.0` | `insufficient_valid_history` |

The detailed nested diagnostics included:

| Term | Additional details |
| --- | --- |
| cadence | `kind = rank1_projected_cadence_success`, `rank = 1` |
| recent melodic range | `valid_pitch_count = 1`, `max_recent_range = 12`, `observed_range = null`, `penalty_applied = false` |
| large leap recovery | `current_action = 1`, `previous_interval = null`, `triggered = false`, `opposite_direction = false`, `small_step = false`, `success = false` |

This is expected for a one-step run from a one-state window: no cadence can be
recognized, no recent range can be computed, and no prior leap exists.

## Checkpoint Evidence

The checkpoint loaded successfully with:

```text
torch.load(..., weights_only=True)
```

Observed checkpoint metadata:

| Field | Value |
| --- | --- |
| rank | `1` |
| lineage_id | `reward-diagnostics-tiny` |
| episode_index | `0` |
| artifact_schema_version | `1` |

Observed checkpoint keys:

```text
artifact_schema_version
config
episode_index
lineage_id
optimizer_state_dict
policy_state_dict
rank
stats
```

Checkpoint stats matched the training metrics row.

## MIDI Evidence

The MIDI file exists at:

```text
artifacts/tower/reward-diagnostics-tiny/rank_1/example_episode.mid
```

The file begins with the standard MIDI header magic:

```text
MThd
```

The first 32 bytes observed by `xxd`:

```text
00000000: 4d54 6864 0000 0006 0000 0001 01e0 4d54  MThd..........MT
00000010: 726b 0000 0016 0090 3c40 8360 803c 0000  rk......<@.`.<..
```

## What This Proves

This run proves:

| Claim | Evidence |
| --- | --- |
| diagnostics artifact is produced | `reward_diagnostics.jsonl` exists |
| manifest exposes diagnostics artifact | manifest includes `reward_diagnostics` path |
| training diagnostics are persisted | row with `episode_kind = training` exists |
| final inference diagnostics are persisted | row with `episode_kind = final_inference` exists |
| Slice A reward diagnostics survive JSONL serialization | rows include `reward_diagnostics.kind = rank1_reward` |
| term-level reasons are inspectable | cadence/range/leap reasons are preserved |
| metrics remain scalar summaries | `metrics.jsonl` did not absorb verbose diagnostics |

## Limitations

This run is intentionally tiny.

| Limitation | Meaning |
| --- | --- |
| one episode | no learning claim |
| one step | no long-horizon reward behavior claim |
| all term reasons are insufficient history | expected from one-state history, but does not prove positive/penalty term firing in artifact evidence |
| rank 1 only | rank-2 diagnostics are covered by tests, not this evidence run |

## Next Recommendation

The next useful move is to run a tiny scripted diagnostics probe that forces each
Slice A term path at least once, without doing full training.

Recommended next action:

```text
Post-Slice-8 Phase 7.Stage 7.9.Action 7.9.1:
Specify Slice A Reward Term Evidence Probe
```

Rationale:

The artifact path now preserves diagnostics correctly. The remaining evidence
gap is term coverage in saved artifacts: a one-step training run can only show
the no-history paths. A tiny deterministic probe can demonstrate cadence
success, range penalty, and leap-recovery success/failure diagnostics in an
artifact-like JSONL file before longer training experiments.
