# Post-Slice-8 Reward Evidence

This document is the Post-Slice-8 Phase 7 / Stage 7.5 / Action 7.5.1
deliverable.

The purpose is to record the first tiny artifact-backed rank-1 training run
using Reward Expansion Slice A.

This is an evidence document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.5: Reward Evidence |
| Action | Action 7.5.1: Run Tiny Rank-1 Reward Evidence |

## Source Authority

This evidence follows:

| Source | Role |
| --- | --- |
| `docs/design/tower/reward_expansion_slice_a_contract.md` | accepted Slice A reward contract |
| `tower/reward/factory.py` | rank-1 reward factory implementation |
| `tower/reward/melody.py` | rank-1 melodic reward terms |
| `scripts/tower_train.py` | artifact-backed training command |

## Script Wiring Change

Before this evidence run, `scripts/tower_train.py` still used a constant reward:

```text
TowerRewardResult(reward=1.0)
```

That would not have proven the Slice A reward path. The script was therefore
minimally wired to:

```text
build_rank1_reward_fn(...)
```

The script now records the reward settings in `reward_config` and prints:

```text
reward: rank1_slice_a
```

## Command Run

Executed from repo root:

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

## Script Output

Observed stdout:

```text
run_dir: artifacts/tower/reward-slice-a-tiny/rank_1
lineage_dir: artifacts/tower/reward-slice-a-tiny
rank: 1
episodes: 1
max_steps: 1
reward: rank1_slice_a
latest checkpoint: artifacts/tower/reward-slice-a-tiny/rank_1/checkpoint_latest.pt
final midi: artifacts/tower/reward-slice-a-tiny/rank_1/example_episode.mid
final episode return: 0.0
final terminal_success: False
```

## Artifact Paths

The run produced:

| Artifact | Path |
| --- | --- |
| lineage manifest | `artifacts/tower/reward-slice-a-tiny/manifest.json` |
| rank config | `artifacts/tower/reward-slice-a-tiny/rank_1/config.json` |
| metrics JSONL | `artifacts/tower/reward-slice-a-tiny/rank_1/metrics.jsonl` |
| latest checkpoint | `artifacts/tower/reward-slice-a-tiny/rank_1/checkpoint_latest.pt` |
| final MIDI | `artifacts/tower/reward-slice-a-tiny/rank_1/example_episode.mid` |

Observed file sizes at run time:

| Artifact | Size |
| --- | --- |
| manifest | 266 bytes |
| config | 850 bytes |
| metrics | 638 bytes |
| checkpoint | 32463 bytes |
| MIDI | 44 bytes |

## Reward Config Evidence

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

This proves the artifact-backed run used the Slice A reward configuration rather
than the earlier constant reward placeholder.

## Training Config Evidence

The rank config also recorded:

| Field | Value |
| --- | --- |
| rank | `1` |
| lineage_id | `reward-slice-a-tiny` |
| episode_budget | `1` |
| measure_size | `4` |
| context_measures | `2` |
| max_step_size | `1` |
| parent_checkpoint | `null` |
| parent_sampler_config | `{}` |
| seed | `123` |
| training max_steps | `1` |
| training learning_rate | `0.001` |

The policy config recorded the small script defaults:

| Field | Value |
| --- | --- |
| d_model | `8` |
| num_layers | `1` |
| num_heads | `2` |
| ff_dim | `16` |
| dropout | `0.0` |

## Metrics Evidence

The metrics file contained two rows:

| Row | Meaning |
| --- | --- |
| 0 | training episode |
| 1 | final no-train inference episode |

Training row:

| Metric | Value |
| --- | --- |
| episode_index | `0` |
| rank | `1` |
| episode_return | `0.0` |
| episode_length | `1` |
| mean_step_reward | `0.0` |
| terminated | `false` |
| truncated | `true` |
| loss | `-0.0` |
| invalid_extension_count | `0` |
| empty_lift_fiber_count | `0` |
| parent_failure_count | `0` |
| terminal_success | `false` |

Final inference row:

| Metric | Value |
| --- | --- |
| episode_index | `1` |
| kind | `final_inference` |
| final_inference | `true` |
| rank | `1` |
| episode_return | `0.0` |
| episode_length | `1` |
| mean_step_reward | `0.0` |
| terminated | `false` |
| truncated | `true` |
| final_state | `[61]` |
| midi_path | `rank_1/example_episode.mid` |
| terminal_success | `false` |

The zero return is expected for this deliberately tiny one-step run. It is not a
cadence, it does not exceed the recent range threshold, and it does not follow a
prior large leap.

## Manifest Evidence

The lineage manifest recorded:

| Field | Value |
| --- | --- |
| lineage_id | `reward-slice-a-tiny` |
| artifact_schema_version | `1` |
| rank 1 status | `accepted` |
| rank 1 checkpoint | `rank_1/checkpoint_latest.pt` |
| rank 1 config | `rank_1/config.json` |
| rank 1 metrics | `rank_1/metrics.jsonl` |

## Checkpoint Evidence

The checkpoint loaded successfully with:

```text
torch.load(..., weights_only=True)
```

Observed checkpoint metadata:

| Field | Value |
| --- | --- |
| rank | `1` |
| lineage_id | `reward-slice-a-tiny` |
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
artifacts/tower/reward-slice-a-tiny/rank_1/example_episode.mid
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

## Verification

Focused verification after script wiring:

```bash
uv run pytest tests/tower/train/test_runner_script.py \
  tests/tower/reward/test_factory.py \
  tests/tower/test_import_boundaries.py
```

Observed result:

```text
13 passed
```

## What This Proves

This run proves:

| Claim | Evidence |
| --- | --- |
| script uses Slice A reward factory | stdout prints `reward: rank1_slice_a` |
| reward config is artifact-backed | `config.json` includes `reward_config.kind = rank1_slice_a` |
| runner accepts the factory reward | training episode and final inference both complete |
| artifact layout still works | config/metrics/checkpoint/manifest/MIDI exist |
| checkpoint path still works | checkpoint loads with expected keys and stats |
| final MIDI export still works | MIDI file exists and has `MThd` header |

## Limitations

This run is intentionally tiny.

| Limitation | Meaning |
| --- | --- |
| one episode | no learning claim |
| one step | no long-horizon musical behavior claim |
| zero return | expected because no cadence/range/leap-recovery condition triggered |
| metrics lack reward diagnostics | current metrics summarize scalar reward but do not persist per-term diagnostics |
| rank 1 only | Slice B/rank-2 reward work remains future work |

## Next Recommendation

The next useful move is to decide whether to improve reward evidence observability
before moving on.

Recommended next action:

```text
Post-Slice-8 Phase 7.Stage 7.6.Action 7.6.1:
Specify Reward Diagnostics Evidence Contract
```

Rationale:

The reward factory works in the runner, but metrics currently record only scalar
episode summaries. A small diagnostics-evidence contract would decide whether
per-term reward diagnostics should be persisted in metrics, trajectories, or a
separate artifact before larger reward experiments begin.
