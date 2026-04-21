# Reward Diagnostics Evidence Contract

This document is the Post-Slice-8 Phase 7 / Stage 7.6 / Action 7.6.1
deliverable.

The purpose is to specify how reward diagnostics should be persisted before
larger reward experiments begin.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.6: Reward Diagnostics Evidence Contract |
| Action | Action 7.6.1: Specify Reward Diagnostics Evidence Contract |

## Source Authority

This contract is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/post_slice_8_reward_evidence.md` | evidence gap: Slice A reward runs, but per-term diagnostics are not persisted |
| `tower/train/trajectory.py` | in-memory trajectory steps already carry `reward.diagnostics` |
| `tower/reward/result.py` | reward result stores structured diagnostics |
| `tower/train/checkpoint.py` | current artifact path and JSONL conventions |
| `tower/reward/factory.py` | current Slice A reward diagnostics shape |

## Problem

The artifact-backed training path currently persists scalar episode summaries in:

```text
metrics.jsonl
```

Those rows include:

```text
episode_return
mean_step_reward
terminal_success
loss
truncated
terminated
```

They do not include the per-step reward diagnostics already present in memory on:

```text
TowerTrajectoryStep.reward.diagnostics
```

For Slice A, this means artifacts can prove that the reward factory ran, but not
which reward terms fired or why.

## Design Decision

Keep `metrics.jsonl` scalar and summary-friendly.

Add a separate JSONL artifact for per-step trajectory and reward diagnostics.

Recommended filename:

```text
reward_diagnostics.jsonl
```

Recommended artifact location:

```text
artifacts/tower/<lineage_id>/rank_<k>/reward_diagnostics.jsonl
```

Recommended manifest field:

```json
{
  "reward_diagnostics": "rank_1/reward_diagnostics.jsonl"
}
```

## Why Separate From Metrics

| Reason | Explanation |
| --- | --- |
| metrics stay compact | episode-level scalar rows remain easy to scan and compare |
| diagnostics can be verbose | reward terms may include nested per-term evidence |
| per-step cardinality differs | one episode row can correspond to many trajectory-step rows |
| final inference also needs rows | diagnostics should cover training and final no-train inference |
| future ranks need room | rank-2 and higher diagnostics may include parent/lift context |

## Artifact Row Contract

Each row in `reward_diagnostics.jsonl` should represent one trajectory step.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `artifact_schema_version` | int | artifact schema version |
| `lineage_id` | str | lineage id |
| `rank` | int | active tower rank |
| `episode_index` | int | training episode index, or final synthetic episode index |
| `episode_kind` | str | `training` or `final_inference` |
| `step_index` | int | rollout step index |
| `source_state` | list[int] | pre-action state |
| `assembled_action` | list[int] | full rank-local action |
| `attempted_target_state` | list[int] | target before outcome handling |
| `realized_next_state` | list[int] | state recorded after outcome handling |
| `reward` | float | scalar reward for the step |
| `hard_violation` | bool | reward hard-violation flag |
| `is_terminal_success` | bool | reward terminal-success flag |
| `reward_diagnostics` | object | `TowerRewardResult.diagnostics` |
| `terminated` | bool | trajectory step termination flag |
| `truncated` | bool | trajectory step truncation flag |
| `outcome` | str | trajectory outcome string |

Optional fields:

| Field | Include when | Meaning |
| --- | --- | --- |
| `parent_state` | rank > 1 | projected parent state |
| `parent_action` | rank > 1 | chosen parent action |
| `active_choice` | valid active choice exists | active coordinate choice |
| `step_diagnostics` | always acceptable | sampler/outcome diagnostics from the trajectory step |

Do not include tensor objects or log-prob tensors directly. If log probabilities
are later needed, convert detached scalar tensor values to floats in a separate
contract action.

## JSON Compatibility

Rows must be JSON-compatible.

Conversion rules:

| Source value | Persist as |
| --- | --- |
| tuple state/action | list |
| tuple diagnostics | list |
| scalar tensor | float, if explicitly included later |
| non-scalar tensor | not allowed |
| Path | relative string |

The writer should reject non-JSON-compatible diagnostics instead of silently
dropping them.

## Episode Indexing

Training episodes should use their real `episode_index`.

Final inference should use the same synthetic index currently used in metrics:

```text
episode_index = config.episode_count
episode_kind = final_inference
```

This keeps diagnostics aligned with existing `metrics.jsonl` rows.

## Expected Slice A Diagnostics

For the current rank-1 Slice A reward, rows should preserve the top-level
composite diagnostics:

```text
kind = rank1_reward
key_pitch_class = <configured key>
terms = [...]
```

The `terms` list should include diagnostics for:

| Term | Expected diagnostics |
| --- | --- |
| cadence | predicate kind, reason, pitch classes when available |
| recent melodic range | valid pitch count, threshold, observed range, penalty flag |
| large leap recovery | previous interval, current action, thresholds, recovery flags |

## Current Evidence Gap Example

The `reward-slice-a-tiny` evidence run recorded:

| Evidence | Status |
| --- | --- |
| reward config | persisted |
| scalar reward | persisted |
| terminal success | persisted |
| cadence diagnostics | not persisted |
| range diagnostics | not persisted |
| leap-recovery diagnostics | not persisted |

This contract closes that gap without changing reward semantics.

## Suggested Implementation Files

Expected source files:

| File | Work |
| --- | --- |
| `tower/train/diagnostics.py` | serialize trajectory reward diagnostics rows |
| `tower/train/checkpoint.py` | add artifact path and read/write helpers |
| `tower/train/runner.py` | write diagnostics for training and final inference episodes |

Expected tests:

| File | Work |
| --- | --- |
| `tests/tower/train/test_diagnostics.py` | focused serializer/writer tests |
| `tests/tower/train/test_checkpoint.py` | artifact path/helper tests |
| `tests/tower/train/test_runner.py` | runner writes training and final diagnostics rows |
| `tests/tower/train/test_runner_script.py` | script artifact includes diagnostics file |

## Minimal Implementation Behavior

The first implementation should:

| Behavior | Requirement |
| --- | --- |
| write diagnostics rows | one JSONL row per trajectory step |
| include training episodes | append rows after each training episode |
| include final inference | append rows after final no-train inference |
| preserve reward diagnostics | write full `reward.diagnostics` structure |
| keep metrics unchanged | do not move existing scalar metrics |
| update manifest | include relative diagnostics path |
| test JSON round trip | read back rows as JSON-compatible objects |

## Non-Goals

Do not include in the first implementation:

| Non-goal | Reason |
| --- | --- |
| HTML/plot reports | artifact observability first |
| model logits/probability traces | separate policy diagnostics concern |
| MIDI note-level annotation | separate music artifact concern |
| reward-weight sweeps | requires diagnostics artifact first |
| checkpoint promotion by reward quality | acceptance remains episode-count based |

## Verification Command

Recommended focused verification after implementation:

```bash
uv run pytest tests/tower/train/test_diagnostics.py \
  tests/tower/train/test_checkpoint.py \
  tests/tower/train/test_runner.py \
  tests/tower/train/test_runner_script.py \
  tests/tower/reward \
  tests/tower/test_import_boundaries.py
```

## Proposed Implementation Action

Recommended next action:

```text
Post-Slice-8 Phase 7.Stage 7.7.Action 7.7.1:
Implement Reward Diagnostics Artifact
```

Expected files:

```text
tower/train/diagnostics.py
tower/train/checkpoint.py
tower/train/runner.py
tests/tower/train/test_diagnostics.py
tests/tower/train/test_checkpoint.py
tests/tower/train/test_runner.py
tests/tower/train/test_runner_script.py
```
