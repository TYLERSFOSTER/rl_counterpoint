# Artifact And Checkpoint Dependencies

This document is the Phase 4 / Stage 12 deliverable for the tower redesign.

The purpose is to define where tower training artifacts live, how rank checkpoints are named and loaded, how parent checkpoint dependencies are recorded, and what must be reproducible across ranks.

This is a mathematical and design contract, not an implementation file.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 4: Freeze Training Protocol |
| Stage | Stage 12: Define artifact and checkpoint dependencies |
| Action | Specify rank-specific artifacts, checkpoint lineage, and reproducibility metadata |

Stage 12 exit criterion:

| Requirement | Status |
| --- | --- |
| define tower artifact root | drafted here |
| define shared lineage directory structure | drafted here |
| define rank-specific run directories | drafted here |
| define rank checkpoint contents | drafted here |
| define parent checkpoint dependency metadata | drafted here |
| define lineage manifest | drafted here |
| define accepted checkpoint lookup | drafted here |
| define reproducibility metadata | drafted here |

## Old-System Baseline

The old flat `rl_counterpoint` training harness already made several useful artifact decisions.

| Artifact question | Old flat answer |
| --- | --- |
| run artifact directory | `artifacts/train_reinforce/` |
| config persistence | `config.json` |
| metrics persistence | `metrics.jsonl` |
| checkpoint style | rolling `checkpoint_latest.pt` |
| checkpoint contents | episode index, config, stats, policy state dict, optimizer state dict |
| MIDI export | `example_episode.mid` |
| end-of-training export | yes |
| printed summaries | run dir, config knobs, episode metrics, latest checkpoint path |

The tower should carry this forward rankwise rather than inventing a much heavier system.

## Tower Artifact Root

The tower artifact root is:

```text
artifacts/tower/
```

Every tower training lineage lives under this root.

## Shared Lineage Directory

Tower training should use a shared lineage directory:

```text
artifacts/tower/<lineage_id>/
```

The lineage directory exists because tower training is sequential and dependent:

\[
\pi^1 \longrightarrow \pi^2 \longrightarrow \pi^3 \longrightarrow \cdots
\]

The lineage captures one no-rollback chain of rank checkpoints. If a lower-rank flaw requires retraining, that should create a new lineage rather than silently mutating the old one.

## Directory Layout

The standard artifact layout is:

```text
artifacts/tower/<lineage_id>/
  manifest.json
  rank_1/
    config.json
    metrics.jsonl
    checkpoint_latest.pt
    example_episode.mid
  rank_2/
    config.json
    metrics.jsonl
    checkpoint_latest.pt
    example_episode.mid
  rank_3/
    config.json
    metrics.jsonl
    checkpoint_latest.pt
    example_episode.mid
```

Additional rank directories continue the same pattern:

```text
rank_<k>/
```

## Rank Directory

Each rank directory owns artifacts for one active training tier:

\[
\pi^k.
\]

Required files:

| File | Meaning |
| --- | --- |
| `config.json` | persisted rank-\(k\) training config |
| `metrics.jsonl` | append-only episode metrics |
| `checkpoint_latest.pt` | rolling latest rank-\(k\) checkpoint |

Recommended generated artifact:

| File | Meaning |
| --- | --- |
| `example_episode.mid` | end-of-training example rollout export for that rank |

Optional future artifacts:

| File | Meaning |
| --- | --- |
| `diagnostics.jsonl` | detailed rollout/reward diagnostics |
| `eval_metrics.json` | validation/evaluation summary |
| `sample_*.mid` | multiple sampled exports |

## Rolling Latest Checkpoint

The tower should preserve the old rolling-latest checkpoint behavior:

```text
checkpoint_latest.pt
```

For now, do not save every episode checkpoint by default.

The purpose is to keep local research runs simple and avoid artifact sprawl.

If archival checkpoints become necessary later, they should be added deliberately, for example:

```text
checkpoint_episode_<episode_index>.pt
```

but that is not part of the Stage 12 default.

## Rank Checkpoint Contents

Every rank checkpoint should contain:

| Field | Meaning |
| --- | --- |
| `rank` | active rank \(k\) |
| `lineage_id` | lineage directory identifier |
| `episode_index` | latest completed episode |
| `config` | persisted rank config payload |
| `stats` | latest episode stats |
| `policy_state_dict` | active rank policy parameters |
| `optimizer_state_dict` | active rank optimizer state |
| `artifact_schema_version` | checkpoint schema version |

For \(k>1\), the checkpoint must also contain parent dependency metadata:

| Field | Meaning |
| --- | --- |
| `parent_rank` | \(k-1\) |
| `parent_checkpoint` | relative path to accepted parent checkpoint |
| `parent_checkpoint_id` | stable parent checkpoint identifier if available |
| `parent_config_hash` | hash or digest of parent config if available |
| `parent_artifact_schema_version` | parent checkpoint schema version if available |

The checkpoint should not duplicate the parent policy state dict. It should reference the parent checkpoint.

## Rank Config Contents

Each rank's `config.json` should include:

| Field | Meaning |
| --- | --- |
| `rank` | active rank \(k\) |
| `lineage_id` | lineage identifier |
| `episode_budget` | \(E_k\) |
| `measure_size` | meter measure size |
| `context_measures` | window length in measures |
| `max_step_size` | horizontal step budget |
| `reward_config` | rank-\(k\) reward weights and knobs |
| `graph_config` | rank-\(k\) graph legality/spec knobs |
| `policy_config` | architecture and action-space knobs |
| `training_config` | learning rate, gamma, entropy/exploration knobs |
| `parent_sampler_config` | top-\(m\) parent sampler settings, for \(k>1\) |
| `parent_checkpoint` | parent checkpoint path, for \(k>1\) |

The config should be sufficient to reconstruct the rank-\(k\) training setup, assuming the code version is also known.

## Lineage Manifest

Each lineage directory has:

```text
manifest.json
```

The manifest summarizes the tower lineage.

Minimum structure:

```json
{
  "lineage_id": "...",
  "artifact_schema_version": 1,
  "ranks": {
    "1": {
      "status": "accepted",
      "checkpoint": "rank_1/checkpoint_latest.pt",
      "config": "rank_1/config.json",
      "metrics": "rank_1/metrics.jsonl"
    },
    "2": {
      "status": "accepted",
      "checkpoint": "rank_2/checkpoint_latest.pt",
      "config": "rank_2/config.json",
      "metrics": "rank_2/metrics.jsonl",
      "parent_rank": 1,
      "parent_checkpoint": "rank_1/checkpoint_latest.pt"
    }
  }
}
```

The manifest should make it easy for rank \(k+1\) training to locate the accepted rank-\(k\) checkpoint.

## Accepted Checkpoint Lookup

Stage 10 defined acceptance as completing the configured episode budget:

\[
\text{accepted}(\pi^k)
\quad\Longleftrightarrow\quad
\text{episodes completed}=E_k.
\]

Therefore the accepted checkpoint for rank \(k\) is:

```text
artifacts/tower/<lineage_id>/rank_<k>/checkpoint_latest.pt
```

provided that the manifest marks rank \(k\) as accepted.

Rank \(k+1\) training locates the frozen parent by reading:

```text
artifacts/tower/<lineage_id>/manifest.json
```

and resolving rank \(k\)'s accepted checkpoint path.

## Parent Dependency Rule

For \(k>1\), rank \(k\) depends on exactly one accepted parent checkpoint:

\[
\operatorname{parent}(\pi^k)=\operatorname{checkpoint}(\pi^{k-1}).
\]

This dependency must be recorded in:

1. the rank-\(k\) `config.json`,
2. the rank-\(k\) `checkpoint_latest.pt`,
3. the lineage `manifest.json`.

This is lineage tracking only. The parent checkpoint remains read-only.

## Reproducibility Metadata

Each rank should record enough metadata to make the run understandable and reproducible.

Required or strongly recommended:

| Metadata | Why |
| --- | --- |
| `lineage_id` | ties rank artifacts to tower lineage |
| `rank` | identifies active tier |
| `episode_budget` | explains stop condition |
| `graph_config` | graph legality affects action spaces |
| `reward_config` | reward weights define training objective |
| `training_config` | optimizer and RL parameters |
| `parent_checkpoint` | scaffold dependency for \(k>1\) |
| `parent_sampler_config` | parent rollout behavior for \(k>1\) |
| `seed_config` | reproduce sampling where intended |
| `artifact_schema_version` | guard against future format drift |

If code-version metadata is available, record it:

| Metadata | Why |
| --- | --- |
| `git_commit` | identifies source version |
| `git_dirty` | indicates local modifications |

If this metadata is unavailable, training should still run, but diagnostics should say it is unavailable.

## Metrics

Each rank writes:

```text
metrics.jsonl
```

Each line should be one JSON object for one episode.

Minimum rank-\(k\) episode metrics:

| Metric | Meaning |
| --- | --- |
| `rank` | active rank |
| `episode_index` | episode number |
| `episode_return` | sum of rank-\(k\) rewards |
| `episode_length` | number of steps |
| `mean_step_reward` | average reward |
| `terminated` | terminal success flag |
| `truncated` | deadline/max-step stop flag |
| `loss` | training loss |
| `invalid_extension_count` | invalid lift count |
| `empty_lift_fiber_count` | empty fiber count |
| `parent_failure_count` | parent failure count |
| `terminal_success` | rank-local success flag |

Additional reward-specific diagnostics may be added by rank.

## MIDI Exports

Each rank may export:

```text
example_episode.mid
```

The old system already exports one example MIDI after training. The tower should preserve this behavior rankwise.

The export for rank \(k\) should render the realized rank-\(k\) chord sequence.

If a higher-rank export depends on frozen parent checkpoints, the export metadata should include the same parent checkpoint references as the training run.

## Artifact Status Values

The lineage manifest should track rank status.

Suggested statuses:

| Status | Meaning |
| --- | --- |
| `not_started` | rank directory may not exist yet |
| `running` | rank training is in progress |
| `accepted` | rank completed its episode budget |
| `failed` | rank training failed before acceptance |
| `superseded` | rank was replaced by a new lineage, not mutated in place |

The no-rollback rule means a rank in an accepted lineage should not be silently overwritten to fix later problems.

## Old-System Carryovers

Carry forward:

| Old behavior | Tower decision |
| --- | --- |
| persist config at run start | yes, per rank |
| append metrics as JSONL | yes, per rank |
| save rolling latest checkpoint | yes, per rank |
| checkpoint includes policy state | yes |
| checkpoint includes optimizer state | yes for active rank |
| checkpoint includes latest stats | yes |
| export example MIDI | yes, per rank when useful |
| print artifact paths during training | yes |

## Tower-Only Additions

Add:

| Addition | Why needed |
| --- | --- |
| shared lineage directory | captures no-rollback tower chain |
| rank-specific artifact directories | each rank has distinct config/checkpoint/metrics |
| lineage manifest | lets child ranks find accepted parent checkpoints |
| parent checkpoint metadata | records scaffold dependency |
| parent config hash/digest | guards against wrong parent |
| rank status | tracks whether a parent is accepted |
| invalid-extension metrics | tower-specific rollout diagnostics |
| empty-lift-fiber metrics | detects graph/spec calibration issues |

## Unresolved Choices

These do not block the Stage 12 draft, but should be resolved before implementation:

| Choice | Proposed default |
| --- | --- |
| exact lineage ID format | timestamp or user-provided slug |
| exact config hash function | implementation-stage decision |
| whether to save archival checkpoints | no by default |
| whether every rank must export MIDI | recommended but not mandatory |
| whether manifest updates are atomic | yes if practical |
| whether dirty git state blocks training | no, but record it |

## Stage 12 Completion Checklist

Stage 12 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| tower artifacts live under `artifacts/tower/` | yes |
| each training chain uses a shared lineage directory | yes |
| each rank has its own `rank_<k>/` directory | yes |
| each rank writes `config.json` | yes |
| each rank writes `metrics.jsonl` | yes |
| each rank writes rolling `checkpoint_latest.pt` | yes |
| each rank may write `example_episode.mid` | yes |
| checkpoints record rank and lineage ID | yes |
| checkpoints for \(k>1\) record parent checkpoint dependency | yes |
| parent checkpoint state is referenced, not duplicated | yes |
| lineage manifest records accepted checkpoints | yes |
| rank \(k+1\) locates parent via manifest | yes |
| no-rollback lineage is preserved | yes |

Once accepted, Phase 4 is complete. The next phase is Phase 5: Freeze System Architecture.
