# Training Runner Contract

This document is the Post-Slice-8 Phase 1 / Stage 1.1 / Action 1.1.1
deliverable.

The purpose is to specify the rank training lifecycle that the eventual tower
training runner must execute.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 1: Freeze Next-Build Contracts |
| Stage | Stage 1.1: Runner Lifecycle Contract |
| Action | Action 1.1.1: Specify Rank Training Lifecycle |

Action 1.1.1 exit criterion:

| Requirement | Status |
| --- | --- |
| rank-1 training lifecycle recorded | drafted here |
| rank-2 training lifecycle recorded | drafted here |
| final no-train inference requirement recorded | drafted here |
| MIDI artifact requirement recorded | drafted here |
| checkpoint/manifest evidence requirement recorded | drafted here |
| parent sampling default recorded | drafted here |

## Source Authority

This contract is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/training_protocol.md` | rankwise training order and freeze rule |
| `docs/design/tower/artifact_checkpoint_dependencies.md` | artifact layout and parent dependency metadata |
| `docs/design/tower/post_slice_8_build_plan.md` | accepted post-Slice-8 decisions |
| `docs/design/tower/post_slice_8_phase_stage_action_plan.md` | current Phase.Stage.Action plan |
| `docs/design/tower/post_slice_8_questions.md` | owner answers; do not modify unless reopened |

## Runner Purpose

The training runner is the orchestration layer that turns existing lower-level
tower training helpers into a reproducible rank training run.

It must eventually coordinate:

```text
config
-> policy construction
-> optimizer construction
-> episode training loop
-> metrics/config/checkpoint/manifest artifacts
-> final no-train inference episode
-> final MIDI artifact
```

The runner is not the graph core, reward core, policy architecture, or MIDI
writer itself. It wires those pieces together.

## Non-Goals

The first runner should not attempt to solve the entire experiment-management
problem.

Out of scope:

| Out of scope | Reason |
| --- | --- |
| archival checkpoint per episode | current artifact contract uses rolling latest checkpoint |
| multiple accepted parent checkpoints per rank-2 run | owner selected one accepted rank-1 parent checkpoint |
| automatic smart promotion metrics | owner selected episode count for now |
| rich validation suite | first milestone is training evidence plus final MIDI |
| rank-k generalization | rank-1/rank-2 first |
| reward expansion | later pass, using `assets/rules/tc21m_rules.md` |
| suspensions/style variants | later project update such as `beta.1` |

## Shared Lifecycle

Every rank training run follows the same high-level lifecycle:

```text
1. create or load rank config
2. create deterministic artifact paths
3. construct active rank policy
4. construct active rank optimizer
5. if rank > 1, load accepted parent checkpoint and freeze parent policy
6. train for configured episode count
7. write config, metrics, latest checkpoint, and manifest updates
8. run one final no-train inference episode
9. write final inference episode to MIDI
10. record final cadence/MIDI evidence in metrics or summary artifacts
```

The runner should preserve the existing rank-local training helpers rather than
replacing them. Existing lower-level helpers already prove:

| Helper area | Existing module |
| --- | --- |
| rank config | `tower/train/config.py` |
| artifact paths/checkpoints | `tower/train/checkpoint.py` |
| rank-1/rank-2 training episodes | `tower/train/protocol.py` |
| rollout | `tower/train/rollout.py` |
| policy loss | `tower/train/losses.py` |

## Rank-1 Training Lifecycle

Rank 1 trains without a parent policy.

Required lifecycle:

| Step | Requirement |
| --- | --- |
| config | rank is 1 |
| parent checkpoint | absent |
| policy | active rank-1 policy is trainable |
| optimizer | active rank-1 optimizer is trainable |
| stop condition | configured episode count |
| metrics | append per-episode metrics |
| checkpoint | rolling `rank_1/checkpoint_latest.pt` |
| manifest | record rank 1 status |
| final inference | run one no-train episode after training |
| final MIDI | write final inference trajectory to `rank_1/example_episode.mid` or configured equivalent |
| final evidence | record cadence success and MIDI path |

For now, rank-1 checkpoint acceptance may remain episode-budget based, but the
runner must record evidence that will make smarter acceptance possible later.

Evidence to record:

| Evidence | Purpose |
| --- | --- |
| final inference return | inspect final run quality |
| final cadence success | rank-local musical target evidence |
| final MIDI path | human inspection artifact |
| final episode length | basic rollout sanity |
| invalid/no-op counts | training/legality diagnostics |

## Rank-2 Training Lifecycle

Rank 2 trains over one accepted rank-1 parent checkpoint.

Required lifecycle:

| Step | Requirement |
| --- | --- |
| config | rank is 2 |
| parent checkpoint | exactly one accepted rank-1 checkpoint |
| parent policy | loaded from parent checkpoint and frozen |
| parent optimizer | not used for training |
| active child policy | rank-2 policy is trainable |
| child optimizer | rank-2 optimizer is trainable |
| parent sampling | frozen parent samples top-m legal parent actions |
| `parent_top_m` | real-training default is 3 |
| child action | constrained through lift fiber over sampled parent action |
| stop condition | configured episode count |
| metrics | append per-episode rank-2 metrics |
| checkpoint | rolling `rank_2/checkpoint_latest.pt` |
| manifest | record rank-2 status and parent checkpoint dependency |
| final inference | run one no-train rank-2 episode after training |
| final MIDI | write final inference trajectory to `rank_2/example_episode.mid` or configured equivalent |
| final evidence | record cadence success and MIDI path |

The runner must not duplicate the parent policy state inside the rank-2
checkpoint. The rank-2 checkpoint references the parent checkpoint.

Parent safety invariants:

| Invariant | Requirement |
| --- | --- |
| parent checkpoint | read-only |
| parent params | unchanged |
| parent optimizer | not stepped |
| parent logprobs | diagnostic only |
| parent logits/distribution | not child model input |
| top-m candidates | not child model input |

The child "knows" the sampled parent action through the lift-fiber constrained
candidate set, not through rich parent-policy distribution features.

## Final No-Train Inference Episode

Every completed training run must end with exactly one final no-train inference
episode.

Required behavior:

| Behavior | Requirement |
| --- | --- |
| no optimizer step | yes |
| no gradient accumulation | yes |
| policy mode | evaluation/no-train mode |
| trajectory | returned or persisted for artifact export |
| reward context | same rank-local reward contract |
| cadence evidence | recorded |
| MIDI export | generated from this final trajectory |

This final episode is not a training episode. It is an inspection/evidence
episode.

The owner does not require continuous MIDI export during training. The required
artifact is the final no-train inference MIDI.

## Artifact Contract

The runner must use the tower artifact layout:

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
```

Required rank artifacts:

| Artifact | Requirement |
| --- | --- |
| `config.json` | write rank config |
| `metrics.jsonl` | append per-episode metrics |
| `checkpoint_latest.pt` | save rolling active-rank checkpoint |
| `manifest.json` | update lineage status and parent dependency |
| `example_episode.mid` | write final no-train inference MIDI |

The runner may later add additional summary artifacts, but it must not require
them for the first implementation.

## Checkpoint Status And Acceptance Evidence

The current stop condition is episode count.

For now:

| Concept | Near-term rule |
| --- | --- |
| training stop | configured episode count |
| checkpoint status | may become accepted when episode budget is reached |
| musical evidence | final cadence success recorded |
| human inspection | final MIDI path recorded |
| smarter promotion | deferred until real training evidence exists |

Important distinction:

```text
Episode count can stop training.
Cadence/MIDI evidence tells us whether the result is musically promising.
```

The runner should record both, even if acceptance status remains budget-based
initially.

## Parent Sampling Contract

For rank 2, parent sampling uses the frozen accepted rank-1 policy.

Required configuration:

| Setting | Requirement |
| --- | --- |
| `parent_top_m` | config hyperparameter |
| real-training default | 3 |
| test default | may use 1 for determinism |
| parent actions | legal rank-1 actions only |
| sampling set | top-m legal parent actions |

The parent sampler must remain a controlled scaffold. It must not perform broad
uniform exploration.

## Required Runner Outputs

At minimum, a successful rank run should make these facts inspectable:

| Output | Required evidence |
| --- | --- |
| run directory | printed or returned |
| config path | written |
| metrics path | written |
| checkpoint path | written |
| manifest path | written |
| final MIDI path | printed or returned |
| final cadence success | recorded in metrics/summary |

## Expected Future Implementation Files

This contract does not approve these files yet, but the expected runner
implementation path is:

| Area | Expected file |
| --- | --- |
| runner orchestration | `tower/train/runner.py` |
| runner tests | `tests/tower/train/test_runner.py` |
| script entrypoint | `scripts/tower_train.py` |
| script tests | `tests/tower/train/test_runner_script.py` |
| MIDI rendering | `tower/music/render.py` |
| MIDI tests | `tests/tower/music/test_render.py` |

The next Phase.Stage.Action should still be followed before these files are
created or edited.

## Stop Conditions

Pause and resynchronize if:

| Stop condition | Why |
| --- | --- |
| runner lifecycle conflicts with current artifact code | reality break |
| final inference would require training-only APIs | lifecycle mismatch |
| rank-2 parent checkpoint would be mutated | violates freeze/artifact contract |
| rank-2 child would require parent logits/logprobs as input | violates owner decision |
| acceptance cannot be separated from episode count | design ambiguity |
| MIDI export requires importing `rl_counterpoint` at runtime | violates tower boundary |

## Next Phase.Stage.Action

After this contract is accepted, the next proposed action is:

```text
Post-Slice-8 Phase 1.Stage 1.2.Action 1.2.1:
Specify Tower Encoded Window Contract
```

That action would create:

```text
docs/design/tower/real_policy_observation_contract.md
```

No code implementation is approved by this document.
