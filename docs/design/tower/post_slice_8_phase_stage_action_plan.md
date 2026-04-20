# Post-Slice-8 Phase.Stage.Action Plan

This document turns the accepted post-Slice-8 assessment into an explicit
Phase.Stage.Action plan for the next tower build.

This is a planning document, not implementation approval.

## Source

This plan is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/post_slice_8_build_plan.md` | accepted post-Slice-8 assessment baseline |
| `docs/design/tower/post_slice_8_questions.md` | owner answers to post-Slice-8 planning questions |
| `docs/design/tower/build_plan.md` | completed Slice 1-8 implementation plan |
| `assets/rules/tc21m_rules.md` | future reward/shaping source |

## Build Goal

The next big build should produce the first real tower training path:

```text
rank-local tower window
-> model-ready observation tensor
-> transformer-family rank policy
-> artifact-backed training runner
-> final no-train inference episode
-> MIDI artifact
```

The build should prove that training is working, not merely that weak smoke tests
execute.

## Owner Decisions Incorporated

| Topic | Decision |
| --- | --- |
| main priority | learnable real policy architecture |
| architecture family | shared transformer-family policy across ranks |
| rank differences | config-driven tensor/action differences |
| observation source | adapt old `rl_counterpoint` timed-window transformer pattern |
| child access to parent | constrain through lift fiber, do not feed parent logits/logprobs/top-m candidates |
| rank-2 parent checkpoint | one accepted rank-1 checkpoint |
| rank-2 parent sampling | frozen parent policy, top-m action sampling |
| real-training `parent_top_m` | 3 |
| training stop condition for now | episode count |
| final evaluation | one no-train inference episode |
| final artifact | MIDI file saved as training artifact |
| reward expansion | do not block first real-policy path; draw later from `assets/rules/tc21m_rules.md` |
| suspensions | defer to later style update such as `beta.1` |

## Global Guardrails

These apply to every action in this plan.

| Guardrail | Requirement |
| --- | --- |
| owner approval | discuss before each action; implement only after owner approval |
| old project boundary | do not modify files under `rl_counterpoint/` |
| imports | do not import `rl_counterpoint/` from `tower/` runtime code |
| reuse | old code may be read and copied into tower-owned modules |
| representation | keep tuple state/action as graph core |
| model tensors | tensorization belongs in policy/observation layer, not graph core |
| parent model | rank-k child training uses frozen rank-(k-1) parent |
| parent information | parent logits/logprobs/top-m candidates remain diagnostics, not child inputs |
| tests | every implementation action gets focused tests |
| failure handling | unexpected failures pause the game plan for investigation |

## Phase 1: Freeze Next-Build Contracts

Purpose:

Define the exact contracts needed before model and runner implementation.

Exit criterion:

The owner accepts the files/functions/tests to build in Phases 2-5.

### Stage 1.1: Runner Lifecycle Contract

Goal:

Define the training lifecycle that the eventual runner must execute.

#### Action 1.1.1: Specify Rank Training Lifecycle

Files:

| File | Work |
| --- | --- |
| `docs/design/tower/training_runner_contract.md` | create lifecycle contract |

Required content:

| Item | Decision to record |
| --- | --- |
| rank-1 training | episode-count stop condition |
| rank-2 training | one accepted rank-1 parent checkpoint |
| final evaluation | one no-train inference episode after training |
| final artifact | MIDI written from final inference trajectory |
| checkpoint status | record cadence/MIDI evidence even if acceptance is episode-budget based |
| rank-2 parent sampling | `parent_top_m`, real default 3 |

Tests:

None. Documentation action only.

Stop if:

| Stop condition |
| --- |
| lifecycle conflicts with existing artifact/checkpoint design |
| owner wants acceptance smarter than episode count now |

### Stage 1.2: Observation Contract Plan

Goal:

Define the tower-local observation contract before writing model code.

#### Action 1.2.1: Specify Tower Encoded Window Contract

Files:

| File | Work |
| --- | --- |
| `docs/design/tower/real_policy_observation_contract.md` | create model-observation contract |

Required content:

| Item | Decision to record |
| --- | --- |
| source pattern | old timed-window transformer policy |
| tower input | `TowerWindow` plus context |
| padding | use valid mask |
| meter | expose bar/beat information to model input |
| tonic/key | expose key context |
| target | expose target context where configured |
| rank | config-driven state width |
| output | active-choice logits before external masking |

Tests:

None. Documentation action only.

Stop if:

| Stop condition |
| --- |
| observation design requires canonical graph tensors |
| observation design requires child policy to consume parent logits/logprobs |

### Stage 1.3: Implementation Cut Lines

Goal:

Decide exact files for implementation phases before code changes begin.

#### Action 1.3.1: Produce File-Level Build Map

Files:

| File | Work |
| --- | --- |
| `docs/design/tower/post_slice_8_file_map.md` | create file/test map for Phases 2-5 |

Required content:

| Area | Expected files |
| --- | --- |
| observation | `tower/policy/observation.py`, `tests/tower/policy/test_observation.py` |
| transformer policy | `tower/policy/transformer.py`, `tests/tower/policy/test_transformer.py` |
| MIDI render/export | `tower/music/render.py`, `tests/tower/music/test_render.py` |
| runner | `tower/train/runner.py`, `tests/tower/train/test_runner.py` |
| script/CLI | likely `scripts/tower_train.py`, `tests/tower/train/test_runner_script.py` |

Tests:

None. Documentation action only.

Stop if:

| Stop condition |
| --- |
| proposed files import from frozen `rl_counterpoint/` |
| proposed files blur graph core with tensor/model code |

## Phase 2: Real Policy Observation Contract

Purpose:

Create tower-local tensorization for rank-local model input.

Exit criterion:

A tower window can be converted into a model-ready observation object with tests
covering padding, context, rank width, and shape validation.

### Stage 2.1: Observation Data Structures

Goal:

Add explicit tower model-observation objects.

#### Action 2.1.1: Add Encoded Tower Window Shell

Files:

| File | Work |
| --- | --- |
| `tower/policy/observation.py` | add encoded observation dataclass/protocol |
| `tests/tower/policy/test_observation.py` | add construction/validation tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| event tensor | rank-local sequence tensor |
| valid mask | boolean padding mask |
| bar positions | exposed as tensor or explicit metadata |
| rank | carried or inferable |
| context | tonic/key/target fields represented |
| validation | reject shape mismatches |

Run:

```bash
uv run pytest tests/tower/policy/test_observation.py
```

### Stage 2.2: Window-To-Tensor Encoding

Goal:

Convert `TowerWindow` to the encoded observation shell.

#### Action 2.2.1: Implement Tower Window Encoder

Files:

| File | Work |
| --- | --- |
| `tower/policy/observation.py` | add `encode_tower_window(...)` |
| `tests/tower/policy/test_observation.py` | add encoding tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| tuple states | encode rank-local state values |
| padding | valid mask preserves padded positions |
| meter | bar positions included |
| tonic/key | context included |
| target | optional target included |
| rank variability | rank 1 and rank 2 both work |

Run:

```bash
uv run pytest tests/tower/policy/test_observation.py
```

Stop if:

| Stop condition |
| --- |
| encoding requires changing `TowerWindow` graph contract |
| encoding drops padding/meter/context needed by the old model pattern |

## Phase 3: Transformer-Family Rank Policy

Purpose:

Implement the first real rank-local policy architecture behind the existing
`RankPolicy` protocol.

Exit criterion:

A transformer-family tower policy maps encoded tower observations to active
choice logits for rank 1 and rank 2.

### Stage 3.1: Policy Config

Goal:

Represent transformer hyperparameters and rank-specific action dimensions.

#### Action 3.1.1: Add Policy Config

Files:

| File | Work |
| --- | --- |
| `tower/policy/transformer.py` | add config dataclass |
| `tests/tower/policy/test_transformer.py` | add config validation tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| d_model/layers/heads/ff/dropout | validated |
| max window length | validated |
| rank | validated |
| action/choice dimension | config-driven |
| invalid configs | raise clear errors |

Run:

```bash
uv run pytest tests/tower/policy/test_transformer.py
```

### Stage 3.2: Transformer Policy Module

Goal:

Add a tower-owned transformer encoder policy.

#### Action 3.2.1: Implement Rank Transformer Policy

Files:

| File | Work |
| --- | --- |
| `tower/policy/transformer.py` | add transformer policy class |
| `tests/tower/policy/test_transformer.py` | add forward/shape tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| accepts encoded observation | yes |
| uses valid mask | yes |
| selects final valid state | yes |
| returns `PolicyOutput` | yes |
| output logits dimension | active choice/action dimension |
| rank attribute | satisfies `RankPolicy` |

Run:

```bash
uv run pytest tests/tower/policy/test_transformer.py
```

Stop if:

| Stop condition |
| --- |
| policy API conflicts with existing sampler API |
| model requires parent diagnostic inputs |

### Stage 3.3: Frontier Contract And Sampler Integration

Goal:

Resolve the discovered policy API ambiguity, then use the real policy through
existing active and parent samplers.

#### Action 3.3.0: Clarify Policy Frontier/Window Contract

Files:

| File | Work |
| --- | --- |
| `docs/design/tower/policy_frontier_window_contract.md` | create policy frontier/window contract |

Required content:

| Item | Decision to record |
| --- | --- |
| policy input | window-primary / encoded-window-primary |
| frontier state | final valid state of the window |
| independent state input | not part of model input |
| rollout state | may remain local graph/legality/application state |
| sampler responsibility | derive frontier from window or encoded window where needed |
| transformer policy | consumes `EncodedTowerWindow` directly |
| existing protocol | decide adapter/refactor path for `RankPolicy` and samplers |

Tests:

None. Documentation/design-correction action only.

Stop if:

| Stop condition |
| --- |
| the correction implies parent logits/logprobs/top-m candidates as model input |
| the correction requires changing graph/window tuple contracts |
| the correction makes rollout state and window frontier intentionally divergent |

#### Action 3.3.1: Verify Real Policy Sampler Compatibility

Files:

| File | Work |
| --- | --- |
| `tests/tower/policy/test_samplers.py` | add transformer-backed sampler tests |
| `tests/tower/train/test_protocol.py` | add minimal transformer-backed training episode if cheap |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| active sampler | handles transformer policy output |
| parent top-m sampler | handles transformer policy output |
| logprobs | remain differentiable for active child |
| parent logprobs | remain detached/diagnostic in rank-2 training |

Run:

```bash
uv run pytest tests/tower/policy tests/tower/train/test_protocol.py
```

## Phase 4: Tower MIDI Artifact

Purpose:

Write tower trajectories to MIDI so final no-train inference episodes are
inspectable.

Exit criterion:

A tower trajectory can be exported to a MIDI file without importing
`rl_counterpoint`.

### Stage 4.1: Tower MIDI Render Utility

Goal:

Copy/adapt old MIDI rendering into tower-owned code.

#### Action 4.1.1: Implement Tower MIDI Writer

Files:

| File | Work |
| --- | --- |
| `tower/music/__init__.py` | package marker |
| `tower/music/render.py` | tower-owned MIDI render helpers |
| `tests/tower/music/test_render.py` | MIDI writer tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| chord/state sequence | writes MIDI |
| MIDI range | validates 0-127 |
| deterministic output | stable enough for tests |
| no old imports | no `rl_counterpoint` runtime import |

Run:

```bash
uv run pytest tests/tower/music/test_render.py tests/tower/test_import_boundaries.py
```

### Stage 4.2: Trajectory-To-MIDI Export

Goal:

Convert final inference trajectory states into a MIDI artifact.

#### Action 4.2.1: Add Tower Trajectory MIDI Export

Files:

| File | Work |
| --- | --- |
| `tower/music/render.py` | add trajectory export helper |
| `tests/tower/music/test_render.py` | add trajectory export tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| initial state included | yes |
| subsequent target states included | yes |
| final path | writes to artifact path |
| rank 1 and rank 2 | both supported if states are MIDI tuples |

Run:

```bash
uv run pytest tests/tower/music/test_render.py
```

## Phase 5: Training Runner

Purpose:

Create the first artifact-backed tower training runner.

Exit criterion:

The runner can train rank 1 for a small episode count, run a final no-train
inference episode, save metrics/checkpoint/config/manifest, and write final MIDI.
Rank-2 runner support should either work over one accepted rank-1 checkpoint or
be explicitly gated to the next action.

### Stage 5.1: Runner Config

Goal:

Represent run-level settings separate from rank config internals.

#### Action 5.1.1: Add Runner Config

Files:

| File | Work |
| --- | --- |
| `tower/train/runner.py` | add runner config dataclass |
| `tests/tower/train/test_runner.py` | add config tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| lineage id | configured |
| rank | configured |
| episode count | configured |
| seed | configured |
| artifact root | configured |
| parent checkpoint | rank 2 only |
| parent_top_m | default 3 for real rank-2 configs |
| final MIDI export | enabled |

Run:

```bash
uv run pytest tests/tower/train/test_runner.py
```

### Stage 5.2: Final Inference Episode

Goal:

Run one no-train episode after training using the learned policy.

#### Action 5.2.1: Add Final Inference Helper

Files:

| File | Work |
| --- | --- |
| `tower/train/runner.py` | add no-train inference helper |
| `tests/tower/train/test_runner.py` | add no-gradient/no-optimizer tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| no optimizer step | yes |
| no gradient accumulation | yes |
| policy eval mode | yes |
| trajectory returned | yes |
| metrics returned | cadence success recorded |

Run:

```bash
uv run pytest tests/tower/train/test_runner.py
```

### Stage 5.3: Rank-1 Runner

Goal:

Train rank 1 with a real policy and artifacts.

#### Action 5.3.1: Implement Rank-1 Training Runner

Files:

| File | Work |
| --- | --- |
| `tower/train/runner.py` | add rank-1 run function |
| `tests/tower/train/test_runner.py` | add rank-1 runner tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| builds policy | transformer-family policy |
| trains N episodes | episode count from config |
| writes artifacts | config, metrics, latest checkpoint, manifest |
| final inference | after training |
| MIDI artifact | final inference episode exported |
| cadence evidence | final success metric recorded |

Run:

```bash
uv run pytest tests/tower/train/test_runner.py tests/tower/policy tests/tower/music
```

Stop if:

| Stop condition |
| --- |
| runner cannot prove optimizer changes policy |
| final inference accidentally trains |
| MIDI artifact is not generated |

### Stage 5.4: Rank-2 Runner

Goal:

Train rank 2 over one accepted frozen rank-1 checkpoint.

#### Action 5.4.1: Implement Rank-2 Training Runner

Files:

| File | Work |
| --- | --- |
| `tower/train/runner.py` | add rank-2 run function |
| `tests/tower/train/test_runner.py` | add rank-2 runner tests |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| loads accepted parent checkpoint | yes |
| freezes parent policy | yes |
| samples parent top-m actions | `parent_top_m`, default 3 |
| trains child only | yes |
| records parent dependency | checkpoint and manifest |
| final inference | yes |
| MIDI artifact | final rank-2 inference episode exported |

Run:

```bash
uv run pytest tests/tower/train/test_runner.py tests/tower/train/test_protocol.py
```

Stop if:

| Stop condition |
| --- |
| parent checkpoint is mutated |
| parent optimizer or params change |
| rank-2 child receives parent logits/logprobs as inputs |

### Stage 5.5: Script Entrypoint

Goal:

Expose the runner as a repo script.

#### Action 5.5.1: Add Tower Training Script

Files:

| File | Work |
| --- | --- |
| `scripts/tower_train.py` | add script entrypoint |
| `tests/tower/train/test_runner_script.py` | add direct script execution test |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| direct file execution | imports project from repo root |
| small config | can run a tiny test config |
| prints run dir | yes |
| prints final MIDI path | yes |
| writes artifacts | yes |

Run:

```bash
uv run pytest tests/tower/train/test_runner_script.py
```

## Phase 6: First Training Evidence

Purpose:

Run the first small end-to-end tower training evidence pass.

Exit criterion:

The repo contains a reproducible command and artifact output showing that a real
tower policy trains, performs final no-train inference, and writes MIDI.

### Stage 6.1: Focused Verification

#### Action 6.1.1: Run Focused Test Suite

Run:

```bash
uv run pytest tests/tower/policy tests/tower/train tests/tower/music
```

Pass condition:

All focused tests pass.

### Stage 6.2: Full Verification

#### Action 6.2.1: Run Full Test Suite

Run:

```bash
uv run pytest
```

Pass condition:

All repo tests pass.

### Stage 6.3: Tiny Training Run

#### Action 6.3.1: Run Tiny Rank-1 Training Command

Run:

```bash
uv run python scripts/tower_train.py --rank 1 --episodes <small-number>
```

Expected evidence:

| Artifact | Requirement |
| --- | --- |
| config | written |
| metrics | written |
| latest checkpoint | written |
| manifest | written |
| final MIDI | written |
| stdout | prints run dir and final MIDI path |

Stop if:

| Stop condition |
| --- |
| training does not update policy |
| final inference trains |
| final MIDI is missing |
| artifacts are not reproducible enough to inspect |

### Stage 6.4: Closeout

#### Action 6.4.1: Write Training Evidence Report

Files:

| File | Work |
| --- | --- |
| `docs/design/tower/post_slice_8_training_evidence.md` | summarize test/run evidence |

Required content:

| Section | Content |
| --- | --- |
| command run | exact command |
| artifact paths | config/metrics/checkpoint/manifest/MIDI |
| observed metrics | episode count, return, final success evidence |
| limitations | what is still skeletal |
| next recommendation | likely reward expansion or rank-2 real run |

## Phase 7: Reward Expansion Planning

Purpose:

Plan, not necessarily implement, the first TC21M-derived reward expansion after
real training evidence exists.

Exit criterion:

The owner accepts a narrow reward-expansion slice.

### Stage 7.1: TC21M Reward Triage

#### Action 7.1.1: Produce Reward Expansion Plan

Files:

| File | Work |
| --- | --- |
| `docs/design/tower/post_slice_8_reward_expansion_plan.md` | create narrow reward plan |

Candidate areas:

| Area | Initial stance |
| --- | --- |
| cadence rules | include |
| motion rewards | include narrow subset |
| dissonance treatment | include narrow subset |
| voice-leading constraints | decide graph vs reward boundary |
| harmonic templates | consider after observation/policy evidence |
| suspensions | defer to `beta.1` |
| six-four logic | defer |

## Overall Dependency Order

The dependency graph is:

```text
Phase 1 contracts
  -> Phase 2 observation
  -> Phase 3 transformer policy
  -> Phase 4 MIDI export
  -> Phase 5 runner
  -> Phase 6 evidence
  -> Phase 7 reward expansion planning
```

Phase 4 can be implemented before Phase 3 if desired, because MIDI export uses
trajectories rather than model internals. Phase 5 should not be implemented
before Phases 2 and 3 are accepted, unless the runner is explicitly built with a
temporary toy policy only.

## Immediate Next Action

The next proposed action is:

```text
Post-Slice-8 Phase 1.Stage 1.1.Action 1.1.1:
Specify Rank Training Lifecycle
```

Owner approval needed before creating:

```text
docs/design/tower/training_runner_contract.md
```

## Non-Approval Statement

This document does not approve implementation of any code.

Each Phase.Stage.Action still requires discussion and owner approval before
implementation.
