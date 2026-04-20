# Post-Slice-8 File-Level Build Map

This document is the Post-Slice-8 Phase 1 / Stage 1.3 / Action 1.3.1
deliverable.

The purpose is to define exact source and test file cut lines for the next
implementation phases before code work begins.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 1: Freeze Next-Build Contracts |
| Stage | Stage 1.3: Implementation Cut Lines |
| Action | Action 1.3.1: Produce File-Level Build Map |

Action 1.3.1 exit criterion:

| Requirement | Status |
| --- | --- |
| observation files mapped | drafted here |
| transformer policy files mapped | drafted here |
| MIDI render/export files mapped | drafted here |
| runner files mapped | drafted here |
| script/CLI files mapped | drafted here |
| layer boundaries recorded | drafted here |

## Source Authority

This map is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/post_slice_8_phase_stage_action_plan.md` | current explicit plan |
| `docs/design/tower/training_runner_contract.md` | runner lifecycle contract |
| `docs/design/tower/real_policy_observation_contract.md` | model-observation contract |
| `docs/design/tower/post_slice_8_build_plan.md` | accepted post-Slice-8 baseline |
| `docs/design/tower/post_slice_8_questions.md` | owner decisions; read-only unless reopened |

## Global File Rules

These rules apply to all files in this map.

| Rule | Requirement |
| --- | --- |
| old project boundary | no runtime imports from `rl_counterpoint` in `tower/` |
| graph core | keep tuple state/action/window contracts in graph/core modules |
| model tensors | tensorization belongs under `tower/policy/` |
| MIDI | tower-owned rendering belongs under `tower/music/` |
| runner | orchestration belongs under `tower/train/runner.py` |
| scripts | user-facing command entrypoint belongs under `scripts/` |
| tests | focused tests live under matching `tests/tower/` subtrees |
| approval | no file in this map is approved for implementation until its action is approved |

## Existing Source Baseline

Current tower source groups:

| Area | Existing files |
| --- | --- |
| core | `tower/state_action.py`, `tower/window.py` |
| graph | `tower/graph/projection.py`, `tower/graph/spec.py`, `tower/graph/legality.py`, `tower/graph/actions.py` |
| action | `tower/action/assembly.py` |
| reward | `tower/reward/context.py`, `tower/reward/result.py`, `tower/reward/success.py`, `tower/reward/terms.py` |
| policy | `tower/policy/base.py`, `tower/policy/samplers.py` |
| train | `tower/train/config.py`, `tower/train/checkpoint.py`, `tower/train/trajectory.py`, `tower/train/rollout.py`, `tower/train/losses.py`, `tower/train/protocol.py` |

Current tower test groups:

| Area | Existing tests |
| --- | --- |
| core | `tests/tower/test_state_action.py`, `tests/tower/test_window.py` |
| graph/action | `tests/tower/graph/*`, `tests/tower/action/test_assembly.py` |
| reward | `tests/tower/reward/*` |
| policy | `tests/tower/policy/test_base.py`, `tests/tower/policy/test_samplers.py` |
| train | `tests/tower/train/*` |
| boundary | `tests/tower/test_import_boundaries.py` |

## Phase 2 File Map: Real Policy Observation Contract

### Source Files

| File | Status | Ownership |
| --- | --- | --- |
| `tower/policy/observation.py` | new | policy-layer tensorization |

`tower/policy/observation.py` owns:

| Object/function | Purpose |
| --- | --- |
| `EncodedTowerWindow` or equivalent | transformer-ready rank-local observation object |
| observation context dataclass if needed | key/meter/target/rank context |
| `encode_tower_window(...)` | convert `TowerWindow` plus context into encoded observation |
| validation helpers | reject mismatched shapes/lengths/context |

Allowed imports:

| Import source | Reason |
| --- | --- |
| `tower.window` | consume `TowerWindow`, pad/meter constants |
| `tower.state_action` | rank/state validation |
| `torch` | tensor construction |
| standard library | dataclasses/typing |

Forbidden imports:

| Import source | Reason |
| --- | --- |
| `rl_counterpoint` | frozen project boundary |
| `tower.train` | observation layer must not depend on training orchestration |
| `tower.graph.actions` | masking belongs outside observation encoding |

### Test Files

| File | Status | Purpose |
| --- | --- | --- |
| `tests/tower/policy/test_observation.py` | new | focused observation encoding tests |

Required test coverage:

| Test | Proves |
| --- | --- |
| construct encoded observation | object contract works |
| reject mismatched sequence lengths | validation works |
| preserve valid mask | padding contract works |
| encode rank-1 window | rank-1 width works |
| encode rank-2 window | rank-2 width works |
| expose meter context | bar positions/measure size available |
| expose key/target context | musical goal context available |
| no old imports | covered with import-boundary test |

Focused verification:

```bash
uv run pytest tests/tower/policy/test_observation.py tests/tower/test_import_boundaries.py
```

## Phase 3 File Map: Transformer-Family Rank Policy

### Source Files

| File | Status | Ownership |
| --- | --- | --- |
| `tower/policy/transformer.py` | new | real transformer-family rank policy |

`tower/policy/transformer.py` owns:

| Object/function | Purpose |
| --- | --- |
| policy config dataclass | transformer/rank/action-dim hyperparameters |
| positional encoding helper if needed | sequence modeling support |
| rank transformer policy class | maps encoded observation to `PolicyOutput` |
| shape validation | model-level input validation |

Allowed imports:

| Import source | Reason |
| --- | --- |
| `tower.policy.base` | `PolicyOutput`, `RankPolicy` compatibility |
| `tower.policy.observation` | encoded observation input |
| `tower.state_action` | rank validation if needed |
| `torch` | model implementation |
| standard library | dataclasses/typing |

Forbidden imports:

| Import source | Reason |
| --- | --- |
| `rl_counterpoint` | frozen project boundary |
| `tower.train.runner` | policy must not depend on runner |
| `tower.train.protocol` | policy must remain training-loop agnostic |
| `tower.graph.actions` | external masking stays in sampler/rollout layer |

### Test Files

| File | Status | Purpose |
| --- | --- | --- |
| `tests/tower/policy/test_transformer.py` | new | transformer config/forward tests |
| `tests/tower/policy/test_samplers.py` | existing | add real-policy sampler integration tests |
| `tests/tower/train/test_protocol.py` | existing | optionally add cheap real-policy protocol smoke test |

Required test coverage:

| Test | Proves |
| --- | --- |
| config validation | invalid hyperparameters rejected |
| forward rank-1 observation | logits shape correct |
| forward rank-2 observation | logits shape correct |
| valid mask use | padding does not become final valid event |
| returns `PolicyOutput` | sampler compatibility |
| active sampler integration | active logprob remains differentiable |
| parent sampler integration | parent logprob remains diagnostic/detached |

Focused verification:

```bash
uv run pytest tests/tower/policy tests/tower/train/test_protocol.py
```

## Phase 4 File Map: Tower MIDI Artifact

### Source Files

| File | Status | Ownership |
| --- | --- | --- |
| `tower/music/__init__.py` | new | package marker |
| `tower/music/render.py` | new | tower-owned MIDI rendering/export |

`tower/music/render.py` owns:

| Object/function | Purpose |
| --- | --- |
| pitch/name helpers if needed | tower-owned copy of display helpers |
| chord/state sequence MIDI writer | write MIDI from tower state tuples |
| trajectory export helper | convert final inference trajectory to MIDI |
| MIDI validation helpers | validate 0-127 pitch range |

Allowed imports:

| Import source | Reason |
| --- | --- |
| `tower.state_action` | state validation |
| `tower.train.trajectory` | trajectory export helper |
| standard library | bytes/path writing |

Forbidden imports:

| Import source | Reason |
| --- | --- |
| `rl_counterpoint.music.render` | copy/adapt only, no runtime import |
| `tower.policy` | MIDI export must not depend on model internals |
| `tower.train.runner` | render utility should be reusable by runner |

### Test Files

| File | Status | Purpose |
| --- | --- | --- |
| `tests/tower/music/test_render.py` | new | MIDI rendering/export tests |

Required test coverage:

| Test | Proves |
| --- | --- |
| write state sequence MIDI | basic file output works |
| reject out-of-range MIDI pitch | validation works |
| deterministic header/body basics | output is inspectable/testable |
| export trajectory MIDI | final trajectory can become artifact |
| include initial state | exported passage starts correctly |
| no old imports | boundary preserved |

Focused verification:

```bash
uv run pytest tests/tower/music/test_render.py tests/tower/test_import_boundaries.py
```

## Phase 5 File Map: Training Runner

### Source Files

| File | Status | Ownership |
| --- | --- | --- |
| `tower/train/runner.py` | new | runner orchestration |

`tower/train/runner.py` owns:

| Object/function | Purpose |
| --- | --- |
| runner config dataclass | run-level settings |
| policy/optimizer factory hooks | construct active policy and optimizer |
| final inference helper | no-train final episode |
| rank-1 runner | artifact-backed rank-1 training lifecycle |
| rank-2 runner | artifact-backed rank-2 training lifecycle |
| evidence summary helpers | final cadence/MIDI evidence |

Allowed imports:

| Import source | Reason |
| --- | --- |
| `tower.train.config` | rank config |
| `tower.train.checkpoint` | artifacts/checkpoints/manifests |
| `tower.train.protocol` | existing train episode helpers |
| `tower.train.rollout` | final inference if needed |
| `tower.policy.transformer` | real policy construction |
| `tower.policy.observation` | policy input path |
| `tower.music.render` | final MIDI artifact |
| `tower.reward` modules | cadence evidence |
| `tower.graph.spec` | graph config |

Forbidden imports:

| Import source | Reason |
| --- | --- |
| `rl_counterpoint` | frozen project boundary |
| `scripts.*` | runner should be library code |

### Test Files

| File | Status | Purpose |
| --- | --- | --- |
| `tests/tower/train/test_runner.py` | new | runner lifecycle tests |

Required test coverage:

| Test | Proves |
| --- | --- |
| runner config validation | run-level config works |
| final inference no optimizer step | final episode does not train |
| final inference no gradients | no gradient accumulation |
| rank-1 runner writes artifacts | config/metrics/checkpoint/manifest/MIDI |
| rank-1 runner updates policy | training actually connects |
| rank-2 runner loads parent | accepted parent dependency works |
| rank-2 parent frozen | parent unchanged |
| rank-2 child trains | child optimizer changes child |
| rank-2 final MIDI | final child inference exported |

Focused verification:

```bash
uv run pytest tests/tower/train/test_runner.py tests/tower/policy tests/tower/music
```

## Phase 5 Script/CLI File Map

### Source Files

| File | Status | Ownership |
| --- | --- | --- |
| `scripts/tower_train.py` | new | user-facing training script |

`scripts/tower_train.py` owns:

| Behavior | Requirement |
| --- | --- |
| direct execution | imports repo package from root when run by file path |
| argument parsing | enough for rank, episodes, lineage/artifact root |
| tiny run support | can execute a small local run |
| output | prints run dir and final MIDI path |

Allowed imports:

| Import source | Reason |
| --- | --- |
| `tower.train.runner` | run orchestration |
| standard library | argparse/path/sys |

Forbidden imports:

| Import source | Reason |
| --- | --- |
| `rl_counterpoint` | frozen project boundary |
| direct low-level graph internals unless needed | script should stay thin |

### Test Files

| File | Status | Purpose |
| --- | --- | --- |
| `tests/tower/train/test_runner_script.py` | new | script execution tests |

Required test coverage:

| Test | Proves |
| --- | --- |
| direct script execution | file-path execution imports project |
| tiny rank-1 run | script can produce artifacts |
| stdout includes run dir | user can find artifacts |
| stdout includes MIDI path | user can inspect final artifact |

Focused verification:

```bash
uv run pytest tests/tower/train/test_runner_script.py
```

## Boundary Matrix

| Layer | May depend on | Must not depend on |
| --- | --- | --- |
| `tower/graph/*` | `tower.state_action`, standard library | `tower.policy`, `tower.train`, `torch` model code |
| `tower/window.py` | `tower.state_action` | `tower.policy`, `tower.train` |
| `tower/policy/observation.py` | `tower.window`, `tower.state_action`, `torch` | `tower.train`, `rl_counterpoint` |
| `tower/policy/transformer.py` | `tower.policy.base`, `tower.policy.observation`, `torch` | `tower.train.runner`, `rl_counterpoint` |
| `tower/music/render.py` | `tower.state_action`, `tower.train.trajectory` | `tower.policy`, `rl_counterpoint` |
| `tower/train/runner.py` | `tower.train.*`, `tower.policy.*`, `tower.music.render`, `tower.reward.*` | `rl_counterpoint`, `scripts.*` |
| `scripts/tower_train.py` | `tower.train.runner` | `rl_counterpoint` |

## Implementation Order

Implementation should proceed in this order unless the owner explicitly changes
it:

| Order | Action | Files |
| --- | --- | --- |
| 1 | Phase 2.Stage 2.1.Action 2.1.1 | `tower/policy/observation.py`, `tests/tower/policy/test_observation.py` |
| 2 | Phase 2.Stage 2.2.Action 2.2.1 | same observation files |
| 3 | Phase 3.Stage 3.1.Action 3.1.1 | `tower/policy/transformer.py`, `tests/tower/policy/test_transformer.py` |
| 4 | Phase 3.Stage 3.2.Action 3.2.1 | same transformer files |
| 5 | Phase 3.Stage 3.3.Action 3.3.1 | sampler/protocol tests |
| 6 | Phase 4.Stage 4.1.Action 4.1.1 | `tower/music/*`, `tests/tower/music/test_render.py` |
| 7 | Phase 4.Stage 4.2.Action 4.2.1 | same music files |
| 8 | Phase 5.Stage 5.1.Action 5.1.1 | `tower/train/runner.py`, `tests/tower/train/test_runner.py` |
| 9 | Phase 5.Stage 5.2.Action 5.2.1 | same runner files |
| 10 | Phase 5.Stage 5.3.Action 5.3.1 | same runner files |
| 11 | Phase 5.Stage 5.4.Action 5.4.1 | same runner files |
| 12 | Phase 5.Stage 5.5.Action 5.5.1 | `scripts/tower_train.py`, `tests/tower/train/test_runner_script.py` |

## Stop Conditions

Pause and resynchronize if:

| Stop condition | Why |
| --- | --- |
| any proposed tower source imports `rl_counterpoint` | violates frozen-project boundary |
| graph/window modules need torch tensors | violates tuple graph core |
| observation file needs training runner imports | layer leak |
| transformer file needs runner imports | layer leak |
| child model needs parent logits/logprobs/top-m candidates | violates owner model decision |
| MIDI utility needs policy internals | layer leak |
| script starts owning training logic | runner/script boundary leak |

## Next Phase.Stage.Action

After this map is accepted, the next proposed action is:

```text
Post-Slice-8 Phase 2.Stage 2.1.Action 2.1.1:
Add Encoded Tower Window Shell
```

That is the first code implementation action in the post-Slice-8 plan.

Expected files:

```text
tower/policy/observation.py
tests/tower/policy/test_observation.py
```

No code implementation is approved by this document.
