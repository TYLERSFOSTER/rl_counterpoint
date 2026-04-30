# Tower Implementation Plan

This document is the Phase 5 / Stage 13 deliverable for the tower redesign.

The purpose is to map the accepted tower design documents to concrete files under the top-level `tower/` package before implementation begins.

This is still a design document. It does not authorize implementation by itself.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 5: Freeze System Architecture |
| Stage | Stage 13: Map design to files |
| Action | Specify planned files, responsibilities, import boundaries, inputs/outputs, and non-ownership rules |

Stage 13 exit criterion:

| Requirement | Status |
| --- | --- |
| map design concepts to planned files | drafted here |
| identify each file's responsibility | drafted here |
| identify allowed imports | drafted here |
| identify inputs and outputs | drafted here |
| identify what each file must not own | drafted here |

## Ground Rules

The old `rl_counterpoint/` subproject is frozen for tower work. Tower implementation should happen under the top-level `tower/` package.

During implementation, do not modify files under `rl_counterpoint/` unless the project manager explicitly reopens that decision.

The tower package may read or import stable old utilities only after Stage 14 decides which old utilities remain shared and which need tower-local copies.

## Existing Top-Level Skeleton

The current intended tower skeleton is:

```text
tower/
  action/
  graph/
  policy/
  reward/
  train/
```

Stage 13 assigns responsibilities inside this skeleton. Stage 14 will decide which old utilities can be shared. Stage 15 will choose the first implementation slice.

## Architecture Layers

The tower package should be layered from pure mathematical structure upward to training orchestration:

```text
state/action primitives
  -> graph projection and legality
  -> reward context and reward terms
  -> policy interfaces and samplers
  -> rollout
  -> training protocol
  -> artifacts/checkpoints
```

Imports should generally point downward. Graph code should not import training code. Reward terms should not import training orchestration. Policy modules should not own graph legality.

## Planned File Map

### `tower/__init__.py`

Responsibility:

| Category | Content |
| --- | --- |
| package marker | marks `tower` as a package |
| public API | maybe exports a small stable subset later |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| none by default | keep package import lightweight |

Inputs/outputs:

| Input | Output |
| --- | --- |
| package import | no side effects |

Must not own:

| Must not own |
| --- |
| training startup |
| checkpoint loading |
| global config creation |
| model construction |

### `tower/state_action.py`

Responsibility:

| Category | Content |
| --- | --- |
| state aliases or dataclasses | rank-$k$ state representation |
| action aliases or dataclasses | rank-$k$ move-vector representation |
| realization helpers | convert state/action objects to tuples if needed |
| invariants | tuple length, integer MIDI/action coordinates |

Core concepts:

$$
s^k=(\lambda_0,\dots,\lambda_{k-1})
$$

$$
\Delta s^k=(\Delta\lambda_0,\dots,\Delta\lambda_{k-1})
$$

Allowed imports:

| Allowed | Notes |
| --- | --- |
| standard library | dataclasses, typing |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| construct state | rank, tuple of ints | validated state |
| construct action | rank, tuple of ints | validated action |
| realize state | state object | tuple[int, ...] |
| realize action | action object | tuple[int, ...] |

Must not own:

| Must not own |
| --- |
| graph legality |
| reward semantics |
| training behavior |
| policy sampling |
| artifact paths |

### `tower/action/assembly.py`

Responsibility:

| Category | Content |
| --- | --- |
| action assembly | assemble rank-$k$ action from parent action plus new coordinate |
| new-coordinate indexing | identify active coordinate at each rank |
| projection compatibility checks | verify assembled action projects to parent action |

Core rules:

For $k=2$:

$$
\Delta s^2=(\Delta\lambda_0,\Delta\lambda_1).
$$

For $k\ge 3$:

$$
\Delta s^k
=
(\Delta\lambda_0,\dots,\Delta\lambda_{k-3},\Delta\lambda_{k-2},\Delta\lambda_{k-1}).
$$

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.state_action` | state/action types |
| `tower.graph.projection` | action projection |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| new voice index | rank $k$ | coordinate index |
| assemble action | parent action, new coordinate, rank | rank-$k$ action |
| validate assembly | assembled action, parent action | bool or diagnostic |

Must not own:

| Must not own |
| --- |
| policy logits |
| reward scoring |
| graph state legality beyond projection compatibility |
| training loop |

### `tower/graph/spec.py`

Responsibility:

| Category | Content |
| --- | --- |
| tower graph config | global and per-rank graph knobs |
| pruning parameters | pitch range, adjacent gap, outer width, forbidden intervals |
| validation | ensure graph spec values are coherent |

Core parameters:

| Parameter family | Examples |
| --- | --- |
| MIDI bounds | $0,\dots,127$ |
| adjacent gap | $M_{\mathrm{adj}},F_{\mathrm{adj}}$ |
| outer width | $L(n,N),U(n,N)$ |
| edge motion | $M_{\mathrm{move}}$ |
| tonic/root classes | $\tau$, allowed root/outer pitch classes |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| standard library | dataclasses, typing |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| construct spec | config values | validated spec |
| rank spec | global spec, rank | rank-specific view |

Must not own:

| Must not own |
| --- |
| projection functions |
| action masks |
| reward weights |
| policy architecture |
| artifact paths |

### `tower/graph/projection.py`

Responsibility:

| Category | Content |
| --- | --- |
| state projection | $G(k)_0\to G(k-1)_0$ |
| action projection | $\mathbb Z^k\to\mathbb Z^{k-1}$ |
| window projection | $W_t^k\to W_t^{k-1}$ |
| trajectory projection | optional helper for diagnostics |

Core rules:

$$
\operatorname{pr}^2(\lambda_0,\lambda_1)=(\lambda_0)
$$

For $k\ge 3$:

$$
\operatorname{pr}^k(\lambda_0,\dots,\lambda_{k-3},\lambda_{k-2},\lambda_{k-1})
=
(\lambda_0,\dots,\lambda_{k-3},\lambda_{k-1}).
$$

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.state_action` | state/action types |
| `tower.reward.context` | only for window/context projection if no cycle is created |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| project state | rank-$k$ state | rank-$k-1$ state |
| project action | rank-$k$ action | rank-$k-1$ action |
| project window | rank-$k$ window | rank-$k-1$ window |

Must not own:

| Must not own |
| --- |
| graph legality |
| reward scoring |
| policy sampling |
| training loop |

Design note:

If importing `tower.reward.context` creates a cycle, window projection should move to a neutral module such as `tower/window.py`.

### `tower/graph/legality.py`

Responsibility:

| Category | Content |
| --- | --- |
| node legality | rank-$k$ state validity |
| edge legality | rank-$k$ transition validity |
| graph morphism check | valid higher edge projects to valid lower edge |
| no crossing / no parallel fifths | hard edge rules |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.state_action` | state/action types |
| `tower.graph.spec` | graph parameters |
| `tower.graph.projection` | projection compatibility |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| valid state | state, graph spec | bool / diagnostic |
| valid transition | source, action or target, graph spec | bool / diagnostic |
| edge diagnostics | source, action, target, spec | reason codes |

Must not own:

| Must not own |
| --- |
| policy masks as tensors |
| reward penalties |
| rollout time advancement |
| checkpoint behavior |

### `tower/graph/actions.py`

Responsibility:

| Category | Content |
| --- | --- |
| candidate action generation | bounded rank action candidates |
| lift-fiber generation | legal actions over parent action |
| action masks | Boolean masks for legal actions |
| empty-fiber detection | identify `empty_lift_fiber` |

Core set:

$$
A_k(s_t^k;\Delta s_t^{k-1})
=
\left\{
\Delta s_t^k\in\partial_0^{-1}(s_t^k)
\mid
\operatorname{pr}^k(\Delta s_t^k)=\Delta s_t^{k-1}
\right\}.
$$

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.state_action` | action/state types |
| `tower.action.assembly` | action assembly helpers |
| `tower.graph.legality` | legality checks |
| `tower.graph.projection` | fiber checks |
| `tower.graph.spec` | action bounds |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| action candidates | rank, max step | tuple/list of actions |
| legal action mask | state, candidates, spec | bool mask |
| lift fiber | state, parent action, spec | legal child actions |
| extension mask | state, parent action, child candidates, spec | bool mask |

Must not own:

| Must not own |
| --- |
| policy logits |
| sampler randomness |
| reward scoring |
| trajectory storage |

### `tower/window.py`

Responsibility:

| Category | Content |
| --- | --- |
| rank window object | fixed-length padded rank-$k$ history view |
| padding | PAD state and valid mask |
| bar positions | metrical positions per window slot |
| window projection | may live here to avoid graph/reward cycles |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.state_action` | state types |
| `tower.graph.projection` | state projection if no cycle |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| build window | history, step index, measure size, context measures | window |
| pad state | rank | PAD state |
| project window | rank-$k$ window | rank-$k-1$ window |

Must not own:

| Must not own |
| --- |
| policy encoding |
| reward scoring |
| rollout stepping |
| graph legality |

### `tower/reward/context.py`

Responsibility:

| Category | Content |
| --- | --- |
| reward context dataclasses | rank, source, target, action, window, meter, goal, deadline |
| new-facts payload | active coordinate and new vertical facts |
| context projection | possibly delegated to `tower/window.py` and `tower/graph/projection.py` |
| context validation | consistency checks |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.state_action` | state/action types |
| `tower.window` | window types |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| build reward context | rollout step components | rank context |
| derive new facts | state/action/rank | new-facts payload |
| project context | rank-$k$ context | rank-$k-1$ context |

Must not own:

| Must not own |
| --- |
| reward term formulas |
| policy behavior |
| graph action masks |
| checkpoint paths |

### `tower/reward/result.py`

Responsibility:

| Category | Content |
| --- | --- |
| reward output dataclass | scalar reward, hard violation, terminal success, diagnostics |
| diagnostics conventions | common keys and merge helpers |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| standard library | dataclasses, typing |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| construct result | reward, flags, diagnostics | reward result |
| merge diagnostics | component diagnostics | combined diagnostics |

Must not own:

| Must not own |
| --- |
| reward formulas |
| context construction |
| environment stepping |
| training loss |

### `tower/reward/terms.py`

Responsibility:

| Category | Content |
| --- | --- |
| reward term protocol | interface for rank-local terms |
| composite reward | weighted sum of rank terms |
| terminal reward hooks | terminal condition integration |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.reward.context` | reward input |
| `tower.reward.result` | reward output |
| maybe shared music utilities | Stage 14 decision |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| reward term call | reward context | reward result or component result |
| composite call | reward context | reward result |

Must not own:

| Must not own |
| --- |
| graph legality |
| policy sampling |
| rollout stepping |
| artifact writing |

### `tower/reward/success.py`

Responsibility:

| Category | Content |
| --- | --- |
| rank success predicates | $\mathsf{Success}_k(W_t^k)$ |
| lifted success | parent success through projection plus new terminal condition |
| cadence terminal hooks | terminal predicate structure |

Core rule:

$$
\mathsf{Success}_k(W_t^k)
=
\mathsf{Success}_{k-1}(\operatorname{pr}^k W_t^k)
\wedge
\mathsf{NewTerminalCondition}_k(W_t^k).
$$

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.window` | windows |
| `tower.graph.projection` | projected windows |
| maybe shared music utilities | Stage 14 decision |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| success predicate | rank, window, context | bool plus diagnostics |
| terminal diagnostics | window/context | diagnostics |

Must not own:

| Must not own |
| --- |
| dense reward weights |
| policy sampling |
| checkpointing |
| rollout storage |

### `tower/policy/base.py`

Responsibility:

| Category | Content |
| --- | --- |
| policy protocol | rank-local policy call contract |
| policy output structure | logits/probs over active child choices |
| masking contract | apply legal masks to policy outputs |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.window` | policy inputs |
| `tower.state_action` | action types |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| policy forward | window/context | logits or distribution |
| masked distribution | logits, mask | legal distribution |

Must not own:

| Must not own |
| --- |
| graph legality computation |
| parent sampler policy |
| reward computation |
| optimizer steps |

### `tower/policy/samplers.py`

Responsibility:

| Category | Content |
| --- | --- |
| active exploration sampler | $(1-\epsilon)\pi+\epsilon U$ |
| parent top-$m$ sampler | mostly-greedy frozen-parent sampling |
| export sampler | optional temperature sampler |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.policy.base` | policy outputs |
| standard/random/torch | implementation-stage decision |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| sample active action | policy logits, legal mask, epsilon | action and logprob |
| sample parent action | parent logits, legal mask, top-m config | action and diagnostic logprob |
| sample export action | logits, legal mask, temperature | action |

Must not own:

| Must not own |
| --- |
| graph legality construction |
| reward scoring |
| checkpoint loading |
| trajectory object definition |

### `tower/train/trajectory.py`

Responsibility:

| Category | Content |
| --- | --- |
| rollout step record | Option C record fields |
| trajectory container | sequence of step records |
| return computation helpers | discounted returns if needed |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.state_action` | states/actions |
| `tower.window` | windows |
| `tower.reward.result` | reward outputs |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| create step record | per-step rollout data | `TowerStepRecord` |
| create trajectory | step records | trajectory |
| compute returns | trajectory, gamma | returns |

Must not own:

| Must not own |
| --- |
| rollout decision logic |
| policy sampling |
| reward formulas |
| checkpoint paths |

### `tower/train/rollout.py`

Responsibility:

| Category | Content |
| --- | --- |
| per-step rollout choreography | Stage 11 semantics |
| parent-first sampling | call frozen parent sampler first |
| lift-fiber mask | get legal child extensions over parent action |
| invalid-extension behavior | no-op, penalty, diagnostics |
| trajectory emission | build Option C records |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.graph.actions` | masks and lift fibers |
| `tower.action.assembly` | assembled actions |
| `tower.reward.context` | context building |
| `tower.reward.terms` | reward call |
| `tower.policy.samplers` | sampling behavior |
| `tower.train.trajectory` | records |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| run one rollout | env/state/spec/policies/config | trajectory |
| step once | rollout state | step record and next state |

Must not own:

| Must not own |
| --- |
| policy architecture |
| reward formulas |
| checkpoint serialization |
| lineage manifest |

### `tower/train/protocol.py`

Responsibility:

| Category | Content |
| --- | --- |
| stagewise lifecycle | train rank $1,2,3,\dots$ |
| freeze enforcement | load lower policies read-only |
| active training loop | run rollouts, update active policy |
| acceptance by episode budget | rank accepted after $E_k$ |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.train.rollout` | rollout execution |
| `tower.train.trajectory` | returns/loss inputs |
| `tower.train.checkpoint` | save/load artifacts |
| `tower.policy.*` | active and parent policies |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| train rank | lineage id, rank config, parent refs | accepted checkpoint |
| train lineage | tower config | lineage manifest |

Must not own:

| Must not own |
| --- |
| low-level graph legality |
| reward formulas |
| policy model internals |
| artifact schema details beyond calling checkpoint module |

### `tower/train/checkpoint.py`

Responsibility:

| Category | Content |
| --- | --- |
| artifact paths | `artifacts/tower/<lineage_id>/rank_<k>/...` |
| config persistence | write/read `config.json` |
| metrics persistence | append `metrics.jsonl` |
| checkpoint persistence | save/load rolling latest checkpoint |
| lineage manifest | read/write `manifest.json` |
| parent lookup | locate accepted parent checkpoint |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| standard library | pathlib, json, dataclasses |
| torch | checkpoint serialization if torch is used |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| lineage paths | lineage id | canonical paths |
| save config | rank config | config path |
| append metrics | episode stats | metrics path |
| save checkpoint | policy/optimizer/config/stats | checkpoint path |
| load parent | lineage id, parent rank | checkpoint payload |
| update manifest | rank status/checkpoint | manifest path |

Must not own:

| Must not own |
| --- |
| training loop |
| policy forward pass |
| reward calculation |
| graph legality |

### `tower/train/losses.py`

Responsibility:

| Category | Content |
| --- | --- |
| policy-gradient loss | active-tier logprob and returns |
| entropy regularization | if used by tower training |
| diagnostics | loss scalars |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| `tower.train.trajectory` | trajectory/logprob data |
| torch | implementation-stage decision |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| compute loss | trajectory, returns, config | scalar loss and diagnostics |

Must not own:

| Must not own |
| --- |
| rollout sampling |
| optimizer stepping |
| reward formulas |
| checkpoint paths |

### `tower/train/config.py`

Responsibility:

| Category | Content |
| --- | --- |
| config dataclasses | tower lineage config and per-rank config |
| validation | check required rank/parent fields |
| serialization helpers | JSON-friendly config payloads |

Allowed imports:

| Allowed | Notes |
| --- | --- |
| standard library | dataclasses, typing |
| `tower.graph.spec` | graph config types |

Inputs/outputs:

| Function family | Input | Output |
| --- | --- | --- |
| build config | user/default values | validated config |
| serialize config | config object | JSON payload |

Must not own:

| Must not own |
| --- |
| artifact writing |
| training loop |
| policy model construction if avoidable |
| reward formulas |

## Script Entry Points

Stage 13 does not require scripts, but implementation will probably need thin scripts later:

| Script | Responsibility |
| --- | --- |
| `scripts/train_tower.py` | CLI entrypoint for tower lineage/rank training |
| `scripts/export_tower_midi.py` | export from a lineage/rank checkpoint |
| `scripts/smoke_tower_rank1.py` | first vertical slice smoke |
| `scripts/smoke_tower_rank2.py` | second vertical slice smoke |

Scripts should be thin wrappers. They should not own core training logic.

## Import Boundary Summary

Allowed direction:

```text
state_action
  <- graph
  <- reward context/result
  <- policy interfaces
  <- train rollout
  <- train protocol
  <- scripts
```

Checkpoint/artifact code is a side service used by training protocol, not by graph/reward/policy primitives.

## Files Not Yet Planned

The following may be needed later but should not be invented until Stage 15 or implementation readiness:

| Possible file | Why deferred |
| --- | --- |
| `tower/env.py` | unclear if tower needs Gym-style env or rollout can be direct |
| `tower/music.py` | Stage 14 must decide shared vs copied music utilities |
| `tower/policy/transformer.py` | architecture choice can wait until implementation slice |
| `tower/eval.py` | evaluation semantics can follow training protocol |

## Stage 13 Completion Checklist

Stage 13 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| keep implementation under top-level `tower/` | yes |
| keep old `rl_counterpoint/` frozen | yes |
| separate state/action primitives from graph legality | yes |
| separate graph legality from reward scoring | yes |
| separate reward context/result from reward terms | yes |
| separate policy samplers from graph action masks | yes |
| place rollout choreography in `tower/train/rollout.py` | yes |
| place artifact/checkpoint behavior in `tower/train/checkpoint.py` | yes |
| defer shared-vs-copied old utilities to Stage 14 | yes |
| defer first implementation slice to Stage 15 | yes |

Once accepted, the next stage is Phase 5 / Stage 14: Decide shared-vs-copied utilities.
