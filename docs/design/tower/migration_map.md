# Tower Redesign Migration Map

## Purpose

This document maps the current flat `rl_counterpoint` system into the proposed tower-based redesign.

## Attribution

The redesign direction documented here originates with the project manager.

In particular, the following ideas are project-manager-originated:

- nested graphs across chord size
- projections as full graph morphisms
- the observation that voiceleading builds tier-by-tier, with lower-rank voiceleading genuinely contained in higher-rank voiceleading
- stagewise policy growth by rank
- semidirect / hierarchical-search framing
- the expectation that this structure can produce a major search/training speedup
- the dimensionality argument that tiering the search radically reduces effective search dimension when out-degree is the operative local complexity bound
- the idea that extensions/sections become comparatively easy to build once lower-rank structure has been trained

This document is therefore an implementation-planning map for those ideas, not a claim of authorship over them.

The goal is not to preserve file-for-file structure. The goal is to preserve the right ideas while replacing the flat ontology:

- flat chord state
- flat whole-chord action
- one-rank graph
- one-rank policy

with a tower-based ontology:

- downward state projection across chord size
- upward action assemblability across chord size
- staged rank-local rewards
- staged policy hierarchy

This is a design map, not an implementation plan.

## Top-Level Recommendation

Keep the current flat system intact as a working reference.

Build the new system in the top-level `tower/` subtree.

Use the current repo selectively:

- copy reusable musical/runtime primitives
- adapt good protocol and encoding ideas
- replace the flat graph/action/policy core from scratch

## Buckets

The old code falls into three buckets:

1. Keep mostly as-is
2. Keep the ideas, but refactor heavily
3. Replace from scratch

---

## 1. Keep Mostly As-Is

These files are good reusable infrastructure and are not deeply committed to the flat graph design.

### Music Primitives

- `rl_counterpoint/music/pitch.py`
- `rl_counterpoint/music/intervals.py`
- `rl_counterpoint/music/consonance.py`
- `rl_counterpoint/music/render.py`

Why keep:

- these are general musical utility functions
- they are not inherently tied to flat `G(n)`
- they will still be needed for rendering, interval arithmetic, consonance scoring, and symbolic diagnostics

Recommended destination:

- `tower/music/` or shared repo-level music utilities if we later decide they should remain common

### Metrical / Observation Utilities

- `rl_counterpoint/envs/observation.py`
- `rl_counterpoint/envs/termination.py`

Why keep:

- `TimedChordWindow`
- padding logic
- bar-position helpers
- strong/weak beat helpers
- max-step truncation helper

These are all likely still relevant even if the state object changes.

Recommended destination:

- `tower/observation/` or `tower/env/observation.py`
- `tower/env/termination.py`

### Reward Protocol Shape

- `rl_counterpoint/reward/protocol.py`

Why keep:

- the structured `RewardContext` / `RewardResult` split is good
- this is the right general idea for rank-local rewards too

What should change:

- the context fields should become tower-aware
- source/target should not be assumed to be raw flat chord tuples

Recommended destination:

- `tower/reward/protocol.py`

Status:

- keep conceptually
- rewrite dataclasses for tower objects

---

## 2. Keep The Ideas, But Refactor Heavily

These areas contain valuable decisions, but they are currently expressed in a flat-system way.

### Symbolic Encoding / Context Conditioning

- `rl_counterpoint/models/policy.py`

What should survive:

- symbolic chord rendering as model input
- tonic / meter / target conditioning
- sequence-window encoding idea
- transformer-style policy path as an option

What should change:

- inputs should become hierarchical state/action objects, not only flat chord tuples
- encoders should be rank-aware
- context should likely include parent/child rank structure

Recommended destination:

- `tower/policy/encoder.py`
- `tower/policy/sequence_models.py`

Status:

- preserve the encoding philosophy
- redesign the actual contracts

### Reward Ideas

- `rl_counterpoint/reward/black_box.py`

Ideas worth carrying forward:

- static consonance scoring
- beat-role contrast ideas
- goal pressure / deadline ideas
- early-arrival bonus ideas
- structured diagnostics

What should change:

- rewards should become rank-local extension rewards
- each stage `n+1` should score the newly added coordinate over the frozen/scaffolded lower-rank structure
- root-level reward should likely differ substantially from inner-voice reward

Recommended destination:

- `tower/reward/root.py`
- `tower/reward/outer_interval.py`
- `tower/reward/interior_extension.py`

Status:

- keep many formulas
- completely refactor ownership and scope

### Environment Interface Pattern

- `rl_counterpoint/envs/counterpoint_env.py`

What should survive:

- explicit env object
- structured info dict / diagnostics philosophy
- separation between transition legality and reward evaluation
- reset / step discipline

What should change:

- state object type
- action object type
- legal action computation
- stage/rank-aware transitions
- parent-policy conditioning / frozen lower-rank scaffold

Recommended destination:

- `tower/train/env_stage.py`
- or `tower/env/stage_env.py`

Status:

- preserve the explicit environment pattern
- rebuild the dynamics around tower objects

### Goal-Bias Policy Wrapper

- `rl_counterpoint/algos/rollout.py`
- `rl_counterpoint/algos/reinforce.py`
- `scripts/train_reinforce.py`

What should survive:

- the general idea of wrapping model logits with a domain prior
- consistency between rollout and loss when a wrapper defines the true policy

What should change:

- the wrapper should become rank-aware
- it may operate differently on root actions, outer-interval actions, and interior-line actions

Recommended destination:

- `tower/policy/wrappers.py`

Status:

- keep the principle
- redesign the implementation around tower action types

---

## 3. Replace From Scratch

These files are the ones most deeply tied to the flat graph/action ontology and should not be ported directly.

### Flat State-Space Construction

- `rl_counterpoint/graph/state_space.py`

Why replace:

- assumes a node is just a raw increasing chord tuple
- has no parent/child projection structure
- no explicit notion of rank-extension

Tower replacement should provide:

- rank-specific node types
- canonical projection maps
- realization maps from hierarchical nodes to flat chord tuples

Recommended destination:

- `tower/graph/nodes.py`
- `tower/graph/projections.py`

### Flat Graph Spec

- `rl_counterpoint/graph/graph_spec.py`

Why replace:

- one monolithic spec object for the whole flat graph
- not organized around rank-specific extension constraints

Tower replacement should provide:

- global shared musical constraints
- plus rank-local extension constraints
- possibly one spec per stage in the tower

Recommended destination:

- `tower/graph/spec.py`

### Flat Edge Predicates

- `rl_counterpoint/graph/non_crossing.py`

Why replace:

- edge legality is currently defined as full-chord transition legality
- tower system needs legality of extension actions relative to parent scaffold

Tower replacement should provide:

- root-stage legality
- outer-interval extension legality
- inner-line extension legality
- compatibility across projections

Recommended destination:

- `tower/graph/edge_rules.py`

### Flat Action Lattice

- `rl_counterpoint/graph/actions.py`

Why replace:

- `StepDelta` is the flat-system assumption made concrete
- whole-chord bounded-lattice actions are exactly what the tower redesign is moving away from

Tower replacement should provide:

- `Action1` for pedal/root motion
- `Action2` for outer interval over pedal
- `Action3` and onward for interior extension coordinates
- explicit upward assemblability of full action

Recommended destination:

- `tower/action/types.py`
- `tower/action/assembly.py`

### Flat Rollout / REINFORCE Core

- `rl_counterpoint/algos/rollout.py`
- `rl_counterpoint/algos/reinforce.py`

Why replace:

- rollout semantics are currently “one policy over one flat action set”
- new design is staged and rank-conditional

What might survive conceptually:

- explicit trajectory records
- loss-replay discipline

What should be rebuilt:

- staged rollout
- frozen parent-policy conditioning
- rank-local action sampling
- stagewise training loop

Recommended destination:

- `tower/train/rollout.py`
- `tower/train/stagewise.py`

---

## Suggested New Tower Package Layout

This is the current recommended layout.

```text
tower/
    graph/
        spec.py
        nodes.py
        projections.py
        edge_rules.py
    action/
        types.py
        assembly.py
    policy/
        encoder.py
        wrappers.py
        stage_policies.py
    reward/
        protocol.py
        root.py
        outer_interval.py
        interior_extension.py
    train/
        stage_env.py
        rollout.py
        stagewise.py
```

Optional later additions:

```text
tower/
    music/
    observation/
```

if we decide not to share the existing helpers directly.

---

## Design Principles To Preserve

These are the old-system decisions that seem strongest and should survive in spirit.

### 1. Explicit Contracts

The current repo does well when it makes interfaces explicit:

- reward protocol
- observation dataclasses
- action masks
- diagnostics

The tower system should keep that style.

### 2. Symbolic Interpretability

The repo already prefers symbolic chord/context rendering over opaque tensors only.

That interpretability is worth preserving in the tower system.

### 3. Diagnostics-Rich Runtime

The current env/reward path exposes a lot of structured information.

That is very useful and should remain a design norm in `tower/`.

### 4. Music Logic Separated From RL Logic

The repo already has a healthy split between:

- music primitives
- graph legality
- reward logic
- training logic

That separation should be preserved, even though the tower ontology changes.

---

## Design Principles To Drop

These current assumptions should not be carried forward.

### 1. Chord State As Bare Tuple Is The True State

In the tower design, the bare tuple should become a realized view, not the primary state object.

### 2. One Flat Action Object Solves Everything

The new design depends on staged actions by rank.

### 3. One Flat Reward Must Judge Everything

The new design depends on rank-local extension rewards.

### 4. One Graph Per Chord Size Is Independent

The new design depends on projection structure across rank.

---

## First Design Targets

Before implementing anything substantial in `tower/`, the following design targets should be pinned down.

1. Canonical node projection

```text
pr^(n+1 -> n)(s^(n+1)) = s^n
```

2. Canonical action assembly

```text
alpha^n built from alpha^1, ..., alpha^n
```

3. Rank-local reward ownership

- what exactly does `R^1` score?
- what exactly does `R^2` score?
- what exactly does `R^3` score?

4. Stagewise training protocol

- freeze parent?
- partially fine-tune parent?
- when is joint refinement allowed?

5. Representation choice

- recursive dataclasses
- algebraic object model
- flat realized chord as derived view only

---

## Summary

The strongest migration decision is:

- preserve the repo’s musical semantics, protocols, and diagnostics philosophy
- do not preserve the flat graph/state/action ontology

That means:

- reuse content
- replace shape

The `tower/` subtree should be treated as a new system built from selected old ideas, not as a refactor of the current flat runtime.
