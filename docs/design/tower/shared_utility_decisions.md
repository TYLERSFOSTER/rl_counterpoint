# Shared Utility Decisions

This document is the Phase 5 / Stage 14 deliverable for the tower redesign.

The purpose is to decide whether the new `tower/` system directly imports old `rl_counterpoint/` utilities, copies them, rewrites them, or treats them as reference only.

This is a design contract, not an implementation file.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 5: Freeze System Architecture |
| Stage | Stage 14: Decide shared-vs-copied utilities |
| Action | Classify old utilities as legacy reference, copy source, rewrite source, or disallowed runtime dependency |

Stage 14 exit criterion:

| Requirement | Status |
| --- | --- |
| decide whether `tower/` may import `rl_counterpoint/` | drafted here |
| classify music helpers | drafted here |
| classify graph helpers | drafted here |
| classify observation/window helpers | drafted here |
| classify reward helpers | drafted here |
| classify policy/training helpers | drafted here |
| define copy-and-own rule | drafted here |

## Core Decision

The old `rl_counterpoint/` system is legacy.

It is frozen reference material, not a runtime dependency for `tower/`.

The reason is structural: the tower redesign exists because the old flat `rl_counterpoint` model does not scale for dimension/search reasons. Therefore the new tower system should not remain coupled to the old system at runtime.

The core Stage 14 rule is:

> The tower implementation may copy from `rl_counterpoint/`, but should not import from it. Once copied, the code becomes tower-owned and may be modified to fit tower semantics.

So:

| Relationship | Decision |
| --- | --- |
| `tower/` imports from `rl_counterpoint/` | disallowed by default |
| `tower/` copies stable utility code from `rl_counterpoint/` | allowed |
| copied code is modified for tower semantics | allowed |
| old files are edited to support tower | disallowed unless explicitly reopened |
| old files are read for design reference | allowed |

## System Boundary

The intended boundary is:

```text
rl_counterpoint/     frozen legacy flat system
tower/              new active tower system
```

This means `rl_counterpoint/` should be treated like a fossil record of useful decisions and code, not like a package dependency.

If a helper is generic enough to reuse, it should be copied into `tower/` or later extracted into a neutral shared package that is not conceptually owned by the legacy system.

## Classification Labels

This document uses these labels:

| Label | Meaning |
| --- | --- |
| copy | copy into `tower/` with minimal changes |
| copy-and-rework | copy into `tower/`, then modify for tower semantics |
| rewrite | implement tower-local version from design, using old code only as reference |
| reference only | read old code for ideas, do not copy mechanically |
| disallowed runtime import | `tower/` should not import this old module |

All old `rl_counterpoint/` modules are disallowed runtime imports by default.

## Music Helpers

### `rl_counterpoint/music/pitch.py`

Decision: copy.

Reason:

Pitch helpers such as MIDI note name rendering and pitch-class computation are generic. They do not depend on the flat graph architecture.

Tower destination:

```text
tower/music/pitch.py
```

Notes:

| Helper type | Tower status |
| --- | --- |
| MIDI bounds checks | copy |
| pitch class | copy |
| MIDI note name rendering | copy |

### `rl_counterpoint/music/intervals.py`

Decision: copy.

Reason:

Pitch-class interval helpers are generic:

\[
(\lambda_j-\lambda_i)\bmod 12.
\]

Tower destination:

```text
tower/music/intervals.py
```

### `rl_counterpoint/music/render.py`

Decision: copy.

Reason:

MIDI rendering of chord sequences is not specific to the flat graph. A tower rank-\(k\) passage is still a tuple sequence of MIDI-note chords:

\[
(s_0^k,\dots,s_T^k).
\]

Tower destination:

```text
tower/music/render.py
```

Notes:

The copied renderer may later be extended to include tower/rank metadata, but the base MIDI writer can start as a copy.

### `rl_counterpoint/music/consonance.py`

Decision: copy-and-rework.

Reason:

The numeric consonance helper may be useful, but consonance scoring is musically and reward-semantically sensitive. The tower reward model distinguishes new vertical facts, full-sonority facts, downbeat/upbeat roles, and rank-local ownership.

Tower destination:

```text
tower/music/consonance.py
```

Notes:

The low-level interval-to-score mapping can be copied initially, but reward-facing consonance computations should live in tower reward modules.

## Observation And Windows

### `rl_counterpoint/envs/observation.py`

Decision: copy-and-rework.

Reason:

The old observation module already has useful mechanics:

| Old idea | Tower status |
| --- | --- |
| fixed-length history windows | keep |
| left padding | keep |
| valid mask | keep |
| bar positions | keep |
| PAD chord | keep |

But tower windows must project:

\[
W_t^k\longrightarrow W_t^{k-1}.
\]

Therefore this should become tower-owned code.

Tower destination:

```text
tower/window.py
```

Do not import the old observation module directly.

## Graph Helpers

### `rl_counterpoint/graph/graph_spec.py`

Decision: rewrite.

Reason:

The old graph spec is flat. Tower graph specs need rankwise projections, outer-width bands, adjacent gap rules, parent/child lift semantics, and graph-morphism compatibility.

Tower destination:

```text
tower/graph/spec.py
```

### `rl_counterpoint/graph/state_space.py`

Decision: rewrite.

Reason:

Some low-level checks carry over, but the tower node model has rank projection semantics and a different ownership split between state/action primitives and graph legality.

Tower destination:

```text
tower/graph/legality.py
```

or:

```text
tower/state_action.py
```

depending on whether a check is structural or graph-legal.

### `rl_counterpoint/graph/actions.py`

Decision: rewrite.

Reason:

The old action helpers generate flat step-delta spaces. Tower action helpers must construct lift fibers:

\[
A_k(s_t^k;\Delta s_t^{k-1})
=
\left\{
\Delta s_t^k
\mid
\operatorname{pr}^k(\Delta s_t^k)=\Delta s_t^{k-1}
\right\}.
\]

Tower destination:

```text
tower/graph/actions.py
```

and:

```text
tower/action/assembly.py
```

### `rl_counterpoint/graph/non_crossing.py`

Decision: copy-and-rework.

Reason:

No crossing and no parallel fifths remain important edge pruning ideas. But tower edge legality must also enforce projection compatibility, and graph legality belongs in tower-owned code.

Tower destination:

```text
tower/graph/legality.py
```

## Reward Helpers

### `rl_counterpoint/reward/protocol.py`

Decision: rewrite.

Reason:

The old reward protocol inspired the Stage 8 tower contract, but tower reward context has rank, parent projection, new-facts bookkeeping, and full-sonority diagnostics.

Tower destination:

```text
tower/reward/context.py
tower/reward/result.py
```

### `rl_counterpoint/reward/black_box.py`

Decision: reference only.

Reason:

The old black-box reward contains useful experiments:

| Old idea | Tower status |
| --- | --- |
| target/root deadline | reference |
| terminal window reward | reference |
| reciprocal early-goal bonus | reference |
| beat-role consonance bonus | reference |
| static consonance diagnostics | reference |

But the tower reward system is rank-local and should be implemented from the Stage 7 reward spec.

Tower destination:

```text
tower/reward/terms.py
tower/reward/success.py
```

## Environment And Termination

### `rl_counterpoint/envs/counterpoint_env.py`

Decision: reference only.

Reason:

The old environment defines useful semantics for reset, invalid action, history, truncation, and reward context construction. But tower rollouts are not a flat Gym-style environment by default. They involve frozen parent policies, lift-fiber masks, invalid extensions, and lineage-aware training.

Tower destination:

```text
tower/train/rollout.py
```

Possibly later:

```text
tower/env.py
```

### `rl_counterpoint/envs/termination.py`

Decision: copy or rewrite trivially.

Reason:

The max-step truncation helper is tiny and generic. It can be copied into tower if needed, but direct import is still disallowed by the legacy boundary.

Tower destination:

```text
tower/train/termination.py
```

or inline in rollout/training protocol if simple.

## Policy And Model Helpers

### `rl_counterpoint/models/policy.py`

Decision: reference only initially.

Reason:

The old policy code may contain useful transformer/window encoding ideas, but it likely assumes:

| Old assumption | Tower issue |
| --- | --- |
| flat action dimension | tower samples rank-local new coordinate over lift fiber |
| old timed chord encoding | tower windows project rankwise |
| old context token format | tower context includes rank and parent scaffold |
| old action mask shape | tower masks over lift fibers |

Tower destination:

```text
tower/policy/base.py
tower/policy/samplers.py
```

Later architecture-specific files may be added, such as:

```text
tower/policy/transformer.py
```

but this should happen after Stage 15 chooses the implementation slice.

## Algorithm Helpers

### `rl_counterpoint/algos/rollout.py`

Decision: reference only.

Reason:

Old rollout logic is flat. Tower rollout semantics are parent-first and lift-fiber based.

Tower destination:

```text
tower/train/rollout.py
```

### `rl_counterpoint/algos/reinforce.py`

Decision: reference only.

Reason:

The old REINFORCE implementation may inform return/loss computation, but tower training uses active-tier-only log probabilities and parent diagnostics. Loss code should be tower-owned.

Tower destination:

```text
tower/train/losses.py
```

## Training Script

### `scripts/train_reinforce.py`

Decision: reference only.

Reason:

The old training script provides useful artifact decisions:

| Old idea | Tower status |
| --- | --- |
| write `config.json` | carry forward |
| append `metrics.jsonl` | carry forward |
| save `checkpoint_latest.pt` | carry forward |
| export `example_episode.mid` | carry forward |
| print training summaries | carry forward |

But the tower training lifecycle, rollout semantics, parent checkpoints, and lineage manifest are different.

Tower destination:

```text
tower/train/protocol.py
tower/train/checkpoint.py
```

Thin scripts may later call tower modules:

```text
scripts/train_tower.py
```

## Summary Table

| Old module | Decision | Tower destination |
| --- | --- | --- |
| `rl_counterpoint/music/pitch.py` | copy | `tower/music/pitch.py` |
| `rl_counterpoint/music/intervals.py` | copy | `tower/music/intervals.py` |
| `rl_counterpoint/music/render.py` | copy | `tower/music/render.py` |
| `rl_counterpoint/music/consonance.py` | copy-and-rework | `tower/music/consonance.py` |
| `rl_counterpoint/envs/observation.py` | copy-and-rework | `tower/window.py` |
| `rl_counterpoint/envs/termination.py` | copy or rewrite trivially | `tower/train/termination.py` or inline |
| `rl_counterpoint/envs/counterpoint_env.py` | reference only | `tower/train/rollout.py` or future `tower/env.py` |
| `rl_counterpoint/graph/graph_spec.py` | rewrite | `tower/graph/spec.py` |
| `rl_counterpoint/graph/state_space.py` | rewrite | `tower/graph/legality.py`, `tower/state_action.py` |
| `rl_counterpoint/graph/actions.py` | rewrite | `tower/graph/actions.py`, `tower/action/assembly.py` |
| `rl_counterpoint/graph/non_crossing.py` | copy-and-rework | `tower/graph/legality.py` |
| `rl_counterpoint/reward/protocol.py` | rewrite | `tower/reward/context.py`, `tower/reward/result.py` |
| `rl_counterpoint/reward/black_box.py` | reference only | `tower/reward/terms.py`, `tower/reward/success.py` |
| `rl_counterpoint/models/policy.py` | reference only initially | `tower/policy/base.py`, `tower/policy/samplers.py` |
| `rl_counterpoint/algos/rollout.py` | reference only | `tower/train/rollout.py` |
| `rl_counterpoint/algos/reinforce.py` | reference only | `tower/train/losses.py` |
| `scripts/train_reinforce.py` | reference only | `tower/train/protocol.py`, `tower/train/checkpoint.py` |

## Stage 14 Completion Checklist

Stage 14 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| `rl_counterpoint/` is legacy reference | yes |
| `tower/` should not import `rl_counterpoint/` at runtime | yes |
| copying old utilities into tower is allowed | yes |
| copied utilities become tower-owned | yes |
| music pitch/interval/render helpers can be copied | yes |
| consonance helper can be copied and reworked | yes |
| old graph modules should be rewritten tower-locally | yes |
| old window helper should be copied and reworked | yes |
| old reward modules are reference/rewrite only | yes |
| old training and algorithm modules are reference only | yes |

Once accepted, the next stage is Phase 5 / Stage 15: Define first implementation slice.
