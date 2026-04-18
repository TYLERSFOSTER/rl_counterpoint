# Tower Contradiction Pass

This document is the Phase 6 / Stage 16 deliverable for the tower redesign.

The purpose is to check the accepted tower design documents against each other before producing a machine-implementable build plan.

This is a review document, not an implementation file.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 6: Implementation Readiness Review |
| Stage | Stage 16: Run contradiction pass |
| Action | Check accepted design docs for contradictions, unresolved decisions, and implementation blockers |

Stage 16 exit criterion:

| Requirement | Status |
| --- | --- |
| check state/action/projection consistency | reviewed here |
| check graph legality vs rollout semantics | reviewed here |
| check reward spec vs reward context | reviewed here |
| check success semantics vs training protocol | reviewed here |
| check artifact plan vs training protocol | reviewed here |
| check utility decisions vs implementation plan | reviewed here |
| check implementation slices vs implementation plan | reviewed here |
| identify blockers before Stage 17 | reviewed here |

## Authority Order

Some documents were written earlier and contain design ideas that were later corrected. For implementation planning, use this authority order:

| Priority | Document family |
| --- | --- |
| 1 | accepted Stage 7-15 docs |
| 2 | `mathematical_model.md` where it records final mathematical decisions |
| 3 | `implementation_plan.md` and `implementation_slices.md` for file/slice planning |
| 4 | `system_design.md` and `migration_map.md` as historical design context |

The project manager's later corrections override earlier speculative wording.

## Summary Verdict

The core tower design is internally consistent enough to proceed to Stage 17.

There are no conceptual blockers to writing a machine-implementable build plan for the first slices.

However, there are several required amendments or cautions:

| Severity | Finding | Resolution |
| --- | --- | --- |
| required amendment | Stage 13 allows possible direct old utility imports, but Stage 14 forbids runtime imports from `rl_counterpoint/` | Stage 14 wins; Stage 17 must enforce copy/rewrite/reference-only |
| required amendment | older `system_design.md` and `migration_map.md` contain explicit parent-object language that conflicts with later projection-only mathematical model | later docs win; do not implement explicit parent fields as canonical state structure |
| non-blocking typo | `mathematical_model.md` has a coordinate-index typo for intermediate voices | fix before or during Stage 17 documentation cleanup |
| deferred decision | hard-violation termination is not fully settled | not needed for Slice 1-3; decide before rollout/training slices |
| deferred decision | some reward grammar choices remain unresolved | not needed before reward slices |

## Check 1: State, Action, Projection

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `mathematical_model.md` | states, actions, graph projections |
| `training_protocol.md` | action assembly during training |
| `rollout_semantics.md` | parent-first action and lift-fiber mask |
| `implementation_slices.md` | Slice 1-3 tests |

Consistent decisions:

| Topic | Consistent answer |
| --- | --- |
| state | \(s^k=(\lambda_0,\dots,\lambda_{k-1})\) |
| action | \(\Delta s^k=(\Delta\lambda_0,\dots,\Delta\lambda_{k-1})\) |
| rank-2 projection | \(\operatorname{pr}^2(\lambda_0,\lambda_1)=(\lambda_0)\) |
| rank-\(k\ge3\) projection | remove second-from-top coordinate |
| action projection | remove corresponding action coordinate |
| action assembly | parent action plus new coordinate |
| commuting law | projection commutes with transition |

No contradiction in the accepted docs.

Required implementation tests:

\[
\operatorname{pr}^k(s_t^k+\Delta s_t^k)
=
\operatorname{pr}^k(s_t^k)+\operatorname{pr}^k(\Delta s_t^k).
\]

### Non-Blocking Typo

`mathematical_model.md` says:

```text
intermediate coordinates λ_k, for 10≤k≤n-1
```

This should be:

\[
1\le k\le n-2.
\]

It also has at least one missing comma in the displayed rank-\(n\) projection tuple. This is not a conceptual contradiction, but it should be cleaned up before future engineers rely on the prose.

## Check 2: Parent/Child Structure

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `mathematical_model.md` | parent/child encoded in projections |
| `system_design.md` | older explicit parent-object examples |
| `migration_map.md` | older migration sketch |
| `implementation_plan.md` | file responsibilities |

Accepted final decision:

Parent/child structure is encoded by projections, not by requiring the canonical state object to store a parent object.

\[
\operatorname{pr}^k(s^k)=s^{k-1}.
\]

Contradiction:

Older `system_design.md` includes early implementation sketches where higher states/actions carry explicit parent fields. That language conflicts with the later mathematical clarification that the parent structure is encoded by projection.

Resolution:

The later mathematical model wins. Stage 17 must not produce canonical state dataclasses of the form:

```text
StateK(parent=...)
```

unless that is explicitly framed as a cache or convenience view, not the mathematical state object.

Implementation rule:

Store rank-\(k\) state as the rank-\(k\) coordinate tuple or tower-owned equivalent. Compute parent by projection.

## Check 3: Graph Legality Vs Rollout Semantics

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `mathematical_model.md` | graph morphism, node/edge pruning |
| `rollout_semantics.md` | lift-fiber masks and invalid extension |
| `success_failure_semantics.md` | invalid action/extension behavior |
| `implementation_slices.md` | Slice 2-4 |

Consistent decisions:

| Topic | Consistent answer |
| --- | --- |
| higher valid edge projects to lower valid edge | yes |
| child mask is fiber over parent action | yes |
| active child samples legal lift when nonempty | yes |
| empty fiber is exceptional | yes |
| invalid extension is distinct from parent failure | yes |
| invalid extension is no-op, penalty, time-advancing | yes |

No contradiction.

Implementation caution:

The rollout doc uses:

\[
A_k(s_t^k;\Delta s_t^{k-1})
=
\{\Delta s_t^k\in\partial_0^{-1}(s_t^k)\mid \operatorname{pr}^k(\Delta s_t^k)=\Delta s_t^{k-1}\}.
\]

In implementation, `tower/graph/actions.py` must make clear whether `A_k` contains full rank-\(k\) action vectors or active new-coordinate choices. Both views are valid, but policy-facing code should expose active choices while legality code can work with full action vectors.

This is not a contradiction. It is an API clarity requirement for Stage 17.

## Check 4: Reward Spec Vs Reward Context

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `rank_local_reward_spec.md` | reward ownership and term inventory |
| `reward_context_contracts.md` | context fields and projections |
| `training_protocol.md` | optimize only \(R_k\) |

Consistent decisions:

| Topic | Consistent answer |
| --- | --- |
| reward is rank-local | yes |
| inherited facts not rescored by default | yes |
| full-sonority terms are allowed exceptions | yes |
| reward context includes rank/window/action/goal/meter/new facts | yes |
| parent rewards diagnostic only during child training | yes |
| active gradient uses only \(R_k\) | yes |

No contradiction.

Deferred reward choices:

| Choice | Blocks early slices? |
| --- | --- |
| exact consonance set | no |
| exact cadence-template vocabulary | no |
| exact chord-template vocabulary | no |
| whether \(\mathrm{V}\to\mathrm{IV}\) is pruning or hard violation | no for Slice 1-4, yes before full reward suite |
| six-four chord treatment | no |

## Check 5: Success Semantics Vs Training Protocol

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `success_failure_semantics.md` | terminal success, truncation, hard violation |
| `training_protocol.md` | rank-local lifted success |
| `rank_local_reward_spec.md` | cadence and terminal rewards |

Consistent final decision:

\[
\mathsf{Success}_k(W_t^k)
=
\mathsf{Success}_{k-1}(\operatorname{pr}^k W_t^k)
\wedge
\mathsf{NewTerminalCondition}_k(W_t^k).
\]

Rank 1 success is the projected pedal/root part of a perfect cadence. Rank 2 success requires the rank-1 cadence-root condition plus the outer voice supplying the third of the cadence chords.

No contradiction.

Deferred decision:

`success_failure_semantics.md` leaves hard-violation termination configurable:

\[
\mathsf{hardViolation}\Rightarrow \mathsf{terminated}
\]

is not yet accepted as mandatory.

This does not block Slice 1-3. It should be decided before Slice 4 or any rollout/training implementation that can emit hard violations.

## Check 6: Artifact Plan Vs Training Protocol

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `artifact_checkpoint_dependencies.md` | lineage directories and checkpoint dependencies |
| `training_protocol.md` | freeze rule and no rollback |
| `implementation_slices.md` | artifact skeleton before learning loops |

Consistent decisions:

| Topic | Consistent answer |
| --- | --- |
| shared lineage directory | yes |
| per-rank artifacts | yes |
| rolling `checkpoint_latest.pt` | yes |
| parent checkpoint recorded for \(k>1\) | yes |
| parent checkpoint read-only | yes |
| no rollback within lineage | yes |
| accepted checkpoint after episode budget | yes |

No contradiction.

Implementation caution:

The artifact doc allows optional `git_commit` / `git_dirty` metadata. Stage 17 should decide whether this is implemented in the first artifact slice or deferred. It is not needed for Slice 1-5.

## Check 7: Utility Decisions Vs Implementation Plan

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `implementation_plan.md` | planned files and import boundaries |
| `shared_utility_decisions.md` | legacy/copy/rewrite rule |

Contradiction:

`implementation_plan.md` says:

```text
The tower package may read or import stable old utilities only after Stage 14 decides which old utilities remain shared and which need tower-local copies.
```

Stage 14 later decided:

```text
tower/ should not import rl_counterpoint/ at runtime.
```

Resolution:

Stage 14 wins. Stage 17 must enforce:

| Old relationship | Allowed? |
| --- | --- |
| direct runtime import from `rl_counterpoint/` | no |
| copy old utility code into `tower/` | yes |
| modify copied code | yes |
| read old code for reference | yes |

Required amendment:

Before implementation or as part of Stage 17 cleanup, update `implementation_plan.md` Ground Rules to replace "read or import" with "read or copy".

## Check 8: Implementation Slices Vs Implementation Plan

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `implementation_plan.md` | planned files |
| `implementation_slices.md` | build order |

Consistent decisions:

| Topic | Consistent answer |
| --- | --- |
| first slice rank-1 contracts | yes |
| second slice rank-2 projection/assembly | yes |
| graph actions after projection/legality basics | yes |
| rollout before neural training | yes |
| artifact skeleton before multi-rank training | yes |
| rank-1 learning before rank-2 frozen-parent learning | yes |

No contradiction.

Implementation caution:

`implementation_slices.md` introduces `tower/reward/success.py` in Slice 1 as "skeleton or minimal detector." Stage 17 should be careful not to overbuild cadence detection in Slice 1. A skeleton plus tests for shape may be enough, with real rank-1/rank-2 success predicates in Slice 5.

## Check 9: File Naming Consistency

Reviewed docs:

| Document | Relevant content |
| --- | --- |
| `implementation_plan.md` | canonical file map |
| `migration_map.md` | older suggested paths |
| `system_design.md` | older suggested paths |

Contradiction:

Older docs sometimes use names like:

```text
tower/graph/projections.py
```

while the accepted implementation plan uses:

```text
tower/graph/projection.py
```

Resolution:

`implementation_plan.md` wins. Use singular:

```text
tower/graph/projection.py
```

unless Stage 17 explicitly changes the file map.

## Check 10: Missing Originally Planned Docs

The initial preparation gameplan mentioned:

| Planned doc | Current status |
| --- | --- |
| `state_action_spec.md` | not created |
| `graph_legality_spec.md` | not created |
| `reward_training_spec.md` | split into accepted reward/training docs |

This is not a conceptual contradiction because later accepted docs covered the same decisions in a different document structure.

Resolution:

Stage 17 should not assume these exact filenames exist. It should use the accepted current docs:

| Current doc | Covers |
| --- | --- |
| `mathematical_model.md` | state/action/projection/graph model |
| `rank_local_reward_spec.md` | reward ownership |
| `reward_context_contracts.md` | reward context |
| `success_failure_semantics.md` | terminal/failure |
| `training_protocol.md` | lifecycle |
| `rollout_semantics.md` | rollout |
| `artifact_checkpoint_dependencies.md` | artifacts |
| `implementation_plan.md` | file map |
| `implementation_slices.md` | build order |

## Blocking Items Before Stage 17

No conceptual blockers.

Required Stage 17 accommodations:

| Item | Required handling |
| --- | --- |
| no runtime imports from `rl_counterpoint/` | enforce in build actions |
| explicit parent fields in older docs | ignore as superseded |
| singular `projection.py` | use accepted file name |
| hard violation termination | either defer beyond early slices or ask before rollout slice |
| real cadence/reward grammar | defer until reward slice |

## Recommended Cleanup Before Implementation

These are not blockers, but cleaning them up would reduce future confusion:

| Cleanup | Reason |
| --- | --- |
| amend `implementation_plan.md` Ground Rules to say copy, not import | aligns Stage 13 with accepted Stage 14 |
| fix `mathematical_model.md` coordinate typo | avoids future engineer confusion |
| add a short "superseded by later docs" note to `system_design.md` and `migration_map.md` | prevents old parent-field sketches from being misread as final |

## Stage 16 Completion Checklist

Stage 16 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| core tower design is internally consistent | yes |
| no conceptual blocker prevents Stage 17 | yes |
| Stage 14 legacy boundary overrides Stage 13 import wording | yes |
| older explicit-parent sketches are superseded | yes |
| hard-violation termination can be deferred until rollout/training slices | yes |
| unresolved reward grammar can be deferred until reward slices | yes |
| Stage 17 should produce machine-implementable build actions from current accepted docs | yes |

Once accepted, the next stage is Phase 6 / Stage 17: Produce machine-implementable build plan.
