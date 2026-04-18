# Implementation Slices

This document is the Phase 5 / Stage 15 deliverable for the tower redesign.

The purpose is to define the first buildable/testable implementation slices before any implementation begins.

This is a design contract, not an implementation file.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 5: Freeze System Architecture |
| Stage | Stage 15: Define first implementation slice |
| Action | Choose the smallest safe build order for the tower system |

Stage 15 exit criterion:

| Requirement | Status |
| --- | --- |
| choose first implementation slice | drafted here |
| choose second implementation slice | drafted here |
| define files touched per slice | drafted here |
| define behavior required per slice | drafted here |
| define tests required per slice | drafted here |
| define out-of-scope work per slice | drafted here |

## Guiding Principle

The tower design should be implemented in small slices that prove one concept at a time.

Do not begin with neural training.

Training failures are hard to interpret unless the following are already proven:

| Contract | Why it must be proven first |
| --- | --- |
| state/action representation | everything else depends on it |
| projection | tower correctness depends on it |
| action assembly | policies assemble upward through actions |
| lift-fiber masks | this is the core search-space reduction |
| reward context | rewards depend on windows/actions/goals |
| rollout records | training loss depends on recorded logprobs/rewards |
| artifacts | rank dependencies need reproducible lineage |

The build order should move from pure contracts to rollout to training.

## Slice Order

The accepted slice order is:

| Slice | Name |
| --- | --- |
| 1 | Rank-1 core contracts |
| 2 | Rank-2 projection and action assembly |
| 3 | Rank-2 lift-fiber masks |
| 4 | Rank-2 rollout without neural policy |
| 5 | Rank-1/rank-2 reward context and success predicates |
| 6 | Artifact/checkpoint skeleton |
| 7 | Rank-1 learning loop |
| 8 | Rank-2 learning over frozen rank 1 |

## Slice 1: Rank-1 Core Contracts

Purpose:

Implement the base of the tower without parent policies, lift fibers, neural policy, or checkpoint lineage.

Core math:

\[
s_t^1=(\lambda_0),
\qquad
\Delta s_t^1=(\Delta\lambda_0),
\qquad
s_{t+1}^1=s_t^1+\Delta s_t^1.
\]

Files:

| File | Work |
| --- | --- |
| `tower/__init__.py` | package marker |
| `tower/state_action.py` | rank state/action objects or aliases |
| `tower/window.py` | padded fixed-length rank window |
| `tower/reward/context.py` | rank-1 reward context shell |
| `tower/reward/result.py` | reward output shell |
| `tower/reward/success.py` | rank-1 success predicate skeleton or minimal detector |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| state construction | construct/validate \(s^1=(\lambda_0)\) |
| action construction | construct/validate \(\Delta s^1=(\Delta\lambda_0)\) |
| realization | return tuple form |
| window construction | build left-padded fixed-length \(W_t^1\) |
| padding | PAD state and valid mask work |
| meter positions | bar positions align with step index |
| reward context | carries rank, source, action, target, window, tonic/meter/goal fields |

Tests:

| Test | Proves |
| --- | --- |
| rank-1 state/action construction | primitive contracts work |
| invalid rank/length rejected | invariants are enforced |
| rank-1 window padding | old padding idea works tower-locally |
| reward context accepts rank-1 fields | Stage 8 context shell works |
| reward result defaults | Stage 8 output shell works |

Out of scope:

| Out of scope |
| --- |
| rank-2 projection |
| action assembly over parent |
| lift-fiber masks |
| neural policies |
| training loops |
| checkpoint lineage |

## Slice 2: Rank-2 Projection And Action Assembly

Purpose:

Implement the first real tower transition:

\[
G(2)_\bullet\to G(1)_\bullet.
\]

Core math:

\[
\operatorname{pr}^2(\lambda_0,\lambda_1)=(\lambda_0)
\]

and:

\[
\operatorname{pr}^2(\Delta\lambda_0,\Delta\lambda_1)=(\Delta\lambda_0).
\]

Action assembly:

\[
\Delta s_t^2=(\Delta\lambda_{0,t},\Delta\lambda_{1,t}).
\]

Files:

| File | Work |
| --- | --- |
| `tower/graph/projection.py` | rank-2 state/action/window projection |
| `tower/action/assembly.py` | assemble \(\Delta s^2\) from parent action plus outer-voice coordinate |
| `tower/graph/spec.py` | minimal rank-1/rank-2 graph spec shell |
| `tower/graph/legality.py` | minimal rank-1/rank-2 state/edge checks |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| state projection | project \(s^2\mapsto s^1\) |
| action projection | project \(\Delta s^2\mapsto \Delta s^1\) |
| window projection | project \(W_t^2\mapsto W_t^1\) |
| rank-2 assembly | assemble parent \(\Delta s^1\) and outer coordinate into \(\Delta s^2\) |
| projection compatibility | assembled action projects to parent action |
| commuting law | projection commutes with state update |

Required test equation:

\[
\operatorname{pr}^2(s_t^2+\Delta s_t^2)
=
\operatorname{pr}^2(s_t^2)+\operatorname{pr}^2(\Delta s_t^2).
\]

Tests:

| Test | Proves |
| --- | --- |
| project rank-2 state | node projection works |
| project rank-2 action | action projection works |
| assemble rank-2 action | parent plus outer action works |
| assembled action projects to parent | graph-morphism action law |
| window projection preserves bar/mask | Stage 8 projection law |
| commuting update law | projection and action update are coherent |

Out of scope:

| Out of scope |
| --- |
| full graph enumeration |
| rank-3 projection |
| lift-fiber masks |
| rollout |
| reward formulas |
| learning |

## Slice 3: Rank-2 Lift-Fiber Masks

Purpose:

Implement the first core search-space reduction mechanism.

Given:

\[
s_t^2\mapsto s_t^1
\]

and parent action:

\[
\Delta s_t^1,
\]

the rank-2 child action mask keeps only arrows lying over the parent arrow:

\[
A_2(s_t^2;\Delta s_t^1)
=
\left\{
\Delta s_t^2
\mid
\operatorname{pr}^2(\Delta s_t^2)=\Delta s_t^1
\right\}.
\]

Files:

| File | Work |
| --- | --- |
| `tower/graph/actions.py` | candidate action generation and masks |
| `tower/graph/legality.py` | legality filtering for candidates |
| `tower/action/assembly.py` | active coordinate helpers |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| candidate actions | bounded step-delta candidates exist |
| lift-fiber filtering | only actions over parent action survive |
| legality filtering | graph-invalid lifts are removed |
| empty-fiber diagnostic | empty fiber can be detected |

Tests:

| Test | Proves |
| --- | --- |
| lift mask filters by parent action | only matching parent coordinate remains |
| lift fiber combines projection and legality | illegal arrows removed after fiber check |
| empty fiber detected | exceptional Stage 11 case is observable |
| active coordinate candidates known | policy can target new coordinate |

Out of scope:

| Out of scope |
| --- |
| parent policy sampling |
| rollout records |
| invalid-extension penalty |
| neural policy logits |
| reward terms |

## Slice 4: Rank-2 Rollout Without Neural Policy

Purpose:

Test the runtime choreography without learning.

Use deterministic or scripted policies/samplers so failures isolate rollout semantics, not neural training.

Files:

| File | Work |
| --- | --- |
| `tower/train/rollout.py` | parent-first rollout loop |
| `tower/train/trajectory.py` | Option C step records |
| `tower/policy/samplers.py` | trivial/scripted samplers and top-\(m\) helpers |
| `tower/reward/result.py` | simple scalar reward output |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| parent sampled first | frozen/scripted parent action chosen before child |
| child fiber mask | child choice restricted over parent action |
| assembled action | full \(\Delta s^2\) produced |
| valid transition | valid action advances state |
| invalid extension | no-op, penalty, time advances |
| Option C record | parent diagnostics plus active logprob stored |

Tests:

| Test | Proves |
| --- | --- |
| rank-2 scripted rollout advances | basic choreography works |
| invalid extension records diagnostic | Stage 11 failure semantics work |
| empty fiber records diagnostic | exceptional case is visible |
| trajectory record contains Option C fields | training data shape is ready |
| active logprob is distinct from parent logprob | gradient ownership is clear |

Out of scope:

| Out of scope |
| --- |
| neural policy model |
| optimizer step |
| checkpoint writing |
| MIDI export |
| full TC21M reward terms |

## Slice 5: Rank-1/Rank-2 Reward Context And Success

Purpose:

Implement the first rank-local terminal success predicates and reward context extensions.

Core rule:

\[
\mathsf{Success}_k(W_t^k)
=
\mathsf{Success}_{k-1}(\operatorname{pr}^k W_t^k)
\wedge
\mathsf{NewTerminalCondition}_k(W_t^k).
\]

Rank 1:

\[
\mathsf{Success}_1(W_t^1)
=
\left[
W_t^1\models \operatorname{pr}^2(\text{perfect cadence})
\right].
\]

Rank 2:

\[
\mathsf{Success}_2(W_t^2)
=
\mathsf{Success}_1(\operatorname{pr}^2 W_t^2)
\wedge
\left[
\text{outer voice supplies the third of the cadence chords}
\right].
\]

Files:

| File | Work |
| --- | --- |
| `tower/reward/context.py` | new-facts context payload |
| `tower/reward/success.py` | rank-1 and rank-2 success predicates |
| `tower/reward/terms.py` | simple placeholder/composite reward protocol |
| `tower/reward/result.py` | diagnostics support |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| rank-1 success | projected cadence root motion detectable |
| rank-2 lifted success | parent success plus outer-third condition |
| new facts | rank-2 context identifies outer voice and new intervals |
| diagnostics | success/failure reason reported |

Tests:

| Test | Proves |
| --- | --- |
| rank-1 cadence root success | base terminal predicate works |
| rank-1 non-cadence failure | predicate rejects wrong terminal window |
| rank-2 requires parent success | lifted success uses projection |
| rank-2 requires outer-third condition | new terminal condition is rank-local |
| new-facts diagnostics | reward context avoids double-scoring |

Out of scope:

| Out of scope |
| --- |
| full TC21M reward inventory |
| neural training |
| artifact lineage |
| MIDI export |

## Slice 6: Artifact/Checkpoint Skeleton

Purpose:

Implement artifact contracts without full training.

Files:

| File | Work |
| --- | --- |
| `tower/train/checkpoint.py` | lineage/rank paths, config, metrics, checkpoint, manifest |
| `tower/train/config.py` | config dataclasses and JSON serialization |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| lineage root | `artifacts/tower/<lineage_id>/` |
| rank dirs | `rank_<k>/` directories |
| config write/read | `config.json` |
| metrics append | `metrics.jsonl` |
| latest checkpoint path | `checkpoint_latest.pt` |
| manifest update | accepted parent checkpoint discoverable |

Tests:

| Test | Proves |
| --- | --- |
| lineage paths deterministic | artifact contract stable |
| rank config round trips | config persistence works |
| metrics append JSONL | old behavior carried forward |
| manifest records parent | Stage 12 lineage rule works |
| parent lookup from manifest | rank \(k+1\) can find rank \(k\) |

Out of scope:

| Out of scope |
| --- |
| actual model checkpoint payload |
| optimizer state |
| real training loop |
| generated MIDI |

## Slice 7: Rank-1 Learning Loop

Purpose:

Add the first trainable policy loop only after primitive contracts, windows, reward context, and artifacts are stable.

Files:

| File | Work |
| --- | --- |
| `tower/policy/base.py` | policy protocol |
| `tower/policy/samplers.py` | active exploration sampler |
| `tower/train/losses.py` | policy-gradient loss |
| `tower/train/protocol.py` | rank-1 train loop |
| `tower/train/checkpoint.py` | actual checkpoint payload |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| rank-1 rollout | produces trainable trajectory |
| active logprob | recorded for every step |
| returns/loss | computed from rank-1 reward |
| optimizer step | updates only \(\pi^1\) |
| checkpoint | saves rank-1 latest checkpoint |

Tests:

| Test | Proves |
| --- | --- |
| one rank-1 training episode runs | end-to-end minimal training |
| one optimizer step changes params | learning path connected |
| checkpoint writes latest | artifact contract integrated |
| metrics JSONL records rank | metrics contract integrated |

Out of scope:

| Out of scope |
| --- |
| rank-2 frozen-parent training |
| top-\(m\) parent sampler |
| multi-rank lineage beyond rank 1 |
| full TC21M reward suite |

## Slice 8: Rank-2 Learning Over Frozen Rank 1

Purpose:

Train the first true tower child policy over a frozen rank-1 checkpoint.

Files:

| File | Work |
| --- | --- |
| `tower/train/protocol.py` | train rank 2 from accepted rank 1 |
| `tower/train/checkpoint.py` | load parent checkpoint and record dependency |
| `tower/policy/samplers.py` | parent top-\(m\) sampler |
| `tower/train/rollout.py` | parent-first rollout with active child |

Required behavior:

| Behavior | Requirement |
| --- | --- |
| load frozen \(\pi^1\) | parent checkpoint read-only |
| sample parent first | top-\(m\) mostly-greedy sampler |
| child mask | rank-2 lift fiber over parent action |
| optimize only \(R_2\) | active-tier-only gradient |
| parent diagnostics | parent logprob and sampler details recorded |
| checkpoint lineage | rank-2 checkpoint records rank-1 parent |

Tests:

| Test | Proves |
| --- | --- |
| rank-2 train setup finds parent | manifest lookup works |
| parent params do not change | freeze rule works |
| child params update | active tier trains |
| trajectory uses lift-fiber mask | Stage 11 semantics in training |
| rank-2 checkpoint records parent | Stage 12 lineage works |

Out of scope:

| Out of scope |
| --- |
| rank-3 |
| full higher-rank repeated pattern |
| production-quality policy architecture |
| final reward tuning |

## Phase 5 Completion Criteria

Phase 5 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| first slice is rank-1 core contracts | yes |
| second slice is rank-2 projection/action assembly | yes |
| third slice is rank-2 lift-fiber masks | yes |
| rollout comes before neural training | yes |
| reward success predicates come before full reward suite | yes |
| artifacts come before full multi-rank training | yes |
| first learning loop is rank 1 | yes |
| first true tower training loop is rank 2 over frozen rank 1 | yes |

Once accepted, the next phase is Phase 6: Implementation Readiness Review.
