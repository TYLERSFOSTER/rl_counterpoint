# Post-Slice-8 Build Plan

This document pauses tower implementation after completion of the first build
plan's Slice 8.

The purpose is to reconcile the original implementation plan with the code that
now exists, identify what is real versus skeletal, and define the next game plan
only after that assessment is accepted.

This is a planning document, not implementation approval.

## Stage Location

This document belongs to the post-implementation assessment following:

| Plan level | Current location |
| --- | --- |
| Source phase | Phase 6: Implementation Readiness Review |
| Source stage | Stage 17: Produce machine-implementable build plan |
| Completed build range | Implementation Slices 1-8 |
| Current action | Pause, assess, and write the next game plan |

## Pause Gate

All new tower feature work should pause until the project owner accepts a new
post-Slice-8 plan.

Allowed work before acceptance:

| Work | Allowed |
| --- | --- |
| document current state | yes |
| inspect existing code/tests | yes |
| run verification commands | yes |
| fix newly discovered test breakage | only with owner approval |
| add new tower behavior | no |
| generalize rank-2 to rank-k | no |
| build transformer policies | no |
| begin real training runs | no |

## Source Plans Being Reconciled

Primary sources:

| Source | Role |
| --- | --- |
| `docs/design/tower/mathematical_model.md` | mathematical ground truth |
| `docs/design/tower/contradiction_pass.md` | resolution of design contradictions |
| `docs/design/tower/implementation_slices.md` | accepted Slice 1-8 order |
| `docs/design/tower/build_plan.md` | machine-implementable Slice 1-8 plan |
| `docs/design/tower/slice4_rollout_clarifications.md` | rollout ambiguity answers |
| `docs/design/tower/training_protocol.md` | training ownership and freeze rules |
| `docs/design/tower/artifact_checkpoint_dependencies.md` | lineage and artifact intent |
| `assets/rules/tc21m_rules.md` | existing TC21M reward-rule summary for future reward expansion |

Owner-answer source:

| Source | Role |
| --- | --- |
| `docs/design/tower/post_slice_8_questions.md` | project-owner answers to post-Slice-8 planning questions |

Continuity source:

| Source | Role |
| --- | --- |
| `docs/engineer_continuity/2026/04/18/01_001_tower_design_to_build_handoff.md` | handoff from design to implementation |

## Current Implementation State

Tower source currently exists under:

```text
tower/
```

Tower tests currently exist under:

```text
tests/tower/
```

Implemented module groups:

| Area | Files |
| --- | --- |
| core contracts | `tower/state_action.py`, `tower/window.py` |
| graph | `tower/graph/projection.py`, `tower/graph/spec.py`, `tower/graph/legality.py`, `tower/graph/actions.py` |
| action assembly | `tower/action/assembly.py` |
| reward | `tower/reward/context.py`, `tower/reward/result.py`, `tower/reward/success.py`, `tower/reward/terms.py` |
| policy | `tower/policy/base.py`, `tower/policy/samplers.py` |
| training | `tower/train/trajectory.py`, `tower/train/rollout.py`, `tower/train/losses.py`, `tower/train/config.py`, `tower/train/checkpoint.py`, `tower/train/protocol.py` |

Test module groups:

| Area | Files |
| --- | --- |
| core contracts | `tests/tower/test_state_action.py`, `tests/tower/test_window.py` |
| graph/action | `tests/tower/graph/*`, `tests/tower/action/test_assembly.py` |
| reward | `tests/tower/reward/*` |
| policy | `tests/tower/policy/*` |
| training | `tests/tower/train/*` |
| import boundary | `tests/tower/test_import_boundaries.py` |

## Slice Completion Audit

### Slice 1: Rank-1 Core Contracts

Status: implemented and tested.

Evidence:

| Contract | Current state |
| --- | --- |
| tuple state/action representation | implemented |
| rank validation | implemented |
| state/action realization | implemented |
| fixed-length padded window | implemented |
| reward context shell | implemented |
| reward result shell | implemented |
| success result shell | implemented |

Notes:

The tuple representation is still the canonical tower graph representation.
Tensor use remains downstream and policy-local.

### Slice 2: Rank-2 Projection And Action Assembly

Status: implemented and tested.

Evidence:

| Contract | Current state |
| --- | --- |
| rank-2 state projection | implemented |
| rank-2 action projection | implemented |
| window projection | implemented |
| rank-2 action assembly | implemented |
| projection compatibility | tested |
| commuting update law | tested |

Notes:

Projection remains computed from tuple position. No canonical parent field was
introduced.

### Slice 3: Rank-2 Lift-Fiber Masks

Status: implemented and tested.

Evidence:

| Contract | Current state |
| --- | --- |
| candidate action generation | implemented |
| legal action filtering | implemented |
| lift-fiber action filtering | implemented |
| active coordinate choices | implemented |
| empty lift-fiber detection | implemented |

Notes:

The lift fiber is now an executable object in the codebase, not just a design
claim.

### Slice 4: Rank-2 Rollout Without Neural Policy

Status: implemented and tested.

Evidence:

| Contract | Current state |
| --- | --- |
| trajectory step records | implemented |
| scripted sampler protocol | implemented |
| rank-1 rollout | implemented |
| rank-2 parent-first rollout | implemented |
| invalid extension diagnostic | implemented |
| empty lift-fiber diagnostic | implemented |
| parent failure diagnostic | implemented |
| parent and active logprobs separated | implemented |

Notes:

The clarifications in `slice4_rollout_clarifications.md` were incorporated:
invalid child extensions advance as no-ops, empty lift fibers do not call the
active sampler, and parent failures truncate.

### Slice 5: Reward Context And Success Predicates

Status: implemented as a minimal, tested reward layer.

Evidence:

| Contract | Current state |
| --- | --- |
| reward term protocol | implemented |
| composite reward term | implemented |
| rank-1 cadence success | implemented |
| rank-2 lifted cadence success | implemented |
| rank-2 outer-third condition | implemented |

Notes:

This is intentionally not a full TC21M reward grammar. It is enough to prove
rank-local reward ownership, context construction, terminal success plumbing,
and lifted rank-2 success.

### Slice 6: Artifact/Checkpoint Skeleton

Status: implemented and tested.

Evidence:

| Contract | Current state |
| --- | --- |
| rank config dataclass | implemented |
| config JSON round trip | implemented |
| deterministic artifact paths | implemented |
| metrics JSONL append/read | implemented |
| lineage manifest read/write | implemented |
| latest checkpoint save/load | implemented |
| accepted parent lookup | implemented |

Notes:

The skeleton became a usable local artifact contract. It is still not a full
experiment-management system.

### Slice 7: Rank-1 Learning Loop

Status: implemented and tested.

Evidence:

| Contract | Current state |
| --- | --- |
| rank policy protocol | implemented |
| policy output container | implemented |
| policy-gradient loss | implemented |
| discounted returns | implemented |
| active choice sampler | implemented |
| one rank-1 train episode | implemented |
| artifact-backed rank-1 episode | implemented |

Notes:

The train loop is intentionally minimal. It proves that rollout logprobs,
returns, optimizer stepping, metrics, and checkpoint writing connect.

### Slice 8: Rank-2 Learning Over Frozen Rank 1

Status: implemented and tested.

Evidence:

| Contract | Current state |
| --- | --- |
| frozen parent policy wrapper | implemented |
| parent top-m sampler | implemented |
| rank-2 train episode | implemented |
| child-only optimizer step | implemented |
| parent params preserved | tested |
| parent logprobs diagnostic-only for loss | tested |
| artifact-backed rank-2 episode | implemented |
| rank-2 checkpoint records parent dependency | tested |
| manifest records parent dependency | tested |
| accepted parent checkpoint remains unchanged | tested |

Notes:

This slice proves the core tiered training rule: a rank-2 child can train over a
frozen rank-1 parent while using lift-fiber masking and recording parent lineage.

## Current Verification Evidence

Most recent verification:

```bash
uv run pytest tests/tower/train/test_protocol.py
uv run pytest tests/tower/policy tests/tower/train
uv run pytest tests/tower
uv run pytest
```

Observed result:

| Command | Result |
| --- | --- |
| `uv run pytest tests/tower/train/test_protocol.py` | 21 passed |
| `uv run pytest tests/tower/policy tests/tower/train` | 144 passed |
| `uv run pytest tests/tower` | 246 passed |
| `uv run pytest` | 401 passed |

## Known Deviations And Clarifications

### Slices 4-8 Were More Concrete Than The Original Build Plan

`build_plan.md` fully specified early slices and only outlined later slices.
Implementation necessarily made additional decisions for rollout, loss,
checkpoint payloads, and rank-2 artifact-backed training.

Those decisions should now be treated as implemented contracts unless the
project owner deliberately revises them.

### The Current System Is Rank-1/Rank-2 First

The mathematical model supports the general tower idea, but the implementation
currently proves only the rank-1/rank-2 base case.

Rank-3 and general rank-k behavior remain future work.

### The Current Policy Is A Protocol, Not A Final Architecture

The policy layer supports rank-local policy outputs and samplers. Tests use tiny
policies.

The final production architecture, likely transformer-backed per tier, has not
been implemented.

### The Current Reward Layer Is Minimal

Success predicates and reward terms prove rank-local ownership and terminal
plumbing.

The full musical reward vocabulary remains future work, but it should not be
invented from scratch. Future reward expansion should draw from
`assets/rules/tc21m_rules.md`, which already contains candidate reward/pruning
ideas for motion, cadence, dissonance treatment, harmonic templates, and related
voice-leading behavior.

### Artifact Support Is Local And Minimal

Config, metrics, checkpoints, and manifests are implemented.

Missing future concerns include run naming, multiple checkpoints, best checkpoint
selection, resumable optimizers across long runs, richer metadata, and optional
example MIDI artifacts.

## What Is Real

The following should be considered real implementation ground:

| Area | Real contract |
| --- | --- |
| representation | tuple state/action core |
| projection | parent computed by projection |
| lift fiber | active choices constrained over parent action |
| rollout ownership | parent acts first, child chooses lift coordinate |
| diagnostics | invalid/empty/parent failure outcomes recorded |
| loss ownership | active child logprobs drive child loss |
| freezing | parent policy is frozen during child training |
| lineage | rank-2 artifacts reference accepted rank-1 parent |
| boundary | tower code must not import frozen `rl_counterpoint` |

## What Is Skeletal

The following are not final system capabilities:

| Area | Skeletal state |
| --- | --- |
| policy architecture | protocol and toy test policies only |
| reward grammar | minimal cadence/success proof only |
| training runner | no production CLI or experiment runner |
| rank generality | rank-1/rank-2 only |
| MIDI output | artifact path exists, full tower MIDI writing not implemented |
| evaluation | no musical evaluation suite beyond unit-level reward/success tests |
| performance | no batched graph pruning or optimized tensor graph operations |

## Open Questions Before Future Work

The questions in `post_slice_8_questions.md` have owner answers. The planning
state should now be treated as mostly resolved, with a small number of deferred
questions.

### Resolved Product Decisions

| Question | Decision |
| --- | --- |
| next priority | learnable real policy architecture |
| audible MIDI milestone | training should end with a final no-train inference episode converted to MIDI and saved as an artifact |
| reward richness before long training | do not block first real-policy/training work on large reward expansion |
| weak rewards | acceptable for infrastructure only, but the near-term target is to see training working, not merely to run weak smoke tests |

### Resolved Model Decisions

| Question | Decision |
| --- | --- |
| shared architecture | use the same transformer-family architecture across ranks, with rank-specific tensor shape/config differences |
| rank-specific differences | make them config-driven |
| observation format | use the same conceptual schema across ranks, not identical raw tensor dimensions |
| source of observation contract | derive from old `rl_counterpoint` timed-window transformer policy and tower design docs |
| parent diagnostics as child inputs | do not feed parent logprobs, top-m candidates, or distributions into the child policy |
| child awareness of parent action | child is constrained through the lift fiber over the sampled parent action |

The first real tower policy observation should adapt the old timed-window model
pattern:

| Old project pattern | Tower adaptation |
| --- | --- |
| fixed-length timed chord window | fixed-length rank-local `TowerWindow` |
| left padding | tower window padding and valid mask |
| tonic/meter/target context | explicit model context features |
| symbolic event encoding | rank-local state/event encoding |
| transformer over sequence | shared transformer-family rank policy |
| final valid event hidden state to logits | active-choice logits, masked externally by legality/lift-fiber constraints |

### Resolved Training Decisions

| Question | Decision |
| --- | --- |
| rank-1 promotion gate for now | episode count |
| smarter promotion metrics | deferred until after basic training works |
| musical acceptance target | final no-train episode should realize the rank-local perfect cadence goal in correct octave/key/meter placement |
| parent checkpoint selection | one rank-2 run is tied to one accepted rank-1 checkpoint |
| parent action sampling | sample actions from the frozen rank-1 policy during rank-2 rollout |
| parent randomness | `parent_top_m` is a training/config hyperparameter |
| preferred real-training `parent_top_m` | 3 |
| multiple accepted parent checkpoint mixture | not needed now |
| lineage | record the single parent checkpoint dependency, not multiple possible parent passages |

### Resolved Music-Semantics Decisions

| Question | Decision |
| --- | --- |
| cadence rules | include |
| motion rules | include |
| dissonance treatment | include |
| voice-leading constraints | include |
| harmonic template rules | include |
| six-four logic | defer |
| suspension handling | defer to a later project update such as `beta.1`; treat as a style expansion, not a near-term core requirement |
| cadence-only reward sparsity | likely sparse; use TC21M-derived shaping carefully |

### Deferred Questions

| Question | Deferred reason |
| --- | --- |
| exact rank-k voice-leading ownership beyond rank 2 | project owner wants to revisit after more thought |
| exact promotion metrics beyond episode count | should wait until real training evidence exists |
| exact TC21M reward subset for the first reward-expansion slice | should be planned after real policy/runner path is settled |

## Recommended Future Game Plan

The next plan should be accepted before implementation resumes.

Owner-selected order:

| Proposed slice | Name | Purpose |
| --- | --- | --- |
| Post-8 Slice A | Tower Training Runner | Add a runner plan for artifact-backed rank training, including the final no-train inference episode and MIDI artifact requirement. |
| Post-8 Slice B | Real Policy Observation Contract | Convert tower windows/states into model-ready tensors while preserving tuple graph contracts and following the old timed-window transformer pattern. |
| Post-8 Slice C | Transformer Rank Policy | Implement the first real rank-local transformer-family policy behind the existing policy protocol, with rank-specific config-driven differences. |
| Post-8 Slice D | Example MIDI Artifact | Mimic the previous `rl_counterpoint` MIDI-export path for tower trajectories; this may be implemented as part of Slice A if the runner needs the final artifact immediately. |
| Post-8 Slice E | Reward Expansion Pass | Add the next small set of musically meaningful reward/shaping terms, drawing from `assets/rules/tc21m_rules.md`. |
| Post-8 Slice F | Rank-k Generalization Assessment | Decide whether to generalize graph/rollout/training after rank-2 real-policy experiments provide evidence. |

Although the selected first planning target is the training runner, the product
priority is learnable real policy architecture. Therefore the next detailed plan
should be careful about dependencies: the runner can be planned first, but it
must either include or explicitly sequence the observation contract and
transformer policy work required to make training real.

## Near-Term Acceptance Policy

For the next training-capable milestone:

| Item | Decision |
| --- | --- |
| training stop condition | episode count |
| final evaluation | one no-train inference episode after training |
| final artifact | write final inference episode to MIDI |
| success evidence | record whether the final episode satisfies the rank-local cadence success predicate |
| checkpoint acceptance | may remain episode-budget based initially, but should record cadence/MIDI evidence for later smarter acceptance |
| rank-2 parent dependency | one accepted rank-1 checkpoint |
| rank-2 parent sampling | frozen parent policy, top-m action sampling |
| default real-training `parent_top_m` | 3 |

## Reward Expansion Direction

Reward expansion is not the immediate blocker for the first real-policy runner,
but it should not be treated as empty design space.

`assets/rules/tc21m_rules.md` already points toward:

| Area | Near-term status |
| --- | --- |
| cadence rules | near-term candidate |
| motion rewards | near-term candidate |
| dissonance treatment | near-term candidate, likely shaping/reward before pruning |
| voice-leading constraints | near-term candidate, with graph/reward boundary decisions |
| harmonic templates | near-term candidate after observation/policy work |
| suspensions | defer to later style update such as `beta.1` |
| six-four logic | defer |

The next reward pass should be narrow. It should add enough shaping to make
training interpretable without replacing the main cadence objective or turning
style refinements into core blockers.

## Recommended Immediate Next Action

The recommended immediate next action is:

```text
Post-Slice-8 Planning / Action 1:
Accept or revise this updated assessment, then produce the detailed build plan
for Post-8 Slice A: Tower Training Runner.
```

Suggested owner decision:

| Option | Meaning |
| --- | --- |
| accept as baseline | this document becomes the post-Slice-8 assessment baseline |
| revise | update the assessment before any future implementation |
| authorize next planning doc | create a detailed build plan for Post-8 Slice A |

## Non-Approval Statement

This document does not approve implementation of any post-Slice-8 work.

It creates a pause point and a shared map. Future work should proceed by the
project owner's normal approval loop: discuss, approve, implement, verify.
