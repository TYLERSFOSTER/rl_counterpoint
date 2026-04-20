# Post-Slice-8 Tower Implementation And Planning Report

This continuity report captures the engineering session that moved the tower
project from "ready to implement Slice 1" through completed implementation of
the original Slice 1-8 tower build plan, then into the accepted post-Slice-8
assessment and the new explicit Phase.Stage.Action plan for the next big build.

The most important state change is:

```text
The first tower implementation build plan has been executed through Slice 8.
The project is now paused at a post-Slice-8 planning gate.
The next approved direction is to plan the real training path with real
rank-local policy architecture, final no-train inference, and MIDI artifact
output.
```

This report is intentionally detailed. It is written so a future LLM engineer
can recover the session state without reading the full chat transcript.

## Project Grounding

The project root is:

```text
/Users/foster/rl_counterpoint
```

The old flat project lives under:

```text
rl_counterpoint/
```

The new active project lives under:

```text
tower/
```

The old `rl_counterpoint/` subproject is frozen for tower work.

The active guardrails remain:

| Guardrail | Requirement |
| --- | --- |
| frozen old project | do not modify `rl_counterpoint/` for tower work |
| import boundary | do not import `rl_counterpoint/` from `tower/` runtime code |
| reuse | old code may be read or copied, but copied code becomes tower-owned |
| owner approval | proceed one Phase.Stage.Action at a time |
| implementation approval | discuss before each action, implement only after owner approval |
| reality breaks | unexpected failures pause the game plan for investigation |

The owner explicitly wants to proceed by Phase.Stage.Action, not broad
implementation batches.

## Prime Directive And Process Notes

At the beginning of the session, the engineer was instructed to read:

```text
docs/prime_directive/
docs/engineer_continuity/2026/04/
docs/design/
docs/design/tower/
```

The most important continuity documents for this session were:

```text
docs/engineer_continuity/2026/04/15/01_002_preparation_gameplan.md
docs/engineer_continuity/2026/04/18/01_001_tower_design_to_build_handoff.md
```

The session followed the owner's preferred rhythm:

```text
discuss
implement
next Phase.Stage.Action
discuss
implement
...
```

The owner repeatedly supplied approval to implement the next action after each
discussion step.

## Important Process Trick Added To Prime Directive

During the session, the engineer asked ambiguity questions about Slice 4 rollout
semantics. The owner then jumped back to a previous LLM thread, asked that
previous LLM to answer those questions, and had it create:

```text
docs/design/tower/slice4_rollout_clarifications.md
```

The engineer initially thought this document had been missed, but the owner
explained that it had not existed before. It was created through a previous
thread as an "echo of past self" maneuver.

In response, a new prime directive document was created:

```text
docs/prime_directive/consultant_tricks.md
```

It records the "Echoes Of Past Self" trick:

| Trick | Meaning |
| --- | --- |
| Echoes Of Past Self | ask a previous LLM thread to answer current ambiguity questions and write the answers into the repo |
| When useful | when current context is missing prior design intent or when the owner can use another thread as a design consultant |
| Engineer behavior | notice when this could help and suggest it as an option |

This process note matters because future engineers should not assume all helpful
documents existed at the start of a session. Some may have been created by
cross-thread owner orchestration.

## Mathematical And Design Ground Truth Used

The authoritative mathematical model remains:

```text
docs/design/tower/mathematical_model.md
```

The contradiction-resolution document remains:

```text
docs/design/tower/contradiction_pass.md
```

The implementation source plans that were executed were:

```text
docs/design/tower/implementation_slices.md
docs/design/tower/build_plan.md
```

Important accepted model facts:

| Topic | Accepted fact |
| --- | --- |
| state | `TowerState = tuple[int, ...]` |
| action | `TowerAction = tuple[int, ...]` |
| rank | normally `len(state)` |
| parent | computed by projection, not stored canonically |
| rank-2 projection | drop second coordinate/action, yielding `(x0,)` |
| rank >= 3 projection | drop second-from-top coordinate/action |
| child action | assembled from parent action plus one active coordinate |
| lift fiber | legal child actions over a sampled parent action |
| reward ownership | rank-k rewards score newly introduced rank-k facts |
| training ownership | child loss uses active child logprobs, parent logprobs diagnostic |
| parent policy | frozen during child training |

Important rollout clarifications accepted from
`slice4_rollout_clarifications.md`:

| Case | Accepted behavior |
| --- | --- |
| invalid child extension | no-op, time advances, diagnostic, default zero reward |
| empty lift fiber | no-op, diagnostic, active sampler is not called |
| parent failure | truncating step, active sampler is not called |
| parent logprob | diagnostic only for child training |
| rank-2 choreography | parent sampled first, then child chooses active lift coordinate |

## Side Quest: Old Training Test Failures

At one point the owner ran:

```bash
uv run pytest
```

and reported failures in:

```text
tests/test_train_reinforce.py
```

The failures concerned default training-script hyperparameters:

| Failed expectation | Observed issue |
| --- | --- |
| `num_episodes == 100_000` | script/test patching exposed `2` |
| `entropy_coefficient: 0.01` printed | output did not include it |

The owner requested that the tests be modified so they explicitly input those
hyperparameters, rather than depending on how variables are set inside the
training script.

A focused test-only fix was made in:

```text
tests/test_train_reinforce.py
```

The intent was:

| Fix | Reason |
| --- | --- |
| inject expected hyperparameters explicitly | tests should not depend on incidental script defaults |
| keep old training code behavior separate from tower work | old stack is not the tower focus |

Later, a separate full-suite collection issue occurred because tower test files
had the same basenames as old test files:

```text
tests/graph/test_actions.py
tests/tower/graph/test_actions.py
tests/algos/test_rollout.py
tests/tower/train/test_rollout.py
```

Pytest reported import-file-mismatch errors. The fix was to update:

```text
pyproject.toml
```

with:

```toml
addopts = ["--import-mode=importlib"]
```

This allowed duplicate test basenames under different packages without pytest
collection conflicts.

## Tower Implementation Summary

The tower implementation moved through Slices 1-8. The latest full-suite result
after Slice 8 closeout was:

```text
401 passed
```

The latest focused tower-suite result after Slice 8 closeout was:

```text
246 passed
```

The sections below summarize what was built.

## Slice 1: Rank-1 Core Contracts

Purpose:

Implement the base tuple contracts without projection, lift fibers, rollout, or
training.

Implemented source:

```text
tower/__init__.py
tower/state_action.py
tower/window.py
tower/reward/__init__.py
tower/reward/result.py
tower/reward/context.py
tower/reward/success.py
```

Implemented tests:

```text
tests/tower/test_state_action.py
tests/tower/test_window.py
tests/tower/reward/test_result.py
tests/tower/reward/test_context.py
tests/tower/reward/test_success.py
```

Core behavior implemented:

| Contract | Behavior |
| --- | --- |
| MIDI pitch | int with range validation |
| tower state | tuple of MIDI pitches |
| tower action | tuple of integer deltas |
| rank validation | invalid ranks rejected |
| state/action length | rank and tuple length must agree |
| state realization | tuple form remains canonical |
| action realization | tuple form remains canonical |
| window | fixed-length, left-padded, valid-mask aware |
| reward result | structured reward/hard-violation/success output |
| reward context | carries rank, source, target, action, window, key/meter/goal fields |
| success result | structured success diagnostics |

Important implementation stance:

Tuple state/action objects are the graph core. Tensor use is not part of the
canonical graph representation.

## Slice 2: Rank-2 Projection And Action Assembly

Purpose:

Implement the first tower projection and action assembly law.

Implemented source:

```text
tower/graph/__init__.py
tower/graph/projection.py
tower/graph/spec.py
tower/graph/legality.py
tower/action/__init__.py
tower/action/assembly.py
```

Implemented tests:

```text
tests/tower/graph/test_projection.py
tests/tower/graph/test_spec.py
tests/tower/graph/test_legality.py
tests/tower/action/test_assembly.py
```

Core behavior implemented:

| Contract | Behavior |
| --- | --- |
| rank-2 state projection | `(x0, x1) -> (x0,)` |
| rank-2 action projection | `(dx0, dx1) -> (dx0,)` |
| rank>=3 projection | removes second-from-top coordinate |
| window projection | projects each valid state and preserves masks/meter |
| graph spec | pitch range and max step size |
| legality | pitch range, strict ordering, movement bounds |
| action assembly | assemble full child action from parent action and active coordinate |
| commuting law | projection commutes with state update |

Important implementation stance:

No canonical parent field was introduced. Parent is computed by projection.

## Slice 3: Rank-2 Lift-Fiber Masks

Purpose:

Implement the core search-space reduction: child action choices restricted to
the legal lift fiber over a sampled parent action.

Implemented source:

```text
tower/graph/actions.py
```

Implemented tests:

```text
tests/tower/graph/test_actions.py
```

Core behavior implemented:

| Functionality | Behavior |
| --- | --- |
| action space | generate candidate actions |
| legal actions | filter by graph legality |
| lift fiber actions | legal child actions projecting to parent action |
| active choices | active coordinate choices extracted from lift fiber |
| empty fiber | detectable and test-covered |

Important implementation stance:

The lift fiber is now executable code, not only design language. It is the
mechanism that keeps the child policy searching a much smaller action set.

## Slice 4: Rank-2 Rollout Without Neural Policy

Purpose:

Implement rollout choreography and trajectory records before real neural policy
training.

Implemented source:

```text
tower/train/__init__.py
tower/train/trajectory.py
tower/train/rollout.py
tower/policy/__init__.py
tower/policy/samplers.py
```

Implemented tests:

```text
tests/tower/train/test_trajectory.py
tests/tower/train/test_rollout.py
tests/tower/policy/test_samplers.py
tests/tower/test_import_boundaries.py
```

Core behavior implemented:

| Contract | Behavior |
| --- | --- |
| `TowerTrajectoryStep` | records source/action/target/reward/logprobs/diagnostics |
| `TowerTrajectory` | rank-local trajectory with total reward |
| outcome labels | normal, invalid extension, empty lift fiber, parent failure |
| scripted sampler | deterministic sampler for tests |
| rank-1 rollout | active sampler only |
| rank-2 rollout | parent-first then child active coordinate |
| invalid extension | no-op, time advances, diagnostic |
| empty lift fiber | no-op, diagnostic, no active sampler call |
| parent failure | truncation, diagnostic, no active sampler call |
| import boundary | static test ensures tower does not import frozen old project |

Important implementation stance:

Parent and active logprobs are stored separately. Parent logprob is diagnostic in
child training.

## Slice 5: Reward Context And Success Predicates

Purpose:

Implement minimal rank-local reward terms and success predicates without
overbuilding the full TC21M grammar.

Implemented source:

```text
tower/reward/success.py
tower/reward/terms.py
```

Implemented tests:

```text
tests/tower/reward/test_success.py
tests/tower/reward/test_terms.py
```

Core behavior implemented:

| Functionality | Behavior |
| --- | --- |
| `RewardTerm` protocol | callable over tower reward context |
| `CompositeRewardTerm` | sums structured rewards and diagnostics |
| `SuccessRewardTerm` | adapts success predicate to reward |
| rank-1 cadence success | final V-I root motion in key, on ending beat |
| rank-2 lifted cadence success | requires projected rank-1 success and outer major thirds |

Current reward layer:

| Exists now | Does not yet exist |
| --- | --- |
| cadence/success predicates | full TC21M reward grammar |
| composite reward plumbing | motion rewards |
| diagnostics | dissonance treatment |
| terminal success flag | suspension handling |
| rank-local success tests | harmonic-template rewards |

Later planning decided future reward expansion should draw from:

```text
assets/rules/tc21m_rules.md
```

## Slice 6: Artifact/Checkpoint Skeleton

Purpose:

Implement deterministic artifact paths, rank configs, metrics, checkpoints, and
lineage manifest support.

Implemented source:

```text
tower/train/config.py
tower/train/checkpoint.py
```

Implemented tests:

```text
tests/tower/train/test_config.py
tests/tower/train/test_checkpoint.py
```

Core behavior implemented:

| Contract | Behavior |
| --- | --- |
| `TowerRankConfig` | rank-local config dataclass |
| JSON round trip | deterministic config serialization |
| `TowerArtifactPaths` | deterministic lineage/rank paths |
| metrics JSONL | append/read metrics rows |
| manifest | read/write lineage manifest |
| checkpoint payload | save/load latest checkpoint |
| parent lookup | find/load accepted parent checkpoint |
| parent validation | rank/lineage/manifest checks |

Important implementation stance:

Artifact support is local and minimal. It is not yet a complete experiment
management system.

## Slice 7: Rank-1 Learning Loop

Purpose:

Connect rank-1 rollout, policy-gradient loss, optimizer stepping, and artifacts.

Implemented source:

```text
tower/policy/base.py
tower/train/losses.py
tower/train/protocol.py
```

Implemented tests:

```text
tests/tower/policy/test_base.py
tests/tower/train/test_losses.py
tests/tower/train/test_protocol.py
```

Core behavior implemented:

| Functionality | Behavior |
| --- | --- |
| `PolicyOutput` | policy output container |
| `RankPolicy` | protocol for rank-local policy |
| `freeze_parent_policy` | freezes module parameters and eval mode |
| `discounted_returns` | computes returns |
| `trajectory_returns` | returns from trajectory rewards |
| `policy_gradient_loss` | uses active logprobs |
| rank-1 active sampler | samples active action from policy logits |
| `train_rank1_episode` | rollout, loss, backward, optimizer step |
| `train_rank1_episode_with_artifacts` | writes config, metrics, checkpoint, manifest |

Important implementation stance:

The rank-1 loop is intentionally small. It proves the training path connects,
not that the model is final.

## Slice 8: Rank-2 Learning Over Frozen Rank 1

Purpose:

Train a rank-2 child policy over a frozen rank-1 parent policy using lift-fiber
masking and parent-linked artifacts.

Implemented source:

```text
tower/policy/samplers.py
tower/train/checkpoint.py
tower/train/protocol.py
```

Implemented tests:

```text
tests/tower/policy/test_samplers.py
tests/tower/train/test_checkpoint.py
tests/tower/train/test_protocol.py
```

Core behavior implemented:

| Functionality | Behavior |
| --- | --- |
| parent top-m sampler | samples from top-m legal parent actions |
| parent logprob | detached diagnostic |
| frozen parent | parameters unchanged and requires_grad false |
| rank-2 train episode | parent-first rollout, child loss, child optimizer step |
| lift-fiber child mask | child choices constrained over parent action |
| child loss | active child logprob only |
| parent checkpoint loading | accepted parent checkpoint found via manifest |
| rank-2 artifact training | writes parent-linked checkpoint/manifest |
| parent artifact safety | accepted parent checkpoint bytes/payload unchanged |

Final Slice 8 closeout added test coverage for:

```text
tests/tower/train/test_protocol.py
```

Specifically:

| Test addition | Purpose |
| --- | --- |
| parent top-m sampler diagnostics | proves rank-2 trajectory records parent top-m details |
| parent checkpoint unchanged | proves artifact-backed rank-2 training does not mutate accepted parent |
| renamed legacy-boundary test | test name now matches no-old-import assertion |

Focused verification after Slice 8 closeout:

```bash
uv run pytest tests/tower/train/test_protocol.py
```

Result:

```text
21 passed
```

Broader focused verification:

```bash
uv run pytest tests/tower/policy tests/tower/train
```

Result:

```text
144 passed
```

Tower verification:

```bash
uv run pytest tests/tower
```

Result:

```text
246 passed
```

Full verification:

```bash
uv run pytest
```

Result:

```text
401 passed
```

## Post-Slice-8 Pause

After Slice 8 closeout, the engineer initially proposed an invented
"Implementation.Extension.Stage 9" next step. The owner questioned whether that
matched:

```text
docs/design/tower/build_plan.md
```

This exposed an important numbering issue.

There are two numbering systems:

| Numbering system | Meaning |
| --- | --- |
| Phase/Stage | design/process document sequence |
| Slice | implementation build order inside the Stage 17 build plan |

`docs/design/tower/build_plan.md` is itself:

```text
Phase 6 / Stage 17 / Action: Break implementation into concrete file/function/test actions
```

Inside that Stage 17 document, implementation proceeded through:

```text
Slice 1
...
Slice 8
```

There was no pre-existing Slice 9 or Stage 9 in the accepted build plan.

The owner asked to pause and create a post-Slice-8 plan rather than continue by
momentum.

## Post-Slice-8 Assessment Document

The following document was created:

```text
docs/design/tower/post_slice_8_build_plan.md
```

Purpose:

| Purpose | Meaning |
| --- | --- |
| pause work | all new tower feature work paused until new plan accepted |
| reconcile | compare original Slice 1-8 plan to actual implementation |
| audit | record what is real, skeletal, deferred |
| plan future | propose future work without automatically approving it |

Major sections:

| Section | Content |
| --- | --- |
| Stage Location | identifies this as post-Slice-8 assessment |
| Pause Gate | explicitly stops new feature work |
| Source Plans | lists source docs |
| Current Implementation State | lists existing tower modules/tests |
| Slice Completion Audit | audits Slices 1-8 |
| Verification Evidence | records latest test evidence |
| Known Deviations | describes where implementation concretized outlined slices |
| What Is Real | lists actual implemented contracts |
| What Is Skeletal | lists non-final capabilities |
| Questions | originally open, later resolved from owner answers |
| Future Game Plan | owner-selected order |
| Immediate Next Action | accept/revise, then plan next slice |

The owner then requested a hierarchical list of all questions that needed
answers before approving it.

## Post-Slice-8 Questions Worksheet

The following worksheet was created:

```text
docs/design/tower/post_slice_8_questions.md
```

Purpose:

| Purpose | Meaning |
| --- | --- |
| answer questions | owner-answer worksheet |
| preserve hierarchy | group questions by product/model/training/music/future plan |
| guide assessment update | answers should update `post_slice_8_build_plan.md` |

Question groups:

```text
1. Product Direction
2. Model Architecture
3. Training Protocol
4. Music Semantics
5. Future Game Plan Selection
```

The owner filled in answers and embedded follow-up questions. The engineer then
added consultant-answer blocks directly into the worksheet without modifying the
owner's existing text.

The owner then filled in follow-up answers and instructed:

```text
Do not modify docs/design/tower/post_slice_8_questions.md further.
Modify docs/design/tower/post_slice_8_build_plan.md accordingly.
```

From that point forward, the questions worksheet was treated as read-only.

## Owner Answers Incorporated Into Post-Slice-8 Build Plan

The updated `post_slice_8_build_plan.md` now incorporates these owner decisions:

| Topic | Decision |
| --- | --- |
| next priority | learnable real policy architecture |
| audible milestone | final no-train inference episode converted to MIDI artifact |
| reward richness | do not block first real-policy/training path on major reward expansion |
| weak rewards | acceptable for infrastructure, but owner wants to see training working |
| architecture | shared transformer-family across ranks |
| rank differences | config-driven tensor/action differences |
| observation source | derive from old `rl_counterpoint` timed-window transformer pattern and tower docs |
| parent diagnostics | do not feed parent logprobs/top-m candidates/distributions to child |
| child parent awareness | child constrained by lift fiber over sampled parent action |
| rank-1 promotion | episode count for now |
| musical acceptance | final episode should realize rank-local perfect cadence goal |
| rank-2 parent | one accepted rank-1 checkpoint |
| parent action sampling | frozen parent policy samples top-m actions |
| real-training `parent_top_m` | 3 |
| lineage | record one parent checkpoint dependency |
| cadence/motion/dissonance/voice-leading/harmony | include later from TC21M source |
| suspensions | defer to later style update such as `beta.1` |
| six-four logic | defer |
| rank-k voice-leading ownership | deferred for owner to revisit |

The updated build plan also now cites:

```text
assets/rules/tc21m_rules.md
```

as the source for future reward/shaping work.

Important answer about TC21M:

The owner observed that many shaping rewards already appear in:

```text
assets/rules/tc21m_rules.md
```

The plan now reflects that future reward expansion should not be invented from
scratch.

## Post-Slice-8 Phase.Stage.Action Plan

After the owner accepted the updated post-Slice-8 build plan as satisfactory,
they asked whether it was ready to turn into a super explicit
Phase.Stage.Action plan for the next big build.

The engineer confirmed yes and created:

```text
docs/design/tower/post_slice_8_phase_stage_action_plan.md
```

This document is the current next-gameplan source.

It is a planning document, not implementation approval.

Its build goal is:

```text
rank-local tower window
-> model-ready observation tensor
-> transformer-family rank policy
-> artifact-backed training runner
-> final no-train inference episode
-> MIDI artifact
```

The plan's phases are:

| Phase | Name | Purpose |
| --- | --- | --- |
| 1 | Freeze Next-Build Contracts | define exact contracts before code |
| 2 | Real Policy Observation Contract | create tower-local tensorization |
| 3 | Transformer-Family Rank Policy | implement real rank-local policy |
| 4 | Tower MIDI Artifact | write tower trajectories to MIDI |
| 5 | Training Runner | create artifact-backed training runner |
| 6 | First Training Evidence | run tests and tiny training evidence |
| 7 | Reward Expansion Planning | plan narrow TC21M-derived reward expansion |

The dependency graph recorded in the plan is:

```text
Phase 1 contracts
  -> Phase 2 observation
  -> Phase 3 transformer policy
  -> Phase 4 MIDI export
  -> Phase 5 runner
  -> Phase 6 evidence
  -> Phase 7 reward expansion planning
```

The plan also notes:

```text
Phase 4 can be implemented before Phase 3 if desired,
because MIDI export uses trajectories rather than model internals.
Phase 5 should not be implemented before Phases 2 and 3 are accepted,
unless the runner is explicitly built with a temporary toy policy only.
```

## Current Next Action

The current next proposed action is:

```text
Post-Slice-8 Phase 1.Stage 1.1.Action 1.1.1:
Specify Rank Training Lifecycle
```

This is a documentation action.

It would create:

```text
docs/design/tower/training_runner_contract.md
```

The action should specify:

| Item | Decision to record |
| --- | --- |
| rank-1 training | episode-count stop condition |
| rank-2 training | one accepted rank-1 parent checkpoint |
| final evaluation | one no-train inference episode after training |
| final artifact | MIDI written from final inference trajectory |
| checkpoint status | record cadence/MIDI evidence even if acceptance is episode-budget based |
| rank-2 parent sampling | `parent_top_m`, real default 3 |

No code should be written before the owner discusses and approves this next
action.

## Current Design Documents Created Or Updated In This Session

Created:

```text
docs/prime_directive/consultant_tricks.md
docs/design/tower/post_slice_8_build_plan.md
docs/design/tower/post_slice_8_questions.md
docs/design/tower/post_slice_8_phase_stage_action_plan.md
docs/engineer_continuity/2026/04/20/01_001_post_slice_8_tower_implementation_and_planning_report.md
```

Updated:

```text
docs/design/tower/post_slice_8_build_plan.md
```

The questions worksheet was modified once to add consultant answers, then later
treated as read-only after the owner asked for no further changes to it:

```text
docs/design/tower/post_slice_8_questions.md
```

## Current Tower Source Shape

At the end of Slice 8, tower source existed in these groups:

```text
tower/__init__.py
tower/state_action.py
tower/window.py

tower/action/__init__.py
tower/action/assembly.py

tower/graph/__init__.py
tower/graph/actions.py
tower/graph/legality.py
tower/graph/projection.py
tower/graph/spec.py

tower/policy/__init__.py
tower/policy/base.py
tower/policy/samplers.py

tower/reward/__init__.py
tower/reward/context.py
tower/reward/result.py
tower/reward/success.py
tower/reward/terms.py

tower/train/__init__.py
tower/train/checkpoint.py
tower/train/config.py
tower/train/losses.py
tower/train/protocol.py
tower/train/rollout.py
tower/train/trajectory.py
```

Future planned source from the new Phase.Stage.Action plan includes:

```text
tower/policy/observation.py
tower/policy/transformer.py
tower/music/__init__.py
tower/music/render.py
tower/train/runner.py
scripts/tower_train.py
```

These future files are not yet approved for implementation.

## Current Tower Test Shape

At the end of Slice 8, tower tests existed in these groups:

```text
tests/tower/test_import_boundaries.py
tests/tower/test_state_action.py
tests/tower/test_window.py

tests/tower/action/test_assembly.py

tests/tower/graph/test_actions.py
tests/tower/graph/test_legality.py
tests/tower/graph/test_projection.py
tests/tower/graph/test_spec.py

tests/tower/policy/test_base.py
tests/tower/policy/test_samplers.py

tests/tower/reward/test_context.py
tests/tower/reward/test_result.py
tests/tower/reward/test_success.py
tests/tower/reward/test_terms.py

tests/tower/train/test_checkpoint.py
tests/tower/train/test_config.py
tests/tower/train/test_losses.py
tests/tower/train/test_protocol.py
tests/tower/train/test_rollout.py
tests/tower/train/test_trajectory.py
```

Future planned tests from the new Phase.Stage.Action plan include:

```text
tests/tower/policy/test_observation.py
tests/tower/policy/test_transformer.py
tests/tower/music/test_render.py
tests/tower/train/test_runner.py
tests/tower/train/test_runner_script.py
```

These future tests are not yet approved for implementation.

## Current Verification Commands And Known Good Results

Most recent known-good commands:

```bash
uv run pytest tests/tower/train/test_protocol.py
uv run pytest tests/tower/policy tests/tower/train
uv run pytest tests/tower
uv run pytest
```

Most recent known-good results:

| Command | Result |
| --- | --- |
| `uv run pytest tests/tower/train/test_protocol.py` | 21 passed |
| `uv run pytest tests/tower/policy tests/tower/train` | 144 passed |
| `uv run pytest tests/tower` | 246 passed |
| `uv run pytest` | 401 passed |

Future engineers should rerun focused tests after any changes. Do not assume
these counts remain stable after new files are added.

## Important Distinctions For Future Engineers

### Stage 17 vs Slice 8

Do not confuse:

| Term | Meaning |
| --- | --- |
| Phase 6 / Stage 17 | design-process stage that produced `build_plan.md` |
| Slice 1-8 | implementation sequence inside that build plan |
| post-Slice-8 | current pause/planning state after completing original Slice 8 |

The current next work is not "Stage 18" unless the owner explicitly names it so.
The active new planning source is:

```text
docs/design/tower/post_slice_8_phase_stage_action_plan.md
```

### Runner First vs Policy Priority

The owner selected "Tower Training Runner" as the first post-8 planning target,
but also clarified that the product priority is real learnable policy
architecture.

Therefore, future planning must be careful:

| Apparent order | Real dependency |
| --- | --- |
| runner plan first | yes, as planning |
| runner code first | not before observation/policy contracts unless explicitly toy-policy-only |
| real training evidence | requires observation contract and transformer policy |

### MIDI Requirement

The owner does not need continuous MIDI exports during training.

The requirement is:

```text
After training ends, run one final no-train inference episode and save it as MIDI.
```

This final MIDI artifact is important for inspecting training efficacy.

### Reward Requirement

The current minimal reward layer is acceptable for the first real-policy path,
but the owner wants to see training working, not merely weak infrastructure
smoke tests.

Reward expansion should later use:

```text
assets/rules/tc21m_rules.md
```

Suspensions should be deferred to a later style update such as:

```text
beta.1
```

### Parent-Child Model Information

For rank-2 child training:

| Parent data | Child input? |
| --- | --- |
| sampled parent action | operationally yes, via lift-fiber constraint |
| parent logprob | no, diagnostic only |
| parent top-m candidates | no |
| parent distribution/logits | no |

The child should not secretly regain the capacity burden of the old flat model
by receiving rich parent-policy distribution data.

## Recommended Next Engineer Behavior

The next engineer should:

1. Read this report.
2. Read `docs/design/tower/post_slice_8_build_plan.md`.
3. Read `docs/design/tower/post_slice_8_phase_stage_action_plan.md`.
4. Treat `docs/design/tower/post_slice_8_questions.md` as read-only unless the owner explicitly reopens it.
5. Discuss the next action with the owner:

```text
Post-Slice-8 Phase 1.Stage 1.1.Action 1.1.1:
Specify Rank Training Lifecycle
```

6. Only after approval, create:

```text
docs/design/tower/training_runner_contract.md
```

7. Do not implement code until the relevant documentation actions are accepted
   and the owner approves a code action.

## End State Of This Session

The session ends with:

| Area | State |
| --- | --- |
| original tower Slice 1-8 implementation | complete |
| full test suite | last known `401 passed` |
| post-Slice-8 assessment | created and owner accepted as satisfactory |
| post-Slice-8 questions | answered and used as source |
| new Phase.Stage.Action plan | created |
| current next action | documentation: training runner lifecycle contract |
| code implementation | paused pending next owner approval |

The project is in a much better place than at session start:

```text
Before: design was ready for Slice 1 implementation.
After: Slice 1-8 are implemented and tested, and the next big build has an
explicit Phase.Stage.Action plan.
```

Proceed deliberately from here.
