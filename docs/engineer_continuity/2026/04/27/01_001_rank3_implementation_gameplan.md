# Rank-3 Implementation Gameplan

Date: 2026-04-27  
Scope: extremely detailed implementation plan for bringing actual rank-3 / tier-3
training into the `tower/` stack.

This note is intended to be a build handoff, not a design-brainstorm artifact.
The design is now far enough along that we can sequence the implementation work.

## Executive Summary

Rank 3 is now blocked mainly by missing implementation, not by missing abstract
math.

The most important accepted design decisions entering this plan are:

1. rank 3 inserts an **interior voice**
2. final-rank construction modifies the **whole tower of graphs**
3. rank-3 graph legality should be **hard-pruned**
4. allowed interval classes should use the current rank-2-style set:
   $$
   \{3,4,7,8,9\}\pmod{12}
   $$
5. rank-3 edges should hard-prune:
   - voice crossing
   - stationary voices
   - parallel fifths
   - parallel octaves
6. `max_step_size` is sufficient as the motion cap for the first pass
7. rank-3 reward ownership is **global-triad**
8. inserted interior voice has **no goal octave/register**
9. rank 3 should use **no staging initially**
10. rank 3 should be implemented as a **concrete slice first**, not a rank-`k`
    refactor-first exercise
11. rank-3 success is:
    - pedal in goal octave, and
    - a perfect cadence of triads in that octave

The highest-level implementation strategy is:

1. write the explicit rank-3 contract docs
2. implement rank-3 graph legality and rank-3-induced lower-tier graph artifacts
3. implement rank-3 reward slice A
4. implement rank-3 rollout/training/runner/script
5. run rank-3 smoke lineage
6. only then consider longer training or generalization cleanup

## Current Ground Truth

The repo today has:

- generic projection machinery that already supports rank 3:
  - [tower/graph/projection.py](/Users/foster/rl_counterpoint/tower/graph/projection.py)
- generic action assembly that already supports rank 3:
  - [tower/action/assembly.py](/Users/foster/rl_counterpoint/tower/action/assembly.py)
- generic action-space and lift-fiber helpers that can represent rank 3:
  - [tower/graph/actions.py](/Users/foster/rl_counterpoint/tower/graph/actions.py)
- rank-1 and rank-2 training infrastructure only:
  - [tower/train/protocol.py](/Users/foster/rl_counterpoint/tower/train/protocol.py)
  - [tower/train/runner.py](/Users/foster/rl_counterpoint/tower/train/runner.py)
- rank-1 and rank-2 reward infrastructure only:
  - [tower/reward/factory.py](/Users/foster/rl_counterpoint/tower/reward/factory.py)
  - [tower/reward/melody.py](/Users/foster/rl_counterpoint/tower/reward/melody.py)
  - [tower/reward/harmony.py](/Users/foster/rl_counterpoint/tower/reward/harmony.py)
- rank-1 induced graph-from-rank-2 machinery:
  - [tower/graph/induced.py](/Users/foster/rl_counterpoint/tower/graph/induced.py)

What does **not** yet exist:

- rank-3 legality
- rank-3 induced rank-2 artifact
- rank-3 reward factory
- rank-3 success predicate
- rank-3 rollout/training path
- rank-3 script
- rank-3 smoke test

So the right posture is:

- do not start with big cleanup
- do not start with generic rank-`k`
- do not start by tuning musical reward weights
- first create a minimal but real rank-3 vertical slice

## Primary Implementation Risks

These are the main places rank-3 work can go wrong.

### 1. Mixing Up The Mathematical Tower With Convenience Implementations

The new final-rank rule is stronger than the earlier induced-rank1 correction.

For final rank 3, the build should conceptually be:

$$
G(3)_\bullet \rightsquigarrow (\operatorname{pr}^{3})^{-1}\!\big((\operatorname{pr}^{2})^{-1}(G(1)_\bullet)\big)
$$

then:

$$
G(2)_\bullet \leftarrow \operatorname{pr}^{3}(G(3)_\bullet)
$$

and then:

$$
G(1)_\bullet \leftarrow \operatorname{pr}^{2}(G(2)_\bullet)
$$

Risk:

- implementing only a local rank-3 graph without rebuilding the lower tiers from
  it

Mitigation:

- treat induced-rank2-from-rank3 artifact generation as part of rank-3 graph work,
  not as a later optimization

### 2. Letting Reward Compensate For Missing Graph Pruning

We already learned in rank 2 that bad graph permissiveness creates ugly musical
regions and then reward has to fight uphill.

Risk:

- implementing rank-3 reward before rank-3 graph legality is hard enough

Mitigation:

- graph legality first
- reward second

### 3. Over-Generalizing Too Early

Risk:

- turning this into a rank-`k` framework refactor before rank 3 is working

Mitigation:

- implement rank 3 concretely
- only generalize where the code already wants it

### 4. Under-Specifying Triadic Cadence Success

Risk:

- creating a vague "rank-3 cadence" term that does not actually encode the
  owner's chosen success semantics

Mitigation:

- define success predicate explicitly before training code

### 5. Breaking Existing Rank-1 / Rank-2 Runs

Risk:

- rank-3 changes silently perturb lower-tier active workflows

Mitigation:

- keep all new rank-3 code paths additive at first
- preserve rank-1/rank-2 scripts as-is until rank-3 artifact induction is
  intentionally switched on

## Detailed Implementation Sequence

This is the recommended order.

## Phase 1: Lock The Rank-3 Contract In Docs

Goal:

- remove remaining ambiguity before code changes

Deliverables:

1. rank-3 node legality contract doc
2. rank-3 edge legality contract doc
3. rank-3 success predicate doc
4. rank-3 reward slice A doc
5. rank-3 graph-induction contract doc

Recommended filenames:

- `docs/design/tower/rank3_node_edge_contract.md`
- `docs/design/tower/rank3_success_contract.md`
- `docs/design/tower/rank3_reward_slice_a_contract.md`
- `docs/design/tower/rank3_induced_graph_contract.md`

Minimum content to write:

### Rank-3 node legality

The first implementation should hard-require:

1. state length exactly 3
2. pitch range bounds
3. strict ordering
4. projected lower voice valid in rank 1
5. projected outer pair valid in rank 2
6. lower adjacent interval class in `{3,4,7,8,9}`
7. upper adjacent interval class in `{3,4,7,8,9}`
8. outer interval class in `{3,4,7,8,9}`
9. width bounded by chosen rule

Need to specify explicitly whether "projected outer pair valid in rank 2" is
evaluated using the currently active rank-2 legality contract or a copied subset.

Recommendation:

- use the active rank-2 legality contract

### Rank-3 edge legality

The first implementation should hard-require:

1. source legal
2. target legal
3. no self-loop
4. no voice crossing
5. no stationary coordinates
6. no parallel fifths for any voice pair
7. no parallel octaves for any voice pair
8. projected edge valid in rank 2 after induction cutover

Need to spell out whether all pairwise perfects are tested among:

- `(0,1)`
- `(1,2)`
- `(0,2)`

Recommendation:

- yes, all three pairs

### Rank-3 success predicate

Need exact triadic cadence language, not prose only.

Current owner rule:

- pedal in goal octave
- perfect cadence of triads in that octave

This should be expanded into concrete pitch-class and octave tests over the final
and penultimate states.

This is the only remaining substantial unresolved design area.

## Phase 2: Implement Rank-3 Graph Legality

Goal:

- create `G(3)_bullet` legality before any training/reward work

Primary files:

- [tower/graph/legality.py](/Users/foster/rl_counterpoint/tower/graph/legality.py)
- [tower/graph/spec.py](/Users/foster/rl_counterpoint/tower/graph/spec.py)
- [tests/tower/graph/test_legality.py](/Users/foster/rl_counterpoint/tests/tower/graph/test_legality.py)
- [tests/tower/graph/test_actions.py](/Users/foster/rl_counterpoint/tests/tower/graph/test_actions.py)

Implementation tasks:

1. Add rank-3 interval-class constants
2. Add rank-3 width-bound helper
3. Add rank-3 pairwise parallel-perfect detection helpers
4. Add rank-3 stationary-voice helper
5. Add rank-3 state legality branch inside `is_valid_state`
6. Add rank-3 transition legality branch inside `is_valid_transition`

Recommended helper structure:

- `_rank3_adjacent_intervals(state)`
- `_rank3_outer_interval(state)`
- `_rank3_has_stationary_voice(action)`
- `_rank3_has_voice_crossing(source, target)`
- `_rank3_has_parallel_fifth(source, target)`
- `_rank3_has_parallel_octave(source, target)`

Important design rule:

- rank-3 legality should reuse lower-rank legality where possible

Concretely:

- build a projected rank-1 spec and require `is_valid_state((state[0],), projected_rank1_spec)`
- build a projected rank-2 spec and require `is_valid_state(project_state(state), projected_rank2_spec)`

This keeps the tower honest.

Testing tasks:

1. valid rank-3 state example
2. invalid adjacent interval example
3. invalid outer interval example
4. invalid projected rank-2 example
5. invalid projected rank-1 example
6. crossing edge rejection
7. stationary-voice rejection
8. parallel fifth rejection
9. parallel octave rejection

## Phase 3: Implement Induced Rank-2-From-Rank-3 Artifact

Goal:

- make final-rank-3 construction able to prune `G(2)` and then `G(1)`

Primary files:

- [tower/graph/induced.py](/Users/foster/rl_counterpoint/tower/graph/induced.py)
- [tests/tower/graph/test_induced.py](/Users/foster/rl_counterpoint/tests/tower/graph/test_induced.py)

Current state:

- induced graph machinery exists only for rank-2 to rank-1

Implementation tasks:

1. generalize the induced artifact writer to support:
   - source rank 2 -> target rank 1
   - source rank 3 -> target rank 2
2. keep deterministic payload and cache key behavior
3. include legality-contract versioning in cache key
4. include all graph-contract-relevant fields in cache key

Recommended artifact structure:

- keep the same shape:
  - `source_spec`
  - `node_image`
  - `edge_image`
  - `schema_version`
  - `legality_contract_version`

But parameterize rank instead of hardcoding rank-1 image.

Need new API shape:

- `build_induced_graph_payload(source_spec=..., target_rank=...)`
- `write_induced_graph_artifact(source_spec=..., target_rank=..., artifact_root=...)`

Keep the older rank-1 helper as a thin wrapper if that preserves lower-tier
stability.

Testing tasks:

1. rank-3 -> rank-2 projected node image deterministic
2. rank-3 -> rank-2 projected edge image deterministic
3. self-loop filtering where needed
4. cache key changes when legality contract changes

## Phase 4: Decide And Implement Rank-3-Induced Lower-Tier Consumption

Goal:

- make lower tiers optionally consumable from final-rank-3 induction

Primary files:

- [tower/train/runner.py](/Users/foster/rl_counterpoint/tower/train/runner.py)
- [tower/graph/spec.py](/Users/foster/rl_counterpoint/tower/graph/spec.py)

Questions to operationalize:

1. does rank-2 training under final-rank-3 mode use induced rank-2 immediately?
2. does rank-1 training under final-rank-3 mode use rank-3-induced rank-2, then
   induced rank-1 from that?

Recommended first implementation:

- keep rank-1/rank-2 standalone workflows unchanged by default
- add explicit final-rank-3 graph-construction mode
- only rank-3 workflows activate the top-down induced tower rebuild

This reduces regression risk.

## Phase 5: Implement Rank-3 Reward Slice A

Goal:

- create the first trainable rank-3 reward bundle

Primary files:

- [tower/reward/factory.py](/Users/foster/rl_counterpoint/tower/reward/factory.py)
- [tower/reward/harmony.py](/Users/foster/rl_counterpoint/tower/reward/harmony.py)
- [tower/reward/success.py](/Users/foster/rl_counterpoint/tower/reward/success.py)
- tests under:
  - [tests/tower/reward/](/Users/foster/rl_counterpoint/tests/tower/reward)

Owner decision already made:

- reward ownership is global-triad
- no inserted-voice octave-goal term

Likely first reward bundle:

1. terminal rank-3 cadence success reward
2. triadic consonance reward over both adjacent intervals and outer interval
3. spacing reward for healthy triad spread
4. maybe cadence-endpoint shaping if terminal reward is too sparse

Avoid for first pass:

- separate interior octave-goal reward
- stylistic flourishes before cadence works

Need new objects:

- `Rank3RewardFactoryConfig`
- `Rank3RewardFunction`
- `build_rank3_reward_fn(...)`

Likely new reward terms:

- `Rank3TriadConsonanceReward`
- `Rank3SpacingControlReward`
- `Rank3CadenceEndpointReward` if needed

Testing tasks:

1. rank-3 reward rejects non-rank-3 context
2. valid triad scores positively
3. invalid-but-reachable test fixtures rejected or penalized appropriately
4. success predicate wiring works

## Phase 6: Implement Rank-3 Success Predicate

Goal:

- formalize terminal success for tier 3

Primary files:

- [tower/reward/success.py](/Users/foster/rl_counterpoint/tower/reward/success.py)
- [tests/tower/reward/test_success.py](/Users/foster/rl_counterpoint/tests/tower/reward/test_success.py)

Required work:

1. translate the owner’s rule into exact code:
   - pedal in goal octave
   - perfect cadence of triads in that octave
2. specify:
   - required penultimate triad
   - required final triad
   - required pitch classes for the inserted interior voice
   - whether projected rank-2 success is diagnostic, required, or redundant

Recommended first implementation style:

- write the predicate against explicit final and penultimate rank-3 states
- add rich diagnostics explaining failure reason

Do not bury cadence logic inside reward terms.

## Phase 7: Implement Rank-3 Rollout And Protocol

Goal:

- create the first actual rank-3 train episode

Primary files:

- [tower/train/rollout.py](/Users/foster/rl_counterpoint/tower/train/rollout.py)
- [tower/train/protocol.py](/Users/foster/rl_counterpoint/tower/train/protocol.py)
- [tower/train/losses.py](/Users/foster/rl_counterpoint/tower/train/losses.py)
- [tests/tower/train/test_rollout.py](/Users/foster/rl_counterpoint/tests/tower/train/test_rollout.py)
- [tests/tower/train/test_protocol.py](/Users/foster/rl_counterpoint/tests/tower/train/test_protocol.py)

Current state:

- rank-1 rollout exists
- rank-2 parent-first rollout exists
- no rank-3 path exists

Implementation tasks:

1. add rank-3 rollout mode over frozen rank-2 parent
2. project rank-3 state/window to rank-2 for parent policy
3. sample frozen parent action from rank-2 policy
4. compute legal lift fiber for rank 3
5. sample inserted-voice active choice
6. assemble rank-3 action
7. score reward
8. record diagnostics

Need to preserve:

- parent feasibility filtering
- no empty-lift-fiber silent failures
- proper parent/active logprob separation

Potential subtlety:

- with interior-voice insertion, active lift choices may have a different local
  combinatorics than rank 2

This should be explicitly tested.

Testing tasks:

1. valid rollout step over rank-2 parent
2. projected parent state/action recorded correctly
3. feasibility filter removes empty lifts
4. no-stationary rule reflected in active choice set
5. rollout terminates/truncates coherently

## Phase 8: Implement Rank-3 Runner

Goal:

- support artifact-backed rank-3 training runs

Primary files:

- [tower/train/runner.py](/Users/foster/rl_counterpoint/tower/train/runner.py)
- [tests/tower/train/test_runner.py](/Users/foster/rl_counterpoint/tests/tower/train/test_runner.py)

New needed entrypoint:

- `run_rank3_training(...)`

Responsibilities:

1. enforce accepted rank-2 parent checkpoint requirement
2. build / load child policy
3. build / load optimizer
4. build rank-3 graph spec
5. build rank-3 reward function
6. run episode loop
7. append diagnostics
8. write final inference MIDI artifacts

Need companion config behavior:

- `TowerRunnerConfig(rank=3, parent_checkpoint=...)`

This should be implemented conservatively:

- add rank-3 support without altering rank-1/rank-2 behavior

## Phase 9: Implement Rank-3 Script

Goal:

- make rank 3 runnable from the command line

Primary file:

- `scripts/tower_train_rank3.py`

Recommended behavior:

1. find one accepted rank-2 parent checkpoint from lineage
2. reconstruct parent policy from saved config/checkpoint
3. build rank-3 reward slice A
4. run artifact-backed rank-3 training
5. print run dir, checkpoint, final MIDI, terminal summary

Do not stage rank 3 initially.

Testing tasks:

1. parse args defaults
2. tiny end-to-end one-episode run
3. file-path execution
4. missing-parent failure

## Phase 10: Rank-3 Smoke Lineage

Goal:

- prove the whole stack actually works before scaling

Recommended first smoke:

1. rank-1 staged `100/100`
2. rank-2 `100`
3. rank-3 `100`

Only after that should we do:

1. rank-1 `100/100`
2. rank-2 `100`
3. rank-3 `100/100` or larger variants

Smoke evaluation questions:

1. does rank-3 run complete?
2. are there empty lift fibers?
3. are final triads actually constrained musically?
4. does the rank-3 success predicate ever fire?
5. do final inference MIDIs sound structurally cleaner than ad hoc triads?

## Phase 11: Post-Smoke Tightening

Only after the smoke run works:

1. inspect reward diagnostics
2. inspect interval distributions
3. inspect spacing distributions
4. inspect cadence failure reasons
5. then tune reward weights or graph constraints

Do **not** start by tuning before smoke passes.

## Concrete File Checklist

Expected new or substantially changed files:

### New docs

- `docs/design/tower/rank3_node_edge_contract.md`
- `docs/design/tower/rank3_success_contract.md`
- `docs/design/tower/rank3_reward_slice_a_contract.md`
- `docs/design/tower/rank3_induced_graph_contract.md`

### New/changed implementation

- `tower/graph/legality.py`
- `tower/graph/induced.py`
- `tower/graph/spec.py`
- `tower/reward/factory.py`
- `tower/reward/harmony.py`
- `tower/reward/success.py`
- `tower/train/rollout.py`
- `tower/train/protocol.py`
- `tower/train/runner.py`
- `scripts/tower_train_rank3.py`

### New/changed tests

- `tests/tower/graph/test_legality.py`
- `tests/tower/graph/test_induced.py`
- `tests/tower/graph/test_actions.py`
- `tests/tower/reward/test_factory.py`
- `tests/tower/reward/test_harmony.py`
- `tests/tower/reward/test_success.py`
- `tests/tower/train/test_rollout.py`
- `tests/tower/train/test_protocol.py`
- `tests/tower/train/test_runner.py`
- `tests/tower/train/test_rank3_runner_script.py`

## Recommended Execution Order In Actual Work Sessions

The practical session order should be:

1. write rank-3 contract docs
2. implement graph legality
3. implement induced rank-2 artifact
4. implement reward factory + success predicate
5. implement rollout/protocol
6. implement runner
7. implement script
8. run smoke
9. inspect diagnostics

This order matters. It keeps the debugging surface narrow.

## Things Not To Do Yet

1. do not refactor all rank-2 code into generic rank-`k` abstractions first
2. do not introduce a staged rank-3 curriculum immediately
3. do not add interior-voice octave-goal reward
4. do not start tuning long training runs before smoke is green
5. do not silently change rank-1/rank-2 active workflows while adding rank 3

## Success Condition For This Plan

This gameplan is successful when:

1. a new engineer can start rank-3 implementation without guessing the order
2. lower-tier regressions are minimized
3. the first rank-3 smoke run is positioned as a build milestone, not as an
   afterthought

## Immediate Next Action

The next concrete action after this continuity note should be:

- write the explicit rank-3 node/edge contract docs, then implement
  `tower/graph/legality.py` rank-3 branches and induced rank-2-from-rank-3
  artifacts.

