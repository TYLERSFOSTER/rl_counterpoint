# Final-Rank=3 Wiring And Rank-3 Reward/Success Contracts

Date: 2026-04-27  
Context: follow-on continuity after Phase 10.Stage 1.Actions 2-5.

## What Just Landed

The repo now has the first meaningful whole-tower wiring for a known final rank
of 3.

Concretely:

1. rank-3 legality exists
2. induced rank-2-from-rank-3 artifacts exist
3. runner graph construction can now rebuild lower tiers from `final_rank = 3`
4. rank-1 and rank-2 script surfaces now expose the graph knobs needed to use
   that behavior explicitly

This is not yet operational rank-3 training, but it is the lower-tier graph
plumbing that rank-3 training will depend on.

## Files Affected In This Slice

### Graph legality / induction

- [tower/graph/legality.py](/Users/foster/rl_counterpoint/tower/graph/legality.py)
- [tower/graph/induced.py](/Users/foster/rl_counterpoint/tower/graph/induced.py)
- [tower/graph/spec.py](/Users/foster/rl_counterpoint/tower/graph/spec.py)

### Runner / script surface

- [tower/train/runner.py](/Users/foster/rl_counterpoint/tower/train/runner.py)
- [scripts/tower_train.py](/Users/foster/rl_counterpoint/scripts/tower_train.py)
- [scripts/tower_train_staged.py](/Users/foster/rl_counterpoint/scripts/tower_train_staged.py)
- [scripts/tower_train_rank2.py](/Users/foster/rl_counterpoint/scripts/tower_train_rank2.py)
- [scripts/tower_train_rank2_staged.py](/Users/foster/rl_counterpoint/scripts/tower_train_rank2_staged.py)

### Design docs added this round

- [final_rank_graph_construction_contract.md](/Users/foster/rl_counterpoint/docs/design/tower/final_rank_graph_construction_contract.md)
- [rank3_success_contract.md](/Users/foster/rl_counterpoint/docs/design/tower/rank3_success_contract.md)
- [rank3_reward_slice_a_contract.md](/Users/foster/rl_counterpoint/docs/design/tower/rank3_reward_slice_a_contract.md)

## Operational Ground Truth

The important new invariant is:

- if `graph_config.final_rank == 3`,
  lower-tier graph construction is no longer supposed to behave like an
  independent local rebuild

Instead:

1. construct rank-3 source graph
2. prune by rank-3 legality
3. induce rank 2 from that graph
4. induce rank 1 from the induced rank 2 graph

So:

- rank-2 can now be represented as an induced projected graph from rank 3
- rank-1 can now be represented as induced from that induced rank 2 image

This is the first repo state where the “final rank modifies the whole tower”
principle is actually true in runner construction, not just in design prose.

## What Is Still Missing

We still do **not** have:

1. rank-3 reward code
2. rank-3 success code
3. rank-3 rollout/training protocol
4. rank-3 runner
5. rank-3 training script

So the next phase should not re-open graph philosophy. It should implement the
rank-3 reward/success slice directly from the new contracts.

## Contracts Now Ready For Coding

Two new docs should be treated as the source of truth for the next slice:

### Rank-3 success

- [rank3_success_contract.md](/Users/foster/rl_counterpoint/docs/design/tower/rank3_success_contract.md)

Current accepted first-slice rule:

- final pedal in goal octave
- penultimate triad is dominant
- final triad is tonic
- outer voices remain layered with lower-tier cadence semantics
- interior voice completes the dominant triad before cadence and tonic triad at
  cadence

### Rank-3 reward slice A

- [rank3_reward_slice_a_contract.md](/Users/foster/rl_counterpoint/docs/design/tower/rank3_reward_slice_a_contract.md)

Current accepted first-slice reward shape:

- terminal success term
- global-triad consonance term
- global spacing term
- cadence-endpoint triad shaping term

No target-octave reward for the inserted interior voice.

## Testing State

At the end of this slice, the graph and runner-facing tests relevant to lower-tier
rebuilding were green.

Key validations included:

- rank-3 legality tests
- induced rank-2-from-rank-3 artifact tests
- runner tests for:
  - rank-2 induced from final rank 3
  - rank-1 induced from final rank 3 through rank 2
- rank-1 and rank-2 script tests for the new graph-config surface

## Recommendation For The Very Next Action

Do **not** start by writing a runner.

The next action should be:

- implement rank-3 success predicate in `tower/reward/success.py`
- implement rank-3 reward config + factory branch in `tower/reward/factory.py`
- implement rank-3 harmony terms in `tower/reward/harmony.py`

Only after that should we wire rank-3 rollout/training.

That sequencing keeps the next layer honest:

- graph legality already exists
- success/reward contract now exists
- then the training path can be written against something stable
