# Rank-3 Preflight Design Decisions

This note answers one question:

- before we start implementing tier/rank 3 in `tower`, which design decisions are
  already implicitly settled by legacy `rl_counterpoint`, and which ones do we
  still need to choose explicitly?

The goal is to avoid starting rank 3 with fuzzy ownership boundaries.

## Bottom Line

Rank 3 is **not** blocked on abstract tower math. The basic projection/assembly
machinery already generalizes beyond rank 2.

Rank 3 **is** blocked on musical and training-contract decisions.

Legacy `rl_counterpoint` already answers a meaningful subset of the **graph**
questions. It does **not** answer the `tower`-specific questions about:

- parent-child rollout semantics,
- reward ownership by tier,
- checkpoint lineage for rank-3-over-rank-2,
- staged curricula for rank 3,
- success semantics for tiered triadic cadence.

So the next rank-3 work should begin with a short explicit spec pass, not with
blind implementation.

## What Is Already Implicitly Decided By Legacy

The following decisions are already present in legacy `rl_counterpoint`, even if
they were never written as "rank-3 design decisions."

### 1. Nodes Should Be Hard-Pruned Musically, Not Cleaned Up Later By Reward

Legacy `G(n)_0` is not a permissive pitch box. It is a trimmed node set defined by:

- pitch range,
- strict voice ordering,
- adjacent vertical interval constraints,
- outer interval constraints,
- lower/root pitch-class constraints.

See:

- [rl_counterpoint/graph/graph_spec.py](/Users/foster/rl_counterpoint/rl_counterpoint/graph/graph_spec.py)
- [rl_counterpoint/graph/state_space.py](/Users/foster/rl_counterpoint/rl_counterpoint/graph/state_space.py)

This strongly suggests the following rank-3 design decision is already implicit:

- `G(3)_0` should be built from **hard musical admissibility**, not from a loose
  graph plus soft penalties.

### 2. Root Is Interpreted Relative To A Fixed Tonic

Legacy uses:

- `allowed_root_intervals_mod_12`
- `allowed_root_pitch_classes`

relative to a pre-given tonic.

So one implicit decision already made is:

- rank-3 legality should still be defined relative to the global tonic, not
  relative only to the current pedal or immediate local sonority.

This aligns with the new rank-1 diatonic prune we just added in `tower`.

### 3. Outer-Interval Constraints Are Structural

Legacy treats outer interval admissibility as part of node legality, via:

- `max_chord_width`
- `allowed_outer_intervals_mod_12`

This implies that for rank 3 we should decide outer-interval legality as a graph
question, not postpone it into reward.

### 4. Adjacent-Interval Constraints Are Structural

Legacy also treats adjacent vertical intervals structurally:

- forbidden adjacent intervals,
- maximum adjacent interval,
- strict increasing order.

For rank 3 that means the adjacent intervals

\[
\lambda_1 - \lambda_0,\qquad \lambda_2 - \lambda_1
\]

should almost certainly be part of `G(3)_0` legality, not merely reward shaping.

### 5. Voice Crossing Is A Hard Edge Constraint

Legacy edge legality forbids voice crossing by default:

- [rl_counterpoint/graph/non_crossing.py](/Users/foster/rl_counterpoint/rl_counterpoint/graph/non_crossing.py)

That answer clearly ports:

- rank-3 transitions should hard-prune crossing, not merely discourage it.

### 6. Parallel Perfect Fifths Are A Hard Edge Constraint

Legacy hard-prunes parallel fifths at the edge level.

That does **not** fully answer all parallel-motion policy for rank 3, but it does
implicitly answer:

- at least some parallel-perfect constraints belong in graph pruning, not reward.

### 7. Node And Edge Contracts Should Be Spec-Driven

Legacy has a small graph spec object that owns graph knobs.

This implies a useful architectural decision for rank 3:

- if rank-3 legality needs new musical knobs, they should live in a proper graph
  spec / reward config contract, not get smuggled into ad hoc rollout code.

## What Legacy Only Partially Answers

These are areas where legacy gives us a direction, but not a complete `tower`
decision.

### 1. Three-Voice Node Pruning Pattern

Legacy supports generic `n`, so it implicitly says "yes, use the same style of
trimmed graph for 3 voices."

But it does **not** answer exactly which interval-class sets we want in present
`tower`, because we have already diverged from legacy in rank 2:

- we removed perfect fourths,
- we changed some root/diatonic assumptions,
- we added tiered projection-induced pruning.

So the general pattern is answered, but the exact `tower` interval vocabulary is
still ours to choose.

### 2. Single-Line Motion Bounds

Legacy uses `max_single_line_interval`.

In `tower`, bounded action space and `max_step_size` currently play a similar role.

So for rank 3 we still need to decide:

- is `max_step_size` enough, or
- do we also want an explicit edge predicate in the style of legacy?

### 3. Root/Outer/Adjacent Ownership In A Tiered Tower

Legacy assumes a one-shot `G(n)` graph.

`tower` adds a new question:

- when `G(3)` projects to `G(2)` and `G(1)`, which legality constraints belong
  intrinsically to rank 3, and which should be inherited downward only through
  projection image?

Legacy does not answer that because it has no tiered scaffold training rule.

## Design Decisions We Still Need To Make Before Rank 3

These are the real preflight decisions.

### 1. What Is The New Voice In Rank 3?

This is the biggest one.

The projection/assembly code already implies the new rank-3 coordinate is inserted
 at index `1`:

\[
(\lambda_0, \lambda_2) \leadsto (\lambda_0, \lambda_1, \lambda_2)
\]

See:

- [tower/action/assembly.py](/Users/foster/rl_counterpoint/tower/action/assembly.py)
- [tower/graph/projection.py](/Users/foster/rl_counterpoint/tower/graph/projection.py)

So the implicit mathematical choice is:

- rank 3 adds an **interior voice**, not a new outermost voice.

What is still not explicitly written down is the musical interpretation:

- is rank 3 always "insert the inner third voice between bass and top"?
- is there ever any alternate insertion policy?

This should be made explicit before implementation.

### 2. Exact Rank-3 Node Legality Contract

We need to decide exactly which of these are hard-pruned in `G(3)_0`:

1. lower voice must be valid rank-1 note over tonic
2. projected rank-2 pair must be valid rank-2 state
3. both adjacent vertical intervals must be in an allowed consonance set
4. outer interval must be in an allowed set
5. total chord width bound
6. strict increasing order
7. perhaps diatonic membership for the inserted voice

This is the most important unresolved graph decision.

### 3. Exact Rank-3 Edge Legality Contract

We need to decide which edge predicates are hard:

1. no self-loop
2. no voice crossing
3. no stationary voices, if that should generalize past rank 2
4. no parallel fifths between any voice pair
5. whether to prune parallel octaves too
6. whether to prune other parallel perfects
7. whether to impose explicit per-voice motion caps beyond `max_step_size`

Legacy answers some of this, but not all.

### 4. Projection-Induced Pruning Direction

We already decided for rank 1:

\[
G(1)_\bullet \leftarrow \operatorname{pr}^2(G(2)_\bullet)
\]

For rank 3 we need the next analogue:

\[
G(2)_\bullet \leftarrow \operatorname{pr}^3(G(3)_\bullet)
\]

Questions still to settle:

1. do we build a cached induced rank-2 artifact from rank 3 just as we do for rank 1 from rank 2?
2. does rank-2 training eventually switch to that induced artifact?
3. do we keep the currently hand-tightened rank-2 graph as a base, or replace it entirely by the projection image?

This is a major architectural decision.

### 5. Parent Policy For Rank 3

Rank 2 trains over one accepted rank-1 parent checkpoint.

For rank 3 we need to decide:

1. does one rank-3 run consume exactly one accepted rank-2 checkpoint?
2. is the rank-2 parent frozen exactly as rank-1 is frozen for rank 2?
3. do we keep top-`m` parent sampling?
4. do we keep the feasibility filter over the child lift fiber?

The likely answer is "yes" to all four, but it should be made explicit.

### 6. Reward Ownership For Rank 3

Legacy gives no direct answer here.

We need to decide what rank-3 reward is allowed to score.

The rank-2 policy we adopted has leaned toward:

- score facts introduced by the new voice,
- avoid leaking lower-tier responsibilities upward.

For rank 3 that becomes:

- should reward only evaluate intervals involving the inserted inner voice?
- can it score triadic sonority globally?
- can it score spacing on both sides of the inserted voice?
- how much terminal cadence logic belongs at rank 3 versus rank 2?

This is a genuinely open design decision.

### 7. Rank-3 Success Semantics

We need an explicit lifted cadence predicate for rank 3.

Questions:

1. must projected rank-2 success already hold?
2. what exact terminal triadic sonority is required?
3. what pitch classes must the inserted voice realize at pre-cadential and final steps?
4. is there one accepted cadence pattern or several?

Legacy does not answer this in `tower` terms.

### 8. Coupled/Decoupled Curriculum For Rank 3

For rank 1 and rank 2 we now use staged curricula.

Before implementing rank 3 we should decide whether we want:

1. rank-3 stage 1: inserted voice starts in target/root octave or target-register band
2. rank-3 stage 2: decoupled start/goal

and more importantly:

- what "coupled" even means for an inserted interior voice.

That is still unresolved.

### 9. Register Contract For The Inserted Voice

For rank 3, the new voice is interior.

So we need to decide:

1. should it be constrained between the lower and upper parent voices only by strict ordering?
2. do we impose minimum spacing from both neighbors?
3. do we bias it toward a preferred band?
4. does it have its own target octave at all, or is that the wrong concept for an interior voice?

This is currently open, and it affects both graph and reward design.

### 10. Whether Rank-3 Should Be Built As Generic Rank-k Infrastructure Or A Concrete Rank-3 Slice

This is a project-shape decision.

We need to decide whether the next step is:

- "implement rank 3 concretely, end to end,"

or:

- "generalize rank-2 machinery into true rank-k machinery before rank 3."

Given where the repo is today, the safer recommendation is:

- implement rank 3 as a **concrete slice first**, while keeping names and data
  shapes generic where cheap.

## Recommended Decisions Before Coding Rank 3

These are the decisions I would lock first.

### Already Safe To Treat As Settled

1. rank 3 inserts an interior voice at projection index `1`
2. rank-3 graph should be hard-pruned, not reward-cleaned
3. strict ordering, no crossing, and at least some parallel-perfect pruning are graph concerns
4. tonic-relative legality remains global, not merely pedal-relative
5. rank-3 should train over one accepted rank-2 parent checkpoint

### Still Need Explicit Spec

1. exact allowed adjacent and outer interval classes for `G(3)_0`
2. exact rank-3 terminal cadence predicate
3. whether inserted voice has a goal octave / goal register at all
4. which rank-3 facts belong to reward versus hard legality
5. whether to induce `G(2)` from `G(3)` immediately or postpone that cutover

## Practical Recommendation

Before coding tier 3, write one short contract doc covering:

1. rank-3 node legality
2. rank-3 edge legality
3. rank-3 parent-child rollout contract
4. rank-3 reward slice A
5. rank-3 terminal success predicate

That should be enough to start implementation without opening another design fog bank.

