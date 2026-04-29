# Scaffold-First Initialization Contract

This note defines the correct episode-initialization contract for hierarchical
training in `tower`.

It exists to pin down one specific design invariant:

- hierarchical control is not enough;
- episode initialization must also respect the lower-rank scaffold.

Without that, higher-rank training can be locally hierarchical during rollout
while still being globally mis-specified at episode start.

## Core Rule

For rank \(k \ge 2\), training episodes should start from a lower-rank scaffold
first, not from an arbitrary legal rank-\(k\) full state sampled directly from
\(G(k)_0\).

Operationally:

1. initialize the lower scaffold,
2. derive the parent state from that scaffold,
3. place the active rank on top of that parent structure,
4. train only the active rank's added coordinate.

The active policy should never receive a training curriculum whose diversity
comes primarily from arbitrary full higher-rank state sampling.

## Why This Matters

The tower design says:

- rank 1 learns the pedal line,
- rank 2 learns the added outer voice over the pedal scaffold,
- rank 3 learns the inserted interior voice over the frozen lower scaffold.

That principle has two parts:

1. **control hierarchy**
   - lower ranks are frozen while the active rank learns;
2. **initialization hierarchy**
   - the episode start state is induced from the lower scaffold rather than
     sampled as an arbitrary full higher-rank state.

If only (1) is true but (2) is false, then the training loop is only
partially hierarchical.

## Rank-by-Rank Contract

### Rank 1

Rank 1 is the base tier, so there is no lower scaffold below it.

Its initialization may be configured directly:

- fixed initial pitch,
- sampled initial pitch,
- coupled start/goal octave curriculum,
- decoupled continuation curriculum.

Rank 1 is the only tier where direct state initialization is primary.

### Rank 2

Rank 2 should be initialized from a rank-1 scaffold.

That means:

1. choose or sample the rank-1 initial pedal state,
2. derive the rank-2 parent state from that pedal scaffold,
3. let the rank-2 policy learn only the added outer voice coordinate.

The important prohibition is:

- do **not** sample an arbitrary legal two-voice state
  \[
  (\lambda_0,\lambda_1) \in G(2)_0
  \]
  as the default training start.

The correct source of diversity for rank 2 is:

- variation in the pedal scaffold,
- variation in target octave / task context,
- variation in frozen-parent behavior if explicitly configured.

It is **not** arbitrary legal chord sampling.

### Rank 3

Rank 3 should be initialized from a frozen lower scaffold:

1. initialize the lower scaffold,
2. derive the parent outer pair,
3. insert the active interior-voice slot relative to that parent state,
4. let the rank-3 policy learn only the new interior coordinate.

Equivalently:

- rank 3 starts from a pre-existing pedal + outer-voice scaffold;
- it does not start from an arbitrary sampled legal triad.

The important prohibition is:

- do **not** sample an arbitrary legal three-voice state
  \[
  (\lambda_0,\lambda_1,\lambda_2) \in G(3)_0
  \]
  as the default training start.

The correct source of diversity for rank 3 is:

- variation in the frozen lower scaffold,
- variation in target octave / task context,
- variation in frozen-parent behavior if explicitly configured.

It is **not** arbitrary legal full-triad sampling.

## Decoupling: Correct Meaning

For higher ranks, "decoupled" should refer to task/context variation, not to
arbitrary full-state sampling.

Allowed examples of decoupling:

- target root octave sampled from a configured set,
- lower-scaffold start condition not rigidly tied to the same octave,
- different frozen-parent trajectories over the same class of lower scaffold.

Disallowed example of "decoupling":

- defaulting to `sample_initial_state=True` for rank 2 or rank 3 and drawing
  arbitrary legal full states from the higher-rank node set.

That is not scaffold-first hierarchical training.

## Present Diagnosis

The current codebase already satisfies the **control** part of this contract:

- rank 2 trains over a frozen rank-1 parent,
- rank 3 trains over a frozen rank-1/rank-2 parent stack.

But the codebase has recently violated the **initialization** part in the higher
tiers by allowing rank-2 and rank-3 episode starts to come from arbitrary legal
full-state sampling.

That behavior should be treated as a design mistake, not as the intended tower
contract.

## Required Implementation Consequences

The operational fixes implied by this contract are:

1. rank-2 and rank-3 should not default to arbitrary full-state sampling;
2. rank-2 initialization should be rebuilt from a rank-1 scaffold;
3. rank-3 initialization should be rebuilt from a frozen lower scaffold;
4. final-inference diversity may still sample varied starts if explicitly
   desired, but that should remain separate from the training initialization
   contract.

## Non-Goals

This note does **not** specify:

- exact sampling distributions for the lower scaffold,
- whether higher-rank target-octave sampling should remain enabled by default,
- whether parent scaffolds should be cached or recomputed.

Those are implementation and curriculum questions.

This note specifies only the initialization invariant:

- higher-rank training starts from the lower scaffold first.

## Short Operational Summary

The correct mental model is:

- rank 1: initialize directly;
- rank 2: initialize from pedal scaffold;
- rank 3: initialize from pedal + outer scaffold.

If a rank-\(k\) episode begins by sampling an arbitrary full legal state in
\(G(k)_0\), then the initialization is not correctly hierarchical.
