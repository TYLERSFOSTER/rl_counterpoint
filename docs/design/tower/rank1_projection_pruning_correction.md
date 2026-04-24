# Rank-1 Projection Pruning Correction

## Purpose

This document records a design correction for how rank-1 graph pruning should work in `tower`.

It replaces the current heuristic idea:

- prune rank 1 by reserving some arbitrary amount of upper register "headroom" for future voices

with the stronger tower-native idea:

- prune rank 1 by upward extendability into the already-pruned rank-2 graph

The motivating project-manager point is simple:

> rank 1 should not be pruned by a hand-made ceiling rule when rank 2 already tells us which rank-1 states and edges are genuinely extendable.

## Problem With The Current Heuristic

The current rank-1 ceiling rule says, in effect:

- choose a final chord size `n`
- reserve a fixed number of semitones per future voice
- lower the rank-1 `pitch_max` accordingly

That is a blunt proxy.

It does not actually ask:

- which rank-1 states extend to valid rank-2 states
- which rank-1 transitions extend to valid rank-2 transitions

So it can over-prune:

- remove rank-1 material that would in fact admit good rank-2 lifts

and it can under-prune:

- keep rank-1 material that still cannot be extended into valid rank-2 behavior for the real graph-theoretic reasons

This is especially unsatisfying in `tower`, because the system already has the machinery needed to define the lower-rank scaffold by image under projection.

## Corrected Idea

Build rank 2 first, prune rank 2 by the actual two-voice graph constraints, then project downward.

So instead of:

```text
heuristic headroom -> prune G(1)
```

we should use:

```text
construct G(1)
construct G(2)
prune G(2)
project G(2) downward
induce the surviving part of G(1) from that image
```

This is much more faithful to the tower philosophy.

## Mathematical Contract

Let:

\[
\mathrm{pr}^{2 \to 1} : G(2)_\bullet \to G(1)_\bullet
\]

be the canonical rank projection.

Then the corrected induced rank-1 graph is:

### Node Image

\[
G(1)_0^{\mathrm{induced}}
=
\operatorname{Im}\left(
\mathrm{pr}^{2 \to 1} : G(2)_0 \to G(1)_0
\right).
\]

So a rank-1 node survives iff it is the projection of at least one valid rank-2 node.

### Edge Image

\[
G(1)_1^{\mathrm{induced}}
=
\operatorname{Im}\left(
\mathrm{pr}^{2 \to 1} : G(2)_1 \to G(1)_1
\right).
\]

So a rank-1 edge survives iff it is the projection of at least one valid rank-2 edge.

This second clause is important.

Node-only pruning is not enough, because a rank-1 state may be extendable upward somewhere, while a particular outgoing rank-1 transition from that state may still admit no valid rank-2 lift.

The intended corrected rule is therefore:

- induce both nodes and edges from the image of the pruned rank-2 graph

not merely:

- keep rank-1 nodes that appear as projections of rank-2 nodes

## Practical Interpretation

This means rank 1 should be treated as:

- the base rank in the training order

but not as:

- an independently pruned graph chosen before looking upward

Instead:

- rank-1 legality remains structurally simple in itself
- but the *training scaffold* for rank 1 is replaced by the upward-extendable image induced from rank 2

This is a genuine tower idea:

- lower rank is not arbitrary
- lower rank is the shadow cast by valid higher-rank structure

## What Still Belongs To Rank 1

This correction does **not** mean that all rank-2-specific musical predicates should be copied into rank-1 legality directly.

For example:

- rank-2 vertical interval classes are still a rank-2 notion
- rank-2 crossing and parallel-fifth predicates are still rank-2 notions

Those should remain implemented at rank 2.

What rank 1 receives is not those predicates themselves, but the projection image of the graph that already satisfies them.

So the correction is:

- do not port every rank-2 rule downward as a rank-1 local predicate
- instead prune rank 1 by image under projection from the graph where those predicates already live

## Replacement Of The Current Headroom Rule

The current rank-1 final-chord-size headroom rule should be considered provisional and ultimately replaceable.

The intended replacement is:

1. build the admissible rank-2 graph for the chosen Slice / graph spec
2. compute the projected rank-1 node image
3. compute the projected rank-1 edge image
4. train rank 1 on that induced graph rather than on a heuristic pitch-box trim

Until that implementation exists, the headroom rule may remain as a temporary approximation, but it should no longer be treated as the target design.

## Implementation Consequences

The implementation should eventually introduce an induced-rank-1 graph artifact or cache, rather than recomputing the projection image on every training step.

The natural shape is:

- construct or enumerate the relevant pruned slice of `G(2)`
- compute the projected node and edge images
- persist that induced rank-1 graph description as an artifact keyed by the rank-2 graph spec

Then rank-1 rollout can query:

- node membership in the induced image
- outgoing actions in the induced edge image

without repeatedly rebuilding the whole rank-2 object online.

## Decision

Accepted design correction:

- rank-1 should ultimately be pruned by projection image from pruned rank 2
- this applies to both nodes and edges
- the current rank-1 headroom ceiling is a temporary heuristic, not the intended final method

## Next Build Consequence

The next implementation slice implied by this correction is:

- specify and build an induced rank-1 graph artifact from rank-2 projection image

That work should begin by deciding:

1. whether the induced graph is fully enumerated or lazily cached
2. whether the current rank-1 graph is replaced by the image or intersected with an independently defined base graph
3. how the induced graph artifact is keyed and versioned
