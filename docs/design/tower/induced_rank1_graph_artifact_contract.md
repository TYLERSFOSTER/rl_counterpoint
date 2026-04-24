# Induced Rank-1 Graph Artifact Contract

This document is the Phase 9 / Stage 2 / Action 2 deliverable.

The purpose is to specify how rank 1 should consume the projection image of the
pruned rank-2 graph as an artifact, rather than relying on the current
headroom-based pitch ceiling heuristic.

This is a design contract, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Phase 9: Graph-Pruning Corrections |
| Stage | Stage 2: Rank-1 Projection-Pruning Correction |
| Action | Action 2: Specify induced rank-1 graph artifact |

Action 2 exit criterion:

| Requirement | Status |
| --- | --- |
| define induced node image artifact | drafted here |
| define induced edge image artifact | drafted here |
| decide replacement vs intersection semantics | drafted here |
| define cache key inputs | drafted here |
| define artifact location | drafted here |
| define rank-1 runner dependency on artifact | drafted here |

## Source Authority

This contract depends on:

- [rank1_projection_pruning_correction.md](/Users/foster/rl_counterpoint/docs/design/tower/rank1_projection_pruning_correction.md)
- [artifact_checkpoint_dependencies.md](/Users/foster/rl_counterpoint/docs/design/tower/artifact_checkpoint_dependencies.md)
- [training_runner_contract.md](/Users/foster/rl_counterpoint/docs/design/tower/training_runner_contract.md)
- [system_design.md](/Users/foster/rl_counterpoint/docs/design/tower/system_design.md)

## Purpose Of The Artifact

The rank-1 training graph should eventually be pruned not by a manually chosen
upper-register ceiling, but by actual upward extendability into the pruned
rank-2 graph.

So rank 1 needs access to:

1. the projected image of valid rank-2 nodes
2. the projected image of valid rank-2 edges

This information should be materialized as an artifact, because computing the
full projection image online during every rollout step would be wasteful and
would entangle rank-1 training with repeated rank-2 graph reconstruction.

## Core Decision

The induced rank-1 graph artifact is a cached graph description derived from a
specific pruned rank-2 graph spec.

The rank-1 runner should query that artifact as its effective graph, rather than
using a heuristic pitch-range trim.

## Artifact Semantics

Let:

\[
\mathrm{pr}^{2 \to 1} : G(2)_\bullet \to G(1)_\bullet
\]

be the canonical tower projection.

Then the artifact represents:

\[
G(1)_0^{\mathrm{induced}}
=
\operatorname{Im}\left(\mathrm{pr}^{2 \to 1} : G(2)_0 \to G(1)_0\right)
\]

and

\[
G(1)_1^{\mathrm{induced}}
=
\operatorname{Im}\left(\mathrm{pr}^{2 \to 1} : G(2)_1 \to G(1)_1\right).
\]

So the artifact must include both:

- induced nodes
- induced directed edges

Node-only artifacts are explicitly insufficient.

## Replacement Versus Intersection

Decision:

- the induced image replaces the effective rank-1 graph used for training

not:

- intersect induced image with a separately defined heuristic rank-1 graph

Reason:

The whole point of the correction is to stop rank 1 from being governed by an
independent approximate pruning rule once a better upward-induced rule exists.

So the intended semantics are:

```text
effective rank-1 training graph
= induced image from pruned rank-2 graph
```

The only remaining rank-1-local legality checks should be structural sanity
checks such as:

- correct rank
- MIDI bounds
- basic action shape

Those are not alternative musical pruning rules. They are invariant structural
guards.

## Artifact Contents

Minimum artifact contents:

```json
{
  "artifact_schema_version": 1,
  "kind": "induced_rank1_graph",
  "source_rank": 2,
  "target_rank": 1,
  "source_graph_spec": { "...": "..." },
  "target_graph_spec_base": { "...": "..." },
  "node_image": [[60], [61], [62]],
  "edge_image": [
    {"source": [60], "target": [61]},
    {"source": [61], "target": [62]}
  ]
}
```

The payload should remain deterministic and JSON-serializable.

Required fields:

| Field | Meaning |
| --- | --- |
| `artifact_schema_version` | versioning |
| `kind` | artifact discriminator |
| `source_rank` | should be 2 for this first slice |
| `target_rank` | should be 1 |
| `source_graph_spec` | full rank-2 graph spec used to build image |
| `target_graph_spec_base` | structural rank-1 base spec before image pruning |
| `node_image` | sorted unique projected rank-1 states |
| `edge_image` | sorted unique projected rank-1 directed edges |

Recommended additional fields:

| Field | Meaning |
| --- | --- |
| `node_count` | summary count |
| `edge_count` | summary count |
| `construction_mode` | `enumerated` / future alternatives |
| `source_lineage_id` | optional provenance if built from one lineage-specific graph artifact |
| `notes` | optional human-readable provenance/comments |

## Representation Decision

Decision:

- use a fully enumerated artifact for the first implementation

not:

- a lazy memo cache only

Reason:

The first implementation should be maximally inspectable and debuggable.
Enumerated node and edge images make it easy to:

- inspect the induced rank-1 graph by hand
- compare two artifact builds
- write deterministic tests
- explain why a rank-1 action is missing

If the artifact later becomes too large, lazy or hybrid indexing can be added as
an optimization layer. But the semantic contract should start from explicit
enumeration.

## Artifact Location

The artifact should live under the tower lineage artifact tree, but it is not a
rank checkpoint.

Recommended path:

```text
artifacts/tower/derived_graphs/induced_rank1_from_rank2/<cache_key>.json
```

This keeps it:

- outside any one rank directory
- cacheable across multiple rank-1 training runs that use the same rank-2 graph
- clearly separate from episode/checkpoint artifacts

The artifact should not be placed inside `rank_1/`, because it is conceptually
derived from rank 2, even though it is consumed by rank 1.

## Cache Key Contract

The cache key must depend on the semantic inputs that determine the induced
image.

Required cache key inputs:

1. rank-2 graph spec
2. projection convention version
3. artifact schema version

For the first implementation, the cache key should *not* depend on:

- policy weights
- reward weights
- training episode count

Reason:

This artifact is a graph-theoretic object, not a learned-policy object.

So if two training runs use the same rank-2 graph legality spec, they should
reuse the same induced rank-1 graph artifact.

## Rank-1 Runner Dependency

The rank-1 runner should eventually accept:

- either an explicit induced rank-1 graph artifact path
- or a rank-2 graph spec from which the runner can resolve or build the artifact

Effective lifecycle:

```text
rank-1 config
-> resolve induced-rank1 artifact for the chosen rank-2 graph spec
-> construct rank-1 rollout legality from node_image + edge_image
-> train on induced graph
```

The runner should fail loudly if:

- the induced graph artifact is requested but missing and cannot be built
- the induced node image is empty
- the induced edge image is empty

## Query Model

The first implementation only needs two queries:

1. node membership

```text
is this rank-1 state in the induced node image?
```

2. edge membership

```text
is this rank-1 transition in the induced edge image?
```

That is enough for rollout legality and candidate action filtering.

## Construction Contract

The artifact builder should:

1. enumerate or otherwise obtain valid rank-2 nodes
2. enumerate or otherwise obtain valid rank-2 edges
3. project all rank-2 nodes to rank 1
4. project all rank-2 edges to rank 1
5. deduplicate and sort the resulting images
6. persist the artifact

The builder must not infer rank-1 image membership from nodes alone. Edge image
must be computed from actual valid rank-2 edges.

## Temporary Compatibility

Until the induced artifact is implemented, the current rank-1 headroom rule may
remain temporarily in code.

But once the induced artifact path exists, it should become the preferred and
intended rank-1 pruning mechanism.

## Decision Summary

Accepted decisions:

- rank-1 should consume an induced graph artifact derived from rank-2 projection
- artifact contains both node image and edge image
- artifact replaces the effective training graph rather than intersecting with a
  separate heuristic graph
- first implementation should use a fully enumerated, inspectable artifact
- artifact should be cached outside rank directories under a derived-graphs area

## Next Build Consequence

The next implementation slice implied by this contract is:

- build the induced rank-1 graph artifact generator and the rank-1 legality
  adapter that consumes it
