# Tower Design Notes

This folder collects design work for the tower-based redesign of `rl_counterpoint`.

The tower system is intended to sit beside the current flat graph RL system, not replace it immediately.

## Attribution

The core ideas collected in this folder come from the human project manager / project owner.

That includes:

- the nested-graph view across chord size
- the insistence that the cross-rank maps are full graph morphisms
- the observation that voiceleading builds upward by tier, so that one-voice motion is genuinely part of two-voice motion, two-voice motion is part of three-voice motion, and so on
- the state-downward / action-upward asymmetry
- the semidirect / HNSW-like search intuition
- the expectation of a major training/search speedup from the hierarchical structure
- the dimensionality insight that, when local search burden is reflected by out-degree, reducing the problem tier by tier is effectively a radical reduction in the searched dimension
- the idea that higher-rank sections/extensions become easy to construct once lower-rank scaffolds exist

These documents are technical writeups of those project-manager ideas for implementation planning.

Current documents:

- [migration_map.md](/Users/foster/rl_counterpoint/docs/design/tower/migration_map.md)
- [system_design.md](/Users/foster/rl_counterpoint/docs/design/tower/system_design.md)
- [rank1_projection_pruning_correction.md](/Users/foster/rl_counterpoint/docs/design/tower/rank1_projection_pruning_correction.md)
- [induced_rank1_graph_artifact_contract.md](/Users/foster/rl_counterpoint/docs/design/tower/induced_rank1_graph_artifact_contract.md)
