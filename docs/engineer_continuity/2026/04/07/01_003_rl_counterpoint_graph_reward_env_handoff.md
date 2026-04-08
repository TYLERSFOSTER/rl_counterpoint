# Engineer Continuity Report: RL Counterpoint Graph/Reward/Env Handoff

Date: 2026-04-08  
Session role: LLM consultant / assistant engineer under Project Owner direction  
Repository: `/Users/foster/rl_counterpoint`

## Scope Of This Session

This session established the first real project architecture for `rl_counterpoint`, with the Project Owner explicitly directing the work. The major result is that the graph layer is now implemented and tested, the reward protocol layer has an initial placeholder implementation and tests, and the environment layer has been discussed but not implemented.

The Project Owner emphasized that this is not an agent independently building the whole system. Work is organized under a human-directed Phase / Stage / Action hierarchy.

## Prime Directive And Collaboration Protocol

The session began by reading:

- `/Users/foster/rl_counterpoint/docs/prime_directive/prime_directive.md`
- `/Users/foster/rl_counterpoint/docs/prime_directive/common_failure_mode_001.md`

Key operational constraints taken into account:

- User directs scope and sequencing.
- Reality is verified against files, commands, and tests.
- Unexpected errors are treated as reality breaks.
- Avoid assuming that prior context still reflects the filesystem.
- Maintain explicit uncertainty around design decisions.

## Continuity Context Read

Two previous continuity docs were read:

- `/Users/foster/rl_counterpoint/docs/engineer_continuity/2026/04/07/01_001_rl_counterpoint_modern_rl_baseline.md`
- `/Users/foster/rl_counterpoint/docs/engineer_continuity/2026/04/07/01_002_counterpoint_rl_notes_from_attachment.md`

Important inherited framing:

- The project is a reinforcement learning system for voice leading / counterpoint.
- The learner should not initially depend on a fully formalized TC21M reward.
- The graph/state mechanics should become stable before larger RL training work.
- The reward internals are intentionally black-boxed at first.

## Repo Structure Work

The following directory scaffold was created earlier in the session:

- `rl_counterpoint/music`
- `rl_counterpoint/graph`
- `rl_counterpoint/reward`
- `rl_counterpoint/envs`
- `rl_counterpoint/models`
- `rl_counterpoint/algos`
- `scripts`
- `tests`

Later, test subdirectories were added to mirror the package:

- `tests/music`
- `tests/graph`
- `tests/models`
- `tests/algos`
- `tests/reward`
- `tests/envs`

Package initializer files were also added as needed so pytest/imports work:

- `rl_counterpoint/__init__.py`
- `rl_counterpoint/graph/__init__.py`
- `rl_counterpoint/music/__init__.py`
- `rl_counterpoint/models/__init__.py`
- `rl_counterpoint/algos/__init__.py`
- `rl_counterpoint/reward/__init__.py`
- `rl_counterpoint/envs/__init__.py`

The following empty script/module placeholders exist for future work:

- `rl_counterpoint/envs/counterpoint_env.py`
- `rl_counterpoint/envs/observation.py`
- `rl_counterpoint/envs/termination.py`
- `rl_counterpoint/models/policy.py`
- `rl_counterpoint/models/value.py`
- `rl_counterpoint/algos/rollout.py`
- `rl_counterpoint/algos/reinforce.py`
- `scripts/smoke_graph.py`
- `scripts/smoke_reward.py`
- `scripts/smoke_env.py`
- `scripts/train_reinforce.py`

## Design Docs Created

### Project Gameplan

The project gameplan was first created and then renamed by the Project Owner to:

- `/Users/foster/rl_counterpoint/docs/design/project_gameplan_001.md`

It lays out provisional phases:

- project grounding and contracts
- pitch/chord/state representation
- non-crossing graph and action mechanics
- reward protocol and black-box placeholder
- Gymnasium-style environment
- minimal training loop
- later TC21M reward formalization

Important project decision captured there: reward internals should not block environment/learner scaffolding. The reward protocol is the early dependency, not the final TC21M formula.

### Graph Spec Design

The graph design spec was created at:

- `/Users/foster/rl_counterpoint/docs/design/graph_spec_001.md`

It defines the current mathematical contract for \(G(n)\):

- \(G(n)_0\) is the node set.
- \(G(n)_1\) is the edge set.
- \(\delta_0,\delta_1:G(n)_1\to G(n)_0\) are source and target maps.
- Nodes are ordered MIDI pitch tuples \((\lambda_0,\dots,\lambda_{n-1})\).
- Self-loops are excluded.

Current node trims:

- adjacent vertical interval cap \(M=11\)
- forbidden adjacent vertical intervals `{1, 2, 6, 10, 11}`
- total chord width cap \(\lambda_{n-1}-\lambda_0\le \lceil 6n\rceil\)
- tonic-root trim \(\lambda_0-\tau\in\{3,4,5,7,8,9\}\pmod{12}\)
- outer interval consonance \(\lambda_{n-1}-\lambda_0\in\{3,4,5,7,8,9\}\pmod{12}\)

Current edge trims:

- no self-loops
- optional voice-crossing trim, default disallow
- optional parallel-fifth trim, default disallow
- single-line upward interval cap \(\mu_i-\lambda_i\le M\)

Important note: the single-line edge trim currently caps upward motion only, not absolute displacement.

### Sparsity Count Script

A design script exists at:

- `/Users/foster/rl_counterpoint/docs/design/count_gn_sparsity.py`

It evaluates formula / finite-sum counts over gap vectors, not full graph materialization. It requires a tonic:

```bash
uv run python docs/design/count_gn_sparsity.py 3 --tonic 60
```

The script was rewritten after a reality break: an earlier brute-force / DP approach was too slow and did not match the Project Owner's intent to use formulas. The current script works by gap vectors and relative target-root offsets.

There was also a file disappearance incident: the source file vanished while the compiled cache remained. The source was later restored from the conversation state and reverified.

## Graph Layer Implemented

### `graph_spec.py`

File:

- `/Users/foster/rl_counterpoint/rl_counterpoint/graph/graph_spec.py`

Defines:

- `CounterpointGraphSpec`
- default forbidden adjacent vertical intervals
- default allowed root intervals mod 12
- default allowed outer intervals mod 12

Important fields:

- `n`
- `tonic`
- `pitch_min`
- `pitch_max`
- `max_interval`
- `max_chord_width_factor`
- `forbidden_adjacent_vertical_intervals`
- `allowed_root_intervals_mod_12`
- `allowed_outer_intervals_mod_12`
- `allow_voice_crossing`
- `allow_parallel_fifths`

Important derived properties:

- `pitch_count`
- `tonic_pitch_class`
- `max_chord_width`
- `allowed_root_pitch_classes`
- `allowed_outer_interval_classes`
- `max_adjacent_vertical_interval`
- `max_single_line_interval`

The class is a frozen dataclass. The generated `__init__` is provided by `@dataclass(frozen=True)`, and validation runs in `__post_init__`.

### `state_space.py`

File:

- `/Users/foster/rl_counterpoint/rl_counterpoint/graph/state_space.py`

Defines:

- `ChordState = tuple[int, ...]`
- `pitch_class`
- `adjacent_intervals`
- `outer_interval`
- `is_strictly_increasing`
- `is_in_pitch_range`
- `has_valid_length`
- `has_valid_adjacent_intervals`
- `has_valid_outer_interval`
- `has_valid_root`
- `is_valid_node`
- `iter_gap_vectors`
- `state_from_root_and_gaps`
- `iter_node_states`

Design decision: the graph is currently implicit/predicate-defined. There is no NetworkX/DGL graph and no per-node class instantiation.

### `non_crossing.py`

File:

- `/Users/foster/rl_counterpoint/rl_counterpoint/graph/non_crossing.py`

Defines:

- `has_voice_crossing`
- `is_non_crossing`
- `has_parallel_fifth`
- `respects_single_line_interval`
- `is_valid_edge`

Current edge predicate applies:

- endpoint node validity
- self-loop rejection
- single-line upward interval cap
- optional voice-crossing rejection
- optional parallel-fifth rejection

Parallel fifth detection ranges over all voice pairs, not just adjacent pairs.

### `actions.py`

File:

- `/Users/foster/rl_counterpoint/rl_counterpoint/graph/actions.py`

Defines a minimal direct-next-state action representation:

- `DirectNextStateAction = ChordState`
- `action_to_next_state`
- `is_valid_action`
- `candidate_next_states`
- `action_mask`

Important later design discussion: the direct next-state representation is good for debugging, but the model should likely output over movement vectors \(\Delta\), not absolute adjacent nodes, because \(|\mathrm{Adj}(\lambda)|\) is variable and action distributions over adjacent nodes are awkward for fixed-shape neural policies.

## Reward Layer Implemented

### `protocol.py`

File:

- `/Users/foster/rl_counterpoint/rl_counterpoint/reward/protocol.py`

Defines:

- `RewardContext`
- `RewardResult`
- `RewardFn` protocol

The reward protocol is intended as the stable interface between future environment code and reward implementations. It does not encode the TC21M formula.

### `black_box.py`

File:

- `/Users/foster/rl_counterpoint/rl_counterpoint/reward/black_box.py`

Defines:

- `ConstantReward`

This is explicitly a placeholder / plumbing reward, not a music-theory evaluator. It returns a fixed scalar and diagnostic payload.

## Tests Added

Pytest was added to the project by the Project Owner via `uv add pytest`. `pyproject.toml` now includes:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

This was needed because `uv run pytest -q tests/graph` initially failed to import `rl_counterpoint` even though the package was importable under `.venv/bin/python -m pytest`. Adding `pythonpath = ["."]` made `uv run pytest` work directly.

### Graph tests

Files:

- `/Users/foster/rl_counterpoint/tests/graph/test_graph_spec.py`
- `/Users/foster/rl_counterpoint/tests/graph/test_state_space.py`
- `/Users/foster/rl_counterpoint/tests/graph/test_non_crossing.py`
- `/Users/foster/rl_counterpoint/tests/graph/test_actions.py`

The test files include module-level preambles and function-level docstrings.

Current graph test coverage:

- graph spec validation and derived properties
- node membership in \(G(n)_0\)
- forbidden adjacent intervals
- root and outer interval rules
- gap vector generation
- known `n=2, tonic=60` node count of 363
- voice crossing predicates
- parallel fifth predicates over all voice pairs
- self-loop rejection
- single-line upward interval cap
- direct-state action helper behavior

### Reward tests

Files:

- `/Users/foster/rl_counterpoint/tests/reward/test_protocol.py`
- `/Users/foster/rl_counterpoint/tests/reward/test_black_box.py`

Current reward test coverage:

- `RewardContext` defaults and history
- `RewardResult` defaults and diagnostics
- `ConstantReward` returns `RewardResult`
- configured scalar reward
- transition diagnostics
- custom diagnostic merging

## Current Test Status

Final verified command:

```bash
uv run pytest -q tests/graph tests/reward
```

Final observed result:

```text
45 passed in 0.04s
```

## Environment Layer Discussion

The environment layer has been discussed but not implemented.

Current intended files:

- `/Users/foster/rl_counterpoint/rl_counterpoint/envs/counterpoint_env.py`
- `/Users/foster/rl_counterpoint/rl_counterpoint/envs/observation.py`
- `/Users/foster/rl_counterpoint/rl_counterpoint/envs/termination.py`

Current conceptual split:

- `CounterpointEnv` should hold the running episode state.
- `observation.py` should translate internal state/history into agent observation.
- `termination.py` should hold episode stopping/truncation logic.

The environment should consume graph predicates and reward functions. It should not redefine graph legality.

## Important Design Discussion: Action Representation

Clarification reached:

- Conceptually, the action is a distribution over adjacent nodes.
- But for learning, output over \(\mathrm{Adj}(\lambda)\) is awkward because \(|\mathrm{Adj}(\lambda)|\) changes with \(\lambda\).
- A better learning representation may be a distribution over movement vectors:

\[
\Delta=(\Delta_0,\dots,\Delta_{n-1})
\]

with:

\[
\mu=\lambda+\Delta.
\]

The model input should include current state and likely history/context. The model output should likely be over movement distances rather than absolute next states.

Likely future flow:

- model input: current state plus history
- model output: distribution over movement vectors
- environment/action layer: maps movement vector to candidate target
- graph layer: validates target and transition
- reward layer: scores source, target, and context

The `argmax` / sampling step belongs in policy/rollout/evaluation code, not in `envs/`.

## Current Known Caveats

- `docs/engineer_continuity/2026/04/08 ` exists with a trailing space in the directory name. This report was placed directly in `docs/engineer_continuity/2026/04` because the Project Owner explicitly requested that path and `01_003_` filename prefix.
- The graph is currently implicit/predicate-defined, not materialized as a stored graph object.
- `candidate_next_states` currently filters all generated node states and is acceptable for early testing but may not be efficient enough for later large runs.
- The reward is a placeholder. TC21M rule-derived reward logic remains future work.
- The environment layer is not yet implemented.
- The action representation is not yet final. Direct next-state action exists for debugging, but movement-vector action is likely the learning direction.

## Suggested Next Step

The natural next work item is to continue designing `envs/`, especially:

- what the environment stores as runtime state
- direct next-state actions vs movement-vector actions for the first environment
- how much history appears in observation
- how history is passed into `RewardContext`
- invalid action behavior
- termination/truncation rules before cadence reward exists

Do not proceed with implementation without Project Owner direction.
