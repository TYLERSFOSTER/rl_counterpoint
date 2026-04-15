## April 15 Session Continuity

This note captures the work and design discussion completed so far in the current session.

### Implemented Before The Design Pivot

The flat REINFORCE training stack was extended with a goal-directed inference wrapper.

Core idea:

- the model produces logits over legal actions
- a hand-coded goal-progress bias is added to those logits
- the wrapped logits define the operative policy

Mathematically:

```text
tilde_z(alpha) = z_theta(alpha) + beta * b(alpha; s, g)
```

where `b(alpha; s, g)` is root-octave progress toward target octave `g`.

This wrapper is now used consistently in:

- training rollout collection
- REINFORCE replay loss
- final MIDI export

Files changed:

- `rl_counterpoint/algos/rollout.py`
- `rl_counterpoint/algos/reinforce.py`
- `scripts/train_reinforce.py`
- tests in:
  - `tests/algos/test_rollout.py`
  - `tests/algos/test_reinforce.py`
  - `tests/test_train_reinforce.py`

Important behavior:

- the new knob is `goal_bias_weight`
- default is `0.0`, so repo behavior is unchanged until the user enables it

Verification completed:

- focused rollout / reinforce / train tests passed
- full test suite passed

Final verified result at that point:

```text
155 passed in 35.28s
```

### Graph Statistics Script

A new graph stats script was added at:

- `scripts/graph_stats.py`

It currently computes:

- average out-star size
- a rough goal step count
- a heuristic depth `d`
- a rough branch growth estimate `b^d`

The current interpretation discussed in-session:

- the graph has high local branching
- naive exploration is combinatorially enormous
- this supports a search-space / curse-of-dimensionality reading
- therefore graph pruning is likely a more fundamental first lever than tuning RL alone

### Review Of Existing Sparsity / Pruning

We checked `docs/design/count_gn_sparsity.md` against the live runtime graph.

Conclusion:

- most of the document’s node and edge pruning ideas are already live in the repo and therefore active in training
- main mismatch: the design note describes chord-width cap `ceil(6n)`, while the live repo currently uses `ceil(5n)`
- the single-line interval edge trim is implemented one-sided (`target_i - source_i <= M`), so large downward motion is still allowed

### Major Design Pivot: Hierarchical Tower Model

The session then shifted away from the flat graph RL design toward a new hierarchical/tower design.

Main conceptual conclusions:

- the present repo architecture is not the right long-term shape for inductive training by chord size
- the new system should live beside the current one, not be mixed into the existing flat graph/runtime code
- the right mathematical picture is a tower of projections:

```text
... -> G(3) -> G(2) -> G(1)
```

Key correction made during discussion:

- the canonical map is child to parent by projection
- not parent to child by a unique section

So:

- states project downward
- actions assemble upward

Example interpretation discussed:

- `alpha^1_t`: pedal motion
- `alpha^2_t`: outer interval above pedal
- `alpha^3_t`: inner voice interval inside the outer span
- etc.

This gives:

- downward projectability of state
- upward assemblability of action

Policy hierarchy discussed:

- `pi^n(alpha^n_t | s^n_t)` at each chord size `n`
- train `pi^(n+1)` only after `pi^n` has been trained some amount
- `pi^n` provides the scaffold
- `pi^(n+1)` learns the newly added action coordinate
- each level has its own reward function for evaluating the new added structure

This was recognized as having:

- inverse/projective-limit flavor for states
- semidirect-extension flavor for actions

Important vocabulary that emerged:

- compatible tower of projections
- staged policy hierarchy
- successive semidirect extensions

### Code-Structure Recommendation From Discussion

Recommended design for the new system:

- keep the current flat system intact
- create a genuinely separate top-level subtree for the tower-based redesign

Recommended directories:

- `tower/graph`
- `tower/action`
- `tower/policy`
- `tower/reward`
- `tower/train`

Rationale:

- new ontology is too different from the current flat graph/env/algos structure
- this should be developed as a parallel system, not as a patch layered into the old one

### Directory Work Done

Top-level tower directories were created:

- `tower/`
- `tower/graph/`
- `tower/action/`
- `tower/policy/`
- `tower/reward/`
- `tower/train/`

There was one mistake during setup:

- directories were first created under `rl_counterpoint/tower/`, which was the wrong location

That was corrected in-session:

- the correct top-level `tower/` tree was created
- only empty mistaken directories were removed

Current intended location is the top-level repo subtree:

- `tower/...`

### Current Design Direction

The active design direction at the end of this segment of the session is:

- build a new tower-based system in `tower/`
- represent hierarchical rank structure explicitly
- treat lower-rank state as canonical parent projection
- treat higher-rank action as assembled extension
- design rewards rank-by-rank
- aim for inductive trainability across chord size

### What Has Not Yet Been Done

Not yet done in this session:

- no tower implementation files beyond empty directories
- no new graph formalism under `tower/`
- no new policy classes under `tower/`
- no reward or training code under `tower/`

So the repo state is:

- flat RL system still exists and remains working
- top-level `tower/` subtree exists as the placeholder for the redesign
- the conceptual direction for that redesign has been clarified substantially
