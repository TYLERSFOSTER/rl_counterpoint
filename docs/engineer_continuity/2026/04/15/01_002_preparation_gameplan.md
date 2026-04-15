## Preparation Gameplan

This note captures the initial gameplan for walking through the remaining tower-design decisions before implementation.

The purpose is not to implement yet.

The purpose is to decide the remaining mathematical and systems questions in the right order so that a later build plan can become fully machine-implementable.

### Phase 1: Freeze The Mathematical Core

#### Stage 1: Define canonical state objects

- Write `docs/design/tower/state_action_spec.md`
- Specify `State1`, `State2`, `State3`, and the general `State(n)` pattern
- Specify exact field names
- Specify exact invariants
- Specify `realize(state_n) -> tuple[int, ...]`

#### Stage 2: Define canonical action objects

- In the same doc, specify `Action1`, `Action2`, `Action3`, and general `Action(n)`
- Specify exact meaning of each coordinate
- Specify legal ranges for each coordinate
- Specify whether coordinates are absolute, relative, or interval-valued

#### Stage 3: Define projections and assembly

- Specify node projection `pr^(n+1->n)`
- Specify edge/action projection induced by graph morphism requirement
- Specify action assembly rule from `alpha^1, ..., alpha^n`
- State the commuting law explicitly

#### Exit criterion for Phase 1

- we can write the exact type signatures for state, action, projection, realization, and assembly without hand-waving

### Phase 2: Freeze Graph Legality

#### Stage 4: Define node legality by rank

- Write `docs/design/tower/graph_legality_spec.md`
- Specify what makes a rank-1 node legal
- Specify what makes a rank-2 extension legal over rank-1
- Specify what makes a rank-3 extension legal over rank-2
- Separate shared constraints from rank-local constraints

#### Stage 5: Define edge legality by rank

- In the same doc, specify valid transition rules for each rank
- Make explicit the rule:
  - valid higher-rank edge projects to valid lower-rank edge
- Specify whether legality is checked on hierarchical objects, realized tuples, or both

#### Stage 6: Define graph-spec ownership

- Decide whether there is one global `TowerSpec` plus per-rank specs, or only per-rank specs
- List every tunable parameter and who owns it

#### Exit criterion for Phase 2

- we can describe `is_valid_state_n(...)` and `is_valid_transition_n(...)` precisely for the first few ranks

### Phase 3: Freeze Reward Ownership

#### Stage 7: Define rank-local reward semantics

- Write `docs/design/tower/reward_training_spec.md`
- Specify what `R^1` is responsible for
- Specify what `R^2` is responsible for
- Specify what `R^3` is responsible for
- Explicitly separate inherited scaffold from newly scored extension

#### Stage 8: Define reward context contracts

- Specify exact context objects passed to each rank reward
- Specify diagnostics fields
- Specify whether rewards are purely local, measure-local, or horizon-aware

#### Stage 9: Define success/failure semantics

- For each rank, specify:
  - terminal success
  - truncation
  - hard violation
  - invalid extension behavior

#### Exit criterion for Phase 3

- every rank has a clearly bounded reward responsibility

### Phase 4: Freeze Training Protocol

#### Stage 10: Define stagewise training lifecycle

- In `reward_training_spec.md` or a new `training_protocol.md`, specify:
  - train `pi^1`
  - freeze or partially freeze `pi^1`
  - train `pi^2`
  - etc.
- Decide exactly what “freeze” means

#### Stage 11: Define rollout semantics

- Specify how parent scaffold is produced during stage `n+1`
- Specify whether parent policy samples, acts greedily, or uses a wrapper
- Specify what data are recorded in trajectories

#### Stage 12: Define artifact and checkpoint dependencies

- Specify where rank-specific checkpoints live
- Specify how stage `n+1` locates and loads stage `n`
- Specify what must be reproducible across stages

#### Exit criterion for Phase 4

- we can describe one full training run for `pi^2` and `pi^3` step by step

### Phase 5: Freeze System Architecture

#### Stage 13: Map design to files

- Write `docs/design/tower/implementation_plan.md`
- For each planned file in `tower/`, specify:
  - responsibility
  - imports allowed
  - inputs/outputs
  - what it must not own

#### Stage 14: Decide shared-vs-copied utilities

- Decide which old modules remain shared:
  - music helpers
  - observation helpers
  - rendering
- Decide which are tower-local copies

#### Stage 15: Define first implementation slice

- Choose the smallest vertical slice:
  - probably rank-1 only
- Then choose second slice:
  - rank-2 extension over rank-1

#### Exit criterion for Phase 5

- each planned file has a clear responsibility and the first implementation slice is bounded

### Phase 6: Implementation Readiness Review

#### Stage 16: Run a contradiction pass

- Check state/action spec against reward spec
- Check reward spec against training protocol
- Check graph-morphism requirement against action assembly
- Check all docs for unresolved ambiguity

#### Stage 17: Produce machine-implementable build plan

- Break implementation into stages
- Each stage should be phrased as:
  - create file X
  - add dataclass Y
  - add function Z
  - add tests A/B/C

#### Exit criterion for Phase 6

- we have a build plan that can be executed without inventing new theory mid-implementation

### Recommended Immediate Next Documents

The next two documents should be written in this order:

1. `docs/design/tower/state_action_spec.md`
2. `docs/design/tower/reward_training_spec.md`

These are the bottlenecks.

Once they exist, the remaining work becomes much more mechanical.
