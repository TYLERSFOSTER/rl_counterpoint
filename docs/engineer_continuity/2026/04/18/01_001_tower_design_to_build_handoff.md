# Tower Design-To-Build Handoff

This note captures the tower-redesign work completed since the last engineer continuity document on April 15.

The most important state change is that the tower redesign has moved from conceptual design into implementation readiness.

No tower implementation code has been written yet in this segment. The work was design, alignment, contradiction review, and build planning.

## Current Project State

The old flat `rl_counterpoint/` subproject is frozen for tower work.

That means:

- do not modify files under `rl_counterpoint/` while working on tower
- do not import `rl_counterpoint/` from `tower/` at runtime
- old code may be read as legacy reference
- useful old utilities may be copied into `tower/`, after which the copy becomes tower-owned

The active new implementation area is the top-level `tower/` package.

Existing top-level tower directories:

```text
tower/
tower/action/
tower/graph/
tower/policy/
tower/reward/
tower/train/
```

The next implementation work should happen there, with tests under:

```text
tests/tower/
```

## Authorship And Attribution

The tower model is the project manager's design.

This includes:

- the nested graph tower $G(4)_\bullet \to G(3)_\bullet \to G(2)_\bullet \to G(1)_\bullet$
- the insight that voiceleading builds by rank
- the idea that lower-rank voiceleading is genuinely contained in higher-rank voiceleading
- the projection-only parent/child structure
- the rank-local action-extension view
- the HNSW-like / logarithmic-search-speedup analogy
- the observation that the speedup comes from searching a much smaller active action fiber at each tier
- the dimension-reduction interpretation in terms of reducing effective local out-degree tier by tier

Docs in `docs/design/tower/` were updated to reflect this attribution.

## Accepted Mathematical Model

The authoritative mathematical document is:

```text
docs/design/tower/mathematical_model.md
```

Core state convention:

$$
s^n=(\lambda_0,\dots,\lambda_{n-1})\in\{0,\dots,127\}^n
$$

with strictly increasing MIDI pitches.

Core action convention:

$$
\Delta s^n=(\Delta\lambda_0,\dots,\Delta\lambda_{n-1})\in\mathbb Z^n.
$$

Transition:

$$
s^n_{t+1}=s^n_t+\Delta s^n_t.
$$

Projection:

$$
\operatorname{pr}^2(\lambda_0,\lambda_1)=(\lambda_0).
$$

For $n\ge3$, projection removes the second-from-top coordinate:

$$
\operatorname{pr}^n(\lambda_0,\dots,\lambda_{n-3},\lambda_{n-2},\lambda_{n-1})
=
(\lambda_0,\dots,\lambda_{n-3},\lambda_{n-1}).
$$

The same convention applies to action projection.

Important correction from the discussion:

- parent/child structure is encoded by graph projections
- the mathematical state object does not store an explicit parent object
- any parent field in implementation would only be a cache or convenience view, not the canonical model

Graph projections are full graph morphisms, not just node projections.

So every valid higher-rank edge must project to a valid lower-rank edge.

## Action Assembly Model

Each rank policy decides only the new action coordinate.

For rank $k$, the parent action $\Delta s^{k-1}$ is already supplied by lower-rank policy/scaffold. The active rank supplies the missing coordinate and assembles a full $\Delta s^k$.

For $k=2$, the new coordinate is the top/outer voice.

For $k\ge3$, the new coordinate is the second-from-top inserted interior voice.

This gives:

- states project downward
- actions assemble upward
- each active policy searches a much smaller local action space

Lift fiber:

$$
A_k(s_t^k;\Delta s_t^{k-1})
=
\{\Delta s_t^k\in\partial_0^{-1}(s_t^k)
\mid
\operatorname{pr}^k(\Delta s_t^k)=\Delta s_t^{k-1}\}.
$$

This fiber is the concrete implementation target for the speedup. During rank-$k$ training, the active child policy should select among legal lift choices over the parent action rather than search the whole rank-$k$ action space.

## Graph Pruning Decisions

The graph-pruning decisions were clarified and recorded in:

```text
docs/design/tower/mathematical_model.md
```

Node legality uses:

- outer gap $\lambda_{n-1}-\lambda_0$
- adjacent gaps $\lambda_{i+1}-\lambda_i$

The only gaps that matter for node pruning are:

- the outer gap
- adjacent gaps

The adjacent gap rule includes the top adjacent gap. "Adjacent" means $\lambda_{i+1}-\lambda_i$, as opposed to the outer gap.

The outer width band is:

$$
L(n,N)\le \lambda_{n-1}-\lambda_0\le U(n,N).
$$

The accepted interpretation is:

$$
C(N)=\lceil M_{\mathrm{adj}}N\rceil
$$

and:

$$
L(n,N)=C(N)-E_{\mathrm{vert}},
\qquad
U(n,N)=C(N)+E_{\mathrm{vert}}.
$$

Here $E_{\mathrm{vert}}$ is a maximum vertical error.

Accepted edge pruning includes:

- source and target must be legal states
- edge is realized by an action vector
- no self-loop
- no voice crossing
- no parallel fifths
- one-sided movement cap $\mu_i-\lambda_i\le M_{\mathrm{move}}$
- $M_{\mathrm{move}}$ is rank-independent
- $M_{\mathrm{move}}$ is not the same as vertical spread
- valid higher-rank edge must project to a valid lower-rank edge

## Reward Design Decisions

The accepted reward documents are:

```text
docs/design/tower/rank_local_reward_spec.md
docs/design/tower/reward_context_contracts.md
docs/design/tower/success_failure_semantics.md
```

Core principle:

$$
R_k
$$

scores rank-$k$ facts that are newly introduced at rank $k$, rather than rescoring all lower-rank facts.

Parent rewards are diagnostic during child training. The active training objective uses the active rank reward.

For rank $k$, the reward context is built from the rank-$k$ window/passage object. Lower-rank context can be obtained by projection when needed.

"Passage so far" means the fixed reward window $W_t^k$, following the old system's window idea but interpreted at rank $k$.

Some rewards can depend on recent previous steps, not only the current transition. Example: checking for a cadence may need the last several chords.

Important success redesign:

- rank 1 does not terminate on the full perfect cadence
- rank 1 terminates on the projected pedal/root part of a perfect cadence
- rank 2 adds an additional terminal condition for the outer voice, such as supplying the third of cadence chords
- higher ranks add their own rank-local terminal requirements

In general:

$$
\operatorname{Success}_k(W_t^k)
=
\operatorname{Success}_{k-1}(\operatorname{pr}^k W_t^k)
\wedge
\operatorname{NewTerminalCondition}_k(W_t^k).
$$

## Training Protocol Decisions

The accepted training document is:

```text
docs/design/tower/training_protocol.md
```

Accepted lifecycle:

1. Train $\pi^1$.
2. Freeze $\pi^1$.
3. Train $\pi^2$ over the frozen rank-1 scaffold.
4. Freeze $\pi^2$.
5. Continue upward rank by rank.

Training order is $k=1,2,3,\dots$.

When rank $k$ is training:

- all lower ranks are frozen
- no higher ranks are active
- lower-rank policies provide scaffold actions
- active rank policy supplies only its new action coordinate
- the objective uses only $R_k$, unless a later design explicitly changes this

Parent policy action selection during child training:

- generally greedy
- with some added randomness
- randomness is more like top-k sampling than full epsilon-uniform exploration

There is no backtracking to retrain lower ranks in the accepted protocol.

## Rollout Semantics

The accepted rollout document is:

```text
docs/design/tower/rollout_semantics.md
```

During rank-$k$ rollout:

1. Build or maintain current rank-$k$ state/window.
2. Project to parent context as needed.
3. Parent frozen policies choose the parent action scaffold.
4. Active rank policy chooses only the new coordinate.
5. The assembled full action is checked against the lift fiber.
6. If legal lift choices exist, active policy samples only legal extensions.
7. If no legal extension exists, sample outside the fiber as an exceptional invalid-extension event.
8. Invalid extension advances time and is treated as no-op/penalty according to rollout semantics.

The accepted trajectory record shape is Option C:

- store the full rank-$k$ record
- include parent diagnostics
- store only $W_t^k$ as required
- compute projected parent windows on demand

Policy-gradient training uses only the active tier's log-probability.

Parent log-probabilities are diagnostics only.

## Artifact And Checkpoint Decisions

The accepted artifact document is:

```text
docs/design/tower/artifact_checkpoint_dependencies.md
```

Core accepted idea:

- each rank has its own artifacts/checkpoints
- child-rank training depends on locating parent-rank artifacts
- reproducibility must be explicit enough that the parent scaffold can be recovered

Exact artifact implementation is deferred until the artifact/checkpoint slice.

## Legacy Utility Decision

The accepted utility decision document is:

```text
docs/design/tower/shared_utility_decisions.md
```

Final decision:

- `rl_counterpoint/` is legacy reference only
- tower must not runtime-import old code
- copying is allowed
- copied modules become tower-owned

Planned copy/rework targets include:

- pitch helpers
- interval helpers
- rendering helpers
- consonance helpers
- observation/window logic

Old training, policy, and algorithm code are reference only. Tower training must be tower-owned.

## Implementation Planning Completed

The accepted implementation documents are:

```text
docs/design/tower/implementation_plan.md
docs/design/tower/implementation_slices.md
docs/design/tower/contradiction_pass.md
docs/design/tower/build_plan.md
```

Phase 5 was completed:

- Stage 13: map design to files
- Stage 14: decide shared-vs-copied utilities
- Stage 15: define implementation slices

Phase 6 was completed:

- Stage 16: contradiction pass
- Stage 17: machine-implementable build plan

The build plan was accepted by the project manager.

This means the design phase in `01_002_preparation_gameplan.md` is complete.

There is no Phase 7 in that gameplan. The next work is implementation, starting with Slice 1.

## Contradiction Pass Result

The contradiction pass found no conceptual blocker to implementation.

Important outcomes:

- Stage 14 overrides any earlier wording that allowed direct imports from old `rl_counterpoint/`
- older docs that mention explicit parent-object state structures are superseded by the projection-only mathematical model
- hard-violation termination remains deferred until rollout/training slices
- some reward grammar choices remain deferred until reward slices
- the first implementation slices do not require those deferred decisions

One known non-blocking typo was identified in `mathematical_model.md`:

```text
intermediate coordinates λ_k, for 10≤k≤n-1
```

This should be:

$$
1\le k\le n-2.
$$

The typo is conceptual noise only, not an implementation blocker.

## Accepted Build Plan

The machine-implementable build plan is:

```text
docs/design/tower/build_plan.md
```

Global implementation guardrails:

- do not modify `rl_counterpoint/`
- do not import `rl_counterpoint/` from `tower/`
- copy old utilities only when needed
- canonical state/action representation is tuple/list-like
- tensor conversion is downstream or internal-only
- parent is computed by projection, not stored as required state
- use `tower/graph/projection.py`, singular
- do not introduce neural training before Slice 7
- do not overbuild reward grammar in early slices

Canonical early representation:

```python
MidiPitch = int
TowerState = tuple[int, ...]
TowerAction = tuple[int, ...]
```

Rank is normally:

```python
rank = len(state)
```

Public graph/projection/legality APIs should accept and return tuple-like objects.

Tensor use is allowed later for optimized batched pruning or model internals, but it is not the core graph contract.

## Next Implementation Step

The next approved direction is not another design stage.

The next proposed implementation step is:

```text
Implementation Slice 1: Rank-1 Core Contracts
```

Start with:

```text
Slice 1 / Action 1.1: Create Package Markers
```

Then continue through:

```text
Slice 1 / Action 1.2: Add Tuple-Based State/Action Helpers
Slice 1 / Action 1.3: Add Tower Window
Slice 1 / Action 1.4: Add Reward Result Shell
Slice 1 / Action 1.5: Add Reward Context Shell
Slice 1 / Action 1.6: Add Success Result Shell
```

Expected first files:

```text
tower/__init__.py
tower/reward/__init__.py
tower/state_action.py
tower/window.py
tower/reward/result.py
tower/reward/context.py
tower/reward/success.py
```

Expected first tests:

```text
tests/tower/test_state_action.py
tests/tower/test_window.py
```

The build plan says to stop before projection, graph, rollout, or learning logic in Slice 1.

## Current Git/Workspace Notes

At the time this continuity document was written, recent design docs may still be untracked.

Known recently created design docs include:

```text
docs/design/tower/contradiction_pass.md
docs/design/tower/implementation_plan.md
docs/design/tower/implementation_slices.md
docs/design/tower/shared_utility_decisions.md
docs/design/tower/build_plan.md
```

Before implementation, check:

```bash
git status --short
```

Do not assume a clean worktree.

## How To Resume

The next engineer should:

1. Read `docs/design/tower/build_plan.md`.
2. Reconfirm with the project manager before writing code if they have not already approved implementation.
3. Implement only Slice 1.
4. Do not touch `rl_counterpoint/`.
5. Add focused tests under `tests/tower/`.
6. Run the Slice 1 tests with `uv run pytest`.
7. Stop and report before proceeding to Slice 2.

The project manager has accepted the design-to-build handoff, but the working style remains discussion-first. Implementation should proceed in small bounded slices.
