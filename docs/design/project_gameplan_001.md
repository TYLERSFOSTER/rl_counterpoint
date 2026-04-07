# RL Counterpoint Project Gameplan

## Purpose

This document is a high-level project gameplan for building `rl_counterpoint`: a reinforcement learning system for voice leading and counterpoint over ordered pitch-state graphs.

The plan is intentionally decision-gated. The Project Owner defines the musical and mathematical interpretation of the system. The engineering role is to make those decisions executable, testable, and inspectable without prematurely freezing design choices that still belong to the Project Owner.

## Current High-Level Direction

The system should model counterpoint as sequential movement through a structured space of voiced chord states. A state is an ordered tuple of discrete pitch values, with persistent voice identity and strict vertical ordering. Legal movement begins with a non-crossing transition idea: voices may move, but the resulting voice leading should preserve ordering rather than tangle or swap identities.

The project also needs a reward function derived eventually from *Tonal Counterpoint for the 21st-Century Musician*. However, deriving that reward function is substantial music-theory and specification work. It should not block early RL infrastructure. Therefore, the initial implementation should define a stable reward protocol and use a black-box placeholder behind that protocol. The internals can later be replaced by a book-derived evaluator once the rule formalization is ready.

The main early engineering goal is not to train a good agent immediately. It is to create a small, reality-checkable system where state representation, action mechanics, reward interface, environment semantics, and a minimal learner can all run end-to-end.

## Design Decisions Reserved For The Project Owner

The following decisions should not be silently made by the engineering agent:

- Exact pitch universe: whether to use `0..127`, `0..128`, or another bounded pitch lattice.
- Voice count defaults and whether voice count is fixed globally or configurable per environment.
- Exact definition of non-crossing transitions.
- Whether actions represent direct next-chord choice or factorized per-voice movement.
- Whether all non-crossing transitions are legal initially, or whether there are additional baseline constraints before the reward function.
- Episode objective and terminal cadence representation.
- Observation encoding.
- Reward protocol fields beyond the minimal early contract.
- Whether early reward should be random, heuristic, externally injected, table-driven, or manually scripted.
- Whether the first trainer should be REINFORCE, PPO, or an even smaller smoke-test learner.

Engineering work should expose these as explicit choices and proceed only after the relevant choice is made.

## Work Hierarchy

Work is organized as:

```text
Project
└── Phase
    └── Stage
        └── Action
```

A phase is a broad project band. A stage is the smallest meaningful unit that implements a coherent new capability. An action is the smallest machine-executable work unit.

Each action should contain:

1. High-level explanation.
2. Files to inspect or establish as ground truth.
3. Machine-executable operation.
4. Associated unit tests.
5. Failure hypotheses.

## Phase 0: Project Grounding And Contracts

Goal: establish the repo as a controlled engineering workspace before implementing behavior.

### Stage 0.1: Confirm Repository Shape

Purpose: verify that the intended package, script, test, docs, and asset directories exist.

Action sketch:

- Explanation: confirm the filesystem matches the intended scaffold without creating code files.
- Ground truth files/directories: repo root, `rl_counterpoint/`, `scripts/`, `tests/`, `assets/rules/`, `docs/design/`.
- Machine operation: list the directory tree at a bounded depth.
- Tests: none; this is a structural check.
- Failure hypotheses:
  - A directory was not created.
  - An existing file conflicts with a planned directory.
  - The working directory is not the intended repo root.

### Stage 0.2: Define Naming And Import Contract

Purpose: decide whether the package name remains `rl_counterpoint` and define initial import expectations.

Action sketch:

- Explanation: avoid future drift between repo name, package name, and module imports.
- Ground truth files/directories: `pyproject.toml`, `rl_counterpoint/`, `main.py`.
- Machine operation: inspect project metadata and package layout.
- Tests: later import smoke test, once files exist.
- Failure hypotheses:
  - `pyproject.toml` points to a different package pattern.
  - The empty package directory lacks `__init__.py`.
  - Existing tooling assumes a different module name.

## Phase 1: Pitch, Chord, And State Representation

Goal: define the core data objects that the graph, environment, and reward protocol will all share.

### Stage 1.1: Pitch Lattice Contract

Purpose: decide and encode the pitch universe.

Owner decision needed:

- What is the exact pitch set: `0..127`, `0..128`, or another range?
- Are pitch values MIDI-like integers, abstract semitone ranks, or something else?

Action sketch:

- Explanation: create a minimal pitch representation and validation boundary.
- Ground truth files/directories: `rl_counterpoint/music/pitch.py`, `tests/test_state_space.py`, `pyproject.toml`.
- Machine operation: add pitch range constants and validation helpers.
- Tests: valid lower/upper bounds pass; out-of-range values fail.
- Failure hypotheses:
  - Upper bound is off by one relative to owner intent.
  - Pitch semantics are confused with literal frequency.
  - Validation is too strict for future transposition or register experiments.

### Stage 1.2: Chord State Representation

Purpose: represent an ordered `n`-voice chord state with strict vertical ordering and persistent voice identity.

Owner decision needed:

- Should a state be a tuple of ints, a dataclass, a named tuple, or another structure?
- Should voice names exist now, or later?

Action sketch:

- Explanation: create the canonical chord/state object used throughout the system.
- Ground truth files/directories: `rl_counterpoint/music/chord.py`, `rl_counterpoint/graph/state_space.py`, `tests/test_state_space.py`.
- Machine operation: implement state construction and ordering validation.
- Tests: valid strictly increasing states pass; duplicate/crossed/out-of-range states fail.
- Failure hypotheses:
  - Voice identity gets confused with sorted pitch sets.
  - The implementation silently sorts invalid input instead of rejecting it.
  - The state object becomes too heavyweight too early.

### Stage 1.3: Basic Interval And Voice Motion Utilities

Purpose: define small reusable calculations needed by graph mechanics and later reward logic.

Action sketch:

- Explanation: add interval and per-voice motion helpers without implementing book-derived scoring.
- Ground truth files/directories: `rl_counterpoint/music/intervals.py`, `rl_counterpoint/music/voiceleading.py`, tests.
- Machine operation: implement basic interval magnitude/direction helpers.
- Tests: ascending, descending, repeated, and multi-voice movement cases.
- Failure hypotheses:
  - Direction conventions are ambiguous.
  - Signed and absolute intervals are mixed.
  - Helpers accidentally encode stylistic reward assumptions.

## Phase 2: Non-Crossing Graph And Action Mechanics

Goal: define how the agent can move from one ordered chord state to another.

### Stage 2.1: Non-Crossing Transition Definition

Purpose: turn the owner-approved non-crossing rule into a deterministic predicate.

Owner decision needed:

- What is the exact mathematical condition for a non-crossing edge?
- Are equalities allowed anywhere, or is strict ordering always required?

Action sketch:

- Explanation: implement `is_non_crossing(prev_state, next_state)` only after the formula is clarified.
- Ground truth files/directories: `rl_counterpoint/graph/non_crossing.py`, `rl_counterpoint/music/voiceleading.py`, `tests/test_non_crossing.py`.
- Machine operation: add predicate and representative examples.
- Tests: clear crossing, non-crossing, boundary, and equal-pitch cases.
- Failure hypotheses:
  - The original symbolic formula is malformed or ambiguous.
  - Adjacent-voice constraints are implemented with wrong indices.
  - Legal voice overlap versus illegal voice crossing is not yet distinguished.

### Stage 2.2: Action Representation Decision

Purpose: choose the action contract before environment code depends on it.

Owner decision needed:

- Should an action be a direct next-state choice?
- Should an action be factorized per voice, then projected or rejected?
- Should action masks enumerate legal next states, legal moves, or both?

Action sketch:

- Explanation: document and implement the first action representation.
- Ground truth files/directories: `rl_counterpoint/graph/actions.py`, `rl_counterpoint/envs/counterpoint_env.py`, tests.
- Machine operation: add action type and conversion to candidate next state.
- Tests: action decoding, invalid action rejection, legal candidate generation.
- Failure hypotheses:
  - Direct next-state actions create an enormous action space too early.
  - Factorized actions create illegal combinations requiring projection.
  - Projection hides invalid behavior that should be visible in diagnostics.

### Stage 2.3: Small Candidate Generator

Purpose: provide bounded candidate transitions for smoke testing.

Owner decision needed:

- What movement radius should be allowed in the first tiny system?

Action sketch:

- Explanation: generate local candidate next states from a current state for testing and early rollout.
- Ground truth files/directories: `rl_counterpoint/graph/state_space.py`, `rl_counterpoint/graph/actions.py`, `tests/test_non_crossing.py`.
- Machine operation: implement a bounded candidate generator.
- Tests: generated candidates are valid states and satisfy non-crossing predicate.
- Failure hypotheses:
  - Candidate generation explodes combinatorially.
  - The generator accidentally excludes musically important moves.
  - The generator includes illegal states due to weak validation.

## Phase 3: Reward Protocol And Black-Box Placeholder

Goal: define the reward interface without blocking on full TC21M rule formalization.

### Stage 3.1: Reward Result Contract

Purpose: make the environment and learner depend on a stable reward signature, not a specific formula.

Owner decision needed:

- What fields must the reward result expose initially?
- Should legality live in the reward result, the graph predicate, or both?

Initial proposed contract:

```text
RewardFn(prev_state, action, next_state, context) -> RewardResult
```

Initial proposed `RewardResult` fields:

```text
reward: float
is_terminal_success: bool
hard_violation: bool
diagnostics: dict
```

Action sketch:

- Explanation: define reward protocol and result object.
- Ground truth files/directories: `rl_counterpoint/reward/protocol.py`, `rl_counterpoint/reward/diagnostics.py`, `tests/test_reward_protocol.py`.
- Machine operation: implement protocol types and result validation.
- Tests: reward result construction and stable field availability.
- Failure hypotheses:
  - Contract is too narrow for cadence and trajectory rules.
  - Contract mixes legality with reward too early.
  - Diagnostics become unstructured junk instead of useful debugging data.

### Stage 3.2: Black-Box Reward Placeholder

Purpose: allow environment and learner development while book-derived scoring remains unresolved.

Owner decision needed:

- What placeholder behavior should be used first?

Possible placeholder types:

- constant reward per legal step,
- distance-to-target heuristic,
- externally injected callable,
- manually scripted toy reward,
- random reward for plumbing only.

Action sketch:

- Explanation: implement a swappable placeholder reward behind the protocol.
- Ground truth files/directories: `rl_counterpoint/reward/black_box.py`, `tests/test_reward_protocol.py`.
- Machine operation: add placeholder implementation selected by explicit config or constructor.
- Tests: deterministic output for fixed inputs; diagnostics payload present.
- Failure hypotheses:
  - Placeholder reward creates misleading learning behavior.
  - Placeholder becomes sticky and is mistaken for the real reward.
  - Environment code starts depending on placeholder internals.

### Stage 3.3: Future TC21M Evaluator Boundary

Purpose: reserve the later replacement path without implementing it now.

Action sketch:

- Explanation: document that the book-derived evaluator must satisfy the same reward protocol.
- Ground truth files/directories: `assets/rules/tc21m_rules.md`, `rl_counterpoint/reward/protocol.py`, `docs/design/`.
- Machine operation: add design notes or TODO markers only if desired.
- Tests: none initially beyond reward protocol compatibility.
- Failure hypotheses:
  - Later evaluator requires context not included in the early protocol.
  - Rule-derived hard constraints conflict with graph-defined legality.
  - Terminal cadence scoring needs fields not anticipated by the placeholder.

## Phase 4: Gymnasium-Style Environment

Goal: wrap the graph and reward protocol as a sequential decision problem.

### Stage 4.1: Environment State And Reset

Purpose: define episode initialization.

Owner decision needed:

- What is the fixed initial chord?
- Is the initial chord globally fixed, sampled, or configurable?

Action sketch:

- Explanation: implement reset semantics and initial observation.
- Ground truth files/directories: `rl_counterpoint/envs/counterpoint_env.py`, `rl_counterpoint/envs/observation.py`, `tests/test_env_api.py`.
- Machine operation: add minimal environment class with `reset`.
- Tests: reset returns observation and info in expected shape.
- Failure hypotheses:
  - Reset is nondeterministic without seed control.
  - Observation does not preserve voice identity.
  - Initial state violates pitch/order constraints.

### Stage 4.2: Step Semantics

Purpose: connect action decoding, transition legality, reward protocol, and episode state update.

Action sketch:

- Explanation: implement one-step transition behavior.
- Ground truth files/directories: `rl_counterpoint/envs/counterpoint_env.py`, `rl_counterpoint/graph/actions.py`, `rl_counterpoint/reward/protocol.py`, tests.
- Machine operation: add `step(action)` with diagnostics in `info`.
- Tests: legal step updates state; illegal step behavior is explicit; info includes reward diagnostics.
- Failure hypotheses:
  - Illegal action handling is ambiguous.
  - Termination and truncation are conflated.
  - Reward is computed against the wrong previous/next state.

### Stage 4.3: Observation Encoding And Action Mask

Purpose: define what the policy sees and how illegal actions are hidden or rejected.

Owner decision needed:

- What should the first observation encoding be?
- Should invalid actions be masked, rejected, penalized, or impossible by construction?

Action sketch:

- Explanation: implement the minimal policy-facing observation and legality mask.
- Ground truth files/directories: `rl_counterpoint/envs/observation.py`, `rl_counterpoint/graph/actions.py`, `tests/test_env_api.py`.
- Machine operation: add observation builder and action mask builder.
- Tests: observation shape stable; action mask agrees with transition predicate.
- Failure hypotheses:
  - Observation leaks future target information unintentionally.
  - Mask calculation is too slow for even small experiments.
  - Mask and step legality disagree.

### Stage 4.4: Termination Contract

Purpose: define when episodes end.

Owner decision needed:

- What counts as terminal success?
- Is there a max episode length independent of cadence success?

Action sketch:

- Explanation: implement terminal and truncation checks.
- Ground truth files/directories: `rl_counterpoint/envs/termination.py`, `rl_counterpoint/envs/counterpoint_env.py`, tests.
- Machine operation: add termination helper called by `step`.
- Tests: terminal success, max-length truncation, and continuing step cases.
- Failure hypotheses:
  - Cadence success depends on reward internals not yet implemented.
  - Episode length interacts poorly with target chord.
  - Terminated and truncated flags are returned incorrectly.

## Phase 5: Minimal Training Loop

Goal: verify that the environment, reward protocol, and a tiny learner can run end-to-end.

### Stage 5.1: Rollout Collection

Purpose: collect short trajectories from the environment with a simple policy.

Action sketch:

- Explanation: implement explicit rollout logic before adding algorithmic complexity.
- Ground truth files/directories: `rl_counterpoint/algos/rollout.py`, `scripts/smoke_env.py`, tests.
- Machine operation: add rollout collector for fixed number of steps or episodes.
- Tests: rollout length, observations, actions, rewards, done flags align.
- Failure hypotheses:
  - Reset after termination is mishandled.
  - Reward diagnostics are dropped.
  - Rollout storage confuses time and batch dimensions.

### Stage 5.2: Tiny Policy And Value Modules

Purpose: add minimal PyTorch modules sufficient for smoke training.

Owner decision needed:

- Should the first model consume raw integer states, one-hot encodings, or normalized numeric features?

Action sketch:

- Explanation: implement the smallest model compatible with the observation/action contract.
- Ground truth files/directories: `rl_counterpoint/models/policy.py`, `rl_counterpoint/models/value.py`, `pyproject.toml`, tests.
- Machine operation: add simple modules.
- Tests: forward pass shape checks and finite outputs.
- Failure hypotheses:
  - Action space shape is not yet stable enough for model code.
  - Observation encoding changes force model rewrite.
  - Invalid-action masking is not integrated.

### Stage 5.3: First Explicit Trainer

Purpose: prove that the stack is executable, not musically good.

Owner decision needed:

- Should the first trainer be REINFORCE, PPO, or a smaller smoke-only loop?

Action sketch:

- Explanation: implement a tiny readable training script.
- Ground truth files/directories: `scripts/train_reinforce.py`, `rl_counterpoint/algos/reinforce.py`, `tests/test_smoke_train.py`.
- Machine operation: run a short fixed-seed training loop.
- Tests: no NaNs; at least one update occurs; optional checkpoint write if desired.
- Failure hypotheses:
  - Reward placeholder provides no useful learning signal.
  - Action masking creates zero valid actions.
  - Tensor shapes fail during return/loss computation.

## Phase 6: TC21M Rule Formalization And Reward Replacement

Goal: replace the placeholder reward internals with a deterministic evaluator derived from the book rules, without changing the environment/learner contract.

This phase should happen when the Project Owner is ready to invest in rule formalization. It should not block phases 1-5.

### Stage 6.1: Rule Markdown As Normative Spec

Purpose: turn `tc21m_rules.md` into a structured rule source.

Action sketch:

- Explanation: add rule IDs, scope, hard/soft status, required context, machine feature, ambiguities, and pass/fail examples.
- Ground truth files/directories: `assets/rules/tc21m_rules.md`, `assets/rules/images/`.
- Machine operation: edit Markdown rule tables or sections.
- Tests: none initially; later schema validation.
- Failure hypotheses:
  - Book prose depends on human judgment.
  - A rule is phrase-level but gets mislabeled as edge-local.
  - Multiple rules encode the same musical fact.

### Stage 6.2: Machine-Readable Rule Spec

Purpose: produce JSON/schema artifacts from the Markdown spec.

Action sketch:

- Explanation: encode formalized rules without parsing prose directly.
- Ground truth files/directories: `assets/rules/`, tests.
- Machine operation: add JSON spec and schema.
- Tests: schema validation and golden examples.
- Failure hypotheses:
  - JSON weights are chosen before raw features are trustworthy.
  - Hard constraints and soft preferences are mixed.
  - Terminal rules need more context than the spec allows.

### Stage 6.3: Deterministic Evaluator

Purpose: implement book-derived reward internals behind the existing protocol.

Action sketch:

- Explanation: replace black-box internals with structured scoring while preserving protocol compatibility.
- Ground truth files/directories: `rl_counterpoint/reward/`, `rl_counterpoint/music/`, `assets/rules/`, tests.
- Machine operation: implement feature extraction, scoring, and diagnostics.
- Tests: golden trajectory scoring and per-rule diagnostic checks.
- Failure hypotheses:
  - Reward double-counting dominates behavior.
  - Feature extraction and scoring become inseparable.
  - Diagnostics fail to identify implicated voices.

## Phase 7: Evaluation, Iteration, And Experiment Discipline

Goal: make the project inspectable enough to improve without guessing.

### Stage 7.1: Smoke Scripts

Purpose: provide fast executable checks for graph, reward, environment, and training.

Action sketch:

- Explanation: create small scripts that demonstrate each subsystem in isolation.
- Ground truth files/directories: `scripts/smoke_graph.py`, `scripts/smoke_reward.py`, `scripts/smoke_env.py`.
- Machine operation: add smoke scripts.
- Tests: scripts execute successfully under `uv run`.
- Failure hypotheses:
  - Smoke scripts drift away from tests.
  - Scripts depend on ambient working directory.
  - Diagnostics are too terse to debug failures.

### Stage 7.2: Experiment Configuration

Purpose: define reproducible knobs once the early stack exists.

Action sketch:

- Explanation: introduce config only when repeated runs need it.
- Ground truth files/directories: future `configs/`, training scripts, README.
- Machine operation: add minimal config files or CLI arguments.
- Tests: fixed config produces repeatable smoke behavior.
- Failure hypotheses:
  - Config abstraction arrives too early.
  - Defaults hide important owner decisions.
  - Path handling depends on ambient current working directory.

### Stage 7.3: Continuity Reporting

Purpose: preserve project state between sessions and engineers.

Action sketch:

- Explanation: write continuity reports under the dated engineer continuity path.
- Ground truth files/directories: `docs/engineer_continuity/2026/04/07/`.
- Machine operation: add reports named `02_001_...`, `02_002_...`, etc. when directed.
- Tests: none.
- Failure hypotheses:
  - Reports summarize intent but omit actual file changes.
  - Reports claim unverified system state.
  - Report naming collides with another engineer/session.

## Near-Term Practical Sequence

The likely first implementation sequence is:

1. Confirm or decide the pitch lattice.
2. Implement pitch and chord/state representation.
3. Clarify and implement non-crossing transition mechanics.
4. Decide initial action representation.
5. Define reward protocol and black-box placeholder.
6. Build environment reset/step around those contracts.
7. Add smoke tests and a minimal training loop.

This sequence can change at the Project Owner's direction.

## Standing Rule

The reward function's internal formula is not a blocker for early infrastructure. The reward protocol is the blocker. Once the protocol is stable, early environment and learner work can proceed with a black-box placeholder, and the TC21M-derived evaluator can replace it later.
