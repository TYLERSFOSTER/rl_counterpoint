# Timed Chord Sequence Gameplan

Date: 2026-04-08  
Role: actual staged gameplan for the timed-sequence / transformer direction  
Status: supersedes the prior checkpoint-note role of this file

## Purpose

This document is the actual gameplan for the architectural turn reached in discussion:

- the serious policy should be sequence-based
- the policy information state should be a timed chord sequence
- bar-relative position and time signature must become explicit parts of the model

This file is not just a note. It is the working staged plan for the next band of project architecture.

## Core Model Direction

The RL skeleton remains:

```text
history -> policy -> action -> transition -> reward
```

But the true policy-facing information state is now taken to be a sequence of timed chord events rather than a single bare chord.

At quarter-note time `t`:

- chord state:
  ```text
  s_t
  ```
- metrical coordinate:
  ```text
  m_t
  ```
- timed event:
  ```text
  e_t = (s_t, m_t)
  ```
- policy information state:
  ```text
  x_t = (e_{t-L+1}, ..., e_t)
  ```

where `L` is a fixed context window measured in quarter-note events.

The working interpretation from discussion is:

```text
x_t = chord sequence over the last three bars
```

so the actual event count depends on time signature.

## Why The Plan Must Change

This is not just a model-internals swap.

Making the policy sequence-based changes multiple contracts:

- representation of timed musical events
- environment observation semantics
- rollout record shape
- policy input contract
- later trainer assumptions
- eventually reward/evaluator context needs

Because of that, the project now needs new explicit phases/stages rather than leaving this as commentary attached to an earlier plan.

## Work Hierarchy

Work is organized as:

```text
Project
└── Phase
    └── Stage
        └── Action
```

An action contains:

1. High-level explanation.
2. Ground-truth files or artifacts.
3. Machine-executable operation.
4. Associated unit tests.
5. Failure hypotheses.

## Phase T1 — Timed Musical Event Representation

Goal: make metrical structure an explicit engineering object rather than hidden burden on policy code.

### Stage T1.1: Time-Signature Contract

Purpose: decide how the package represents meter at the system level.

Owner decisions needed:

- Which time signatures are in scope initially?
- Is the first implementation fixed to one meter or configurable from the start?
- Are chord events always quarter-note indexed?

Action sketch:

- Explanation: define the first time-signature contract before sequence observation code is built.
- Ground truth files/directories: this gameplan, `rl_counterpoint/envs/counterpoint_env.py`, `rl_counterpoint/reward/protocol.py`.
- Machine operation: discussion only until the Project Owner approves a concrete representation.
- Tests: none at this checkpoint.
- Failure hypotheses:
  - Meter stays implicit and later leaks into multiple modules ad hoc.
  - The project commits to a fixed meter too early when configurability is needed.
  - The project allows too much meter variability before any one case is executable.

### Stage T1.2: Timed Event Schema

Purpose: define the object representing a quarter-note chord event with metrical coordinates.

Owner decisions needed:

- What exact fields belong in `m_t`?
- Is metrical data just beat-in-bar, or more than that?
- Does the timed event belong in `music/`, `envs/`, or another layer?

Action sketch:

- Explanation: define the representation of `e_t = (s_t, m_t)` explicitly.
- Ground truth files/directories: `rl_counterpoint/graph/state_space.py`, `rl_counterpoint/envs/observation.py`, `rl_counterpoint/music/`, this gameplan.
- Machine operation: implement timed event representation after approval.
- Tests: field validation and basic construction tests.
- Failure hypotheses:
  - Timed event semantics get duplicated between env, rollout, and model code.
  - Metrical fields are underspecified and later force breaking changes.

## Phase T2 — Sequence Observation Contract

Goal: define what the policy actually sees.

### Stage T2.1: Context Window Decision

Purpose: decide how much history the policy consumes.

Owner decisions needed:

- Is the first context exactly the last three bars?
- Is the context window fixed globally?
- Does the first implementation use truncation, padding, or both?

Action sketch:

- Explanation: decide the event-window contract before replacing the placeholder single-state policy.
- Ground truth files/directories: `rl_counterpoint/envs/counterpoint_env.py`, `rl_counterpoint/algos/rollout.py`, `rl_counterpoint/models/policy.py`, this gameplan.
- Machine operation: discussion only until the Project Owner approves the sequence contract.
- Tests: none at this checkpoint.
- Failure hypotheses:
  - More code is built around single-chord observations and then churns.
  - Full history is chosen when a fixed window is the real target.
  - Padding/truncation rules are left implicit and drift across modules.

### Stage T2.2: Observation Ownership Boundary

Purpose: decide whether sequence observations are produced by the environment or assembled later by rollout/training code.

Owner decisions needed:

- Should `CounterpointEnv.reset/step` return sequence observations directly?
- Should env return only current state plus history in `info`, with rollout assembling windows?
- Where should padding happen?

Action sketch:

- Explanation: fix ownership of sequence observation construction before more rollout/model code lands.
- Ground truth files/directories: `rl_counterpoint/envs/counterpoint_env.py`, `rl_counterpoint/envs/observation.py`, `rl_counterpoint/algos/rollout.py`.
- Machine operation: implementation after approval.
- Tests: observation-shape and ownership tests.
- Failure hypotheses:
  - Environment and rollout both start constructing sequence observations differently.
  - Sequence ownership gets decided accidentally through helper code instead of contract.

## Phase T3 — Transformer Policy Contract

Goal: replace the placeholder single-chord policy with the actual sequence-model direction.

### Stage T3.1: Sequence Tokenization And Encoding

Purpose: define how timed chord events become model inputs.

Owner decisions needed:

- Are chord pitches fed as raw integers, embeddings, or something else?
- How is metrical information encoded?
- Are positional encodings purely temporal, purely metrical, or both?

Action sketch:

- Explanation: define the transformer input representation before building the model.
- Ground truth files/directories: `rl_counterpoint/models/policy.py`, this gameplan, future timed-event representation.
- Machine operation: implement encoding helpers and tests after approval.
- Tests: tensor-shape, batching, and encoding sanity tests.
- Failure hypotheses:
  - The model shape gets built before the token semantics are defined.
  - Meter encoding is left implicit and leaks into handcrafted feature hacks.

### Stage T3.2: Transformer Policy Module

Purpose: implement the real sequence policy that emits logits over the fixed `StepDelta` lattice.

Owner decisions needed:

- What transformer depth/width is appropriate for the first executable version?
- Should the first policy emit logits from the final token only?
- Does the mask remain outside the policy? Current expectation: yes.

Action sketch:

- Explanation: replace the placeholder policy contract with a sequence policy over timed chord windows.
- Ground truth files/directories: `rl_counterpoint/models/policy.py`, `rl_counterpoint/graph/actions.py`, `rl_counterpoint/algos/rollout.py`.
- Machine operation: implement transformer-like policy after approval.
- Tests: forward shape checks, finite logits, batch handling, action-dimension agreement.
- Failure hypotheses:
  - The transformer is introduced before sequence observation ownership is settled.
  - Mask application responsibility blurs between model and rollout.

## Phase T4 — Rollout Refactor For Sequence Policy

Goal: make rollout collection operate over sequence observations rather than single-chord placeholders.

### Stage T4.1: Policy-Driven Rollout Update

Purpose: replace masked random action choice with policy-driven masked action selection under the new sequence contract.

Prerequisite:

- Phases T1 through T3 must be settled enough that sequence observations are real.

Action sketch:

- Explanation: revise rollout collection to consume sequence observations and a real policy output.
- Ground truth files/directories: `rl_counterpoint/algos/rollout.py`, `rl_counterpoint/models/policy.py`, `rl_counterpoint/envs/counterpoint_env.py`.
- Machine operation: implementation after approval.
- Tests: rollout records align with sequence observations, action indices, masks, and next observations.
- Failure hypotheses:
  - Rollout record schema still assumes single-state observations.
  - Sequence window building and policy input drift apart.

### Stage T4.2: Rollout Smoke Revision

Purpose: update smoke artifacts so they expose the real sequence-policy path.

Action sketch:

- Explanation: revise smoke scripts once sequence rollout exists.
- Ground truth files/directories: `scripts/smoke_rollout.py`, `scripts/smoke_env.py`, `rl_counterpoint/algos/rollout.py`.
- Machine operation: implementation after approval.
- Tests: script-level smoke tests and direct-execution checks.
- Failure hypotheses:
  - Smoke artifacts continue demonstrating only scaffold behavior after the architecture has moved on.

## Phase T5 — Real Training Loop

Goal: implement actual learning only after the sequence observation and transformer policy contracts are stable.

### Stage T5.1: Trainer Design Decision

Purpose: decide the first actual learner update path under the sequence-policy architecture.

Owner decisions needed:

- Is the first learner still REINFORCE?
- Does the first trainer remain intentionally tiny and explicit?
- When should a value model become real rather than placeholder?

Action sketch:

- Explanation: discussion checkpoint before replacing the explicit wait-state in `scripts/train_reinforce.py`.
- Ground truth files/directories: `rl_counterpoint/algos/rollout.py`, `rl_counterpoint/algos/reinforce.py`, `rl_counterpoint/models/policy.py`, `rl_counterpoint/models/value.py`, `scripts/train_reinforce.py`.
- Machine operation: discussion only until the Project Owner approves the trainer direction.
- Tests: none at this checkpoint.
- Failure hypotheses:
  - Training code starts before sequence rollout and policy contracts stabilize.
  - Trainer code ends up compensating for unsettled observation/model interfaces.

### Stage T5.2: First Real Trainer Implementation

Purpose: replace the training wait-state with an actual learner update loop.

Action sketch:

- Explanation: implement real learner code only after the preceding sequence-policy stages are settled.
- Ground truth files/directories: `rl_counterpoint/algos/reinforce.py`, `scripts/train_reinforce.py`, `rl_counterpoint/models/policy.py`, `rl_counterpoint/models/value.py`.
- Machine operation: implementation after approval.
- Tests: smoke-train tests, no-NaN checks, executable training script.
- Failure hypotheses:
  - Reward placeholder gives too weak a signal for meaningful learning.
  - Sequence batching or masking semantics break the loss code.

## Phase T6 — TC21M Reward Replacement

Goal: replace the placeholder reward internals once the sequence/timed-event architecture is stable enough to support beat-sensitive rules.

### Stage T6.1: Beat-Sensitive Rule Boundary

Purpose: ensure reward formalization is compatible with explicit metrical information.

Action sketch:

- Explanation: revisit the reward formalization boundary now that meter is explicit in the model.
- Ground truth files/directories: `assets/rules/tc21m_rules.md`, `rl_counterpoint/reward/protocol.py`, future timed-event schema.
- Machine operation: discussion or implementation after approval.
- Tests: golden examples and protocol compatibility.
- Failure hypotheses:
  - Meter-sensitive rules get bolted on after the architecture instead of being carried through the contract cleanly.

## Immediate Interpretation

This file now changes the actual planning picture.

The project is no longer merely choosing between a placeholder MLP and some later nicer model.

The active architectural direction is:

```text
timed chord event representation
-> fixed sequence observation contract
-> transformer-like StepDelta policy
-> rollout refactor
-> real training loop
```

That means the immediate next planning work should happen inside:

```text
Phase T1 and Phase T2
```

because those phases define the representation and observation contracts that every later learning component depends on.
