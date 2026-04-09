# Session Gameplan From Ground Truth

Date: 2026-04-09  
Role: first engineer continuity artifact for today's session  
Authority: grounded planning artifact built from prior continuity docs plus current on-disk repo state

## Purpose

This document is the gameplan for the current session.

It is not a speculative roadmap.

It is a ground-truth touchdown artifact built from:

- the prior continuity/gameplan documents in `docs/engineer_continuity`
- the current repository as inspected on disk
- the current implementation/test surface that actually exists now

The intent is to prevent planning drift.

In particular:

- the 2026-04-08 continuity documents remain authoritative for yesterday's decisions
- this document restates the next session gameplan against present repo reality
- empty placeholder files are treated as empty until populated
- implemented files are treated as implemented only where code and tests actually exist

## Ground-Truth Repo Snapshot

### Implemented and populated now

The following project areas are materially implemented and non-empty:

- Graph layer:
  - `rl_counterpoint/graph/graph_spec.py`
  - `rl_counterpoint/graph/state_space.py`
  - `rl_counterpoint/graph/non_crossing.py`
  - `rl_counterpoint/graph/actions.py`
- Environment layer:
  - `rl_counterpoint/envs/counterpoint_env.py`
  - `rl_counterpoint/envs/observation.py`
  - `rl_counterpoint/envs/termination.py`
- Reward scaffold:
  - `rl_counterpoint/reward/protocol.py`
  - `rl_counterpoint/reward/black_box.py`
- Policy / sequence model path:
  - `rl_counterpoint/models/policy.py`
- Rollout / training path:
  - `rl_counterpoint/algos/rollout.py`
  - `rl_counterpoint/algos/reinforce.py`
  - `scripts/smoke_env.py`
  - `scripts/smoke_rollout.py`
  - `scripts/train_reinforce.py`
- Test surface:
  - graph, env, reward, policy, rollout, reinforce, and smoke tests all exist as non-empty test files under `tests/`

### Empty or placeholder-only now

The following files are still empty on disk and therefore remain placeholder territory:

- `rl_counterpoint/models/value.py`
- `scripts/smoke_graph.py`
- `scripts/smoke_reward.py`

The following package files are empty initializer markers only:

- `rl_counterpoint/__init__.py`
- `rl_counterpoint/graph/__init__.py`
- `rl_counterpoint/envs/__init__.py`
- `rl_counterpoint/models/__init__.py`
- `rl_counterpoint/algos/__init__.py`
- `rl_counterpoint/reward/__init__.py`
- `rl_counterpoint/music/__init__.py`

### Important current asymmetry

The `music/` layer planned in earlier gameplans is still not developed beyond an empty package marker.

Musical/state/voiceleading responsibilities currently live primarily in:

- `rl_counterpoint/graph/*`
- `rl_counterpoint/envs/observation.py`
- `rl_counterpoint/models/policy.py`

This means future work should not pretend there is already a separate mature music-theory module boundary.

### Rules asset reality

The rule source file exists:

- `assets/rules/tc21m_rules.md`

But it is not yet a clean executable specification.

Ground-truth characteristics of the current file:

- it contains useful rule families
- it contains unresolved prose notes such as `***[FIX THIS...]***`
- it contains ambiguity markers and open questions
- it ends in unfinished placeholder text

Therefore the next reward-stage work is not "simply code the rules."

It is at least:

1. classify the current rule text
2. identify what is executable now
3. identify what is beat-sensitive / cadence-sensitive / history-sensitive
4. decide the minimal reward-boundary change required before implementation

## Continuity Alignment

The most recent continuity handoff from 2026-04-08 says the next live work boundary is:

- Phase T6
- Stage T6.1
- Action: Beat-Sensitive Rule Boundary

That remains consistent with present repo reality.

Reason:

- the timed-sequence observation and transformer-policy path are already implemented
- REINFORCE plumbing already exists
- reward remains black-box placeholder only
- meter/time context already exists in environment and reward context, but it is not yet bound to TC21M rule families

So the session should center on reward-boundary formalization, not on rebuilding graph/env/policy scaffolding that already exists.

## Session Gameplan

### Phase 0

#### Stage 0.1

##### Action 0.1.1

Purpose:
Lock the current engineering reality before any reward-boundary change is proposed.

Ground-truth files:

- `docs/engineer_continuity/2026/04/08 /01_004_engineer_continuity_session_report.md`
- `rl_counterpoint/reward/protocol.py`
- `rl_counterpoint/reward/black_box.py`
- `rl_counterpoint/envs/counterpoint_env.py`
- `rl_counterpoint/envs/observation.py`
- `rl_counterpoint/models/policy.py`
- `rl_counterpoint/algos/reinforce.py`

Machine operation:
Inspect these files and bind what is already real versus what is still placeholder.

Associated tests:

- inspect the existing reward/env/policy/reinforce tests before altering boundaries

Failure hypotheses:

- prior continuity overstates implementation status
- current repo has drifted from the 2026-04-08 report
- reward context already lacks something needed for beat-sensitive rules

#### Stage 0.2

##### Action 0.2.1

Purpose:
Bind the placeholder frontier explicitly so the session does not drift into already-complete work or nonexistent modules.

Ground-truth files:

- `rl_counterpoint/models/value.py`
- `scripts/smoke_graph.py`
- `scripts/smoke_reward.py`
- `rl_counterpoint/music/__init__.py`

Machine operation:
Treat these as absent capabilities unless the Project Owner explicitly directs otherwise.

Associated tests:

- none; these are placeholder-boundary confirmations

Failure hypotheses:

- session planning assumes a value model exists
- session planning assumes reward smoke tooling already exists
- session planning assumes a `music/` abstraction layer already owns rule logic

### Phase 1

#### Stage 1.1

##### Action 1.1.1

Purpose:
Re-read the current TC21M rules asset as a specification source rather than as prose inspiration.

Ground-truth files:

- `assets/rules/tc21m_rules.md`

Machine operation:
Classify the rule text into executable families without implementing scoring yet.

Associated tests:

- none at classification time

Failure hypotheses:

- rules that look local are actually phrase- or cadence-dependent
- prose ambiguity is too high to bind directly into code
- current markdown mixes melodic, vertical, harmonic, and cadence rules without enough separation

#### Stage 1.2

##### Action 1.2.1

Purpose:
Map TC21M rule families onto the current explicit context already carried by the environment and reward protocol.

Ground-truth files:

- `assets/rules/tc21m_rules.md`
- `rl_counterpoint/reward/protocol.py`
- `rl_counterpoint/envs/counterpoint_env.py`
- `rl_counterpoint/envs/observation.py`

Machine operation:
For each candidate rule family, determine whether it is:

- transition-local
- beat-sensitive
- cadence-position-sensitive
- multi-step/history-sensitive
- not yet formalizable from current context

Associated tests:

- none yet; this is boundary mapping

Failure hypotheses:

- `RewardContext` carries less metrical information than the environment actually knows
- rule families require explicit bar-role context rather than only `step_index` and `measure_size`
- cadence-sensitive rules require horizon/ending metadata not yet exposed cleanly

### Phase 2

#### Stage 2.1

##### Action 2.1.1

Purpose:
Decide whether the current reward protocol is already sufficient for the first beat-sensitive replacement step.

Ground-truth files:

- `rl_counterpoint/reward/protocol.py`
- `rl_counterpoint/reward/black_box.py`
- `tests/reward/test_protocol.py`
- `tests/reward/test_black_box.py`

Machine operation:
Produce a minimal sufficiency decision:

- `RewardContext` is already sufficient, or
- `RewardContext` needs a minimal extension before TC21M replacement begins

Associated tests:

- reward protocol tests must be updated only if the context contract changes

Failure hypotheses:

- adding too much reward context will hard-freeze unapproved design decisions
- adding too little reward context will force later breaking changes
- env `info` may expose useful signals that reward code cannot yet receive structurally

#### Stage 2.2

##### Action 2.2.1

Purpose:
If protocol extension is needed, define the smallest owner-visible contract change before implementation.

Ground-truth files:

- `rl_counterpoint/reward/protocol.py`
- `rl_counterpoint/envs/counterpoint_env.py`
- `docs/engineer_continuity/2026/04/08 /01_003_model_checkpoint_timed_chord_sequence.md`

Machine operation:
State the exact proposed reward-context additions in concrete file/symbol terms before editing code.

Associated tests:

- future protocol and env tests if approval is given

Failure hypotheses:

- protocol change leaks environment internals instead of exposing stable musical context
- protocol fields are named too generically and become ambiguous later
- cadence logic is encoded prematurely before the Project Owner approves it

### Phase 3

#### Stage 3.1

##### Action 3.1.1

Purpose:
After the reward boundary is settled, identify the smallest first executable TC21M replacement slice.

Ground-truth files:

- `assets/rules/tc21m_rules.md`
- `rl_counterpoint/reward/black_box.py`
- `tests/reward/test_black_box.py`

Machine operation:
Choose one narrowly scoped first replacement target such as:

- beat-role diagnostics only
- one local rule family with diagnostics
- one beat-sensitive rule family with placeholder weights

Associated tests:

- add or revise narrowly targeted reward tests only for the approved first slice

Failure hypotheses:

- too broad a first slice turns rule formalization into uncontrolled implementation
- diagnostics and scoring collapse into one opaque number too early
- reward replacement starts before the input contract is truly settled

## Current Session Interpretation

The repo does not currently need a new graph/action/env/policy scaffold.

The repo does currently need a reward-boundary touchdown:

- what the TC21M rules asset actually contains now
- which of those rules depend on beat/time/cadence context
- whether the current `RewardContext` is sufficient
- what the smallest approved first replacement slice should be

That is the present gameplan according to on-disk reality.
