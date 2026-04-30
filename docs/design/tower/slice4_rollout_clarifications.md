# Slice 4 Rollout Clarifications

This document answers the Slice 4 ambiguity list raised after the accepted build plan.

It is a design clarification document for:

```text
Implementation Slice 4: Rank-2 Rollout Without Neural Policy
```

It does not authorize changes to the frozen `rl_counterpoint/` legacy subproject.

## Authority

Use the following documents as authority, in this order:

| Priority | Document |
| --- | --- |
| 1 | `docs/design/tower/build_plan.md` |
| 2 | `docs/design/tower/rollout_semantics.md` |
| 3 | `docs/design/tower/implementation_slices.md` |
| 4 | `docs/design/tower/success_failure_semantics.md` |
| 5 | `docs/design/tower/training_protocol.md` |

The project manager's later clarifications override earlier speculative text.

## Summary Answer

The next implementation action should be:

```text
Slice 4 / Action 4.1: Add trajectory record types.
```

Recommended file:

```text
tower/train/trajectory.py
```

Recommended tests:

```text
tests/tower/train/test_trajectory.py
```

This action should not add a rollout loop yet.

The purpose of Action 4.1 is to freeze the data shape that later rollout, reward, and policy-gradient code will consume.

## A. Slice Boundary

### Decision

Slice 4 is:

```text
Rank-2 rollout without neural policy.
```

It should prove rollout choreography, not learning.

### Answers

| Question | Answer |
| --- | --- |
| Is Slice 4 only rank-2 rollout, or reusable scaffolding for ranks 3+? | Implement rank-2 behavior first, but choose generic names and field shapes that clearly generalize to rank $k$. Do not implement rank-3 logic in Slice 4. |
| Should Slice 4 include minimal reward plumbing? | Yes, but only callback-shaped reward plumbing. Do not implement real reward formulas here. |
| Should Slice 4 include termination semantics? | Include `terminated` and `truncated` fields and support fixed-length truncation. Real success/failure hooks may be callback-shaped but should remain minimal. |
| Should Slice 4 create final public APIs? | Create provisional-but-serious APIs. They should be stable enough for Slice 5-7, but not treated as a frozen external API. |

### Rationale

Slice 4 sits between already-proven graph mechanics and later reward/training mechanics. It should establish the rollout data contract without pulling in neural policy or full TC21M reward grammar.

## B. Phase.Stage.Action Decomposition

### Decision

Proceed in this order:

1. `Slice 4 / Action 4.1`: trajectory record types.
2. `Slice 4 / Action 4.2`: scripted sampler interfaces.
3. `Slice 4 / Action 4.3`: rank-2 happy-path rollout.
4. `Slice 4 / Action 4.4`: invalid extension and empty-fiber outcomes.
5. `Slice 4 / Action 4.5`: rollout verification tests and cleanup.

### Atomicity Rule

Each approved action should be either:

- one file plus focused tests, or
- one behavior plus focused tests.

Avoid implementing the entire Slice 4 in one unreviewed jump.

## C. Trajectory Record Shape

### Decision

Use a full rank-$k$ step record with parent diagnostics.

Recommended primary dataclass:

```python
TowerTrajectoryStep
```

Recommended rollout container:

```python
TowerTrajectory
```

### Required Step Fields

Every rollout step should store:

| Field | Status | Meaning |
| --- | --- | --- |
| `rank` | required | active rollout rank |
| `step_index` | required | integer timestep before transition |
| `source_state` | required | $s_t^k$ |
| `window` | required | $W_t^k$ |
| `parent_state` | required for $k>1$ | $\operatorname{pr}^k(s_t^k)$ |
| `parent_action` | required for $k>1$ | $\Delta s_t^{k-1}$ |
| `active_choice` | required | active child coordinate choice |
| `assembled_action` | required | full $\Delta s_t^k$ assembled from parent plus active choice |
| `attempted_target_state` | required | state that would result from applying `assembled_action` |
| `realized_next_state` | required | actual next state after validity/no-op semantics |
| `active_logprob` | required | trainable log probability for active tier |
| `reward` | required | `TowerRewardResult`, not just scalar |
| `terminated` | required | true after terminal success |
| `truncated` | required | true after deadline, cap, parent failure, or other truncation |
| `outcome` | required | short outcome label |
| `diagnostics` | required | implementation/debug metadata |

### Reward Field

Store reward as:

```python
TowerRewardResult
```

not only as a scalar.

Reason: Slice 5 and later training need hard-violation, terminal, and diagnostics information. A scalar-only field would immediately need replacement.

### Attempted Vs Realized State

Store both:

- `attempted_target_state`
- `realized_next_state`

This matters for invalid extensions, where:

$$
\text{attempted target}\ne\text{realized next state}.
$$

For no-op invalid extensions:

$$
\text{realized next state}=s_t^k.
$$

### Mathematical/Sampled/Applied Action Distinction

For Slice 4, do not over-split action fields into many competing names.

Use:

- `active_choice`: sampled active child coordinate
- `assembled_action`: mathematical rank-$k$ action assembled from parent plus active choice
- `attempted_target_state`: result attempted by that assembled action
- `realized_next_state`: state actually recorded after validity semantics

This is enough to distinguish sampled action, assembled action, and realized transition.

## D. Option C Rollout Semantics

### Decision

Store full rank-$k$ records. Compute projected parent windows on demand.

### Answers

| Question | Answer |
| --- | --- |
| Store rank-2 records only, or cache projected parent windows/actions? | Store rank-2 records. Parent action/state diagnostics are stored; parent windows are computed on demand. |
| Is projected parent data diagnostic or training contract? | Parent action and active projection checks are part of the rollout contract. Parent windows are derived data, not required stored data. |
| Should the rollout record be generic or rank-2-specific? | Generic enough to be called `TowerTrajectoryStep`, but only tested/used for rank 2 in Slice 4. |

## E. Parent Sampler

### Decision

In Slice 4, parent actions should come from a sampler callback.

Recommended file:

```text
tower/policy/samplers.py
```

Recommended return type:

```python
SamplerResult
```

### Parent Sampler Return

The parent sampler should return:

| Field | Meaning |
| --- | --- |
| `choice` | parent action tuple |
| `logprob` | optional diagnostic log probability |
| `diagnostics` | sampler metadata |

For deterministic scripted parent actions, use:

```python
logprob = None
```

Reason: parent log probability is not used for active training. Using `0.0` can be misread as a real probability statement. `None` cleanly means "not trainable / not applicable."

### Invalid Parent Action

If the parent sampler proposes an invalid rank-1 action, rollout should:

1. record a `parent_failure` diagnostic,
2. emit a truncating step or truncate before child sampling, depending on implementation convenience,
3. not treat it as an invalid child extension.

Parent failure and invalid extension are different events.

Do not silently no-op a parent failure.

### Location

Put scripted sampler code in:

```text
tower/policy/samplers.py
```

Even without neural policy, this keeps rollout code from owning sampling policy.

## F. Active Child Sampler

### Decision

The active sampler should normally choose from:

```python
active_lift_choices(...)
```

However, Slice 4 must also allow a scripted invalid child proposal for testing invalid-extension semantics.

### Active Sampler Return

Use the same `SamplerResult` shape:

| Field | Meaning |
| --- | --- |
| `choice` | active child coordinate delta |
| `logprob` | active log probability or `None` for deterministic scripted tests |
| `diagnostics` | sampler metadata |

For deterministic scripted active choices in Slice 4:

```python
logprob = None
```

Later neural/stochastic training will supply a real active log probability.

### Fiber Visibility

The active sampler should receive the active coordinate choices, not the whole full-action fiber, by default.

Reason: the active policy's mathematical responsibility is choosing the new coordinate. Rollout/graph code owns assembly and validity checking.

For diagnostics or test utilities, passing the full fiber is acceptable, but it should not be required by the policy-facing sampler interface.

## G. Lift Fiber And Invalid Extension

### Decision

If the child proposes a coordinate outside the active lift choices, this is an invalid extension.

On invalid extension, rollout must:

1. keep the rank-$k$ state unchanged,
2. advance time,
3. record the attempted action and attempted target,
4. record `realized_next_state == source_state`,
5. apply an invalid-extension reward result,
6. set a diagnostic label such as `invalid_extension`,
7. not mark terminal success.

### Default Penalty

Slice 4 should avoid fixing musical penalty scale.

Recommended default:

```python
invalid_extension_reward = TowerRewardResult(
    reward=0.0,
    hard_violation=False,
    is_terminal_success=False,
    diagnostics={"outcome": "invalid_extension"},
)
```

If a penalty parameter is needed for tests, expose it as a rollout argument with default `0.0`.

Reason: Slice 4 tests mechanics, not reward calibration.

### Representation

Invalid extension should be represented as a normal trajectory step with `outcome="invalid_extension"`.

Do not use a separate error object.

### Termination

Invalid extension does not terminate by default.

It can contribute to truncation only if the fixed step cap/deadline is reached.

## H. Empty Fiber

### Decision

Empty lift fiber is exceptional but should be represented inside rollout, not as an immediate crash.

Slice 4 should emit a normal no-op trajectory step with:

```python
outcome = "empty_lift_fiber"
```

and diagnostics indicating that no legal child lift existed.

### Distinction From Invalid Child Choice

Test empty fiber independently from invalid child choice.

The distinction is:

| Case | Meaning |
| --- | --- |
| `invalid_extension` | legal lift choices existed, but child proposed outside them |
| `empty_lift_fiber` | no legal lift choices existed over the parent action |

### Parent Failure?

Empty fiber is not automatically a parent-policy failure.

It means the parent action was valid in the lower graph but had no legal lift at the active rank. This may indicate calibration trouble, but it is not the same event as an invalid parent action.

## I. State And Window Ownership

### Decision

Rollout owns building the rank-$k$ window from rank-$k$ history.

For Slice 4:

- build $W_t^2$ from rank-2 history at every step
- project parent windows on demand
- do not store projected parent windows in the trajectory step

### Records And Windows

The trajectory step should store:

```python
window
```

where this is the full rank-$k$ `TowerWindow`.

Repeated states from invalid no-op steps must appear in history.

Reason:

The invalid extension advances time. Therefore the next window must reflect that time advanced even though state did not change.

## J. Termination

### Decision

Slice 4 should support fixed-length truncation and data fields for future termination, but should not implement full musical success semantics.

### Required Fields

Use both:

```python
terminated: bool
truncated: bool
```

Meaning:

| Field | Meaning |
| --- | --- |
| `terminated` | terminal success or terminal failure hook has ended the episode |
| `truncated` | rollout ended because of step cap, deadline, parent failure, or external cutoff |

### Timing

Termination flags are per-step after transition.

The rollout container may also expose aggregate properties later, but the step record should store the post-transition flags.

### Hard Violations

Hard violations should not be fully implemented in Slice 4 unless needed by the invalid-extension mechanics.

Use `TowerRewardResult.hard_violation` as the future-compatible field.

Do not decide the full hard-violation termination policy until rollout supports real reward/success hooks.

## K. Reward Plumbing

### Decision

Slice 4 should accept a reward callback returning `TowerRewardResult`.

It should not compute real musical reward terms.

### Callback Shape

Use a callback conceptually like:

```python
reward_fn(context) -> TowerRewardResult
```

The exact context type should use the existing Slice 1 reward-context shell if available.

### Invalid Extension Penalty

Invalid-extension reward is rollout mechanics in Slice 4.

Reason:

Invalid extension is not a normal musical transition to score. It is an outcome of failing the lift-fiber legality contract.

Later reward code can wrap or override the default invalid-extension reward.

### Reward Context

Construct minimal reward context only if it already exists from Slice 1.

Do not expand reward grammar inside Slice 4.

## L. API And File Layout

### Decision

Use the file layout from the accepted build plan:

```text
tower/train/trajectory.py
tower/policy/samplers.py
tower/train/rollout.py
```

Also add package markers if not already present:

```text
tower/train/__init__.py
tower/policy/__init__.py
```

### Neutral Rollout Directory?

Do not introduce a new `tower/rollout/` directory for Slice 4.

Reason:

The accepted implementation plan places rollout under `tower/train/`, and this is close enough to later training trajectory ownership.

### Generic Or Rank-Specific Function

Use an explicit rank-2 rollout function first:

```python
rollout_rank2_scripted(...)
```

or:

```python
rollout_rank2(...)
```

Do not prematurely implement a fully generic rank-$k$ rollout engine.

However, name record types generically:

```python
TowerTrajectoryStep
TowerTrajectory
```

## M. Test Strategy

### Decision

Use focused tests after each Slice 4 action.

Run:

```bash
uv run pytest tests/tower
```

after each completed action if the tests are fast.

Avoid old full-suite runs unless specifically requested or unless tower changes appear to affect shared project configuration.

### Required Slice 4 Tests

At minimum, Slice 4 should test:

| Test | Required? |
| --- | --- |
| trajectory record construction | yes |
| trajectory stores `TowerRewardResult` | yes |
| parent-first sampling order | yes |
| child fiber assembly | yes |
| rank-2 happy path | yes |
| invalid child no-op | yes |
| empty fiber diagnostic | yes |
| parent failure diagnostic/truncation | yes |
| active logprob separate from parent logprob | yes |
| projected parent data recoverable on demand | yes |
| no runtime import from `rl_counterpoint` | yes |

## N. Naming

### Decision

Use the following names.

| Concept | Name |
| --- | --- |
| active child value | `active_choice` |
| active child coordinate delta in prose | active child coordinate delta |
| parent action | `parent_action` |
| assembled full rank action | `assembled_action` |
| attempted target | `attempted_target_state` |
| actual next state | `realized_next_state` |
| invalid child proposal | `invalid_extension` |
| no legal lift over parent | `empty_lift_fiber` |
| lower-rank scaffold failure | `parent_failure` |

Avoid `invalid_lift` as the main label because it blurs two separate cases:

- a bad proposal outside a nonempty lift fiber
- an actually empty lift fiber

Avoid calling the assembled action just `action` in trajectory records because it obscures the parent/active assembly structure.

## Final Recommendation

Approve the next action as:

```text
Slice 4 / Action 4.1: Add trajectory record types.
```

Implementation target:

```text
tower/train/trajectory.py
tests/tower/train/test_trajectory.py
```

Action 4.1 should add only:

- generic trajectory dataclasses,
- outcome labels,
- reward-result storage,
- parent diagnostic fields,
- attempted-vs-realized state fields,
- focused construction/default tests.

It should not add:

- rollout loop,
- sampler implementations,
- neural policy,
- reward formulas,
- rank-3 support,
- imports from `rl_counterpoint/`.

Once Action 4.1 passes, proceed to:

```text
Slice 4 / Action 4.2: Add scripted sampler interfaces.
```
