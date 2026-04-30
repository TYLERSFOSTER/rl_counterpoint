# Success And Failure Semantics

This document is the Phase 3 / Stage 9 deliverable for the tower redesign.

The purpose is to specify what success, truncation, hard violation, invalid action, invalid extension, parent failure, and reward cutoff mean in the tower model.

This is a mathematical and design contract, not an implementation file.

The old `rl_counterpoint` system already made several decisions that should carry forward:

| Existing behavior | Old source |
| --- | --- |
| invalid step-delta actions are no-ops with an invalid-action penalty | `rl_counterpoint/envs/counterpoint_env.py` |
| episodes truncate at the effective max step | `rl_counterpoint/envs/termination.py` |
| effective max step is the minimum of configured max steps and reward deadline | `rl_counterpoint/envs/counterpoint_env.py` |
| reward functions may declare terminal success | `rl_counterpoint/reward/protocol.py` |
| reward functions may declare hard violation | `rl_counterpoint/reward/protocol.py` |
| final-step context controls terminal rewards | `rl_counterpoint/reward/protocol.py` |
| goal-directed reward shuts off after the derived deadline | `rl_counterpoint/reward/black_box.py` |

The tower adds one new family of cases: failures of rankwise projection and extension. These are not mysterious. They are just the success/failure semantics induced by the graph-morphism requirement.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 3: Freeze Reward Ownership |
| Stage | Stage 9: Define success/failure semantics |
| Action | Extract old termination semantics and lift them to rankwise tower training |

Stage 9 exit criterion:

| Requirement | Status |
| --- | --- |
| define terminal success | drafted here |
| define truncation | drafted here |
| define hard violation | drafted here |
| define invalid action behavior | drafted here |
| define invalid extension behavior | drafted here |
| define parent failure behavior | drafted here |
| define reward cutoff after deadline | drafted here |
| define terminal/failure diagnostics | drafted here |

## Outcome Vocabulary

At each step of a rank-$k$ episode, the environment/action/reward system returns an outcome:

$$
\mathcal E_t^k
=
\left(
s_{t+1}^k,\;
r_t^k,\;
\mathsf{terminated}_t^k,\;
\mathsf{truncated}_t^k,\;
\mathsf{info}_t^k
\right).
$$

The outcome vocabulary is:

| Outcome | Meaning |
| --- | --- |
| valid step | proposed action is a valid rank-$k$ graph edge |
| invalid action | proposed action does not decode to a valid rank-$k$ edge |
| invalid extension | parent action is valid, but the proposed rank-$k$ extension is not |
| terminal success | reward/goal contract declares the episode successful |
| truncation | episode stops because deadline or max-step limit is reached |
| hard violation | reward contract declares a severe rule violation |
| parent failure | lower-rank scaffold fails before or during higher-rank extension |

These terms should remain separate. In particular, terminal success and truncation are not the same event.

## Valid Step

A rank-$k$ step is valid when:

$$
s_t^k\to s_{t+1}^k
$$

is an edge of $G(k)_\bullet$, and its projection

$$
\operatorname{pr}(s_t^k)\to \operatorname{pr}(s_{t+1}^k)
$$

is the corresponding parent edge in $G(k-1)_\bullet$.

Equivalently, in action form:

$$
s_{t+1}^k=s_t^k+\Delta s_t^k
$$

and:

$$
\Delta s_t^{k-1}=\operatorname{pr}(\Delta s_t^k).
$$

Valid steps are scored by $R_k$.

## Invalid Action

The old system behavior should carry forward:

| Behavior | Tower rule |
| --- | --- |
| invalid action is not applied | preserve |
| state remains unchanged | preserve |
| history records the no-op state | preserve unless implementation chooses stricter rejection |
| invalid-action penalty is returned | preserve |
| terminated is false | preserve |
| truncated may still become true if the no-op advances past the step cap | preserve |
| diagnostics explain invalidity | preserve |

Mathematically, if $\Delta s_t^k$ does not define a valid rank-$k$ edge, then:

$$
s_{t+1}^k=s_t^k
$$

and:

$$
r_t^k=r_{\mathrm{invalid}}.
$$

The info payload should contain:

| Diagnostic | Meaning |
| --- | --- |
| `valid_action` | false |
| `invalid_action_reason` | why the action failed |
| `source` | attempted source |
| `target` | decoded attempted target, if available |
| `step_delta` | attempted action |
| `rank` | active rank |

## Invalid Extension

Invalid extension is the tower-specific version of invalid action.

At rank $k>1$, suppose the lower-rank parent action is valid:

$$
\Delta s_t^{k-1}
$$

but the proposed extension coordinate fails to assemble into a valid rank-$k$ edge:

$$
\Delta s_t^k
\notin
(\operatorname{pr}^k)^{-1}(\Delta s_t^{k-1})
\cap
\partial_0^{-1}(s_t^k).
$$

Then the rank-$k$ action is an invalid extension.

The proposed semantics are:

| Behavior | Rule |
| --- | --- |
| parent action remains conceptually valid | yes |
| rank-$k$ state should not advance | yes |
| rank-$k$ reward should be invalid-extension penalty | yes |
| rank-$k$ episode should not be terminal success | yes |
| rank-$k$ episode may truncate if step cap is reached | yes |
| diagnostics must distinguish invalid extension from invalid parent action | yes |

So:

$$
s_{t+1}^k=s_t^k,
\qquad
r_t^k=r_{\mathrm{invalidExtension}}.
$$

This distinction matters because an invalid extension means the tower policy failed to add a legal new coordinate over an otherwise legal scaffold.

## Parent Failure

During training of $\pi^k$, the lower-rank scaffold is assumed to be produced by an already-trained parent policy or by an accepted parent trajectory.

If the parent process fails, the higher-rank episode cannot honestly continue as a rank-$k$ extension. The proposed semantics are:

| Parent event | Rank-$k$ consequence |
| --- | --- |
| parent terminal success | rank-$k$ may continue only if its own terminal/goal condition still allows it |
| parent truncation | rank-$k$ truncates |
| parent invalid action | rank-$k$ marks parent failure |
| parent hard violation | rank-$k$ marks parent failure |
| parent has no legal action | rank-$k$ truncates or fails with diagnostic |

The main rule:

$$
\text{a higher-rank trajectory cannot outlive an invalid lower-rank scaffold.}
$$

This follows from the graph-morphism requirement. If the projected edge is not valid in $G(k-1)_\bullet$, then the higher edge cannot be valid in $G(k)_\bullet$.

## Terminal Success

The old system lets reward functions declare terminal success:

$$
\mathsf{terminalSuccess}\in\{\mathrm{true},\mathrm{false}\}.
$$

The tower should preserve reward-owned terminal success.

At rank $k$, terminal success means:

1. the rank-$k$ trajectory satisfied its goal condition,
2. the success happens before or at the reward deadline,
3. the relevant terminal musical condition for rank $k$ is satisfied,
4. the projected parent trajectory is not failed.

In notation:

$$
\mathsf{terminalSuccess}^k
=
\mathsf{goalSuccess}^k
\wedge
\mathsf{deadlineOK}
\wedge
\mathsf{terminalMusicOK}^k
\wedge
\neg\mathsf{parentFailure}^{k-1}.
$$

The exact $k$-specific musical condition belongs to the reward spec, not this document. Examples:

| Rank | Terminal success may require |
| --- | --- |
| $G(1)$ | melody reaches target octave/region |
| $G(2)$ | dyadic cadence or outer-voice terminal condition |
| $G(3)$ | triadic terminal sonority condition |
| $G(4)$ | four-voice terminal sonority/cadence condition |

## Truncation

The old system truncates when:

$$
t\ge \min(\mathsf{maxSteps},\mathsf{rewardDeadlineStep}).
$$

The tower should preserve this:

$$
\mathsf{truncated}^k_t
=
\left[
t\ge
\min(\mathsf{maxSteps},\mathsf{rewardDeadlineStep})
\right].
$$

The deadline is shared through projection because it is based on the root/pedal line and horizontal movement budget, not on vertical chord width:

$$
\operatorname{pr}(\mathsf{deadline}^k)=\mathsf{deadline}^{k-1}.
$$

Truncation is not success. A truncated episode may still have accumulated useful rewards, but it did not terminate by success unless the reward output also declares terminal success before truncation.

## Reward Cutoff After Deadline

The old goal reward shuts off after the derived deadline. The tower should preserve this.

For goal-directed reward terms:

$$
t\ge \mathsf{rewardDeadlineStep}
\quad\Longrightarrow\quad
R^{\mathrm{goal}}_k=0.
$$

Likewise, terminal goal bonuses should not pay after deadline:

$$
t\ge \mathsf{rewardDeadlineStep}
\quad\Longrightarrow\quad
R^{\mathrm{terminalGoal}}_k=0.
$$

This does not necessarily mean every diagnostic or every non-goal penalty is zero after deadline. It means no further positive goal/scoring pressure should be awarded after the rewardable deadline. The environment should normally truncate at that boundary anyway.

## Hard Violation

The old reward output includes `hard_violation`.

In the tower design, most absolute impossibilities should be graph pruning rather than reward-level hard violations. Examples:

| Rule family | Preferred handling |
| --- | --- |
| invalid node | pruning |
| invalid edge | pruning |
| voice crossing | pruning |
| parallel fifths | pruning |
| projection-incompatible edge | pruning |
| doubled leading tone if adopted as hard rule | pruning |

Hard violation should be reserved for cases that are not graph-pruned but should still be treated as severe episode failures or training diagnostics.

Proposed Stage 9 semantics:

| Hard violation does | Hard violation does not necessarily do |
| --- | --- |
| reports severe reward-level failure | automatically replace graph pruning |
| may terminate the episode if configured | always terminate by definition |
| must be exposed in diagnostics | imply terminal success |

Before implementation, the training protocol should decide whether:

$$
\mathsf{hardViolation}\Rightarrow \mathsf{terminated}
$$

or whether it remains only an info flag plus large penalty.

## Rankwise Success Responsibilities

Each rank has its own success/failure responsibility:

| Rank | Success responsibility | Failure responsibility |
| --- | --- | --- |
| $G(1)$ | root/pedal melody reaches target region and satisfies melody terminal terms | invalid melody move, missed deadline, hard melody violation |
| $G(2)$ | valid outer extension over $G(1)$, dyadic cadence/terminal terms | invalid extension, parent failure, missed deadline |
| $G(3)$ | valid inserted voice over $G(2)$, triadic terminal terms | invalid extension, parent failure, missed deadline |
| $G(4)$ | valid inserted voice over $G(3)$, four-voice terminal terms | invalid extension, parent failure, missed deadline |
| $G(n>4)$ | repeat $G(4)$-style extension | invalid extension, parent failure, missed deadline |

This matches the rank-local reward ownership from Stage 7.

## Diagnostic Contract

Terminal and failure events should be explicit in diagnostics.

Required outcome diagnostics:

| Diagnostic | Meaning |
| --- | --- |
| `rank` | active rank |
| `outcome_kind` | valid, invalid_action, invalid_extension, parent_failure, terminal_success, truncated, hard_violation |
| `terminated` | whether environment returned terminated |
| `truncated` | whether environment returned truncated |
| `is_terminal_success` | reward-declared success |
| `hard_violation` | reward-declared hard violation |
| `valid_action` | whether rank-$k$ action was valid |
| `valid_parent_action` | whether projected parent action was valid |
| `invalid_action_reason` | reason for invalid rank-$k$ action |
| `invalid_extension_reason` | reason parent action could not be legally extended |
| `parent_failure_reason` | lower-rank failure reason, if any |
| `step_index` | current step |
| `max_steps` | configured max steps |
| `reward_deadline_step` | reward deadline |
| `deadline_active` | whether deadline has shut off goal rewards |
| `is_final_step` | whether terminal rewards may fire |

## Old-System Carryovers

Carry forward:

| Old behavior | Tower decision |
| --- | --- |
| invalid actions are no-op penalties | yes |
| invalid actions include diagnostics | yes |
| truncation is step-limit based | yes |
| reward deadline can be earlier than max steps | yes |
| target/deadline persists across an episode | yes |
| reward owns terminal success flag | yes |
| reward owns hard violation flag | yes |
| final-step flag gates terminal rewards | yes |
| goal rewards stop after deadline | yes |

## Tower-Only Additions

Add:

| Addition | Why needed |
| --- | --- |
| invalid extension | distinguishes bad extension from bad parent action |
| parent failure | higher-rank trajectory depends on lower-rank scaffold |
| projected validity diagnostics | confirms graph-morphism compatibility |
| rankwise outcome kind | makes terminal/failure traces interpretable |
| extension-specific penalty | trains tier $k$ to choose legal additions |

## Unresolved Choices

These do not block the Stage 9 draft, but they should be resolved before implementation:

| Choice | Proposed default |
| --- | --- |
| should hard violation terminate immediately? | probably yes for tower training, but needs explicit acceptance |
| should invalid extension advance time? | yes, matching old invalid-action no-op semantics |
| should invalid extension record repeated state in history? | yes, matching old invalid-action behavior |
| should parent truncation be inherited as truncation or failure? | truncation |
| should parent terminal success stop child rollout immediately? | depends on whether child has already satisfied its terminal condition |
| should $\mathrm{V}\to\mathrm{IV}$ be pruning or hard violation? | unresolved from Stage 7 |

## Stage 9 Completion Checklist

Stage 9 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| invalid rank-$k$ actions are no-op penalties | yes |
| invalid extensions are a distinct invalid-action subtype | yes |
| higher-rank trajectories cannot continue over failed parent scaffold | yes |
| truncation is inherited from old max-step/deadline semantics | yes |
| goal rewards stop after deadline | yes |
| terminal success remains reward-owned | yes |
| hard violation remains reward-owned but may terminate if configured | yes |
| terminal/failure diagnostics must distinguish all outcome kinds | yes |

Once accepted, Phase 3 is complete. The next phase is Phase 4: Freeze Training Protocol.
