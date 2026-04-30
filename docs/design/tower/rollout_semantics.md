# Rollout Semantics

This document is the Phase 4 / Stage 11 deliverable for the tower redesign.

The purpose is to define what happens inside one rank-$k$ rollout: how parent policies are stepped, how child actions are masked and sampled, how rank-$k$ actions are assembled, how invalid extensions are handled, and what data each timestep records.

This is a mathematical and design contract, not an implementation file.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 4: Freeze Training Protocol |
| Stage | Stage 11: Define rollout semantics |
| Action | Specify per-step rollout choreography for rank-$k$ training |

Stage 11 exit criterion:

| Requirement | Status |
| --- | --- |
| define parent-first sampling | drafted here |
| define lift-fiber action masking | drafted here |
| define active child sampling | drafted here |
| define empty-fiber behavior | drafted here |
| define invalid-extension behavior | drafted here |
| define rollout record fields | drafted here |
| define policy-gradient log-probability ownership | drafted here |
| define parent diagnostics | drafted here |

## Big Picture

During rank-$k$ training, the active trainable policy is $\pi^k$.

Lower policies

$$
\pi^1,\dots,\pi^{k-1}
$$

are frozen and provide the parent scaffold. Higher policies are not involved.

A rank-$k$ rollout is a trajectory:

$$
\tau^k
=
\left(
\mathcal Z_0^k,\mathcal Z_1^k,\dots,\mathcal Z_T^k
\right)
$$

where each $\mathcal Z_t^k$ is the recorded data for one timestep.

The fundamental choreography is:

1. construct the rank-$k$ window $W_t^k$,
2. project as needed to obtain lower-rank parent inputs,
3. sample frozen parent action first,
4. mask child choices to actions lying over the parent action,
5. sample the active child extension,
6. assemble the full rank-$k$ action,
7. check rank-$k$ graph legality,
8. apply reward and outcome semantics,
9. record the full rank-$k$ step plus parent diagnostics.

## Parent-First Sampling

For rank $k>1$, the parent action is sampled first.

At time $t$, the current rank-$k$ state is:

$$
s_t^k.
$$

Its parent state is:

$$
s_t^{k-1}=\operatorname{pr}^k(s_t^k).
$$

The rank-$k$ window is:

$$
W_t^k.
$$

The parent window is computed by projection when needed:

$$
W_t^{k-1}=\operatorname{pr}^k(W_t^k).
$$

The frozen parent policy proposes:

$$
\Delta s_t^{k-1}.
$$

This parent action is sampled using the Stage 10 parent sampler:

$$
\mu_{\mathrm{parent}}^{k-1}
=
\operatorname{TopMRandomized}
\left(
\pi^{k-1},\;
m_{\mathrm{parent}}^{k-1},\;
\rho_{\mathrm{parent}}^{k-1}
\right).
$$

The parent sampler is mostly greedy with controlled top-$m$ randomness. It is not broad uniform exploration.

## Lift-Fiber Child Mask

Once the parent action $\Delta s_t^{k-1}$ is specified, the active child policy may only choose rank-$k$ actions lying over that parent action.

The legal lift fiber is:

$$
A_k(s_t^k;\Delta s_t^{k-1})
=
\left\{
\Delta s_t^k
\in
\partial_0^{-1}(s_t^k)
\;\middle|\;
\operatorname{pr}^k(\Delta s_t^k)=\Delta s_t^{k-1}
\right\}.
$$

Equivalently, the mask removes all outgoing arrows from $s_t^k$ that do not lie over the chosen parent arrow:

$$
\Delta s_t^k
\text{ is mask-legal}
\quad\Longleftrightarrow\quad
\Delta s_t^k\in A_k(s_t^k;\Delta s_t^{k-1}).
$$

This is the concrete training-speed mechanism: the active tier does not search the whole outgoing star. It searches only the fiber of legal lifts over the parent action.

## Active Child Sampling

The active policy $\pi^k$ chooses only the rank-local new coordinate.

For $k=1$:

$$
\Delta s_t^1=(\Delta\lambda_{0,t}).
$$

For $k=2$, the parent action is:

$$
\Delta s_t^1=(\Delta\lambda_{0,t})
$$

and the child chooses:

$$
\Delta\lambda_{1,t}.
$$

For $k\ge 3$, the child chooses:

$$
\Delta\lambda_{k-2,t}.
$$

The active child samples from the legal lift fiber when the fiber is nonempty:

$$
\Delta s_t^k\sim \mu^k(\cdot\mid W_t^k,\Delta s_t^{k-1})
\quad\text{with support contained in}\quad
A_k(s_t^k;\Delta s_t^{k-1}).
$$

The active training sampler may include forced exploration:

$$
\mu^k
=
(1-\epsilon_k)\pi^k+\epsilon_k U_k,
$$

but $U_k$ is interpreted over the legal child extension choices when the lift fiber is nonempty.

## Action Assembly

For $k=1$:

$$
\Delta s_t^1=(\Delta\lambda_{0,t}).
$$

For $k=2$:

$$
\Delta s_t^2
=
(\Delta\lambda_{0,t},\Delta\lambda_{1,t}).
$$

For $k\ge 3$, the parent action has the projected form:

$$
\Delta s_t^{k-1}
=
(\Delta\lambda_{0,t},\dots,\Delta\lambda_{k-3,t},\Delta\lambda_{k-1,t}),
$$

and the active child inserts the new coordinate:

$$
\Delta s_t^k
=
(\Delta\lambda_{0,t},\dots,\Delta\lambda_{k-3,t},\Delta\lambda_{k-2,t},\Delta\lambda_{k-1,t}).
$$

The assembled action must satisfy:

$$
\operatorname{pr}^k(\Delta s_t^k)=\Delta s_t^{k-1}.
$$

If this fails, the action is not a valid lift of the parent action.

## Empty Lift Fiber

The empty-fiber case is exceptional:

$$
A_k(s_t^k;\Delta s_t^{k-1})=\varnothing.
$$

This should usually be rare if the graph and parent sampler are well calibrated. Possible causes include:

| Cause | Example |
| --- | --- |
| MIDI boundary | no new voice move stays in $\{0,\dots,127\}$ |
| spacing constraints | no inserted voice can satisfy adjacent/outer spacing |
| crossing constraints | every possible lift crosses another voice |
| aggressive parent action | parent move is legal below but has no legal higher lift |
| cadence/deadline corner | parent move is legal but cannot be realized by a legal higher sonority |

The Stage 11 rule is:

1. diagnose the event as `empty_lift_fiber`,
2. sample outside the empty legal fiber from a broader child proposal space,
3. mark the resulting event as an invalid extension,
4. apply invalid-extension semantics.

This fallback exists so training records the failure rather than silently crashing or disappearing. If `empty_lift_fiber` happens often, it is evidence of a graph/spec/training calibration problem.

## Invalid Extension Semantics

An invalid extension occurs when the parent action is conceptually valid but the active tier fails to produce a valid rank-$k$ lift.

This can happen because:

1. the legal lift fiber is empty,
2. a fallback action was sampled outside the legal fiber,
3. an implementation bug or stale mask allowed an illegal child action,
4. the assembled action fails rank-$k$ graph legality.

Invalid extension follows the Stage 9 no-op penalty behavior:

$$
s_{t+1}^k=s_t^k,
\qquad
r_t^k=r_{\mathrm{invalidExtension}}^k.
$$

It advances time:

$$
t\mapsto t+1.
$$

It is not terminal success. It may still cause truncation if the step cap/deadline is reached.

Diagnostics must distinguish invalid extension from invalid parent action.

## Parent Failure

If the frozen parent policy cannot produce a valid parent action, then the rank-$k$ rollout cannot continue honestly.

The rollout must record a parent-failure event:

$$
\mathsf{parentFailure}^{k-1}_t=\mathrm{true}.
$$

The rank-$k$ outcome then follows the Stage 9 parent-failure semantics: the child episode truncates or fails with explicit diagnostics.

This is separate from invalid extension. Parent failure means the lower scaffold itself failed. Invalid extension means the parent scaffold was valid but the current rank failed to lift it.

## Policy-Gradient Ownership

The policy-gradient loss for rank $k$ uses only the active tier's log probability.

For a sampled active choice $\alpha_t^k$, usually the new coordinate:

$$
\alpha_t^k=\Delta\lambda_{\mathrm{new},t},
$$

the trainable log probability is:

$$
\log\pi^k(\alpha_t^k\mid W_t^k,\Delta s_t^{k-1}).
$$

The frozen parent log probabilities are diagnostics only. They are not included in the rank-$k$ gradient.

This remains true even though many rank-$k$ reward computations involve lower-rank data inside the rank-$k$ setting. The point is that no specific reward fact should be double-scored across ranks.

## Rollout Record

The rollout record uses Option C: full rank-$k$ step record plus parent diagnostics.

Each timestep record $\mathcal Z_t^k$ should contain:

| Field | Required? | Meaning |
| --- | --- | --- |
| `rank` | yes | active rank $k$ |
| `source_state` | yes | $s_t^k$ |
| `window` | yes | $W_t^k$ |
| `parent_action` | yes for $k>1$ | $\Delta s_t^{k-1}$ |
| `active_action` | yes | $\alpha_t^k=\Delta\lambda_{\mathrm{new},t}$ |
| `assembled_action` | yes | $\Delta s_t^k$ |
| `target_state` | yes | $s_{t+1}^k$ |
| `active_logprob` | yes | trainable log probability for $\pi^k$ |
| `reward` | yes | $r_t^k$ |
| `terminated` | yes | terminal-success endpoint flag |
| `truncated` | yes | deadline/max-step endpoint flag |
| `outcome_diagnostics` | yes | validity, terminal, truncation, and failure diagnostics |
| `reward_diagnostics` | yes | reward-term diagnostics |
| `parent_logprob` | diagnostic | frozen parent log probability |
| `parent_sampler_mode` | diagnostic | greedy/top-$m$/randomness details |
| `parent_top_m` | diagnostic | top-$m$ sampler width |
| `parent_randomness` | diagnostic | parent randomness strength |
| `parent_action_rank` | diagnostic | rank of parent action |
| `projection_diagnostics` | diagnostic | checks that assembled action projects correctly |

The projected parent window/context is not required as stored data. Store $W_t^k$ as the required window and compute:

$$
\operatorname{pr}^k(W_t^k)
$$

on demand. Implementations may cache projected windows for debugging or speed, but the mathematical rollout record requires only the full rank-$k$ window.

## Timestep Data Flow

Each rank-$k$ timestep has three phases.

### Before action

Available:

$$
s_t^k,\quad
W_t^k,\quad
\operatorname{pr}^k(W_t^k),\quad
\text{meter/goal/deadline context}.
$$

For $k>1$, frozen parent policies are also available.

### During action

Compute:

$$
\Delta s_t^{k-1}
\quad\text{from frozen parent policies},
$$

$$
A_k(s_t^k;\Delta s_t^{k-1})
\quad\text{as the legal lift fiber},
$$

$$
\alpha_t^k
\quad\text{as the active tier's sampled new coordinate},
$$

and assemble:

$$
\Delta s_t^k.
$$

### After action

Compute:

$$
s_{t+1}^k,
\qquad
R_k,
\qquad
\mathsf{terminated},
\qquad
\mathsf{truncated},
\qquad
\mathsf{diagnostics}.
$$

Then append $\mathcal Z_t^k$ to the trajectory.

## Rank 1 Special Case

At $k=1$, there is no parent.

The legal action set is simply the outgoing action set from $s_t^1$:

$$
A_1(s_t^1)=\partial_0^{-1}(s_t^1).
$$

The active policy samples:

$$
\Delta s_t^1.
$$

The rollout record omits parent-specific required fields or sets them to null diagnostics.

## Stage 11 Completion Checklist

Stage 11 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| parent action is sampled first | yes |
| child action mask is the lift fiber over parent action | yes |
| active child samples from legal lift fiber when nonempty | yes |
| empty lift fiber is exceptional and diagnosed | yes |
| empty lift fiber falls back to sampled invalid extension event | yes |
| invalid extension is no-op, penalty, and time-advancing | yes |
| parent failure is distinct from invalid extension | yes |
| policy gradient uses only active tier log probability | yes |
| parent log probabilities are diagnostics only | yes |
| rollout record uses Option C | yes |
| only $W_t^k$ is required as stored window | yes |
| projected parent windows are computed on demand | yes |

Once accepted, the next stage is Phase 4 / Stage 12: Define artifact and checkpoint dependencies.
