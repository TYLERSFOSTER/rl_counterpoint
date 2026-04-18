# Training Protocol

This document is the Phase 4 / Stage 10 deliverable for the tower redesign.

The purpose is to define the stagewise training lifecycle for the tower policies:

\[
\pi^1,\pi^2,\pi^3,\pi^4,\dots
\]

This is a mathematical and design contract, not an implementation file.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 4: Freeze Training Protocol |
| Stage | Stage 10: Define stagewise training lifecycle |
| Action | Specify the rankwise training order, freezing rule, parent sampling behavior, active exploration, reward ownership, and terminal-goal lifting |

Stage 10 exit criterion:

| Requirement | Status |
| --- | --- |
| define training order | drafted here |
| define what "freeze" means | drafted here |
| define parent policy behavior during child training | drafted here |
| define active child exploration | drafted here |
| define reward used for optimization | drafted here |
| define rank-local terminal success | drafted here |
| define rollback/no-rollback behavior | drafted here |
| define checkpoint lineage requirement | drafted here |

## Core Lifecycle

Training proceeds sequentially by rank:

\[
k=1,2,3,\dots
\]

The policies are trained in order:

\[
\pi^1 \longrightarrow \pi^2 \longrightarrow \pi^3 \longrightarrow \pi^4 \longrightarrow \cdots
\]

The rank \(k=2\) policy is the outer-voice policy. It extends the \(G(1)\) root/pedal line by adding the top voice. Ranks \(k\ge 3\) add interior voices according to the tower projection convention.

Each tier has its own configured episode budget:

\[
E_1,E_2,E_3,\dots
\]

Training for rank \(k\) stops when the rank-\(k\) episode count reaches \(E_k\). For the present design stage, there is no additional automatic acceptance rule beyond completing the configured episode budget.

## Freeze Rule

When training \(\pi^k\):

\[
\pi^1,\dots,\pi^{k-1}
\]

are frozen.

Frozen means:

| Property | Meaning |
| --- | --- |
| no gradient updates | parent policy parameters do not change |
| no optimizer updates | parent optimizer state does not change |
| checkpoint read-only | parent policy is loaded as a fixed artifact |
| parent reward not optimized | parent reward may be logged, but does not drive gradients |

Policies above rank \(k\),

\[
\pi^{k+1},\pi^{k+2},\dots
\]

are not trained and are not relevant yet.

Only the active policy \(\pi^k\) is trainable.

## Parent Scaffold Behavior

During training of \(\pi^k\), the lower-rank frozen policies provide the parent scaffold.

The parent policies should not perform full training-style exploration. They should act mostly greedily, with a small controlled local randomness knob.

The intended parent sampler is a top-\(m\) greedy sampler:

1. compute the frozen parent policy distribution or scores,
2. identify the top \(m_{\mathrm{parent}}\) legal actions,
3. choose among those top actions using a configured randomness rule,
4. fall back to greedy/argmax when \(m_{\mathrm{parent}}=1\) or randomness is zero.

Symbolically:

\[
\mu_{\mathrm{parent}}^j
=
\operatorname{TopMRandomized}
\left(
\pi^j,\;
m_{\mathrm{parent}}^j,\;
\rho_{\mathrm{parent}}^j
\right),
\qquad
j<k.
\]

Here:

| Symbol | Meaning |
| --- | --- |
| \(m_{\mathrm{parent}}^j\) | number of top parent actions considered |
| \(\rho_{\mathrm{parent}}^j\) | parent randomness strength |
| \(m_{\mathrm{parent}}^j=1\) | pure greedy parent behavior |
| \(\rho_{\mathrm{parent}}^j=0\) | pure greedy parent behavior |

This is intentionally different from broad uniform exploration. The frozen parent is supposed to provide a competent scaffold, not rediscover its own problem.

## Active Child Exploration

The active rank-\(k\) policy should use forced exploration during training.

The default exploration model remains:

\[
\mu^k(\alpha\mid s)
=
(1-\epsilon_k)\pi^k(\alpha\mid s)
+\epsilon_k U_k(\alpha\mid s),
\]

where \(U_k\) is the uniform distribution over legal rank-\(k\) actions or legal rank-\(k\) extension choices.

The purpose is to let the active tier discover useful extensions rather than immediately collapsing to its current greedy behavior.

The exploration parameter is rank-local:

\[
\epsilon_1,\epsilon_2,\epsilon_3,\dots
\]

may differ by tier.

## Action Assembly During Training

At rank \(k\), the full action is assembled from the frozen parent action and the active tier's new coordinate.

For \(k=1\), the policy chooses:

\[
\Delta s_t^1=(\Delta\lambda_{0,t}).
\]

For \(k=2\), the parent provides:

\[
\Delta s_t^1=(\Delta\lambda_{0,t})
\]

and \(\pi^2\) chooses the outer-voice coordinate:

\[
\Delta\lambda_{1,t}.
\]

The assembled action is:

\[
\Delta s_t^2=(\Delta\lambda_{0,t},\Delta\lambda_{1,t}).
\]

For \(k\ge 3\), the parent action is:

\[
\Delta s_t^{k-1}
=
(\Delta\lambda_{0,t},\dots,\Delta\lambda_{k-3,t},\Delta\lambda_{k-1,t}),
\]

and \(\pi^k\) chooses the new interior coordinate:

\[
\Delta\lambda_{k-2,t}.
\]

The assembled action is:

\[
\Delta s_t^k
=
(\Delta\lambda_{0,t},\dots,\Delta\lambda_{k-3,t},\Delta\lambda_{k-2,t},\Delta\lambda_{k-1,t}).
\]

This is the action-side counterpart of the graph projection:

\[
\operatorname{pr}^k(\Delta s_t^k)=\Delta s_t^{k-1}.
\]

## Reward Used For Optimization

When training \(\pi^k\), the optimization objective uses only the rank-\(k\) reward:

\[
R_k.
\]

Parent rewards may be computed as diagnostics, but they do not drive gradients by default:

\[
\nabla_\theta J_k
\text{ uses }R_k\text{ only.}
\]

The reason is rank-local ownership. The lower ranks have already been trained for their own responsibilities. The active rank should learn the new extension problem.

If this causes an obvious practical problem later, parent rewards can be reconsidered as auxiliary diagnostics or auxiliary terms, but the Stage 10 default is:

\[
\text{optimize only }R_k.
\]

## Rank-Local Terminal Success

The terminal goal is not one identical target predicate reused at every rank.

The correct tower rule is:

\[
\mathsf{Success}_k(W_t^k)
=
\mathsf{Success}_{k-1}(\operatorname{pr}^k W_t^k)
\wedge
\mathsf{NewTerminalCondition}_k(W_t^k).
\]

So success is also rank-local and lifted through the tower.

For \(k=1\), the terminal condition is the projection of the two-voice perfect cadence onto the root/pedal line:

\[
\mathsf{Success}_1(W_t^1)
=
\left[
W_t^1
\models
\operatorname{pr}^2(\text{perfect cadence})
\right].
\]

Concretely, the rank-1 passage terminates when the root/pedal line realizes the root-note motion of a perfect cadence in the correct terminal window and metrical position.

For \(k=2\), the terminal condition is:

\[
\mathsf{Success}_2(W_t^2)
=
\mathsf{Success}_1(\operatorname{pr}^2 W_t^2)
\wedge
\left[
\text{outer voice supplies the third of the cadence chords}
\right].
\]

For \(k\ge 3\), the same pattern repeats:

\[
\mathsf{Success}_k(W_t^k)
=
\mathsf{Success}_{k-1}(\operatorname{pr}^k W_t^k)
\wedge
\mathsf{NewTerminalCondition}_k(W_t^k).
\]

The exact new terminal condition for each higher tier belongs to the reward specification and can be refined rank by rank.

## Passage Clock And Deadline

There is one passage clock during a rank-\(k\) episode:

\[
t=0,1,2,\dots
\]

The parent scaffold and active child extension are evaluated at the same time step. The parent is not running a separate episode clock.

The deadline remains a shared passage budget because it is based on the root/pedal line and horizontal movement budget:

\[
\operatorname{pr}(\mathsf{deadline}^k)=\mathsf{deadline}^{k-1}.
\]

However, the terminal success predicate is rank-local as described above. So the system has:

| Shared across ranks | Rank-local |
| --- | --- |
| passage clock | success predicate |
| target/root budget | new terminal condition |
| deadline | reward ownership |
| measure structure | extension legality |

## Parent Failure

If a frozen parent policy cannot produce a valid parent action, then the rank-\(k\) episode cannot honestly continue.

The training protocol inherits the Stage 9 decision:

\[
\text{parent failure breaks the current rank-}k\text{ episode.}
\]

This should produce a parent-failure diagnostic and truncate or fail the child episode according to the success/failure semantics document.

## Parent Early Success

Parent success should not be treated as a separate parent episode ending early.

During rank-\(k\) training, the parent scaffold is part of the same passage. The active rank evaluates:

\[
\mathsf{Success}_{k-1}(\operatorname{pr}^k W_t^k)
\]

inside its own terminal condition.

So parent success is necessary for child terminal success, but not by itself sufficient to stop the child unless the full rank-\(k\) success predicate is also satisfied.

## Trajectory Storage

The default trajectory record should store the full rank-\(k\) trajectory:

\[
\left(
s_t^k,\;
\Delta s_t^k,\;
r_t^k,\;
\mathcal C_t^k,\;
\mathcal O_t^k
\right)_{t}
\]

Projected lower-rank trajectories can be computed from the full rank-\(k\) trajectory when needed:

\[
\operatorname{pr}^k(s_t^k),
\qquad
\operatorname{pr}^k(\Delta s_t^k),
\qquad
\operatorname{pr}^k(W_t^k).
\]

Therefore the Stage 10 default is:

\[
\text{store full rank-}k\text{ trajectory; compute projections when needed.}
\]

Implementation may cache projections for diagnostics or speed, but the mathematical record is the full rank-\(k\) rollout.

## Checkpoint Lineage

Every rank-\(k\) checkpoint for \(k>1\) should record the exact parent checkpoint used during training.

For example:

```text
pi_3_checkpoint:
  parent_rank: 2
  parent_checkpoint: pi_2_checkpoint_...
```

The purpose is lineage tracking. If a higher-rank policy succeeds or fails, we need to know which frozen scaffold it was trained over.

This does not mean the parent is modified. It only records dependency.

## Acceptance Rule

For the current design stage, a rank checkpoint is accepted when its configured episode budget completes:

\[
\text{accepted}(\pi^k)
\quad\Longleftrightarrow\quad
\text{episodes completed}=E_k.
\]

Future stages may add validation metrics, generated MIDI review, or manual approval gates. Stage 10 does not require those.

## No Rollback Within A Lineage

If training \(\pi^{k+1}\) exposes a flaw in \(\pi^k\), the current lineage does not go back and mutate \(\pi^k\).

The rule is:

\[
\text{no rollback within a lineage.}
\]

If a lower-rank flaw is serious, a later experiment can start a new lineage from a retrained lower-rank policy. But the current rank-\(k\) training run treats its frozen parents as fixed.

## Hyperparameters

The training protocol introduces these rankwise hyperparameters:

| Symbol | Meaning |
| --- | --- |
| \(E_k\) | episode budget for rank \(k\) |
| \(\epsilon_k\) | active child forced-exploration rate |
| \(m_{\mathrm{parent}}^k\) | parent top-\(m\) sampler width |
| \(\rho_{\mathrm{parent}}^k\) | parent randomness strength |
| \(r_{\mathrm{invalidExtension}}^k\) | penalty for invalid rank-\(k\) extension |
| \(r_{\mathrm{invalid}}^k\) | penalty for invalid rank-\(k\) action |

Defaults are implementation-stage decisions.

## Stage 10 Completion Checklist

Stage 10 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| train ranks sequentially | yes |
| rank \(2\) is the outer-voice tier | yes |
| each rank has its own episode budget | yes |
| lower policies are frozen while current rank trains | yes |
| higher policies are irrelevant until their rank is reached | yes |
| frozen parents use top-\(m\) mostly-greedy randomness | yes |
| active child policy uses forced exploration | yes |
| optimize only \(R_k\) while training \(\pi^k\) | yes |
| terminal success is rank-local and lifted | yes |
| parent failure breaks the current child episode | yes |
| parent early success is necessary but not sufficient for child success | yes |
| store full rank-\(k\) trajectory and compute projections as needed | yes |
| checkpoints record parent lineage | yes |
| no rollback inside a lineage | yes |

Once accepted, the next stage is Phase 4 / Stage 11: Define rollout semantics.
