# Reward Context Contracts

This document is reserved for Phase 3 / Stage 8: Define reward context contracts.

Status: draft.

This is a mathematical and design contract, not an implementation file.

The old `rl_counterpoint` system already made most of the relevant decisions:

| Existing decision | Old source |
| --- | --- |
| rewards receive source state, target state, and a context object | `rl_counterpoint/reward/protocol.py` |
| context carries step index, max steps, max step size, measure size, history, action delta, key pitch class, timed window, target octave, and final-step flag | `rl_counterpoint/reward/protocol.py` |
| context is produced by the environment but reward functions do not depend on the environment object | `rl_counterpoint/envs/counterpoint_env.py` |
| timed windows are fixed-length, left-padded, and carry a valid mask | `rl_counterpoint/envs/observation.py` |
| reward outputs are structured scalar results with hard-violation, terminal-success, and diagnostics fields | `rl_counterpoint/reward/protocol.py` |

Stage 8 should therefore preserve the old contract shape and lift it rankwise through the tower projections.

The project manager's observation is that this generalization is routine because the reward context tower is induced by the graph tower:

\[
W_t^k \longrightarrow W_t^{k-1},
\qquad
\Delta s_t^k \longmapsto \Delta s_t^{k-1},
\qquad
s_t^k \longmapsto s_t^{k-1}.
\]

Thus there is no need for ad hoc translation between ranks. The same projection structure that acts on states and actions acts on reward contexts.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 3: Freeze Reward Ownership |
| Stage | Stage 8: Define reward context contracts |
| Action | Extract old reward context decisions and lift them to the tower |

Stage 8 exit criterion:

| Requirement | Status |
| --- | --- |
| specify the exact context object passed to each rank reward | drafted here |
| specify projected parent context semantics | drafted here |
| specify window, meter, goal, and deadline fields | drafted here |
| specify diagnostics expectations | drafted here |
| separate old-system carryovers from tower-only additions | drafted here |

## Contract Shape

At rank \(k\), a reward function should conceptually have the form

\[
R_k:
\left(
s_t^k,\;
s_{t+1}^k,\;
\mathcal C_t^k
\right)
\longrightarrow
\mathcal O_t^k,
\]

where:

| Symbol | Meaning |
| --- | --- |
| \(s_t^k\) | rank-\(k\) source state |
| \(s_{t+1}^k\) | rank-\(k\) target state |
| \(\mathcal C_t^k\) | rank-\(k\) reward context |
| \(\mathcal O_t^k\) | structured reward output |

Equivalently, using the action explicitly:

\[
R_k(W_t^k,\Delta s_t^k;\mathcal C_t^k).
\]

The source-target and window-action presentations are equivalent because

\[
s_{t+1}^k=s_t^k+\Delta s_t^k
\]

whenever the transition is valid.

## Rank-\(k\) Reward Context

The rank-\(k\) context is:

\[
\mathcal C_t^k
=
\left(
k,\;
t,\;
s_t^k,\;
s_{t+1}^k,\;
\Delta s_t^k,\;
W_t^k,\;
\tau,\;
\mathsf{meter},\;
\mathsf{goal},\;
\mathsf{deadline},\;
\mathsf{terminal},\;
\mathsf{newFacts}_t^k
\right).
\]

This tuple should be read structurally, not as a required implementation order.

| Field | Meaning | Old-system source |
| --- | --- | --- |
| \(k\) | active rank | tower-only |
| \(t\) | current step index | `step_index` |
| \(s_t^k\) | source state | reward call argument |
| \(s_{t+1}^k\) | target state | reward call argument |
| \(\Delta s_t^k\) | step delta / action | `step_delta` |
| \(W_t^k\) | padded recent rank-\(k\) passage window | `timed_chord_window`, `history` |
| \(\tau\) | tonic or key pitch class | `key_pitch_class` |
| \(\mathsf{meter}\) | measure size and beat role | `measure_size`, bar-position helpers |
| \(\mathsf{goal}\) | target octave or target region | `target_root_octave` |
| \(\mathsf{deadline}\) | reward deadline/max-step semantics | `max_steps`, derived deadline |
| \(\mathsf{terminal}\) | final-step / success flags | `is_final_step`, reward result flags |
| \(\mathsf{newFacts}_t^k\) | facts newly introduced at rank \(k\) | tower-only |

## Projection Law

For \(k>1\), the context must project:

\[
\operatorname{pr}(\mathcal C_t^k)=\mathcal C_t^{k-1}.
\]

This means:

\[
\operatorname{pr}(s_t^k)=s_t^{k-1},
\qquad
\operatorname{pr}(s_{t+1}^k)=s_{t+1}^{k-1},
\qquad
\operatorname{pr}(\Delta s_t^k)=\Delta s_t^{k-1},
\qquad
\operatorname{pr}(W_t^k)=W_t^{k-1}.
\]

The scalar episode fields are shared:

\[
\operatorname{pr}(\tau)=\tau,
\qquad
\operatorname{pr}(\mathsf{meter})=\mathsf{meter},
\qquad
\operatorname{pr}(\mathsf{goal})=\mathsf{goal},
\qquad
\operatorname{pr}(\mathsf{deadline})=\mathsf{deadline}.
\]

So the projection is nontrivial on state-like and action-like objects, and identity-like on global episode metadata.

## Window Contract

The old system uses a fixed-length left-padded window with three synchronized sequences:

| Sequence | Meaning |
| --- | --- |
| chord sequence | recent chord states plus left padding |
| bar positions | metrical position for each chord or PAD metrical position |
| valid mask | whether each position is real history or padding |

The tower contract preserves this exactly, replacing flat chord states by rank-\(k\) states:

\[
W_t^k
=
\left(
(s_i^k)_{i\in I_t},\;
(b_i)_{i\in I_t},\;
(m_i)_{i\in I_t}
\right),
\]

where:

| Symbol | Meaning |
| --- | --- |
| \(I_t\) | fixed-length context index set |
| \(s_i^k\) | rank-\(k\) chord or PAD chord |
| \(b_i\) | bar position or PAD metrical position |
| \(m_i\in\{0,1\}\) | validity mask |

The projection of windows is coordinatewise:

\[
\operatorname{pr}(W_t^k)
=
\left(
(\operatorname{pr}(s_i^k))_{i\in I_t},\;
(b_i)_{i\in I_t},\;
(m_i)_{i\in I_t}
\right).
\]

Padding must project to padding. Since old padding is the zero chord of the current chord size, the tower interpretation is:

\[
\operatorname{pr}(\operatorname{PAD}_k)=\operatorname{PAD}_{k-1}.
\]

The old default window length is:

\[
\text{window length}=\text{context measures}\cdot\text{measure size}.
\]

The old default context-measure count is \(3\). Stage 8 does not need to change this; later implementation may expose it as a tower reward hyperparameter.

## History Contract

The old system separately carries raw history:

\[
\operatorname{Hist}_t=(s_0,\dots,s_t).
\]

The tower keeps:

\[
\operatorname{Hist}_t^k=(s_0^k,\dots,s_t^k).
\]

and projection is coordinatewise:

\[
\operatorname{pr}(\operatorname{Hist}_t^k)
=
\operatorname{Hist}_t^{k-1}.
\]

The window \(W_t^k\) is the fixed-length padded view of this history. Rewards may use raw history when they need the true unpadded episode prefix, and may use \(W_t^k\) when they need fixed-shape tensor-style computation.

## Action Context

The old system uses step-delta actions:

\[
\Delta s_t=(\Delta\lambda_0,\dots,\Delta\lambda_{n-1}).
\]

The tower keeps this representation rankwise:

\[
\Delta s_t^k=(\Delta\lambda_0,\dots,\Delta\lambda_{k-1}).
\]

For \(k>1\), the context also has the projected parent action:

\[
\Delta s_t^{k-1}=\operatorname{pr}(\Delta s_t^k).
\]

For \(k\ge 3\), the newly chosen coordinate is:

\[
\Delta\lambda_{\mathrm{new},t}=\Delta\lambda_{k-2,t}.
\]

For \(k=2\), the newly chosen coordinate is:

\[
\Delta\lambda_{\mathrm{new},t}=\Delta\lambda_{1,t}.
\]

This matches the tower-policy interpretation: the lower-rank scaffold supplies the parent action, and the current rank supplies the one new coordinate needed to assemble the rank-\(k\) action.

## Meter Contract

The old system already exposes:

| Quantity | Meaning |
| --- | --- |
| measure size | number of steps per measure |
| bar position | \(t\bmod\text{measure size}\) |
| leading beat | bar position \(0\) |
| downbeat | even step index |
| ending beat | final beat position in measure |

Stage 7 chose downbeat/upbeat as the preferred reward vocabulary.

Therefore Stage 8 defines:

\[
\operatorname{bar}(t)=t\bmod M,
\]

\[
\operatorname{leading}(t)=
\left[\operatorname{bar}(t)=0\right],
\]

\[
\operatorname{downbeat}(t)=
\left[t\equiv 0\pmod 2\right],
\]

\[
\operatorname{ending}(t)=
\left[\operatorname{bar}(t)=M-1\right],
\]

where \(M\) is the measure size.

The downbeat/upbeat reward language should be derived from \(\operatorname{downbeat}(t)\). If finer beat strength is later needed, it should extend this contract rather than replace it.

## Tonic And Pitch-Class Context

The old system passes `key_pitch_class`, not necessarily the full tonic MIDI value, into reward functions.

The tower contract requires at least:

\[
\tau_{12}=\tau\bmod 12.
\]

Reward terms then use:

\[
\operatorname{pc}_{\tau}(\lambda)
=
(\lambda-\tau_{12})\bmod 12.
\]

If a later reward term needs absolute tonic register, the context may also carry the full tonic MIDI value \(\tau\), but the Stage 8 minimum is the tonic pitch class.

## Goal And Deadline Context

The old system has a target root octave and a derived reward deadline.

The tower contract preserves:

| Quantity | Meaning |
| --- | --- |
| target root octave | target register region for the root/pedal |
| max steps | hard episode step cap |
| max step size | horizontal movement budget used for deadline |
| reward deadline step | last step at which goal-directed rewards are active |
| reward deadline measures | deadline measured in whole measures |
| final-step flag | whether current transition is the final rewardable step |

The old deadline rule is important:

\[
\text{deadline depends on horizontal root/pedal distance and max step size, not vertical chord spread.}
\]

The tower should preserve that rule. Since every rank projects to the same \(\lambda_0\) pedal/root line, the deadline projects trivially through the tower.

The reward contract should distinguish:

| Concept | Meaning |
| --- | --- |
| final step | this is the last rewardable step |
| terminal success | reward function declares success |
| truncation | episode stops because deadline/max-steps was reached |
| hard violation | reward function declares a forbidden behavior |

## New-Facts Contract

Stage 7 defined the "new facts only" reward principle. Stage 8 requires the context to expose those facts or enough information to compute them.

At rank \(k\), define:

\[
\mathsf{newFacts}_t^k
=
\left(
\mathsf{newVoice}_k,\;
\mathsf{newAction}_t^k,\;
\mathsf{newVerticalFacts}_t^k,\;
\mathsf{fullSonorityAllowed}_k
\right).
\]

For \(k=1\), all melodic facts are new.

For \(k=2\):

\[
\mathsf{newVoice}_2=1.
\]

For \(k\ge 3\):

\[
\mathsf{newVoice}_k=k-2.
\]

The new action coordinate is:

\[
\mathsf{newAction}_t^k=\Delta\lambda_{\mathsf{newVoice}_k,t}.
\]

The new vertical interval facts are:

\[
\mathsf{newVerticalFacts}_t^k
=
\left\{
(\lambda_{\mathsf{newVoice}_k,t}-\lambda_{i,t})\bmod 12
\;\middle|\;
i\ne \mathsf{newVoice}_k
\right\}.
\]

Full-sonority terms may intentionally look at all of \(s_t^k\). When they do, diagnostics should state that full-sonority scoring was used.

## Reward Output Contract

The old structured reward output is preserved:

\[
\mathcal O_t^k
=
\left(
r_t^k,\;
\mathsf{hardViolation},\;
\mathsf{terminalSuccess},\;
\mathsf{diagnostics}
\right).
\]

| Field | Meaning |
| --- | --- |
| reward | scalar reward |
| hard violation | transition should be treated as rule violation |
| terminal success | goal/cadence/success condition is met |
| diagnostics | structured explanatory payload |

The output contract should not force every reward term to expose every diagnostic. It should require enough common fields that training traces can be understood.

## Required Diagnostics

Every rank-\(k\) reward result should be able to report:

| Diagnostic | Meaning |
| --- | --- |
| `rank` | active rank \(k\) |
| `kind` or `term_name` | reward term or composite reward type |
| `source` | \(s_t^k\) |
| `target` | \(s_{t+1}^k\) |
| `step_index` | \(t\) |
| `step_delta` | \(\Delta s_t^k\) |
| `parent_step_delta` | projected \(\Delta s_t^{k-1}\), if \(k>1\) |
| `new_voice_index` | coordinate introduced at this rank |
| `new_action` | action coordinate introduced at this rank |
| `new_facts_count` | number of new facts inspected |
| `inherited_facts_ignored` | whether parent facts were deliberately skipped |
| `full_sonority_used` | whether a full-sonority term intentionally used all voices |
| `measure_size` | meter size |
| `bar_position` | current bar position |
| `is_downbeat` | downbeat/upbeat flag |
| `is_ending_beat` | final beat of measure |
| `key_pitch_class` | \(\tau\bmod 12\) |
| `target_root_octave` | goal octave when present |
| `deadline_active` | whether goal rewards are shut off after deadline |
| `is_final_step` | whether terminal rewards may fire |

## Old-System Carryovers

The following decisions should be carried forward unchanged unless later contradicted:

| Decision | Carry forward |
| --- | --- |
| reward functions are called with source, target, context | yes |
| context is separate from environment object | yes |
| actions are step deltas | yes |
| history is available in raw form | yes |
| timed windows are padded fixed-length views of history | yes |
| padding is identified by a validity mask | yes |
| measure size is part of context | yes |
| downbeat can be derived from step parity | yes |
| key pitch class is part of context | yes |
| target root octave is part of context | yes |
| final-step flag is part of context | yes |
| reward output has reward, hard violation, terminal success, diagnostics | yes |

## Tower-Only Additions

The tower requires these additions:

| Addition | Why needed |
| --- | --- |
| rank \(k\) | rewards are rank-local |
| projected parent context | lower-rank scaffold is explicit |
| projected parent action | current action extends parent action |
| new voice index | identifies the current extension coordinate |
| new action coordinate | policy at rank \(k\) chooses one new coordinate |
| new vertical facts | Stage 7 rewards score only new facts by default |
| inherited-facts diagnostic | prevents accidental double scoring |
| full-sonority diagnostic | records intentional exceptions to new-facts-only scoring |

## Unresolved Choices

These do not block Stage 8, but they should be resolved before implementation:

| Choice | Notes |
| --- | --- |
| exact context class names | implementation-stage decision |
| whether context stores parent context directly or computes projection on demand | implementation-stage decision |
| whether full tonic MIDI value is always carried | needed only if rewards use absolute tonic register |
| whether beat strength becomes richer than downbeat/upbeat | Stage 7 only requires downbeat/upbeat |
| default context window length | old default is three measures |
| exact diagnostics schema enforcement | implementation-stage decision |

## Stage 8 Completion Checklist

Stage 8 can be considered complete if the project manager accepts:

| Decision | Proposed answer |
| --- | --- |
| Does the tower preserve the old source-target-context reward call shape? | yes |
| Does \(W_t^k\) project to \(W_t^{k-1}\)? | yes |
| Does \(\Delta s_t^k\) project to \(\Delta s_t^{k-1}\)? | yes |
| Are meter, tonic, goal, and deadline shared through projection? | yes |
| Is raw history still available? | yes |
| Is a padded fixed-length window still available? | yes |
| Does context expose rank and new-facts bookkeeping? | yes |
| Does reward output preserve reward, hard violation, terminal success, diagnostics? | yes |

Once accepted, Stage 9 should define success/failure semantics: terminal success, truncation, hard violation, and invalid extension behavior for each rank.
