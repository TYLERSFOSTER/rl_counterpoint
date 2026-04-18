# Rank-Local Reward Specification

This document is the Phase 3 / Stage 7 deliverable for the tower redesign.

The purpose is to turn the musical and computational rule notes in `assets/rules/tc21m_rules.md` into a clean mathematical reward specification for the tower model. This is still a design document, not an implementation plan.

The tower model, including the rank-local "new facts only" reward principle, is the project manager's design. The central idea is that voiceleading builds by rank: a higher-rank voiceleading problem extends a lower-rank one rather than replacing it. Therefore rewards at rank \(k\) should evaluate the new musical facts introduced at rank \(k\), while the inherited scaffold remains the responsibility of earlier trained ranks.

## Stage Location

This document belongs to:

| Plan level | Current location |
| --- | --- |
| Phase | Phase 3: Freeze Reward Ownership |
| Stage | Stage 7: Define rank-local reward semantics |
| Action | Translate musical rule prose into computational reward/pruning terms |

Stage 7 exit criterion:

| Requirement | Status |
| --- | --- |
| Specify what \(R^1\) is responsible for | Drafted here |
| Specify what \(R^2\) is responsible for | Drafted here |
| Specify what \(R^3\) is responsible for | Drafted here |
| Separate inherited scaffold from newly scored extension | Drafted here |
| Identify reward vs pruning vs terminal semantics | Drafted here |
| List unresolved reward choices | Drafted here |

## Source Notes

The source rule notes are in `assets/rules/tc21m_rules.md`.

Those notes are intentionally closer to musical prose. This document reorganizes them into the reward vocabulary that the tower model needs:

| Source concept | Spec concept |
| --- | --- |
| consonance/dissonance notes | interval-class and pitch-class reward terms |
| accented/unaccented | downbeat/upbeat |
| melodic rules | rank-local horizontal action/window terms |
| vertical interval rules | newly introduced vertical-fact terms |
| chord progression rules | tonic-relative template detectors |
| cadence rules | terminal or near-terminal template rewards |
| forbidden voiceleading | graph pruning, not reward |

## Mathematical Setting

For each rank \(k\), a state is a chord

\[
s^k_t=(\lambda_0,\dots,\lambda_{k-1})\in\{0,\dots,127\}^k
\]

with

\[
\lambda_0<\cdots<\lambda_{k-1}.
\]

An action is a move vector

\[
\Delta s^k_t=(\Delta\lambda_0,\dots,\Delta\lambda_{k-1})\in\mathbb Z^k.
\]

The realized next state is

\[
s^k_{t+1}=s^k_t+\Delta s^k_t
\]

whenever this transition is graph-legal.

The reward at rank \(k\) is written

\[
R_k(W^k_t,\Delta s^k_t;\Theta_k),
\]

where \(W^k_t\) is the rank-\(k\) window up to time \(t\), \(\Delta s^k_t\) is the proposed or realized action at time \(t\), and \(\Theta_k\) is the collection of rank-\(k\) reward hyperparameters.

## Rank-Local Principle

The defining reward principle is:

\[
R_k \text{ scores rank-}k\text{ musical facts not already owned by }R_{k-1}.
\]

This is not only an engineering convenience. It is part of the project manager's mathematical design. The tower is useful because training does not repeatedly search and score the entire full-chord space at every rank. Instead, each tier learns the extension problem over an inherited scaffold.

Concretely:

| Rank | Inherited scaffold | New reward responsibility |
| --- | --- | --- |
| \(G(1)\) | none | one-line melody and horizontal motion |
| \(G(2)\) | \(G(1)\) pedal/melody | outer voice extension, dyadic vertical facts, two-voice relational motion |
| \(G(3)\) | projected \(G(2)\) scaffold | newly inserted voice, new vertical facts involving that voice, three-voice sonority facts |
| \(G(4)\) | projected \(G(3)\) scaffold | newly inserted voice, new vertical facts involving that voice, four-voice sonority facts |
| \(G(n)\), \(n>4\) | projected \(G(n-1)\) scaffold | repeat the \(G(4)\)-style extension pattern |

## New Facts

The phrase "new facts" means the additional musical facts made available when passing from \(G(k-1)\) to \(G(k)\).

For \(k=1\), all facts are new:

\[
\mathcal N_1(W^1_t)=\text{one-line melodic facts in }W^1_t.
\]

For \(k=2\), the new voice is the outer/top voice over the pedal:

\[
s^2_t=(\lambda_0,\lambda_1).
\]

The new facts include:

| Fact type | Example |
| --- | --- |
| new horizontal fact | motion of \(\lambda_1\) |
| new vertical fact | \(\lambda_1-\lambda_0\) |
| new relational motion fact | expansion/contraction, contrary/similar/parallel motion |
| new cadence fact | dyadic dominant-to-tonic or perfect-cadence evidence |

For \(k\ge 3\), the projection removes the second-from-top coordinate:

\[
\operatorname{pr}^k(\lambda_0,\dots,\lambda_{k-3},\lambda_{k-2},\lambda_{k-1})
=
(\lambda_0,\dots,\lambda_{k-3},\lambda_{k-1}).
\]

Thus the newly introduced voice at rank \(k\) is

\[
\lambda_{\mathrm{new}}=\lambda_{k-2}.
\]

The new vertical facts are the intervals involving \(\lambda_{\mathrm{new}}\):

\[
\mathcal V^{\mathrm{new}}_k(s)
=
\{(\lambda_{\mathrm{new}}-\lambda_i)\bmod 12\mid i\ne k-2\}.
\]

The inherited outer interval \(\lambda_{k-1}-\lambda_0\) is not automatically rescored at \(G(k)\), because it was already present in the projected scaffold.

Full-sonority facts are allowed when the rule genuinely requires the whole rank-\(k\) chord. Examples include chord-factor occupancy, omission of root or third, seventh-chord evidence, and doubling the leading tone.

## Reward Decomposition

The rank-\(k\) reward is decomposed as

\[
R_k
=
w_{\mathrm{mel}}R^{\mathrm{mel}}_k
+w_{\mathrm{vert}}R^{\mathrm{vert}}_k
+w_{\mathrm{rel}}R^{\mathrm{rel}}_k
+w_{\mathrm{harm}}R^{\mathrm{harm}}_k
+w_{\mathrm{cad}}R^{\mathrm{cad}}_k
+w_{\mathrm{goal}}R^{\mathrm{goal}}_k.
\]

Some terms are dense step rewards. Some inspect a window. Some inspect the proposed next transition. Some pay only at the terminal boundary.

| Term | Reads | Timing |
| --- | --- | --- |
| \(R^{\mathrm{mel}}_k\) | horizontal motion and recent melody window | step or window |
| \(R^{\mathrm{vert}}_k\) | new vertical intervals | step or window |
| \(R^{\mathrm{rel}}_k\) | relative motion among voices | step or window |
| \(R^{\mathrm{harm}}_k\) | chord-factor and progression templates | window |
| \(R^{\mathrm{cad}}_k\) | cadence templates and meter position | terminal or near-terminal |
| \(R^{\mathrm{goal}}_k\) | passage goal condition | terminal or deadline-aware |

Note: \(R^{\mathrm{goal}}_k\) preserves the old system's idea that the agent is given a starting chord and an ending octave or target region. This document focuses on the musical rule rewards, but goal-pressure remains part of the total reward.

## Pitch-Class Normalization

All harmonic and scale-degree reward terms use the same normalization pattern already used in the old system:

\[
\operatorname{pc}_{\tau}(\lambda)
=
(\lambda-\tau)\bmod 12.
\]

Vertical interval classes use

\[
\operatorname{int}(\lambda_i,\lambda_j)
=
(\lambda_j-\lambda_i)\bmod 12.
\]

This lets reward terms detect tonic-relative facts independently of absolute register.

## Beat Vocabulary

The reward vocabulary should use downbeat/upbeat, not accented/unaccented.

Let

\[
\operatorname{beat}(t)\in\{\mathrm{downbeat},\mathrm{upbeat}\}
\]

or later, if needed,

\[
\operatorname{strength}(t)\in\mathbb R.
\]

For Stage 7, downbeat/upbeat is enough.

## Reward-Term Inventory

### Rank 1: \(R_1\)

\(R_1\) owns one-line melody and horizontal motion.

| Term | Type | Reads | Sketch |
| --- | --- | --- | --- |
| horizontal consonance distribution | reward | \(W^1_t\) | match consonant/dissonant action distribution to target ratio |
| sparse dissonant leaps | reward | \(W^1_t,\Delta s^1_t\) | disfavor overuse of dissonant horizontal intervals |
| switchback after large leap | reward | previous action and current action | if \(|\Delta s^1_{t-1}|\) is large, reward opposite direction |
| diminished/augmented resolution | reward | previous action and current action | if previous interval is unstable, reward stepwise resolution |
| recent range bound | reward or penalty | \(W^1_t\) | reward recent melody range \(\le 12\) semitones |
| avoid repeated fourths/fifths | reward or penalty | recent actions | penalize consecutive equal \(5\) or \(7\) semitone moves |
| large leap followed by small step | reward | previous and current action | if large previous leap, reward current small step |
| avoid step-then-opposite-leap | penalty | previous and current action | penalize step followed by large opposite-direction leap |
| tonic emphasis | reward | spectral density of \(W^1_t\) | reward pitch-class density near tonic targets |
| structural pitch placement | reward | pitch classes and beat position | reward structural scale degrees on stronger beats |
| leading-tone resolution | reward | window until resolution | reward reciprocal time-to-resolution |
| unstable pitches on downbeats | penalty | pitch classes and beat position | penalize unstable classes on downbeats, especially first beat |
| terminal V-I gesture | terminal reward | final window | reward V-I cadence at passage end |

### Rank 2: \(R_2\)

\(R_2\) owns the first extension over the melody/pedal: dyadic vertical facts, outer interval behavior, and two-voice relational motion.

| Term | Type | Reads | Sketch |
| --- | --- | --- | --- |
| expansion/contraction variety | reward | \(W^2_t\) | distribute \(\operatorname{sgn}\Delta r_t\) across expand/parallel/contract |
| parallel/similar/contrary variety | reward | \(W^2_t\) | distribute relative-motion classes |
| oblique motion avoidance | penalty | transition | penalize one voice stationary while another moves |
| repetition avoidance | reward or penalty | \(W^2_t\) | use autocorrelation/convolution to detect immediate repeated notes/chords |
| consonant outer interval | reward or pruning candidate | current dyad | check \(\lambda_1-\lambda_0\) in consonance set |
| structural vertical loci | reward | phrase-position masks | reward tonic P8 and dominant P5/P8 in phrase positions |
| imperfect consonance flow | reward | \(W^2_t\) | reward density of 3rds and 6ths between structural loci |
| excessive parallel 3rds/6ths | penalty | \(W^2_t\) | penalize runs longer than \(H_{\mathrm{parallel36}}\) |
| parallel perfect intervals | pruning | edge | forbid parallel/similar perfect 5ths and octaves |
| common progressions | reward | chord-template window | reward tonic-relative progression templates |
| retrogressions | penalty or pruning candidate | chord-template window | penalize retrogression templates |
| retrogression near cadence | penalty | chord-template window and goal distance | increase penalty near cadence |
| vertical 3rd density | reward | \(W^2_t\) | reward sufficient density of 3rds |
| terminal cadence | terminal reward | final \(L_{\mathrm{cad}}\) steps | reward final cadence template |
| perfect cadence | terminal reward | final \(G(2)\) window | reward dominant-to-tonic perfect cadence |

### Rank 3: \(R_3\)

\(R_3\) owns the insertion of the first interior voice. It should not rescore the inherited outer dyad from \(G(2)\) except when a full-sonority rule explicitly requires the entire three-voice chord.

| Term | Type | Reads | Sketch |
| --- | --- | --- | --- |
| upbeat dissonance preference | reward/penalty | new vertical facts and beat | reward new dissonances on upbeats, penalize on downbeats |
| downbeat dissonance resolution | transition reward | \(W^3_t,\Delta s^3_t\) | downbeat dissonance must resolve to consonance |
| new consonance ratio | reward | new vertical facts in \(W^3_t\) | reward ratio above \(q_{\mathrm{cons}}\) |
| pedal downbeat/upbeat repetition | optional reward | pedal action and beat pair | small style reward for \(\Delta\lambda_0=0\) across beat pair |
| chord-tone/non-chord-tone balance | reward | newly added note and chord-template evidence | match target ratio |
| stepwise dissonance handling | reward | new voice actions | dissonance approached and left stepwise |
| upbeat passing tones | reward | new voice three-note pattern | reward stepwise same-direction upbeat dissonance |
| downbeat non-harmonic resolution | reward/penalty | next event | require next event to be chord tone |
| leaps only between chord tones | penalty or pruning candidate | new voice source/target | penalize leaps involving non-chord tones |
| chordal leap exception | gating rule | new voice and vertical facts | no leap penalty if endpoints are chord tones and vertical facts consonant |
| root/third presence | reward | full sonority | reward root and third occupancy |
| triad implication | reward | full sonority | reward \(\{R,3,5\}\) or \(\{R,R,3\}\) occupancy |
| non-harmonic support | reward | full sonority | if non-harmonic tone present, reward root and third support |
| double leading tone | pruning | full sonority | forbid doubled leading tone |
| seventh-chord implication | reward | full sonority | reward root, third, seventh evidence |
| six-four classification | unresolved | full sonority/window | ignore for now |

### Rank 4 And Higher

\(R_4\) repeats the same extension logic for a four-voice sonority. For \(k>4\), the system should repeat the \(G(4)\)-style pattern unless a later design stage introduces special high-rank behavior.

| Term family | Ownership |
| --- | --- |
| new vertical facts | intervals involving the newly inserted voice |
| full-sonority occupancy | chord-factor rules needing the whole rank-\(k\) chord |
| non-harmonic handling | newly inserted voice and its relation to the sonority |
| cadence/progression evidence | full template only when rank-\(k\) information is genuinely needed |

## Pruning Versus Reward

Some musical rules should not be ordinary reward terms because they define legal graph structure.

| Rule | Current classification |
| --- | --- |
| no voice crossing | pruning |
| no parallel fifths | pruning |
| no parallel perfect octaves | pruning candidate |
| no doubled leading tone | pruning |
| \(\mathrm{V}\to\mathrm{IV}\) or \(\mathrm{V}\to\mathrm{iv}\) | pruning candidate or large penalty |
| chord width and spacing constraints | pruning |
| rank projection compatibility | pruning / graph morphism law |

The design rule is:

\[
\text{If a transition should never be explored, make it pruning.}
\]

\[
\text{If a transition is legal but stylistically worse, make it reward or penalty.}
\]

## Hyperparameters

The following knobs are implied by the current reward design.

| Symbol | Meaning |
| --- | --- |
| \(w_{\mathrm{mel}}\) | melody reward weight |
| \(w_{\mathrm{vert}}\) | vertical interval reward weight |
| \(w_{\mathrm{rel}}\) | relative-motion reward weight |
| \(w_{\mathrm{harm}}\) | harmonic template reward weight |
| \(w_{\mathrm{cad}}\) | cadence reward weight |
| \(w_{\mathrm{goal}}\) | passage-goal reward weight |
| \(q_{\mathrm{cons}}\) | minimum desired consonant ratio |
| \(S_{\mathrm{step}}\) | maximum size of a stepwise move |
| \(H_{\mathrm{parallel36}}\) | tolerated run length for parallel 3rds/6ths |
| \(L_{\mathrm{cad}}\) | cadence detector window length |
| \(r_{\mathrm{chord}}\) | target chord-tone/non-chord-tone ratio |
| \(p_{\mathrm{downbeat\_diss}}\) | tolerance or penalty strength for downbeat dissonance |
| \(w_{\mathrm{retro\_cad}}\) | extra retrogression penalty near cadence |

Defaults are not fixed in Stage 7. Stage 8 or Stage 10 should decide which defaults are needed before implementation.

## Diagnostics

Every reward term should eventually expose diagnostics. This is not yet the implementation contract, but the design expectation is:

| Diagnostic | Meaning |
| --- | --- |
| `rank` | active rank \(k\) |
| `term_name` | reward component name |
| `new_facts_count` | number of new facts considered |
| `inherited_facts_ignored` | whether inherited facts were intentionally skipped |
| `beat_role` | downbeat/upbeat when relevant |
| `template_name` | progression/cadence template when relevant |
| `is_terminal_term` | whether term pays only at terminal boundary |
| `is_pruning_candidate` | whether term may migrate into graph legality |

## Unresolved Choices

The following are deliberately not finalized here.

| Choice | Why unresolved |
| --- | --- |
| exact consonance pitch-class set | source notes still mark uncertainty |
| exact structural pitch set | source notes suggest \(\{0,5,7\}\), not finalized |
| exact chord-template vocabulary | needs a compact representation before implementation |
| exact cadence-template vocabulary | depends on final goal/deadline semantics |
| whether parallel octaves are hard pruning | likely yes, but old system explicitly handled fifths |
| whether \(\mathrm{V}\to\mathrm{IV}\) is pruning or large penalty | design boundary still open |
| six-four chord handling | explicitly ignored for now |
| default hyperparameter values | belongs to later training protocol/design |

## Stage 7 Completion Checklist

Stage 7 can be considered complete when the project manager accepts the following decisions:

| Decision | Proposed answer |
| --- | --- |
| Is reward rank-local? | yes |
| Are inherited facts rescored? | no, unless a full-sonority template explicitly requires them |
| Is downbeat/upbeat the meter vocabulary? | yes |
| Are hard illegality rules pruned from the graph? | yes |
| Are progression/cadence rewards template detectors? | yes |
| Is \(R_1\) melody-owned? | yes |
| Is \(R_2\) dyad/relation-owned? | yes |
| Is \(R_3\) inserted-voice/triad-owned? | yes |
| Does \(R_4\) repeat the higher-rank insertion pattern? | yes |

Once accepted, Stage 8 should define the exact reward context contracts: what object is passed to \(R_k\), what fields exist on \(W^k_t\), how meter/goal/deadline information is represented, and what diagnostics each reward term must return.
