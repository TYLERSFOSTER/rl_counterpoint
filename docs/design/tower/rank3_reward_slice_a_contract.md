# Rank-3 Reward Slice A Contract

This document defines the first concrete shaping-reward bundle for rank 3 in
`tower`.

The accepted design constraints are:

- reward ownership should be **global-triad**
- the inserted interior voice has **no goal octave/register**
- the pedal octave remains part of terminal success, not a general interior-voice
  shaping target

So rank-3 reward slice A should primarily score:

1. whole-triad sonority quality
2. whole-triad spacing quality
3. cadence-nearness at the triad level

It should **not** introduce a new target-octave reward for the inserted interior
voice.

## Composition

The rank-3 reward should be a sum:

\[
R^{(3)}(c) = R_{\mathrm{succ}}^{(3)}(c)
          + R_{\mathrm{triad}}(c)
          + R_{\mathrm{spacing}}(c)
          + R_{\mathrm{cad-end}}(c).
\]

The first implementation does not need more than this.

## 1. Terminal Success Term

Use the dedicated rank-3 success predicate from:

- [rank3_success_contract.md](./rank3_success_contract.md)

with the same success/failure term pattern already used elsewhere:

\[
R_{\mathrm{succ}}^{(3)}(c)=
\begin{cases}
r_{\mathrm{succ}}, & S_3(c)=1 \\
r_{\mathrm{fail}}, & S_3(c)=0
\end{cases}
\]

As with rank 1 and rank 2, this term is evaluated every step, but usually
contributes the failure value, which may be zero.

## 2. Global-Triad Consonance Term

Let the realized target state be

\[
s'=(\lambda_0,\lambda_1,\lambda_2).
\]

Define the three pairwise intervals:

\[
I_{01} = \lambda_1-\lambda_0,\qquad
I_{12} = \lambda_2-\lambda_1,\qquad
I_{02} = \lambda_2-\lambda_0.
\]

The allowed interval-class set remains

\[
\mathcal C=\{3,4,7,8,9\}\pmod{12}.
\]

Because graph legality already hard-prunes states outside this family, the
reward term should not pretend to be the primary dissonance enforcer. Instead,
it should rank **legal** triads by the quality of their interval content.

The first concrete form should be:

\[
R_{\mathrm{triad}}(c)
=
w_{01} C(I_{01}\bmod 12)
+ w_{12} C(I_{12}\bmod 12)
+ w_{02} C(I_{02}\bmod 12)
\]

where \(C(\cdot)\) is the same consonance score table already used in lower-tier
harmonic rewards.

The default first-slice choice should be:

\[
w_{01}=w_{12}=w_{02}=1.
\]

This is simple, global, and consistent with the accepted reward-ownership rule.

## 3. Global Spacing Term

Rank 3 has two adjacent gaps and one total span, so spacing should also be scored
globally rather than only from the inserted voice’s perspective.

Let:

\[
g_{01} = \lambda_1-\lambda_0,\qquad
g_{12} = \lambda_2-\lambda_1,\qquad
g_{02} = \lambda_2-\lambda_0.
\]

The first slice should reward legal, not-overly-compressed triads:

\[
R_{\mathrm{spacing}}(c)
=
\sigma(g_{01}) + \sigma(g_{12}) + \tau(g_{02})
\]

where:

- \(\sigma\) scores adjacent spacing,
- \(\tau\) scores total span.

The first implementation should keep this simple:

\[
\sigma(g)=
\begin{cases}
r_{\mathrm{adj}}, & g \ge g_{\min}^{\mathrm{adj}} \\
\pi_{\mathrm{adj}}, & g < g_{\min}^{\mathrm{adj}}
\end{cases}
\]

\[
\tau(g)=
\begin{cases}
r_{\mathrm{span}}, & g \le W_3 \\
\pi_{\mathrm{span}}, & g > W_3
\end{cases}
\]

with \(W_3\) matching the graph contract’s width cap.

This gives us a small positive signal for comfortable legal spacing without
inventing a complicated register objective too early.

## 4. Cadence-Endpoint Triad Shaping Term

Rank-3 success will still be sparse, so the first bundle should include a final-
step cadence-endpoint shaping term that scores closeness to the intended
dominant and tonic triads.

At final-step contexts only:

\[
R_{\mathrm{cad-end}}(c)
=
w_{\mathrm{prev}} D^{-}(c)
+ w_{\mathrm{final}} D^{+}(c)
\]

where:

- \(D^{-}(c)\) scores closeness of the penultimate triad to the dominant triad
  \(\{k+7, k+11, k+2\}\)
- \(D^{+}(c)\) scores closeness of the final triad to the tonic triad
  \(\{k, k+4, k+7\}\)

The first simple form should be inverse-distance over pitch classes:

\[
D^{+}(c)=\sum_{i=0}^{2}\frac{1}{|\delta_i^{+}|+1}
\]

and analogously for \(D^{-}(c)\), after matching each voice to its intended
cadential pitch-class target.

This is the rank-3 analog of the child-local cadence-endpoint shaping we already
introduced for rank 2, but now elevated to the whole triad.

## What Slice A Intentionally Does Not Include

Not yet:

- no target-octave reward for the inserted interior voice
- no separate melodic reward for the interior voice alone
- no beat-class pitch reward for the interior voice
- no elaborate inversion-preference table
- no staging-specific reward toggles

Those may all become useful later, but slice A should stay compact.

## Implementation Target

The first implementation should add:

- rank-3 reward config in [tower/reward/factory.py](../../../tower/reward/factory.py)
- rank-3 harmonic terms in [tower/reward/harmony.py](../../../tower/reward/harmony.py)
- rank-3 success predicate hook in [tower/reward/success.py](../../../tower/reward/success.py)

The key discipline is:

- graph legality handles structural impossibility
- reward slice A ranks the legal rank-3 world

That keeps rank 3 aligned with what we learned the hard way in rank 2.
