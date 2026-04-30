# Rank-3 Success Contract

This document defines the first concrete terminal-success contract for rank 3 in
`tower`.

It is intentionally narrow. The goal is not to encode every musically desirable
feature of three-voice cadence at once. The goal is to make the owner's accepted
success rule operational and testable.

## Accepted Owner Rule

Rank-3 success is:

1. pedal in goal octave
2. perfect cadence of triads in that octave

This should be interpreted as a **terminal** condition over the final transition,
not as a shaping preference distributed over the whole episode.

## Scope

This contract defines:

- what terminal rank-3 success means,
- which state coordinates it inspects,
- which pitch-class conditions are required,
- what octave condition is required.

It does **not** define:

- rank-3 shaping rewards,
- nonterminal cadence-endpoint shaping,
- stylistic refinements beyond the first concrete slice.

## State Vocabulary

Let the penultimate state be

$$
s^{-} = (\lambda_0^{-}, \lambda_1^{-}, \lambda_2^{-})
$$

and the final state be

$$
s^{+} = (\lambda_0^{+}, \lambda_1^{+}, \lambda_2^{+}).
$$

The pedal voice is the lower voice:

$$
\lambda_0.
$$

The final-rank-3 outer scaffold is:

$$
(\lambda_0,\lambda_2).
$$

The inserted interior voice is:

$$
\lambda_1.
$$

Let the tonic pitch class be $k$.

Let the target pedal octave be $o_\ast$.

## Terminal Preconditions

Rank-3 success can only fire when all of the following hold:

1. the episode step is terminal,
2. the metrical position is the cadence endpoint used by lower tiers,
3. the final state exists and is legal rank 3.

The first implementation should continue using the same terminal-step and
measure-end conventions already used for rank 1 and rank 2 success.

## Pedal-Octave Condition

The final pedal must be in the configured goal octave:

$$
\operatorname{oct}(\lambda_0^{+}) = o_\ast.
$$

This is the only octave-goal requirement in the first rank-3 slice.

There is **no** corresponding target-octave requirement on the inserted interior
voice.

## Cadential Outer-Voice Condition

The outer voices must realize the same perfect-cadence scaffold already expected
by the lower tower.

That means:

### Penultimate outer scaffold

$$
\operatorname{pc}(\lambda_0^{-}) = k + 7 \pmod{12}
$$

$$
\operatorname{pc}(\lambda_2^{-}) = k + 11 \pmod{12}
$$

### Final outer scaffold

$$
\operatorname{pc}(\lambda_0^{+}) = k \pmod{12}
$$

$$
\operatorname{pc}(\lambda_2^{+}) = k + 4 \pmod{12}
$$

So the lower tower still contributes:

- dominant pedal to tonic pedal
- leading-tone outer voice to tonic-third outer voice

## Interior-Voice Triad Condition

The inserted interior voice must complete the intended dominant-to-tonic triads.

For the first slice:

### Penultimate interior voice

$$
\operatorname{pc}(\lambda_1^{-}) = k + 2 \pmod{12}
$$

### Final interior voice

$$
\operatorname{pc}(\lambda_1^{+}) = k + 7 \pmod{12}
$$

This yields:

### Penultimate triad

$$
\{\operatorname{pc}(\lambda_0^{-}), \operatorname{pc}(\lambda_1^{-}), \operatorname{pc}(\lambda_2^{-})\}
=
\{k+7, k+11, k+2\}
$$

which is the dominant triad

$$
\{5,7,2\}
$$

relative to tonic.

### Final triad

$$
\{\operatorname{pc}(\lambda_0^{+}), \operatorname{pc}(\lambda_1^{+}), \operatorname{pc}(\lambda_2^{+})\}
=
\{k, k+4, k+7\}
$$

which is the tonic triad.

## Complete First-Slice Rank-3 Success Predicate

The first concrete rank-3 success predicate should return `True` iff all of the
following hold:

1. lower-tier terminal timing conditions hold
2. $\operatorname{oct}(\lambda_0^{+}) = o_\ast$
3. penultimate outer scaffold is dominant:
   - $\operatorname{pc}(\lambda_0^{-}) = k+7$
   - $\operatorname{pc}(\lambda_2^{-}) = k+11$
4. final outer scaffold is tonic:
   - $\operatorname{pc}(\lambda_0^{+}) = k$
   - $\operatorname{pc}(\lambda_2^{+}) = k+4$
5. penultimate interior voice is:
   - $\operatorname{pc}(\lambda_1^{-}) = k+2$
6. final interior voice is:
   - $\operatorname{pc}(\lambda_1^{+}) = k+7$

## Consequence

This is a strictly layered but genuinely triadic success rule.

It preserves lower-tier cadence structure while giving rank 3 a real local
terminal job:

- complete the dominant triad before cadence,
- complete the tonic triad at cadence,
- keep the pedal in the requested goal octave.

## Implementation Notes

The first implementation should live beside the existing success helpers, likely
in:

- [tower/reward/success.py](../../../tower/reward/success.py)

The implementation should expose a dedicated rank-3 predicate rather than
trying to shoehorn this into the rank-2 lifted cadence helper.
