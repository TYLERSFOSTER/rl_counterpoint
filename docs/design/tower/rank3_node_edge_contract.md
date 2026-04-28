# Rank-3 Node And Edge Contract

This document is the explicit graph contract for the first operational rank-3
slice in `tower`.

It turns the already accepted rank-3 design choices into an implementation target
for:

- [tower/graph/legality.py](/Users/foster/rl_counterpoint/tower/graph/legality.py)
- [tower/graph/induced.py](/Users/foster/rl_counterpoint/tower/graph/induced.py)
- [tower/train/runner.py](/Users/foster/rl_counterpoint/tower/train/runner.py)

The intended use is:

- code should implement this contract directly,
- tests should assert this contract directly,
- later reward work should assume this contract already holds.

## Scope

This is a **graph** contract, not a reward contract.

It defines:

1. how the rank-3 graph is built,
2. what rank-3 node legality means,
3. what rank-3 edge legality means,
4. how rank-3 induces lower-tier graphs.

It does **not** define:

- rank-3 reward terms,
- rank-3 terminal cadence success in full detail,
- rank-3 training script UX.

Those belong in companion docs.

## Accepted Structural Premises

The following are already fixed.

### 1. Rank 3 Inserts An Interior Voice

Rank 3 lifts a rank-2 outer scaffold

\[
(\lambda_0,\lambda_2)
\]

to a three-voice state

\[
(\lambda_0,\lambda_1,\lambda_2)
\]

by inserting the new coordinate in the interior position.

So the projection is:

\[
\operatorname{pr}^{3}(\lambda_0,\lambda_1,\lambda_2)=(\lambda_0,\lambda_2).
\]

Likewise, for actions:

\[
\operatorname{pr}^{3}(\Delta\lambda_0,\Delta\lambda_1,\Delta\lambda_2)
=
(\Delta\lambda_0,\Delta\lambda_2).
\]

### 2. Final-Rank Construction Rebuilds The Whole Tower

When the intended final rank is 3, graph construction is a whole-tower operation.

The conceptual pattern is:

1. build a candidate rank-3 graph,
2. prune it by rank-3 legality,
3. induce rank 2 from it by projection image,
4. induce rank 1 from that lower projection.

Symbolically:

\[
G(3)_\bullet \rightsquigarrow G(3)_\bullet^{\mathrm{legal}}
\]

\[
G(2)_\bullet \leftarrow \operatorname{pr}^{3}\!\left(G(3)_\bullet^{\mathrm{legal}}\right)
\]

\[
G(1)_\bullet \leftarrow \operatorname{pr}^{2}\!\left(G(2)_\bullet\right).
\]

Equivalently, the owner’s intended phrasing is that rank-3 construction should be
understood as replacing the initial graph by the appropriate preimage over the
lower tower and then projecting downward to replace lower tiers.

The implementation implication is:

- rank-3 workflows should eventually consume lower-tier induced artifacts derived
  from rank 3, rather than treating lower tiers as independent fixed graphs.

## Rank-3 Graph Vocabulary

Let

\[
s=(\lambda_0,\lambda_1,\lambda_2)
\]

be a rank-3 state with strict ordering

\[
\lambda_0<\lambda_1<\lambda_2.
\]

Define the adjacent intervals:

\[
I_{01}=\lambda_1-\lambda_0,\qquad I_{12}=\lambda_2-\lambda_1.
\]

Define the outer interval:

\[
I_{02}=\lambda_2-\lambda_0.
\]

Allowed interval classes are:

\[
\mathcal C=\{3,4,7,8,9\}\pmod{12}.
\]

This same interval-class vocabulary is used for:

- lower adjacent interval,
- upper adjacent interval,
- outer interval,

in the first rank-3 slice.

## Rank-3 Node Contract: \(G(3)_0\)

A rank-3 state belongs to \(G(3)_0\) iff **all** of the following hold.

### 1. Correct rank

\[
|s|=3.
\]

### 2. Pitch-range legality

Every coordinate lies in the configured MIDI range:

\[
\lambda_i \in [\texttt{pitch\_min},\texttt{pitch\_max}]
\quad\text{for }i=0,1,2.
\]

### 3. Strict ordering

\[
\lambda_0<\lambda_1<\lambda_2.
\]

### 4. Lower voice valid as rank 1

The projected pedal voice must be a valid rank-1 state under the current rank-1
legality contract:

\[
(\lambda_0)\in G(1)_0.
\]

This means rank-3 legality inherits all currently active rank-1 tonic-relative
constraints on the lower voice.

### 5. Projected outer pair valid as rank 2

The outer scaffold must already be a valid rank-2 state under the current rank-2
legality contract:

\[
(\lambda_0,\lambda_2)\in G(2)_0.
\]

This keeps rank 3 honest with respect to the existing lower tier.

### 6. Lower adjacent interval consonant

\[
I_{01}\bmod 12 \in \mathcal C.
\]

### 7. Upper adjacent interval consonant

\[
I_{12}\bmod 12 \in \mathcal C.
\]

### 8. Outer interval consonant

\[
I_{02}\bmod 12 \in \mathcal C.
\]

### 9. Width bounded

The total span must satisfy the chosen rank-3 width bound:

\[
I_{02}\le W_3
\]

for a configured width cap \(W_3\).

The first implementation does not need a new elaborate width formula. A direct
rank-3 cap in the style already used for lower tiers is acceptable.

For the first concrete slice, the implementation sets

\[
W_3 = 15.
\]

## Consequence Of The Node Contract

Rank 3 is intentionally strict.

It is **not** enough for a state merely to:

- be ordered,
- fit in pitch range,
- and contain consonant outer voices.

The inserted interior voice must also produce a fully legal triad shape:

- valid lower voice,
- valid outer scaffold,
- consonant lower adjacency,
- consonant upper adjacency,
- consonant outer span.

This is a deliberate hard-pruning choice, not a reward choice.

## Rank-3 Edge Contract: \(G(3)_1\)

Let

\[
s=(\lambda_0,\lambda_1,\lambda_2),\qquad
t=(\mu_0,\mu_1,\mu_2)
\]

and let the action be

\[
a=(\Delta\lambda_0,\Delta\lambda_1,\Delta\lambda_2)
\]

with

\[
t=s+a.
\]

An edge belongs to \(G(3)_1\) iff **all** of the following hold.

### 1. Source legal

\[
s\in G(3)_0.
\]

### 2. Target legal

\[
t\in G(3)_0.
\]

### 3. No self-loop

\[
t\neq s.
\]

### 4. Action rank matches state rank

\[
|a|=3.
\]

### 5. No stationary voices

Every coordinate must move:

\[
\Delta\lambda_i\neq 0
\quad\text{for }i=0,1,2.
\]

### 6. No voice crossing

Adjacent voice order must be preserved across the transition. Equivalently:

\[
\mu_0<\mu_1<\mu_2
\]

and no source-target crossing pattern is permitted.

Operationally, this should be checked in the same "no crossing through each other"
style already used for lower tiers.

### 7. No parallel perfect fifths

For any pair \(i<j\), if the source interval is a perfect fifth and the pair moves
in parallel with equal displacement, the edge is illegal.

That is, forbid:

\[
\lambda_j-\lambda_i = 7
\]

and

\[
\mu_i-\lambda_i=\mu_j-\lambda_j.
\]

Apply this to all three pairs:

- \((0,1)\)
- \((1,2)\)
- \((0,2)\)

### 8. No parallel octaves

Likewise forbid parallel octaves for any voice pair:

\[
\lambda_j-\lambda_i \equiv 0 \pmod{12}
\]

with equal pairwise displacement.

Again, apply to all three pairs.

### 9. Motion bounded by `max_step_size`

The first implementation relies on the action lattice bound:

\[
|\Delta\lambda_i|\le \texttt{max\_step\_size}
\]

for each coordinate, as already enforced by the generated action space.

No separate legacy-style single-line interval predicate is required in the first
rank-3 slice.

### 10. Projected edge valid in rank 2

The projected outer-voice transition must be legal in rank 2:

\[
\operatorname{pr}^{3}(s)\to \operatorname{pr}^{3}(t)\in G(2)_1.
\]

Equivalently:

\[
(\lambda_0,\lambda_2)\to(\mu_0,\mu_2)
\]

must satisfy active rank-2 edge legality.

This is the graph-morphism requirement, not an optional consistency check.

## Consequence Of The Edge Contract

Rank-3 transitions are not merely "legal triads at both ends."

They also require:

- full lower-tier projection legality,
- no local freezing,
- no crossing,
- no parallel fifths,
- no parallel octaves.

This makes rank 3 a real structural extension of rank 2 rather than a loosely
connected extra voice.

## Induced Lower-Tier Artifacts From Rank 3

Rank-3 workflows should produce an induced rank-2 artifact analogous to the
existing induced rank-1-from-rank-2 artifact.

### Rank-3-Induced Rank-2 Node Image

\[
G(2)_0^{(3)}=\operatorname{pr}^{3}(G(3)_0).
\]

### Rank-3-Induced Rank-2 Edge Image

\[
G(2)_1^{(3)}=\operatorname{pr}^{3}(G(3)_1).
\]

This induced artifact should be:

- deterministic,
- cacheable,
- versioned by legality contract,
- inspectable on disk.

Then rank-1 may be induced from that rank-2 image as already planned.

## Implementation Rules

To keep code and contract aligned, the first implementation should follow these
rules.

### Rule 1

Do not implement rank-3 legality as a collection of ad hoc checks spread across:

- rollout,
- reward,
- scripts.

It belongs centrally in graph legality.

### Rule 2

Lower-tier legality should be reused, not copied by hand.

That means rank-3 legality should literally reuse:

- rank-1 validity for the lower voice,
- rank-2 validity for the projected outer pair.

### Rule 3

The induced rank-2 artifact should be additive at first.

Do not immediately force all existing rank-2 training workflows to consume it by
default. First get the artifact correct, then opt rank-3 workflows into the
top-down construction path.

### Rule 4

Graph legality should come before reward expansion.

If a musical relation is intended to be impossible, it should be graph-pruned
first rather than "rewarded away later."

## Test Matrix

The first implementation should add or update tests covering at least:

### Node legality

1. valid rank-3 state
2. invalid lower adjacent interval
3. invalid upper adjacent interval
4. invalid outer interval
5. invalid projected rank-1 lower voice
6. invalid projected rank-2 outer pair
7. excessive width

### Edge legality

1. valid rank-3 transition
2. self-loop rejection
3. stationary-voice rejection
4. voice-crossing rejection
5. parallel fifth rejection
6. parallel octave rejection
7. projected rank-2 edge rejection

### Induced graph artifact

1. rank-3 -> rank-2 node image
2. rank-3 -> rank-2 edge image
3. cache-key sensitivity to legality-contract changes

## Non-Goals Of This Document

This doc does **not** settle:

- exact rank-3 reward terms
- exact rank-3 terminal cadence implementation details
- script UX
- long-run hyperparameters

Those should be handled separately, after this contract is treated as authoritative.

## Immediate Follow-On Work

After accepting this contract, the next implementation step should be:

1. add rank-3 branches to [tower/graph/legality.py](/Users/foster/rl_counterpoint/tower/graph/legality.py)
2. generalize [tower/graph/induced.py](/Users/foster/rl_counterpoint/tower/graph/induced.py) to support rank-3 -> rank-2 artifacts
3. add graph tests before reward or runner work
