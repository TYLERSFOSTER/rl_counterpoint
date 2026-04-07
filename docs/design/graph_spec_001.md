# Counterpoint Graph Spec 001

## Purpose

This document specifies the current design contract for the counterpoint graph $G(n)$ and the environment-level knobs that define it.

The key architectural point is that these knobs define the graph the agent traverses. They should not be scattered as unrelated arguments across graph code, environment code, scripts, and tests.

The likely future code home for this contract is a graph spec object, for example `CounterpointGraphSpec`, consumed by the environment rather than duplicated inside it.

## Environment Graph

Call the environment graph: $G(n)$ where: $n \in \mathbb{Z}_{\ge 1}$ The parameter $n$ is the number of voices. It should be treated as a basic defining parameter of the environment, likely provided through an environment constructor or through a graph spec object passed into the environment.

## Node And Edge Notation

Use: $G(n)_0$ for the node set, and: $G(n)_1$ for the edge set.

The source and target maps are: $\delta_0,\delta_1:G(n)_1\longrightarrow G(n)_0$ For an edge: $e:\lambda\to\mu$ with:

$$
\lambda=(\lambda_0,\dots,\lambda_{n-1}),\qquad
\mu=(\mu_0,\dots,\mu_{n-1}),
$$

the maps are: $\delta_0(e)=\lambda$ and $\delta_1(e)=\mu.$ Self-loops are always excluded: $\lambda\to\lambda\notin G(n)_1.$ ## Raw Node Set

Before trimming, nodes are ordered $n\)-tuples of distinct tones in the standard MIDI scale:

$$
(\lambda_0,\dots,\lambda_{n-1})\in\{0,\dots,127\}^n
$$

with: $\lambda_0<\lambda_1<\cdots<\lambda_{n-1}.$ This is a voiced chord state, not an unordered pitch set. Voice identity is persistent and ordered from low to high.

## Node Trims

The following trims affect $G(n)_0$ itself. They occur before edge counting.

### Vertical Interval Cap

For adjacent voices: $\lambda_{i+1}-\lambda_i\le M$ where $M$ is the max interval parameter. The current default is: $M=11.$ ### Adjacent Vertical Consonance

Adjacent vertical intervals must be consonant according to the current trim rule.

Forbidden adjacent vertical intervals: $\{1,2,6,10,11\}$ corresponding to:

- minor second,
- major second,
- tritone,
- minor seventh,
- major seventh.

So for all adjacent voices: $\lambda_{i+1}-\lambda_i\notin\{1,2,6,10,11\}.$ ### Total Chord Width Cap

The total chord width is capped by: $\lambda_{n-1}-\lambda_0\le \lceil 6n\rceil.$ Since $n$ is an integer here, this is currently equivalent to: $\lambda_{n-1}-\lambda_0\le 6n.$ The ceiling notation is retained because that was the design language used when the constraint was introduced.

### Tonic Root Scale Trim

Another defining parameter is the tonic note: $\tau$ where $\tau$ is a MIDI-note value. Only $\tau \bmod 12$ matters for the current trim.

The root note of every chord must satisfy: $\lambda_0-\tau\in\{3,4,5,7,8,9\}\pmod{12}.$ Combined with the previous vertical-interval trims, this is intended to amount to a "stay in scale" graph restriction.

### Outer Interval Consonance

The outer interval of every chord must also be consonant according to the same mod-12 consonance set:

$$
\lambda_{n-1}-\lambda_0\in\{3,4,5,7,8,9\}\pmod{12}.
$$

This is a node trim only. It affects $G(n)_0$ before edge counting and does not define a separate edge-trim option.

## Edge Trims

The following trims affect $G(n)_1\), after the node set $G(n)_0$ has already been trimmed.

### Voice Crossing

The graph construction should have a boolean parameter controlling whether voice crossing is allowed.

Suggested field:

```python
allow_voice_crossing: bool = False
```

An edge: $(\lambda_0,\dots,\lambda_{n-1})\to(\mu_0,\dots,\mu_{n-1})$ has voice crossing if there exists an adjacent voice boundary $0\le i<n-1$ such that: $\mu_i\ge \lambda_{i+1}$ or: $\lambda_i\ge \mu_{i+1}.$ So no crossing means: $\mu_i<\lambda_{i+1}$ and: $\lambda_i<\mu_{i+1}$ for every $0\le i<n-1\).

### Parallel Fifths

The graph construction should have a boolean parameter controlling whether parallel fifths are allowed.

Suggested field:

```python
allow_parallel_fifths: bool = False
```

An edge has a parallel fifth if for some pair of voices: $0\le i<j<n,$ we have: $\lambda_j-\lambda_i=7$ and: $\mu_j-\lambda_j=\mu_i-\lambda_i.$ This ranges over all voice pairs, not just adjacent voices.

Because the voices move by the same signed displacement, this also implies: $\mu_j-\mu_i=7.$ ### Single-Line Interval Cap

The graph construction should also have an integer parameter controlling the largest allowed single-line upward interval.

Suggested field:

```python
max_interval: int = 11
```

For every voice: $\mu_i-\lambda_i\le M.$ Important note: as currently written, this caps upward motion only. If the intended rule later becomes largest absolute melodic displacement, the condition should become: $|\mu_i-\lambda_i|\le M.$ That has not been adopted yet.

## Suggested Code Shape

The graph-defining parameters should likely live in a dedicated spec object rather than being duplicated directly across environment code and utility scripts.

A possible future code shape:

```python
@dataclass(frozen=True)
class CounterpointGraphSpec:
    n: int
    tonic: int
    pitch_min: int = 0
    pitch_max: int = 127
    max_interval: int = 11
    max_chord_width_factor: float = 6.0
    forbidden_adjacent_vertical_intervals: frozenset[int] = frozenset(
        {1, 2, 6, 10, 11}
    )
    allowed_root_intervals_mod_12: frozenset[int] = frozenset(
        {3, 4, 5, 7, 8, 9}
    )
    allowed_outer_intervals_mod_12: frozenset[int] = frozenset(
        {3, 4, 5, 7, 8, 9}
    )
    allow_voice_crossing: bool = False
    allow_parallel_fifths: bool = False
```

Then the environment could be constructed from the graph spec:

```python
env = CounterpointEnv(
    graph_spec=CounterpointGraphSpec(n=3, tonic=60)
)
```

rather than duplicating all parameters as loose environment constructor arguments.

## Current Architectural Principle

The graph spec defines $G(n)\). The environment should consume the spec; it should not be the only place where the graph is defined.

The sparsity-count design script should either consume this same spec object later or mirror it exactly until the code object exists.
