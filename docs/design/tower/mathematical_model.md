# Tower Mathematical Model

## Purpose

This document records the mathematical model of the tower-based counterpoint RL system as developed in design discussion.

It is meant to be complete enough that a future engineer can recover the formal picture without having to reconstruct it from scattered notes or implementation details.

This is a document about the **mathematical model**, not the Python implementation model.

So this document concerns:

- graphs $G(n)_\bullet$
- projections
- states
- actions
- policies
- rewards
- windows
- pruning conditions
- the hierarchical-search / training-speedup intuition

It does **not** concern:

- Python classes
- dataclasses
- module boundaries
- code APIs

Those belong to later software-design documents.

---

## Attribution

The mathematical model in this document is the project manager's design.

In particular, the following ideas are project-manager-originated and should be understood that way throughout:

- the nested graph tower across chord size
- the insistence that the cross-rank maps are graph morphisms, not merely node-set maps
- the observation that voiceleading builds by rank, so that 1-part voiceleading is genuinely part of 2-part voiceleading, 2-part is part of 3-part, and so on
- the asymmetry that states project downward while actions assemble upward
- the use of tiered policies that choose one new action coordinate at a time
- the use of tier-local rewards that assess genuinely $k$-part voiceleading
- the interpretation of training as a search process
- the HNSW-like / hierarchical-search intuition for why the tower should produce a major speedup
- the dimensionality claim that reducing the search tier by tier radically reduces effective search dimension when out-degree is the operative local complexity bound
- the fact that, in this voiceleading setting, the quotient/projection morphisms arise canonically from the problem rather than needing to be chosen randomly

This document is a technical writeup of that design.



---

## Summary Of The Model

The mathematical model is now:

- ***Hierarchical spaces*:** For each rank $n$, there is a graph $G(n)_\bullet$.
- ***States*:** A state is a chord $s=(\lambda_0,\dots,\lambda_{n-1})\in\{0,\dots,127\}^n$, where $\lambda_0<\dots<\lambda_{n-1}$, and where the set $\{0,1,\dots,126,127\}$ is just the set of MIDI numbers for the notes of the chromatic scale.
- ***Actions*:** An action is a move vector $\Delta s \in \mathbb{Z}^n$.
- ***Hierarchical projections*:** The projection maps are natural coordinate projections on both states and actions. They constitute graph morphisms, representing the fact that valid higher-rank voiceleadings project to valid lower-rank voiceleadings.
- ***Hierarchical states and actions*:** States and actions assemble upward because each tier introduces one new coordinate of the total state and action vectors.
- ***Hierarchical policies*:** The tier-$k$ policy and the tier-$k$ reward are both defined on the same tier-$k$ window object $W_t^k$.
- ***Hierarchical rewards*:** Each reward $R_k$ is genuinely $k$-part and may depend on recent rank-$k$ passage.

The whole tower is intended to function like a canonical hierarchical search structure for voiceleading, which is why a large training-speedup is expected.

We now explain all of this in more detail.


## State space / environment
The old flat system tries to learn directly on one large graph of full-chord states and full-chord moves. The tower model instead organizes the problem by rank:

$$
G(4)_{\bullet} \overset{\operatorname{pr}^4}{\longrightarrow} G(3)_{\bullet} \overset{\operatorname{pr}^3}{\longrightarrow} G(2)_{\bullet} \overset{\operatorname{pr}^2}{\longrightarrow} G(1)_{\bullet}
$$

For ranks above $4$, the projection pattern is intended to continue in the same way as at the top displayed stage. The guiding musical idea is:

- one-part voiceleading is part of two-part voiceleading
- two-part voiceleading is part of three-part voiceleading
- three-part voiceleading is part of four-part voiceleading

So higher-rank voiceleading should not be modeled as a wholly unrelated search problem. It should be modeled as an extension of lower-rank voiceleading.

### Graphs

For each $n \ge 1$, there is a graph $G(n)_\bullet$ with node set $G(n)_0$ and edge set $G(n)_1$. The nodes are valid $n$-voice chords. The edges are valid $n$-voice voiceleadings. This document does **not** fix every final musical pruning rule, but it does fix the formal structure into which those pruning rules must fit.

### States / graph nodes
A state at rank $n$ is an $n$-chord of MIDI note values:

$$
s = (\lambda_0,\dots,\lambda_{n-1})\;\in\;\{0,\dots,127\}^n.
$$

where $\lambda_0<\lambda_1<\cdots<\lambda_{n-1}$. It is an ordered $n$-tuple of MIDI chromatic scale numbers, and so we can interpretat is as

- $\lambda_0$ is the pedal / lowest voice
- $\lambda_{n-1}$ is the top voice
- intermediate coordinates $\lambda_k$, for $10\le k\le n-1$, are interior voices

The tuple is ordered from bottom to top. Node validity determines which such tuples actually lie in $G(n)_0$.

### Edges
A directed edge in $G(n)_\bullet$ is a valid voiceleading $s \xrightarrow{\Delta s} s'$ where:

- $s,s' \in G(n)_0$
- $s' = s + \Delta s$
- the transition satisfies the rank-$n$ edge-validity rules

So morphisms in these graphs are naturally represented by $n$-tuples of moves. This is why edge projections are natural once node projections are fixed.

### Projections on states
Because a rank-$n$ state is an $n$-tuple, the node sets carry natural coordinate projections. These are fixed as follows.

- **Projection $G(2)_0 \to G(1)_0$.** For $n=2$: $\operatorname{pr}^2(\lambda_0,\lambda_1) = (\lambda_0)$. So the 2-voice parent of a pedal-plus-top state is just the pedal.

- **Projection $G(n)_0 \to G(n-1)_0$ For $n \ge 3$.** For $n\ge 3$: $\operatorname{pr}^n(\lambda_0,\dots,\lambda_{n-3}\lambda_{n-2},\lambda_{n-1})=(\lambda_0,\dots,\lambda_{n-3},\lambda_{n-1})$. So the projection removes the second-from-top coordinate.

Equivalently:
- the pedal is always preserved
- the top voice is always preserved
- each stage removes the newest added interior voice

This exactly matches the intended musical growth pattern:
- $G(2)$ adds the top voice over the pedal
- $G(3)$ adds the first middle voice
- $G(4)$ adds the next middle voice
- and so on

Thus a "parent/child" structure between implicit "node classes" is encoded in the mathematical model within the projections themselves, instead of within the internal data structure of the state object.

### Projections On Edges
The graph projection is not merely a map on nodes. It is a graph morphism. That means:

- valid nodes project to valid nodes
- valid edges project to valid edges
- source and target are respected under projection

If $s \xrightarrow{\Delta s} s'$ is a valid edge in $G(n)_\bullet$, then one requires: $\operatorname{pr}^n(s) \xrightarrow{\operatorname{pr}^n(\Delta s)} \operatorname{pr}^n(s')$ to be a valid edge in $G(n-1)_\bullet$. Equivalently, projection commutes with the transition:

$$
\operatorname{pr}^n(s') = \operatorname{pr}^n(s + \Delta s) = \operatorname{pr}^n(s) + \operatorname{pr}^n(\Delta s).
$$

This is the central structural reason the tower can support hierarchical reuse in training. The project manager's key point is:

> Each valid voiceleading with one extra part present must remain a valid voiceleading after that extra part is removed.

That statement is exactly the graph-morphism requirement in plain language.

## Agent / policy
[...]

### Actions
An action at rank $n$ is a move vector: $\Delta s = (\Delta\lambda_0,\dots,\Delta\lambda_{n-1}) \in \mathbb{Z}^n$. The intended meaning is $s_{t+1} = s_t + \Delta s_t$ coordinatewise, whenever the resulting transition is valid. So the action is not some abstract label. It is literally the chordwise move vector. This is a central spect of the model.


### Projections On Actions
Because actions are also coordinate tuples, they carry the same natural projections.

- **Projection $ \mathbb{Z}^2 \to \mathbb{Z}^1 $.** $\operatorname{pr}^2(\Delta\lambda_0,\Delta\lambda_1) = (\Delta\lambda_0)$.

- **Projection $ \mathbb{Z}^n \to \mathbb{Z}^{n-1} $ For $n \ge 3$.** $\operatorname{pr}^n(\Delta\lambda_0,\dots,\Delta\lambda_{n-3},\Delta\lambda_{n-2},\Delta\lambda_{n-1})=(\Delta\lambda_0,\dots,\Delta\lambda_{n-3},\Delta\lambda_{n-1})$.

So the action projection removes the action coordinate corresponding to the removed state coordinate. This is the natural edge/morphism projection induced by the tuple structure.


### Upward Assembly Of Actions

States project downward. Actions assemble upward. What this means here is that although the total action at rank $n$ is still a vector in $\mathbb{Z}^n$, policies are trained tierwise so that each tier chooses only one new coordinate of that vector. So if the total rank-$n$ action is $\Delta s = (\Delta\lambda_0,\dots,\Delta\lambda_{n-1})$, then the tierwise picture is:

- the first tier decides the first coordinate
- the next tier decides the newly introduced coordinate
- the next tier decides the newly introduced coordinate
- and so on

Thus "upward assembly" does **not** mean that the action is a different mathematical kind of object than a vector. It means:

- the full vector is built tier by tier
- each tier contributes the next action coordinate

This is one of the most important naming clarifications in the model.


### Tiered Window Objects
For each tier $k$, there is a window object $W_t^k$ which is the canonical representation of the recent rank-$k$ passage used by both the tier-$k$ policy and the tier-$k$ reward. This is intentionally inherited from the older `rl_counterpoint` system in spirit. The current design decision is that the old window logic is conceptually reused with $n$ reinterpreted as $k$. So the tier-$k$ window is the rank-$k$ analogue of the old finite rolling passage window. It is intended to include the same general kinds of information:

- recent realized $k$-chords
- padding
- valid-mask structure
- metrical annotations
- start/goal conditioning as appropriate

The crucial point is that policy input at tier $k$ and reward input at tier $k$ are built from the same formal window object. This is a direct carryover from the older system, but reinterpreted through the tiered model.


### Policies
For each tier $k$, there is a policy $\pi^k$. The policy at tier $k$ does **not** choose the whole rank-$k$ action vector at once. Rather, it chooses the newly added coordinate of the total action vector, given the already available lower-tier structure. So the correct conceptual form is:

$$
\pi^k(\text{new action coordinate at time } t \mid W_t^k).
$$

Here $W_t^k$ is the rank-$k$ window object, and the "new action coordinate" means the coordinate newly introduced at tier $k$. Thus the tower of policies is a tower of conditional coordinate-selection policies. This is the mechanism by which the full action vector is assembled.

## Training

[...]

### Rewards
For each tier $k$, there is a reward function $R_k$. This reward is genuinely a $k$-part voiceleading reward. It is **not** required to reduce to $R_{k-1}$. Rather:

- $R_1$ evaluates one-part voiceleading
- $R_2$ evaluates genuinely two-part voiceleading
- $R_3$ evaluates genuinely three-part voiceleading
- and so on

The project manager's point is that each tier introduces a genuinely new musical judgment problem. So:

- higher-tier rewards are not mere decorations on lower-tier rewards
- each tier has real new musical responsibility

### Dependence On Recent Passage
The reward at tier $k$ need not be purely one-step local. It may depend on recent rank-$k$ passage history. For example, checking whether a cadence has occurred may require the last three chords So $R_k$ is naturally allowed to depend on the same window object $W_t^k$ used by $\pi^k$. This keeps policy and reward aligned at the mathematical level.

---

## Other important aspects of the model

[...]

### Special Role Of $G(2)_{\bullet}$
The graph $G(2)_\bullet$ is not just the graph of ordinary standalone two-part writing. It is the graph of two-part writing intended as a scaffold for later higher-part writing. This matters because:

- standalone two-part writing often prefers relatively narrow vertical spacing
- but two-part writing intended as a scaffold for later three- or four-part writing must leave room for later inserted voices

The key design decision is:

- this is **not** treated as an extra mysterious future-room predicate
- it is handled by the ordinary vertical-width pruning rule
- but the width rule depends on the intended maximum rank of the tower

So the graph family should really be understood as depending on a target maximum rank $N$. For fixed current rank $n$ and intended maximum rank $N$, the width condition is not just an upper cap. It is a range:

$$
L(n,N)\;\le\;\operatorname{width}(s)\;\le\;U(n,N).
$$

Here:

- $\operatorname{width}(s)$ is the vertical span of the chord, typically top minus pedal
- $L(n,N)$ is the lower bound needed to leave enough room for future inserted voices
- $U(n,N)$ is the upper bound needed to keep the chord musically coherent

This is another project-manager design insight: if a scaffold chord needs to be wide, it should be structurally required to be wide, not merely permitted to be wide by accident. So the "future room" condition is simply part of the ordinary node pruning, expressed by a width band rather than only a width ceiling.

### Stabilization Above Rank 4

The current design discussion focuses on

$$
G(4)_\bullet \xrightarrow{\operatorname{pr}^4} G(3)_\bullet \xrightarrow{\operatorname{pr}^3} G(2)_\bullet \xrightarrow{\operatorname{pr}^2} G(1)_\bullet.
$$

For ranks above $4$, the intended rule is that the same projection pattern continues as at the highest displayed stage. That is: one always removes the second-from-top coordinate under projection and adds a new interior coordinate under extension. So the large-$n$ regime is intended to repeat the stable interior-voice insertion pattern already visible at the top of the displayed tower.

---

### Why The Tower Should Speed Up Training

The project manager's speedup claim is not incidental. It is one of the main motivations for the entire tower model.

#### Training As Search

The basic viewpoint is:

- training is a kind of search

It searches over:

- policy behavior
- action structure
- legal transition patterns
- successful passage continuations

So any structure that reduces search burden should also reduce training burden.

#### Flat Search Versus Hierarchical Search

In the flat system:

- one searches directly in a high-branching full-rank graph

In the tower system:

- one searches first in lower-rank graphs
- then in extensions of those lower-rank graphs
- then in extensions of those extensions

The comparison made by the project manager is to HNSW-like or hierarchical-search structures:

- a flat search explores one large combinatorial object directly
- a hierarchical search uses coarser levels to guide finer ones

The important difference here is:

- in the voiceleading problem, the quotient/projection structure does not need to be chosen artificially or randomly
- it arises canonically from the coordinate structure of voiceleading itself

That is a major part of the elegance of the model.

#### Dimensionality Interpretation

Another project-manager observation is that if local out-degree is taken as a proxy for local search dimension, then:

- a flat high-rank graph is effectively forcing search on the boundary of a very high-dimensional object
- tiering the problem reduces that burden stage by stage

So the expected benefit is not just "fewer choices."

It is a radical reduction in effective search dimension by making the problem hierarchical.

This is why the hoped-for speedup is described in ordinary hierarchical-search language as something like an $n$-to-$\log n$ style gain:

- not because the exact asymptotics have already been proven in this setting
- but because the search burden is expected to collapse in the standard hierarchical-search way

Thus:

- lower-rank policies build reusable search structure
- higher-rank training searches only over compatible extensions
- the graph-morphism condition guarantees that lower-rank validity remains valid inside higher-rank search

This is the core efficiency idea of the tower.

---

## Final Attribution Note

This mathematical model, including:

- the nested graph tower
- the graph-morphism requirement
- the voiceleading-builds-by-rank observation
- the upward action assembly by coordinate choice
- the use of genuinely $k$-part rewards
- the width-band scaffold condition
- the hierarchical-search / HNSW-style training-speedup intuition

is the project manager's design and should be understood that way by any future engineer reading this document.
