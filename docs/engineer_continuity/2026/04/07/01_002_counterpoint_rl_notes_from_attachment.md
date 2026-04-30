# Counterpoint RL Notes from Attached Conversation

This Markdown file is a cleaned handoff version of the attached pasted text.

## 1. State space and the n-chord non-crossing graph

### User prompt

A state is a $n$-tuple $(\lambda_0,\cdots,\lambda_{n-1})\in\{0,1,2,\cdots,128\}^{\times n}$ ($n$-fold frequency space) such that $\lambda_{i}<\lambda_{i+1}$ for all $0\le i < n-1$. Initially, an edge $(\lambda_0,\cdots,\lambda_{n-1})\longrightarrow(\mu_0,\cdots,\mu_{n-1})$ is present if and only if $\lambda_{i-1}<\mu_{i}$ for all $i>0$ and $\mu_{i}<\lambda_{i-1}$ for all $i<n+1$. This is the "$n$-chord non-crossing graph." Just explain in your own words to show me you understand. If you *actually* understand, you'll be able to say why frequency space would have the form $\{0,1,2,\cdots,128\}$.

### Response

A state is an ordered chord with $n$ voices, where each voice is assigned a discrete pitch value and the voices are listed from low to high. So

$$
(\lambda_0,\dots,\lambda_{n-1}) \in \{0,1,\dots,128\}^n,
\qquad
\lambda_i < \lambda_{i+1}
$$

means:

- each $\lambda_i$ is a pitch slot in a bounded discrete register,
- the strict inequalities force a true vertical ordering,
- no two voices occupy the same pitch,
- voice $0$ is the lowest voice and voice $n-1$ the highest.

So this is not just a set of notes. It is a voiced chord state with persistent voice identity.

The graph structure is trying to encode legal voiceleadings from one chord to the next. An edge

$$
(\lambda_0,\dots,\lambda_{n-1}) \to (\mu_0,\dots,\mu_{n-1})
$$

is meant to say that the new chord $(\mu_i)$ can follow the old chord $(\lambda_i)$ when the voices move in a way that does not cross. In geometric terms, if you draw line segments from old voices to new voices, the segments should preserve vertical order rather than tangling.

So the intended meaning of the non-crossing graph is:

- lower voices remain lower than higher voices after the move,
- the voiceleading respects vertical order,
- the graph is a combinatorial skeleton of admissible chord-to-chord motions.

There is also an important reality check: as written, the edge condition looks malformed. Taken literally, the two inequalities appear to push in opposite directions with the same neighboring indices, so the intended idea is clear — interlacing / non-crossing voiceleading — but the symbolic formula likely needs correction before building on it.

Why would “frequency space” have the form $\{0,1,\dots,128\}$?

Because this is almost certainly not literal physical frequency. Literal frequencies live in a positive real interval, not in a finite integer set. So $\{0,\dots,128\}$ is really a discretized pitch register — most naturally something MIDI-like, or at least semitone-index-like. In other words:

- pitch is being treated as a bounded integer lattice,
- each integer is a pitch slot,
- “frequency” here really means quantized pitch level.

And the reason the bound looks like $128$ is that it evokes the standard MIDI-style finite pitch universe. The caution is that standard MIDI note numbers are usually $0,\dots,127$, not $128$, so using $\{0,\dots,128\}$ suggests one of the following:

- an intentional choice of 129 pitch levels,
- an inclusive top cap at $128$,
- or a mild off-by-one relative to standard MIDI convention.

So the object is: a finite ordered lattice of voiced chords, with edges encoding non-crossing voiceleading in a bounded discrete pitch register.

---

## 2. The book rules as both reward and environment weighting

### User prompt

There is a book called *Tonal Counterpoint for the 21st-Century Musician*. It has this really clear, almost JSON-like statement of voiceleading principles that can be read as both a guide for defining a reward function for a voiceleading agent in this graph, and a weighting on edges themselves, like an environment weighting. Explain this back to me to show me you understand what I'm getting at.

### Response

The book’s rule system can play **two mathematically different roles** on the same chord-transition graph.

#### First role: reward function for an RL agent

- The agent walks through the $n$-chord non-crossing graph.
- A trajectory is a sequence of voiced chords.
- The book’s voiceleading principles assign local and possibly global desirability to what the agent does.

So after a transition

$$
\lambda \to \mu
$$

or after a longer fragment, reward can be assigned according to how well that move satisfies the counterpoint principles:

- contrary vs. parallel motion,
- size of leaps,
- tendency-tone treatment,
- preparation / resolution patterns,
- spacing,
- melodic singability of each voice,
- approach to cadence,
- avoidance of forbidden parallels.

In RL language, the book gives a way to define

$$
r(\lambda,\mu)
\qquad\text{or perhaps}\qquad
r(\lambda,\mu,\text{context})
$$

and the policy learns to choose paths maximizing cumulative musical value.

#### Second role: edge weighting on the graph itself

Instead of saying only that an edge is present or absent, a weight is attached to each legal edge. Then the graph already carries a built-in musical geometry:

- good voiceleadings = low cost or high weight,
- bad but still legal voiceleadings = high cost or low weight,
- impossible / forbidden voiceleadings = no edge at all.

So the book is not just supervising the agent externally. It is also defining the structure of the environment internally.

That means the same principles can be used at three levels:

1. **hard constraints** — remove edges entirely,
2. **soft structural bias** — weight edges by musical quality,
3. **learning signal** — convert those same evaluations into reward along trajectories.

The key idea is that the book’s prose is explicit enough to become something close to a declarative schema, for example:

```text
if outer voices move in parallel perfect fifths:
    forbid edge

if a voice leaps larger than sixth:
    apply penalty

if leading tone resolves correctly:
    apply bonus

if cadence is approached with proper contrary motion:
    apply bonus
```

So the text becomes both:

- a musical evaluator,
- and a combinatorial shaping of the search space.

That is powerful because RL is not trying to discover counterpoint from a formless space of note strings. It is learning inside a graph whose local geometry already reflects the style.

The important distinction is:

- **edge weighting** changes what the environment looks like before learning,
- **reward** changes what the agent is incentivized to do while learning.

Same principles, two different insertion points.

### Possible failure hypotheses noted in the discussion

1. The rules in the book may be more context-dependent than a purely local edge weight can express.
2. Some principles may be path-level or cadence-level and cannot live on single edges without extra state.
3. If too much is encoded as edge weights, RL may collapse into shortest-path search rather than genuinely sequential decision-making.

---

## 3. What is critical when turning the book into Markdown rules and JSON reward scoring

### User prompt

The next thing I want to do is make:

1. an `md` of the rules in the book, and then
2. a JSON to use for an actual reward function scoring system.

I think I can produce these, but tell me what is really critical for RL as I do this.

### Response

The most critical thing is: **do not let the book collapse into a single flat score too early**.

You want **three layers**, kept separate from the start:

1. **Rule extraction layer** — plain-English / Markdown restatement of what the book says,
2. **Feature layer** — machine-evaluable predicates and measurements,
3. **Scoring layer** — weights, penalties, bonuses, aggregation.

If the middle layer is skipped, then later it will be unclear whether a bad result came from:

- the musical rule itself,
- the formalization of the rule,
- or the chosen weight.

That separation is crucial for RL.

### The most critical design points

#### 1. Separate hard constraints from soft preferences

Some rules should remove actions or edges entirely. Others should merely penalize them. That distinction must be explicit in the JSON.

Otherwise you get a confused environment where:

- forbidden things are merely “kind of bad,” or
- stylistic preferences accidentally become impossible constraints.

#### 2. Mark which rules are local and which require history

This is probably the single biggest technical issue.

Some rules are evaluable from just

$$
(\lambda,\mu)
$$

meaning one chord transition.

Others require:

- the previous transition,
- voice trajectory over several steps,
- phrase position,
- approach to cadence,
- beginning / middle / ending context.

So every rule in the Markdown should get a tag like:

- `scope: edge`
- `scope: two_step`
- `scope: phrase`
- `scope: terminal`
- `scope: trajectory`

#### 3. Keep raw musical diagnostics separate from weighted reward

For each rule, store the raw quantity first.

Bad:

```json
{"rule": "large_leap", "reward": -0.7}
```

Better:

```json
{
  "rule": "large_leap",
  "feature": {
    "max_leap_semitones": 9,
    "threshold": 5,
    "violation": true
  },
  "score": -0.7
}
```

This matters because when training goes wrong, you need to know whether the model is making large leaps, or whether your weighting is wrong.

#### 4. Make every rule evaluable by a deterministic function

Each rule in the Markdown should be translatable into something like:

```text
inputs -> feature value -> pass/fail/graded value
```

If a rule cannot be operationalized, mark it honestly. Suggested categories:

- `implemented`
- `needs_context`
- `underspecified_in_source`
- `human_judgment_only`

#### 5. Do not reward the same musical fact five different ways

Reward double-counting will happen easily.

For example, if you separately reward:

- smooth melodic motion,
- small leaps,
- common-tone retention,
- minimal total displacement,
- stepwise contrary motion,

then you may be rewarding the same underlying phenomenon multiple times.

#### 6. Terminal cadence scoring must be its own section

Do not mix cadence reward into ordinary transition reward.

Cadence has different structure:

- it depends on phrase end,
- it may require target octave / register,
- it may depend on approach pattern,
- it may need stronger reward than ordinary local niceness.

So in the JSON, keep at least:

- `transition_rules`
- `terminal_rules`
- optionally `trajectory_rules`

#### 7. Preserve explainability per voice

Because this is counterpoint, many failures are voice-specific.

The system should be able to say not just “bad chord transition,” but things like:

- alto leapt a seventh,
- bass crossed prior tenor range,
- soprano failed tendency resolution,
- outer voices formed parallel fifths.

That means many rule outputs should include:

- implicated voices,
- interval class,
- direction,
- magnitude,
- location in trajectory.

#### 8. Build the Markdown as a normative spec, not just notes

For each rule, record:

- natural-language statement,
- musical intuition,
- exact evaluator inputs,
- whether it is hard or soft,
- scope,
- sign of preference,
- open ambiguities,
- examples of pass/fail.

If this is done well, the JSON becomes almost mechanical.

#### 9. Decide now whether reward is additive

Many RL implementations quietly assume something like

$$
R = \sum_i w_i f_i.
$$

But some musical rules are really lexicographic:

- first avoid forbidden parallels,
- then prefer singable melodic motion,
- then optimize cadence elegance.

That is not the same as a flat weighted sum. While drafting, mark whether each rule belongs to:

- legality,
- primary style,
- secondary style,
- ornament / tie-break.

#### 10. Make a non-RL evaluator first

Before any agent exists, it should be possible to feed in a hand-written trajectory and get back:

- total score,
- per-step score,
- per-rule breakdown,
- implicated voices,
- terminal evaluation.

If the evaluator is not trustworthy before RL, RL will only amplify the confusion.

---

## 4. Minimal JSON shapes suggested in the discussion

### Hard constraint example

```json
{
  "id": "parallel_perfects",
  "description": "Avoid parallel perfect fifths or octaves between specified voice pairs.",
  "scope": "edge",
  "type": "hard_constraint",
  "inputs": ["prev_chord", "next_chord"],
  "applies_to": ["voice_pairs"],
  "feature": "detect_parallel_perfects",
  "aggregation": "any",
  "weight": null,
  "notes": "Need exact source-language interpretation for hidden/direct parallels."
}
```

### Soft preference example

```json
{
  "id": "prefer_stepwise_motion",
  "description": "Prefer smaller melodic motion in each voice.",
  "scope": "edge",
  "type": "soft_preference",
  "inputs": ["prev_chord", "next_chord"],
  "feature": "melodic_interval_size_by_voice",
  "aggregation": "sum_over_voices",
  "weight": -0.2,
  "normalization": "per_voice"
}
```

---

## 5. Highest-risk failure modes called out in the discussion

1. Formalizing prose that actually needs phrase-level context as though it were edge-local.
2. Mixing hard constraints and soft preferences in one undifferentiated score.
3. Double-counting closely related musical facts.
4. Encoding weights before the raw features are trustworthy.
5. Failing to preserve enough diagnostic detail to debug learned behavior.

---

## 6. Strongest recommendation from the discussion

Write the Markdown first with a strict table for every rule:

- `rule_id`
- `book wording`
- `my restatement`
- `scope`
- `hard/soft`
- `required context`
- `machine feature`
- `ambiguities`
- `pass example`
- `fail example`

Then generate the JSON from that table rather than writing the JSON from scratch.

### Next action proposed in the discussion

Draft 5 actual rules from the book in that table format, then pressure-test them for RL suitability before scaling up.
