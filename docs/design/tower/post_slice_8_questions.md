# Post-Slice-8 Questions

This document lists the questions that should be answered before approving or
revising `post_slice_8_build_plan.md`.

Use this as an owner-answer worksheet. Answers here should guide the next
post-Slice-8 build plan.

## 1. Product Direction

### 1.1 Next Priority

What is the next priority?

Answer:

```text
A
```

Candidate directions:

| Option | Direction |
| --- | --- |
| A | learnable real policy architecture |
| B | broader graph generality, such as rank-3 or rank-k |
| C | training runner / artifact workflow |
| D | audible MIDI output |
| E | reward expansion |

### 1.2 Audible MIDI Milestone

Should the next milestone produce audible MIDI examples?

Answer:

```text
I only care that training ends with a last, no-train inference epsiode that is converted to MIDI and saved as an artifact.

```

Related details to decide:

| Question | Answer |
| --- | --- |
| Is inspectable generated music a near-term requirement? | It is important for inspecting training efficacy |
| Should MIDI export become part of the next slice? | Yes |
| Should example MIDI artifacts be written during training/checkpointing? | At end of training, after last episode. |

### 1.3 Reward Before Long Training

Should reward richness come before long training runs?

Answer:

```text
I need clarification here.
```

Consultant answer:

```text
The distinction is between infrastructure-training and meaningful musical
training.

For the next slice, the current minimal reward layer is enough to test that a real policy can connect to rollout, loss, checkpoints, and final MIDI export. It is probably not rich enough to expect musically satisfying convergence from long runs.

Recommendation: do not block the first real-policy/training-runner work on a large reward expansion. Use the current cadence/success reward for smoke-scale training, but treat the results as infrastructure evidence. Add richer reward terms before interpreting training quality seriously.
```

Follow-up Answer:

```text
Ok I agree with your recommendation.
```

Related details to decide:

| Question | Answer |
| --- | --- |
| Are current cadence/success predicates enough for first experiments? | Yes |
| Do we need more TC21M-style reward terms before training is meaningful? | I don't think so... what do we have now? |
| Should weak rewards be accepted temporarily just to test infrastructure? | Tell me more. |

Consultant answer:

```text
What exists now:

- `rank1_projected_cadence_success`: final-step rank-1 V-I root motion in the requested key, on the ending beat.
- `rank2_lifted_cadence_success`: rank-2 success that first requires projected rank-1 success, then requires the outer voice to form major thirds over the dominant and tonic.
- `SuccessRewardTerm`: converts a success predicate into reward.
- `CompositeRewardTerm`: sums structured reward terms and preserves diagnostics.

What does not exist yet:

- motion rewards,
- dissonance treatment,
- suspension logic,
- harmonic-template vocabulary,
- rich voice-leading penalties/rewards,
- dense shaping rewards.

Weak rewards are acceptable temporarily if the goal is to prove infrastructure: the policy samples, the mask is applied, gradients flow, artifacts are written, and the final no-train episode can be exported. Weak rewards are not acceptable as evidence that the musical learning problem is solved.
```

Follow-up Answer:
```text
Ok. Then not just weak. I want to see training working.
```

## 2. Model Architecture

### 2.1 Shared Transformer Family

Should each rank use the same transformer-family architecture?

Answer:

```text
Yes, up to abvious modificaitons coming from differences in tensor shape or anyhting like that. The modifications should be obvious.
```

Related details to decide:

| Question | Answer |
| --- | --- |
| Should rank 1 and rank 2 share one model class? | Yes, unless there are obvious modifications coming from tensor shape differences. |
| Should rank-specific differences be config-driven? | YES. I now see what you're getting at. Yes |
| Should parent and child policies have the same observation format? | What does this mean? |
| Should policy code be designed now for rank-k reuse? | Yes |

Consultant answer:

```text
"Same observation format" means same conceptual schema, not identical tensor shape.

For example, every rank-local policy could receive:

- a fixed-length tower window,
- valid/pad mask,
- bar or beat position,
- tonic/key context,
- optional target context,
- rank/config metadata,
- and then rank-width state features.

Rank 1 and rank 2 would differ because the state tuple width differs and the action space/mask differs. But the model input contract should be one shared family: "rank-local timed tower window plus context", with rank-specific tensor sizes driven by config.

So your answer is: yes, same observation format in the schema sense; no, not same raw tensor dimensions.
```

### 2.2 First Real Policy Observation Sequence

What is the observation sequence for the first real policy?

Answer:

```text
I'm not going to write answers here, because I think all of this can be deduced from details of older `rl_counterpoint` project, and from design documents regarding that and current project.
```

Consultant answer:

```text
Agreed. The first real policy observation contract should be derived rather
than invented.

The old project gives the main pattern:

- a fixed-length timed chord window,
- left padding with a distinguished PAD chord,
- a valid mask,
- tonic/meter/target context,
- symbolic event encoding,
- transformer encoder over the sequence,
- final valid event hidden state feeding action logits.

The tower adaptation should keep the same shape of idea while replacing old
ChordState assumptions with tower-local rank state:

- `TowerWindow.states` becomes the timed sequence,
- `TowerWindow.valid_mask` becomes the padding mask,
- `TowerWindow.bar_positions` or equivalent meter features should be explicit model input, not only reward context,
- each event encodes the rank-local tuple state,
- rank-specific tuple width is config-driven,
- output logits correspond to the rank-local active action choices, then are masked by legality/lift-fiber constraints outside the model.

The first tensorization contract should be small and explicit: one encoded rank-local tower window plus masks/context in, one vector of active-choice logits out.
```

Related details to decide:

| Question | Answer |
| --- | --- |
| What does a tower window become as model input? | - |
| Are tokens pitches, actions, intervals, beats, voices, rank coordinates, or some combination? | - |
| Does the policy see only the current rank window? | - |
| Does the child policy see projected parent history? | - |
| Does the policy see meter/beat position explicitly? | - |
| Does the policy see target/goal information? | - |
| How are padded positions represented? | - |
| What is the first tensorization contract? | - |

### 2.3 Parent Diagnostics As Child Inputs

Should parent diagnostics be inputs to the child policy, or only rollout
constraints?

Answer:

```text
Let's first design a system of models that only follows previous policy by constraining options. This is the whole reason that the speedup in training in supposed to occur. If we start lifting data, we're secretly demanding that the model have nearly the same capacity as previous `rl_counterpoint` models that have trouble converging.
```

Related details to decide:

| Question | Answer |
| --- | --- |
| Does the child policy receive the sampled parent action? | It should know what action happens downstairs, and I think the right way to do this is increase rank of inputs, and then mask output to be contrained over action or *something* along these lines. |
| Does it receive parent action logprob? | No. |
| Does it receive parent top-m candidates? | No. |
| Does it receive parent distribution/logits? | No. |
| Or does the parent only constrain the child through the lift fiber? | Yes. |

Consultant answer:

```text
This is consistent with the tower math.

The child training step must know the sampled parent action operationally, because the lift fiber is defined over that parent action. But the parent action does not need to become a rich learned feature like parent logits or parent distribution.

Recommended interpretation:

- the rollout samples a parent action downstairs,
- the graph code computes the legal lift fiber upstairs,
- the child policy produces logits for its rank-local active choices,
- the sampler masks those logits to the lift fiber,
- parent logprob/top-m/distribution remain diagnostics only.

So the child "knows" the downstairs action through the constrained candidate set, not through extra capacity-heavy parent-distribution input.
```

Follow-up Answer:
```text
Sounds good.
```

## 3. Training Protocol

### 3.1 Rank-1 Checkpoint Acceptance

What makes a rank-1 checkpoint accepted in real training?

Answer:

```text
In words: It does the rank-1 projection of a perfect cadence in the correct octave and key, situated correctly on weak/strong beats. In rank 2, this goal will extend to performing the (outervoice) rank-2 projection of a perfect cadence in the correct octave and key, situated correctly on weak/strong beats.

```

Consultant answer:

```text
This answer defines the musical acceptance target.

For the near-term implementation plan, we should separate two concepts:

1. training stop condition: probably episode count for now, as you answer in 3.3;
2. musical acceptance evidence: the final no-train inference episode should satisfy the appropriate rank-local cadence predicate and be  exported to MIDI.

That means a checkpoint can be mechanically produced after N episodes, but it should be called "accepted" only when the configured acceptance rule says so. For the next pass, the simplest rule can still be episode-budget based, while recording cadence success and MIDI output so we can make acceptance smarter later.
```

Candidate acceptance signals:

| Signal | Use? |
| --- | --- |
| manual owner acceptance | - |
| episode budget reached | - |
| reward threshold | - |
| musical validation score | - |
| generated MIDI inspection | - |
| combination of signals | - |

### 3.2 Parent Selection For Rank-2 Training

Should rank-2 train from one accepted parent or sample among accepted parents?

Answer:

```text
Yes rank 2 should train by sampling from rank 1 policy, and freezing above that. Yes we allow a small bit of randomness here, but lets say only top three... this way we're at least constraing over a low-dimensional space downstairs. Make this a hyperprameter so taht
```

Consultant answer:

```text
Interpreting the unfinished sentence:

Make `parent_top_m` a training/config hyperparameter, with an owner-preferred initial value of 3 for real rank-2 training.

Meaning:

- rank 2 is tied to one accepted rank-1 model checkpoint;
- at each rank-2 rollout step, the frozen rank-1 policy samples from its legal top-m parent actions;
- m defaults to 3 for real experiments, though tests may use 1 for determinism;
- the sampled parent action constrains the rank-2 child through the lift fiber.

This is different from sampling among multiple parent checkpoints. We are sampling actions from one frozen parent policy, not carrying multiple parent models inside one episode.
```

```text
Correct.
```

Related details to decide:

| Question | Answer |
| --- | --- |
| Is there exactly one promoted parent per lineage? | Yes, if I understand the quesiton correctly. What I mean here is that within one episode, there is one promoted parent action. |
| Can a lineage contain multiple accepted rank-1 parents? | This depends on what exactly "lineage" means. If you mean within one training episode, then no: we should be building one musical passage, not carrying multiple possiblities as we go. |
| Should rank-2 training be tied to a specific parent checkpoint? | If you mean model checkpoint, then yes. Downstairs actions restrict possiblities for upstairs actions. |
| Should parent selection become part of the training config? | Parent *what*? |
| How should lineage represent multiple possible parents? | It shouldn't, unless I really misunderstand what you're saying. |

Consultant answer:

```text
Clarification of terms:

- "parent checkpoint" means the accepted rank-1 model checkpoint used by rank-2 training.
- "parent action" means one sampled rank-1 action during a single rollout step.
- "lineage" means the artifact family tying rank-1, rank-2, ... training runs together. It is not the same thing as one episode and it is not a beam of possible musical passages.
- "parent selection" meant selecting which accepted rank-1 checkpoint a rank-2 run depends on. It did not mean selecting one parent action during rollout.

Given your answers, the intended design is:

- one rank-2 run is tied to one accepted rank-1 checkpoint;
- each episode samples a sequence of parent actions from that frozen checkpoint;
- no multiple-parent-checkpoint mixture is needed now;
- lineage should record the single parent checkpoint dependency.
```

### 3.3 Promotion Metric

What is the first success metric that gates promotion to the next rank?

Answer:

```text
Episode count for rank 1 training. Later we can try something more inteligent.
```

Candidate metrics:

| Metric | Use? |
| --- | --- |
| average reward | - |
| terminal success rate | - |
| cadence success rate | - |
| invalid action rate | - |
| musical rule violation rate | - |
| human-audited MIDI examples | - |
| stability across seeds | - |

## 4. Music Semantics

### 4.1 Next TC21M Rules

Which TC21M rules enter the next reward slice?

Answer:

```text
See below.
```

Candidate rule families:

| Rule family | Include? |
| --- | --- |
| cadence rules | Yes. |
| motion rules | Yes. |
| dissonance treatment | Yes. |
| suspension handling | What does this mean? |
| voice-leading constraints | Yes. |
| harmonic template rules | Yes. |
| six-four logic | No. |

Consultant answer:

```text
Suspension handling means rules/rewards for prepared dissonances that resolve by step after being held over from a previous harmony. In classical counterpoint/harmony terms, a suspension usually has:

- preparation: the note is consonant before the suspension,
- suspension: the same note is held while the harmony changes and becomes dissonant,
- resolution: the dissonance resolves, usually downward by step.

Examples include patterns like 4-3, 7-6, and 9-8 suspensions.

For this project, "suspension handling" would mean deciding whether such  dissonances are allowed, rewarded, penalized, or ignored. Since you said no to six-four logic and are trying to keep the next slice focused, I would defer suspensions unless TC21M demands them early.
```

Follow-up Answer:
```
I think this is the kind of thing that should go in a later project update, like `beta.1`, since it's really a style change to final output. Introducing suspensions is just like saying "ok voiceleading, but with this stylistic change".
```

### 4.2 Cadence Predicates For First Audible Training

Are cadence predicates enough for first audible training?

Answer:

```text
See below.
```

Related details to decide:

| Question | Answer |
| --- | --- |
| Can we train useful behavior with the current minimal terminal predicates? | Yes. |
| Do we need intermediate shaping rewards? | Explain. |
| Do cadence-only rewards create too sparse a signal? | My guess is yes. |
| Should rank 1 and rank 2 have different early reward priorities? | They should have different episode counts. |

Consultant answer:

```text
Intermediate shaping rewards are small rewards or penalties given during an episode, not only at the final cadence.

A terminal-only cadence reward says, in effect: "you get useful feedback only at the end, and only if the cadence worked." That can be sparse  because most early episodes fail and receive little information about why.

Shaping rewards can give partial guidance, for example:

- reward moving the rank-1 voice toward the target octave/key region,
- reward landing on dominant near the penultimate cadence position,
- penalize invalid/no-op extensions,
- penalize large or stylistically bad leaps,
- reward stepwise or contrary motion where appropriate,
- penalize unresolved dissonance once those rules exist.

The risk is over-shaping: the model can learn the shaping proxy instead of the musical goal. The safe next version is minimal shaping: keep cadence success as the main terminal goal, add only a few diagnostics or small penalties that make training interpretable.
```

Follow-up Answer:
```text
Don't a bunch of rewards around this already appear in the TC21M summarizing document in this repo, in assets/rules/tc21m_rules.md?
```

### 4.3 Scaling Voice-Leading Constraints

How should voice-leading style constraints scale beyond rank 2?

Answer:

```text
Let's come back to this. I need to think about it more carefully.
```

Related details to decide:

| Question | Answer |
| --- | --- |
| Should they be legality constraints or reward penalties? | - |
| Which constraints belong in graph legality? | - |
| Which constraints belong in reward? | - |
| How do constraints behave when adding a new outer voice? | - |
| Does the rank-k child only own new constraints involving its active coordinate? | - |
| Are cross-rank constraints checked through projection or directly at the full rank? | - |

## 5. Future Game Plan Selection

### 5.1 First Post-8 Slice To Plan

Which proposed post-8 slice should be planned first?

Answer:

```text
See below.

```

Candidate slices:

| Slice | Name | Select? |
| --- | --- | --- |
| Post-8 Slice A | Tower Training Runner | 1 |
| Post-8 Slice B | Real Policy Observation Contract |2 |
| Post-8 Slice C | Transformer Rank Policy | 3 |
| Post-8 Slice D | Example MIDI Artifact | This should be directly mimicked from previous `rl_counterpoint` project. |
| Post-8 Slice E | Reward Expansion Pass | 4 |
| Post-8 Slice F | Rank-k Generalization Assessment | 5 |

### 5.2 Assessment Approval

Should `post_slice_8_build_plan.md` be accepted as-is, revised, or used only as
a draft?

Answer:

```text
Before doing anything to that document, please, wouthout modifying what exists in this document now, answer my new questions in the appropriate places.
```

Possible decisions:

| Decision | Select? |
| --- | --- |
| accept as baseline | - |
| revise current-state audit | - |
| revise real-vs-skeletal classification | - |
| revise recommended future order | - |
| add/remove open questions before approval | - |
