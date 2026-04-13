# RL Attractors and Escape Mechanisms on a Finite Counterpoint Graph

## Purpose

This note captures an abstract RL discussion in language matched to this repository, then identifies which parts already apply to the current `rl_counterpoint` implementation and which parts do not.

## Abstract setup

Suppose the environment state space is a finite directed graph.

- States are graph nodes.
- Actions induce legal transitions.
- A deterministic greedy policy induces a self-map on the finite state set.

If the policy is fully deterministic and greedily chooses one successor at each state, then repeated application of that policy eventually enters a periodic orbit.

On a finite state space, this means:

- every trajectory eventually falls into a cycle or fixed point
- these cycles are the attractors of the greedy policy-induced dynamics

In RL terms, this is the familiar collapse into recurrent classes of a deterministic policy on a finite graph.

## Standard escape mechanisms

### 1. Stochastic policy instead of hard greedy policy

Replace deterministic action choice with a stochastic policy.

Canonical examples:

- epsilon-greedy exploration
- categorical sampling from policy probabilities
- any policy assigning positive probability to more than one legal action

Effect:

- deterministic attractors become escapable
- the induced dynamics is a Markov chain rather than a pure endomorphism
- cycles become metastable rather than strictly absorbing

This is the most standard answer.

### 2. Entropy regularization / soft policies

Instead of optimizing expected return alone, optimize expected return plus a policy-entropy term.

This favors higher-entropy action distributions and discourages premature collapse to one action per state.

Effect:

- keeps multiple legal actions alive longer during training
- reduces early policy freezing
- can be viewed as a temperature-perturbed version of greedy dynamics

This is the modern policy-gradient version of "do not get trapped too early."

### 3. Exploration bonuses / intrinsic reward

If the reward landscape itself pulls the agent into a local basin, modify reward to include novelty pressure.

Canonical examples:

- count-based bonuses
- state visitation bonuses
- curiosity / prediction-error bonuses

Effect:

- previously unseen regions become temporarily more attractive
- local basins become shallower
- the agent is rewarded for leaving familiar trajectories

This changes the objective, not just the action-selection rule.

### 4. Noise in parameter space

Instead of perturbing sampled actions, perturb the policy parameters during training.

Effect:

- helps escape local minima in policy space
- addresses optimizer-level trapping, not only state-space trapping

This is related but distinct from graph-state attractors.

## 5. Non-myopic value-based reasoning

Purely greedy next-step choice is especially vulnerable to local attractors because it overweights immediate gain.

A value-aware policy can accept a temporarily worse immediate move if that move exits a bad local cycle and improves long-horizon return.

Effect:

- replaces one-step greediness with multi-step optimization
- allows deliberate escape moves
- is one of the core motivations for value functions and dynamic programming

In abstract terms: replace local greedy flow with long-horizon optimization.

### 6. Reset, restart, and data-diversification mechanisms

Even if the policy has attractors, training data need not collapse onto them completely.

Canonical examples:

- randomized episode starts
- diverse initial states
- replay or sampling schemes that emphasize rare states

Effect:

- broadens the empirical state distribution seen during training
- reduces overfitting to one narrow basin

## Clean abstract formulation

One clean formalization is:

- `T`: deterministic greedy transition operator
- `K`: exploration kernel
- `P_epsilon = (1 - epsilon) T + epsilon K`

Then `P_epsilon` is a Markov operator rather than a deterministic self-map.

For `epsilon > 0`, strict deterministic attractors are broken unless the exploration mechanism itself preserves them.

The important conceptual shift is:

- hard attractor under deterministic policy
- metastable region under stochastic perturbation

That is the standard abstract mathematical answer.

## What applies to `rl_counterpoint`

These observations are grounded in the current repo state, not just in the abstract discussion.

### Observation 1. We are not currently doing Q-learning

The live training path is policy-only REINFORCE, not Q-learning.

Repo evidence:

- `README.md` describes "rollout and REINFORCE training utilities"
- `rl_counterpoint/algos/reinforce.py` implements explicit REINFORCE episode updates
- `scripts/train_reinforce.py` is the active training entrypoint

So the right comparison is not "how does this modify our Q-learning?" but rather "which of these ideas already exist in our policy-gradient path?"

### Observation 2. We already have stochastic action sampling during training

During rollout collection, the policy does not act greedily.

`rl_counterpoint/algos/rollout.py`:

- builds logits over the legal action set
- applies `torch.softmax` on legal logits
- samples with `rng.choices(..., weights=probabilities, k=1)`

That means the current training-time policy-induced dynamics is already stochastic.

So, in the language of the abstract report:

- we are already replacing a deterministic graph endomorphism with a Markov-style transition kernel during training
- we already have one standard escape mechanism active

This is the most immediate applicable observation.

### Observation 3. We do use greedy selection in evaluation/export

The training script exports one example evaluation trajectory by taking the argmax legal logit at each step.

`scripts/train_reinforce.py`:

- computes legal logits
- picks `torch.argmax(legal_logits)`
- rolls out a deterministic episode for MIDI export

So if attractor behavior shows up in exported examples, that would be expected even if training itself still samples stochastically.

This distinction matters:

- training path: stochastic
- example evaluation path: greedy

Therefore the attractor language applies especially strongly to the evaluation/export rollout.

### Observation 4. Entropy is implicit in sampling, but not explicitly optimized

The current policy samples from softmax probabilities, but there is no explicit entropy bonus in the REINFORCE loss.

`rl_counterpoint/algos/reinforce.py` currently computes:

- discounted returns
- normalized returns
- `-log_prob * return`

There is no additional entropy regularization term.

So:

- soft stochastic sampling is present
- explicit entropy regularization is not

This means the current system may still collapse toward low-entropy behavior as training progresses, even though early training is stochastic.

### Observation 5. Reward-driven local basins are likely a real issue here

This project’s environment is exactly the kind of graph where musically plausible local cycles could emerge:

- finite legal graph
- constrained action mask
- domain-specific reward shaping
- strong possibility of repeated locally acceptable harmonic motion

The repo currently trains with `TargetRootOctaveReward` in `scripts/train_reinforce.py`, while earlier continuity notes explicitly say broader reward design is still evolving.

So exploration-bonus ideas are conceptually relevant here, but they are not currently implemented.

The main practical implication is:

- if the reward strongly favors locally pleasant but globally repetitive movement, the learner can still settle into musically narrow regions even with stochastic sampling

### Observation 6. The "non-myopic objective" observation only partially applies

The abstract discussion says value-based or long-horizon reasoning can help escape local attractors.

In this repo:

- the current learner is REINFORCE, so it is optimizing trajectory return, not one-step greedy reward only
- however, there is no active learned value baseline or Q-function in the current path

So the repo is already more long-horizon than a purely myopic greedy controller, but it is not currently using the stronger value-based escape mechanisms discussed in the abstract note.

### Observation 7. Reset diversity is already structurally present, but limited

Episodes do reset, and rollout seeds vary across episodes in `scripts/train_reinforce.py`.

However:

- the environment build path is fixed
- the initial-state regime does not yet appear to be a deliberately diversified curriculum
- there is no replay system emphasizing rare states

So restart/diversification ideas apply in a weak form now, not a strong form.

## Most relevant takeaways for this repo

If we translate the abstract conversation into immediate repository-level conclusions, the strongest ones are:

1. The repo is not currently using Q-learning; it is using policy-only REINFORCE.
2. Training already includes one standard anti-attractor mechanism: stochastic action sampling from masked softmax probabilities.
3. Evaluation/export currently becomes deterministic greedy rollout, so attractor behavior is more likely to appear there.
4. Explicit entropy regularization is not yet present.
5. Exploration bonus / novelty reward ideas are not yet present.
6. Reward design remains a likely source of local musical basins even if action sampling is stochastic.

## Practical interpretation for counterpoint generation

For this project, "escaping attractors" likely means at least two different things:

1. escaping graph-dynamical cycles in the generated chord trajectory
2. escaping overly narrow musical habits induced by the reward and legality structure

The current code already addresses the first problem partially during training through stochastic sampling.

It does not yet explicitly address:

- entropy preservation during optimization
- novelty-seeking reward terms
- deliberate curriculum over rare or diverse harmonic regions
- value-based planning or Q-learning-style escape logic

## Bottom line

The abstract conversation mostly does apply to `rl_counterpoint`, but with one important correction:

the repo is not currently a Q-learning system.

The most relevant existing mechanism in this codebase is already the standard one from the abstract discussion:

- stochastic policy sampling over legal actions

The most relevant missing mechanisms are:

- explicit entropy regularization
- exploration bonuses / novelty pressure
- stronger long-horizon or value-based escape machinery if later desired
