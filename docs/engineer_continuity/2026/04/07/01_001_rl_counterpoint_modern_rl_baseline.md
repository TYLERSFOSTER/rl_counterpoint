# Modern RL Baseline for `rl_counterpoint`

This document is a baseline handoff for the next engineer. It combines:

1. an overview of the main **modern RL coding environments / stack shapes**, and
2. an opinionated **directory layout and dev workflow** for a custom PyTorch RL research repository, specialized to the counterpoint project.

The intended use case is: **custom RL for counterpoint generation**, where the environment, rule evaluator, and reward logic are domain-critical and should be trusted before any large training loop is trusted.

---

## 1. Modern RL coding environments: what they actually feel like

The right question is usually not “what RL library should I use?” but rather:

**Who owns the loop?**

There are four common answers.

### 1.1 Baseline harness

This is the environment where the **library owns the loop**.

Typical shape:

```python
env = make_env(...)
model = Algo(policy="MlpPolicy", env=env, **cfg)
model.learn(total_timesteps=...)
```

What this feels like in practice:

- you configure the trainer more than you write the trainer
- you mostly edit wrappers, configs, callbacks, logging, and evaluation
- you do **not** usually touch rollout bookkeeping or loss internals

This is the right environment when the goal is:

> “Give me PPO / SAC / DQN that works, fast.”

Typical tools in this lane:

- Gymnasium as environment interface
- Stable-Baselines3 as baseline trainer

---

### 1.2 Single-file research loop

This is the environment where **you own the whole loop**.

Typical shape:

```python
obs = env.reset()
for step in range(T):
    action = policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    buffer.add(obs, action, reward, next_obs, terminated, truncated)
    obs = next_obs

    if update_time:
        batch = buffer.sample()
        loss = compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

What this feels like in practice:

- raw PyTorch + an environment
- you directly see state, action, reward, rollout, and update flow
- debugging is local and implementation details are explicit

This is the right environment when the goal is:

> “I already know RL theory and want to inspect and mutate the algorithm.”

Typical tools in this lane:

- Gymnasium / PettingZoo
- explicit scripts
- optionally CleanRL as reference style

---

### 1.3 PyTorch-native component system

This is the environment where **you own the loop, but using reusable RL components**.

Conceptual shape:

```python
env = TransformedEnv(base_env, transforms...)
collector = SyncDataCollector(env, policy, ...)
replay_buffer = ReplayBuffer(...)
loss_module = PPOLoss(actor, critic, ...)

for batch in collector:
    replay_buffer.extend(batch)
    sub_batch = replay_buffer.sample()
    loss_vals = loss_module(sub_batch)
    loss_vals["loss"].backward()
    optimizer.step()
```

What this feels like in practice:

- closer to “serious PyTorch engineering” than to a black-box trainer
- explicit objects for collectors, replay buffers, modules, transforms, losses, etc.
- cleaner component boundaries, but more moving parts than a single-script trainer

This is the right environment when the goal is:

> “I want custom RL systems, but I still want them to feel like modular PyTorch.”

Typical tools in this lane:

- Gymnasium
- TorchRL
- Tianshou

---

### 1.4 Distributed runtime-owned loop

This is the environment where **the system / cluster owns the loop**.

What this feels like in practice:

- rollout workers are separate processes or nodes
- learners may be separated from actors
- config becomes more important than code elegance
- debugging shifts toward distributed-systems failure modes

This is the right environment when the goal is:

> “My real bottleneck is throughput, parallelism, or multi-node orchestration.”

Typical tools in this lane:

- RLlib / Ray

For the counterpoint project, this is **not** the recommended starting point.

---

## 2. The minimal practical mental model

For a custom RL system, the system boundary is usually not just `nn.Module`.

It is more like:

```text
Env -> Collector / Rollout -> Buffer -> Loss / Update -> Evaluator / Logger
```

The biggest engineering shift from ordinary PyTorch is that the “dataset” is no longer fixed.
The agent interacts with the environment, and interaction itself creates the training data.

This means the first-order bugs are often not “backprop bugs.”
They are more like:

- reset semantics bugs
- rollout slicing bugs
- invalid action masking bugs
- done / termination bugs
- reward leakage bugs
- stale-policy sampling bugs

For a music / counterpoint problem, there is a further shift:

- the **rule evaluator** is as important as the policy learner
- a bad musical evaluator will poison every downstream RL experiment

---

## 3. Project-specific framing for `rl_counterpoint`

The intended project is:

- fixed number of voices `n`
- state space based on ordered voice tuples in a bounded discrete pitch register
- sequential generation of chords
- fixed initial chord
- terminal cadential objective
- reward and/or edge weighting derived from explicit counterpoint rules

A crucial design principle for this project:

> The deterministic musical evaluator must become trustworthy **before** the RL training stack is trusted.

That means the first engineering center of gravity is not the trainer.
It is:

1. rule extraction
2. machine-evaluable features
3. reward / diagnostic evaluator
4. environment contract
5. only then training code

---

## 4. Recommended repo philosophy

The repository should separate three contracts.

### 4.1 Musical truth

This layer answers:

- is this transition legal?
- which rules fired?
- what raw musical features were measured?
- what is the unweighted diagnostic?
- what is the weighted score?

This should be independent of RL algorithm choice.

### 4.2 Environment truth

This layer answers:

- what is the observation?
- what actions are legal now?
- when does the episode end?
- what counts as success?
- what diagnostics go into `info`?

This wraps the musical evaluator as a sequential decision problem.

### 4.3 Learning truth

This layer answers:

- how actions are sampled
- how returns / advantages are computed
- how policies / values are updated
- how checkpoints and evaluation are run

This layer is allowed to change frequently.
The previous two should be more stable.

---

## 5. Exact directory layout for a custom PyTorch RL research repo

```text
rl_counterpoint/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── README.md
├── configs/
│   ├── base.yaml
│   ├── env/
│   │   └── counterpoint.yaml
│   ├── algo/
│   │   ├── ppo.yaml
│   │   └── reinforce.yaml
│   ├── reward/
│   │   └── tonal_rules_v1.yaml
│   └── exp/
│       ├── debug.yaml
│       ├── small.yaml
│       └── paper_baseline.yaml
├── assets/
│   └── rules/
│       ├── tonal_counterpoint_rules.md
│       ├── tonal_counterpoint_rules.json
│       ├── tonal_counterpoint_rules.schema.json
│       └── golden_examples/
│           ├── good_cadence_001.json
│           ├── bad_parallel_001.json
│           └── leap_resolution_001.json
├── scripts/
│   ├── smoke_env.py
│   ├── validate_rules.py
│   ├── score_trajectory.py
│   ├── train_reinforce.py
│   ├── train_ppo.py
│   ├── eval_policy.py
│   ├── sample_episode.py
│   └── export_best.py
├── runs/                  # ignored
├── checkpoints/           # ignored
├── notebooks/             # optional; never source of truth
├── rlcp/
│   ├── __init__.py
│   ├── envs/
│   │   ├── counterpoint_env.py
│   │   ├── observation.py
│   │   ├── action_space.py
│   │   ├── termination.py
│   │   └── wrappers.py
│   ├── rules/
│   │   ├── schema.py
│   │   ├── loader.py
│   │   ├── features.py
│   │   ├── constraints.py
│   │   ├── soft_rules.py
│   │   └── explain.py
│   ├── reward/
│   │   ├── reward_fn.py
│   │   ├── edge_weighting.py
│   │   ├── terminal_reward.py
│   │   └── aggregation.py
│   ├── music/
│   │   ├── pitch.py
│   │   ├── intervals.py
│   │   ├── chord.py
│   │   ├── cadence.py
│   │   └── voiceleading.py
│   ├── models/
│   │   ├── policy.py
│   │   ├── value.py
│   │   ├── encoders.py
│   │   └── distributions.py
│   ├── algos/
│   │   ├── reinforce.py
│   │   ├── ppo.py
│   │   ├── losses.py
│   │   └── rollout.py
│   ├── logging/
│   │   ├── tensorboard.py
│   │   ├── metrics.py
│   │   └── episode_tables.py
│   └── utils/
│       ├── seed.py
│       ├── device.py
│       ├── checkpoint.py
│       └── config.py
└── tests/
    ├── test_rules_schema.py
    ├── test_rule_features.py
    ├── test_reward_examples.py
    ├── test_env_api.py
    ├── test_env_determinism.py
    ├── test_smoke_train.py
    └── test_terminal_cadence.py
```

---

## 6. Why this layout

### `assets/rules/`

This is for **spec artifacts**, not Python code.

It contains:

- the human-readable restatement of the book rules
- the machine-readable JSON rule spec
- a JSON schema for validation
- golden examples used as reference tests

### `rlcp/rules/`

This is for deterministic rule evaluation code.

It should contain:

- feature extractors
- hard-constraint detectors
- soft-rule scorers
- explanation utilities

### `rlcp/reward/`

This layer converts raw features into structured scoring output.
It should know about:

- weighted aggregation
- terminal rewards
- edge weighting
- score breakdowns

### `rlcp/envs/`

This layer should be thin.
It mostly:

- stores episode state
- enforces action legality
- invokes the evaluator
- obeys the Gymnasium API

### `rlcp/algos/` and `scripts/`

The first trainer implementations should remain explicit and readable.
Do not hide the entire research loop inside a large abstraction too early.

The `scripts/` directory is deliberately where the first actual loops live.
That is the place where algorithm experiments stay concrete.

---

## 7. What each important file should do

### `assets/rules/tonal_counterpoint_rules.md`

Human-readable source of truth.

Each rule should record:

- `rule_id`
- prose statement from the source
- your restatement
- scope: `edge`, `two_step`, `terminal`, `trajectory`, etc.
- hard vs soft
- machine inputs required
- ambiguities
- pass/fail examples

### `assets/rules/tonal_counterpoint_rules.json`

Machine-readable rule spec.
The code should load this, not parse prose directly.

### `rlcp/rules/features.py`

Pure feature functions, e.g.

```python
melodic_interval(prev_note, next_note)
parallel_perfect(prev_chord, next_chord, voice_i, voice_j)
voice_crossing(prev_chord, next_chord)
approaches_final_cadence(context)
```

No weights here.
No policy knowledge here.

### `rlcp/reward/reward_fn.py`

Turns feature results into structured scoring output, e.g.

```python
{
    "total": ...,
    "components": {...},
    "violations": [...],
    "by_voice": {...},
}
```

### `rlcp/envs/counterpoint_env.py`

Owns:

- `reset()`
- `step(action)`
- observation construction
- legality checks / masking
- terminal logic
- diagnostics in `info`

### `scripts/train_reinforce.py`

The first explicit training script should:

- instantiate env
- instantiate policy
- roll out episodes
- compute returns
- optimize
- log metrics
- save checkpoints

This script should be readable before it is elegant.

---

## 8. The dev workflow I would use today

### 8.1 Bootstrap the project

Use `uv` with `pyproject.toml` and a checked-in `uv.lock`.

Representative start:

```bash
uv init
uv add torch gymnasium torchrl tensordict numpy scipy pyyaml pydantic pytest ruff tensorboard matplotlib rich
```

Then commit immediately.

---

### 8.2 First make the rule engine, not the learner

The first executable should be something like:

```bash
uv run python scripts/validate_rules.py
```

This should:

- load the JSON rule spec
- validate it against the schema
- run golden examples
- print a per-rule pass/fail summary

Until this works, there is no trustworthy RL system.

---

### 8.3 Then make the reward evaluator standalone

```bash
uv run python scripts/score_trajectory.py assets/rules/golden_examples/good_cadence_001.json
```

This should emit:

- total score
- per-step score
- per-rule contributions
- implicated voices
- diagnostic explanations

You need to trust this before training.

---

### 8.4 Then make the environment smoke test

```bash
uv run python scripts/smoke_env.py
```

This should:

- instantiate the environment
- call `reset()`
- sample or choose legal actions
- step through a short episode
- assert shape and type sanity
- print the `info` payload

---

### 8.5 Only then write the first trainer

Start with a small explicit trainer, e.g.

```bash
uv run python scripts/train_reinforce.py --config configs/exp/debug.yaml
```

Then later:

```bash
uv run python scripts/train_ppo.py --config configs/exp/debug.yaml
```

The first likely failures are not “PPO theory failures.”
They are more like:

- reward weirdness
- masking weirdness
- termination weirdness
- observation leakage
- invalid incentive structure around cadence

These are easiest to debug in explicit scripts.

---

### 8.6 Add componentized RL tooling only when it pays rent

A component system like TorchRL becomes worth it when:

- rollout code is getting annoying
- replay/storage utilities are worth standardizing
- transforms / collectors / modules are stabilizing

Recommended progression:

1. raw scripts + Gymnasium env
2. reusable PyTorch modules
3. optional TorchRL collectors / buffers
4. only later, consider heavier scaling infrastructure

---

## 9. The tests I would insist on

Before any serious training:

```bash
uv run pytest -q
```

These tests should exist.

### `test_rules_schema.py`

The JSON rule spec loads and validates.

### `test_rule_features.py`

Known voiceleadings trigger exactly the intended features.

### `test_reward_examples.py`

Golden trajectories score as expected.

### `test_env_api.py`

The environment obeys the Gymnasium reset / step contract.

### `test_env_determinism.py`

Given a fixed seed and fixed action sequence, the rollout is reproducible.

### `test_smoke_train.py`

A tiny training run completes without NaNs and writes at least one checkpoint.

This is not about final quality. It is about executability.

---

## 10. Daily working loop

### When changing rules

```bash
uv run python scripts/validate_rules.py
uv run python scripts/score_trajectory.py assets/rules/golden_examples/bad_parallel_001.json
uv run pytest -q tests/test_rules_schema.py tests/test_rule_features.py tests/test_reward_examples.py
```

### When changing environment logic

```bash
uv run python scripts/smoke_env.py
uv run pytest -q tests/test_env_api.py tests/test_env_determinism.py
```

### When changing policy / trainer logic

```bash
uv run python scripts/train_reinforce.py --config configs/exp/debug.yaml
uv run pytest -q tests/test_smoke_train.py
```

### When running an actual experiment

```bash
uv run python scripts/train_ppo.py --config configs/exp/small.yaml
uv run python scripts/eval_policy.py --checkpoint checkpoints/latest.pt
uv run python scripts/sample_episode.py --checkpoint checkpoints/latest.pt
```

---

## 11. What I would not do

I would not:

- hide training inside a generic trainer class too early
- mix prose rule extraction with evaluator code
- put weights directly into feature extractors
- make notebooks the only place evaluation exists
- start with distributed infrastructure
- start with a giant monolithic RL framework abstraction

For this problem, the highest-value artifact is the deterministic musical evaluator.
The learner is downstream of that.

---

## 12. The first seven commits I would make

1. `init uv project`
2. `add rule markdown, json schema, and golden examples`
3. `implement rule loader and schema validation`
4. `implement raw musical feature extractors`
5. `implement reward evaluator and explanation objects`
6. `implement gymnasium env smoke test`
7. `implement first explicit reinforce trainer`

This is the order I would trust.

---

## 13. Immediate next architectural question

The next important design decision is not repository tooling.
It is the **action representation**.

In this project, the critical next fork is:

- **direct next-chord choice in the graph**, or
- **factorized per-voice move**, later projected back into the legal ordered chord space

That decision will affect:

- action-space size
- legality checking
- masking strategy
- model architecture
- credit assignment
- how naturally the musical rules attach to decisions

---

## 14. Summary recommendation

For this project, start in the **custom PyTorch research repo** lane, not the large-framework lane.

Concretely:

- use Gymnasium-style environment contracts
- keep the first trainers explicit and script-like
- treat rule extraction / reward evaluation as the most important subsystem
- make musical correctness testable before optimizing policy performance
- delay scale/distribution concerns until the environment and evaluator are stable

That gives the next engineer a repo that is:

- inspectable
- testable
- modular where it matters
- not prematurely abstracted
- aligned with the actual difficulty of the counterpoint problem

