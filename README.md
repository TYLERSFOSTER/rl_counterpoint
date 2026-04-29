# RL Counterpoint
`rl_counterpoint` is a research repo for [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) based [counterpoint](https://en.wikipedia.org/wiki/Counterpoint) generation. The long-term goal is to train an RL agent, equipped with a tranformer-driven policy function, to generate "good" counterpoint passages, and to be able to modify the flavor/feel/vibe of these voiceleading passages by modifying the agent's reward functions.

The repo currently contains two systems:

1. `rl_counterpoint/`: the legacy flat-graph project, kept as a frozen reference and baseline
2. `tower/`: the active hierarchical redesign, where training proceeds rank by rank through a tower of graph problems

Right now, `tower` is the central product in this repo.



## Key PM Engineering Insights

### **Insight 1.** *The book [Tonal Counterpoint for the 21st Century Musician](https://www.bloomsbury.com/us/tonal-counterpoint-for-the-21stcentury-musician-9781442234598/) is accidentially written for RL training pipelines.*
A

### Counterpoint is naturally a problem in [Hierarchical RL](https://arxiv.org/pdf/2506.14045)
The ***second*** key technical insigth about how to do this that we implement in the present repo is:
> The probelm of generating counterpoint passages is naturally hierarchical. For instance, 3-part voiceleading naturally reduces to an inner-voice problem *over* a simpler 2-part voice leading problem, itself natrually reducing to an upper-voice problem *over* a simpler 1-voice pedal problem.

From an engineering perspective, this means a hierarchical system of agents with interrelated, but separate policy models, trained as follows:
- rank 1: learn how to generate a good pedal line
- rank 2: add the top voice over a frozen rank-1 (pedal) scaffold
- rank 3+: add further interior voices over lower-rank scaffolds and under upper-voice.

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/images/hrl_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/images/hrl_light.png">
    <img src="assets/images/hrl_light.png" alt="Hierarchy of voiceleading ranks: rank 1 learns the pedal, rank 2 learns an added outer voice over the frozen pedal, and rank 3 learns an interior voice over the frozen lower scaffold." width="900">
  </picture>

The design lives in `tower/` and is documented in `docs/design/tower/`.

An important reward-design reference for this repo is [docs/design/tower/rank_local_reward_spec.md](/Users/foster/rl_counterpoint/docs/design/tower/rank_local_reward_spec.md), which is built from the TC21M notes in `assets/rules/tc21m_rules.md`.

The insight in using that book as a reference manual is not "copy the prose directly into code." It is to use the book as a structured source of musically meaningful reward ideas: which intervals are stable or unstable, which motions want recovery or resolution, which beat positions matter structurally, and which cadential shapes should count as successful endings. In other words, the book is valuable here as a guide for building reward terms and pruning ideas that reflect real contrapuntal practice.


## Current Status

### Legacy system: `rl_counterpoint/`

This subproject is frozen. It still matters because it provides:

- the original flat graph and environment baseline
- reward and rollout reference behavior
- legacy transformer-policy training code
- tests and scripts that still anchor older functionality

You should think of it as historical ground truth, not the active destination.

### Active system: `tower/`

This is the current build target. It already includes:

- tower graph specs and action/state conventions
- transformer policies over frontier-window observations
- artifact-backed rank-local training runs
- staged rank-1 training
- rank-2 training over accepted rank-1 parent checkpoints
- reward diagnostics, checkpoints, metrics, and MIDI artifact writing
- tests covering graph, reward, rollout, runner, scripts, and protocol behavior

Implemented slices are still intentionally incomplete from a musical point of view, but the engineering substrate is real and actively used.

## Repo Layout

```text
rl_counterpoint/
├── rl_counterpoint/          legacy flat-system code
├── tower/                    active hierarchical redesign
├── scripts/                  training, smoke, and probing entrypoints
├── tests/                    automated test coverage
├── docs/                     design docs, continuity reports, directives
├── artifacts/                generated run outputs
├── assets/                   repo media assets
├── pyproject.toml
└── README.md
```

### Active `tower/` package

```text
tower/
├── action/                   rank-local action objects and helpers
├── graph/                    graph specs, legality, and lift/projection logic
├── music/                    pitch and music-theory utilities
├── policy/                   transformer policies and samplers
├── reward/                   rank-local reward terms and factories
└── train/                    rollout, losses, checkpointing, runners
```

### Key scripts

- `scripts/tower_train.py`
  - single rank-1 training run
- `scripts/tower_train_staged.py`
  - staged rank-1 curriculum, typically coupled start/target then decoupled continuation
- `scripts/tower_train_rank2.py`
  - rank-2 training against an accepted rank-1 parent checkpoint
- `scripts/tower_reward_probe.py`
  - quick reward probing for tower slices
- `scripts/train_reinforce.py`
  - legacy flat-system training entrypoint
- `scripts/smoke_*.py`
  - small sanity scripts for the legacy baseline

## Installation

This repo uses `uv`.

```bash
uv sync
```

Run the full test suite:

```bash
uv run pytest
```

## Training Workflows

### Legacy flat-system training

```bash
uv run python scripts/train_reinforce.py
```

### Rank-1 tower training

Single run:

```bash
uv run python scripts/tower_train.py --rank 1 --episodes 1000
```

Staged curriculum:

```bash
uv run python scripts/tower_train_staged.py \
  --lineage-id my-rank1-lineage \
  --stage1-episodes 5000 \
  --stage2-episodes 5000
```

### Rank-2 tower training

This assumes the lineage already has an accepted rank-1 checkpoint.

```bash
uv run python scripts/tower_train_rank2.py \
  --lineage-id my-rank1-lineage \
  --episodes 5000
```

## Artifacts

Training runs write under `artifacts/`.

Typical tower lineage layout:

```text
artifacts/tower/<lineage-id>-stage1/rank_1/
artifacts/tower/<lineage-id>/rank_1/
artifacts/tower/<lineage-id>/rank_2/
```

Common outputs include:

- `config.json`
- `metrics.jsonl`
- `checkpoint_latest.pt`
- `reward_diagnostics.jsonl` when enabled
- `example_episode.mid` and additional final inference MIDI files
- lineage manifests and rank manifests

## Documentation

The repo has a lot of project memory in `docs/`.

Most important folders:

- `docs/design/tower/`
  - system design, rollout semantics, training protocol, reward contracts, build plans
- `docs/engineer_continuity/`
  - session handoff reports and implementation continuity
- `docs/prime_directive/`
  - operating instructions for the engineering agent working in this repo

Good starting points:

- [docs/design/tower/README.md](/Users/foster/rl_counterpoint/docs/design/tower/README.md)
- [docs/design/tower/system_design.md](/Users/foster/rl_counterpoint/docs/design/tower/system_design.md)
- [docs/design/tower/training_protocol.md](/Users/foster/rl_counterpoint/docs/design/tower/training_protocol.md)
- [docs/design/tower/post_slice_8_phase_stage_action_plan.md](/Users/foster/rl_counterpoint/docs/design/tower/post_slice_8_phase_stage_action_plan.md)

## Development Notes

- Python requirement: `>=3.13`
- dependencies live in [pyproject.toml](/Users/foster/rl_counterpoint/pyproject.toml)
- pytest uses `--import-mode=importlib` to avoid duplicate test-module name collisions
- artifact directories can get large during training; long runs often disable training reward diagnostics to keep disk usage sane

## Honest Snapshot

This is an active research codebase with real infrastructure and moving musical targets.

The old flat system still works and is useful as reference. The new `tower` system is where current implementation effort goes. Rank-1 training is much more mature than rank-2 training, and higher ranks are still ahead of us.

So the repo is not "finished," but it is no longer a sketch either. It contains a working experimental ladder from legacy baseline to hierarchical training.
