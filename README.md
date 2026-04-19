# RL Counterpoint

`rl_counterpoint` is a reinforcement learning project for counterpoint and voice-leading generation over a constrained graph of ordered chord states.

<figure
style="
    margin: 0;
    width: 100%;
    box-sizing: border-box;
"
>
<p align="center">
  <picture>
    <source srcset="assets/images/hrl_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="assets/images/hrl_light.png" media="(prefers-color-scheme: light)">
    <img src="assets/MPK_Mini.jpg" alt="MIDIControl001" width="500">
  </picture>
<figcaption style="margin-top: 8px; text-align: center; font-size: 0.95em;">
    Figure 2. Early experiments with API-call-based image continuation
</figcaption>
</figure><br>

## Overview

The repository includes:

- a graph-based state/action space for valid chord transitions
- a Gymnasium-style environment for sequential decision-making
- reward interfaces and experimental reward implementations
- a transformer-style policy path over timed chord-history windows
- rollout and REINFORCE training utilities
- smoke scripts for fast manual inspection
- tests for graph, environment, reward, model, rollout, and training behavior

## Installation

This project uses `uv` for environment and dependency management.

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

## Training

Run a training session:

```bash
uv run python scripts/train_reinforce.py
```

The training script writes outputs to `artifacts/train_reinforce/`, including:

- JSONL metrics
- PyTorch checkpoints
- a MIDI file for one example evaluation episode

## Project Structure

```text
rl_counterpoint/
├── rl_counterpoint/
│   ├── algos/
│   ├── envs/
│   ├── graph/
│   ├── models/
│   ├── music/
│   └── reward/
├── scripts/
├── tests/
├── docs/
└── assets/
```

Key directories:

- `rl_counterpoint/graph/`: graph specification, node validity, actions, edge legality
- `rl_counterpoint/envs/`: environment, observation windows, truncation helpers
- `rl_counterpoint/reward/`: reward protocol and reward implementations
- `rl_counterpoint/models/`: symbolic encoding and policy models
- `rl_counterpoint/algos/`: rollout and REINFORCE helpers
- `scripts/`: smoke scripts and training entrypoints
- `tests/`: automated test coverage
- `docs/`: design notes, continuity records, and collaboration directives

## Development Notes

- Python version is defined in `.python-version`
- dependencies are defined in `pyproject.toml`
- generated training outputs under `artifacts/` are ignored by git

## Status

This is an active research codebase. Core infrastructure is in place, but reward design, training behavior, and evaluation workflows are still evolving.
