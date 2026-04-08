"""Smoke check for one policy-driven rollout episode."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from rl_counterpoint.algos.rollout import PolicyStepRecord, collect_policy_episode
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.models.policy import (
    SymbolicChordEncoder,
    TransformerStepDeltaPolicy,
    timed_chord_window_to_strings,
)
from rl_counterpoint.reward.black_box import ConstantReward


class DummyTextEmbedder:
    """Deterministic local embedder for rollout smoke checks."""

    def embed_text(self, text: str) -> torch.Tensor:
        base = float(len(text))
        return torch.tensor([base, base + 1.0, base + 2.0, base + 3.0], dtype=torch.float32)


def print_step_record(index: int, step: PolicyStepRecord) -> None:
    """Print a compact human-readable summary of one policy-driven rollout step."""
    legal_action_count = sum(step.action_mask)
    window_strings = timed_chord_window_to_strings(step.timed_window)
    print(f"step {index} observation: {step.observation}")
    print(f"step {index} window: {window_strings}")
    print(f"step {index} action_index: {step.action_index}")
    print(f"step {index} step_delta: {step.step_delta}")
    print(f"step {index} legal_action_count: {legal_action_count}")
    print(f"step {index} logits_dim: {step.logits.shape[0]}")
    print(f"step {index} reward: {step.reward}")
    print(f"step {index} next_observation: {step.next_observation}")
    print(f"step {index} terminated: {step.terminated}")
    print(f"step {index} truncated: {step.truncated}")


def main() -> None:
    env = CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=ConstantReward(reward=1.0),
        initial_state=(3, 6),
        max_steps=4,
        measure_size=4,
        max_step_size=2,
        invalid_action_penalty=-1.0,
    )
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())
    policy = TransformerStepDeltaPolicy(
        embedding_dim=4,
        action_dim=len(env.action_space),
        max_window_len=env.measure_size * 3,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )

    trajectory = collect_policy_episode(
        env,
        policy=policy,
        encoder=encoder,
        context_measures=3,
        seed=0,
    )
    print(f"episode length: {len(trajectory)}")
    for index, step in enumerate(trajectory):
        print_step_record(index, step)


if __name__ == "__main__":
    main()
