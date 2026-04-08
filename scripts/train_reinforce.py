"""Tiny explicit REINFORCE training entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.algos.reinforce import run_reinforce_episode
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.models.policy import SymbolicChordEncoder, TransformerStepDeltaPolicy
from rl_counterpoint.reward.black_box import ConstantReward


class DummyTextEmbedder:
    """Deterministic local embedder for the first training smoke path."""

    def embed_text(self, text: str) -> torch.Tensor:
        base = float(len(text))
        return torch.tensor([base, base + 1.0, base + 2.0, base + 3.0], dtype=torch.float32)


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
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for episode_index in range(3):
        stats = run_reinforce_episode(
            env,
            policy=policy,
            encoder=encoder,
            optimizer=optimizer,
            gamma=0.99,
            context_measures=3,
            seed=episode_index,
        )
        print(f"episode {episode_index} return: {stats.episode_return}")
        print(f"episode {episode_index} length: {stats.episode_length}")
        print(f"episode {episode_index} loss: {stats.loss}")


if __name__ == "__main__":
    main()
