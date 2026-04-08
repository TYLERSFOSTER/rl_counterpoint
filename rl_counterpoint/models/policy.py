"""Policy modules for fixed-lattice StepDelta action selection."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rl_counterpoint.graph.state_space import ChordState


def observation_to_tensor(observation: ChordState) -> Tensor:
    """Convert a raw ChordState observation into a float tensor."""
    return torch.tensor(observation, dtype=torch.float32)


class StepDeltaPolicy(nn.Module):
    """Tiny policy that maps a chord-state observation to action logits."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if observation_dim < 1:
            raise ValueError("observation_dim must be at least 1")
        if action_dim < 1:
            raise ValueError("action_dim must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """Return logits over the fixed StepDelta action lattice."""
        if observation.ndim == 1:
            if observation.shape[0] != self.observation_dim:
                raise ValueError("unbatched observation has wrong shape")
            return self.network(observation)

        if observation.ndim == 2:
            if observation.shape[1] != self.observation_dim:
                raise ValueError("batched observation has wrong shape")
            return self.network(observation)

        raise ValueError("observation tensor must be rank 1 or 2")
