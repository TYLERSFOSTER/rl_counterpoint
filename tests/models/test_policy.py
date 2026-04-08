"""Tests for the first StepDelta policy contract."""

from __future__ import annotations

import pytest
import torch

from rl_counterpoint.models.policy import StepDeltaPolicy, observation_to_tensor


def test_observation_to_tensor_preserves_chord_state_values() -> None:
    """A raw ChordState is converted to a float tensor in voice order."""
    tensor = observation_to_tensor((3, 6, 10))

    assert tensor.dtype == torch.float32
    assert tensor.shape == (3,)
    assert torch.equal(tensor, torch.tensor([3.0, 6.0, 10.0]))


def test_policy_returns_logits_for_unbatched_observation() -> None:
    """The policy maps one observation to one logits vector."""
    policy = StepDeltaPolicy(observation_dim=2, action_dim=8, hidden_dim=16)
    observation = observation_to_tensor((3, 6))

    logits = policy(observation)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()


def test_policy_returns_logits_for_batched_observations() -> None:
    """The policy also accepts a batch of chord-state observations."""
    policy = StepDeltaPolicy(observation_dim=2, action_dim=8, hidden_dim=16)
    observation_batch = torch.tensor([[3.0, 6.0], [4.0, 7.0]])

    logits = policy(observation_batch)

    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()


def test_policy_rejects_invalid_constructor_dimensions() -> None:
    """Observation, action, and hidden dimensions must all be positive."""
    with pytest.raises(ValueError, match="observation_dim must be at least 1"):
        StepDeltaPolicy(observation_dim=0, action_dim=8)

    with pytest.raises(ValueError, match="action_dim must be at least 1"):
        StepDeltaPolicy(observation_dim=2, action_dim=0)

    with pytest.raises(ValueError, match="hidden_dim must be at least 1"):
        StepDeltaPolicy(observation_dim=2, action_dim=8, hidden_dim=0)


def test_policy_rejects_wrong_observation_shapes() -> None:
    """Forward shape checks protect the fixed observation contract."""
    policy = StepDeltaPolicy(observation_dim=2, action_dim=8)

    with pytest.raises(ValueError, match="unbatched observation has wrong shape"):
        policy(torch.tensor([3.0, 6.0, 10.0]))

    with pytest.raises(ValueError, match="batched observation has wrong shape"):
        policy(torch.tensor([[3.0, 6.0, 10.0]]))

    with pytest.raises(ValueError, match="observation tensor must be rank 1 or 2"):
        policy(torch.tensor([[[3.0, 6.0]]]))
