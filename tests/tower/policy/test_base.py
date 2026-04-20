"""Tests for tower policy contracts."""

from __future__ import annotations

import pytest
import torch

from tower.policy.base import (
    PolicyOutput,
    RankPolicy,
    freeze_parent_policy,
    policy_parameters_frozen,
)
from tower.window import TowerWindow, build_window


class DummyRankPolicy:
    rank = 1

    def __call__(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert state == (60,)
        assert window.valid_mask[-1]
        return PolicyOutput(
            logits=torch.tensor([0.0, 1.0, -1.0]),
            diagnostics={"kind": "dummy"},
        )


class TinyModulePolicy(torch.nn.Module):
    rank = 1

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        return PolicyOutput(logits=self.logits + float(state[0]) * 0.0)


def test_dummy_policy_satisfies_rank_policy_protocol() -> None:
    policy: RankPolicy = DummyRankPolicy()
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    output = policy(state=(60,), window=window)

    assert policy.rank == 1
    assert torch.equal(output.logits, torch.tensor([0.0, 1.0, -1.0]))
    assert output.diagnostics == {"kind": "dummy"}


def test_policy_output_accepts_tensor_logits() -> None:
    logits = torch.tensor([1.0, 2.0])

    output = PolicyOutput(logits=logits)

    assert output.logits is logits
    assert output.diagnostics == {}


def test_policy_output_preserves_diagnostics() -> None:
    output = PolicyOutput(
        logits=torch.tensor([1.0]),
        diagnostics={"rank": 1},
    )

    assert output.diagnostics == {"rank": 1}


def test_policy_output_rejects_non_tensor_logits() -> None:
    with pytest.raises(TypeError, match="logits must be a torch.Tensor"):
        PolicyOutput(logits=[1.0, 2.0])  # type: ignore[arg-type]


def test_freeze_parent_policy_sets_eval_and_freezes_params() -> None:
    policy = TinyModulePolicy()
    policy.train()

    returned = freeze_parent_policy(policy)

    assert returned is policy
    assert not policy.training
    assert policy_parameters_frozen(policy)


def test_freeze_parent_policy_is_idempotent() -> None:
    policy = TinyModulePolicy()

    freeze_parent_policy(policy)
    freeze_parent_policy(policy)

    assert policy_parameters_frozen(policy)


def test_frozen_policy_forward_still_returns_output() -> None:
    policy = freeze_parent_policy(TinyModulePolicy())
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    output = policy(state=(60,), window=window)

    assert isinstance(output, PolicyOutput)
    assert torch.equal(output.logits, torch.tensor([0.0, 1.0]))


def test_child_optimizer_step_does_not_change_frozen_parent_params() -> None:
    parent = freeze_parent_policy(TinyModulePolicy())
    child = TinyModulePolicy()
    parent_before = parent.logits.detach().clone()
    optimizer = torch.optim.SGD(child.parameters(), lr=0.1)
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    loss = child(state=(60,), window=window).logits.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.equal(parent.logits.detach(), parent_before)
    assert not torch.equal(child.logits.detach(), torch.tensor([0.0, 1.0]))
