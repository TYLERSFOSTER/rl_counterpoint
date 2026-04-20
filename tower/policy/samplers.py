"""Sampler contracts and scripted sampler helpers for tower rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Mapping, TypeVar

import torch

from tower.policy.base import RankPolicy
from tower.state_action import TowerAction, TowerState, rank_of_state
from tower.window import TowerWindow

ChoiceT = TypeVar("ChoiceT")


@dataclass(frozen=True)
class SamplerResult(Generic[ChoiceT]):
    """One policy-facing sampler choice plus optional probability metadata."""

    choice: ChoiceT
    logprob: float | torch.Tensor | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logprob is None or isinstance(self.logprob, float):
            return
        if isinstance(self.logprob, torch.Tensor):
            if self.logprob.ndim != 0:
                raise ValueError("logprob tensor must be scalar")
            return
        raise TypeError("logprob must be a float, scalar tensor, or None")


@dataclass
class ScriptedSampler(Generic[ChoiceT]):
    """Deterministic sampler that emits a fixed sequence of choices."""

    script: tuple[ChoiceT | SamplerResult[ChoiceT], ...]
    exhausted_message: str = "scripted sampler is exhausted"
    _index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.script, tuple):
            raise TypeError("script must be a tuple")

    def __call__(self, *args: object, **kwargs: object) -> SamplerResult[ChoiceT]:
        """Return the next scripted result, ignoring policy context arguments."""
        if self._index >= len(self.script):
            raise RuntimeError(self.exhausted_message)

        item = self.script[self._index]
        self._index += 1
        if isinstance(item, SamplerResult):
            return item
        return SamplerResult(choice=item)

    @property
    def remaining(self) -> int:
        """Return how many scripted entries have not yet been sampled."""
        return len(self.script) - self._index


def scripted_result(
    choice: ChoiceT,
    *,
    logprob: float | torch.Tensor | None = None,
    diagnostics: Mapping[str, object] | None = None,
) -> SamplerResult[ChoiceT]:
    """Build a sampler result for scripted tests and deterministic policies."""
    return SamplerResult(
        choice=choice,
        logprob=logprob,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def sample_active_choice_from_policy(
    *,
    policy: RankPolicy,
    state: TowerState,
    window: TowerWindow,
    active_choices: tuple[int, ...],
    generator: torch.Generator | None = None,
) -> SamplerResult[int]:
    """Sample an active coordinate choice from rank-local policy logits."""
    if not active_choices:
        raise ValueError("active_choices must not be empty")
    state_rank = rank_of_state(state)
    if policy.rank != state_rank:
        raise ValueError("policy rank must match state rank")

    output = policy(state=state, window=window)
    logits = output.logits
    if logits.ndim != 1:
        raise ValueError("policy logits must be a vector")
    if logits.shape[0] != len(active_choices):
        raise ValueError("policy logits length must match active_choices")

    probabilities = torch.softmax(logits, dim=0)
    selected_index_tensor = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )
    selected_index = int(selected_index_tensor.item())
    logprob = torch.log_softmax(logits, dim=0)[selected_index]

    return SamplerResult(
        choice=active_choices[selected_index],
        logprob=logprob,
        diagnostics={
            "selected_index": selected_index,
            "active_choices": active_choices,
            "policy": output.diagnostics,
        },
    )


def sample_parent_top_m_from_policy(
    *,
    policy: RankPolicy,
    state: TowerState,
    window: TowerWindow,
    parent_actions: tuple[TowerAction, ...],
    top_m: int = 1,
    generator: torch.Generator | None = None,
) -> SamplerResult[TowerAction]:
    """Sample a frozen-parent action from the top-m policy logits."""
    if not parent_actions:
        raise ValueError("parent_actions must not be empty")
    if top_m < 1:
        raise ValueError("top_m must be at least 1")
    state_rank = rank_of_state(state)
    if policy.rank != state_rank:
        raise ValueError("policy rank must match state rank")

    output = policy(state=state, window=window)
    logits = output.logits
    if logits.ndim != 1:
        raise ValueError("policy logits must be a vector")
    if logits.shape[0] != len(parent_actions):
        raise ValueError("policy logits length must match parent_actions")

    effective_top_m = min(top_m, len(parent_actions))
    top = torch.topk(logits.detach(), k=effective_top_m)
    if effective_top_m == 1:
        selected_top_position = 0
    else:
        selected_top_position = int(
            torch.randint(
                low=0,
                high=effective_top_m,
                size=(1,),
                generator=generator,
            ).item()
        )

    selected_index = int(top.indices[selected_top_position].item())
    diagnostic_logprobs = torch.log_softmax(logits.detach(), dim=0)

    return SamplerResult(
        choice=parent_actions[selected_index],
        logprob=diagnostic_logprobs[selected_index],
        diagnostics={
            "selected_index": selected_index,
            "top_indices": tuple(int(index.item()) for index in top.indices),
            "top_m": effective_top_m,
            "parent_actions": parent_actions,
            "policy": output.diagnostics,
        },
    )
