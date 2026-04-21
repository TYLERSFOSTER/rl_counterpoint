"""Sampler contracts and scripted sampler helpers for tower rollout."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Mapping, TypeVar, cast

import torch

from tower.policy.base import PolicyOutput, RankPolicy
from tower.policy.observation import encode_tower_window
from tower.policy.transformer import TowerTransformerPolicy
from tower.state_action import TowerAction, TowerState, rank_of_state
from tower.window import TowerWindow, frontier_state

ChoiceT = TypeVar("ChoiceT")
LegacyRankPolicyCall = Callable[..., PolicyOutput]


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
    window: TowerWindow,
    active_choices: tuple[int, ...],
    state: TowerState | None = None,
    measure_size: int = 4,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    max_step_size: int | None = None,
    generator: torch.Generator | None = None,
) -> SamplerResult[int]:
    """Sample an active coordinate choice from rank-local policy logits."""
    if not active_choices:
        raise ValueError("active_choices must not be empty")
    current_state = _frontier_for_policy(
        policy=policy,
        window=window,
        state=state,
    )

    output = _policy_output(
        policy=policy,
        window=window,
        state=current_state,
        measure_size=measure_size,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        max_step_size=max_step_size,
    )
    logits = output.logits
    if logits.ndim != 1:
        raise ValueError("policy logits must be a vector")
    choice_logits = _active_choice_logits(
        logits=logits,
        active_choices=active_choices,
        policy_rank=policy.rank,
        max_step_size=max_step_size,
    )

    probabilities = torch.softmax(choice_logits, dim=0)
    selected_index_tensor = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )
    selected_index = int(selected_index_tensor.item())
    logprob = torch.log_softmax(choice_logits, dim=0)[selected_index]

    return SamplerResult(
        choice=active_choices[selected_index],
        logprob=logprob,
        diagnostics={
            "selected_index": selected_index,
            "active_choices": active_choices,
            "frontier_state": current_state,
            "policy": output.diagnostics,
        },
    )


def sample_parent_top_m_from_policy(
    *,
    policy: RankPolicy,
    window: TowerWindow,
    parent_actions: tuple[TowerAction, ...],
    state: TowerState | None = None,
    measure_size: int = 4,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    max_step_size: int | None = None,
    top_m: int = 1,
    generator: torch.Generator | None = None,
) -> SamplerResult[TowerAction]:
    """Sample a frozen-parent action from the top-m policy logits."""
    if not parent_actions:
        raise ValueError("parent_actions must not be empty")
    if top_m < 1:
        raise ValueError("top_m must be at least 1")
    current_state = _frontier_for_policy(
        policy=policy,
        window=window,
        state=state,
    )

    output = _policy_output(
        policy=policy,
        window=window,
        state=current_state,
        measure_size=measure_size,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        max_step_size=max_step_size,
    )
    logits = output.logits
    if logits.ndim != 1:
        raise ValueError("policy logits must be a vector")
    action_logits = _parent_action_logits(
        logits=logits,
        parent_actions=parent_actions,
        policy_rank=policy.rank,
        max_step_size=max_step_size,
    )

    effective_top_m = min(top_m, len(parent_actions))
    top = torch.topk(action_logits.detach(), k=effective_top_m)
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
    diagnostic_logprobs = torch.log_softmax(action_logits.detach(), dim=0)

    return SamplerResult(
        choice=parent_actions[selected_index],
        logprob=diagnostic_logprobs[selected_index],
        diagnostics={
            "selected_index": selected_index,
            "top_indices": tuple(int(index.item()) for index in top.indices),
            "top_m": effective_top_m,
            "parent_actions": parent_actions,
            "frontier_state": current_state,
            "policy": output.diagnostics,
        },
    )


def _frontier_for_policy(
    *,
    policy: RankPolicy,
    window: TowerWindow,
    state: TowerState | None,
) -> TowerState:
    current_state = frontier_state(window)
    state_rank = rank_of_state(current_state)
    if policy.rank != state_rank:
        raise ValueError("policy rank must match state rank")
    if state is not None and state != current_state:
        raise ValueError("state must match window frontier")
    return current_state


def _policy_output(
    *,
    policy: RankPolicy,
    window: TowerWindow,
    state: TowerState,
    measure_size: int,
    key_pitch_class: int | None,
    target_root_octave: int | None,
    max_step_size: int | None,
) -> PolicyOutput:
    if isinstance(policy, TowerTransformerPolicy):
        return policy(
            encode_tower_window(
                window=window,
                measure_size=measure_size,
                rank=policy.rank,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
                max_step_size=max_step_size,
            )
        )

    legacy_policy = cast(LegacyRankPolicyCall, policy)
    return legacy_policy(state=state, window=window)


def _active_choice_logits(
    *,
    logits: torch.Tensor,
    active_choices: tuple[int, ...],
    policy_rank: int,
    max_step_size: int | None,
) -> torch.Tensor:
    if logits.shape[0] == len(active_choices):
        return logits
    if max_step_size is None:
        raise ValueError("policy logits length must match active_choices")

    indices = tuple(
        _active_choice_index(
            choice=choice,
            policy_rank=policy_rank,
            max_step_size=max_step_size,
        )
        for choice in active_choices
    )
    if max(indices) >= logits.shape[0]:
        raise ValueError("policy logits length must cover active_choices")
    return logits[torch.tensor(indices, dtype=torch.long, device=logits.device)]


def _parent_action_logits(
    *,
    logits: torch.Tensor,
    parent_actions: tuple[TowerAction, ...],
    policy_rank: int,
    max_step_size: int | None,
) -> torch.Tensor:
    if logits.shape[0] == len(parent_actions):
        return logits
    if max_step_size is None:
        raise ValueError("policy logits length must match parent_actions")
    if policy_rank != 1:
        raise ValueError("full-lattice parent action logits require rank 1")

    indices = tuple(
        _rank1_nonzero_action_index(
            choice=action[0],
            max_step_size=max_step_size,
        )
        for action in parent_actions
    )
    if max(indices) >= logits.shape[0]:
        raise ValueError("policy logits length must cover parent_actions")
    return logits[torch.tensor(indices, dtype=torch.long, device=logits.device)]


def _active_choice_index(
    *,
    choice: int,
    policy_rank: int,
    max_step_size: int,
) -> int:
    if policy_rank == 1:
        return _rank1_nonzero_action_index(
            choice=choice,
            max_step_size=max_step_size,
        )
    if choice < -max_step_size or choice > max_step_size:
        raise ValueError("active choice is outside max_step_size")
    return choice + max_step_size


def _rank1_nonzero_action_index(
    *,
    choice: int,
    max_step_size: int,
) -> int:
    if choice == 0:
        raise ValueError("rank-1 full action lattice does not include zero")
    if choice < -max_step_size or choice > max_step_size:
        raise ValueError("action choice is outside max_step_size")
    if choice < 0:
        return choice + max_step_size
    return choice + max_step_size - 1
