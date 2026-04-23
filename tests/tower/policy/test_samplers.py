"""Tests for tower sampler contracts."""

from __future__ import annotations

import pytest
import torch

from tower.policy.base import PolicyOutput
from tower.policy.samplers import (
    SamplerResult,
    ScriptedSampler,
    sample_active_choice_from_policy,
    sample_parent_top_m_from_policy,
    scripted_result,
)
from tower.policy.transformer import (
    TowerTransformerPolicy,
    TowerTransformerPolicyConfig,
)
from tower.window import TowerWindow, build_window


class DummyPolicy:
    def __init__(self, *, rank: int, logits: torch.Tensor) -> None:
        self.rank = rank
        self.logits = logits

    def __call__(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert window.valid_mask[-1]
        return PolicyOutput(
            logits=self.logits,
            diagnostics={"state": state},
        )


def make_rank_1_window() -> TowerWindow:
    return build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )


def make_transformer_policy(
    *,
    rank: int,
    action_dim: int,
    input_feature_dim: int | None = None,
    max_window_len: int = 4,
) -> TowerTransformerPolicy:
    return TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=rank,
            input_feature_dim=rank + 5
            if input_feature_dim is None
            else input_feature_dim,
            action_dim=action_dim,
            max_window_len=max_window_len,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    )


def test_sampler_result_stores_choice_logprob_and_diagnostics() -> None:
    result = SamplerResult(
        choice=(1,),
        logprob=None,
        diagnostics={"kind": "parent"},
    )

    assert result.choice == (1,)
    assert result.logprob is None
    assert result.diagnostics == {"kind": "parent"}


def test_sampler_result_rejects_non_float_logprob() -> None:
    with pytest.raises(TypeError, match="logprob must be a float, scalar tensor"):
        SamplerResult(choice=1, logprob=0)  # type: ignore[arg-type]


def test_sampler_result_accepts_scalar_tensor_logprob() -> None:
    logprob = torch.tensor(-0.25, requires_grad=True)

    result = SamplerResult(choice=1, logprob=logprob)

    assert result.logprob is logprob


def test_sampler_result_rejects_vector_tensor_logprob() -> None:
    with pytest.raises(ValueError, match="logprob tensor must be scalar"):
        SamplerResult(choice=1, logprob=torch.tensor([-0.25, -0.5]))


def test_scripted_sampler_returns_plain_choices_in_order_with_none_logprob() -> None:
    sampler = ScriptedSampler(script=((1,), (-1,)))

    first = sampler()
    second = sampler()

    assert first == SamplerResult(choice=(1,))
    assert first.logprob is None
    assert second == SamplerResult(choice=(-1,))
    assert second.logprob is None
    assert sampler.remaining == 0


def test_scripted_sampler_returns_explicit_results_in_order() -> None:
    sampler = ScriptedSampler(
        script=(
            scripted_result(1, diagnostics={"kind": "active"}),
            scripted_result(-1, logprob=-0.25),
        )
    )

    first = sampler()
    second = sampler()

    assert first.choice == 1
    assert first.logprob is None
    assert first.diagnostics == {"kind": "active"}
    assert second.choice == -1
    assert second.logprob == -0.25


def test_scripted_sampler_accepts_policy_context_arguments() -> None:
    sampler = ScriptedSampler(script=(2,))

    result = sampler(
        state=(60, 64),
        window="ignored",
        active_choices=(-1, 0, 1),
    )

    assert result == SamplerResult(choice=2)


def test_scripted_sampler_allows_invalid_active_proposals() -> None:
    sampler = ScriptedSampler(script=(99,))

    result = sampler(active_choices=(-1, 0, 1))

    assert result.choice == 99


def test_scripted_sampler_raises_when_exhausted() -> None:
    sampler = ScriptedSampler(script=(1,))

    assert sampler().choice == 1
    with pytest.raises(RuntimeError, match="scripted sampler is exhausted"):
        sampler()


def test_scripted_sampler_requires_tuple_script() -> None:
    with pytest.raises(TypeError, match="script must be a tuple"):
        ScriptedSampler(script=[1, 2])  # type: ignore[arg-type]


def test_sample_active_choice_from_policy_returns_choice_from_active_choices() -> None:
    generator = torch.Generator().manual_seed(0)
    policy = DummyPolicy(
        rank=1,
        logits=torch.tensor([0.0, 8.0, -8.0], requires_grad=True),
    )

    result = sample_active_choice_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        active_choices=(-1, 0, 1),
        generator=generator,
    )

    assert result.choice in (-1, 0, 1)
    assert result.choice == 0
    assert isinstance(result.logprob, torch.Tensor)
    assert result.logprob.ndim == 0
    assert result.diagnostics["active_choices"] == (-1, 0, 1)
    assert result.diagnostics["frontier_state"] == (60,)
    assert result.diagnostics["policy"] == {"state": (60,)}


def test_sample_active_choice_from_policy_derives_frontier_state() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 8.0, -8.0]))

    result = sample_active_choice_from_policy(
        policy=policy,
        window=make_rank_1_window(),
        active_choices=(-1, 0, 1),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice == 0
    assert result.diagnostics["frontier_state"] == (60,)
    assert result.diagnostics["policy"] == {"state": (60,)}


def test_sample_active_choice_from_policy_rejects_state_window_mismatch() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match="state must match window frontier"):
        sample_active_choice_from_policy(
            policy=policy,
            state=(61,),
            window=make_rank_1_window(),
            active_choices=(-1, 1),
        )


def test_sample_active_choice_from_policy_supports_transformer_policy() -> None:
    policy = make_transformer_policy(rank=1, action_dim=3)

    result = sample_active_choice_from_policy(
        policy=policy,  # type: ignore[arg-type]
        window=make_rank_1_window(),
        active_choices=(-1, 0, 1),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice in {-1, 0, 1}
    assert isinstance(result.logprob, torch.Tensor)
    assert result.logprob.requires_grad
    assert result.diagnostics["frontier_state"] == (60,)
    assert result.diagnostics["policy"]["policy"] == "tower_transformer"


def test_sample_active_choice_from_policy_conditions_transformer_on_key_and_target() -> None:
    policy = make_transformer_policy(rank=1, input_feature_dim=8, action_dim=3)

    result = sample_active_choice_from_policy(
        policy=policy,  # type: ignore[arg-type]
        window=make_rank_1_window(),
        active_choices=(-1, 0, 1),
        key_pitch_class=2,
        target_root_octave=5,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice in {-1, 0, 1}
    assert result.diagnostics["policy"]["context"]["key_pitch_class"] == 2
    assert result.diagnostics["policy"]["context"]["target_root_octave"] == 5


def test_sample_active_choice_from_policy_is_reproducible_with_generator() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 0.0, 0.0]))

    first = sample_active_choice_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        active_choices=(-1, 0, 1),
        generator=torch.Generator().manual_seed(123),
    )
    second = sample_active_choice_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        active_choices=(-1, 0, 1),
        generator=torch.Generator().manual_seed(123),
    )

    assert first.choice == second.choice
    assert first.diagnostics["selected_index"] == second.diagnostics["selected_index"]


def test_sample_active_choice_from_policy_logprob_backpropagates_to_logits() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
    policy = DummyPolicy(rank=1, logits=logits)

    result = sample_active_choice_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        active_choices=(-1, 0, 1),
        generator=torch.Generator().manual_seed(0),
    )
    assert isinstance(result.logprob, torch.Tensor)
    (-result.logprob).backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0


def test_sample_active_choice_from_policy_accepts_sampling_temperature_and_uniform_mix() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([8.0, -8.0]))

    result = sample_active_choice_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        active_choices=(-1, 1),
        temperature=2.0,
        uniform_mix=0.25,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice in (-1, 1)
    assert result.diagnostics["temperature"] == 2.0
    assert result.diagnostics["uniform_mix"] == 0.25


@pytest.mark.parametrize(
    ("temperature", "uniform_mix", "message"),
    (
        (0.0, 0.0, "temperature must be positive"),
        (1.0, -0.1, "uniform_mix must be in \\[0.0, 1.0\\]"),
        (1.0, 1.1, "uniform_mix must be in \\[0.0, 1.0\\]"),
    ),
)
def test_sample_active_choice_from_policy_rejects_invalid_sampling_controls(
    temperature: float,
    uniform_mix: float,
    message: str,
) -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match=message):
        sample_active_choice_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            active_choices=(-1, 1),
            temperature=temperature,
            uniform_mix=uniform_mix,
        )


def test_sample_active_choice_from_policy_masks_full_rank1_action_lattice() -> None:
    logits = torch.tensor([0.0, 8.0, -8.0, 0.0], requires_grad=True)
    policy = DummyPolicy(rank=1, logits=logits)

    result = sample_active_choice_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        active_choices=(-1, 1),
        max_step_size=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice == -1
    assert isinstance(result.logprob, torch.Tensor)
    (-result.logprob).backward()

    assert logits.grad is not None
    assert logits.grad[0].item() == 0.0
    assert torch.count_nonzero(logits.grad[1:3]).item() > 0
    assert logits.grad[3].item() == 0.0


def test_sample_active_choice_from_policy_rejects_empty_choices() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([]))

    with pytest.raises(ValueError, match="active_choices must not be empty"):
        sample_active_choice_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            active_choices=(),
        )


def test_sample_active_choice_from_policy_rejects_policy_rank_mismatch() -> None:
    policy = DummyPolicy(rank=2, logits=torch.tensor([0.0]))

    with pytest.raises(ValueError, match="policy rank must match state rank"):
        sample_active_choice_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            active_choices=(0,),
        )


def test_sample_active_choice_from_policy_rejects_logits_choice_mismatch() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match="policy logits length must match"):
        sample_active_choice_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            active_choices=(-1, 0, 1),
        )


def test_sample_active_choice_from_policy_rejects_uncovered_active_choice() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match="policy logits length must cover"):
        sample_active_choice_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            active_choices=(-2, -1, 1),
            max_step_size=2,
        )


def test_sample_parent_top_m_from_policy_top_1_chooses_argmax() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 3.0, 1.0]))

    result = sample_parent_top_m_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        parent_actions=((-1,), (0,), (1,)),
        top_m=1,
    )

    assert result.choice == (0,)
    assert result.diagnostics["selected_index"] == 1
    assert result.diagnostics["top_indices"] == (1,)
    assert result.diagnostics["parent_actions"] == ((-1,), (0,), (1,))
    assert result.diagnostics["frontier_state"] == (60,)
    assert isinstance(result.logprob, torch.Tensor)
    assert result.logprob.requires_grad is False


def test_sample_parent_top_m_from_policy_supports_transformer_policy() -> None:
    policy = make_transformer_policy(rank=1, action_dim=3)

    result = sample_parent_top_m_from_policy(
        policy=policy,  # type: ignore[arg-type]
        window=make_rank_1_window(),
        parent_actions=((-1,), (0,), (1,)),
        top_m=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice in {(-1,), (0,), (1,)}
    assert isinstance(result.logprob, torch.Tensor)
    assert result.logprob.requires_grad is False
    assert result.diagnostics["frontier_state"] == (60,)
    assert result.diagnostics["policy"]["policy"] == "tower_transformer"


def test_sample_parent_top_m_from_policy_samples_only_from_top_m_actions() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 3.0, 2.0, -5.0]))

    result = sample_parent_top_m_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        parent_actions=((-2,), (-1,), (1,), (2,)),
        top_m=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice in {(-1,), (1,)}
    assert result.diagnostics["top_indices"] == (1, 2)
    assert result.diagnostics["top_m"] == 2


def test_sample_parent_top_m_from_policy_is_reproducible_with_generator() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 3.0, 2.0, 1.0]))

    first = sample_parent_top_m_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        parent_actions=((-2,), (-1,), (1,), (2,)),
        top_m=3,
        generator=torch.Generator().manual_seed(123),
    )
    second = sample_parent_top_m_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        parent_actions=((-2,), (-1,), (1,), (2,)),
        top_m=3,
        generator=torch.Generator().manual_seed(123),
    )

    assert first.choice == second.choice
    assert first.diagnostics["selected_index"] == second.diagnostics["selected_index"]


def test_sample_parent_top_m_from_policy_accepts_sampling_controls() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 3.0, 2.0, 1.0]))

    result = sample_parent_top_m_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        parent_actions=((-2,), (-1,), (1,), (2,)),
        top_m=3,
        temperature=1.75,
        uniform_mix=0.2,
        generator=torch.Generator().manual_seed(123),
    )

    assert result.choice in {(-1,), (1,), (2,)}
    assert result.diagnostics["temperature"] == 1.75
    assert result.diagnostics["uniform_mix"] == 0.2


def test_sample_parent_top_m_from_policy_clamps_top_m_to_available_actions() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 3.0]))

    result = sample_parent_top_m_from_policy(
        policy=policy,
        state=(60,),
        window=make_rank_1_window(),
        parent_actions=((-1,), (1,)),
        top_m=5,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.choice in {(-1,), (1,)}
    assert result.diagnostics["top_m"] == 2


def test_sample_parent_top_m_from_policy_rejects_empty_parent_actions() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([]))

    with pytest.raises(ValueError, match="parent_actions must not be empty"):
        sample_parent_top_m_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            parent_actions=(),
        )


def test_sample_parent_top_m_from_policy_rejects_invalid_top_m() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0]))

    with pytest.raises(ValueError, match="top_m must be at least 1"):
        sample_parent_top_m_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            parent_actions=((1,),),
            top_m=0,
        )


def test_sample_parent_top_m_from_policy_rejects_policy_rank_mismatch() -> None:
    policy = DummyPolicy(rank=2, logits=torch.tensor([0.0]))

    with pytest.raises(ValueError, match="policy rank must match state rank"):
        sample_parent_top_m_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            parent_actions=((1,),),
        )


def test_sample_parent_top_m_from_policy_rejects_logits_action_mismatch() -> None:
    policy = DummyPolicy(rank=1, logits=torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match="policy logits length must match"):
        sample_parent_top_m_from_policy(
            policy=policy,
            state=(60,),
            window=make_rank_1_window(),
            parent_actions=((1,),),
        )
