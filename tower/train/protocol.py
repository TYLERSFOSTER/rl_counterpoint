"""Training protocol helpers for tower policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from tower.action.assembly import assemble_action
from tower.graph.actions import active_lift_choices, legal_actions_for_state
from tower.graph.projection import project_state, project_window
from tower.graph.spec import TowerGraphSpec
from tower.policy.base import RankPolicy, freeze_parent_policy
from tower.policy.samplers import (
    SamplerResult,
    sample_active_choice_from_policy,
    sample_parent_top_m_from_policy,
)
from tower.reward.result import TowerRewardResult
from tower.train.checkpoint import (
    TowerArtifactPaths,
    append_rank_metrics,
    build_checkpoint_payload,
    find_accepted_parent_checkpoint,
    load_accepted_parent_checkpoint,
    record_rank_manifest_entry,
    save_latest_checkpoint,
    write_rank_config,
)
from tower.train.config import TowerRankConfig
from tower.train.losses import PolicyGradientLossResult, policy_gradient_loss
from tower.train.rollout import RewardFunction, rollout_rank1, rollout_rank2, rollout_rank3
from tower.train.trajectory import TowerTrajectory


@dataclass(frozen=True)
class TrainEpisodeResult:
    """Result of one rank-local training episode."""

    trajectory: TowerTrajectory
    loss: PolicyGradientLossResult
    metrics: Mapping[str, object]


def train_rank1_episode(
    *,
    policy: RankPolicy,
    optimizer: torch.optim.Optimizer,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    max_steps: int,
    graph_spec: TowerGraphSpec | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    sampling_temperature: float = 1.0,
    sampling_uniform_mix: float = 0.0,
    gamma: float = 1.0,
    normalize_returns: bool = False,
    generator: torch.Generator | None = None,
) -> TrainEpisodeResult:
    """Run one rank-1 rollout and optimizer step."""
    if policy.rank != 1:
        raise ValueError("train_rank1_episode requires a rank-1 policy")

    spec = TowerGraphSpec(rank=1) if graph_spec is None else graph_spec
    if spec.rank != 1:
        raise ValueError("train_rank1_episode requires a rank-1 graph spec")

    def active_sampler(**kwargs: object):
        return sample_active_choice_from_policy(
            policy=policy,
            state=kwargs["state"],  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            active_choices=kwargs["active_choices"],  # type: ignore[arg-type]
            measure_size=measure_size,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=spec.max_step_size,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
            generator=generator,
        )

    trajectory = rollout_rank1(
        initial_state=initial_state,
        max_steps=max_steps,
        active_sampler=active_sampler,
        reward_fn=reward_fn,
        graph_spec=spec,
        measure_size=measure_size,
        context_measures=context_measures,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
    )
    loss = policy_gradient_loss(
        trajectory,
        gamma=gamma,
        normalize_returns=normalize_returns,
    )

    optimizer.zero_grad()
    loss.loss.backward()
    optimizer.step()

    return TrainEpisodeResult(
        trajectory=trajectory,
        loss=loss,
        metrics=_rank1_episode_metrics(
            trajectory=trajectory,
            loss_value=float(loss.loss.detach().cpu().item()),
        ),
    )


def train_rank1_episode_with_artifacts(
    *,
    policy: RankPolicy,
    optimizer: torch.optim.Optimizer,
    config: TowerRankConfig,
    paths: TowerArtifactPaths,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    episode_index: int,
    graph_spec: TowerGraphSpec | None = None,
    gamma: float | None = None,
    normalize_returns: bool | None = None,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    generator: torch.Generator | None = None,
) -> TrainEpisodeResult:
    """Run one rank-1 episode and persist rank artifacts."""
    if config.rank != 1:
        raise ValueError("rank-1 artifact training requires rank-1 config")
    if paths.rank != 1:
        raise ValueError("rank-1 artifact training requires rank-1 paths")
    if config.lineage_id != paths.lineage_id:
        raise ValueError("config lineage_id must match artifact paths lineage_id")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")

    training_config = dict(config.training_config)
    result = train_rank1_episode(
        policy=policy,
        optimizer=optimizer,
        initial_state=initial_state,
        reward_fn=reward_fn,
        max_steps=_int_config_value(training_config, "max_steps", default=1),
        graph_spec=graph_spec
        if graph_spec is not None
        else TowerGraphSpec(
            rank=1,
            max_step_size=config.max_step_size,
        ),
        measure_size=config.measure_size,
        context_measures=config.context_measures,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        sampling_temperature=_float_config_value(
            training_config,
            "sampling_temperature",
            default=1.0,
        ),
        sampling_uniform_mix=_float_config_value(
            training_config,
            "sampling_uniform_mix",
            default=0.0,
        ),
        gamma=gamma
        if gamma is not None
        else _float_config_value(training_config, "gamma", default=1.0),
        normalize_returns=normalize_returns
        if normalize_returns is not None
        else _bool_config_value(
            training_config,
            "normalize_returns",
            default=False,
        ),
        generator=generator,
    )

    metrics = {
        **result.metrics,
        "episode_index": episode_index,
        "initial_state": list(initial_state),
        "target_root_octave": target_root_octave,
    }
    checkpoint_payload = build_checkpoint_payload(
        config=config,
        episode_index=episode_index,
        stats=metrics,
        policy_state_dict=policy.state_dict(),  # type: ignore[attr-defined]
        optimizer_state_dict=optimizer.state_dict(),
    )

    write_rank_config(config=config, paths=paths)
    append_rank_metrics(paths=paths, metrics=metrics)
    save_latest_checkpoint(paths=paths, payload=checkpoint_payload)
    record_rank_manifest_entry(
        paths=paths,
        status="accepted"
        if episode_index + 1 >= config.episode_budget
        else "running",
    )

    return TrainEpisodeResult(
        trajectory=result.trajectory,
        loss=result.loss,
        metrics=metrics,
    )


def train_rank2_episode(
    *,
    parent_policy: RankPolicy,
    child_policy: RankPolicy,
    child_optimizer: torch.optim.Optimizer,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    max_steps: int,
    graph_spec: TowerGraphSpec | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
    parent_top_m: int = 1,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    sampling_temperature: float = 1.0,
    sampling_uniform_mix: float = 0.0,
    gamma: float = 1.0,
    normalize_returns: bool = False,
    generator: torch.Generator | None = None,
) -> TrainEpisodeResult:
    """Run one rank-2 episode over a frozen rank-1 parent policy."""
    if parent_policy.rank != 1:
        raise ValueError("train_rank2_episode requires a rank-1 parent policy")
    if child_policy.rank != 2:
        raise ValueError("train_rank2_episode requires a rank-2 child policy")

    spec = TowerGraphSpec(rank=2) if graph_spec is None else graph_spec
    if spec.rank != 2:
        raise ValueError("train_rank2_episode requires a rank-2 graph spec")

    frozen_parent = freeze_parent_policy(parent_policy)  # type: ignore[arg-type]
    parent_spec = TowerGraphSpec(
        rank=1,
        key_pitch_class=spec.key_pitch_class,
        pitch_min=spec.pitch_min,
        pitch_max=spec.pitch_max,
        max_step_size=spec.max_step_size,
    )

    def parent_sampler(**kwargs: object):
        parent_state = kwargs["state"]  # type: ignore[assignment]
        full_state = kwargs["full_state"]  # type: ignore[assignment]
        parent_actions = legal_actions_for_state(
            state=parent_state,  # type: ignore[arg-type]
            spec=parent_spec,
        )
        feasible_parent_actions = tuple(
            action
            for action in parent_actions
            if active_lift_choices(
                state=full_state,  # type: ignore[arg-type]
                parent_action=action,
                spec=spec,
            )
        )
        sampled = sample_parent_top_m_from_policy(
            policy=frozen_parent,  # type: ignore[arg-type]
            state=parent_state,  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            parent_actions=(
                feasible_parent_actions
                if feasible_parent_actions
                else parent_actions
            ),
            measure_size=measure_size,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=parent_spec.max_step_size,
            top_m=parent_top_m,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
            generator=generator,
        )
        return SamplerResult(
            choice=sampled.choice,
            logprob=sampled.logprob,
            diagnostics={
                **sampled.diagnostics,
                "unfiltered_parent_actions": parent_actions,
                "feasible_parent_actions": feasible_parent_actions,
                "parent_feasibility_filter_applied": (
                    feasible_parent_actions != parent_actions
                ),
            },
        )

    def active_sampler(**kwargs: object):
        return sample_active_choice_from_policy(
            policy=child_policy,
            state=kwargs["state"],  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            active_choices=kwargs["active_choices"],  # type: ignore[arg-type]
            measure_size=measure_size,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=spec.max_step_size,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
            generator=generator,
        )

    trajectory = rollout_rank2(
        initial_state=initial_state,
        max_steps=max_steps,
        parent_sampler=parent_sampler,
        active_sampler=active_sampler,
        reward_fn=reward_fn,
        graph_spec=spec,
        measure_size=measure_size,
        context_measures=context_measures,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
    )
    loss = policy_gradient_loss(
        trajectory,
        gamma=gamma,
        normalize_returns=normalize_returns,
    )

    child_optimizer.zero_grad()
    no_child_gradient = (
        int(loss.diagnostics.get("active_step_count", 0)) == 0
    )
    if not no_child_gradient:
        loss.loss.backward()
        child_optimizer.step()

    metrics = _episode_metrics(
        trajectory=trajectory,
        rank=2,
        loss_value=float(loss.loss.detach().cpu().item()),
    )
    metrics["no_child_gradient_count"] = 1 if no_child_gradient else 0
    return TrainEpisodeResult(
        trajectory=trajectory,
        loss=loss,
        metrics=metrics,
    )


def train_rank3_episode(
    *,
    grandparent_policy: RankPolicy,
    parent_policy: RankPolicy,
    child_policy: RankPolicy,
    child_optimizer: torch.optim.Optimizer,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    max_steps: int,
    graph_spec: TowerGraphSpec | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
    parent_top_m: int = 1,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    sampling_temperature: float = 1.0,
    sampling_uniform_mix: float = 0.0,
    gamma: float = 1.0,
    normalize_returns: bool = False,
    generator: torch.Generator | None = None,
) -> TrainEpisodeResult:
    """Run one rank-3 episode over frozen rank-1/rank-2 parent tiers."""
    if grandparent_policy.rank != 1:
        raise ValueError("train_rank3_episode requires a rank-1 grandparent policy")
    if parent_policy.rank != 2:
        raise ValueError("train_rank3_episode requires a rank-2 parent policy")
    if child_policy.rank != 3:
        raise ValueError("train_rank3_episode requires a rank-3 child policy")

    spec = TowerGraphSpec(rank=3) if graph_spec is None else graph_spec
    if spec.rank != 3:
        raise ValueError("train_rank3_episode requires a rank-3 graph spec")

    frozen_grandparent = freeze_parent_policy(grandparent_policy)  # type: ignore[arg-type]
    frozen_parent = freeze_parent_policy(parent_policy)  # type: ignore[arg-type]
    parent_spec = TowerGraphSpec(
        rank=2,
        key_pitch_class=spec.key_pitch_class,
        pitch_min=spec.pitch_min,
        pitch_max=spec.pitch_max,
        max_step_size=spec.max_step_size,
    )
    grandparent_spec = TowerGraphSpec(
        rank=1,
        key_pitch_class=spec.key_pitch_class,
        pitch_min=spec.pitch_min,
        pitch_max=spec.pitch_max,
        max_step_size=spec.max_step_size,
    )

    def parent_sampler(**kwargs: object):
        parent_state = kwargs["state"]  # type: ignore[assignment]
        full_state = kwargs["full_state"]  # type: ignore[assignment]
        parent_window = kwargs["window"]  # type: ignore[assignment]
        grandparent_state = project_state(parent_state)  # type: ignore[arg-type]
        grandparent_window = project_window(parent_window)  # type: ignore[arg-type]

        def child_feasible_parent_choices(
            grandparent_action: tuple[int, ...],
        ) -> tuple[int, ...]:
            raw_choices = active_lift_choices(
                state=parent_state,  # type: ignore[arg-type]
                parent_action=grandparent_action,
                spec=parent_spec,
            )
            return tuple(
                choice
                for choice in raw_choices
                if active_lift_choices(
                    state=full_state,  # type: ignore[arg-type]
                    parent_action=assemble_action(
                        rank=2,
                        parent_action=grandparent_action,
                        new_action=choice,
                    ),
                    spec=spec,
                )
            )

        grandparent_actions = legal_actions_for_state(
            state=grandparent_state,  # type: ignore[arg-type]
            spec=grandparent_spec,
        )
        feasible_grandparent_actions = tuple(
            action
            for action in grandparent_actions
            if child_feasible_parent_choices(action)
        )
        sampled_grandparent = sample_parent_top_m_from_policy(
            policy=frozen_grandparent,  # type: ignore[arg-type]
            state=grandparent_state,  # type: ignore[arg-type]
            window=grandparent_window,  # type: ignore[arg-type]
            parent_actions=(
                feasible_grandparent_actions
                if feasible_grandparent_actions
                else grandparent_actions
            ),
            measure_size=measure_size,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=grandparent_spec.max_step_size,
            top_m=parent_top_m,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
            generator=generator,
        )
        grandparent_action = sampled_grandparent.choice
        raw_parent_choices = active_lift_choices(
            state=parent_state,  # type: ignore[arg-type]
            parent_action=grandparent_action,
            spec=parent_spec,
        )
        feasible_parent_choices = child_feasible_parent_choices(grandparent_action)
        if not raw_parent_choices:
            return SamplerResult(
                choice=(0, 0),
                logprob=sampled_grandparent.logprob,
                diagnostics={
                    "grandparent_sampler": {
                        **sampled_grandparent.diagnostics,
                        "unfiltered_parent_actions": grandparent_actions,
                        "feasible_parent_actions": feasible_grandparent_actions,
                        "parent_feasibility_filter_applied": (
                            feasible_grandparent_actions != grandparent_actions
                        ),
                    },
                    "parent_active_choices": raw_parent_choices,
                    "feasible_parent_active_choices": feasible_parent_choices,
                    "empty_parent_lift_fiber": True,
                },
            )

        sampled_parent_active = sample_active_choice_from_policy(
            policy=frozen_parent,  # type: ignore[arg-type]
            state=parent_state,  # type: ignore[arg-type]
            window=parent_window,  # type: ignore[arg-type]
            active_choices=(
                feasible_parent_choices
                if feasible_parent_choices
                else raw_parent_choices
            ),
            measure_size=measure_size,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=parent_spec.max_step_size,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
            generator=generator,
        )
        parent_action = assemble_action(
            rank=2,
            parent_action=grandparent_action,
            new_action=sampled_parent_active.choice,
        )
        return SamplerResult(
            choice=parent_action,
            logprob=_sum_logprobs(
                sampled_grandparent.logprob,
                sampled_parent_active.logprob,
            ),
            diagnostics={
                "grandparent_sampler": {
                    **sampled_grandparent.diagnostics,
                    "unfiltered_parent_actions": grandparent_actions,
                    "feasible_parent_actions": feasible_grandparent_actions,
                    "parent_feasibility_filter_applied": (
                        feasible_grandparent_actions != grandparent_actions
                    ),
                },
                "parent_sampler": sampled_parent_active.diagnostics,
                "parent_active_choices": raw_parent_choices,
                "feasible_parent_active_choices": feasible_parent_choices,
                "child_feasibility_filter_applied": (
                    feasible_parent_choices != raw_parent_choices
                ),
            },
        )

    def active_sampler(**kwargs: object):
        return sample_active_choice_from_policy(
            policy=child_policy,
            state=kwargs["state"],  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            active_choices=kwargs["active_choices"],  # type: ignore[arg-type]
            measure_size=measure_size,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=spec.max_step_size,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
            generator=generator,
        )

    trajectory = rollout_rank3(
        initial_state=initial_state,
        max_steps=max_steps,
        parent_sampler=parent_sampler,
        active_sampler=active_sampler,
        reward_fn=reward_fn,
        graph_spec=spec,
        measure_size=measure_size,
        context_measures=context_measures,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
    )
    loss = policy_gradient_loss(
        trajectory,
        gamma=gamma,
        normalize_returns=normalize_returns,
    )

    child_optimizer.zero_grad()
    loss.loss.backward()
    child_optimizer.step()

    return TrainEpisodeResult(
        trajectory=trajectory,
        loss=loss,
        metrics=_episode_metrics(
            trajectory=trajectory,
            rank=3,
            loss_value=float(loss.loss.detach().cpu().item()),
        ),
    )


def train_rank2_episode_with_artifacts(
    *,
    parent_policy: RankPolicy,
    child_policy: RankPolicy,
    child_optimizer: torch.optim.Optimizer,
    config: TowerRankConfig,
    paths: TowerArtifactPaths,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    episode_index: int,
    graph_spec: TowerGraphSpec | None = None,
    gamma: float | None = None,
    normalize_returns: bool | None = None,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    generator: torch.Generator | None = None,
) -> TrainEpisodeResult:
    """Run one rank-2 episode and persist parent-linked rank artifacts."""
    if config.rank != 2:
        raise ValueError("rank-2 artifact training requires rank-2 config")
    if paths.rank != 2:
        raise ValueError("rank-2 artifact training requires rank-2 paths")
    if config.lineage_id != paths.lineage_id:
        raise ValueError("config lineage_id must match artifact paths lineage_id")
    if config.parent_checkpoint is None:
        raise ValueError("rank-2 config requires parent_checkpoint")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")

    parent_payload = load_accepted_parent_checkpoint(paths)
    accepted_parent_checkpoint = find_accepted_parent_checkpoint(paths)
    parent_relative_checkpoint = accepted_parent_checkpoint.relative_to(
        paths.lineage_dir
    ).as_posix()
    if config.parent_checkpoint != parent_relative_checkpoint:
        raise ValueError("config parent_checkpoint must match accepted parent")
    parent_schema_version = parent_payload.get("artifact_schema_version")
    if not isinstance(parent_schema_version, int):
        raise ValueError("parent checkpoint artifact_schema_version must be an int")

    training_config = dict(config.training_config)
    parent_sampler_config = dict(config.parent_sampler_config)
    result = train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=initial_state,
        reward_fn=reward_fn,
        max_steps=_int_config_value(training_config, "max_steps", default=1),
        graph_spec=graph_spec
        if graph_spec is not None
        else TowerGraphSpec(
            rank=2,
            max_step_size=config.max_step_size,
        ),
        measure_size=config.measure_size,
        context_measures=config.context_measures,
        parent_top_m=_int_config_value(parent_sampler_config, "top_m", default=1),
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        sampling_temperature=_float_config_value(
            training_config,
            "sampling_temperature",
            default=1.0,
        ),
        sampling_uniform_mix=_float_config_value(
            training_config,
            "sampling_uniform_mix",
            default=0.0,
        ),
        gamma=gamma
        if gamma is not None
        else _float_config_value(training_config, "gamma", default=1.0),
        normalize_returns=normalize_returns
        if normalize_returns is not None
        else _bool_config_value(
            training_config,
            "normalize_returns",
            default=False,
        ),
        generator=generator,
    )

    metrics = {
        **result.metrics,
        "episode_index": episode_index,
    }
    checkpoint_payload = build_checkpoint_payload(
        config=config,
        episode_index=episode_index,
        stats=metrics,
        policy_state_dict=child_policy.state_dict(),  # type: ignore[attr-defined]
        optimizer_state_dict=child_optimizer.state_dict(),
        parent_checkpoint=parent_relative_checkpoint,
        parent_artifact_schema_version=parent_schema_version,
    )

    write_rank_config(config=config, paths=paths)
    append_rank_metrics(paths=paths, metrics=metrics)
    save_latest_checkpoint(paths=paths, payload=checkpoint_payload)
    record_rank_manifest_entry(
        paths=paths,
        status="accepted"
        if episode_index + 1 >= config.episode_budget
        else "running",
        parent_checkpoint=parent_relative_checkpoint,
    )

    return TrainEpisodeResult(
        trajectory=result.trajectory,
        loss=result.loss,
        metrics=metrics,
    )


def train_rank3_episode_with_artifacts(
    *,
    grandparent_policy: RankPolicy,
    parent_policy: RankPolicy,
    child_policy: RankPolicy,
    child_optimizer: torch.optim.Optimizer,
    config: TowerRankConfig,
    paths: TowerArtifactPaths,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    episode_index: int,
    graph_spec: TowerGraphSpec | None = None,
    gamma: float | None = None,
    normalize_returns: bool | None = None,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    generator: torch.Generator | None = None,
) -> TrainEpisodeResult:
    """Run one rank-3 episode and persist parent-linked rank artifacts."""
    if config.rank != 3:
        raise ValueError("rank-3 artifact training requires rank-3 config")
    if paths.rank != 3:
        raise ValueError("rank-3 artifact training requires rank-3 paths")
    if config.lineage_id != paths.lineage_id:
        raise ValueError("config lineage_id must match artifact paths lineage_id")
    if config.parent_checkpoint is None:
        raise ValueError("rank-3 config requires parent_checkpoint")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")

    parent_payload = load_accepted_parent_checkpoint(paths)
    accepted_parent_checkpoint = find_accepted_parent_checkpoint(paths)
    parent_relative_checkpoint = accepted_parent_checkpoint.relative_to(
        paths.lineage_dir
    ).as_posix()
    if config.parent_checkpoint != parent_relative_checkpoint:
        raise ValueError("config parent_checkpoint must match accepted parent")
    parent_schema_version = parent_payload.get("artifact_schema_version")
    if not isinstance(parent_schema_version, int):
        raise ValueError("parent checkpoint artifact_schema_version must be an int")

    training_config = dict(config.training_config)
    parent_sampler_config = dict(config.parent_sampler_config)
    result = train_rank3_episode(
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=initial_state,
        reward_fn=reward_fn,
        max_steps=_int_config_value(training_config, "max_steps", default=1),
        graph_spec=graph_spec
        if graph_spec is not None
        else TowerGraphSpec(
            rank=3,
            max_step_size=config.max_step_size,
        ),
        measure_size=config.measure_size,
        context_measures=config.context_measures,
        parent_top_m=_int_config_value(parent_sampler_config, "top_m", default=1),
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        sampling_temperature=_float_config_value(
            training_config,
            "sampling_temperature",
            default=1.0,
        ),
        sampling_uniform_mix=_float_config_value(
            training_config,
            "sampling_uniform_mix",
            default=0.0,
        ),
        gamma=gamma
        if gamma is not None
        else _float_config_value(training_config, "gamma", default=1.0),
        normalize_returns=normalize_returns
        if normalize_returns is not None
        else _bool_config_value(
            training_config,
            "normalize_returns",
            default=False,
        ),
        generator=generator,
    )

    metrics = {
        **result.metrics,
        "episode_index": episode_index,
    }
    checkpoint_payload = build_checkpoint_payload(
        config=config,
        episode_index=episode_index,
        stats=metrics,
        policy_state_dict=child_policy.state_dict(),  # type: ignore[attr-defined]
        optimizer_state_dict=child_optimizer.state_dict(),
        parent_checkpoint=parent_relative_checkpoint,
        parent_artifact_schema_version=parent_schema_version,
    )

    write_rank_config(config=config, paths=paths)
    append_rank_metrics(paths=paths, metrics=metrics)
    save_latest_checkpoint(paths=paths, payload=checkpoint_payload)
    record_rank_manifest_entry(
        paths=paths,
        status="accepted"
        if episode_index + 1 >= config.episode_budget
        else "running",
        parent_checkpoint=parent_relative_checkpoint,
    )

    return TrainEpisodeResult(
        trajectory=result.trajectory,
        loss=result.loss,
        metrics=metrics,
    )


def _rank1_episode_metrics(
    *,
    trajectory: TowerTrajectory,
    loss_value: float,
) -> dict[str, object]:
    episode_length = len(trajectory.steps)
    return _episode_metrics(
        trajectory=trajectory,
        rank=1,
        loss_value=loss_value,
    )


def _episode_metrics(
    *,
    trajectory: TowerTrajectory,
    rank: int,
    loss_value: float,
) -> dict[str, object]:
    episode_length = len(trajectory.steps)
    episode_return = trajectory.total_reward
    terminated = any(step.terminated for step in trajectory.steps)
    truncated = bool(trajectory.steps and trajectory.steps[-1].truncated)

    return {
        "rank": rank,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "mean_step_reward": episode_return / episode_length
        if episode_length > 0
        else 0.0,
        "terminated": terminated,
        "truncated": truncated,
        "loss": loss_value,
        "invalid_extension_count": sum(
            1 for step in trajectory.steps if step.outcome == "invalid_extension"
        ),
        "empty_lift_fiber_count": sum(
            1 for step in trajectory.steps if step.outcome == "empty_lift_fiber"
        ),
        "parent_failure_count": sum(
            1 for step in trajectory.steps if step.outcome == "parent_failure"
        ),
        "tail_stagnation_count": sum(
            1 for step in trajectory.steps if step.outcome == "tail_stagnation"
        ),
        "terminal_success": any(
            step.reward.is_terminal_success for step in trajectory.steps
        ),
    }


def _int_config_value(
    config: dict[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"training_config.{key} must be an int")
    return value


def _sum_logprobs(
    left: float | torch.Tensor | None,
    right: float | torch.Tensor | None,
) -> float | torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _float_config_value(
    config: dict[str, object],
    key: str,
    *,
    default: float,
) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float):
        raise TypeError(f"training_config.{key} must be a number")
    return float(value)


def _bool_config_value(
    config: dict[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"training_config.{key} must be a bool")
    return value
