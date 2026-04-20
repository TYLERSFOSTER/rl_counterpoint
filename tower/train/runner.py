"""Artifact-backed tower training runner contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch

from tower.graph.actions import action_space
from tower.graph.legality import is_valid_transition
from tower.graph.spec import TowerGraphSpec
from tower.policy.base import RankPolicy
from tower.policy.samplers import (
    sample_active_choice_from_policy,
    sample_parent_top_m_from_policy,
)
from tower.policy.transformer import (
    TowerTransformerPolicy,
    TowerTransformerPolicyConfig,
)
from tower.state_action import validate_rank
from tower.music.render import write_trajectory_to_midi
from tower.train.checkpoint import (
    DEFAULT_TOWER_ARTIFACT_ROOT,
    TowerArtifactPaths,
    append_rank_metrics,
)
from tower.train.config import TowerRankConfig, _validate_json_mapping
from tower.train.protocol import (
    TrainEpisodeResult,
    train_rank1_episode_with_artifacts,
    train_rank2_episode_with_artifacts,
)
from tower.train.rollout import RewardFunction, rollout_rank1, rollout_rank2
from tower.train.trajectory import TowerTrajectory


@dataclass(frozen=True)
class TowerRunnerConfig:
    """Run-level settings for one artifact-backed tower training job."""

    lineage_id: str
    rank: int
    episode_count: int
    seed: int
    artifact_root: Path = DEFAULT_TOWER_ARTIFACT_ROOT
    measure_size: int = 4
    context_measures: int = 2
    max_step_size: int = 4
    parent_checkpoint: str | None = None
    parent_top_m: int = 3
    final_midi_enabled: bool = True
    reward_config: Mapping[str, object] = field(default_factory=dict)
    graph_config: Mapping[str, object] = field(default_factory=dict)
    policy_config: Mapping[str, object] = field(default_factory=dict)
    training_config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        if not isinstance(self.lineage_id, str):
            raise TypeError("lineage_id must be a string")
        if not self.lineage_id:
            raise ValueError("lineage_id must not be empty")
        if not isinstance(self.episode_count, int):
            raise TypeError("episode_count must be an int")
        if self.episode_count < 1:
            raise ValueError("episode_count must be at least 1")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.artifact_root, Path):
            raise TypeError("artifact_root must be a Path")
        if self.measure_size < 1:
            raise ValueError("measure_size must be at least 1")
        if self.context_measures < 1:
            raise ValueError("context_measures must be at least 1")
        if self.max_step_size < 1:
            raise ValueError("max_step_size must be at least 1")
        if self.parent_checkpoint is not None and not isinstance(
            self.parent_checkpoint,
            str,
        ):
            raise TypeError("parent_checkpoint must be a string or None")
        if not isinstance(self.parent_top_m, int):
            raise TypeError("parent_top_m must be an int")
        if self.parent_top_m < 1:
            raise ValueError("parent_top_m must be at least 1")
        if not isinstance(self.final_midi_enabled, bool):
            raise TypeError("final_midi_enabled must be a bool")

        if self.rank == 1 and self.parent_checkpoint is not None:
            raise ValueError("rank 1 runner config must not set parent_checkpoint")
        if self.rank > 1 and self.parent_checkpoint is None:
            raise ValueError("rank greater than 1 requires parent_checkpoint")

        _validate_json_mapping(self.reward_config, field_name="reward_config")
        _validate_json_mapping(self.graph_config, field_name="graph_config")
        _validate_json_mapping(self.policy_config, field_name="policy_config")
        _validate_json_mapping(self.training_config, field_name="training_config")

    def artifact_paths(self) -> TowerArtifactPaths:
        """Return deterministic artifact paths for this run."""
        return TowerArtifactPaths(
            lineage_id=self.lineage_id,
            rank=self.rank,
            artifact_root=self.artifact_root,
        )

    def to_rank_config(self) -> TowerRankConfig:
        """Return the rank-local artifact config represented by this run."""
        return TowerRankConfig(
            rank=self.rank,
            lineage_id=self.lineage_id,
            episode_budget=self.episode_count,
            measure_size=self.measure_size,
            context_measures=self.context_measures,
            max_step_size=self.max_step_size,
            reward_config=dict(self.reward_config),
            graph_config=dict(self.graph_config),
            policy_config=dict(self.policy_config),
            training_config={
                **dict(self.training_config),
                "episode_count": self.episode_count,
            },
            parent_sampler_config={}
            if self.rank == 1
            else {"top_m": self.parent_top_m},
            parent_checkpoint=self.parent_checkpoint,
            seed_config={"seed": self.seed},
        )


@dataclass(frozen=True)
class FinalInferenceResult:
    """Result of one final no-train inference episode."""

    trajectory: TowerTrajectory
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class Rank1TrainingRunResult:
    """Artifacts and summaries from one rank-1 runner invocation."""

    config: TowerRunnerConfig
    rank_config: TowerRankConfig
    paths: TowerArtifactPaths
    policy: torch.nn.Module
    episode_results: tuple[TrainEpisodeResult, ...]
    final_inference: FinalInferenceResult
    final_midi_path: Path | None


@dataclass(frozen=True)
class Rank2TrainingRunResult:
    """Artifacts and summaries from one rank-2 runner invocation."""

    config: TowerRunnerConfig
    rank_config: TowerRankConfig
    paths: TowerArtifactPaths
    parent_policy: torch.nn.Module
    child_policy: torch.nn.Module
    episode_results: tuple[TrainEpisodeResult, ...]
    final_inference: FinalInferenceResult
    final_midi_path: Path | None


def run_rank1_training(
    *,
    config: TowerRunnerConfig,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    policy: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    graph_spec: TowerGraphSpec | None = None,
) -> Rank1TrainingRunResult:
    """Train rank 1 for the configured episode count and write artifacts."""
    if config.rank != 1:
        raise ValueError("run_rank1_training requires rank-1 runner config")
    active_policy = _build_rank1_policy(config) if policy is None else policy
    if not hasattr(active_policy, "rank") or active_policy.rank != 1:
        raise ValueError("rank-1 runner policy must have rank 1")

    active_optimizer = (
        _build_optimizer(policy=active_policy, config=config)
        if optimizer is None
        else optimizer
    )
    paths = config.artifact_paths()
    rank_config = config.to_rank_config()
    spec = graph_spec if graph_spec is not None else _graph_spec_from_config(config)
    if spec.rank != 1:
        raise ValueError("rank-1 runner requires rank-1 graph spec")

    generator = torch.Generator().manual_seed(config.seed)
    episode_results = []
    for episode_index in range(config.episode_count):
        episode_results.append(
            train_rank1_episode_with_artifacts(
                policy=active_policy,  # type: ignore[arg-type]
                optimizer=active_optimizer,
                config=rank_config,
                paths=paths,
                initial_state=initial_state,
                reward_fn=reward_fn,
                episode_index=episode_index,
                graph_spec=spec,
                generator=generator,
            )
        )

    max_steps = _training_int(config, "max_steps", default=1)
    final_inference = run_final_inference_episode(
        policy=active_policy,  # type: ignore[arg-type]
        initial_state=initial_state,
        reward_fn=reward_fn,
        max_steps=max_steps,
        graph_spec=spec,
        measure_size=config.measure_size,
        context_measures=config.context_measures,
        generator=generator,
    )
    final_midi_path = None
    if config.final_midi_enabled:
        final_midi_path = write_trajectory_to_midi(
            trajectory=final_inference.trajectory,
            path=paths.example_episode_path,
        )

    append_rank_metrics(
        paths=paths,
        metrics={
            **dict(final_inference.metrics),
            "episode_index": config.episode_count,
            "kind": "final_inference",
            "midi_path": None
            if final_midi_path is None
            else final_midi_path.relative_to(paths.lineage_dir).as_posix(),
        },
    )

    return Rank1TrainingRunResult(
        config=config,
        rank_config=rank_config,
        paths=paths,
        policy=active_policy,
        episode_results=tuple(episode_results),
        final_inference=final_inference,
        final_midi_path=final_midi_path,
    )


def run_rank2_training(
    *,
    config: TowerRunnerConfig,
    parent_policy: torch.nn.Module,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    child_policy: torch.nn.Module | None = None,
    child_optimizer: torch.optim.Optimizer | None = None,
    graph_spec: TowerGraphSpec | None = None,
) -> Rank2TrainingRunResult:
    """Train rank 2 over one accepted rank-1 parent checkpoint."""
    if config.rank != 2:
        raise ValueError("run_rank2_training requires rank-2 runner config")
    if not hasattr(parent_policy, "rank") or parent_policy.rank != 1:
        raise ValueError("rank-2 runner parent policy must have rank 1")

    active_child_policy = (
        _build_rank2_policy(config) if child_policy is None else child_policy
    )
    if not hasattr(active_child_policy, "rank") or active_child_policy.rank != 2:
        raise ValueError("rank-2 runner child policy must have rank 2")

    active_child_optimizer = (
        _build_optimizer(policy=active_child_policy, config=config)
        if child_optimizer is None
        else child_optimizer
    )
    paths = config.artifact_paths()
    rank_config = config.to_rank_config()
    spec = graph_spec if graph_spec is not None else _graph_spec_from_config(config)
    if spec.rank != 2:
        raise ValueError("rank-2 runner requires rank-2 graph spec")

    generator = torch.Generator().manual_seed(config.seed)
    episode_results = []
    for episode_index in range(config.episode_count):
        episode_results.append(
            train_rank2_episode_with_artifacts(
                parent_policy=parent_policy,  # type: ignore[arg-type]
                child_policy=active_child_policy,  # type: ignore[arg-type]
                child_optimizer=active_child_optimizer,
                config=rank_config,
                paths=paths,
                initial_state=initial_state,
                reward_fn=reward_fn,
                episode_index=episode_index,
                graph_spec=spec,
                generator=generator,
            )
        )

    max_steps = _training_int(config, "max_steps", default=1)
    final_inference = run_final_inference_episode(
        policy=active_child_policy,  # type: ignore[arg-type]
        parent_policy=parent_policy,  # type: ignore[arg-type]
        initial_state=initial_state,
        reward_fn=reward_fn,
        max_steps=max_steps,
        graph_spec=spec,
        measure_size=config.measure_size,
        context_measures=config.context_measures,
        parent_top_m=config.parent_top_m,
        generator=generator,
    )
    final_midi_path = None
    if config.final_midi_enabled:
        final_midi_path = write_trajectory_to_midi(
            trajectory=final_inference.trajectory,
            path=paths.example_episode_path,
        )

    append_rank_metrics(
        paths=paths,
        metrics={
            **dict(final_inference.metrics),
            "episode_index": config.episode_count,
            "kind": "final_inference",
            "midi_path": None
            if final_midi_path is None
            else final_midi_path.relative_to(paths.lineage_dir).as_posix(),
        },
    )

    return Rank2TrainingRunResult(
        config=config,
        rank_config=rank_config,
        paths=paths,
        parent_policy=parent_policy,
        child_policy=active_child_policy,
        episode_results=tuple(episode_results),
        final_inference=final_inference,
        final_midi_path=final_midi_path,
    )


def run_final_inference_episode(
    *,
    policy: RankPolicy,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    max_steps: int,
    graph_spec: TowerGraphSpec | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
    parent_policy: RankPolicy | None = None,
    parent_top_m: int = 1,
    generator: torch.Generator | None = None,
) -> FinalInferenceResult:
    """Run one final inference episode without training or gradients."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")
    if context_measures < 1:
        raise ValueError("context_measures must be at least 1")
    if parent_top_m < 1:
        raise ValueError("parent_top_m must be at least 1")

    rank = policy.rank
    if rank not in {1, 2}:
        raise ValueError("final inference currently supports rank 1 or rank 2")

    spec = TowerGraphSpec(rank=rank) if graph_spec is None else graph_spec
    if spec.rank != rank:
        raise ValueError("graph spec rank must match policy rank")

    _eval_if_module(policy)
    if parent_policy is not None:
        _eval_if_module(parent_policy)

    with torch.no_grad():
        if rank == 1:
            if parent_policy is not None:
                raise ValueError("rank 1 final inference must not set parent_policy")
            trajectory = _run_rank1_final_inference(
                policy=policy,
                initial_state=initial_state,
                reward_fn=reward_fn,
                max_steps=max_steps,
                graph_spec=spec,
                measure_size=measure_size,
                context_measures=context_measures,
                generator=generator,
            )
        else:
            if parent_policy is None:
                raise ValueError("rank 2 final inference requires parent_policy")
            if parent_policy.rank != 1:
                raise ValueError("rank 2 final inference requires rank-1 parent policy")
            trajectory = _run_rank2_final_inference(
                parent_policy=parent_policy,
                child_policy=policy,
                initial_state=initial_state,
                reward_fn=reward_fn,
                max_steps=max_steps,
                graph_spec=spec,
                measure_size=measure_size,
                context_measures=context_measures,
                parent_top_m=parent_top_m,
                generator=generator,
            )

    return FinalInferenceResult(
        trajectory=trajectory,
        metrics=_final_inference_metrics(trajectory=trajectory, rank=rank),
    )


def _run_rank1_final_inference(
    *,
    policy: RankPolicy,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    max_steps: int,
    graph_spec: TowerGraphSpec,
    measure_size: int,
    context_measures: int,
    generator: torch.Generator | None,
) -> TowerTrajectory:
    def active_sampler(**kwargs: object):
        return sample_active_choice_from_policy(
            policy=policy,
            state=kwargs["state"],  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            active_choices=kwargs["active_choices"],  # type: ignore[arg-type]
            measure_size=measure_size,
            generator=generator,
        )

    return rollout_rank1(
        initial_state=initial_state,
        max_steps=max_steps,
        active_sampler=active_sampler,
        reward_fn=reward_fn,
        graph_spec=graph_spec,
        measure_size=measure_size,
        context_measures=context_measures,
    )


def _run_rank2_final_inference(
    *,
    parent_policy: RankPolicy,
    child_policy: RankPolicy,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    max_steps: int,
    graph_spec: TowerGraphSpec,
    measure_size: int,
    context_measures: int,
    parent_top_m: int,
    generator: torch.Generator | None,
) -> TowerTrajectory:
    parent_spec = TowerGraphSpec(
        rank=1,
        pitch_min=graph_spec.pitch_min,
        pitch_max=graph_spec.pitch_max,
        max_step_size=graph_spec.max_step_size,
    )

    def parent_sampler(**kwargs: object):
        parent_state = kwargs["state"]  # type: ignore[assignment]
        parent_actions = tuple(
            action
            for action in action_space(
                rank=1,
                max_step_size=parent_spec.max_step_size,
            )
            if is_valid_transition(parent_state, action, parent_spec)  # type: ignore[arg-type]
        )
        return sample_parent_top_m_from_policy(
            policy=parent_policy,
            state=parent_state,  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            parent_actions=parent_actions,
            measure_size=measure_size,
            top_m=parent_top_m,
            generator=generator,
        )

    def active_sampler(**kwargs: object):
        return sample_active_choice_from_policy(
            policy=child_policy,
            state=kwargs["state"],  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            active_choices=kwargs["active_choices"],  # type: ignore[arg-type]
            measure_size=measure_size,
            generator=generator,
        )

    return rollout_rank2(
        initial_state=initial_state,
        max_steps=max_steps,
        parent_sampler=parent_sampler,
        active_sampler=active_sampler,
        reward_fn=reward_fn,
        graph_spec=graph_spec,
        measure_size=measure_size,
        context_measures=context_measures,
    )


def _eval_if_module(policy: RankPolicy) -> None:
    if isinstance(policy, torch.nn.Module):
        policy.eval()


def _final_inference_metrics(
    *,
    trajectory: TowerTrajectory,
    rank: int,
) -> dict[str, object]:
    episode_length = len(trajectory.steps)
    episode_return = trajectory.total_reward
    return {
        "rank": rank,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "mean_step_reward": episode_return / episode_length
        if episode_length > 0
        else 0.0,
        "terminated": any(step.terminated for step in trajectory.steps),
        "truncated": bool(trajectory.steps and trajectory.steps[-1].truncated),
        "terminal_success": any(
            step.reward.is_terminal_success for step in trajectory.steps
        ),
        "invalid_extension_count": sum(
            1 for step in trajectory.steps if step.outcome == "invalid_extension"
        ),
        "empty_lift_fiber_count": sum(
            1 for step in trajectory.steps if step.outcome == "empty_lift_fiber"
        ),
        "parent_failure_count": sum(
            1 for step in trajectory.steps if step.outcome == "parent_failure"
        ),
        "final_state": trajectory.final_state,
        "final_inference": True,
    }


def _build_rank1_policy(config: TowerRunnerConfig) -> TowerTransformerPolicy:
    policy_config = dict(config.policy_config)
    return TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=1,
            action_dim=_policy_int(
                policy_config,
                "action_dim",
                default=2 * config.max_step_size,
            ),
            max_window_len=config.measure_size * config.context_measures,
            d_model=_policy_int(policy_config, "d_model", default=32),
            num_layers=_policy_int(policy_config, "num_layers", default=1),
            num_heads=_policy_int(policy_config, "num_heads", default=4),
            ff_dim=_policy_int(policy_config, "ff_dim", default=64),
            dropout=_policy_float(policy_config, "dropout", default=0.0),
        )
    )


def _build_rank2_policy(config: TowerRunnerConfig) -> TowerTransformerPolicy:
    policy_config = dict(config.policy_config)
    return TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=2,
            input_feature_dim=2,
            action_dim=_policy_int(
                policy_config,
                "action_dim",
                default=2 * config.max_step_size + 1,
            ),
            max_window_len=config.measure_size * config.context_measures,
            d_model=_policy_int(policy_config, "d_model", default=32),
            num_layers=_policy_int(policy_config, "num_layers", default=1),
            num_heads=_policy_int(policy_config, "num_heads", default=4),
            ff_dim=_policy_int(policy_config, "ff_dim", default=64),
            dropout=_policy_float(policy_config, "dropout", default=0.0),
        )
    )


def _build_optimizer(
    *,
    policy: torch.nn.Module,
    config: TowerRunnerConfig,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        policy.parameters(),
        lr=_training_float(config, "learning_rate", default=1e-3),
    )


def _graph_spec_from_config(config: TowerRunnerConfig) -> TowerGraphSpec:
    graph_config = dict(config.graph_config)
    return TowerGraphSpec(
        rank=config.rank,
        pitch_min=_mapping_int(graph_config, "pitch_min", default=0),
        pitch_max=_mapping_int(graph_config, "pitch_max", default=127),
        max_step_size=config.max_step_size,
    )


def _training_int(
    config: TowerRunnerConfig,
    key: str,
    *,
    default: int,
) -> int:
    return _mapping_int(dict(config.training_config), key, default=default)


def _training_float(
    config: TowerRunnerConfig,
    key: str,
    *,
    default: float,
) -> float:
    return _mapping_float(dict(config.training_config), key, default=default)


def _policy_int(
    config: Mapping[str, object],
    key: str,
    *,
    default: int,
) -> int:
    return _mapping_int(config, key, default=default)


def _policy_float(
    config: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    return _mapping_float(config, key, default=default)


def _mapping_int(
    config: Mapping[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value


def _mapping_float(
    config: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number")
    return float(value)
