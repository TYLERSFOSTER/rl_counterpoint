"""Artifact-backed tower training runner contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

import torch

from tower.action.assembly import assemble_action
from tower.graph.actions import action_space, active_lift_choices
from tower.graph.induced import (
    induced_rank1_graph_artifact_path,
    induced_rank2_graph_artifact_path,
    write_induced_rank1_graph_artifact,
    write_induced_rank2_graph_artifact,
)
from tower.graph.legality import is_valid_state, is_valid_transition
from tower.graph.projection import project_state, project_window
from tower.graph.spec import TowerGraphSpec
from tower.policy.base import RankPolicy
from tower.policy.samplers import (
    SamplerResult,
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
    append_reward_diagnostics,
    append_rank_metrics,
)
from tower.train.config import TowerRankConfig, _validate_json_mapping
from tower.train.diagnostics import reward_diagnostics_rows
from tower.train.protocol import (
    TrainEpisodeResult,
    train_rank1_episode_with_artifacts,
    train_rank2_episode_with_artifacts,
    train_rank3_episode_with_artifacts,
)
from tower.train.rollout import RewardFunction, rollout_rank1, rollout_rank2, rollout_rank3
from tower.train.trajectory import TowerTrajectory

TARGET_ROOT_OCTAVE_CHOICES = tuple(range(10))


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
    final_inferences: tuple[FinalInferenceResult, ...]
    final_midi_paths: tuple[Path | None, ...]


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
    final_inferences: tuple[FinalInferenceResult, ...]
    final_midi_paths: tuple[Path | None, ...]


@dataclass(frozen=True)
class Rank3TrainingRunResult:
    """Artifacts and summaries from one rank-3 runner invocation."""

    config: TowerRunnerConfig
    rank_config: TowerRankConfig
    paths: TowerArtifactPaths
    grandparent_policy: torch.nn.Module
    parent_policy: torch.nn.Module
    child_policy: torch.nn.Module
    episode_results: tuple[TrainEpisodeResult, ...]
    final_inference: FinalInferenceResult
    final_midi_path: Path | None
    final_inferences: tuple[FinalInferenceResult, ...]
    final_midi_paths: tuple[Path | None, ...]


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
    key_pitch_class = _optional_reward_int(config, "key_pitch_class")
    target_root_octave = _optional_reward_int(config, "target_root_octave")
    episode_results = []
    for episode_index in range(config.episode_count):
        episode_target_root_octave = _rank1_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=config,
            generator=generator,
        )
        episode_initial_state = _rank1_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=episode_target_root_octave,
            spec=spec,
            config=config,
            generator=generator,
        )
        episode_result = train_rank1_episode_with_artifacts(
            policy=active_policy,  # type: ignore[arg-type]
            optimizer=active_optimizer,
            config=rank_config,
            paths=paths,
            initial_state=episode_initial_state,
            reward_fn=reward_fn,
            episode_index=episode_index,
            graph_spec=spec,
            key_pitch_class=key_pitch_class,
            target_root_octave=episode_target_root_octave,
            generator=generator,
        )
        episode_results.append(episode_result)
        if _training_bool(config, "log_reward_diagnostics", default=True):
            append_reward_diagnostics(
                paths=paths,
                rows=reward_diagnostics_rows(
                    trajectory=episode_result.trajectory,
                    lineage_id=config.lineage_id,
                    episode_index=episode_index,
                    episode_kind="training",
                ),
            )
        _maybe_print_episode_progress(
            config=config,
            episode_index=episode_index,
            metrics=episode_result.metrics,
        )

    final_inferences: list[FinalInferenceResult] = []
    final_midi_paths: list[Path | None] = []
    max_steps = _training_int(config, "max_steps", default=1)
    final_inference_sample_target_root_octave = _training_bool(
        config,
        "final_inference_sample_target_root_octave",
        default=_training_bool(config, "sample_target_root_octave", default=False),
    )
    final_inference_sample_initial_state = _training_bool(
        config,
        "final_inference_sample_initial_state",
        default=_training_bool(config, "sample_initial_pitch", default=False),
    )
    for final_inference_index in range(4):
        final_target_root_octave = _rank1_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=config,
            generator=generator,
            force_sample=final_inference_sample_target_root_octave,
        )
        final_initial_state = _rank1_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=final_target_root_octave,
            spec=spec,
            config=config,
            generator=generator,
            force_sample=final_inference_sample_initial_state,
        )
        final_inference = run_final_inference_episode(
            policy=active_policy,  # type: ignore[arg-type]
            initial_state=final_initial_state,
            reward_fn=reward_fn,
            max_steps=max_steps,
            graph_spec=spec,
            measure_size=config.measure_size,
            context_measures=config.context_measures,
            key_pitch_class=key_pitch_class,
            target_root_octave=final_target_root_octave,
            sampling_temperature=_training_float(
                config,
                "sampling_temperature",
                default=1.0,
            ),
            sampling_uniform_mix=_training_float(
                config,
                "sampling_uniform_mix",
                default=0.0,
            ),
            generator=generator,
        )
        final_midi_path = _write_final_midi(
            config=config,
            paths=paths,
            final_inference=final_inference,
            final_inference_index=final_inference_index,
        )
        _append_final_inference_artifacts(
            paths=paths,
            final_inference=final_inference,
            lineage_id=config.lineage_id,
            episode_index=config.episode_count + final_inference_index,
            final_inference_index=final_inference_index,
            final_midi_path=final_midi_path,
        )
        final_inferences.append(final_inference)
        final_midi_paths.append(final_midi_path)

    final_inference = final_inferences[0]
    final_midi_path = final_midi_paths[0]

    return Rank1TrainingRunResult(
        config=config,
        rank_config=rank_config,
        paths=paths,
        policy=active_policy,
        episode_results=tuple(episode_results),
        final_inference=final_inference,
        final_midi_path=final_midi_path,
        final_inferences=tuple(final_inferences),
        final_midi_paths=tuple(final_midi_paths),
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
    key_pitch_class = _optional_reward_int(config, "key_pitch_class")
    target_root_octave = _optional_reward_int(config, "target_root_octave")
    episode_results = []
    for episode_index in range(config.episode_count):
        episode_target_root_octave = _rank2_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=config,
            generator=generator,
        )
        episode_initial_state = _rank2_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=episode_target_root_octave,
            spec=spec,
            config=config,
            generator=generator,
        )
        episode_result = train_rank2_episode_with_artifacts(
            parent_policy=parent_policy,  # type: ignore[arg-type]
            child_policy=active_child_policy,  # type: ignore[arg-type]
            child_optimizer=active_child_optimizer,
            config=rank_config,
            paths=paths,
            initial_state=episode_initial_state,
            reward_fn=reward_fn,
            episode_index=episode_index,
            graph_spec=spec,
            key_pitch_class=key_pitch_class,
            target_root_octave=episode_target_root_octave,
            generator=generator,
        )
        episode_results.append(episode_result)
        if _training_bool(config, "log_reward_diagnostics", default=True):
            append_reward_diagnostics(
                paths=paths,
                rows=reward_diagnostics_rows(
                    trajectory=episode_result.trajectory,
                    lineage_id=config.lineage_id,
                    episode_index=episode_index,
                    episode_kind="training",
                ),
            )
        _maybe_print_episode_progress(
            config=config,
            episode_index=episode_index,
            metrics=episode_result.metrics,
        )

    final_inferences = []
    final_midi_paths = []
    max_steps = _training_int(config, "max_steps", default=1)
    final_inference_sample_target_root_octave = _training_bool(
        config,
        "final_inference_sample_target_root_octave",
        default=_training_bool(config, "sample_target_root_octave", default=False),
    )
    final_inference_sample_initial_state = _training_bool(
        config,
        "final_inference_sample_initial_state",
        default=_training_bool(config, "sample_initial_state", default=False),
    )
    for final_inference_index in range(4):
        final_target_root_octave = _rank2_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=config,
            generator=generator,
            force_sample=final_inference_sample_target_root_octave,
        )
        final_initial_state = _rank2_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=final_target_root_octave,
            spec=spec,
            config=config,
            generator=generator,
            force_sample=final_inference_sample_initial_state,
        )
        final_inference = run_final_inference_episode(
            policy=active_child_policy,  # type: ignore[arg-type]
            parent_policy=parent_policy,  # type: ignore[arg-type]
            initial_state=final_initial_state,
            reward_fn=reward_fn,
            max_steps=max_steps,
            graph_spec=spec,
            measure_size=config.measure_size,
            context_measures=config.context_measures,
            parent_top_m=config.parent_top_m,
            key_pitch_class=key_pitch_class,
            target_root_octave=final_target_root_octave,
            sampling_temperature=_training_float(
                config,
                "sampling_temperature",
                default=1.0,
            ),
            sampling_uniform_mix=_training_float(
                config,
                "sampling_uniform_mix",
                default=0.0,
            ),
            generator=generator,
        )
        final_midi_path = _write_final_midi(
            config=config,
            paths=paths,
            final_inference=final_inference,
            final_inference_index=final_inference_index,
        )
        _append_final_inference_artifacts(
            paths=paths,
            final_inference=final_inference,
            lineage_id=config.lineage_id,
            episode_index=config.episode_count + final_inference_index,
            final_inference_index=final_inference_index,
            final_midi_path=final_midi_path,
        )
        final_inferences.append(final_inference)
        final_midi_paths.append(final_midi_path)

    final_inference = final_inferences[0]
    final_midi_path = final_midi_paths[0]

    return Rank2TrainingRunResult(
        config=config,
        rank_config=rank_config,
        paths=paths,
        parent_policy=parent_policy,
        child_policy=active_child_policy,
        episode_results=tuple(episode_results),
        final_inference=final_inference,
        final_midi_path=final_midi_path,
        final_inferences=tuple(final_inferences),
        final_midi_paths=tuple(final_midi_paths),
    )


def run_rank3_training(
    *,
    config: TowerRunnerConfig,
    grandparent_policy: torch.nn.Module,
    parent_policy: torch.nn.Module,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    child_policy: torch.nn.Module | None = None,
    child_optimizer: torch.optim.Optimizer | None = None,
    graph_spec: TowerGraphSpec | None = None,
) -> Rank3TrainingRunResult:
    """Train rank 3 over one accepted rank-2 parent stack."""
    if config.rank != 3:
        raise ValueError("run_rank3_training requires rank-3 runner config")
    if not hasattr(grandparent_policy, "rank") or grandparent_policy.rank != 1:
        raise ValueError("rank-3 runner grandparent policy must have rank 1")
    if not hasattr(parent_policy, "rank") or parent_policy.rank != 2:
        raise ValueError("rank-3 runner parent policy must have rank 2")

    active_child_policy = (
        _build_rank3_policy(config) if child_policy is None else child_policy
    )
    if not hasattr(active_child_policy, "rank") or active_child_policy.rank != 3:
        raise ValueError("rank-3 runner child policy must have rank 3")

    active_child_optimizer = (
        _build_optimizer(policy=active_child_policy, config=config)
        if child_optimizer is None
        else child_optimizer
    )
    paths = config.artifact_paths()
    rank_config = config.to_rank_config()
    spec = graph_spec if graph_spec is not None else _graph_spec_from_config(config)
    if spec.rank != 3:
        raise ValueError("rank-3 runner requires rank-3 graph spec")

    generator = torch.Generator().manual_seed(config.seed)
    key_pitch_class = _optional_reward_int(config, "key_pitch_class")
    target_root_octave = _optional_reward_int(config, "target_root_octave")
    episode_results = []
    for episode_index in range(config.episode_count):
        episode_target_root_octave = _rank3_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=config,
            generator=generator,
        )
        episode_initial_state = _rank3_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=episode_target_root_octave,
            spec=spec,
            config=config,
            generator=generator,
        )
        episode_result = train_rank3_episode_with_artifacts(
            grandparent_policy=grandparent_policy,  # type: ignore[arg-type]
            parent_policy=parent_policy,  # type: ignore[arg-type]
            child_policy=active_child_policy,  # type: ignore[arg-type]
            child_optimizer=active_child_optimizer,
            config=rank_config,
            paths=paths,
            initial_state=episode_initial_state,
            reward_fn=reward_fn,
            episode_index=episode_index,
            graph_spec=spec,
            key_pitch_class=key_pitch_class,
            target_root_octave=episode_target_root_octave,
            generator=generator,
        )
        episode_results.append(episode_result)
        if _training_bool(config, "log_reward_diagnostics", default=True):
            append_reward_diagnostics(
                paths=paths,
                rows=reward_diagnostics_rows(
                    trajectory=episode_result.trajectory,
                    lineage_id=config.lineage_id,
                    episode_index=episode_index,
                    episode_kind="training",
                ),
            )
        _maybe_print_episode_progress(
            config=config,
            episode_index=episode_index,
            metrics=episode_result.metrics,
        )

    final_inferences: list[FinalInferenceResult] = []
    final_midi_paths: list[Path | None] = []
    max_steps = _training_int(config, "max_steps", default=1)
    final_inference_sample_target_root_octave = _training_bool(
        config,
        "final_inference_sample_target_root_octave",
        default=_training_bool(config, "sample_target_root_octave", default=False),
    )
    final_inference_sample_initial_state = _training_bool(
        config,
        "final_inference_sample_initial_state",
        default=_training_bool(config, "sample_initial_state", default=False),
    )
    for final_inference_index in range(4):
        final_target_root_octave = _rank3_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=config,
            generator=generator,
            force_sample=final_inference_sample_target_root_octave,
        )
        final_initial_state = _rank3_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=final_target_root_octave,
            spec=spec,
            config=config,
            generator=generator,
            force_sample=final_inference_sample_initial_state,
        )
        final_inference = run_final_inference_episode(
            policy=active_child_policy,  # type: ignore[arg-type]
            parent_policy=parent_policy,  # type: ignore[arg-type]
            grandparent_policy=grandparent_policy,  # type: ignore[arg-type]
            initial_state=final_initial_state,
            reward_fn=reward_fn,
            max_steps=max_steps,
            graph_spec=spec,
            measure_size=config.measure_size,
            context_measures=config.context_measures,
            parent_top_m=config.parent_top_m,
            key_pitch_class=key_pitch_class,
            target_root_octave=final_target_root_octave,
            sampling_temperature=_training_float(
                config,
                "sampling_temperature",
                default=1.0,
            ),
            sampling_uniform_mix=_training_float(
                config,
                "sampling_uniform_mix",
                default=0.0,
            ),
            generator=generator,
        )
        final_midi_path = _write_final_midi(
            config=config,
            paths=paths,
            final_inference=final_inference,
            final_inference_index=final_inference_index,
        )
        _append_final_inference_artifacts(
            paths=paths,
            final_inference=final_inference,
            lineage_id=config.lineage_id,
            episode_index=config.episode_count + final_inference_index,
            final_inference_index=final_inference_index,
            final_midi_path=final_midi_path,
        )
        final_inferences.append(final_inference)
        final_midi_paths.append(final_midi_path)

    final_inference = final_inferences[0]
    final_midi_path = final_midi_paths[0]

    return Rank3TrainingRunResult(
        config=config,
        rank_config=rank_config,
        paths=paths,
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        child_policy=active_child_policy,
        episode_results=tuple(episode_results),
        final_inference=final_inference,
        final_midi_path=final_midi_path,
        final_inferences=tuple(final_inferences),
        final_midi_paths=tuple(final_midi_paths),
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
    grandparent_policy: RankPolicy | None = None,
    parent_top_m: int = 1,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    sampling_temperature: float = 1.0,
    sampling_uniform_mix: float = 0.0,
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
    if rank not in {1, 2, 3}:
        raise ValueError("final inference currently supports rank 1, rank 2, or rank 3")

    spec = TowerGraphSpec(rank=rank) if graph_spec is None else graph_spec
    if spec.rank != rank:
        raise ValueError("graph spec rank must match policy rank")

    _eval_if_module(policy)
    if parent_policy is not None:
        _eval_if_module(parent_policy)
    if grandparent_policy is not None:
        _eval_if_module(grandparent_policy)

    with torch.no_grad():
        if rank == 1:
            if parent_policy is not None:
                raise ValueError("rank 1 final inference must not set parent_policy")
            if grandparent_policy is not None:
                raise ValueError("rank 1 final inference must not set grandparent_policy")
            trajectory = _run_rank1_final_inference(
                policy=policy,
                initial_state=initial_state,
                reward_fn=reward_fn,
                max_steps=max_steps,
                graph_spec=spec,
                measure_size=measure_size,
                context_measures=context_measures,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
                sampling_temperature=sampling_temperature,
                sampling_uniform_mix=sampling_uniform_mix,
                generator=generator,
            )
        elif rank == 2:
            if parent_policy is None:
                raise ValueError("rank 2 final inference requires parent_policy")
            if parent_policy.rank != 1:
                raise ValueError("rank 2 final inference requires rank-1 parent policy")
            if grandparent_policy is not None:
                raise ValueError("rank 2 final inference must not set grandparent_policy")
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
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
                sampling_temperature=sampling_temperature,
                sampling_uniform_mix=sampling_uniform_mix,
                generator=generator,
            )
        else:
            if parent_policy is None:
                raise ValueError("rank 3 final inference requires parent_policy")
            if parent_policy.rank != 2:
                raise ValueError("rank 3 final inference requires rank-2 parent policy")
            if grandparent_policy is None:
                raise ValueError("rank 3 final inference requires grandparent_policy")
            if grandparent_policy.rank != 1:
                raise ValueError("rank 3 final inference requires rank-1 grandparent policy")
            trajectory = _run_rank3_final_inference(
                grandparent_policy=grandparent_policy,
                parent_policy=parent_policy,
                child_policy=policy,
                initial_state=initial_state,
                reward_fn=reward_fn,
                max_steps=max_steps,
                graph_spec=spec,
                measure_size=measure_size,
                context_measures=context_measures,
                parent_top_m=parent_top_m,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
                sampling_temperature=sampling_temperature,
                sampling_uniform_mix=sampling_uniform_mix,
                generator=generator,
            )

    return FinalInferenceResult(
        trajectory=trajectory,
        metrics={
            **_final_inference_metrics(trajectory=trajectory, rank=rank),
            "initial_state": list(initial_state),
            "target_root_octave": target_root_octave,
        },
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
    key_pitch_class: int | None,
    target_root_octave: int | None,
    sampling_temperature: float,
    sampling_uniform_mix: float,
    generator: torch.Generator | None,
) -> TowerTrajectory:
    def active_sampler(**kwargs: object):
        return sample_active_choice_from_policy(
            policy=policy,
            state=kwargs["state"],  # type: ignore[arg-type]
            window=kwargs["window"],  # type: ignore[arg-type]
            active_choices=kwargs["active_choices"],  # type: ignore[arg-type]
            measure_size=measure_size,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=graph_spec.max_step_size,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
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
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
    )


def _rank1_episode_initial_state(
    *,
    initial_state: tuple[int, ...],
    target_root_octave: int | None,
    spec: TowerGraphSpec,
    config: TowerRunnerConfig,
    generator: torch.Generator,
    force_sample: bool = False,
) -> tuple[int, ...]:
    if not force_sample and not _training_bool(
        config, "sample_initial_pitch", default=False
    ):
        return initial_state

    pitch_min = _training_int(
        config,
        "initial_pitch_min",
        default=max(spec.pitch_min, 36),
    )
    pitch_max = _training_int(
        config,
        "initial_pitch_max",
        default=min(spec.pitch_max, 84),
    )
    pitch_min = max(spec.pitch_min, pitch_min)
    pitch_max = min(spec.pitch_max, pitch_max)
    sample_in_target_octave = _training_bool(
        config,
        "sample_initial_pitch_in_target_octave",
        default=False,
    )
    octave_pitch_min: int | None = None
    octave_pitch_max: int | None = None
    if sample_in_target_octave:
        if target_root_octave is None:
            raise ValueError(
                "target_root_octave is required to sample initial pitch in target octave"
            )
        octave_pitch_min = 12 * (target_root_octave + 1)
        octave_pitch_max = octave_pitch_min + 11
        pitch_min = max(pitch_min, octave_pitch_min)
        pitch_max = min(pitch_max, octave_pitch_max)

    if spec.induced_node_image is not None:
        eligible_pitches = sorted(
            state[0]
            for state in spec.induced_node_image
            if pitch_min <= state[0] <= pitch_max
            and (
                octave_pitch_min is None
                or octave_pitch_max is None
                or octave_pitch_min <= state[0] <= octave_pitch_max
            )
            and is_valid_state(state, spec)
        )
        if not eligible_pitches:
            raise ValueError(
                "initial pitch sampling range must intersect induced rank-1 node image"
            )
        choice_index = int(
            torch.randint(
                low=0,
                high=len(eligible_pitches),
                size=(1,),
                generator=generator,
            ).item()
        )
        return (eligible_pitches[choice_index],)

    if pitch_min > pitch_max:
        raise ValueError("initial pitch sampling range must not be empty")
    eligible_pitches = [
        pitch
        for pitch in range(pitch_min, pitch_max + 1)
        if is_valid_state((pitch,), spec)
    ]
    if not eligible_pitches:
        raise ValueError("initial pitch sampling range must contain a legal rank-1 pitch")
    choice_index = int(
        torch.randint(
            low=0,
            high=len(eligible_pitches),
            size=(1,),
            generator=generator,
        ).item()
    )
    return (eligible_pitches[choice_index],)


def _rank1_episode_target_root_octave(
    *,
    target_root_octave: int | None,
    config: TowerRunnerConfig,
    generator: torch.Generator,
    force_sample: bool = False,
) -> int | None:
    if not force_sample and not _training_bool(
        config, "sample_target_root_octave", default=False
    ):
        return target_root_octave

    choices = _target_root_octave_choices(config)
    choice_index = int(
        torch.randint(
            low=0,
            high=len(choices),
            size=(1,),
            generator=generator,
        ).item()
    )
    return choices[choice_index]


def _rank2_episode_target_root_octave(
    *,
    target_root_octave: int | None,
    config: TowerRunnerConfig,
    generator: torch.Generator,
    force_sample: bool = False,
) -> int | None:
    if not force_sample and not _training_bool(
        config, "sample_target_root_octave", default=False
    ):
        return target_root_octave
    choices = _target_root_octave_choices(config)
    choice_index = int(
        torch.randint(
            low=0,
            high=len(choices),
            size=(1,),
            generator=generator,
        ).item()
    )
    return choices[choice_index]


def _rank3_episode_target_root_octave(
    *,
    target_root_octave: int | None,
    config: TowerRunnerConfig,
    generator: torch.Generator,
    force_sample: bool = False,
) -> int | None:
    if not force_sample and not _training_bool(
        config, "sample_target_root_octave", default=False
    ):
        return target_root_octave
    choices = _target_root_octave_choices(config)
    choice_index = int(
        torch.randint(
            low=0,
            high=len(choices),
            size=(1,),
            generator=generator,
        ).item()
    )
    return choices[choice_index]


def _target_root_octave_choices(config: TowerRunnerConfig) -> tuple[int, ...]:
    raw_choices = config.training_config.get("target_root_octave_choices")
    if raw_choices is None:
        return TARGET_ROOT_OCTAVE_CHOICES
    if not isinstance(raw_choices, (list, tuple)):
        raise TypeError("target_root_octave_choices must be a list or tuple")
    choices = tuple(raw_choices)
    if not choices:
        raise ValueError("target_root_octave_choices must not be empty")
    for choice in choices:
        if isinstance(choice, bool) or not isinstance(choice, int):
            raise TypeError("target_root_octave_choices must contain integers")
        if choice < -1 or choice > 9:
            raise ValueError("target_root_octave_choices must be in [-1, 9]")
    return choices


def _rank2_episode_initial_state(
    *,
    initial_state: tuple[int, ...],
    target_root_octave: int | None,
    spec: TowerGraphSpec,
    config: TowerRunnerConfig,
    generator: torch.Generator,
    force_sample: bool = False,
) -> tuple[int, ...]:
    if not force_sample and not _training_bool(
        config, "sample_initial_state", default=False
    ):
        return initial_state

    parent_pitch_min = _training_int(
        config,
        "initial_parent_pitch_min",
        default=spec.pitch_min,
    )
    parent_pitch_max = _training_int(
        config,
        "initial_parent_pitch_max",
        default=spec.pitch_max,
    )
    parent_pitch_min = max(spec.pitch_min, parent_pitch_min)
    parent_pitch_max = min(spec.pitch_max, parent_pitch_max)

    sample_parent_in_target_octave = _training_bool(
        config,
        "sample_initial_parent_pitch_in_target_octave",
        default=False,
    )
    octave_pitch_min: int | None = None
    octave_pitch_max: int | None = None
    if sample_parent_in_target_octave:
        if target_root_octave is None:
            raise ValueError(
                "target_root_octave is required to sample rank-2 parent pitch in target octave"
            )
        octave_pitch_min = 12 * (target_root_octave + 1)
        octave_pitch_max = octave_pitch_min + 11
        parent_pitch_min = max(parent_pitch_min, octave_pitch_min)
        parent_pitch_max = min(parent_pitch_max, octave_pitch_max)

    if parent_pitch_min > parent_pitch_max:
        raise ValueError("rank-2 initial parent pitch sampling range must not be empty")

    eligible_states = [
        (lower, upper)
        for lower in range(parent_pitch_min, parent_pitch_max + 1)
        for upper in range(lower + 1, spec.pitch_max + 1)
        if (
            (octave_pitch_min is None or octave_pitch_min <= lower <= octave_pitch_max)
            and is_valid_state((lower, upper), spec)
        )
    ]
    if not eligible_states:
        raise ValueError("no legal rank-2 initial states satisfy the requested sampling constraints")

    choice_index = int(
        torch.randint(
            low=0,
            high=len(eligible_states),
            size=(1,),
            generator=generator,
        ).item()
    )
    return eligible_states[choice_index]


def _rank3_episode_initial_state(
    *,
    initial_state: tuple[int, ...],
    target_root_octave: int | None,
    spec: TowerGraphSpec,
    config: TowerRunnerConfig,
    generator: torch.Generator,
    force_sample: bool = False,
) -> tuple[int, ...]:
    if not force_sample and not _training_bool(
        config, "sample_initial_state", default=False
    ):
        return initial_state

    lower_pitch_min = _training_int(
        config,
        "initial_parent_pitch_min",
        default=spec.pitch_min,
    )
    lower_pitch_max = _training_int(
        config,
        "initial_parent_pitch_max",
        default=spec.pitch_max,
    )
    lower_pitch_min = max(spec.pitch_min, lower_pitch_min)
    lower_pitch_max = min(spec.pitch_max, lower_pitch_max)

    sample_lower_in_target_octave = _training_bool(
        config,
        "sample_initial_parent_pitch_in_target_octave",
        default=False,
    )
    octave_pitch_min: int | None = None
    octave_pitch_max: int | None = None
    if sample_lower_in_target_octave:
        if target_root_octave is None:
            raise ValueError(
                "target_root_octave is required to sample rank-3 pedal pitch in target octave"
            )
        octave_pitch_min = 12 * (target_root_octave + 1)
        octave_pitch_max = octave_pitch_min + 11
        lower_pitch_min = max(lower_pitch_min, octave_pitch_min)
        lower_pitch_max = min(lower_pitch_max, octave_pitch_max)

    if lower_pitch_min > lower_pitch_max:
        raise ValueError("rank-3 initial pedal pitch sampling range must not be empty")

    eligible_states = [
        (lower, middle, upper)
        for lower in range(lower_pitch_min, lower_pitch_max + 1)
        for middle in range(lower + 1, spec.pitch_max)
        for upper in range(middle + 1, spec.pitch_max + 1)
        if (
            (octave_pitch_min is None or octave_pitch_min <= lower <= octave_pitch_max)
            and is_valid_state((lower, middle, upper), spec)
        )
    ]
    if not eligible_states:
        raise ValueError("no legal rank-3 initial states satisfy the requested sampling constraints")

    choice_index = int(
        torch.randint(
            low=0,
            high=len(eligible_states),
            size=(1,),
            generator=generator,
        ).item()
    )
    return eligible_states[choice_index]


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
    key_pitch_class: int | None,
    target_root_octave: int | None,
    sampling_temperature: float,
    sampling_uniform_mix: float,
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
        full_state = kwargs["full_state"]  # type: ignore[assignment]
        parent_actions = tuple(
            action
            for action in action_space(
                rank=1,
                max_step_size=parent_spec.max_step_size,
            )
            if is_valid_transition(parent_state, action, parent_spec)  # type: ignore[arg-type]
        )
        feasible_parent_actions = tuple(
            action
            for action in parent_actions
            if active_lift_choices(
                state=full_state,  # type: ignore[arg-type]
                parent_action=action,
                spec=graph_spec,
            )
        )
        sampled = sample_parent_top_m_from_policy(
            policy=parent_policy,
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
            max_step_size=graph_spec.max_step_size,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
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
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
    )


def _run_rank3_final_inference(
    *,
    grandparent_policy: RankPolicy,
    parent_policy: RankPolicy,
    child_policy: RankPolicy,
    initial_state: tuple[int, ...],
    reward_fn: RewardFunction,
    max_steps: int,
    graph_spec: TowerGraphSpec,
    measure_size: int,
    context_measures: int,
    parent_top_m: int,
    key_pitch_class: int | None,
    target_root_octave: int | None,
    sampling_temperature: float,
    sampling_uniform_mix: float,
    generator: torch.Generator | None,
) -> TowerTrajectory:
    grandparent_spec = TowerGraphSpec(
        rank=1,
        pitch_min=graph_spec.pitch_min,
        pitch_max=graph_spec.pitch_max,
        max_step_size=graph_spec.max_step_size,
    )
    parent_spec = TowerGraphSpec(
        rank=2,
        pitch_min=graph_spec.pitch_min,
        pitch_max=graph_spec.pitch_max,
        max_step_size=graph_spec.max_step_size,
    )

    def parent_sampler(**kwargs: object):
        parent_state = kwargs["state"]  # type: ignore[assignment]
        full_state = kwargs["full_state"]  # type: ignore[assignment]
        parent_window = kwargs["window"]  # type: ignore[assignment]
        grandparent_state = project_state(parent_state)  # type: ignore[arg-type]
        grandparent_window = project_window(parent_window)  # type: ignore[arg-type]

        def child_feasible_parent_choices(grandparent_action: tuple[int, ...]) -> tuple[int, ...]:
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
                    spec=graph_spec,
                )
            )

        grandparent_actions = tuple(
            action
            for action in action_space(
                rank=1,
                max_step_size=grandparent_spec.max_step_size,
            )
            if is_valid_transition(grandparent_state, action, grandparent_spec)  # type: ignore[arg-type]
        )
        feasible_grandparent_actions = tuple(
            action
            for action in grandparent_actions
            if child_feasible_parent_choices(action)
        )
        sampled_grandparent = sample_parent_top_m_from_policy(
            policy=grandparent_policy,
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
            policy=parent_policy,
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
            max_step_size=graph_spec.max_step_size,
            temperature=sampling_temperature,
            uniform_mix=sampling_uniform_mix,
            generator=generator,
        )

    return rollout_rank3(
        initial_state=initial_state,
        max_steps=max_steps,
        parent_sampler=parent_sampler,
        active_sampler=active_sampler,
        reward_fn=reward_fn,
        graph_spec=graph_spec,
        measure_size=measure_size,
        context_measures=context_measures,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
    )


def _eval_if_module(policy: RankPolicy) -> None:
    if isinstance(policy, torch.nn.Module):
        policy.eval()


def _write_final_midi(
    *,
    config: TowerRunnerConfig,
    paths: TowerArtifactPaths,
    final_inference: FinalInferenceResult,
    final_inference_index: int,
) -> Path | None:
    if not config.final_midi_enabled:
        return None
    return write_trajectory_to_midi(
        trajectory=final_inference.trajectory,
        path=_final_midi_path(paths=paths, final_inference_index=final_inference_index),
    )


def _append_final_inference_artifacts(
    *,
    paths: TowerArtifactPaths,
    final_inference: FinalInferenceResult,
    lineage_id: str,
    episode_index: int,
    final_inference_index: int,
    final_midi_path: Path | None,
) -> None:
    append_rank_metrics(
        paths=paths,
        metrics={
            **dict(final_inference.metrics),
            "episode_index": episode_index,
            "kind": "final_inference",
            "final_inference_index": final_inference_index,
            "midi_path": None
            if final_midi_path is None
            else final_midi_path.relative_to(paths.lineage_dir).as_posix(),
        },
    )
    append_reward_diagnostics(
        paths=paths,
        rows=reward_diagnostics_rows(
            trajectory=final_inference.trajectory,
            lineage_id=lineage_id,
            episode_index=episode_index,
            episode_kind="final_inference",
        ),
    )


def _final_midi_path(
    *,
    paths: TowerArtifactPaths,
    final_inference_index: int,
) -> Path:
    if final_inference_index == 0:
        return paths.example_episode_path
    return paths.rank_dir / f"example_episode_{final_inference_index}.mid"


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
        "tail_stagnation_count": sum(
            1 for step in trajectory.steps if step.outcome == "tail_stagnation"
        ),
        "final_state": trajectory.final_state,
        "final_inference": True,
    }


def _build_rank1_policy(config: TowerRunnerConfig) -> TowerTransformerPolicy:
    policy_config = dict(config.policy_config)
    return TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=_policy_input_feature_dim(config, default_rank=1),
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
            input_feature_dim=_policy_input_feature_dim(config, default_rank=2),
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


def _build_rank3_policy(config: TowerRunnerConfig) -> TowerTransformerPolicy:
    policy_config = dict(config.policy_config)
    return TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=3,
            input_feature_dim=_policy_input_feature_dim(config, default_rank=3),
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
    key_pitch_class = _optional_reward_int(config, "key_pitch_class")
    pitch_min = _mapping_int(graph_config, "pitch_min", default=0)
    requested_pitch_max = _mapping_int(graph_config, "pitch_max", default=127)
    final_rank = _mapping_int(graph_config, "final_rank", default=config.rank)

    if final_rank == 3 and config.rank in {1, 2}:
        induced_rank2_spec = _induced_rank2_spec_from_rank3_config(
            graph_config=graph_config,
            artifact_root=config.artifact_root,
            key_pitch_class=0 if key_pitch_class is None else key_pitch_class,
            default_pitch_min=pitch_min,
            default_pitch_max=requested_pitch_max,
            default_max_step_size=config.max_step_size,
        )
        if config.rank == 2:
            return induced_rank2_spec
        return _induced_rank1_spec_from_rank2_spec(
            source_spec=induced_rank2_spec,
            artifact_root=config.artifact_root,
            key_pitch_class=0 if key_pitch_class is None else key_pitch_class,
            max_step_size=config.max_step_size,
        )

    if (
        config.rank == 1
        and _mapping_bool(graph_config, "use_induced_rank1_graph", default=True)
    ):
        source_spec = TowerGraphSpec(
            rank=2,
            key_pitch_class=0 if key_pitch_class is None else key_pitch_class,
            pitch_min=_mapping_int(
                graph_config,
                "induced_rank2_pitch_min",
                default=pitch_min,
            ),
            pitch_max=_mapping_int(
                graph_config,
                "induced_rank2_pitch_max",
                default=requested_pitch_max,
            ),
            max_step_size=_mapping_int(
                graph_config,
                "induced_rank2_max_step_size",
                default=config.max_step_size,
            ),
        )
        return _induced_rank1_spec_from_rank2_spec(
            source_spec=source_spec,
            artifact_root=config.artifact_root,
            key_pitch_class=0 if key_pitch_class is None else key_pitch_class,
            max_step_size=config.max_step_size,
        )
    effective_pitch_max = _effective_pitch_max_for_rank(
        rank=config.rank,
        requested_pitch_max=requested_pitch_max,
        graph_config=graph_config,
    )
    return TowerGraphSpec(
        rank=config.rank,
        key_pitch_class=0 if key_pitch_class is None else key_pitch_class,
        pitch_min=pitch_min,
        pitch_max=effective_pitch_max,
        max_step_size=config.max_step_size,
    )


def _coerce_rank1_state(value: object) -> tuple[int]:
    if not isinstance(value, list):
        raise TypeError("induced rank1 graph state entries must be lists")
    if len(value) != 1:
        raise ValueError("induced rank1 graph states must have length 1")
    pitch = value[0]
    if not isinstance(pitch, int):
        raise TypeError("induced rank1 graph pitches must be ints")
    return (pitch,)


def _coerce_rank1_edge(value: object) -> tuple[tuple[int], tuple[int]]:
    if not isinstance(value, dict):
        raise TypeError("induced rank1 graph edge entries must be objects")
    return (
        _coerce_rank1_state(value["source"]),
        _coerce_rank1_state(value["target"]),
    )


def _induced_rank2_spec_from_rank3_config(
    *,
    graph_config: Mapping[str, object],
    artifact_root: Path,
    key_pitch_class: int,
    default_pitch_min: int,
    default_pitch_max: int,
    default_max_step_size: int,
) -> TowerGraphSpec:
    source_spec = TowerGraphSpec(
        rank=3,
        key_pitch_class=key_pitch_class,
        pitch_min=_mapping_int(graph_config, "induced_rank3_pitch_min", default=default_pitch_min),
        pitch_max=_mapping_int(graph_config, "induced_rank3_pitch_max", default=default_pitch_max),
        max_step_size=_mapping_int(
            graph_config,
            "induced_rank3_max_step_size",
            default=default_max_step_size,
        ),
    )
    artifact_path = induced_rank2_graph_artifact_path(
        source_spec=source_spec,
        artifact_root=artifact_root,
    )
    if not artifact_path.exists():
        write_induced_rank2_graph_artifact(
            source_spec=source_spec,
            artifact_root=artifact_root,
        )
    payload = _read_induced_graph_payload(
        artifact_path=artifact_path,
        rank_label="rank2",
    )
    induced_node_image = frozenset(
        _coerce_rank2_state(node)
        for node in _payload_list(payload, "node_image", "induced rank2 graph")
    )
    induced_edge_image = frozenset(
        _coerce_rank2_edge(edge)
        for edge in _payload_list(payload, "edge_image", "induced rank2 graph")
    )
    if not induced_node_image:
        raise ValueError("induced rank2 graph node_image must not be empty")
    if not induced_edge_image:
        raise ValueError("induced rank2 graph edge_image must not be empty")
    induced_pitch_min = min(pitch for state in induced_node_image for pitch in state)
    induced_pitch_max = max(pitch for state in induced_node_image for pitch in state)
    return TowerGraphSpec(
        rank=2,
        key_pitch_class=key_pitch_class,
        pitch_min=induced_pitch_min,
        pitch_max=induced_pitch_max,
        max_step_size=default_max_step_size,
        induced_node_image=induced_node_image,
        induced_edge_image=induced_edge_image,
    )


def _induced_rank1_spec_from_rank2_spec(
    *,
    source_spec: TowerGraphSpec,
    artifact_root: Path,
    key_pitch_class: int,
    max_step_size: int,
) -> TowerGraphSpec:
    artifact_path = induced_rank1_graph_artifact_path(
        source_spec=source_spec,
        artifact_root=artifact_root,
    )
    if not artifact_path.exists():
        write_induced_rank1_graph_artifact(
            source_spec=source_spec,
            artifact_root=artifact_root,
        )
    payload = _read_induced_graph_payload(
        artifact_path=artifact_path,
        rank_label="rank1",
    )
    induced_node_image = frozenset(
        _coerce_rank1_state(node)
        for node in _payload_list(payload, "node_image", "induced rank1 graph")
    )
    induced_edge_image = frozenset(
        _coerce_rank1_edge(edge)
        for edge in _payload_list(payload, "edge_image", "induced rank1 graph")
    )
    if not induced_node_image:
        raise ValueError("induced rank1 graph node_image must not be empty")
    if not induced_edge_image:
        raise ValueError("induced rank1 graph edge_image must not be empty")
    induced_pitch_min = min(state[0] for state in induced_node_image)
    induced_pitch_max = max(state[0] for state in induced_node_image)
    return TowerGraphSpec(
        rank=1,
        key_pitch_class=key_pitch_class,
        pitch_min=induced_pitch_min,
        pitch_max=induced_pitch_max,
        max_step_size=max_step_size,
        induced_node_image=induced_node_image,
        induced_edge_image=induced_edge_image,
    )


def _read_induced_graph_payload(*, artifact_path: Path, rank_label: str) -> dict[str, object]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"induced {rank_label} graph artifact must contain an object")
    return payload


def _payload_list(payload: Mapping[str, object], key: str, label: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{label} {key} must be a list")
    return value


def _coerce_rank2_state(value: object) -> tuple[int, int]:
    if not isinstance(value, list):
        raise TypeError("induced rank2 graph state entries must be lists")
    if len(value) != 2:
        raise ValueError("induced rank2 graph states must have length 2")
    left, right = value
    if not isinstance(left, int) or not isinstance(right, int):
        raise TypeError("induced rank2 graph pitches must be ints")
    return (left, right)


def _coerce_rank2_edge(value: object) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(value, dict):
        raise TypeError("induced rank2 graph edge entries must be objects")
    return (
        _coerce_rank2_state(value["source"]),
        _coerce_rank2_state(value["target"]),
    )


def _effective_pitch_max_for_rank(
    *,
    rank: int,
    requested_pitch_max: int,
    graph_config: Mapping[str, object],
) -> int:
    final_chord_size = graph_config.get("final_chord_size")
    if final_chord_size is not None and rank == 1:
        if not isinstance(final_chord_size, int):
            raise TypeError("final_chord_size must be an int")
        if final_chord_size < 1:
            raise ValueError("final_chord_size must be at least 1")
        reserved_upper_semitones_per_voice = graph_config.get(
            "reserved_upper_semitones_per_voice",
            5,
        )
        if not isinstance(reserved_upper_semitones_per_voice, int):
            raise TypeError("reserved_upper_semitones_per_voice must be an int")
        if reserved_upper_semitones_per_voice < 0:
            raise ValueError(
                "reserved_upper_semitones_per_voice must be non-negative"
            )
        return min(
            requested_pitch_max,
            127 - reserved_upper_semitones_per_voice * final_chord_size,
        )

    target_max_rank = graph_config.get("target_max_rank")
    if target_max_rank is None:
        return requested_pitch_max
    if not isinstance(target_max_rank, int):
        raise TypeError("target_max_rank must be an int")
    if target_max_rank < rank:
        raise ValueError("target_max_rank must be at least the current rank")

    reserved_semitones_per_future_voice = graph_config.get(
        "reserved_semitones_per_future_voice",
        4,
    )
    if not isinstance(reserved_semitones_per_future_voice, int):
        raise TypeError("reserved_semitones_per_future_voice must be an int")
    if reserved_semitones_per_future_voice < 0:
        raise ValueError("reserved_semitones_per_future_voice must be non-negative")

    future_voice_count = target_max_rank - rank
    reserved_semitones = (
        reserved_semitones_per_future_voice * future_voice_count
    )
    return min(requested_pitch_max, 127 - reserved_semitones)


def _training_int(
    config: TowerRunnerConfig,
    key: str,
    *,
    default: int,
) -> int:
    return _mapping_int(dict(config.training_config), key, default=default)


def _training_bool(
    config: TowerRunnerConfig,
    key: str,
    *,
    default: bool,
) -> bool:
    value = dict(config.training_config).get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a bool")
    return value


def _training_float(
    config: TowerRunnerConfig,
    key: str,
    *,
    default: float,
) -> float:
    return _mapping_float(dict(config.training_config), key, default=default)


def _maybe_print_episode_progress(
    *,
    config: TowerRunnerConfig,
    episode_index: int,
    metrics: Mapping[str, object],
) -> None:
    progress_every = _training_int(config, "progress_every", default=0)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if progress_every == 0:
        return
    completed = episode_index + 1
    if completed % progress_every != 0 and completed != config.episode_count:
        return
    label = dict(config.training_config).get("progress_label", config.lineage_id)
    if not isinstance(label, str):
        raise TypeError("progress_label must be a string")
    episode_return = float(metrics["episode_return"])
    episode_length = int(metrics["episode_length"])
    terminal_success = bool(metrics["terminal_success"])
    print(
        f"[progress] {label} {completed}/{config.episode_count} "
        f"return={episode_return:.4f} length={episode_length} "
        f"terminal_success={terminal_success}",
        flush=True,
    )


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


def _policy_input_feature_dim(
    config: TowerRunnerConfig,
    *,
    default_rank: int,
) -> int:
    policy_config = dict(config.policy_config)
    if "input_feature_dim" in policy_config:
        return _policy_int(policy_config, "input_feature_dim", default=default_rank)
    return (
        default_rank
        + 4
        + (1 if _optional_reward_int(config, "key_pitch_class") is not None else 0)
        + (1 if _optional_reward_int(config, "target_root_octave") is not None else 0)
        + 1
    )


def _optional_reward_int(
    config: TowerRunnerConfig,
    key: str,
) -> int | None:
    value = dict(config.reward_config).get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"reward_config.{key} must be an int")
    return value


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


def _sum_logprobs(
    left: float | torch.Tensor | None,
    right: float | torch.Tensor | None,
) -> float | torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _mapping_bool(
    config: Mapping[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a bool")
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
