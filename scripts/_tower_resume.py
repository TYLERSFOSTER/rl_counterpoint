"""Reusable resume helpers for interrupted tower training lineages."""

from __future__ import annotations

from pathlib import Path

import torch

from scripts.tower_train_rank2 import _load_parent_policy
from scripts.tower_train_rank3 import _load_parent_stack
from scripts.tower_train_staged import _build_rank1_reward_from_config
from tower.reward.factory import build_rank2_reward_fn, build_rank3_reward_fn
from tower.train.checkpoint import (
    TowerArtifactPaths,
    append_reward_diagnostics,
    load_latest_checkpoint,
    read_rank_config,
)
from tower.train.diagnostics import reward_diagnostics_rows
from tower.train.runner import (
    TowerRunnerConfig,
    _append_final_inference_artifacts,
    _build_optimizer,
    _build_rank1_policy,
    _build_rank2_policy,
    _build_rank3_policy,
    _episode_graph_specs_by_key_pitch_class,
    _graph_spec_from_config,
    _optional_reward_int,
    _rank1_episode_initial_state,
    _rank1_episode_target_root_octave,
    _rank2_episode_initial_state,
    _rank2_episode_key_pitch_class,
    _rank2_episode_target_root_octave,
    _rank3_episode_initial_state,
    _rank3_episode_key_pitch_class,
    _rank3_episode_target_root_octave,
    _training_bool,
    _training_float,
    _training_int,
    _write_final_midi,
    run_final_inference_episode,
)
from tower.train.protocol import (
    train_rank1_episode_with_artifacts,
    train_rank2_episode_with_artifacts,
    train_rank3_episode_with_artifacts,
)


def resume_rank1_lineage(
    *,
    lineage_id: str,
    artifact_root: Path,
    initial_state: tuple[int, ...],
) -> None:
    """Resume one interrupted rank-1 lineage from checkpoint to completion."""
    paths = TowerArtifactPaths(lineage_id=lineage_id, rank=1, artifact_root=artifact_root)
    rank_config = read_rank_config(paths)
    checkpoint = load_latest_checkpoint(paths)
    last_episode_index = _checkpoint_episode_index(checkpoint)

    runner_config = _runner_config_from_rank_config(
        rank_config=rank_config,
        artifact_root=artifact_root,
    )
    reward_fn = _build_rank1_reward_from_config(dict(rank_config.reward_config))
    policy = _build_rank1_policy(runner_config)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer = _build_optimizer(policy=policy, config=runner_config)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    graph_spec = _graph_spec_from_config(runner_config)

    generator = torch.Generator().manual_seed(runner_config.seed)
    key_pitch_class = _optional_reward_int(runner_config, "key_pitch_class")
    target_root_octave = _optional_reward_int(runner_config, "target_root_octave")

    for episode_index in range(last_episode_index + 1, rank_config.episode_budget):
        episode_target_root_octave = _rank1_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
        )
        episode_initial_state = _rank1_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=episode_target_root_octave,
            spec=graph_spec,
            config=runner_config,
            generator=generator,
        )
        episode_result = train_rank1_episode_with_artifacts(
            policy=policy,
            optimizer=optimizer,
            config=rank_config,
            paths=paths,
            initial_state=episode_initial_state,
            reward_fn=reward_fn,
            episode_index=episode_index,
            graph_spec=graph_spec,
            key_pitch_class=key_pitch_class,
            target_root_octave=episode_target_root_octave,
            generator=generator,
        )
        _append_training_reward_diagnostics(
            runner_config=runner_config,
            paths=paths,
            trajectory=episode_result.trajectory,
            episode_index=episode_index,
        )
        _maybe_print_resume_progress(
            runner_config=runner_config,
            episode_index=episode_index,
            episode_budget=rank_config.episode_budget,
            metrics=episode_result.metrics,
        )

    _append_rank1_final_inference_examples(
        runner_config=runner_config,
        rank_config=rank_config,
        paths=paths,
        policy=policy,
        reward_fn=reward_fn,
        initial_state=initial_state,
        graph_spec=graph_spec,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        generator=generator,
    )


def resume_rank2_lineage(
    *,
    lineage_id: str,
    artifact_root: Path,
    initial_state: tuple[int, int],
) -> None:
    """Resume one interrupted rank-2 lineage from checkpoint to completion."""
    paths = TowerArtifactPaths(lineage_id=lineage_id, rank=2, artifact_root=artifact_root)
    rank_config = read_rank_config(paths)
    checkpoint = load_latest_checkpoint(paths)
    last_episode_index = _checkpoint_episode_index(checkpoint)

    runner_config = _runner_config_from_rank_config(
        rank_config=rank_config,
        artifact_root=artifact_root,
    )
    reward_fn = build_rank2_reward_fn(**_rank2_reward_kwargs(rank_config.reward_config))
    parent_policy, _ = _load_parent_policy(lineage_id=lineage_id, artifact_root=artifact_root)
    child_policy = _build_rank2_policy(runner_config)
    child_policy.load_state_dict(checkpoint["policy_state_dict"])
    child_optimizer = _build_optimizer(policy=child_policy, config=runner_config)
    child_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    graph_spec = _graph_spec_from_config(runner_config)

    generator = torch.Generator().manual_seed(runner_config.seed)
    key_pitch_class = _optional_reward_int(runner_config, "key_pitch_class")
    target_root_octave = _optional_reward_int(runner_config, "target_root_octave")
    specs_by_key_pitch_class = _episode_graph_specs_by_key_pitch_class(
        config=runner_config,
        fallback_spec=graph_spec,
    )

    for episode_index in range(last_episode_index + 1, rank_config.episode_budget):
        episode_key_pitch_class = _rank2_episode_key_pitch_class(
            key_pitch_class=key_pitch_class,
            config=runner_config,
            generator=generator,
        )
        episode_spec = specs_by_key_pitch_class[
            0 if episode_key_pitch_class is None else episode_key_pitch_class
        ]
        episode_target_root_octave = _rank2_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
        )
        episode_initial_state = _rank2_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=episode_target_root_octave,
            spec=episode_spec,
            config=runner_config,
            generator=generator,
        )
        episode_result = train_rank2_episode_with_artifacts(
            parent_policy=parent_policy,
            child_policy=child_policy,
            child_optimizer=child_optimizer,
            config=rank_config,
            paths=paths,
            initial_state=episode_initial_state,
            reward_fn=reward_fn,
            episode_index=episode_index,
            graph_spec=episode_spec,
            key_pitch_class=episode_key_pitch_class,
            target_root_octave=episode_target_root_octave,
            generator=generator,
        )
        _append_training_reward_diagnostics(
            runner_config=runner_config,
            paths=paths,
            trajectory=episode_result.trajectory,
            episode_index=episode_index,
        )
        _maybe_print_resume_progress(
            runner_config=runner_config,
            episode_index=episode_index,
            episode_budget=rank_config.episode_budget,
            metrics=episode_result.metrics,
        )

    _append_rank2_final_inference_examples(
        runner_config=runner_config,
        rank_config=rank_config,
        paths=paths,
        parent_policy=parent_policy,
        child_policy=child_policy,
        reward_fn=reward_fn,
        initial_state=initial_state,
        graph_spec=graph_spec,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        specs_by_key_pitch_class=specs_by_key_pitch_class,
        generator=generator,
    )


def resume_rank3_lineage(
    *,
    lineage_id: str,
    artifact_root: Path,
    initial_state: tuple[int, int, int],
) -> None:
    """Resume one interrupted rank-3 lineage from checkpoint to completion."""
    paths = TowerArtifactPaths(lineage_id=lineage_id, rank=3, artifact_root=artifact_root)
    rank_config = read_rank_config(paths)
    checkpoint = load_latest_checkpoint(paths)
    last_episode_index = _checkpoint_episode_index(checkpoint)

    runner_config = _runner_config_from_rank_config(
        rank_config=rank_config,
        artifact_root=artifact_root,
    )
    reward_fn = build_rank3_reward_fn(**_rank3_reward_kwargs(rank_config.reward_config))
    (
        grandparent_policy,
        parent_policy,
        _grandparent_checkpoint,
        _parent_checkpoint,
    ) = _load_parent_stack(lineage_id=lineage_id, artifact_root=artifact_root)
    child_policy = _build_rank3_policy(runner_config)
    child_policy.load_state_dict(checkpoint["policy_state_dict"])
    child_optimizer = _build_optimizer(policy=child_policy, config=runner_config)
    child_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    graph_spec = _graph_spec_from_config(runner_config)

    generator = torch.Generator().manual_seed(runner_config.seed)
    key_pitch_class = _optional_reward_int(runner_config, "key_pitch_class")
    target_root_octave = _optional_reward_int(runner_config, "target_root_octave")
    specs_by_key_pitch_class = _episode_graph_specs_by_key_pitch_class(
        config=runner_config,
        fallback_spec=graph_spec,
    )

    for episode_index in range(last_episode_index + 1, rank_config.episode_budget):
        episode_key_pitch_class = _rank3_episode_key_pitch_class(
            key_pitch_class=key_pitch_class,
            config=runner_config,
            generator=generator,
        )
        episode_spec = specs_by_key_pitch_class[
            0 if episode_key_pitch_class is None else episode_key_pitch_class
        ]
        episode_target_root_octave = _rank3_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
        )
        episode_initial_state = _rank3_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=episode_target_root_octave,
            spec=episode_spec,
            config=runner_config,
            generator=generator,
        )
        episode_result = train_rank3_episode_with_artifacts(
            grandparent_policy=grandparent_policy,
            parent_policy=parent_policy,
            child_policy=child_policy,
            child_optimizer=child_optimizer,
            config=rank_config,
            paths=paths,
            initial_state=episode_initial_state,
            reward_fn=reward_fn,
            episode_index=episode_index,
            graph_spec=episode_spec,
            key_pitch_class=episode_key_pitch_class,
            target_root_octave=episode_target_root_octave,
            generator=generator,
        )
        _append_training_reward_diagnostics(
            runner_config=runner_config,
            paths=paths,
            trajectory=episode_result.trajectory,
            episode_index=episode_index,
        )
        _maybe_print_resume_progress(
            runner_config=runner_config,
            episode_index=episode_index,
            episode_budget=rank_config.episode_budget,
            metrics=episode_result.metrics,
        )

    _append_rank3_final_inference_examples(
        runner_config=runner_config,
        rank_config=rank_config,
        paths=paths,
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        child_policy=child_policy,
        reward_fn=reward_fn,
        initial_state=initial_state,
        graph_spec=graph_spec,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        specs_by_key_pitch_class=specs_by_key_pitch_class,
        generator=generator,
    )


def _runner_config_from_rank_config(
    *,
    rank_config,
    artifact_root: Path,
) -> TowerRunnerConfig:
    return TowerRunnerConfig(
        lineage_id=rank_config.lineage_id,
        rank=rank_config.rank,
        episode_count=rank_config.episode_budget,
        seed=int(rank_config.seed_config.get("seed", 0)),
        artifact_root=artifact_root,
        measure_size=rank_config.measure_size,
        context_measures=rank_config.context_measures,
        max_step_size=rank_config.max_step_size,
        parent_checkpoint=rank_config.parent_checkpoint,
        parent_top_m=rank_config.parent_sampler_config.get("top_m", 3),
        reward_config=dict(rank_config.reward_config),
        graph_config=dict(rank_config.graph_config),
        policy_config=dict(rank_config.policy_config),
        training_config=dict(rank_config.training_config),
    )


def _checkpoint_episode_index(checkpoint: dict[str, object]) -> int:
    episode_index = checkpoint["episode_index"]
    if not isinstance(episode_index, int):
        raise TypeError("checkpoint episode_index must be an int")
    return episode_index


def _append_training_reward_diagnostics(
    *,
    runner_config: TowerRunnerConfig,
    paths: TowerArtifactPaths,
    trajectory,
    episode_index: int,
) -> None:
    if not _training_bool(runner_config, "log_reward_diagnostics", default=True):
        return
    append_reward_diagnostics(
        paths=paths,
        rows=reward_diagnostics_rows(
            trajectory=trajectory,
            lineage_id=runner_config.lineage_id,
            episode_index=episode_index,
            episode_kind="training",
        ),
    )


def _maybe_print_resume_progress(
    *,
    runner_config: TowerRunnerConfig,
    episode_index: int,
    episode_budget: int,
    metrics,
) -> None:
    progress_every = _training_int(runner_config, "progress_every", default=0)
    if progress_every <= 0:
        return
    completed = episode_index + 1
    if completed % progress_every != 0 and completed != episode_budget:
        return
    label = runner_config.training_config.get("progress_label", runner_config.lineage_id)
    print(
        f"[progress] {label} {completed}/{episode_budget} "
        f"return={metrics['episode_return']:.4f} "
        f"length={metrics['episode_length']} "
        f"terminal_success={metrics['terminal_success']}",
        flush=True,
    )


def _append_rank1_final_inference_examples(
    *,
    runner_config: TowerRunnerConfig,
    rank_config,
    paths: TowerArtifactPaths,
    policy,
    reward_fn,
    initial_state: tuple[int, ...],
    graph_spec,
    key_pitch_class: int | None,
    target_root_octave: int | None,
    generator: torch.Generator,
) -> None:
    max_steps = _training_int(runner_config, "max_steps", default=1)
    for final_inference_index in range(4):
        final_target_root_octave = _rank1_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_target_root_octave",
                default=_training_bool(runner_config, "sample_target_root_octave", default=False),
            ),
        )
        final_initial_state = _rank1_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=final_target_root_octave,
            spec=graph_spec,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_initial_state",
                default=_training_bool(runner_config, "sample_initial_pitch", default=False),
            ),
        )
        final_inference = run_final_inference_episode(
            policy=policy,
            initial_state=final_initial_state,
            reward_fn=reward_fn,
            max_steps=max_steps,
            graph_spec=graph_spec,
            measure_size=runner_config.measure_size,
            context_measures=runner_config.context_measures,
            key_pitch_class=key_pitch_class,
            target_root_octave=final_target_root_octave,
            sampling_temperature=_training_float(runner_config, "sampling_temperature", default=1.0),
            sampling_uniform_mix=_training_float(runner_config, "sampling_uniform_mix", default=0.0),
            generator=generator,
        )
        final_midi_path = _write_final_midi(
            config=runner_config,
            paths=paths,
            final_inference=final_inference,
            final_inference_index=final_inference_index,
        )
        _append_final_inference_artifacts(
            paths=paths,
            final_inference=final_inference,
            lineage_id=runner_config.lineage_id,
            episode_index=rank_config.episode_budget + final_inference_index,
            final_inference_index=final_inference_index,
            final_midi_path=final_midi_path,
        )


def _append_rank2_final_inference_examples(
    *,
    runner_config: TowerRunnerConfig,
    rank_config,
    paths: TowerArtifactPaths,
    parent_policy,
    child_policy,
    reward_fn,
    initial_state: tuple[int, int],
    graph_spec,
    key_pitch_class: int | None,
    target_root_octave: int | None,
    specs_by_key_pitch_class,
    generator: torch.Generator,
) -> None:
    max_steps = _training_int(runner_config, "max_steps", default=1)
    for final_inference_index in range(4):
        final_key_pitch_class = _rank2_episode_key_pitch_class(
            key_pitch_class=key_pitch_class,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_key_pitch_class",
                default=_training_bool(runner_config, "sample_key_pitch_class", default=False),
            ),
        )
        final_spec = specs_by_key_pitch_class[
            0 if final_key_pitch_class is None else final_key_pitch_class
        ]
        final_target_root_octave = _rank2_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_target_root_octave",
                default=_training_bool(runner_config, "sample_target_root_octave", default=False),
            ),
        )
        final_initial_state = _rank2_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=final_target_root_octave,
            spec=final_spec,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_initial_state",
                default=_training_bool(runner_config, "sample_initial_state", default=False),
            ),
        )
        final_inference = run_final_inference_episode(
            policy=child_policy,
            parent_policy=parent_policy,
            initial_state=final_initial_state,
            reward_fn=reward_fn,
            max_steps=max_steps,
            graph_spec=final_spec,
            measure_size=runner_config.measure_size,
            context_measures=runner_config.context_measures,
            parent_top_m=runner_config.parent_top_m,
            key_pitch_class=final_key_pitch_class,
            target_root_octave=final_target_root_octave,
            sampling_temperature=_training_float(runner_config, "sampling_temperature", default=1.0),
            sampling_uniform_mix=_training_float(runner_config, "sampling_uniform_mix", default=0.0),
            generator=generator,
        )
        final_midi_path = _write_final_midi(
            config=runner_config,
            paths=paths,
            final_inference=final_inference,
            final_inference_index=final_inference_index,
        )
        _append_final_inference_artifacts(
            paths=paths,
            final_inference=final_inference,
            lineage_id=runner_config.lineage_id,
            episode_index=rank_config.episode_budget + final_inference_index,
            final_inference_index=final_inference_index,
            final_midi_path=final_midi_path,
        )


def _append_rank3_final_inference_examples(
    *,
    runner_config: TowerRunnerConfig,
    rank_config,
    paths: TowerArtifactPaths,
    grandparent_policy,
    parent_policy,
    child_policy,
    reward_fn,
    initial_state: tuple[int, int, int],
    graph_spec,
    key_pitch_class: int | None,
    target_root_octave: int | None,
    specs_by_key_pitch_class,
    generator: torch.Generator,
) -> None:
    max_steps = _training_int(runner_config, "max_steps", default=1)
    for final_inference_index in range(4):
        final_key_pitch_class = _rank3_episode_key_pitch_class(
            key_pitch_class=key_pitch_class,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_key_pitch_class",
                default=_training_bool(runner_config, "sample_key_pitch_class", default=False),
            ),
        )
        final_spec = specs_by_key_pitch_class[
            0 if final_key_pitch_class is None else final_key_pitch_class
        ]
        final_target_root_octave = _rank3_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_target_root_octave",
                default=_training_bool(runner_config, "sample_target_root_octave", default=False),
            ),
        )
        final_initial_state = _rank3_episode_initial_state(
            initial_state=initial_state,
            target_root_octave=final_target_root_octave,
            spec=final_spec,
            config=runner_config,
            generator=generator,
            force_sample=_training_bool(
                runner_config,
                "final_inference_sample_initial_state",
                default=_training_bool(runner_config, "sample_initial_state", default=False),
            ),
        )
        final_inference = run_final_inference_episode(
            policy=child_policy,
            parent_policy=parent_policy,
            grandparent_policy=grandparent_policy,
            initial_state=final_initial_state,
            reward_fn=reward_fn,
            max_steps=max_steps,
            graph_spec=final_spec,
            measure_size=runner_config.measure_size,
            context_measures=runner_config.context_measures,
            parent_top_m=runner_config.parent_top_m,
            key_pitch_class=final_key_pitch_class,
            target_root_octave=final_target_root_octave,
            sampling_temperature=_training_float(runner_config, "sampling_temperature", default=1.0),
            sampling_uniform_mix=_training_float(runner_config, "sampling_uniform_mix", default=0.0),
            generator=generator,
        )
        final_midi_path = _write_final_midi(
            config=runner_config,
            paths=paths,
            final_inference=final_inference,
            final_inference_index=final_inference_index,
        )
        _append_final_inference_artifacts(
            paths=paths,
            final_inference=final_inference,
            lineage_id=runner_config.lineage_id,
            episode_index=rank_config.episode_budget + final_inference_index,
            final_inference_index=final_inference_index,
            final_midi_path=final_midi_path,
        )


def _rank2_reward_kwargs(reward_config: dict[str, object]) -> dict[str, object]:
    return {
        "key_pitch_class": int(reward_config["key_pitch_class"]),
        "use_context_key_pitch_class": bool(
            reward_config.get("use_context_key_pitch_class", False)
        ),
        "target_root_octave": int(reward_config["target_root_octave"]),
        "terminal_cadence_reward": float(reward_config["terminal_cadence_reward"]),
        "cadence_failure_reward": float(reward_config["cadence_failure_reward"]),
        "cadence_endpoint_weight": float(reward_config["cadence_endpoint_weight"]),
        "vertical_consonance_weight": float(reward_config["vertical_consonance_weight"]),
        "vertical_non_consonance_penalty": float(
            reward_config["vertical_non_consonance_penalty"]
        ),
        "upper_register_soft_ceiling": int(reward_config["upper_register_soft_ceiling"]),
        "upper_register_penalty_weight": float(
            reward_config["upper_register_penalty_weight"]
        ),
        "min_vertical_gap": int(reward_config["min_vertical_gap"]),
        "spacing_reward": float(reward_config["spacing_reward"]),
        "spacing_penalty": float(reward_config["spacing_penalty"]),
        "target_vertical_interval": int(reward_config["target_vertical_interval"]),
        "target_vertical_interval_weight": float(
            reward_config["target_vertical_interval_weight"]
        ),
        "onbeat_scale_degree_interval_reward": float(
            reward_config.get("onbeat_scale_degree_interval_reward", 0.0)
        ),
        "onbeat_non_scale_degree_interval_penalty": float(
            reward_config.get("onbeat_non_scale_degree_interval_penalty", 0.0)
        ),
        "offbeat_vertical_consonance_weight": float(
            reward_config.get("offbeat_vertical_consonance_weight", 0.0)
        ),
        "offbeat_vertical_non_consonance_penalty": float(
            reward_config.get("offbeat_vertical_non_consonance_penalty", 0.0)
        ),
    }


def _rank3_reward_kwargs(reward_config: dict[str, object]) -> dict[str, object]:
    return {
        "key_pitch_class": int(reward_config["key_pitch_class"]),
        "use_context_key_pitch_class": bool(
            reward_config.get("use_context_key_pitch_class", False)
        ),
        "target_root_octave": int(reward_config["target_root_octave"]),
        "use_context_target_root_octave": bool(
            reward_config.get("use_context_target_root_octave", False)
        ),
        "terminal_cadence_reward": float(reward_config["terminal_cadence_reward"]),
        "cadence_failure_reward": float(reward_config["cadence_failure_reward"]),
        "triad_consonance_weight": float(reward_config["triad_consonance_weight"]),
        "triad_non_consonance_penalty": float(
            reward_config["triad_non_consonance_penalty"]
        ),
        "min_adjacent_gap": int(reward_config["min_adjacent_gap"]),
        "max_outer_span": int(reward_config["max_outer_span"]),
        "adjacent_spacing_reward": float(reward_config["adjacent_spacing_reward"]),
        "adjacent_spacing_penalty": float(reward_config["adjacent_spacing_penalty"]),
        "outer_span_reward": float(reward_config["outer_span_reward"]),
        "outer_span_penalty": float(reward_config["outer_span_penalty"]),
        "cadence_endpoint_weight": float(reward_config["cadence_endpoint_weight"]),
        "onbeat_all_scale_degree_reward": float(
            reward_config.get("onbeat_all_scale_degree_reward", 0.0)
        ),
        "onbeat_not_all_scale_degree_penalty": float(
            reward_config.get("onbeat_not_all_scale_degree_penalty", 0.0)
        ),
        "offbeat_all_consonant_weight": float(
            reward_config.get("offbeat_all_consonant_weight", 0.0)
        ),
        "offbeat_non_consonance_penalty": float(
            reward_config.get("offbeat_non_consonance_penalty", 0.0)
        ),
    }
