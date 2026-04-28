"""Thin tower rank-3 training script entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower.policy.transformer import TowerTransformerPolicy
from tower.reward.factory import build_rank3_reward_fn
from tower.train.checkpoint import (
    TowerArtifactPaths,
    find_accepted_parent_checkpoint,
    load_latest_checkpoint,
)
from tower.train.config import TowerRankConfig
from tower.train.runner import (
    TowerRunnerConfig,
    _build_rank1_policy,
    _build_rank2_policy,
    _graph_spec_from_config,
    run_rank3_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a local rank-3 tower training run."""
    parser = argparse.ArgumentParser(description="Run a tower rank-3 training job.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--lineage-id", default="local-tower")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tower",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--measure-size", type=int, default=4)
    parser.add_argument("--context-measures", type=int, default=2)
    parser.add_argument("--max-step-size", type=int, default=1)
    parser.add_argument("--pitch-min", type=int, default=36)
    parser.add_argument("--pitch-max", type=int, default=84)
    parser.add_argument("--final-rank", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-pedal-pitch", type=int, default=62)
    parser.add_argument("--initial-middle-pitch", type=int, default=65)
    parser.add_argument("--initial-top-pitch", type=int, default=69)
    parser.add_argument(
        "--sample-initial-state",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--initial-parent-pitch-min", type=int, default=36)
    parser.add_argument("--initial-parent-pitch-max", type=int, default=84)
    parser.add_argument(
        "--sample-initial-parent-pitch-in-target-octave",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--key-pitch-class", type=int, default=0)
    parser.add_argument("--target-root-octave", type=int, default=4)
    parser.add_argument("--parent-top-m", type=int, default=3)
    parser.add_argument(
        "--sample-target-root-octave",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--target-root-octave-choices",
        type=_parse_int_choices,
        default=[2, 3, 4, 5],
    )
    parser.add_argument("--terminal-cadence-reward", type=float, default=10.0)
    parser.add_argument("--cadence-failure-reward", type=float, default=0.0)
    parser.add_argument("--triad-consonance-weight", type=float, default=1.0)
    parser.add_argument("--triad-non-consonance-penalty", type=float, default=0.0)
    parser.add_argument("--min-adjacent-gap", type=int, default=3)
    parser.add_argument("--max-outer-span", type=int, default=15)
    parser.add_argument("--adjacent-spacing-reward", type=float, default=0.1)
    parser.add_argument("--adjacent-spacing-penalty", type=float, default=-0.1)
    parser.add_argument("--outer-span-reward", type=float, default=0.1)
    parser.add_argument("--outer-span-penalty", type=float, default=-0.1)
    parser.add_argument("--cadence-endpoint-weight", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.5)
    parser.add_argument("--sampling-uniform-mix", type=float, default=0.15)
    parser.add_argument(
        "--final-inference-sample-target-root-octave",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--final-inference-sample-initial-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-reward-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _parse_int_choices(value: str) -> list[int]:
    try:
        choices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target octave choices must be comma-separated integers"
        ) from exc
    if not choices:
        raise argparse.ArgumentTypeError("target octave choices must not be empty")
    return choices


def _reward_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "kind": "rank3_slice_a",
        "key_pitch_class": args.key_pitch_class,
        "target_root_octave": args.target_root_octave,
        "use_context_target_root_octave": args.sample_target_root_octave,
        "terminal_cadence_reward": args.terminal_cadence_reward,
        "cadence_failure_reward": args.cadence_failure_reward,
        "triad_consonance_weight": args.triad_consonance_weight,
        "triad_non_consonance_penalty": args.triad_non_consonance_penalty,
        "min_adjacent_gap": args.min_adjacent_gap,
        "max_outer_span": args.max_outer_span,
        "adjacent_spacing_reward": args.adjacent_spacing_reward,
        "adjacent_spacing_penalty": args.adjacent_spacing_penalty,
        "outer_span_reward": args.outer_span_reward,
        "outer_span_penalty": args.outer_span_penalty,
        "cadence_endpoint_weight": args.cadence_endpoint_weight,
    }


def _policy_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }


def _training_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "sample_initial_state": args.sample_initial_state,
        "initial_parent_pitch_min": args.initial_parent_pitch_min,
        "initial_parent_pitch_max": args.initial_parent_pitch_max,
        "sample_initial_parent_pitch_in_target_octave": (
            args.sample_initial_parent_pitch_in_target_octave
        ),
        "sample_target_root_octave": args.sample_target_root_octave,
        "final_inference_sample_target_root_octave": (
            args.final_inference_sample_target_root_octave
        ),
        "final_inference_sample_initial_state": (
            args.final_inference_sample_initial_state
        ),
        "sampling_temperature": args.sampling_temperature,
        "sampling_uniform_mix": args.sampling_uniform_mix,
        "log_reward_diagnostics": args.log_reward_diagnostics,
        **(
            {}
            if args.target_root_octave_choices is None
            else {"target_root_octave_choices": args.target_root_octave_choices}
        ),
    }


def _graph_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "pitch_min": args.pitch_min,
        "pitch_max": args.pitch_max,
        "final_rank": args.final_rank,
    }


def _runner_config_from_parent_rank_config(
    *,
    parent_rank_config: TowerRankConfig,
    artifact_root: Path,
) -> TowerRunnerConfig:
    parent_seed = parent_rank_config.seed_config.get("seed", 0)
    if not isinstance(parent_seed, int):
        raise TypeError("parent seed_config.seed must be an int")
    return TowerRunnerConfig(
        lineage_id=parent_rank_config.lineage_id,
        rank=parent_rank_config.rank,
        episode_count=parent_rank_config.episode_budget,
        seed=parent_seed,
        artifact_root=artifact_root,
        measure_size=parent_rank_config.measure_size,
        context_measures=parent_rank_config.context_measures,
        max_step_size=parent_rank_config.max_step_size,
        parent_checkpoint=parent_rank_config.parent_checkpoint,
        parent_top_m=parent_rank_config.parent_sampler_config.get("top_m", 3),
        reward_config=dict(parent_rank_config.reward_config),
        graph_config=dict(parent_rank_config.graph_config),
        policy_config=dict(parent_rank_config.policy_config),
        training_config=dict(parent_rank_config.training_config),
    )


def _load_parent_stack(
    *,
    lineage_id: str,
    artifact_root: Path,
) -> tuple[TowerTransformerPolicy, TowerTransformerPolicy, str, str]:
    rank3_paths = TowerArtifactPaths(
        lineage_id=lineage_id,
        rank=3,
        artifact_root=artifact_root,
    )
    parent_checkpoint_path = find_accepted_parent_checkpoint(rank3_paths)
    parent_checkpoint_payload = load_latest_checkpoint(parent_checkpoint_path)
    parent_config_payload = parent_checkpoint_payload.get("config")
    if not isinstance(parent_config_payload, dict):
        raise TypeError("rank-2 parent checkpoint config must be an object")
    parent_rank_config = TowerRankConfig.from_json_dict(parent_config_payload)
    parent_runner_config = _runner_config_from_parent_rank_config(
        parent_rank_config=parent_rank_config,
        artifact_root=artifact_root,
    )
    parent_policy = _build_rank2_policy(parent_runner_config)
    parent_policy_state_dict = parent_checkpoint_payload.get("policy_state_dict")
    if not isinstance(parent_policy_state_dict, dict):
        raise TypeError("rank-2 parent policy_state_dict must be an object")
    parent_policy.load_state_dict(parent_policy_state_dict)

    rank2_paths = TowerArtifactPaths(
        lineage_id=lineage_id,
        rank=2,
        artifact_root=artifact_root,
    )
    grandparent_checkpoint_path = find_accepted_parent_checkpoint(rank2_paths)
    grandparent_checkpoint_payload = load_latest_checkpoint(grandparent_checkpoint_path)
    grandparent_config_payload = grandparent_checkpoint_payload.get("config")
    if not isinstance(grandparent_config_payload, dict):
        raise TypeError("rank-1 grandparent checkpoint config must be an object")
    grandparent_rank_config = TowerRankConfig.from_json_dict(grandparent_config_payload)
    grandparent_runner_config = _runner_config_from_parent_rank_config(
        parent_rank_config=grandparent_rank_config,
        artifact_root=artifact_root,
    )
    grandparent_policy = _build_rank1_policy(grandparent_runner_config)
    grandparent_policy_state_dict = grandparent_checkpoint_payload.get("policy_state_dict")
    if not isinstance(grandparent_policy_state_dict, dict):
        raise TypeError("rank-1 grandparent policy_state_dict must be an object")
    grandparent_policy.load_state_dict(grandparent_policy_state_dict)

    parent_relative_checkpoint = parent_checkpoint_path.relative_to(
        rank3_paths.lineage_dir
    ).as_posix()
    grandparent_relative_checkpoint = grandparent_checkpoint_path.relative_to(
        rank3_paths.lineage_dir
    ).as_posix()
    return (
        grandparent_policy,
        parent_policy,
        grandparent_relative_checkpoint,
        parent_relative_checkpoint,
    )


def main(argv: list[str] | None = None) -> int:
    """Run one rank-3 tower training job against an accepted rank-2 parent stack."""
    args = parse_args(argv)
    (
        grandparent_policy,
        parent_policy,
        grandparent_checkpoint,
        parent_checkpoint,
    ) = _load_parent_stack(
        lineage_id=args.lineage_id,
        artifact_root=args.artifact_root,
    )
    reward_config = _reward_config_from_args(args)
    config = TowerRunnerConfig(
        lineage_id=args.lineage_id,
        rank=3,
        episode_count=args.episodes,
        seed=args.seed,
        artifact_root=args.artifact_root,
        measure_size=args.measure_size,
        context_measures=args.context_measures,
        max_step_size=args.max_step_size,
        parent_checkpoint=parent_checkpoint,
        parent_top_m=args.parent_top_m,
        graph_config=_graph_config_from_args(args),
        reward_config=reward_config,
        policy_config=_policy_config_from_args(args),
        training_config=_training_config_from_args(args),
    )
    result = run_rank3_training(
        config=config,
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        initial_state=(
            args.initial_pedal_pitch,
            args.initial_middle_pitch,
            args.initial_top_pitch,
        ),
        reward_fn=build_rank3_reward_fn(
            key_pitch_class=args.key_pitch_class,
            terminal_cadence_reward=args.terminal_cadence_reward,
            cadence_failure_reward=args.cadence_failure_reward,
            target_root_octave=args.target_root_octave,
            use_context_target_root_octave=args.sample_target_root_octave,
            triad_consonance_weight=args.triad_consonance_weight,
            triad_non_consonance_penalty=args.triad_non_consonance_penalty,
            min_adjacent_gap=args.min_adjacent_gap,
            max_outer_span=args.max_outer_span,
            adjacent_spacing_reward=args.adjacent_spacing_reward,
            adjacent_spacing_penalty=args.adjacent_spacing_penalty,
            outer_span_reward=args.outer_span_reward,
            outer_span_penalty=args.outer_span_penalty,
            cadence_endpoint_weight=args.cadence_endpoint_weight,
        ),
        graph_spec=_graph_spec_from_config(config),
    )

    graph_spec = _graph_spec_from_config(config)
    print(f"run_dir: {result.paths.rank_dir}")
    print(f"lineage_dir: {result.paths.lineage_dir}")
    print(f"rank: {config.rank}")
    print(f"episodes: {config.episode_count}")
    print(f"max_steps: {args.max_steps}")
    print(f"pitch_range: [{graph_spec.pitch_min}, {graph_spec.pitch_max}]")
    print(f"reward: {reward_config['kind']}")
    print(f"grandparent checkpoint: {grandparent_checkpoint}")
    print(f"parent checkpoint: {parent_checkpoint}")
    print(f"latest checkpoint: {result.paths.checkpoint_latest_path}")
    print(f"final midi: {result.final_midi_path}")
    print(f"final episode return: {result.final_inference.metrics['episode_return']}")
    print(f"final terminal_success: {result.final_inference.metrics['terminal_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
