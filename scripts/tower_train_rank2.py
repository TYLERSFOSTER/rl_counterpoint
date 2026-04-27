"""Thin tower rank-2 training script entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower.graph.spec import TowerGraphSpec
from tower.policy.transformer import TowerTransformerPolicy
from tower.reward.factory import build_rank2_reward_fn
from tower.train.checkpoint import (
    TowerArtifactPaths,
    find_accepted_parent_checkpoint,
    load_latest_checkpoint,
)
from tower.train.config import TowerRankConfig
from tower.train.runner import (
    TowerRunnerConfig,
    _build_rank1_policy,
    run_rank2_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a local rank-2 tower training run."""
    parser = argparse.ArgumentParser(description="Run a tower rank-2 training job.")
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-parent-pitch", type=int, default=64)
    parser.add_argument("--initial-child-pitch", type=int, default=68)
    parser.add_argument("--key-pitch-class", type=int, default=0)
    parser.add_argument("--target-root-octave", type=int, default=4)
    parser.add_argument("--parent-top-m", type=int, default=3)
    parser.add_argument("--terminal-cadence-reward", type=float, default=10.0)
    parser.add_argument("--cadence-failure-reward", type=float, default=0.0)
    parser.add_argument("--cadence-endpoint-weight", type=float, default=1.0)
    parser.add_argument("--vertical-consonance-weight", type=float, default=1.0)
    parser.add_argument(
        "--vertical-non-consonance-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument("--upper-register-soft-ceiling", type=int, default=80)
    parser.add_argument("--upper-register-penalty-weight", type=float, default=0.05)
    parser.add_argument("--min-vertical-gap", type=int, default=3)
    parser.add_argument("--spacing-reward", type=float, default=0.1)
    parser.add_argument("--spacing-penalty", type=float, default=-0.1)
    parser.add_argument("--target-vertical-interval", type=int, default=4)
    parser.add_argument("--target-vertical-interval-weight", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.5)
    parser.add_argument("--sampling-uniform-mix", type=float, default=0.15)
    parser.add_argument(
        "--log-reward-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _reward_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "kind": "rank2_slice_a",
        "key_pitch_class": args.key_pitch_class,
        "target_root_octave": args.target_root_octave,
        "terminal_cadence_reward": args.terminal_cadence_reward,
        "cadence_failure_reward": args.cadence_failure_reward,
        "cadence_endpoint_weight": args.cadence_endpoint_weight,
        "vertical_consonance_weight": args.vertical_consonance_weight,
        "vertical_non_consonance_penalty": args.vertical_non_consonance_penalty,
        "upper_register_soft_ceiling": args.upper_register_soft_ceiling,
        "upper_register_penalty_weight": args.upper_register_penalty_weight,
        "min_vertical_gap": args.min_vertical_gap,
        "spacing_reward": args.spacing_reward,
        "spacing_penalty": args.spacing_penalty,
        "target_vertical_interval": args.target_vertical_interval,
        "target_vertical_interval_weight": args.target_vertical_interval_weight,
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
        "sampling_temperature": args.sampling_temperature,
        "sampling_uniform_mix": args.sampling_uniform_mix,
        "log_reward_diagnostics": args.log_reward_diagnostics,
    }


def _graph_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "pitch_min": args.pitch_min,
        "pitch_max": args.pitch_max,
    }


def _load_parent_policy(
    *,
    lineage_id: str,
    artifact_root: Path,
) -> tuple[TowerTransformerPolicy, str]:
    parent_paths = TowerArtifactPaths(
        lineage_id=lineage_id,
        rank=2,
        artifact_root=artifact_root,
    )
    parent_checkpoint_path = find_accepted_parent_checkpoint(parent_paths)
    parent_checkpoint_payload = load_latest_checkpoint(parent_checkpoint_path)
    parent_config_payload = parent_checkpoint_payload.get("config")
    if not isinstance(parent_config_payload, dict):
        raise TypeError("parent checkpoint config must be an object")
    parent_rank_config = TowerRankConfig.from_json_dict(parent_config_payload)
    parent_seed = parent_rank_config.seed_config.get("seed", 0)
    if not isinstance(parent_seed, int):
        raise TypeError("parent seed_config.seed must be an int")
    parent_runner_config = TowerRunnerConfig(
        lineage_id=parent_rank_config.lineage_id,
        rank=parent_rank_config.rank,
        episode_count=parent_rank_config.episode_budget,
        seed=parent_seed,
        artifact_root=artifact_root,
        measure_size=parent_rank_config.measure_size,
        context_measures=parent_rank_config.context_measures,
        max_step_size=parent_rank_config.max_step_size,
        reward_config=dict(parent_rank_config.reward_config),
        graph_config=dict(parent_rank_config.graph_config),
        policy_config=dict(parent_rank_config.policy_config),
        training_config=dict(parent_rank_config.training_config),
    )
    parent_policy = _build_rank1_policy(parent_runner_config)
    policy_state_dict = parent_checkpoint_payload.get("policy_state_dict")
    if not isinstance(policy_state_dict, dict):
        raise TypeError("parent checkpoint policy_state_dict must be an object")
    parent_policy.load_state_dict(policy_state_dict)
    parent_relative_checkpoint = parent_checkpoint_path.relative_to(
        parent_paths.lineage_dir
    ).as_posix()
    return parent_policy, parent_relative_checkpoint


def main(argv: list[str] | None = None) -> int:
    """Run one rank-2 tower training job against an accepted rank-1 parent."""
    args = parse_args(argv)
    parent_policy, parent_checkpoint = _load_parent_policy(
        lineage_id=args.lineage_id,
        artifact_root=args.artifact_root,
    )
    reward_config = _reward_config_from_args(args)
    config = TowerRunnerConfig(
        lineage_id=args.lineage_id,
        rank=2,
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
    result = run_rank2_training(
        config=config,
        parent_policy=parent_policy,
        initial_state=(args.initial_parent_pitch, args.initial_child_pitch),
        reward_fn=build_rank2_reward_fn(
            key_pitch_class=args.key_pitch_class,
            target_root_octave=args.target_root_octave,
            terminal_cadence_reward=args.terminal_cadence_reward,
            cadence_failure_reward=args.cadence_failure_reward,
            cadence_endpoint_weight=args.cadence_endpoint_weight,
            vertical_consonance_weight=args.vertical_consonance_weight,
            vertical_non_consonance_penalty=args.vertical_non_consonance_penalty,
            upper_register_soft_ceiling=args.upper_register_soft_ceiling,
            upper_register_penalty_weight=args.upper_register_penalty_weight,
            min_vertical_gap=args.min_vertical_gap,
            spacing_reward=args.spacing_reward,
            spacing_penalty=args.spacing_penalty,
            target_vertical_interval=args.target_vertical_interval,
            target_vertical_interval_weight=args.target_vertical_interval_weight,
        ),
        graph_spec=TowerGraphSpec(
            rank=2,
            key_pitch_class=args.key_pitch_class,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            max_step_size=args.max_step_size,
        ),
    )

    print(f"run_dir: {result.paths.rank_dir}")
    print(f"lineage_dir: {result.paths.lineage_dir}")
    print(f"rank: {config.rank}")
    print(f"episodes: {config.episode_count}")
    print(f"max_steps: {args.max_steps}")
    print(f"pitch_range: [{args.pitch_min}, {args.pitch_max}]")
    print(f"reward: {reward_config['kind']}")
    print(f"parent checkpoint: {parent_checkpoint}")
    print(f"latest checkpoint: {result.paths.checkpoint_latest_path}")
    print(f"final midi: {result.final_midi_path}")
    print(f"final episode return: {result.final_inference.metrics['episode_return']}")
    print(f"final terminal_success: {result.final_inference.metrics['terminal_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
