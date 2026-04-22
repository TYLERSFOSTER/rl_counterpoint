"""Thin tower training script entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower.graph.spec import TowerGraphSpec
from tower.reward.factory import build_rank1_reward_fn
from tower.train.runner import TowerRunnerConfig, run_rank1_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a local tower training run."""
    parser = argparse.ArgumentParser(description="Run a tower training job.")
    parser.add_argument("--rank", type=int, default=1)
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-pitch", type=int, default=60)
    parser.add_argument("--key-pitch-class", type=int, default=0)
    parser.add_argument("--target-root-octave", type=int, default=4)
    parser.add_argument("--terminal-cadence-reward", type=float, default=10.0)
    parser.add_argument("--cadence-failure-reward", type=float, default=0.0)
    parser.add_argument("--max-recent-range", type=int, default=12)
    parser.add_argument("--range-penalty", type=float, default=-1.0)
    parser.add_argument("--large-leap-threshold", type=int, default=6)
    parser.add_argument("--recovery-step-threshold", type=int, default=3)
    parser.add_argument("--recovery-reward", type=float, default=0.5)
    parser.add_argument("--failure-penalty", type=float, default=-0.5)
    parser.add_argument("--measure-start-tonic-reward", type=float, default=1.0)
    parser.add_argument("--onbeat-scale-degree-reward", type=float, default=1.0)
    parser.add_argument("--offbeat-consonance-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the requested tower training job."""
    args = parse_args(argv)
    if args.rank != 1:
        raise ValueError("scripts/tower_train.py currently supports rank 1 only")

    reward_config = {
        "kind": "rank1_slice_a",
        "key_pitch_class": args.key_pitch_class,
        "target_root_octave": args.target_root_octave,
        "terminal_cadence_reward": args.terminal_cadence_reward,
        "cadence_failure_reward": args.cadence_failure_reward,
        "max_recent_range": args.max_recent_range,
        "range_penalty": args.range_penalty,
        "large_leap_threshold": args.large_leap_threshold,
        "recovery_step_threshold": args.recovery_step_threshold,
        "recovery_reward": args.recovery_reward,
        "failure_penalty": args.failure_penalty,
        "measure_start_tonic_reward": args.measure_start_tonic_reward,
        "onbeat_scale_degree_reward": args.onbeat_scale_degree_reward,
        "offbeat_consonance_weight": args.offbeat_consonance_weight,
    }

    config = TowerRunnerConfig(
        lineage_id=args.lineage_id,
        rank=args.rank,
        episode_count=args.episodes,
        seed=args.seed,
        artifact_root=args.artifact_root,
        measure_size=args.measure_size,
        context_measures=args.context_measures,
        max_step_size=args.max_step_size,
        reward_config=reward_config,
        policy_config={
            "d_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "ff_dim": 16,
            "dropout": 0.0,
        },
        training_config={
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
        },
    )
    result = run_rank1_training(
        config=config,
        initial_state=(args.initial_pitch,),
        reward_fn=build_rank1_reward_fn(
            key_pitch_class=args.key_pitch_class,
            target_root_octave=args.target_root_octave,
            terminal_cadence_reward=args.terminal_cadence_reward,
            cadence_failure_reward=args.cadence_failure_reward,
            max_recent_range=args.max_recent_range,
            range_penalty=args.range_penalty,
            large_leap_threshold=args.large_leap_threshold,
            recovery_step_threshold=args.recovery_step_threshold,
            recovery_reward=args.recovery_reward,
            failure_penalty=args.failure_penalty,
            measure_start_tonic_reward=args.measure_start_tonic_reward,
            onbeat_scale_degree_reward=args.onbeat_scale_degree_reward,
            offbeat_consonance_weight=args.offbeat_consonance_weight,
        ),
        graph_spec=TowerGraphSpec(
            rank=1,
            max_step_size=args.max_step_size,
        ),
    )

    print(f"run_dir: {result.paths.rank_dir}")
    print(f"lineage_dir: {result.paths.lineage_dir}")
    print(f"rank: {config.rank}")
    print(f"episodes: {config.episode_count}")
    print(f"max_steps: {args.max_steps}")
    print(f"reward: {reward_config['kind']}")
    print(f"latest checkpoint: {result.paths.checkpoint_latest_path}")
    print(f"final midi: {result.final_midi_path}")
    print(f"final episode return: {result.final_inference.metrics['episode_return']}")
    print(f"final terminal_success: {result.final_inference.metrics['terminal_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
