"""Thin tower training script entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower.graph.spec import TowerGraphSpec
from tower.reward.result import TowerRewardResult
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the requested tower training job."""
    args = parse_args(argv)
    if args.rank != 1:
        raise ValueError("scripts/tower_train.py currently supports rank 1 only")

    config = TowerRunnerConfig(
        lineage_id=args.lineage_id,
        rank=args.rank,
        episode_count=args.episodes,
        seed=args.seed,
        artifact_root=args.artifact_root,
        measure_size=args.measure_size,
        context_measures=args.context_measures,
        max_step_size=args.max_step_size,
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
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
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
    print(f"latest checkpoint: {result.paths.checkpoint_latest_path}")
    print(f"final midi: {result.final_midi_path}")
    print(f"final episode return: {result.final_inference.metrics['episode_return']}")
    print(f"final terminal_success: {result.final_inference.metrics['terminal_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
