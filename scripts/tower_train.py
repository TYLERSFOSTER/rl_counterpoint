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
    parser.add_argument("--pitch-min", type=int, default=0)
    parser.add_argument("--pitch-max", type=int, default=127)
    parser.add_argument("--final-chord-size", type=int, default=4)
    parser.add_argument(
        "--reserved-upper-semitones-per-voice",
        type=int,
        default=5,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-pitch", type=int, default=60)
    parser.add_argument("--initial-pitch-min", type=int, default=36)
    parser.add_argument("--initial-pitch-max", type=int, default=84)
    parser.add_argument(
        "--sample-initial-pitch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--key-pitch-class", type=int, default=0)
    parser.add_argument("--target-root-octave", type=int, default=4)
    parser.add_argument(
        "--sample-target-root-octave",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target-root-octave-choices",
        type=_parse_int_choices,
        default=None,
    )
    parser.add_argument(
        "--sample-initial-pitch-in-target-octave",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
    parser.add_argument("--onbeat-non-scale-penalty", type=float, default=-2.0)
    parser.add_argument("--offbeat-non-consonance-penalty", type=float, default=-2.0)
    parser.add_argument("--step-size-balance-threshold", type=int, default=3)
    parser.add_argument("--step-size-balance-target-small-rate", type=float, default=0.3)
    parser.add_argument("--step-size-balance-weight", type=float, default=1.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.5)
    parser.add_argument("--sampling-uniform-mix", type=float, default=0.15)
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


def main(argv: list[str] | None = None) -> int:
    """Run the requested tower training job."""
    args = parse_args(argv)
    if args.rank != 1:
        raise ValueError("scripts/tower_train.py currently supports rank 1 only")

    reward_config = {
        "kind": "rank1_slice_a",
        "key_pitch_class": args.key_pitch_class,
        "target_root_octave": args.target_root_octave,
        "use_context_target_root_octave": args.sample_target_root_octave,
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
        "onbeat_non_scale_penalty": args.onbeat_non_scale_penalty,
        "offbeat_non_consonance_penalty": args.offbeat_non_consonance_penalty,
        "step_size_balance_threshold": args.step_size_balance_threshold,
        "step_size_balance_target_small_rate": (
            args.step_size_balance_target_small_rate
        ),
        "step_size_balance_weight": args.step_size_balance_weight,
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
        graph_config={
            "pitch_min": args.pitch_min,
            "pitch_max": args.pitch_max,
            "final_chord_size": args.final_chord_size,
            "reserved_upper_semitones_per_voice": (
                args.reserved_upper_semitones_per_voice
            ),
        },
        policy_config={
            "d_model": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
        },
        training_config={
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "sample_initial_pitch": args.sample_initial_pitch,
            "initial_pitch_min": args.initial_pitch_min,
            "initial_pitch_max": args.initial_pitch_max,
            "sample_initial_pitch_in_target_octave": (
                args.sample_initial_pitch_in_target_octave
            ),
            "sample_target_root_octave": args.sample_target_root_octave,
            "sampling_temperature": args.sampling_temperature,
            "sampling_uniform_mix": args.sampling_uniform_mix,
            "log_reward_diagnostics": args.log_reward_diagnostics,
            **(
                {}
                if args.target_root_octave_choices is None
                else {"target_root_octave_choices": args.target_root_octave_choices}
            ),
        },
    )
    result = run_rank1_training(
        config=config,
        initial_state=(args.initial_pitch,),
        reward_fn=build_rank1_reward_fn(
            key_pitch_class=args.key_pitch_class,
            target_root_octave=args.target_root_octave,
            use_context_target_root_octave=args.sample_target_root_octave,
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
            onbeat_non_scale_penalty=args.onbeat_non_scale_penalty,
            offbeat_non_consonance_penalty=args.offbeat_non_consonance_penalty,
            step_size_balance_threshold=args.step_size_balance_threshold,
            step_size_balance_target_small_rate=(
                args.step_size_balance_target_small_rate
            ),
            step_size_balance_weight=args.step_size_balance_weight,
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
    graph_spec = TowerGraphSpec(
        rank=1,
        pitch_min=config.graph_config["pitch_min"],  # type: ignore[index]
        pitch_max=min(
            config.graph_config["pitch_max"],  # type: ignore[index]
            127
            - config.graph_config["reserved_upper_semitones_per_voice"]  # type: ignore[index]
            * config.graph_config["final_chord_size"],  # type: ignore[index]
        ),
        max_step_size=args.max_step_size,
    )
    print(f"pitch_range: [{graph_spec.pitch_min}, {graph_spec.pitch_max}]")
    print(f"reward: {reward_config['kind']}")
    print(f"latest checkpoint: {result.paths.checkpoint_latest_path}")
    print(f"final midi: {result.final_midi_path}")
    print(f"final episode return: {result.final_inference.metrics['episode_return']}")
    print(f"final terminal_success: {result.final_inference.metrics['terminal_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
