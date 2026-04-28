"""Staged tower training script for larger-model rank-1 curricula."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower.reward.factory import build_rank1_reward_fn
from tower.train.checkpoint import load_latest_checkpoint
from tower.train.runner import (
    TowerRunnerConfig,
    _build_optimizer,
    _build_rank1_policy,
    _graph_spec_from_config,
    run_rank1_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for one staged rank-1 training job."""
    parser = argparse.ArgumentParser(
        description="Run a staged larger-model tower rank-1 training job."
    )
    parser.add_argument("--lineage-id", default="local-tower-staged")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tower",
    )
    parser.add_argument("--stage1-episodes", type=int, default=3_000)
    parser.add_argument("--stage2-episodes", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--measure-size", type=int, default=4)
    parser.add_argument("--context-measures", type=int, default=2)
    parser.add_argument("--max-step-size", type=int, default=7)
    parser.add_argument("--pitch-min", type=int, default=0)
    parser.add_argument("--pitch-max", type=int, default=127)
    parser.add_argument("--final-rank", type=int, default=1)
    parser.add_argument(
        "--use-induced-rank1-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--induced-rank2-pitch-min", type=int, default=0)
    parser.add_argument("--induced-rank2-pitch-max", type=int, default=127)
    parser.add_argument("--induced-rank2-max-step-size", type=int, default=7)
    parser.add_argument("--induced-rank3-pitch-min", type=int, default=0)
    parser.add_argument("--induced-rank3-pitch-max", type=int, default=127)
    parser.add_argument("--induced-rank3-max-step-size", type=int, default=7)
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
    parser.add_argument("--goal-octave-direction-weight", type=float, default=0.5)
    parser.add_argument(
        "--sample-target-root-octave",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target-root-octave-choices",
        type=_parse_int_choices,
        default=[2, 3, 4, 5],
    )
    parser.add_argument("--terminal-cadence-reward", type=float, default=100.0)
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
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--sampling-temperature", type=float, default=1.5)
    parser.add_argument("--sampling-uniform-mix", type=float, default=0.15)
    parser.add_argument(
        "--log-reward-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        "kind": "rank1_slice_a",
        "key_pitch_class": args.key_pitch_class,
        "target_root_octave": args.target_root_octave,
        "use_context_target_root_octave": args.sample_target_root_octave,
        "goal_octave_direction_weight": args.goal_octave_direction_weight,
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


def _policy_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }


def _base_training_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "sample_initial_pitch": args.sample_initial_pitch,
        "initial_pitch_min": args.initial_pitch_min,
        "initial_pitch_max": args.initial_pitch_max,
        "sample_target_root_octave": args.sample_target_root_octave,
        "target_root_octave_choices": args.target_root_octave_choices,
        "progress_every": args.progress_every,
        "sampling_temperature": args.sampling_temperature,
        "sampling_uniform_mix": args.sampling_uniform_mix,
        "log_reward_diagnostics": args.log_reward_diagnostics,
    }


def _graph_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "pitch_min": args.pitch_min,
        "pitch_max": args.pitch_max,
        "final_rank": args.final_rank,
        "use_induced_rank1_graph": args.use_induced_rank1_graph,
        "induced_rank2_pitch_min": args.induced_rank2_pitch_min,
        "induced_rank2_pitch_max": args.induced_rank2_pitch_max,
        "induced_rank2_max_step_size": args.induced_rank2_max_step_size,
        "induced_rank3_pitch_min": args.induced_rank3_pitch_min,
        "induced_rank3_pitch_max": args.induced_rank3_pitch_max,
        "induced_rank3_max_step_size": args.induced_rank3_max_step_size,
    }


def _build_rank1_reward_from_config(reward_config: dict[str, object]):
    return build_rank1_reward_fn(
        key_pitch_class=int(reward_config["key_pitch_class"]),
        target_root_octave=int(reward_config["target_root_octave"]),
        use_context_target_root_octave=bool(
            reward_config["use_context_target_root_octave"]
        ),
        goal_octave_direction_weight=float(
            reward_config["goal_octave_direction_weight"]
        ),
        terminal_cadence_reward=float(reward_config["terminal_cadence_reward"]),
        cadence_failure_reward=float(reward_config["cadence_failure_reward"]),
        max_recent_range=int(reward_config["max_recent_range"]),
        range_penalty=float(reward_config["range_penalty"]),
        large_leap_threshold=int(reward_config["large_leap_threshold"]),
        recovery_step_threshold=int(reward_config["recovery_step_threshold"]),
        recovery_reward=float(reward_config["recovery_reward"]),
        failure_penalty=float(reward_config["failure_penalty"]),
        measure_start_tonic_reward=float(reward_config["measure_start_tonic_reward"]),
        onbeat_scale_degree_reward=float(
            reward_config["onbeat_scale_degree_reward"]
        ),
        offbeat_consonance_weight=float(reward_config["offbeat_consonance_weight"]),
        onbeat_non_scale_penalty=float(reward_config["onbeat_non_scale_penalty"]),
        offbeat_non_consonance_penalty=float(
            reward_config["offbeat_non_consonance_penalty"]
        ),
        step_size_balance_threshold=int(
            reward_config["step_size_balance_threshold"]
        ),
        step_size_balance_target_small_rate=float(
            reward_config["step_size_balance_target_small_rate"]
        ),
        step_size_balance_weight=float(reward_config["step_size_balance_weight"]),
    )


def _exception_log_path(*, artifact_root: Path, lineage_id: str) -> Path:
    return artifact_root / "logs" / f"{lineage_id}.exception.log"


def _write_exception_log(
    *,
    exception_log_path: Path,
    argv: list[str] | None,
    args: argparse.Namespace,
) -> None:
    exception_log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_argv = [] if argv is None else list(argv)
    exception_log_path.write_text(
        "\n".join(
            [
                f"lineage_id: {args.lineage_id}",
                f"artifact_root: {args.artifact_root}",
                f"stage1_episodes: {args.stage1_episodes}",
                f"stage2_episodes: {args.stage2_episodes}",
                f"argv: {rendered_argv}",
                "",
                traceback.format_exc().rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the coupled-then-decoupled larger-model rank-1 curriculum."""
    args = parse_args(argv)
    exception_log_path = _exception_log_path(
        artifact_root=args.artifact_root,
        lineage_id=args.lineage_id,
    )
    try:
        reward_config = _reward_config_from_args(args)
        policy_config = _policy_config_from_args(args)
        graph_config = _graph_config_from_args(args)
        base_training_config = _base_training_config_from_args(args)
        reward_fn = _build_rank1_reward_from_config(reward_config)
        stage1_lineage_id = f"{args.lineage_id}-stage1"
        stage2_lineage_id = args.lineage_id

        stage1_config = TowerRunnerConfig(
            lineage_id=stage1_lineage_id,
            rank=1,
            episode_count=args.stage1_episodes,
            seed=args.seed,
            artifact_root=args.artifact_root,
            measure_size=args.measure_size,
            context_measures=args.context_measures,
            max_step_size=args.max_step_size,
            reward_config=reward_config,
            graph_config=graph_config,
            policy_config=policy_config,
            training_config={
                **base_training_config,
                "progress_label": "stage1",
                "sample_initial_pitch_in_target_octave": True,
            },
        )
        stage1_policy = _build_rank1_policy(stage1_config)
        stage1_optimizer = _build_optimizer(policy=stage1_policy, config=stage1_config)
        graph_spec = _graph_spec_from_config(stage1_config)

        print(f"starting stage1: {stage1_lineage_id}")
        stage1_result = run_rank1_training(
            config=stage1_config,
            initial_state=(args.initial_pitch,),
            reward_fn=reward_fn,
            policy=stage1_policy,
            optimizer=stage1_optimizer,
            graph_spec=graph_spec,
        )
        print(f"finished stage1: {stage1_result.paths.rank_dir}")

        checkpoint = load_latest_checkpoint(stage1_result.paths)
        stage2_config = TowerRunnerConfig(
            lineage_id=stage2_lineage_id,
            rank=1,
            episode_count=args.stage2_episodes,
            seed=args.seed,
            artifact_root=args.artifact_root,
            measure_size=args.measure_size,
            context_measures=args.context_measures,
            max_step_size=args.max_step_size,
            reward_config=reward_config,
            graph_config=graph_config,
            policy_config=policy_config,
            training_config={
                **base_training_config,
                "progress_label": "stage2",
                "sample_initial_pitch_in_target_octave": False,
            },
        )
        stage2_policy = _build_rank1_policy(stage2_config)
        stage2_policy.load_state_dict(checkpoint["policy_state_dict"])
        stage2_optimizer = _build_optimizer(policy=stage2_policy, config=stage2_config)
        stage2_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"starting stage2: {stage2_lineage_id}")
        stage2_result = run_rank1_training(
            config=stage2_config,
            initial_state=(args.initial_pitch,),
            reward_fn=reward_fn,
            policy=stage2_policy,
            optimizer=stage2_optimizer,
            graph_spec=graph_spec,
        )
        print(f"finished stage2: {stage2_result.paths.rank_dir}")

        print(f"stage1 run_dir: {stage1_result.paths.rank_dir}")
        print(f"stage2 run_dir: {stage2_result.paths.rank_dir}")
        print(f"stage2 latest checkpoint: {stage2_result.paths.checkpoint_latest_path}")
        print(f"stage2 final midi: {stage2_result.final_midi_path}")
        print(
            f"stage2 final episode return: "
            f"{stage2_result.final_inference.metrics['episode_return']}"
        )
        print(
            f"stage2 final terminal_success: "
            f"{stage2_result.final_inference.metrics['terminal_success']}"
        )
        return 0
    except Exception:
        _write_exception_log(
            exception_log_path=exception_log_path,
            argv=argv,
            args=args,
        )
        print(f"exception log: {exception_log_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
