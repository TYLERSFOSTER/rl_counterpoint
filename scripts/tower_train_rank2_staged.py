"""Staged tower rank-2 training script with coupled then decoupled starts."""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tower_train_rank2 import _load_parent_policy
from tower.graph.spec import TowerGraphSpec
from tower.reward.factory import build_rank2_reward_fn
from tower.train.checkpoint import load_latest_checkpoint
from tower.train.runner import (
    TowerRunnerConfig,
    _build_optimizer,
    _build_rank2_policy,
    run_rank2_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a staged rank-2 tower training curriculum."
    )
    parser.add_argument("--lineage-id", default="local-tower-rank2-staged")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tower",
    )
    parser.add_argument("--stage1-episodes", type=int, default=5_000)
    parser.add_argument("--stage2-episodes", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--measure-size", type=int, default=4)
    parser.add_argument("--context-measures", type=int, default=2)
    parser.add_argument("--max-step-size", type=int, default=7)
    parser.add_argument("--pitch-min", type=int, default=36)
    parser.add_argument("--pitch-max", type=int, default=84)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-parent-pitch", type=int, default=64)
    parser.add_argument("--initial-child-pitch", type=int, default=68)
    parser.add_argument("--initial-parent-pitch-min", type=int, default=36)
    parser.add_argument("--initial-parent-pitch-max", type=int, default=84)
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
        default=[2, 3, 4, 5],
    )
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


def _base_training_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "sampling_temperature": args.sampling_temperature,
        "sampling_uniform_mix": args.sampling_uniform_mix,
        "log_reward_diagnostics": args.log_reward_diagnostics,
        "progress_every": args.progress_every,
        "sample_target_root_octave": args.sample_target_root_octave,
        "target_root_octave_choices": args.target_root_octave_choices,
        "sample_initial_state": True,
        "initial_parent_pitch_min": args.initial_parent_pitch_min,
        "initial_parent_pitch_max": args.initial_parent_pitch_max,
    }


def _graph_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "pitch_min": args.pitch_min,
        "pitch_max": args.pitch_max,
    }


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
    args = parse_args(argv)
    exception_log_path = _exception_log_path(
        artifact_root=args.artifact_root,
        lineage_id=args.lineage_id,
    )
    try:
        parent_policy, parent_checkpoint = _load_parent_policy(
            lineage_id=args.lineage_id,
            artifact_root=args.artifact_root,
        )
        reward_config = _reward_config_from_args(args)
        policy_config = _policy_config_from_args(args)
        graph_config = _graph_config_from_args(args)
        graph_spec = TowerGraphSpec(
            rank=2,
            key_pitch_class=args.key_pitch_class,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            max_step_size=args.max_step_size,
        )
        reward_fn = build_rank2_reward_fn(
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
        )
        base_training_config = _base_training_config_from_args(args)

        stage1_lineage_id = args.lineage_id
        stage2_lineage_id = args.lineage_id

        stage1_config = TowerRunnerConfig(
            lineage_id=stage1_lineage_id,
            rank=2,
            episode_count=args.stage1_episodes,
            seed=args.seed,
            artifact_root=args.artifact_root,
            measure_size=args.measure_size,
            context_measures=args.context_measures,
            max_step_size=args.max_step_size,
            parent_checkpoint=parent_checkpoint,
            parent_top_m=args.parent_top_m,
            graph_config=graph_config,
            reward_config=reward_config,
            policy_config=policy_config,
            training_config={
                **base_training_config,
                "progress_label": "rank2-stage1",
                "sample_initial_parent_pitch_in_target_octave": True,
            },
        )
        stage1_child_policy = _build_rank2_policy(stage1_config)
        stage1_optimizer = _build_optimizer(
            policy=stage1_child_policy,
            config=stage1_config,
        )

        print(f"starting rank2 stage1: {stage1_lineage_id}")
        stage1_result = run_rank2_training(
            config=stage1_config,
            parent_policy=parent_policy,
            initial_state=(args.initial_parent_pitch, args.initial_child_pitch),
            reward_fn=reward_fn,
            child_policy=stage1_child_policy,
            child_optimizer=stage1_optimizer,
            graph_spec=graph_spec,
        )
        print(f"finished rank2 stage1: {stage1_result.paths.rank_dir}")

        archived_stage1_rank_dir = (
            stage1_result.paths.lineage_dir / "rank_2_stage1"
        )
        if archived_stage1_rank_dir.exists():
            shutil.rmtree(archived_stage1_rank_dir)
        shutil.move(
            str(stage1_result.paths.rank_dir),
            str(archived_stage1_rank_dir),
        )

        checkpoint = load_latest_checkpoint(
            archived_stage1_rank_dir / "checkpoint_latest.pt"
        )
        stage2_config = TowerRunnerConfig(
            lineage_id=stage2_lineage_id,
            rank=2,
            episode_count=args.stage2_episodes,
            seed=args.seed,
            artifact_root=args.artifact_root,
            measure_size=args.measure_size,
            context_measures=args.context_measures,
            max_step_size=args.max_step_size,
            parent_checkpoint=parent_checkpoint,
            parent_top_m=args.parent_top_m,
            graph_config=graph_config,
            reward_config=reward_config,
            policy_config=policy_config,
            training_config={
                **base_training_config,
                "progress_label": "rank2-stage2",
                "sample_initial_parent_pitch_in_target_octave": False,
            },
        )
        stage2_child_policy = _build_rank2_policy(stage2_config)
        stage2_child_policy.load_state_dict(checkpoint["policy_state_dict"])
        stage2_optimizer = _build_optimizer(
            policy=stage2_child_policy,
            config=stage2_config,
        )
        stage2_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"starting rank2 stage2: {stage2_lineage_id}")
        stage2_result = run_rank2_training(
            config=stage2_config,
            parent_policy=parent_policy,
            initial_state=(args.initial_parent_pitch, args.initial_child_pitch),
            reward_fn=reward_fn,
            child_policy=stage2_child_policy,
            child_optimizer=stage2_optimizer,
            graph_spec=graph_spec,
        )
        print(f"finished rank2 stage2: {stage2_result.paths.rank_dir}")

        print(f"rank2 stage1 run_dir: {archived_stage1_rank_dir}")
        print(f"rank2 stage2 run_dir: {stage2_result.paths.rank_dir}")
        print(f"rank2 stage2 latest checkpoint: {stage2_result.paths.checkpoint_latest_path}")
        print(f"rank2 stage2 final midi: {stage2_result.final_midi_path}")
        print(
            f"rank2 stage2 final episode return: "
            f"{stage2_result.final_inference.metrics['episode_return']}"
        )
        print(
            f"rank2 stage2 final terminal_success: "
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
