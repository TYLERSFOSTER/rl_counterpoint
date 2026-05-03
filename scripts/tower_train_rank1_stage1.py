"""Explicit rank-1 stage-1 training entrypoint for long staged curricula."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import tower_train_staged
from tower.train.lifecycle import write_run_completion, write_run_failure, write_run_heartbeat
from tower.train.runner import (
    TowerRunnerConfig,
    _build_optimizer,
    _build_rank1_policy,
    _graph_spec_from_config,
    run_rank1_training,
)


def main(argv: list[str] | None = None) -> int:
    args = tower_train_staged.parse_args(argv)
    stage_lineage_id = f"{args.lineage_id}-stage1"
    exception_log_path = args.artifact_root / "logs" / f"{stage_lineage_id}.exception.log"
    write_run_heartbeat(
        args.artifact_root / stage_lineage_id,
        lineage_id=stage_lineage_id,
        stage="rank1-stage1",
        status="running",
        completed_episodes=0,
        episode_budget=args.stage1_episodes,
        extra={"argv": [] if argv is None else list(argv)},
    )
    try:
        reward_config = tower_train_staged._reward_config_from_args(args)
        policy_config = tower_train_staged._policy_config_from_args(args)
        graph_config = tower_train_staged._graph_config_from_args(args)
        base_training_config = tower_train_staged._base_training_config_from_args(args)
        reward_fn = tower_train_staged._build_rank1_reward_from_config(reward_config)

        stage1_config = TowerRunnerConfig(
            lineage_id=stage_lineage_id,
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
        policy = _build_rank1_policy(stage1_config)
        optimizer = _build_optimizer(policy=policy, config=stage1_config)
        graph_spec = _graph_spec_from_config(stage1_config)

        print(f"starting stage1: {stage_lineage_id}")
        result = run_rank1_training(
            config=stage1_config,
            initial_state=(args.initial_pitch,),
            reward_fn=reward_fn,
            policy=policy,
            optimizer=optimizer,
            graph_spec=graph_spec,
        )
        print(f"finished stage1: {result.paths.rank_dir}")
        print(f"stage1 run_dir: {result.paths.rank_dir}")
        write_run_completion(
            result.paths,
            lineage_id=stage_lineage_id,
            stage="rank1-stage1",
            summary={
                "run_dir": result.paths.rank_dir.as_posix(),
                "checkpoint_latest_path": result.paths.checkpoint_latest_path.as_posix(),
                "final_midi_path": None
                if result.final_midi_path is None
                else result.final_midi_path.as_posix(),
            },
        )
        return 0
    except Exception as exc:
        tower_train_staged._write_exception_log(
            exception_log_path=exception_log_path,
            argv=argv,
            args=args,
        )
        write_run_failure(
            args.artifact_root / stage_lineage_id,
            lineage_id=stage_lineage_id,
            stage="rank1-stage1",
            error_type=type(exc).__name__,
            error_message=str(exc),
            exception_log=exception_log_path.as_posix(),
        )
        print(f"exception log: {exception_log_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
