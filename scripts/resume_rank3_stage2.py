"""Resume or start explicit rank-3 stage-2 training from stage-1 checkpoint."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import tower_train_rank3
from scripts._tower_resume import resume_rank3_lineage
from tower.train.checkpoint import TowerArtifactPaths, load_latest_checkpoint
from tower.train.lifecycle import write_run_completion, write_run_failure, write_run_heartbeat
from tower.train.runner import TowerRunnerConfig, _build_optimizer, _build_rank3_policy, run_rank3_training


def main(argv: list[str] | None = None) -> int:
    args = tower_train_rank3.parse_args(argv)
    stage2_lineage_id = args.lineage_id
    exception_log_path = args.artifact_root / "logs" / f"{stage2_lineage_id}-rank3-stage2.exception.log"
    write_run_heartbeat(
        args.artifact_root / stage2_lineage_id,
        lineage_id=stage2_lineage_id,
        stage="rank3-stage2",
        status="running",
        completed_episodes=0,
        episode_budget=args.episodes,
        extra={"argv": [] if argv is None else list(argv)},
    )
    try:
        stage2_paths = TowerArtifactPaths(
            lineage_id=stage2_lineage_id,
            rank=3,
            artifact_root=args.artifact_root,
        )
        if stage2_paths.checkpoint_latest_path.exists():
            print(f"resuming rank3 stage2: {stage2_lineage_id}")
            resume_rank3_lineage(
                lineage_id=stage2_lineage_id,
                artifact_root=args.artifact_root,
                initial_state=(
                    args.initial_pedal_pitch,
                    args.initial_middle_pitch,
                    args.initial_top_pitch,
                ),
            )
            print(f"rank3 stage2 run_dir: {stage2_paths.rank_dir}")
            write_run_completion(
                stage2_paths,
                lineage_id=stage2_lineage_id,
                stage="rank3-stage2",
                summary={
                    "run_dir": stage2_paths.rank_dir.as_posix(),
                    "checkpoint_latest_path": stage2_paths.checkpoint_latest_path.as_posix(),
                },
            )
            return 0

        (
            grandparent_policy,
            parent_policy,
            grandparent_checkpoint,
            parent_checkpoint,
        ) = tower_train_rank3._load_parent_stack(
            lineage_id=args.lineage_id,
            artifact_root=args.artifact_root,
        )
        stage1_checkpoint_path = (
            args.artifact_root
            / args.lineage_id
            / "rank_3_stage1"
            / "checkpoint_latest.pt"
        )
        checkpoint = load_latest_checkpoint(stage1_checkpoint_path)
        reward_config = tower_train_rank3._reward_config_from_args(args)
        config = TowerRunnerConfig(
            lineage_id=stage2_lineage_id,
            rank=3,
            episode_count=args.episodes,
            seed=args.seed,
            artifact_root=args.artifact_root,
            measure_size=args.measure_size,
            context_measures=args.context_measures,
            max_step_size=args.max_step_size,
            parent_checkpoint=parent_checkpoint,
            parent_top_m=args.parent_top_m,
            graph_config=tower_train_rank3._graph_config_from_args(args),
            reward_config=reward_config,
            policy_config=tower_train_rank3._policy_config_from_args(args),
            training_config={
                **tower_train_rank3._training_config_from_args(args),
                "sample_initial_state": True,
                "sample_initial_parent_pitch_in_target_octave": False,
            },
        )
        child_policy = _build_rank3_policy(config)
        child_policy.load_state_dict(checkpoint["policy_state_dict"])
        child_optimizer = _build_optimizer(policy=child_policy, config=config)
        child_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        result = run_rank3_training(
            config=config,
            grandparent_policy=grandparent_policy,
            parent_policy=parent_policy,
            initial_state=(
                args.initial_pedal_pitch,
                args.initial_middle_pitch,
                args.initial_top_pitch,
            ),
            reward_fn=tower_train_rank3.build_rank3_reward_fn(
                key_pitch_class=args.key_pitch_class,
                use_context_key_pitch_class=args.sample_key_pitch_class,
                target_root_octave=args.target_root_octave,
                use_context_target_root_octave=args.sample_target_root_octave,
                terminal_cadence_reward=args.terminal_cadence_reward,
                cadence_failure_reward=args.cadence_failure_reward,
                triad_consonance_weight=args.triad_consonance_weight,
                triad_non_consonance_penalty=args.triad_non_consonance_penalty,
                min_adjacent_gap=args.min_adjacent_gap,
                max_outer_span=args.max_outer_span,
                adjacent_spacing_reward=args.adjacent_spacing_reward,
                adjacent_spacing_penalty=args.adjacent_spacing_penalty,
                outer_span_reward=args.outer_span_reward,
                outer_span_penalty=args.outer_span_penalty,
                cadence_endpoint_weight=args.cadence_endpoint_weight,
                onbeat_all_scale_degree_reward=args.onbeat_all_scale_degree_reward,
                onbeat_not_all_scale_degree_penalty=(
                    args.onbeat_not_all_scale_degree_penalty
                ),
                offbeat_all_consonant_weight=args.offbeat_all_consonant_weight,
                offbeat_non_consonance_penalty=args.offbeat_non_consonance_penalty,
            ),
            child_policy=child_policy,
            child_optimizer=child_optimizer,
            graph_spec=tower_train_rank3._graph_spec_from_config(config),
        )
        print(f"rank3 stage2 run_dir: {result.paths.rank_dir}")
        write_run_completion(
            result.paths,
            lineage_id=stage2_lineage_id,
            stage="rank3-stage2",
            summary={
                "run_dir": result.paths.rank_dir.as_posix(),
                "checkpoint_latest_path": result.paths.checkpoint_latest_path.as_posix(),
                "final_midi_path": None
                if result.final_midi_path is None
                else result.final_midi_path.as_posix(),
                "grandparent_checkpoint": grandparent_checkpoint,
                "parent_checkpoint": parent_checkpoint,
            },
        )
        return 0
    except Exception as exc:
        exception_log_path.parent.mkdir(parents=True, exist_ok=True)
        exception_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_run_failure(
            args.artifact_root / stage2_lineage_id,
            lineage_id=stage2_lineage_id,
            stage="rank3-stage2",
            error_type=type(exc).__name__,
            error_message=str(exc),
            exception_log=exception_log_path.as_posix(),
        )
        print(f"exception log: {exception_log_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
