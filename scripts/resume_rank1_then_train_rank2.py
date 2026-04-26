"""Resume an interrupted rank-1 lineage, then launch rank-2 and copy examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tower_train_rank2 import main as rank2_main
from scripts.tower_train_staged import _build_rank1_reward_from_config
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
    _graph_spec_from_config,
    _optional_reward_int,
    _rank1_episode_initial_state,
    _rank1_episode_target_root_octave,
    _training_bool,
    _training_float,
    _training_int,
    _write_final_midi,
    run_final_inference_episode,
)
from tower.train.protocol import train_rank1_episode_with_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume rank-1 stage 2, then train rank 2."
    )
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tower",
    )
    parser.add_argument("--rank2-episodes", type=int, required=True)
    parser.add_argument("--rank2-initial-parent-pitch", type=int, default=63)
    parser.add_argument("--rank2-initial-child-pitch", type=int, default=67)
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def _resume_rank1(lineage_id: str, artifact_root: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id=lineage_id,
        rank=1,
        artifact_root=artifact_root,
    )
    rank_config = read_rank_config(paths)
    checkpoint = load_latest_checkpoint(paths)
    last_episode_index = checkpoint["episode_index"]
    if not isinstance(last_episode_index, int):
        raise TypeError("checkpoint episode_index must be an int")

    runner_config = TowerRunnerConfig(
        lineage_id=rank_config.lineage_id,
        rank=rank_config.rank,
        episode_count=rank_config.episode_budget,
        seed=int(rank_config.seed_config.get("seed", 0)),
        artifact_root=artifact_root,
        measure_size=rank_config.measure_size,
        context_measures=rank_config.context_measures,
        max_step_size=rank_config.max_step_size,
        parent_checkpoint=rank_config.parent_checkpoint,
        reward_config=dict(rank_config.reward_config),
        graph_config=dict(rank_config.graph_config),
        policy_config=dict(rank_config.policy_config),
        training_config=dict(rank_config.training_config),
    )
    reward_fn = _build_rank1_reward_from_config(dict(rank_config.reward_config))
    policy = _build_rank1_policy(runner_config)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer = _build_optimizer(policy=policy, config=runner_config)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    graph_spec = _graph_spec_from_config(runner_config)

    generator = __import__("torch").Generator().manual_seed(runner_config.seed)
    key_pitch_class = _optional_reward_int(runner_config, "key_pitch_class")
    target_root_octave = _optional_reward_int(runner_config, "target_root_octave")

    for episode_index in range(last_episode_index + 1, rank_config.episode_budget):
        episode_target_root_octave = _rank1_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
        )
        episode_initial_state = _rank1_episode_initial_state(
            initial_state=(60,),
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
        if _training_bool(runner_config, "log_reward_diagnostics", default=True):
            append_reward_diagnostics(
                paths=paths,
                rows=reward_diagnostics_rows(
                    trajectory=episode_result.trajectory,
                    lineage_id=runner_config.lineage_id,
                    episode_index=episode_index,
                    episode_kind="training",
                ),
            )
        progress_every = _training_int(runner_config, "progress_every", default=0)
        if progress_every > 0 and (episode_index + 1) % progress_every == 0:
            metrics = episode_result.metrics
            label = runner_config.training_config.get("progress_label", f"rank{runner_config.rank}")
            print(
                f"[progress] {label} {episode_index + 1}/{rank_config.episode_budget} "
                f"return={metrics['episode_return']:.4f} "
                f"length={metrics['episode_length']} "
                f"terminal_success={metrics['terminal_success']}",
                flush=True,
            )

    max_steps = _training_int(runner_config, "max_steps", default=1)
    for final_inference_index in range(4):
        final_target_root_octave = _rank1_episode_target_root_octave(
            target_root_octave=target_root_octave,
            config=runner_config,
            generator=generator,
        )
        final_initial_state = _rank1_episode_initial_state(
            initial_state=(60,),
            target_root_octave=final_target_root_octave,
            spec=graph_spec,
            config=runner_config,
            generator=generator,
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
            sampling_temperature=_training_float(
                runner_config,
                "sampling_temperature",
                default=1.0,
            ),
            sampling_uniform_mix=_training_float(
                runner_config,
                "sampling_uniform_mix",
                default=0.0,
            ),
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


def _copy_examples(*, lineage_id: str, artifact_root: Path, examples_dir: Path) -> None:
    lineage_dir = artifact_root / lineage_id
    examples_dir.mkdir(parents=True, exist_ok=True)
    for midi_path in sorted((lineage_dir / "rank_1").glob("example_episode*.mid")):
        (examples_dir / f"rank1_{midi_path.name}").write_bytes(midi_path.read_bytes())
    for midi_path in sorted((lineage_dir / "rank_2").glob("example_episode*.mid")):
        (examples_dir / f"rank2_{midi_path.name}").write_bytes(midi_path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"resuming rank1 lineage: {args.lineage_id}", flush=True)
    _resume_rank1(args.lineage_id, args.artifact_root)
    print(f"starting rank2 lineage: {args.lineage_id}", flush=True)
    rank2_exit = rank2_main(
        [
            "--episodes",
            str(args.rank2_episodes),
            "--lineage-id",
            args.lineage_id,
            "--artifact-root",
            str(args.artifact_root),
            "--seed",
            "123",
            "--max-steps",
            "64",
            "--max-step-size",
            "7",
            "--pitch-min",
            "36",
            "--pitch-max",
            "84",
            "--learning-rate",
            "0.001",
            "--initial-parent-pitch",
            str(args.rank2_initial_parent_pitch),
            "--initial-child-pitch",
            str(args.rank2_initial_child_pitch),
            "--key-pitch-class",
            "0",
            "--target-root-octave",
            "4",
            "--parent-top-m",
            "3",
            "--terminal-cadence-reward",
            "10",
            "--cadence-endpoint-weight",
            "1.0",
            "--target-vertical-interval",
            "5",
            "--target-vertical-interval-weight",
            "1.0",
        ]
    )
    if rank2_exit != 0:
        return rank2_exit
    if args.examples_dir is not None:
        _copy_examples(
            lineage_id=args.lineage_id,
            artifact_root=args.artifact_root,
            examples_dir=args.examples_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
