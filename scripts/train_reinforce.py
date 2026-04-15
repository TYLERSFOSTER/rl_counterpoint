"""Explicit REINFORCE training entrypoint with checkpoints and metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.algos.reinforce import ReinforceEpisodeStats, run_reinforce_episode
from rl_counterpoint.algos.rollout import apply_goal_bias_to_logits
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.envs.observation import build_timed_chord_window
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.models.policy import (
    SymbolicChordEncoder,
    TransformerStepDeltaPolicy,
    encode_timed_chord_window,
)
from rl_counterpoint.music.render import write_chord_sequence_to_midi
from rl_counterpoint.reward.black_box import TargetRootOctaveReward

DEFAULT_MEASURE_SIZE = 4
DEFAULT_EPISODE_MEASURES = 16
DEFAULT_CONTEXT_MEASURES = 2
DEFAULT_MAX_STEP_SIZE = 4
DEFAULT_NUM_EPISODES = 500
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_GAMMA = 0.9
DEFAULT_ENTROPY_COEFFICIENT = 0.00
DEFAULT_EXPORT_TEMPERATURE = 1.0
DEFAULT_BEAT_ROLE_CONSONANCE_WEIGHT = 1.0
DEFAULT_EARLY_GOAL_WEIGHT = 1000.0
DEFAULT_EPSILON_BEHAVIOR = 0.9
DEFAULT_GOAL_BIAS_WEIGHT = 1000.0


class DummyTextEmbedder:
    """Deterministic local embedder for the first training harness."""

    def embed_text(self, text: str) -> torch.Tensor:
        base = float(len(text))
        return torch.tensor([base, base + 1.0, base + 2.0, base + 3.0], dtype=torch.float32)


@dataclass(frozen=True)
class TrainConfig:
    """Minimal persisted config for one local REINFORCE run."""

    measure_size: int = DEFAULT_MEASURE_SIZE
    episode_measures: int = DEFAULT_EPISODE_MEASURES
    context_measures: int = DEFAULT_CONTEXT_MEASURES
    max_step_size: int = DEFAULT_MAX_STEP_SIZE
    num_episodes: int = DEFAULT_NUM_EPISODES
    learning_rate: float = DEFAULT_LEARNING_RATE
    gamma: float = DEFAULT_GAMMA
    entropy_coefficient: float = DEFAULT_ENTROPY_COEFFICIENT
    epsilon_behavior: float = DEFAULT_EPSILON_BEHAVIOR
    goal_bias_weight: float = DEFAULT_GOAL_BIAS_WEIGHT
    export_temperature: float = DEFAULT_EXPORT_TEMPERATURE
    beat_role_consonance_weight: float = DEFAULT_BEAT_ROLE_CONSONANCE_WEIGHT
    early_goal_weight: float = DEFAULT_EARLY_GOAL_WEIGHT
    initial_state: tuple[int, ...] | None = None
    tonic: int = 60
    voice_count: int = 3
    invalid_action_penalty: float = -1.0
    target_distance_weight: float = 1.0
    target_terminal_window_reward: float = 10.0

    @property
    def max_steps(self) -> int:
        return self.measure_size * self.episode_measures


def default_run_dir() -> Path:
    """Return the default artifact directory for local REINFORCE runs."""
    return PROJECT_ROOT / "artifacts" / "train_reinforce"


def build_env(config: TrainConfig) -> CounterpointEnv:
    """Build the current local training environment from persisted config."""
    return CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=config.voice_count, tonic=config.tonic),
        reward_fn=TargetRootOctaveReward(
            distance_weight=config.target_distance_weight,
            terminal_window_reward=config.target_terminal_window_reward,
            beat_role_consonance_weight=config.beat_role_consonance_weight,
            early_goal_weight=config.early_goal_weight,
        ),
        initial_state=config.initial_state,
        max_steps=config.max_steps,
        measure_size=config.measure_size,
        max_step_size=config.max_step_size,
        invalid_action_penalty=config.invalid_action_penalty,
    )


def build_policy(
    env: CounterpointEnv,
    *,
    context_measures: int,
) -> TransformerStepDeltaPolicy:
    """Build the current local transformer policy for REINFORCE runs."""
    return TransformerStepDeltaPolicy(
        embedding_dim=4,
        action_dim=len(env.action_space),
        max_window_len=env.measure_size * context_measures,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )


def write_run_config(config: TrainConfig, run_dir: Path) -> Path:
    """Persist the local training config to JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    return config_path


def append_metrics(
    *,
    run_dir: Path,
    episode_index: int,
    stats: ReinforceEpisodeStats,
) -> Path:
    """Append one episode metrics record as JSONL."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    record = {
        "episode_index": episode_index,
        "episode_return": stats.episode_return,
        "episode_length": stats.episode_length,
        "mean_step_reward": stats.mean_step_reward,
        "terminated": stats.terminated,
        "truncated": stats.truncated,
        "loss": stats.loss,
        "target_root_octave": stats.target_root_octave,
        "final_root_octave": stats.final_root_octave,
        "final_octave_distance": stats.final_octave_distance,
        "hit_target_on_final_step": stats.hit_target_on_final_step,
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return metrics_path


def save_checkpoint(
    *,
    run_dir: Path,
    episode_index: int,
    policy: TransformerStepDeltaPolicy,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    stats: ReinforceEpisodeStats,
) -> Path:
    """Persist only the rolling latest checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "episode_index": episode_index,
        "config": asdict(config),
        "stats": {
            "episode_return": stats.episode_return,
            "episode_length": stats.episode_length,
            "mean_step_reward": stats.mean_step_reward,
            "terminated": stats.terminated,
            "truncated": stats.truncated,
            "loss": stats.loss,
            "target_root_octave": stats.target_root_octave,
            "final_root_octave": stats.final_root_octave,
            "final_octave_distance": stats.final_octave_distance,
            "hit_target_on_final_step": stats.hit_target_on_final_step,
        },
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    latest_path = run_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    return latest_path


def print_episode_summary(episode_index: int, stats: ReinforceEpisodeStats) -> None:
    """Print the core metrics for one episode update."""
    print(f"episode {episode_index} return: {stats.episode_return}")
    print(f"episode {episode_index} length: {stats.episode_length}")
    print(f"episode {episode_index} mean_step_reward: {stats.mean_step_reward}")
    print(f"episode {episode_index} terminated: {stats.terminated}")
    print(f"episode {episode_index} truncated: {stats.truncated}")
    print(f"episode {episode_index} loss: {stats.loss}")
    print(f"episode {episode_index} target_root_octave: {stats.target_root_octave}")
    print(f"episode {episode_index} final_root_octave: {stats.final_root_octave}")
    print(f"episode {episode_index} final_octave_distance: {stats.final_octave_distance}")
    print(
        f"episode {episode_index} hit_target_on_final_step: "
        f"{stats.hit_target_on_final_step}"
    )


def choose_export_action_index(
    *,
    legal_indices: list[int],
    legal_logits: torch.Tensor,
    export_temperature: float,
    rng: Random,
) -> int:
    """Choose one legal action for MIDI export under a temperature knob."""
    if not legal_indices:
        raise RuntimeError("no legal StepDelta available during evaluation")
    if legal_logits.ndim != 1:
        raise ValueError("legal_logits must be rank 1 [legal_action_dim]")
    if legal_logits.shape[0] != len(legal_indices):
        raise ValueError("legal_logits length must match legal_indices length")
    if export_temperature < 0.0:
        raise ValueError("export_temperature must be non-negative")

    if export_temperature == 0.0:
        best_legal_position = int(torch.argmax(legal_logits).item())
        return legal_indices[best_legal_position]

    probabilities = torch.softmax(legal_logits / export_temperature, dim=0).tolist()
    return rng.choices(legal_indices, weights=probabilities, k=1)[0]


def export_example_episode_midi(
    *,
    run_dir: Path,
    env: CounterpointEnv,
    policy: TransformerStepDeltaPolicy,
    encoder: SymbolicChordEncoder,
    context_measures: int,
    export_temperature: float,
    goal_bias_weight: float = 0.0,
    seed: int | None = None,
) -> Path:
    """Run one temperature-controlled evaluation episode and export it as MIDI."""
    observation, info = env.reset(seed=seed)
    chord_sequence = [observation]
    rng = Random(seed)

    with torch.no_grad():
        while True:
            timed_window = build_timed_chord_window(
                history=env.history,
                step_index=env.step_index,
                measure_size=env.measure_size,
                context_measures=context_measures,
            )
            encoded_window = encode_timed_chord_window(
                window=timed_window,
                tonic=env.graph_spec.tonic,
                measure_size=env.measure_size,
                encoder=encoder,
                target_root_octave=info.get("target_root_octave"),
            )
            logits = policy(encoded_window)
            action_space = info["action_space"]
            action_mask = info["action_mask"]
            if not isinstance(action_space, tuple) or not isinstance(action_mask, tuple):
                raise TypeError("action_space and action_mask must be tuples")
            adjusted_logits = apply_goal_bias_to_logits(
                current_state=observation,
                action_space=action_space,
                action_mask=action_mask,
                logits=logits,
                target_root_octave=info.get("target_root_octave"),
                goal_bias_weight=goal_bias_weight,
            )

            legal_indices = [
                index for index, is_legal in enumerate(action_mask) if is_legal
            ]
            if not legal_indices:
                raise RuntimeError("no legal StepDelta available during evaluation")

            legal_logits = adjusted_logits[legal_indices]
            action_index = choose_export_action_index(
                legal_indices=legal_indices,
                legal_logits=legal_logits,
                export_temperature=export_temperature,
                rng=rng,
            )
            step_delta = action_space[action_index]

            observation, _reward, terminated, truncated, info = env.step(step_delta)
            chord_sequence.append(observation)

            if terminated or truncated:
                midi_path = write_chord_sequence_to_midi(
                    chord_sequence=tuple(chord_sequence),
                    path=run_dir / "example_episode.mid",
                )
                return midi_path


def main(*, run_dir: Path | None = None) -> None:
    config = TrainConfig()
    output_dir = run_dir or default_run_dir()
    write_run_config(config, output_dir)

    env = build_env(config)
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())
    policy = build_policy(env, context_measures=config.context_measures)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    print(f"run_dir: {output_dir}")
    print(f"measure_size: {config.measure_size}")
    print(f"episode_measures: {config.episode_measures}")
    print(f"max_steps: {config.max_steps}")
    print(f"entropy_coefficient: {config.entropy_coefficient}")
    print(f"epsilon_behavior: {config.epsilon_behavior}")
    print(f"goal_bias_weight: {config.goal_bias_weight}")
    print(f"export_temperature: {config.export_temperature}")
    print(f"beat_role_consonance_weight: {config.beat_role_consonance_weight}")
    print(f"early_goal_weight: {config.early_goal_weight}")

    for episode_index in range(config.num_episodes):
        stats = run_reinforce_episode(
            env,
            policy=policy,
            encoder=encoder,
            optimizer=optimizer,
            gamma=config.gamma,
            entropy_coefficient=config.entropy_coefficient,
            context_measures=config.context_measures,
            epsilon_behavior=config.epsilon_behavior,
            goal_bias_weight=config.goal_bias_weight,
        )
        append_metrics(
            run_dir=output_dir,
            episode_index=episode_index,
            stats=stats,
        )
        latest_checkpoint = save_checkpoint(
            run_dir=output_dir,
            episode_index=episode_index,
            policy=policy,
            optimizer=optimizer,
            config=config,
            stats=stats,
        )
        print_episode_summary(episode_index, stats)
        print(f"latest checkpoint: {latest_checkpoint}")

    midi_path = export_example_episode_midi(
        run_dir=output_dir,
        env=env,
        policy=policy,
        encoder=encoder,
        context_measures=config.context_measures,
        export_temperature=config.export_temperature,
        goal_bias_weight=config.goal_bias_weight,
    )
    print(f"example episode midi: {midi_path}")


if __name__ == "__main__":
    main()
