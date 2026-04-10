"""Explicit REINFORCE training entrypoint with checkpoints and metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.algos.reinforce import ReinforceEpisodeStats, run_reinforce_episode
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.models.policy import SymbolicChordEncoder, TransformerStepDeltaPolicy
from rl_counterpoint.reward.black_box import ConstantReward

DEFAULT_MEASURE_SIZE = 4
DEFAULT_EPISODE_MEASURES = 8
DEFAULT_CONTEXT_MEASURES = 3
DEFAULT_MAX_STEP_SIZE = 2
DEFAULT_NUM_EPISODES = 3
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GAMMA = 0.99


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
    initial_state: tuple[int, ...] = (3, 6)
    tonic: int = 60
    voice_count: int = 2
    invalid_action_penalty: float = -1.0
    reward_value: float = 1.0

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
        reward_fn=ConstantReward(reward=config.reward_value),
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
) -> tuple[Path, Path]:
    """Persist both a per-episode checkpoint and a rolling latest checkpoint."""
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
        },
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    episode_path = run_dir / f"checkpoint_episode_{episode_index:04d}.pt"
    latest_path = run_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, episode_path)
    torch.save(checkpoint, latest_path)
    return episode_path, latest_path


def print_episode_summary(episode_index: int, stats: ReinforceEpisodeStats) -> None:
    """Print the core metrics for one episode update."""
    print(f"episode {episode_index} return: {stats.episode_return}")
    print(f"episode {episode_index} length: {stats.episode_length}")
    print(f"episode {episode_index} mean_step_reward: {stats.mean_step_reward}")
    print(f"episode {episode_index} terminated: {stats.terminated}")
    print(f"episode {episode_index} truncated: {stats.truncated}")
    print(f"episode {episode_index} loss: {stats.loss}")


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

    for episode_index in range(config.num_episodes):
        stats = run_reinforce_episode(
            env,
            policy=policy,
            encoder=encoder,
            optimizer=optimizer,
            gamma=config.gamma,
            context_measures=config.context_measures,
            seed=episode_index,
        )
        append_metrics(
            run_dir=output_dir,
            episode_index=episode_index,
            stats=stats,
        )
        episode_checkpoint, latest_checkpoint = save_checkpoint(
            run_dir=output_dir,
            episode_index=episode_index,
            policy=policy,
            optimizer=optimizer,
            config=config,
            stats=stats,
        )
        print_episode_summary(episode_index, stats)
        print(f"episode {episode_index} checkpoint: {episode_checkpoint}")
        print(f"latest checkpoint: {latest_checkpoint}")


if __name__ == "__main__":
    main()
