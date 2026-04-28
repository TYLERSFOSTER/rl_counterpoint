"""Tests for tower training protocol helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tower.graph.spec import TowerGraphSpec
from tower.policy.base import PolicyOutput
from tower.policy.transformer import (
    TowerTransformerPolicy,
    TowerTransformerPolicyConfig,
)
from tower.reward.context import TowerRewardContext
from tower.reward.result import TowerRewardResult
from tower.train.checkpoint import (
    TowerArtifactPaths,
    build_checkpoint_payload,
    load_latest_checkpoint,
    read_lineage_manifest,
    read_rank_config,
    read_rank_metrics,
    record_rank_manifest_entry,
    save_latest_checkpoint,
)
from tower.train.config import TowerRankConfig
from tower.train.protocol import (
    train_rank1_episode,
    train_rank1_episode_with_artifacts,
    train_rank2_episode,
    train_rank2_episode_with_artifacts,
    train_rank3_episode,
    train_rank3_episode_with_artifacts,
)
from tower.window import TowerWindow


class TinyRank1Policy(torch.nn.Module):
    rank = 1

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert len(state) == 1
        assert window.valid_mask[-1]
        return PolicyOutput(logits=self.logits)


class TinyRank1GrandparentPolicy(torch.nn.Module):
    rank = 1

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([2.0, 0.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert len(state) == 1
        assert window.valid_mask[-1]
        return PolicyOutput(logits=self.logits)


class TinyRank2Policy(torch.nn.Module):
    rank = 2

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 2.0, 0.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert len(state) == 2
        assert window.valid_mask[-1]
        return PolicyOutput(logits=self.logits)


class TinyRank2ParentPolicy(torch.nn.Module):
    rank = 2

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([2.0, 0.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert len(state) == 2
        assert window.valid_mask[-1]
        return PolicyOutput(logits=self.logits)


class TinyRank3Policy(torch.nn.Module):
    rank = 3

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert len(state) == 3
        assert window.valid_mask[-1]
        return PolicyOutput(logits=self.logits)


def make_tiny_transformer_policy(
    *,
    rank: int,
    action_dim: int,
) -> TowerTransformerPolicy:
    return TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=rank,
            input_feature_dim=rank + 5,
            action_dim=action_dim,
            max_window_len=8,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    )


def test_train_rank1_episode_runs_one_episode() -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    result = train_rank1_episode(
        policy=policy,
        optimizer=optimizer,
        initial_state=(60,),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert len(result.trajectory.steps) == 1
    assert result.trajectory.rank == 1
    assert isinstance(result.loss.loss, torch.Tensor)
    assert result.metrics["rank"] == 1
    assert result.metrics["episode_length"] == 1
    assert result.metrics["episode_return"] == 1.0


def test_train_rank1_episode_optimizer_step_changes_params() -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.logits.detach().clone()

    train_rank1_episode(
        policy=policy,
        optimizer=optimizer,
        initial_state=(60,),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=2),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert not torch.allclose(policy.logits.detach(), before)


def test_train_rank1_episode_metrics_include_terminal_and_rollout_counts() -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    def reward_fn(context: TowerRewardContext) -> TowerRewardResult:
        return TowerRewardResult(
            reward=2.0,
            is_terminal_success=True,
        )

    result = train_rank1_episode(
        policy=policy,
        optimizer=optimizer,
        initial_state=(60,),
        max_steps=3,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
        reward_fn=reward_fn,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.metrics["rank"] == 1
    assert result.metrics["episode_return"] == 2.0
    assert result.metrics["episode_length"] == 1
    assert result.metrics["mean_step_reward"] == 2.0
    assert result.metrics["terminated"] is True
    assert result.metrics["truncated"] is False
    assert result.metrics["invalid_extension_count"] == 0
    assert result.metrics["empty_lift_fiber_count"] == 0
    assert result.metrics["parent_failure_count"] == 0
    assert result.metrics["terminal_success"] is True
    assert isinstance(result.metrics["loss"], float)


def test_train_rank1_episode_rejects_non_rank_1_policy() -> None:
    policy = TinyRank1Policy()
    policy.rank = 2  # type: ignore[misc]
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    try:
        train_rank1_episode(
            policy=policy,
            optimizer=optimizer,
            initial_state=(60,),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=1),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
        )
    except ValueError as error:
        assert "rank-1 policy" in str(error)
    else:
        raise AssertionError("expected rank-1 policy validation error")


def test_train_rank1_episode_with_artifacts_writes_config_metrics_and_checkpoint(
    tmp_path,
) -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=2,
        max_step_size=1,
        training_config={"max_steps": 1, "gamma": 1.0},
    )

    result = train_rank1_episode_with_artifacts(
        policy=policy,
        optimizer=optimizer,
        config=config,
        paths=paths,
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    assert read_rank_config(paths) == config
    assert read_rank_metrics(paths) == (
        {
            **result.metrics,
        },
    )
    checkpoint = load_latest_checkpoint(paths)
    assert checkpoint["rank"] == 1
    assert checkpoint["lineage_id"] == "lineage-a"
    assert checkpoint["episode_index"] == 0
    assert checkpoint["config"] == config.to_json_dict()
    assert checkpoint["stats"] == result.metrics
    assert "logits" in checkpoint["policy_state_dict"]
    assert "state" in checkpoint["optimizer_state_dict"]
    assert all(step.active_logprob is not None for step in result.trajectory.steps)


def test_train_rank1_episode_with_artifacts_records_running_before_budget(
    tmp_path,
) -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=2,
        max_step_size=1,
        training_config={"max_steps": 1},
    )

    train_rank1_episode_with_artifacts(
        policy=policy,
        optimizer=optimizer,
        config=config,
        paths=paths,
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    assert read_lineage_manifest(paths)["ranks"]["1"]["status"] == "running"


def test_train_rank1_episode_with_artifacts_records_accepted_at_budget(
    tmp_path,
) -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=1,
        max_step_size=1,
        training_config={"max_steps": 1},
    )

    train_rank1_episode_with_artifacts(
        policy=policy,
        optimizer=optimizer,
        config=config,
        paths=paths,
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    assert read_lineage_manifest(paths)["ranks"]["1"]["status"] == "accepted"


def test_train_rank1_episode_with_artifacts_optimizer_step_changes_params(
    tmp_path,
) -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.logits.detach().clone()

    train_rank1_episode_with_artifacts(
        policy=policy,
        optimizer=optimizer,
        config=TowerRankConfig(
            rank=1,
            lineage_id="lineage-a",
            episode_budget=1,
            max_step_size=2,
            training_config={"max_steps": 1},
        ),
        paths=TowerArtifactPaths(
            lineage_id="lineage-a",
            rank=1,
            artifact_root=tmp_path,
        ),
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    assert not torch.allclose(policy.logits.detach(), before)


def test_training_protocol_does_not_import_frozen_legacy_project() -> None:
    project_root = Path(__file__).parents[3]
    source_text = (project_root / "tower" / "train" / "protocol.py").read_text()

    forbidden_imports = (
        "from rl_counterpoint",
        "import rl_counterpoint",
    )

    assert not any(forbidden in source_text for forbidden in forbidden_imports)


def test_train_rank2_episode_runs_over_frozen_parent() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)

    result = train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(64, 67),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.trajectory.rank == 2
    assert len(result.trajectory.steps) == 1
    step = result.trajectory.steps[0]
    assert step.parent_action == (1,)
    assert step.parent_logprob is not None
    assert step.active_logprob is not None
    assert "parent_sampler" in step.diagnostics
    assert "active_sampler" in step.diagnostics
    assert result.metrics["rank"] == 2
    assert result.metrics["episode_return"] == 1.0


def test_train_rank1_episode_runs_with_transformer_policy() -> None:
    policy = make_tiny_transformer_policy(rank=1, action_dim=2)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

    result = train_rank1_episode(
        policy=policy,  # type: ignore[arg-type]
        optimizer=optimizer,
        initial_state=(60,),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.trajectory.rank == 1
    assert len(result.trajectory.steps) == 1
    step = result.trajectory.steps[0]
    assert step.diagnostics["active_sampler"]["frontier_state"] == (60,)
    assert step.diagnostics["active_sampler"]["policy"]["policy"] == "tower_transformer"
    assert isinstance(step.active_logprob, torch.Tensor)


def test_train_rank2_episode_freezes_and_preserves_parent_params() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    parent_before = parent_policy.logits.detach().clone()

    train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(64, 67),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(parent_policy.logits.detach(), parent_before)
    assert not parent_policy.training
    assert all(not parameter.requires_grad for parameter in parent_policy.parameters())


def test_train_rank2_episode_child_optimizer_step_changes_child_params() -> None:
    class WideRank1Policy(torch.nn.Module):
        rank = 1

        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 2.0]))

        def forward(
            self,
            *,
            state: tuple[int, ...],
            window: TowerWindow,
        ) -> PolicyOutput:
            assert len(state) == 1
            assert window.valid_mask[-1]
            return PolicyOutput(logits=self.logits)

    parent_policy = WideRank1Policy()

    class BinaryRank2Policy(torch.nn.Module):
        rank = 2

        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([0.0, 2.0]))

        def forward(
            self,
            *,
            state: tuple[int, ...],
            window: TowerWindow,
        ) -> PolicyOutput:
            assert len(state) == 2
            assert window.valid_mask[-1]
            return PolicyOutput(logits=self.logits)

    child_policy = BinaryRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    child_before = child_policy.logits.detach().clone()

    train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(36, 40),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, pitch_min=36, pitch_max=84, max_step_size=2),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert not torch.allclose(child_policy.logits.detach(), child_before)


def test_train_rank2_episode_uses_lift_fiber_mask_for_child_choices() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)

    result = train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(64, 67),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    step = result.trajectory.steps[0]
    assert step.diagnostics["active_choices"] == (1,)
    assert step.diagnostics["active_sampler"]["active_choices"] == (1,)


def test_train_rank2_episode_records_parent_top_m_sampler_diagnostics() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)

    result = train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(64, 67),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        parent_top_m=1,
        generator=torch.Generator().manual_seed(0),
    )

    parent_sampler = result.trajectory.steps[0].diagnostics["parent_sampler"]
    assert parent_sampler["top_m"] == 1
    assert parent_sampler["top_indices"] == (0,)
    assert parent_sampler["parent_actions"] == ((1,),)
    assert parent_sampler["unfiltered_parent_actions"] == ((1,),)
    assert parent_sampler["feasible_parent_actions"] == ((1,),)
    assert parent_sampler["parent_feasibility_filter_applied"] is False


def test_train_rank2_episode_filters_parent_actions_to_nonempty_lifts() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)

    result = train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(76, 84),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, pitch_min=36, pitch_max=84, max_step_size=2),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        parent_top_m=1,
        generator=torch.Generator().manual_seed(0),
    )

    parent_sampler = result.trajectory.steps[0].diagnostics["parent_sampler"]
    assert parent_sampler["unfiltered_parent_actions"] == ((-2,), (1,))
    assert parent_sampler["feasible_parent_actions"] == ((-2,),)
    assert parent_sampler["parent_actions"] == ((-2,),)
    assert parent_sampler["parent_feasibility_filter_applied"] is True
    assert result.trajectory.steps[0].parent_action == (-2,)


def test_train_rank2_episode_loss_ignores_parent_logprob_gradient() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)

    train_rank2_episode(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(64, 67),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert parent_policy.logits.grad is None
    assert child_policy.logits.grad is not None


def test_train_rank2_episode_rejects_wrong_policy_ranks() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    parent_policy.rank = 2  # type: ignore[misc]

    try:
        train_rank2_episode(
            parent_policy=parent_policy,
            child_policy=child_policy,
            child_optimizer=child_optimizer,
            initial_state=(64, 67),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
        )
    except ValueError as error:
        assert "rank-1 parent policy" in str(error)
    else:
        raise AssertionError("expected rank-1 parent policy validation error")


def prepare_accepted_rank1_parent(
    *,
    tmp_path: Path,
    lineage_id: str = "lineage-a",
) -> TowerArtifactPaths:
    rank_1_paths = TowerArtifactPaths(
        lineage_id=lineage_id,
        rank=1,
        artifact_root=tmp_path,
    )
    rank_1_config = TowerRankConfig(
        rank=1,
        lineage_id=lineage_id,
        episode_budget=1,
        max_step_size=1,
    )
    save_latest_checkpoint(
        paths=rank_1_paths,
        payload=build_checkpoint_payload(
            config=rank_1_config,
            episode_index=0,
            stats={"episode_return": 1.0},
            policy_state_dict={"parent_weight": [1.0]},
            optimizer_state_dict={"step": 1},
        ),
    )
    record_rank_manifest_entry(paths=rank_1_paths, status="accepted")
    return rank_1_paths


def prepare_accepted_rank2_parent(
    *,
    tmp_path: Path,
    lineage_id: str = "lineage-a",
) -> TowerArtifactPaths:
    prepare_accepted_rank1_parent(tmp_path=tmp_path, lineage_id=lineage_id)
    rank_2_paths = TowerArtifactPaths(
        lineage_id=lineage_id,
        rank=2,
        artifact_root=tmp_path,
    )
    rank_2_config = TowerRankConfig(
        rank=2,
        lineage_id=lineage_id,
        episode_budget=1,
        max_step_size=1,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
        parent_sampler_config={"top_m": 1},
    )
    save_latest_checkpoint(
        paths=rank_2_paths,
        payload=build_checkpoint_payload(
            config=rank_2_config,
            episode_index=0,
            stats={"episode_return": 1.0},
            policy_state_dict={"parent_weight": [1.0]},
            optimizer_state_dict={"step": 1},
            parent_checkpoint="rank_1/checkpoint_latest.pt",
            parent_artifact_schema_version=1,
        ),
    )
    record_rank_manifest_entry(
        paths=rank_2_paths,
        status="accepted",
        parent_checkpoint="rank_1/checkpoint_latest.pt",
    )
    return rank_2_paths


def test_train_rank3_episode_runs_over_frozen_parent_stack() -> None:
    grandparent_policy = TinyRank1GrandparentPolicy()
    parent_policy = TinyRank2ParentPolicy()
    child_policy = TinyRank3Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)

    result = train_rank3_episode(
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(62, 65, 69),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=3, max_step_size=2),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.trajectory.rank == 3
    assert len(result.trajectory.steps) == 1
    step = result.trajectory.steps[0]
    assert step.parent_action == (-2, -1)
    assert step.parent_logprob is not None
    assert step.active_logprob is not None
    assert "grandparent_sampler" in step.diagnostics["parent_sampler"]
    assert "parent_sampler" in step.diagnostics["parent_sampler"]
    assert result.metrics["rank"] == 3
    assert result.metrics["episode_return"] == 1.0


def test_train_rank3_episode_freezes_and_preserves_parent_params() -> None:
    grandparent_policy = TinyRank1GrandparentPolicy()
    parent_policy = TinyRank2ParentPolicy()
    child_policy = TinyRank3Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    grandparent_before = grandparent_policy.logits.detach().clone()
    parent_before = parent_policy.logits.detach().clone()

    train_rank3_episode(
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(62, 65, 69),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=3, max_step_size=2),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(grandparent_policy.logits.detach(), grandparent_before)
    assert torch.equal(parent_policy.logits.detach(), parent_before)
    assert not grandparent_policy.training
    assert not parent_policy.training
    assert all(not parameter.requires_grad for parameter in grandparent_policy.parameters())
    assert all(not parameter.requires_grad for parameter in parent_policy.parameters())


def test_train_rank3_episode_filters_grandparent_actions_to_nonempty_parent_lifts() -> None:
    grandparent_policy = TinyRank1GrandparentPolicy()
    parent_policy = TinyRank2ParentPolicy()
    child_policy = TinyRank3Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)

    result = train_rank3_episode(
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(62, 65, 69),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=3, max_step_size=2),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        parent_top_m=1,
        generator=torch.Generator().manual_seed(0),
    )

    parent_sampler = result.trajectory.steps[0].diagnostics["parent_sampler"]
    grandparent_sampler = parent_sampler["grandparent_sampler"]
    assert grandparent_sampler["unfiltered_parent_actions"] == ((-2,), (2,))
    assert grandparent_sampler["feasible_parent_actions"] == ((-2,),)
    assert grandparent_sampler["parent_actions"] == ((-2,),)
    assert grandparent_sampler["parent_feasibility_filter_applied"] is True
    assert parent_sampler["parent_active_choices"] == (-1,)
    assert result.trajectory.steps[0].parent_action == (-2, -1)


def test_train_rank3_episode_rejects_wrong_policy_ranks() -> None:
    grandparent_policy = TinyRank1GrandparentPolicy()
    parent_policy = TinyRank2ParentPolicy()
    child_policy = TinyRank3Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    parent_policy.rank = 1  # type: ignore[misc]

    with pytest.raises(ValueError, match="rank-2 parent policy"):
        train_rank3_episode(
            grandparent_policy=grandparent_policy,
            parent_policy=parent_policy,
            child_policy=child_policy,
            child_optimizer=child_optimizer,
            initial_state=(62, 65, 69),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=3, max_step_size=2),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
        )


def test_train_rank3_episode_with_artifacts_writes_parent_linked_artifacts(
    tmp_path: Path,
) -> None:
    prepare_accepted_rank2_parent(tmp_path=tmp_path)
    grandparent_policy = TinyRank1GrandparentPolicy()
    parent_policy = TinyRank2ParentPolicy()
    child_policy = TinyRank3Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=3,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=3,
        lineage_id="lineage-a",
        episode_budget=2,
        max_step_size=2,
        parent_checkpoint="rank_2/checkpoint_latest.pt",
        parent_sampler_config={"top_m": 1},
        training_config={"max_steps": 1, "gamma": 1.0},
    )

    result = train_rank3_episode_with_artifacts(
        grandparent_policy=grandparent_policy,
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        config=config,
        paths=paths,
        initial_state=(62, 65, 69),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    assert read_rank_config(paths) == config
    assert read_rank_metrics(paths) == ({**result.metrics},)
    checkpoint = load_latest_checkpoint(paths)
    assert checkpoint["rank"] == 3
    assert checkpoint["lineage_id"] == "lineage-a"
    assert checkpoint["stats"] == result.metrics
    assert checkpoint["parent_rank"] == 2
    assert checkpoint["parent_checkpoint"] == "rank_2/checkpoint_latest.pt"
    assert checkpoint["parent_artifact_schema_version"] == 1
    manifest_entry = read_lineage_manifest(paths)["ranks"]["3"]
    assert manifest_entry["status"] == "running"
    assert manifest_entry["parent_rank"] == 2
    assert manifest_entry["parent_checkpoint"] == "rank_2/checkpoint_latest.pt"


def test_train_rank3_episode_with_artifacts_rejects_missing_accepted_parent(
    tmp_path: Path,
) -> None:
    child_policy = TinyRank3Policy()

    with pytest.raises(ValueError, match="accepted parent checkpoint is missing"):
        train_rank3_episode_with_artifacts(
            grandparent_policy=TinyRank1Policy(),
            parent_policy=TinyRank2ParentPolicy(),
            child_policy=child_policy,
            child_optimizer=torch.optim.SGD(child_policy.parameters(), lr=0.1),
            config=TowerRankConfig(
                rank=3,
                lineage_id="lineage-a",
                episode_budget=1,
                max_step_size=1,
                parent_checkpoint="rank_2/checkpoint_latest.pt",
                parent_sampler_config={"top_m": 1},
                training_config={"max_steps": 1},
            ),
            paths=TowerArtifactPaths(
                lineage_id="lineage-a",
                rank=3,
                artifact_root=tmp_path,
            ),
            initial_state=(62, 65, 69),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
            episode_index=0,
        )


def test_train_rank2_episode_with_artifacts_writes_parent_linked_artifacts(
    tmp_path: Path,
) -> None:
    prepare_accepted_rank1_parent(tmp_path=tmp_path)
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=2,
        lineage_id="lineage-a",
        episode_budget=2,
        max_step_size=1,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
        parent_sampler_config={"top_m": 1},
        training_config={"max_steps": 1, "gamma": 1.0},
    )

    result = train_rank2_episode_with_artifacts(
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        config=config,
        paths=paths,
        initial_state=(64, 67),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    assert read_rank_config(paths) == config
    assert read_rank_metrics(paths) == ({**result.metrics},)
    checkpoint = load_latest_checkpoint(paths)
    assert checkpoint["rank"] == 2
    assert checkpoint["lineage_id"] == "lineage-a"
    assert checkpoint["stats"] == result.metrics
    assert checkpoint["parent_rank"] == 1
    assert checkpoint["parent_checkpoint"] == "rank_1/checkpoint_latest.pt"
    assert checkpoint["parent_artifact_schema_version"] == 1
    assert "parent_policy_state_dict" not in checkpoint
    manifest_entry = read_lineage_manifest(paths)["ranks"]["2"]
    assert manifest_entry["status"] == "running"
    assert manifest_entry["parent_rank"] == 1
    assert manifest_entry["parent_checkpoint"] == "rank_1/checkpoint_latest.pt"


def test_train_rank2_episode_with_artifacts_records_accepted_at_budget(
    tmp_path: Path,
) -> None:
    prepare_accepted_rank1_parent(tmp_path=tmp_path)
    child_policy = TinyRank2Policy()

    train_rank2_episode_with_artifacts(
        parent_policy=TinyRank1Policy(),
        child_policy=child_policy,
        child_optimizer=torch.optim.SGD(child_policy.parameters(), lr=0.1),
        config=TowerRankConfig(
            rank=2,
            lineage_id="lineage-a",
            episode_budget=1,
            max_step_size=1,
            parent_checkpoint="rank_1/checkpoint_latest.pt",
            parent_sampler_config={"top_m": 1},
            training_config={"max_steps": 1},
        ),
        paths=TowerArtifactPaths(
            lineage_id="lineage-a",
            rank=2,
            artifact_root=tmp_path,
        ),
        initial_state=(64, 67),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    manifest = read_lineage_manifest(
        TowerArtifactPaths(
            lineage_id="lineage-a",
            rank=2,
            artifact_root=tmp_path,
        )
    )
    assert manifest["ranks"]["2"]["status"] == "accepted"


def test_train_rank2_episode_with_artifacts_leaves_parent_checkpoint_unchanged(
    tmp_path: Path,
) -> None:
    rank_1_paths = prepare_accepted_rank1_parent(tmp_path=tmp_path)
    parent_checkpoint_bytes = rank_1_paths.checkpoint_latest_path.read_bytes()
    parent_checkpoint_payload = load_latest_checkpoint(rank_1_paths)
    child_policy = TinyRank2Policy()

    train_rank2_episode_with_artifacts(
        parent_policy=TinyRank1Policy(),
        child_policy=child_policy,
        child_optimizer=torch.optim.SGD(child_policy.parameters(), lr=0.1),
        config=TowerRankConfig(
            rank=2,
            lineage_id="lineage-a",
            episode_budget=1,
            max_step_size=1,
            parent_checkpoint="rank_1/checkpoint_latest.pt",
            parent_sampler_config={"top_m": 1},
            training_config={"max_steps": 1},
        ),
        paths=TowerArtifactPaths(
            lineage_id="lineage-a",
            rank=2,
            artifact_root=tmp_path,
        ),
        initial_state=(64, 67),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        episode_index=0,
        generator=torch.Generator().manual_seed(0),
    )

    assert rank_1_paths.checkpoint_latest_path.read_bytes() == parent_checkpoint_bytes
    assert load_latest_checkpoint(rank_1_paths) == parent_checkpoint_payload


def test_train_rank2_episode_with_artifacts_rejects_missing_accepted_parent(
    tmp_path: Path,
) -> None:
    child_policy = TinyRank2Policy()

    with pytest.raises(ValueError, match="accepted parent checkpoint is missing"):
        train_rank2_episode_with_artifacts(
            parent_policy=TinyRank1Policy(),
            child_policy=child_policy,
            child_optimizer=torch.optim.SGD(child_policy.parameters(), lr=0.1),
            config=TowerRankConfig(
                rank=2,
                lineage_id="lineage-a",
                episode_budget=1,
                max_step_size=1,
                parent_checkpoint="rank_1/checkpoint_latest.pt",
                parent_sampler_config={"top_m": 1},
                training_config={"max_steps": 1},
            ),
            paths=TowerArtifactPaths(
                lineage_id="lineage-a",
                rank=2,
                artifact_root=tmp_path,
            ),
            initial_state=(64, 67),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
            episode_index=0,
        )


def test_train_rank2_episode_with_artifacts_rejects_parent_config_mismatch(
    tmp_path: Path,
) -> None:
    prepare_accepted_rank1_parent(tmp_path=tmp_path)
    child_policy = TinyRank2Policy()

    with pytest.raises(ValueError, match="config parent_checkpoint must match"):
        train_rank2_episode_with_artifacts(
            parent_policy=TinyRank1Policy(),
            child_policy=child_policy,
            child_optimizer=torch.optim.SGD(child_policy.parameters(), lr=0.1),
            config=TowerRankConfig(
                rank=2,
                lineage_id="lineage-a",
                episode_budget=1,
                max_step_size=1,
                parent_checkpoint="rank_0/checkpoint_latest.pt",
                parent_sampler_config={"top_m": 1},
                training_config={"max_steps": 1},
            ),
            paths=TowerArtifactPaths(
                lineage_id="lineage-a",
                rank=2,
                artifact_root=tmp_path,
            ),
            initial_state=(64, 67),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
            episode_index=0,
        )
