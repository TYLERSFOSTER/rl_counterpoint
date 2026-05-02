"""Reward factory helpers for tower training."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Real

from tower.reward.context import TowerRewardContext
from tower.reward.harmony import (
    Rank2BeatClassVerticalReward,
    Rank2CadenceEndpointReward,
    Rank2SpacingControlReward,
    Rank2TargetVerticalIntervalReward,
    Rank2VerticalConsonanceReward,
    Rank3BeatClassTriadReward,
    Rank3CadenceEndpointTriadReward,
    Rank3GlobalSpacingReward,
    Rank3GlobalTriadConsonanceReward,
)
from tower.reward.melody import (
    BeatClassPitchReward,
    GoalOctaveDirectionReward,
    LargeLeapRecoveryTerm,
    RecentMelodicRangePenalty,
    StepSizeBinBalanceReward,
    TargetOctaveDistanceReward,
)
from tower.reward.result import TowerRewardResult
from tower.reward.success import (
    rank1_projected_cadence_success,
    rank2_lifted_cadence_success,
    rank3_triadic_cadence_success,
)
from tower.reward.terms import CompositeRewardTerm, RewardTerm, SuccessRewardTerm


@dataclass(frozen=True)
class Rank1RewardFactoryConfig:
    """Configuration for the first rank-1 musical reward bundle."""

    key_pitch_class: int = 0
    terminal_cadence_reward: float = 10.0
    cadence_failure_reward: float = 0.0
    target_root_octave: int = 4
    use_context_target_root_octave: bool = False
    goal_octave_direction_weight: float = 0.5
    max_recent_range: int = 12
    range_penalty: float = -1.0
    large_leap_threshold: int = 6
    recovery_step_threshold: int = 3
    recovery_reward: float = 0.5
    failure_penalty: float = -0.5
    measure_start_tonic_reward: float = 1.0
    onbeat_scale_degree_reward: float = 1.0
    offbeat_consonance_weight: float = 1.0
    onbeat_non_scale_penalty: float = -2.0
    offbeat_non_consonance_penalty: float = -2.0
    step_size_balance_threshold: int = 3
    step_size_balance_target_small_rate: float = 0.3
    step_size_balance_weight: float = 1.0

    def __post_init__(self) -> None:
        _validate_pitch_class(self.key_pitch_class)
        _validate_number(
            self.terminal_cadence_reward,
            field_name="terminal_cadence_reward",
        )
        _validate_number(
            self.cadence_failure_reward,
            field_name="cadence_failure_reward",
        )
        _validate_target_octave(self.target_root_octave)
        if not isinstance(self.use_context_target_root_octave, bool):
            raise TypeError("use_context_target_root_octave must be a bool")
        _validate_number(
            self.goal_octave_direction_weight,
            field_name="goal_octave_direction_weight",
        )
        _validate_number(
            self.measure_start_tonic_reward,
            field_name="measure_start_tonic_reward",
        )
        _validate_number(
            self.onbeat_scale_degree_reward,
            field_name="onbeat_scale_degree_reward",
        )
        _validate_number(
            self.offbeat_consonance_weight,
            field_name="offbeat_consonance_weight",
        )
        _validate_number(
            self.onbeat_non_scale_penalty,
            field_name="onbeat_non_scale_penalty",
        )
        _validate_number(
            self.offbeat_non_consonance_penalty,
            field_name="offbeat_non_consonance_penalty",
        )
        _validate_positive_int(
            self.step_size_balance_threshold,
            field_name="step_size_balance_threshold",
        )
        _validate_rate(
            self.step_size_balance_target_small_rate,
            field_name="step_size_balance_target_small_rate",
        )
        _validate_number(
            self.step_size_balance_weight,
            field_name="step_size_balance_weight",
        )


@dataclass(frozen=True)
class Rank1RewardFunction:
    """Callable rank-1 reward bundle built from composable reward terms."""

    config: Rank1RewardFactoryConfig
    term: RewardTerm = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "term",
            CompositeRewardTerm(
                terms=(
                    SuccessRewardTerm(
                        predicate=rank1_projected_cadence_success,
                        success_reward=float(self.config.terminal_cadence_reward),
                        failure_reward=float(self.config.cadence_failure_reward),
                        diagnostics_key="cadence",
                    ),
                    RecentMelodicRangePenalty(
                        max_recent_range=self.config.max_recent_range,
                        penalty=float(self.config.range_penalty),
                    ),
                    LargeLeapRecoveryTerm(
                        large_leap_threshold=self.config.large_leap_threshold,
                        recovery_step_threshold=self.config.recovery_step_threshold,
                        recovery_reward=float(self.config.recovery_reward),
                        failure_penalty=float(self.config.failure_penalty),
                    ),
                    TargetOctaveDistanceReward(),
                    GoalOctaveDirectionReward(
                        weight=float(self.config.goal_octave_direction_weight),
                    ),
                    BeatClassPitchReward(
                        measure_start_tonic_reward=(
                            self.config.measure_start_tonic_reward
                        ),
                        onbeat_scale_degree_reward=(
                            self.config.onbeat_scale_degree_reward
                        ),
                        offbeat_consonance_weight=(
                            self.config.offbeat_consonance_weight
                        ),
                        onbeat_non_scale_penalty=(
                            self.config.onbeat_non_scale_penalty
                        ),
                        offbeat_non_consonance_penalty=(
                            self.config.offbeat_non_consonance_penalty
                        ),
                    ),
                    StepSizeBinBalanceReward(
                        small_step_threshold=self.config.step_size_balance_threshold,
                        target_small_rate=(
                            self.config.step_size_balance_target_small_rate
                        ),
                        weight=float(self.config.step_size_balance_weight),
                    ),
                ),
                diagnostics={
                    "kind": "rank1_reward",
                    "key_pitch_class": self.config.key_pitch_class,
                    "target_root_octave": self.config.target_root_octave,
                },
            ),
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        if not isinstance(context, TowerRewardContext):
            raise TypeError("context must be a TowerRewardContext")
        if context.rank != 1:
            raise ValueError("rank-1 reward function requires rank 1 context")

        target_root_octave = (
            context.target_root_octave
            if self.config.use_context_target_root_octave
            else self.config.target_root_octave
        )
        if target_root_octave is None:
            raise ValueError("target_root_octave is required for rank-1 reward")

        keyed_context = replace(
            context,
            key_pitch_class=self.config.key_pitch_class,
            target_root_octave=target_root_octave,
        )
        return self.term(keyed_context)


@dataclass(frozen=True)
class Rank2RewardFactoryConfig:
    """Configuration for the first narrow rank-2 reward bundle."""

    key_pitch_class: int = 0
    terminal_cadence_reward: float = 10.0
    cadence_failure_reward: float = 0.0
    cadence_endpoint_weight: float = 1.0
    target_root_octave: int = 4
    use_context_target_root_octave: bool = False
    vertical_consonance_weight: float = 1.0
    vertical_non_consonance_penalty: float = 0.0
    upper_register_soft_ceiling: int = 80
    upper_register_penalty_weight: float = 0.05
    min_vertical_gap: int = 3
    spacing_reward: float = 0.1
    spacing_penalty: float = -0.1
    target_vertical_interval: int = 4
    target_vertical_interval_weight: float = 1.0
    onbeat_scale_degree_interval_reward: float = 1.0
    onbeat_non_scale_degree_interval_penalty: float = 0.0
    offbeat_vertical_consonance_weight: float = 0.0
    offbeat_vertical_non_consonance_penalty: float = -2.0

    def __post_init__(self) -> None:
        _validate_pitch_class(self.key_pitch_class)
        _validate_number(
            self.terminal_cadence_reward,
            field_name="terminal_cadence_reward",
        )
        _validate_number(
            self.cadence_failure_reward,
            field_name="cadence_failure_reward",
        )
        _validate_number(
            self.cadence_endpoint_weight,
            field_name="cadence_endpoint_weight",
        )
        _validate_target_octave(self.target_root_octave)
        if not isinstance(self.use_context_target_root_octave, bool):
            raise TypeError("use_context_target_root_octave must be a bool")
        _validate_number(
            self.vertical_consonance_weight,
            field_name="vertical_consonance_weight",
        )
        _validate_number(
            self.vertical_non_consonance_penalty,
            field_name="vertical_non_consonance_penalty",
        )
        _validate_pitch_bound(
            self.upper_register_soft_ceiling,
            field_name="upper_register_soft_ceiling",
        )
        _validate_positive_int(
            self.min_vertical_gap,
            field_name="min_vertical_gap",
        )
        _validate_number(
            self.upper_register_penalty_weight,
            field_name="upper_register_penalty_weight",
        )
        _validate_number(
            self.spacing_reward,
            field_name="spacing_reward",
        )
        _validate_number(
            self.spacing_penalty,
            field_name="spacing_penalty",
        )
        _validate_positive_int_or_zero(
            self.target_vertical_interval,
            field_name="target_vertical_interval",
        )
        _validate_number(
            self.target_vertical_interval_weight,
            field_name="target_vertical_interval_weight",
        )
        _validate_number(
            self.onbeat_scale_degree_interval_reward,
            field_name="onbeat_scale_degree_interval_reward",
        )
        _validate_number(
            self.onbeat_non_scale_degree_interval_penalty,
            field_name="onbeat_non_scale_degree_interval_penalty",
        )
        _validate_number(
            self.offbeat_vertical_consonance_weight,
            field_name="offbeat_vertical_consonance_weight",
        )
        _validate_number(
            self.offbeat_vertical_non_consonance_penalty,
            field_name="offbeat_vertical_non_consonance_penalty",
        )


@dataclass(frozen=True)
class Rank2RewardFunction:
    """Callable rank-2 reward bundle built from composable reward terms."""

    config: Rank2RewardFactoryConfig
    term: RewardTerm = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "term",
            CompositeRewardTerm(
                terms=(
                    SuccessRewardTerm(
                        predicate=rank2_lifted_cadence_success,
                        success_reward=float(self.config.terminal_cadence_reward),
                        failure_reward=float(self.config.cadence_failure_reward),
                        diagnostics_key="cadence",
                    ),
                    Rank2CadenceEndpointReward(
                        weight=float(self.config.cadence_endpoint_weight),
                    ),
                    Rank2BeatClassVerticalReward(
                        onbeat_scale_degree_interval_reward=float(
                            self.config.onbeat_scale_degree_interval_reward
                        ),
                        onbeat_non_scale_degree_interval_penalty=float(
                            self.config.onbeat_non_scale_degree_interval_penalty
                        ),
                        offbeat_consonance_weight=float(
                            self.config.offbeat_vertical_consonance_weight
                        ),
                        offbeat_non_consonance_penalty=float(
                            self.config.offbeat_vertical_non_consonance_penalty
                        ),
                    ),
                    Rank2VerticalConsonanceReward(
                        consonance_weight=float(
                            self.config.vertical_consonance_weight
                        ),
                        non_consonance_penalty=float(
                            self.config.vertical_non_consonance_penalty
                        ),
                    ),
                    Rank2SpacingControlReward(
                        upper_register_soft_ceiling=int(
                            self.config.upper_register_soft_ceiling
                        ),
                        upper_register_penalty_weight=float(
                            self.config.upper_register_penalty_weight
                        ),
                        min_vertical_gap=int(self.config.min_vertical_gap),
                        spacing_reward=float(self.config.spacing_reward),
                        spacing_penalty=float(self.config.spacing_penalty),
                    ),
                    Rank2TargetVerticalIntervalReward(
                        target_vertical_interval=int(
                            self.config.target_vertical_interval
                        ),
                        interval_reward_weight=float(
                            self.config.target_vertical_interval_weight
                        ),
                    ),
                ),
                diagnostics={
                    "kind": "rank2_reward",
                    "key_pitch_class": self.config.key_pitch_class,
                },
            ),
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        if not isinstance(context, TowerRewardContext):
            raise TypeError("context must be a TowerRewardContext")
        if context.rank != 2:
            raise ValueError("rank-2 reward function requires rank 2 context")

        keyed_context = replace(
            context,
            key_pitch_class=self.config.key_pitch_class,
        )
        return self.term(keyed_context)


@dataclass(frozen=True)
class Rank3RewardFactoryConfig:
    """Configuration for the first concrete rank-3 reward bundle."""

    key_pitch_class: int = 0
    terminal_cadence_reward: float = 10.0
    cadence_failure_reward: float = 0.0
    target_root_octave: int = 4
    use_context_target_root_octave: bool = False
    triad_consonance_weight: float = 1.0
    triad_non_consonance_penalty: float = 0.0
    min_adjacent_gap: int = 3
    max_outer_span: int = 15
    adjacent_spacing_reward: float = 0.1
    adjacent_spacing_penalty: float = -0.1
    outer_span_reward: float = 0.1
    outer_span_penalty: float = -0.1
    cadence_endpoint_weight: float = 1.0
    onbeat_all_scale_degree_reward: float = 1.0
    onbeat_not_all_scale_degree_penalty: float = 0.0
    offbeat_all_consonant_weight: float = 0.0
    offbeat_non_consonance_penalty: float = -2.0

    def __post_init__(self) -> None:
        _validate_pitch_class(self.key_pitch_class)
        _validate_number(
            self.terminal_cadence_reward,
            field_name="terminal_cadence_reward",
        )
        _validate_number(
            self.cadence_failure_reward,
            field_name="cadence_failure_reward",
        )
        _validate_target_octave(self.target_root_octave)
        if not isinstance(self.use_context_target_root_octave, bool):
            raise TypeError("use_context_target_root_octave must be a bool")
        _validate_number(
            self.triad_consonance_weight,
            field_name="triad_consonance_weight",
        )
        _validate_number(
            self.triad_non_consonance_penalty,
            field_name="triad_non_consonance_penalty",
        )
        _validate_positive_int(self.min_adjacent_gap, field_name="min_adjacent_gap")
        _validate_positive_int(self.max_outer_span, field_name="max_outer_span")
        _validate_number(
            self.adjacent_spacing_reward,
            field_name="adjacent_spacing_reward",
        )
        _validate_number(
            self.adjacent_spacing_penalty,
            field_name="adjacent_spacing_penalty",
        )
        _validate_number(
            self.outer_span_reward,
            field_name="outer_span_reward",
        )
        _validate_number(
            self.outer_span_penalty,
            field_name="outer_span_penalty",
        )
        _validate_number(
            self.cadence_endpoint_weight,
            field_name="cadence_endpoint_weight",
        )
        _validate_number(
            self.onbeat_all_scale_degree_reward,
            field_name="onbeat_all_scale_degree_reward",
        )
        _validate_number(
            self.onbeat_not_all_scale_degree_penalty,
            field_name="onbeat_not_all_scale_degree_penalty",
        )
        _validate_number(
            self.offbeat_all_consonant_weight,
            field_name="offbeat_all_consonant_weight",
        )
        _validate_number(
            self.offbeat_non_consonance_penalty,
            field_name="offbeat_non_consonance_penalty",
        )


@dataclass(frozen=True)
class Rank3RewardFunction:
    """Callable rank-3 reward bundle built from composable reward terms."""

    config: Rank3RewardFactoryConfig
    term: RewardTerm = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "term",
            CompositeRewardTerm(
                terms=(
                    SuccessRewardTerm(
                        predicate=rank3_triadic_cadence_success,
                        success_reward=float(self.config.terminal_cadence_reward),
                        failure_reward=float(self.config.cadence_failure_reward),
                        diagnostics_key="cadence",
                    ),
                    Rank3GlobalTriadConsonanceReward(
                        consonance_weight=float(self.config.triad_consonance_weight),
                        non_consonance_penalty=float(
                            self.config.triad_non_consonance_penalty
                        ),
                    ),
                    Rank3BeatClassTriadReward(
                        onbeat_all_scale_degree_reward=float(
                            self.config.onbeat_all_scale_degree_reward
                        ),
                        onbeat_not_all_scale_degree_penalty=float(
                            self.config.onbeat_not_all_scale_degree_penalty
                        ),
                        offbeat_all_consonant_weight=float(
                            self.config.offbeat_all_consonant_weight
                        ),
                        offbeat_non_consonance_penalty=float(
                            self.config.offbeat_non_consonance_penalty
                        ),
                    ),
                    Rank3GlobalSpacingReward(
                        min_adjacent_gap=int(self.config.min_adjacent_gap),
                        max_outer_span=int(self.config.max_outer_span),
                        adjacent_spacing_reward=float(
                            self.config.adjacent_spacing_reward
                        ),
                        adjacent_spacing_penalty=float(
                            self.config.adjacent_spacing_penalty
                        ),
                        outer_span_reward=float(self.config.outer_span_reward),
                        outer_span_penalty=float(self.config.outer_span_penalty),
                    ),
                    Rank3CadenceEndpointTriadReward(
                        weight=float(self.config.cadence_endpoint_weight),
                    ),
                ),
                diagnostics={
                    "kind": "rank3_reward",
                    "key_pitch_class": self.config.key_pitch_class,
                    "target_root_octave": self.config.target_root_octave,
                },
            ),
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        if not isinstance(context, TowerRewardContext):
            raise TypeError("context must be a TowerRewardContext")
        if context.rank != 3:
            raise ValueError("rank-3 reward function requires rank 3 context")

        target_root_octave = (
            context.target_root_octave
            if self.config.use_context_target_root_octave
            else self.config.target_root_octave
        )
        if target_root_octave is None:
            raise ValueError("target_root_octave is required for rank-3 reward")

        keyed_context = replace(
            context,
            key_pitch_class=self.config.key_pitch_class,
            target_root_octave=target_root_octave,
        )
        return self.term(keyed_context)


def build_rank1_reward_fn(
    *,
    key_pitch_class: int = 0,
    terminal_cadence_reward: float = 10.0,
    cadence_failure_reward: float = 0.0,
    target_root_octave: int = 4,
    use_context_target_root_octave: bool = False,
    goal_octave_direction_weight: float = 0.5,
    max_recent_range: int = 12,
    range_penalty: float = -1.0,
    large_leap_threshold: int = 6,
    recovery_step_threshold: int = 3,
    recovery_reward: float = 0.5,
    failure_penalty: float = -0.5,
    measure_start_tonic_reward: float = 1.0,
    onbeat_scale_degree_reward: float = 1.0,
    offbeat_consonance_weight: float = 1.0,
    onbeat_non_scale_penalty: float = -2.0,
    offbeat_non_consonance_penalty: float = -2.0,
    step_size_balance_threshold: int = 3,
    step_size_balance_target_small_rate: float = 0.3,
    step_size_balance_weight: float = 1.0,
) -> Rank1RewardFunction:
    """Build the first rank-1 cadence and melodic-shape reward function."""
    return Rank1RewardFunction(
        config=Rank1RewardFactoryConfig(
            key_pitch_class=key_pitch_class,
            terminal_cadence_reward=terminal_cadence_reward,
            cadence_failure_reward=cadence_failure_reward,
            target_root_octave=target_root_octave,
            use_context_target_root_octave=use_context_target_root_octave,
            goal_octave_direction_weight=goal_octave_direction_weight,
            max_recent_range=max_recent_range,
            range_penalty=range_penalty,
            large_leap_threshold=large_leap_threshold,
            recovery_step_threshold=recovery_step_threshold,
            recovery_reward=recovery_reward,
            failure_penalty=failure_penalty,
            measure_start_tonic_reward=measure_start_tonic_reward,
            onbeat_scale_degree_reward=onbeat_scale_degree_reward,
            offbeat_consonance_weight=offbeat_consonance_weight,
            onbeat_non_scale_penalty=onbeat_non_scale_penalty,
            offbeat_non_consonance_penalty=offbeat_non_consonance_penalty,
            step_size_balance_threshold=step_size_balance_threshold,
            step_size_balance_target_small_rate=step_size_balance_target_small_rate,
            step_size_balance_weight=step_size_balance_weight,
        )
    )


def build_rank2_reward_fn(
    *,
    key_pitch_class: int = 0,
    terminal_cadence_reward: float = 10.0,
    cadence_failure_reward: float = 0.0,
    cadence_endpoint_weight: float = 1.0,
    target_root_octave: int = 4,
    use_context_target_root_octave: bool = False,
    vertical_consonance_weight: float = 1.0,
    vertical_non_consonance_penalty: float = 0.0,
    upper_register_soft_ceiling: int = 80,
    upper_register_penalty_weight: float = 0.05,
    min_vertical_gap: int = 3,
    spacing_reward: float = 0.1,
    spacing_penalty: float = -0.1,
    target_vertical_interval: int = 4,
    target_vertical_interval_weight: float = 1.0,
    onbeat_scale_degree_interval_reward: float = 1.0,
    onbeat_non_scale_degree_interval_penalty: float = 0.0,
    offbeat_vertical_consonance_weight: float = 0.0,
    offbeat_vertical_non_consonance_penalty: float = -2.0,
) -> Rank2RewardFunction:
    """Build the first narrow rank-2 reward function."""
    return Rank2RewardFunction(
        config=Rank2RewardFactoryConfig(
            key_pitch_class=key_pitch_class,
            terminal_cadence_reward=terminal_cadence_reward,
            cadence_failure_reward=cadence_failure_reward,
            cadence_endpoint_weight=cadence_endpoint_weight,
            target_root_octave=target_root_octave,
            use_context_target_root_octave=use_context_target_root_octave,
            vertical_consonance_weight=vertical_consonance_weight,
            vertical_non_consonance_penalty=vertical_non_consonance_penalty,
            upper_register_soft_ceiling=upper_register_soft_ceiling,
            upper_register_penalty_weight=upper_register_penalty_weight,
            min_vertical_gap=min_vertical_gap,
            spacing_reward=spacing_reward,
            spacing_penalty=spacing_penalty,
            target_vertical_interval=target_vertical_interval,
            target_vertical_interval_weight=target_vertical_interval_weight,
            onbeat_scale_degree_interval_reward=onbeat_scale_degree_interval_reward,
            onbeat_non_scale_degree_interval_penalty=(
                onbeat_non_scale_degree_interval_penalty
            ),
            offbeat_vertical_consonance_weight=(
                offbeat_vertical_consonance_weight
            ),
            offbeat_vertical_non_consonance_penalty=(
                offbeat_vertical_non_consonance_penalty
            ),
        )
    )


def build_rank3_reward_fn(
    *,
    key_pitch_class: int = 0,
    terminal_cadence_reward: float = 10.0,
    cadence_failure_reward: float = 0.0,
    target_root_octave: int = 4,
    use_context_target_root_octave: bool = False,
    triad_consonance_weight: float = 1.0,
    triad_non_consonance_penalty: float = 0.0,
    min_adjacent_gap: int = 3,
    max_outer_span: int = 15,
    adjacent_spacing_reward: float = 0.1,
    adjacent_spacing_penalty: float = -0.1,
    outer_span_reward: float = 0.1,
    outer_span_penalty: float = -0.1,
    cadence_endpoint_weight: float = 1.0,
    onbeat_all_scale_degree_reward: float = 1.0,
    onbeat_not_all_scale_degree_penalty: float = 0.0,
    offbeat_all_consonant_weight: float = 0.0,
    offbeat_non_consonance_penalty: float = -2.0,
) -> Rank3RewardFunction:
    """Build the first concrete rank-3 reward function."""
    return Rank3RewardFunction(
        config=Rank3RewardFactoryConfig(
            key_pitch_class=key_pitch_class,
            terminal_cadence_reward=terminal_cadence_reward,
            cadence_failure_reward=cadence_failure_reward,
            target_root_octave=target_root_octave,
            use_context_target_root_octave=use_context_target_root_octave,
            triad_consonance_weight=triad_consonance_weight,
            triad_non_consonance_penalty=triad_non_consonance_penalty,
            min_adjacent_gap=min_adjacent_gap,
            max_outer_span=max_outer_span,
            adjacent_spacing_reward=adjacent_spacing_reward,
            adjacent_spacing_penalty=adjacent_spacing_penalty,
            outer_span_reward=outer_span_reward,
            outer_span_penalty=outer_span_penalty,
            cadence_endpoint_weight=cadence_endpoint_weight,
            onbeat_all_scale_degree_reward=onbeat_all_scale_degree_reward,
            onbeat_not_all_scale_degree_penalty=onbeat_not_all_scale_degree_penalty,
            offbeat_all_consonant_weight=offbeat_all_consonant_weight,
            offbeat_non_consonance_penalty=offbeat_non_consonance_penalty,
        )
    )


def _validate_pitch_class(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("key_pitch_class must be an integer")
    if value < 0 or value > 11:
        raise ValueError("key_pitch_class must be in [0, 11]")


def _validate_number(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")


def _validate_rate(value: float, *, field_name: str) -> None:
    _validate_number(value, field_name=field_name)
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _validate_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")


def _validate_positive_int_or_zero(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_target_octave(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("target_root_octave must be an integer")
    if value < -1 or value > 9:
        raise ValueError("target_root_octave must be in [-1, 9]")


def _validate_pitch_bound(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0 or value > 127:
        raise ValueError(f"{field_name} must be in [0, 127]")
