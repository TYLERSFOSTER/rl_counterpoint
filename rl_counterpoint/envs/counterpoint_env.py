"""Minimal Gymnasium-style counterpoint environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from rl_counterpoint.envs.observation import (
    bar_position,
    build_observation,
    build_timed_chord_window,
    is_downbeat,
    is_ending_beat,
    is_leading_beat,
)
from rl_counterpoint.envs.termination import is_max_step_truncated
from rl_counterpoint.graph.actions import (
    StepDelta,
    is_valid_step_delta_action,
    step_delta_action_mask,
    step_delta_action_space,
    step_delta_to_next_state,
)
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.state_space import ChordState, is_valid_node, iter_node_states
from rl_counterpoint.reward.protocol import RewardContext, RewardFn


Info = dict[str, Any]
TARGET_ROOT_OCTAVE_CHOICES = (2, 3, 4, 5, 6)


@dataclass
class CounterpointEnv:
    """Small explicit environment binding graph actions to a reward protocol."""

    graph_spec: CounterpointGraphSpec
    reward_fn: RewardFn
    initial_state: ChordState | None
    max_steps: int
    measure_size: int
    max_step_size: int
    invalid_action_penalty: float = -1.0
    _state: ChordState = field(init=False, repr=False)
    _step_index: int = field(init=False, default=0, repr=False)
    _history: tuple[ChordState, ...] = field(init=False, repr=False)
    _action_space: tuple[StepDelta, ...] = field(init=False, repr=False)
    _reset_states: tuple[ChordState, ...] = field(init=False, repr=False)
    _target_root_octave: int = field(init=False, default=4, repr=False)
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if self.measure_size < 1:
            raise ValueError("measure_size must be at least 1")

        self._action_space = step_delta_action_space(
            n=self.graph_spec.n,
            max_step_size=self.max_step_size,
        )
        self._reset_states = tuple(
            state
            for state in iter_node_states(self.graph_spec)
            if any(self._action_mask(state))
        )
        if not self._reset_states:
            raise ValueError("graph must contain at least one reset state with a legal StepDelta")

        if self.initial_state is None:
            self.initial_state = self._reset_states[0]
        elif not is_valid_node(self.initial_state, self.graph_spec):
            raise ValueError("initial_state must be a valid graph node")
        elif not any(self._action_mask(self.initial_state)):
            raise ValueError("initial_state must have at least one legal StepDelta")

        self._rng = Random(0)
        self._state = self.initial_state
        self._step_index = 0
        self._history = (self.initial_state,)
        self._target_root_octave = TARGET_ROOT_OCTAVE_CHOICES[0]

    def reset(self, seed: int | None = None) -> tuple[ChordState, Info]:
        """Reset to a sampled valid start chord and episode target octave."""
        if seed is not None:
            self._rng.seed(seed)

        self._state = self._rng.choice(self._reset_states)
        self._step_index = 0
        self._history = (self._state,)
        self._target_root_octave = self._rng.choice(TARGET_ROOT_OCTAVE_CHOICES)

        return build_observation(self._state), self._info_for_state(self._state)

    def step(
        self,
        step_delta: StepDelta,
    ) -> tuple[ChordState, float, bool, bool, Info]:
        """Advance one Gymnasium-style step using a StepDelta action."""
        source = self._state
        target = step_delta_to_next_state(source, step_delta)
        valid_action = is_valid_step_delta_action(source, step_delta, self.graph_spec)

        if not valid_action:
            self._step_index += 1
            self._history = (*self._history, source)
            truncated = is_max_step_truncated(
                step_index=self._step_index,
                max_steps=self.max_steps,
            )
            info = self._info_for_state(
                source,
                source=source,
                target=target,
                step_delta=step_delta,
                valid_action=False,
                invalid_action_reason="decoded target is not a valid edge",
            )
            return (
                build_observation(source),
                self.invalid_action_penalty,
                False,
                truncated,
                info,
            )

        reward_result = self.reward_fn(
            source,
            target,
            RewardContext(
                step_index=self._step_index,
                max_steps=self.max_steps,
                measure_size=self.measure_size,
                history=self._history,
                step_delta=step_delta,
                key_pitch_class=self.graph_spec.tonic_pitch_class,
                target_root_octave=self._target_root_octave,
                is_final_step=(self._step_index + 1) >= self.max_steps,
                timed_chord_window=build_timed_chord_window(
                    history=self._history,
                    step_index=self._step_index,
                    measure_size=self.measure_size,
                ),
            ),
        )

        self._state = target
        self._step_index += 1
        self._history = (*self._history, target)
        truncated = is_max_step_truncated(
            step_index=self._step_index,
            max_steps=self.max_steps,
        )
        info = self._info_for_state(
            self._state,
            source=source,
            target=target,
            step_delta=step_delta,
            valid_action=True,
            reward_diagnostics=reward_result.diagnostics,
            hard_violation=reward_result.hard_violation,
            is_terminal_success=reward_result.is_terminal_success,
        )

        return (
            build_observation(self._state),
            reward_result.reward,
            reward_result.is_terminal_success,
            truncated,
            info,
        )

    @property
    def state(self) -> ChordState:
        return self._state

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def history(self) -> tuple[ChordState, ...]:
        return self._history

    @property
    def action_space(self) -> tuple[StepDelta, ...]:
        return self._action_space

    @property
    def target_root_octave(self) -> int:
        return self._target_root_octave

    def _action_mask(self, state: ChordState) -> tuple[bool, ...]:
        return step_delta_action_mask(state, self._action_space, self.graph_spec)

    def _info_for_state(self, state: ChordState, **extra: Any) -> Info:
        action_mask = self._action_mask(state)
        return {
            "state": state,
            "step_index": self._step_index,
            "measure_size": self.measure_size,
            "bar_position": bar_position(
                step_index=self._step_index,
                measure_size=self.measure_size,
            ),
            "is_leading_beat": is_leading_beat(
                step_index=self._step_index,
                measure_size=self.measure_size,
            ),
            "is_downbeat": is_downbeat(step_index=self._step_index),
            "is_ending_beat": is_ending_beat(
                step_index=self._step_index,
                measure_size=self.measure_size,
            ),
            "history": self._history,
            "target_root_octave": self._target_root_octave,
            "action_space": self._action_space,
            "action_mask": action_mask,
            "has_legal_actions": any(action_mask),
            **extra,
        }
