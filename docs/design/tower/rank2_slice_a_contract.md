# Rank-2 Slice A Contract

This document defines the first real rank-2 reward-and-training slice for
`tower`.

The goal of this slice is not to introduce the full later contrapuntal reward
grammar. The goal is to bring rank-2 into the same operational state that rank-1
reached in its earlier reward slice: a complete, trainable, artifact-backed
pipeline with a narrow but serious reward bundle.

## Scope

Rank-2 Slice A includes exactly three reward families:

1. goal-octave attainment for the new outer voice
2. lifted terminal cadence reward
3. vertical consonance reward over all realized rank-2 vertical intervals

This slice must be implemented at full engineering quality:

- proper reward factory/config validation
- proper diagnostics
- proper rank-2 training script
- proper artifact outputs
- proper tests

What is intentionally deferred:

- beat-class-sensitive vertical exceptions
- dissonance treatment by metrical class
- parallel perfect penalties
- contrary/oblique motion preferences
- spacing/crossing style refinements beyond existing legality
- richer cadence sonority shaping beyond lifted success

## Reward Bundle

The Slice A rank-2 reward bundle is:

1. `rank2_lifted_cadence_success`
2. `rank2_target_octave_distance`
3. `rank2_vertical_consonance`

### 1. Lifted Cadence Success

Terminal success is the existing lifted rank-2 cadence condition:

- projected rank-1 cadence success must hold
- the new outer voice must satisfy the rank-2 outer-third terminal condition

This term must be implemented as a success/failure term in the same general style
as rank-1 terminal cadence reward.

Required config fields:

- `terminal_cadence_reward: float`
- `cadence_failure_reward: float`

### 2. Goal Octave Attainment

The child voice must receive dense shaping toward its target root octave.

The intended semantics match rank-1 target-octave shaping, but now the scored
voice is the new outer voice introduced at rank 2.

Required behavior:

- use the realized rank-2 target state
- isolate the new outer voice pitch
- compare its root octave to `target_root_octave`
- reward should increase as octave distance decreases

Required config fields:

- `target_root_octave: int`
- `use_context_target_root_octave: bool`

Initial formula:

- `1 / (1 + d)` where `d` is outer-voice root-octave distance from target

### 3. Vertical Consonance

The child voice must be rewarded for consonant realized vertical intervals
against every already-present lower voice in the realized rank-2 target state.

For rank 2 specifically, this means:

- the new outer voice forms one vertical interval with the projected rank-1 voice

But the contract should be phrased so the reward term evaluates all realized
vertical intervals involving the new voice, not a rank-2-only special case in its
meaning.

Required behavior:

- score the realized target state, not the source state
- compute interval classes between the new outer voice and each existing lower
  voice
- reward consonant intervals
- penalize non-consonant intervals

Initial consonance vocabulary should match the project’s current consonance
convention used elsewhere unless explicitly overridden later.

Required config fields:

- `vertical_consonance_reward: float`
- `vertical_non_consonance_penalty: float`

## Factory Contract

Add a rank-2 reward factory alongside the existing rank-1 factory.

Expected public entrypoint:

- `build_rank2_reward_fn(...)`

Expected config type:

- `Rank2RewardFactoryConfig`

Expected callable type:

- `Rank2RewardFunction`

Diagnostics top-level payload should identify:

- reward kind
- key pitch class
- target root octave

Term-level diagnostics should expose:

- lifted cadence result and failure reason
- outer-voice target-octave distance
- realized vertical interval classes
- consonant vs non-consonant vertical judgments

## Training Script Contract

Add a real rank-2 training script.

This script must:

- accept an accepted rank-1 parent checkpoint dependency
- construct rank-2 runner config
- build the rank-2 reward function from Slice A config
- run artifact-backed rank-2 training
- write final MIDI examples and metrics in the same style as rank-1 scripts

Minimum required inputs:

- `--rank 2` support or a dedicated rank-2 script
- parent checkpoint path or accepted-parent lineage reference
- `parent_top_m`
- reward config fields for this slice
- policy/training hyperparameters

## Acceptance Criteria

Slice A is complete when all of the following hold:

1. `build_rank2_reward_fn(...)` exists and is tested
2. rank-2 reward tests cover:
   - lifted success reward
   - outer-voice target-octave shaping
   - consonant vs non-consonant vertical intervals
3. rank-2 training script exists and runs by file path
4. a rank-2 smoke train can complete over one accepted rank-1 parent checkpoint
5. artifact outputs include:
   - config
   - metrics
   - checkpoint
   - final inference rows
   - final MIDI examples

## Out Of Scope Reminder

This slice is intentionally not the full rank-2 musical grammar.

It is the first complete rank-2 reward slice with:

- terminal goal
- dense goal shaping
- vertical consonance shaping

Later slices can layer in the more stylistic contrapuntal rules once the rank-2
pipeline is producing stable evidence.
