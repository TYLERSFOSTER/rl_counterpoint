# Post-Slice-8 Reward Expansion Plan

This document is the Post-Slice-8 Phase 7 / Stage 7.1 / Action 7.1.1
deliverable.

The purpose is to choose a narrow first TC21M-derived reward expansion after
the tower training path has produced end-to-end evidence.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 7: Reward Expansion Planning |
| Stage | Stage 7.1: TC21M Reward Triage |
| Action | Action 7.1.1: Produce Reward Expansion Plan |

## Source Authority

This plan is derived from:

| Source | Role |
| --- | --- |
| `assets/rules/tc21m_rules.md` | TC21M-derived rule inventory |
| `docs/design/tower/post_slice_8_training_evidence.md` | proof that runner path exists |
| `docs/design/tower/post_slice_8_build_plan.md` | accepted post-Slice-8 scope |
| `docs/design/tower/training_runner_contract.md` | runner/evidence lifecycle |
| `tower/reward/*` | current reward implementation |
| `tests/tower/reward/*` | current reward coverage |

## Current Reward Baseline

Current tower reward code is intentionally skeletal but well-shaped.

Implemented reward contracts:

| Area | Current status |
| --- | --- |
| `TowerRewardContext` | carries rank, source/target/action, window, meter, key, max-step metadata, and rank-local new facts |
| `TowerRewardResult` | carries scalar reward, hard violation, terminal success, diagnostics |
| `CompositeRewardTerm` | combines multiple structured reward terms |
| `SuccessRewardTerm` | adapts a terminal success predicate into reward |
| rank-1 cadence success | detects terminal V-I root motion |
| rank-2 lifted cadence success | requires projected parent cadence plus outer third motion |

Current tests prove:

| Area | Evidence |
| --- | --- |
| context validation | rank/window/state/action validation |
| result validation | scalar reward and diagnostics shape |
| success predicates | rank-1 and rank-2 cadence success/failure diagnostics |
| term composition | reward summing, hard violations, terminal success propagation |

## Training Evidence Implication

The first tiny training evidence run proved the runner path can train, save
artifacts, run final no-train inference, and export MIDI.

It did not prove musical quality.

Relevant limitation:

| Limitation | Planning consequence |
| --- | --- |
| reward was constant | next reward slice should add real musical signal |
| terminal success was false | cadence reward should become runner-usable |
| one-step run | first expansion should support short focused tests before long training |
| rank 1 only evidence | rank-1 reward should come first; rank-2 should be a follow-up |

## Triage Principles

Use these principles for the first reward expansion:

| Principle | Requirement |
| --- | --- |
| narrow scope | do not implement broad TC21M grammar in one jump |
| rank-local ownership | compute new rank-k facts at the active rank; avoid rescoring inherited parent facts |
| structured diagnostics | every reward term should explain why it paid or penalized |
| reward vs graph boundary | hard impossibilities belong in graph/pruning, not soft reward |
| runner evidence | terms should be usable by the existing runner without new runner architecture |
| suspensions | defer to a later style update such as `beta.1` |

## TC21M Rule Buckets

### Near-Term: First Reward Slice

These are appropriate for the first narrow implementation pass.

| Rule area | Initial implementation stance |
| --- | --- |
| terminal cadence reward | make existing rank-1 cadence predicate easy to wire into runner configs/tests |
| rank-1 melodic leap recovery | reward/penalize large leap followed by small opposite motion |
| rank-1 recent range | penalize recent melodic window range over one octave |
| rank-2 vertical consonance | reward consonant outer interval for the new active dyad |
| rank-2 parallel perfect intervals | plan boundary carefully; likely graph/pruning, not reward |

### Later: Good Candidates After First Slice

These are valuable but should wait until the first reward slice has evidence.

| Rule area | Reason to wait |
| --- | --- |
| tonic spectral density | needs distribution helpers and key-aware pitch-class utilities |
| structural beat emphasis | needs stronger beat-role/meter abstraction |
| leading-tone resolution timing | needs multi-step future/past window logic |
| vertical interval density | needs rank-k interval fact helpers |
| non-harmonic tone classification | needs chord-template evidence |
| harmonic progression templates | needs tonic-relative sonority/template detectors |

### Defer

These should be explicitly out of the first reward expansion.

| Rule area | Reason |
| --- | --- |
| suspensions | owner previously deferred to later style update such as `beta.1` |
| six-four chord logic | source says ignore for now; requires inversion-specific contexts |
| full four-voice style grammar | no runner evidence yet for high-rank training |
| automatic checkpoint promotion by musical quality | acceptance remains episode-count based for now |

## Reward Vs Graph Boundary

Some TC21M rules read like hard legality constraints rather than reward terms.

Initial boundary:

| Rule | Boundary recommendation |
| --- | --- |
| no doubled leading tone | graph/pruning candidate, not first reward term |
| avoid parallel perfect intervals | graph/pruning candidate after reward plan is accepted |
| V to IV retrogression | large penalty first, possible graph rule later |
| MIDI pitch/rank validity | already graph/core validation |
| large melodic range | reward penalty, not graph rule |
| leap recovery | reward shaping |
| cadence success | terminal reward/success predicate |

The first implementation slice should not add graph pruning unless the owner
explicitly approves a graph-boundary action.

## Recommended First Implementation Slice

Recommended next build slice:

```text
Reward Expansion Slice A:
Rank-1 Cadence And Melodic Shape Reward
```

Purpose:

Make rank-1 training use real, inspectable musical reward without requiring
harmonic-template machinery.

Candidate files:

| File | Work |
| --- | --- |
| `tower/reward/melody.py` | add rank-1 melodic shape reward terms |
| `tower/reward/factory.py` | build reward function from runner/rank config |
| `tests/tower/reward/test_melody.py` | focused melodic reward tests |
| `tests/tower/reward/test_factory.py` | reward config/factory tests |
| `tests/tower/train/test_runner.py` | small runner test using factory reward if needed |

### Slice A Terms

#### A1: Terminal Cadence Reward Factory

Use existing predicate:

```text
rank1_projected_cadence_success
```

Wrap with:

```text
SuccessRewardTerm(success_reward=<config>, failure_reward=<config>)
```

Required behavior:

| Behavior | Requirement |
| --- | --- |
| key required | `key_pitch_class` must be supplied through reward context/config |
| meter required | `measure_size` must be supplied |
| terminal only | pays only at final step when predicate succeeds |
| diagnostics | preserve predicate reason |

#### A2: Recent Melodic Range Penalty

TC21M source:

Avoid jumps that give the recent melody a range of more than an octave.

First implementation:

| Item | Rule |
| --- | --- |
| rank | rank 1 only |
| input | valid states in `context.window` |
| condition | `max(pitches) - min(pitches) > max_recent_range` |
| default threshold | `12` semitones |
| result | negative reward when exceeded |
| diagnostics | include observed range and threshold |

#### A3: Large Leap Recovery Reward/Penalty

TC21M source:

Follow large leaps with a small step in the opposite direction.

First implementation:

| Item | Rule |
| --- | --- |
| rank | rank 1 only |
| large leap threshold | default `6` semitones |
| recovery step threshold | default `3` semitones |
| condition | previous melodic interval large |
| positive case | current action is small and opposite direction |
| negative case | current action fails recovery |
| diagnostics | include previous interval, current action, direction check |

#### A4: Optional Step-Leap Penalty

TC21M source:

Avoid stepwise motion followed by a leap in the opposite direction.

This can be included if A2/A3 remain small.

Recommended default:

| Item | Rule |
| --- | --- |
| previous step threshold | `3` semitones |
| current leap threshold | `6` semitones |
| condition | previous motion stepwise and current motion reverses into leap |
| result | mild negative reward |

## First Slice Non-Goals

Do not include these in Slice A:

| Excluded area | Reason |
| --- | --- |
| rank-2 consonance density | should be Slice B after rank-1 reward works |
| harmonic progression templates | requires chord-template detector |
| non-harmonic tone taxonomy | requires chord-tone inference |
| graph pruning | requires separate graph-boundary approval |
| suspensions | deferred |
| six-four logic | deferred |

## Follow-Up Slice B

Recommended second reward slice:

```text
Reward Expansion Slice B:
Rank-2 Vertical Interval And Motion Reward
```

Candidate terms:

| Term | Source rule |
| --- | --- |
| new dyad consonance reward | use consonances for outer intervals |
| imperfect consonance density | favor thirds and sixths |
| parallel perfect interval detector | likely graph/pruning handoff |
| expansion/contraction variety diagnostics | prepare for later style balance |

Slice B should only score new rank-2 facts introduced by the active outer voice,
not inherited rank-1 parent facts.

## Open Questions For Owner

Before implementing Slice A, answer or accept defaults for:

| Question | Recommended default |
| --- | --- |
| key for cadence reward | `key_pitch_class=0` for first tests/runs |
| terminal cadence reward | `10.0` |
| cadence failure reward | `0.0` |
| recent range penalty | `-1.0` |
| large leap recovery success reward | `0.5` |
| large leap recovery failure penalty | `-0.5` |
| include A4 step-leap penalty now | no; defer unless Slice A feels too small |

## Recommended Next Phase.Stage.Action

Recommended next action:

```text
Post-Slice-8 Phase 7.Stage 7.2.Action 7.2.1:
Specify Reward Expansion Slice A Contract
```

Purpose:

Turn this plan into an explicit implementation contract before writing reward
code.

Expected file:

```text
docs/design/tower/reward_expansion_slice_a_contract.md
```
