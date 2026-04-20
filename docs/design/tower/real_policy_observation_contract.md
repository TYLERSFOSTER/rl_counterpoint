# Real Policy Observation Contract

This document is the Post-Slice-8 Phase 1 / Stage 1.2 / Action 1.2.1
deliverable.

The purpose is to specify the tower-local model-observation contract before
writing the real transformer-family rank policy.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 1: Freeze Next-Build Contracts |
| Stage | Stage 1.2: Observation Contract Plan |
| Action | Action 1.2.1: Specify Tower Encoded Window Contract |

Action 1.2.1 exit criterion:

| Requirement | Status |
| --- | --- |
| source observation pattern recorded | drafted here |
| tower window input recorded | drafted here |
| padding/mask contract recorded | drafted here |
| meter/key/target context recorded | drafted here |
| rank-width config rule recorded | drafted here |
| output-logit boundary recorded | drafted here |

## Source Authority

This contract is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/post_slice_8_build_plan.md` | accepted post-Slice-8 decisions |
| `docs/design/tower/post_slice_8_phase_stage_action_plan.md` | current Phase.Stage.Action plan |
| `docs/design/tower/training_runner_contract.md` | runner lifecycle that will consume this observation contract |
| `tower/window.py` | current tower rank-local window implementation |
| `rl_counterpoint/envs/observation.py` | frozen old timed-window reference |
| `rl_counterpoint/models/policy.py` | frozen old transformer observation/policy reference |

The old `rl_counterpoint` files are reference only. Tower runtime code must not
import them.

## Design Principle

The graph core remains tuple-based.

The model observation is a policy-layer view of a rank-local tower window:

```text
TowerWindow + rank/context metadata
-> EncodedTowerWindow
-> transformer-family rank policy
-> active-choice logits
-> external legality/lift-fiber masking
```

Tensorization does not redefine the mathematical state/action objects.

## Old Pattern To Preserve

The old project used:

| Old pattern | Meaning |
| --- | --- |
| `TimedChordWindow` | fixed-length left-padded sequence |
| `chord_sequence` | tuple of old chord states |
| `bar_positions` | metrical position for each event |
| `valid_mask` | distinguishes real events from padding |
| symbolic event encoder | converts chord plus context to event embeddings |
| transformer encoder | reads the full timed window |
| final valid state | drives action logits |

The tower adaptation should preserve the shape of this idea while replacing old
flat chord assumptions with tower-owned rank-local state.

## Tower Input Object

The source input for the first real policy observation is:

```text
TowerWindow
```

Current fields:

| Field | Meaning |
| --- | --- |
| `states` | fixed-length tuple of rank-local `TowerState` values |
| `bar_positions` | fixed-length tuple of metrical positions |
| `valid_mask` | fixed-length tuple marking real states vs padding |

`TowerWindow` remains a graph/reward object. It should not become a tensor
object.

## Required Observation Context

The encoded policy observation must include the tower window plus explicit
context.

Required context:

| Field | Requirement |
| --- | --- |
| `rank` | positive rank; normally `len(state)` |
| `measure_size` | required for meter interpretation |
| `key_pitch_class` | required when key-aware features are enabled |
| `target_root_octave` | optional, but supported |
| `max_step_size` | optional model context, but useful for action semantics |

The first implementation may choose whether context is embedded as numeric
features, symbolic text-like events, learned embeddings, or a hybrid. The
contract requires that the information be available.

## Padding Contract

Padding follows the tower window contract:

| Item | Requirement |
| --- | --- |
| pad state | `(0,) * rank` |
| pad bar position | `PAD_BAR_POSITION` |
| valid mask | `False` for padding, `True` for real events |

The encoded observation must preserve the valid mask.

The transformer policy must use the valid mask so padded positions do not count
as real musical events.

The final hidden state used for logits should correspond to the final valid
event, not necessarily the last physical sequence position if future windows
ever allow right padding. With current left padding, this will normally be the
last sequence position.

## Meter Contract

Meter must be exposed to the model.

At minimum:

| Meter feature | Requirement |
| --- | --- |
| bar position | available for each event |
| padding position | distinguishable from real bar positions |
| measure size | available as context |

A future implementation may encode meter as:

| Encoding | Status |
| --- | --- |
| raw integer feature | allowed |
| learned embedding | allowed |
| sinusoidal/cyclical feature | allowed |
| symbolic context embedding | allowed |

Do not drop meter and leave it only to the reward function. The owner wants the
real policy to have the old timed-window style context.

## State/Event Encoding Contract

Each real event represents one rank-local tower state:

```text
s^k_t = (lambda_0, ..., lambda_{k-1})
```

Rank-specific state width is config-driven.

Required behavior:

| Behavior | Requirement |
| --- | --- |
| rank 1 | one pitch coordinate per event |
| rank 2 | two pitch coordinates per event |
| future rank k | k pitch coordinates per event |
| padding | padding states do not become real events |
| pitch range | preserve MIDI pitch values or a deterministic normalized form |

The first implementation should prefer a simple numeric tensor contract unless a
symbolic embedding is explicitly chosen in a later action.

Acceptable first numeric features include:

| Feature | Reason |
| --- | --- |
| MIDI pitch values | direct tower state representation |
| tonic-relative pitch class | key-aware pattern detection |
| octave or normalized pitch height | target/root octave learning |
| bar position | meter awareness |
| valid mask | padding awareness |

The exact feature vector belongs to the implementation action, but the required
information boundary is fixed here.

## Parent/Child Boundary

For rank-2 child training, the policy observation must not include rich parent
policy diagnostics.

Forbidden child inputs:

| Parent data | Status |
| --- | --- |
| parent action logprob | forbidden as model input |
| parent top-m candidates | forbidden as model input |
| parent distribution/logits | forbidden as model input |
| parent optimizer state | forbidden |

Allowed operational input:

| Parent data | Status |
| --- | --- |
| sampled parent action | used by rollout/graph to compute lift fiber |
| lift-fiber mask | used externally to constrain child active choices |
| projected parent history | allowed only if it is inherent in the rank-local tower state/window, not as parent policy diagnostics |

The child should learn over its rank-local window and active-choice mask. It
should not regain the old flat model's burden by consuming the full parent
policy distribution.

## Encoded Observation Object

The first implementation should introduce a tower-owned encoded observation
object, likely in:

```text
tower/policy/observation.py
```

Expected conceptual fields:

| Field | Meaning |
| --- | --- |
| `event_features` | tensor or tensor-like sequence features |
| `valid_mask` | boolean mask for padding |
| `bar_positions` | metrical positions, if not already embedded in features |
| `rank` | rank represented by this observation |
| `context` | key/measure/target metadata or encoded features |

The exact dataclass name is left to the implementation action, but
`EncodedTowerWindow` is the expected name unless a better local name is chosen.

## Output Boundary

The policy observation feeds a rank-local policy that returns:

```text
PolicyOutput(logits=...)
```

The logits correspond to the policy's rank-local action/choice vocabulary before
external masking.

Masking remains outside the model:

| Mask | Owner |
| --- | --- |
| graph legality | graph/action sampler layer |
| lift fiber | rollout/sampler layer |
| parent top-m | parent sampler layer |

This preserves the separation:

```text
policy scores possibilities
sampler masks to legal/current possibilities
rollout records selected action/logprob
```

## Shape Validation Requirements

The future implementation must validate:

| Validation | Requirement |
| --- | --- |
| sequence length | window fields share length |
| valid mask length | matches event sequence |
| bar positions length | matches event sequence |
| event rank | valid real states have configured rank |
| padded states | padding positions use pad state or are safely ignored |
| tensor rank | event tensor has expected dimensions |
| context | required context fields present when encoder needs them |

Validation errors should be explicit and local to the observation layer.

## Expected Future Implementation Files

This contract does not approve implementation, but the expected file path is:

| Area | Expected file |
| --- | --- |
| observation encoding | `tower/policy/observation.py` |
| observation tests | `tests/tower/policy/test_observation.py` |

## Expected Tests

When implementation is approved, tests should prove:

| Test | Proves |
| --- | --- |
| construct encoded observation | dataclass/protocol works |
| reject mismatched sequence lengths | validation works |
| preserve valid mask | padding contract works |
| encode rank-1 window | rank-1 state width works |
| encode rank-2 window | rank-2 state width works |
| include meter context | bar positions/measure size available |
| include key/target context | model can receive musical goal context |
| no old imports | tower observation does not import `rl_counterpoint` |

Focused verification command for the future implementation action:

```bash
uv run pytest tests/tower/policy/test_observation.py tests/tower/test_import_boundaries.py
```

## Stop Conditions

Pause and resynchronize if:

| Stop condition | Why |
| --- | --- |
| observation design requires canonical graph tensors | violates tuple graph core |
| observation design drops meter/key/target context | conflicts with owner/model direction |
| child model requires parent logits/logprobs/top-m candidates | violates parent/child boundary |
| old `rl_counterpoint` runtime imports seem necessary | violates frozen-project boundary |
| observation contract conflicts with existing sampler API | blocks transformer policy integration |

## Next Phase.Stage.Action

After this contract is accepted, the next proposed action is:

```text
Post-Slice-8 Phase 1.Stage 1.3.Action 1.3.1:
Produce File-Level Build Map
```

That action would create:

```text
docs/design/tower/post_slice_8_file_map.md
```

No code implementation is approved by this document.
