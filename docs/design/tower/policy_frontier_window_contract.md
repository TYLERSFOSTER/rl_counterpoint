# Policy Frontier/Window Contract

This document is the Post-Slice-8 Phase 3 / Stage 3.3 / Action 3.3.0
deliverable.

The purpose is to correct the discovered ambiguity between `state` and `window`
before sampler compatibility work resumes.

This is a planning document, not implementation approval.

## Stage Location

| Plan level | Current location |
| --- | --- |
| Phase | Post-Slice-8 Phase 3: Transformer-Family Rank Policy |
| Stage | Stage 3.3: Frontier Contract And Sampler Integration |
| Action | Action 3.3.0: Clarify Policy Frontier/Window Contract |

Action 3.3.0 exit criterion:

| Requirement | Status |
| --- | --- |
| policy input ownership recorded | drafted here |
| frontier-state derivation recorded | drafted here |
| rollout-state exception recorded | drafted here |
| sampler responsibility recorded | drafted here |
| transformer-policy call shape recorded | drafted here |
| next implementation path recorded | drafted here |

## Source Authority

This contract is derived from:

| Source | Role |
| --- | --- |
| `docs/design/tower/post_slice_8_phase_stage_action_plan.md` | current explicit plan |
| `docs/design/tower/post_slice_8_file_map.md` | current file/action map |
| `docs/design/tower/real_policy_observation_contract.md` | encoded-window model contract |
| `docs/design/tower/slice4_rollout_clarifications.md` | rollout/source-state record contract |
| `tower/window.py` | current tower window implementation |
| `tower/policy/observation.py` | current encoded-window implementation |
| `tower/policy/transformer.py` | current transformer policy implementation |
| `tower/policy/samplers.py` | current sampler API to be reconciled |

## Correction Summary

The tower policy API should be window-primary.

The `state` and `window` values should not be independent model inputs. For
policy/model purposes, the current state is the frontier of the current window:

```text
frontier_state(W_t^k) = final valid state in W_t^k
```

The policy input path should therefore be:

```text
TowerWindow
-> EncodedTowerWindow
-> TowerTransformerPolicy
-> PolicyOutput
-> external sampler/legality/lift-fiber masking
```

This correction preserves tuple state/action graph mechanics. It only clarifies
the model and sampler interface.

## Terms

| Term | Meaning |
| --- | --- |
| `TowerWindow` | fixed-length, padded, rank-local musical context |
| frontier state | final valid state in a `TowerWindow` |
| rollout state | local graph state used by rollout for action enumeration and transition application |
| encoded window | tensorized policy-layer representation of a `TowerWindow` |
| policy input | model-ready observation consumed by a real rank policy |

## Policy Input Rule

Real tower policies should consume an encoded window, not a separate
`state + window` pair.

Required policy stance:

| Item | Rule |
| --- | --- |
| real transformer policy | consumes `EncodedTowerWindow` |
| current state for model | derived from final valid window event |
| independent `state` argument | not part of real model input |
| graph action masks | may use derived frontier state |
| padding | never treated as frontier |

The current `TowerTransformerPolicy` already follows the desired model call
shape by accepting `EncodedTowerWindow` directly and selecting the final valid
event internally for logits.

## Frontier State Rule

The frontier state of a window is the last state whose `valid_mask` entry is
true.

Required behavior:

| Case | Rule |
| --- | --- |
| left-padded window | frontier is normally the last physical sequence state |
| future right-padded or sparse-valid window | frontier is the final valid state, not the final physical position |
| all-padding window | invalid policy/sampler input |
| rank | frontier rank must match the policy/sampler rank |

The current observation encoder already requires at least one valid event. That
is consistent with this contract.

## Rollout State Exception

Rollout may still keep a local `source_state`.

This is not a competing model input. It is graph mechanics:

| Rollout use | Status |
| --- | --- |
| enumerate legal actions | allowed |
| compute lift fiber | allowed |
| apply assembled action | allowed |
| store trajectory diagnostics | allowed |
| feed model as separate context | forbidden for real policy path |

When rollout has both `source_state` and `window`, they must describe the same
frontier:

```text
source_state == frontier_state(window)
```

Any mismatch is a bug or stale-window condition. It should not be interpreted
as meaningful musical data.

## Sampler Responsibility

Samplers sit between policy output and graph action selection, so they may need
both:

| Data | Reason |
| --- | --- |
| encoded/window policy input | get active-choice logits |
| frontier state | enumerate legal actions, masks, and lift fibers |

Under this contract, samplers should derive the frontier from the window rather
than accepting a separately authoritative `state` value for real policy calls.

Acceptable implementation paths:

| Path | Status |
| --- | --- |
| add a `frontier_state(window)` helper and update samplers to use it | preferred |
| add a small adapter around old `RankPolicy` test doubles | allowed |
| broaden policy protocol to support encoded-window policies explicitly | allowed |
| keep independent `state` and `window` as real policy inputs | not allowed |

The sampler compatibility action should choose the smallest path that lets the
real transformer policy work while preserving existing scripted/mock policy
tests.

## Existing Protocol Reconciliation

The current sampler protocol predates the real transformer policy and calls
policies with:

```text
policy(state=state, window=window)
```

That call shape is now legacy/test-helper shaped. It should not define the real
policy architecture.

The next implementation action should reconcile this in one of two ways:

| Option | Meaning |
| --- | --- |
| sampler refactor | samplers accept a window, derive frontier, encode the window, and call encoded-window policies |
| adapter layer | keep sampler internals mostly intact but wrap real policies behind a compatible callable |

The preferred direction is a sampler refactor if it stays small. Use an adapter
only if it avoids a large cross-cutting change.

## Parent/Child Boundary

This correction does not change the parent/child information rule.

Forbidden child model inputs remain forbidden:

| Parent data | Status |
| --- | --- |
| parent logits | forbidden |
| parent logprobs | forbidden |
| parent top-m candidates | forbidden |
| parent optimizer state | forbidden |

Allowed operational data remains allowed:

| Parent data | Status |
| --- | --- |
| sampled parent action | used by rollout to compute lift fiber |
| lift-fiber mask | used externally to constrain child choices |
| parent sampler diagnostics | recorded for audit/debug, not fed to child model |

## Next Implementation Implications

`Phase 3.Stage 3.3.Action 3.3.1` should resume sampler compatibility with this
contract as authority.

Expected implementation decisions:

| Decision | Expected direction |
| --- | --- |
| frontier helper | add or reuse a single helper for final valid state |
| real policy call | call transformer policy with `EncodedTowerWindow` |
| sampler tests | prove active sampler works with transformer policy |
| parent sampler tests | prove parent diagnostics stay detached |
| mismatch behavior | reject or prevent `source_state != frontier_state(window)` |
| old policy doubles | preserve through small adapters or updated test doubles |

Focused tests should cover the real compatibility path rather than only mock
policies.

## Stop Conditions

Pause and resynchronize if:

| Stop condition | Why |
| --- | --- |
| sampler compatibility requires parent logits/logprobs/top-m candidates as child model input | violates owner model decision |
| graph/window tuple contracts need to become torch tensors | violates layer boundary |
| rollout intentionally permits `source_state` to differ from window frontier | contradicts this correction |
| adapter code hides a divergent real policy API instead of clarifying it | preserves the original ambiguity |
