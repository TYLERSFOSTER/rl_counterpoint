# Final-Rank Graph Construction Contract

This note records the operational contract for how `tower` should build lower-tier
graphs when the intended final training rank is known in advance.

## Core Rule

Graph construction is not purely local to the currently trained rank.

When the user knows the intended final rank \(R\), that final rank should modify
the whole lower tower during graph construction.

Operationally:

1. build the highest-rank source graph for \(R\),
2. prune it by the active legality contract for \(R\),
3. project it down one tier at a time,
4. replace lower-tier effective graphs with those projection images.

So if the intended final rank is 3:

\[
G(3)_\bullet^{\mathrm{legal}}
\xrightarrow{\operatorname{pr}^{3}}
G(2)_\bullet^{(3)}
\xrightarrow{\operatorname{pr}^{2}}
G(1)_\bullet^{(3)}.
\]

The lower tiers used in training are the induced projected graphs, not
independently rebuilt graphs that ignore the final tier.

## Current Operational Surface

### Rank 1 scripts

Rank-1 scripts now expose:

- `--final-rank`
- `--induced-rank3-pitch-min`
- `--induced-rank3-pitch-max`
- `--induced-rank3-max-step-size`

Meaning:

- `final_rank = 1`:
  rank 1 may use its direct graph or rank-2-induced graph path, depending on the
  rest of the config.
- `final_rank = 3`:
  rank 1 should be built from the rank-3-induced rank-2 image, and then the
  rank-2-induced rank-1 image.

### Rank 2 scripts

Rank-2 scripts now also expose:

- `--final-rank`
- `--induced-rank3-pitch-min`
- `--induced-rank3-pitch-max`
- `--induced-rank3-max-step-size`

Meaning:

- `final_rank = 2`:
  rank 2 uses its direct graph legality.
- `final_rank = 3`:
  rank 2 uses the induced image of the legal rank-3 graph.

## Present Scope

As of this document:

- rank-3 legality exists,
- induced rank-2-from-rank-3 artifacts exist,
- runner construction can rebuild rank 2 and then rank 1 from final rank 3.

What does **not** yet exist:

- fully mature rank-3 training behavior,
- automatic final-rank propagation from a rank-3 runner without explicit runner
  or script configuration.

So this contract is already active for lower-tier construction, but only through
the runner and script surfaces that explicitly set `final_rank`.

## Design Intent

This contract is meant to prevent a recurring mistake:

- adding new legality at a higher rank,
- but leaving lower-tier training graphs stale because they were rebuilt
  independently.

The intended invariant is:

- if a higher-rank legality change removes projected states or edges,
  lower-tier effective graphs should reflect that removal.

That is the whole point of the tower construction.
