# Count `G(n)` Sparsity

## Purpose

This document preserves the design-side sparsity-counting method for the counterpoint graph `G(n)`.

It is intentionally a Markdown design artifact rather than a standalone executable script.

Reason:

- the live runtime graph authority now lives in the repo code under `rl_counterpoint/graph/`
- this counting logic is useful as mathematical design/reference material
- keeping it as a separate executable script risks creating a second authority that can silently drift away from the live graph contract

So this document is the canonical design-side explanation of the counting method, and it preserves the prior Python implementation as an embedded reference.

## What The Count Measures

The method counts sparsity in the counterpoint graph `G(n)` using finite-sum formulas over gap vectors rather than explicit graph materialization.

It keeps separate:

- raw node counts
- trimmed node counts
- edge counts under combinations of the three edge trims:
  - voice crossing
  - parallel fifths
  - single-line interval cap

This is a design/reference tool, not the runtime graph definition.

## Mathematical Setup

Notation:

- `P` = number of MIDI pitch values, default `128`
- `M` = max interval value, default `11`
- `C(n) = ceil(6n)`
- `tau` = tonic MIDI note
- `S = {3, 4, 5, 7, 8, 9}`

Raw node set:

```text
G(n)_0^raw = {
    lambda in {0, ..., P - 1}^n
    | lambda_0 < ... < lambda_{n-1}
}
```

Trimmed node set:

```text
G(n)_0 = {
    lambda in G(n)_0^raw
    | lambda_{i+1} - lambda_i <= M for all i,
      lambda_{i+1} - lambda_i not in {1, 2, 6, 10, 11} for all i,
      lambda_{n-1} - lambda_0 <= C(n),
      lambda_{n-1} - lambda_0 in S mod 12,
      lambda_0 - tau in S mod 12
}
```

The implementation below counts via admissible gap vectors and relative target-root offsets.

## Embedded Python Reference

```python
#!/usr/bin/env python3
"""Formula counts for sparsity in the counterpoint graph G(n).

This script evaluates finite-sum formulas over gap vectors, not counts over
explicit chord nodes or edge pairs.

Notation:

    P = number of MIDI pitch values, default 128.
    M = max interval value, default 11.
    C(n) = ceil(6n).
    tau = tonic MIDI note.
    S = {3, 4, 5, 7, 8, 9}.

Raw node set:

    G(n)_0^raw = {
        lambda in {0, ..., P - 1}^n
        | lambda_0 < ... < lambda_{n-1}
    }

    |G(n)_0^raw| = binomial(P, n)

Trimmed node set:

    G(n)_0 = {
        lambda in G(n)_0^raw
        | lambda_{i+1} - lambda_i <= M for all i,
          lambda_{i+1} - lambda_i not in {1, 2, 6, 10, 11} for all i,
          lambda_{n-1} - lambda_0 <= C(n),
          lambda_{n-1} - lambda_0 in S mod 12,
          lambda_0 - tau in S mod 12
    }

Gap-vector formula for nodes:

    Write d_i = lambda_{i+1} - lambda_i for 0 <= i < n - 1.
    Let D = sum_i d_i. Then:

    |G(n)_0| =
        sum over d in ({1, ..., M} - {1, 2, 6, 10, 11})^{n-1},
                 sum(d) <= C(n),
                 sum(d) in S mod 12
            #{s in [0, P - 1 - sum(d)] | s - tau in S mod 12}

    For n = 1, the empty gap vector has sum 0.

Edges:

    Source and target maps:

        delta_0(lambda -> mu) = lambda
        delta_1(lambda -> mu) = mu

    Self-loops are always excluded.

    For source gaps d and target gaps e, define prefix offsets:

        D_i = sum_{k < i} d_k
        E_i = sum_{k < i} e_k

    Let source start be s = lambda_0 and target start be t = mu_0.
    Write a = t - s. Then:

        lambda_i = s + D_i
        mu_i = s + a + E_i

    For fixed d, e, and a, the number of valid starts s is:

        #{s in [max(0, -a), min(P - 1 - sum(d), P - 1 - sum(e) - a)]
          | s - tau in S mod 12
            and s + a - tau in S mod 12}

    Voice-crossing trim:

        no crossing iff for all 0 <= i < n - 1:

            mu_i < lambda_{i+1}
            lambda_i < mu_{i+1}

        In gap/offset coordinates this is:

            a <= D_{i+1} - E_i - 1
            a >= D_i - E_{i+1} + 1

    Parallel-fifth trim, all voice pairs:

        A parallel fifth exists iff for some 0 <= i < j < n:

            D_j - D_i = 7
            E_j - E_i = 7

        The relative start offset a cancels out of this condition.

    Single-line interval trim:

        mu_i - lambda_i <= M for all i

        In gap/offset coordinates this is:

            a <= M + D_i - E_i

    The script reports all eight combinations of the three edge trims after
    applying the G(n)_0 node trims.
"""

from __future__ import annotations

import argparse
from itertools import product
from math import ceil, comb

PITCH_COUNT = 128
DEFAULT_MAX_INTERVAL = 11
PARALLEL_FIFTH_SIZE = 7
FORBIDDEN_ADJACENT_VERTICAL_INTERVALS = frozenset({1, 2, 6, 10, 11})
ALLOWED_TONIC_ROOT_INTERVALS = frozenset({3, 4, 5, 7, 8, 9})


def format_int(value: int) -> str:
    return f"{value:,}"


def trim_label(
    *,
    trim_crossing: bool,
    trim_parallel_fifths: bool,
    trim_single_line_interval: bool,
) -> str:
    trim_names = []

    if trim_crossing:
        trim_names.append("no_voice_crossing")

    if trim_parallel_fifths:
        trim_names.append("no_parallel_fifths")

    if trim_single_line_interval:
        trim_names.append("max_single_line_interval")

    if not trim_names:
        return "no_trims"

    return "__".join(trim_names)


def gap_vectors(*, n: int, pitch_count: int, max_interval: int) -> list[tuple[int, ...]]:
    """Return admissible gap vectors for trimmed G(n)_0."""
    if n == 1:
        return [()]

    width_cap = min(ceil(n * 6), pitch_count - 1)
    max_gap = min(max_interval, width_cap)
    allowed_gaps = [
        gap
        for gap in range(1, max_gap + 1)
        if gap not in FORBIDDEN_ADJACENT_VERTICAL_INTERVALS
    ]
    gaps = []

    for candidate in product(allowed_gaps, repeat=n - 1):
        candidate_width = sum(candidate)
        if (
            candidate_width <= width_cap
            and candidate_width % 12 in ALLOWED_TONIC_ROOT_INTERVALS
        ):
            gaps.append(candidate)

    return gaps


def prefix_offsets(gaps: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    current = 0

    for gap in gaps:
        current += gap
        offsets.append(current)

    return tuple(offsets)


def has_parallel_fifth(
    source_offsets: tuple[int, ...],
    target_offsets: tuple[int, ...],
) -> bool:
    n = len(source_offsets)

    for i in range(n):
        for j in range(i + 1, n):
            if (
                source_offsets[j] - source_offsets[i] == PARALLEL_FIFTH_SIZE
                and target_offsets[j] - target_offsets[i] == PARALLEL_FIFTH_SIZE
            ):
                return True

    return False


def count_values_with_residue_classes(
    *,
    lower: int,
    upper: int,
    residues: frozenset[int],
) -> int:
    """Count integers x in [lower, upper] with x mod 12 in residues."""
    if lower > upper:
        return 0

    total = 0
    for residue in residues:
        first = lower + ((residue - lower) % 12)
        if first <= upper:
            total += (upper - first) // 12 + 1

    return total


def allowed_root_residues(*, tonic: int) -> frozenset[int]:
    return frozenset((tonic + interval) % 12 for interval in ALLOWED_TONIC_ROOT_INTERVALS)


def valid_start_count(
    *,
    pitch_count: int,
    source_width: int,
    target_width: int,
    target_offset: int,
    tonic: int,
) -> int:
    """Count starts s with source and target chords inside range and scale."""
    lower = max(0, -target_offset)
    upper = min(
        pitch_count - 1 - source_width,
        pitch_count - 1 - target_width - target_offset,
    )

    source_residues = allowed_root_residues(tonic=tonic)
    target_residues_as_source_values = frozenset(
        (residue - target_offset) % 12 for residue in source_residues
    )
    allowed_residues = source_residues & target_residues_as_source_values

    return count_values_with_residue_classes(
        lower=lower,
        upper=upper,
        residues=allowed_residues,
    )


def offset_bounds(
    *,
    pitch_count: int,
    max_interval: int,
    source_offsets: tuple[int, ...],
    target_offsets: tuple[int, ...],
    trim_crossing: bool,
    trim_single_line_interval: bool,
) -> tuple[int, int]:
    source_width = source_offsets[-1]
    target_width = target_offsets[-1]
    lower = -(pitch_count - 1 - source_width)
    upper = pitch_count - 1 - target_width

    if trim_crossing:
        for i in range(len(source_offsets) - 1):
            upper = min(upper, source_offsets[i + 1] - target_offsets[i] - 1)
            lower = max(lower, source_offsets[i] - target_offsets[i + 1] + 1)

    if trim_single_line_interval:
        for source_offset, target_offset in zip(
            source_offsets,
            target_offsets,
            strict=True,
        ):
            upper = min(upper, max_interval + source_offset - target_offset)

    return lower, upper


def count_nodes(*, gaps: list[tuple[int, ...]], pitch_count: int, tonic: int) -> int:
    residues = allowed_root_residues(tonic=tonic)
    return sum(
        count_values_with_residue_classes(
            lower=0,
            upper=pitch_count - 1 - sum(gap_vector),
            residues=residues,
        )
        for gap_vector in gaps
    )


def count_edges_for_trim(
    *,
    gaps: list[tuple[int, ...]],
    pitch_count: int,
    max_interval: int,
    tonic: int,
    trim_crossing: bool,
    trim_parallel_fifths: bool,
    trim_single_line_interval: bool,
) -> int:
    total = 0
    prefix_by_gaps = {gap_vector: prefix_offsets(gap_vector) for gap_vector in gaps}

    for source_gaps in gaps:
        source_offsets = prefix_by_gaps[source_gaps]
        source_width = source_offsets[-1]

        for target_gaps in gaps:
            target_offsets = prefix_by_gaps[target_gaps]

            if trim_parallel_fifths and has_parallel_fifth(
                source_offsets,
                target_offsets,
            ):
                continue

            lower, upper = offset_bounds(
                pitch_count=pitch_count,
                max_interval=max_interval,
                source_offsets=source_offsets,
                target_offsets=target_offsets,
                trim_crossing=trim_crossing,
                trim_single_line_interval=trim_single_line_interval,
            )

            target_width = target_offsets[-1]

            for target_start_offset in range(lower, upper + 1):
                edge_count = valid_start_count(
                    pitch_count=pitch_count,
                    source_width=source_width,
                    target_width=target_width,
                    target_offset=target_start_offset,
                    tonic=tonic,
                )

                if target_start_offset == 0 and source_gaps == target_gaps:
                    edge_count -= valid_start_count(
                        pitch_count=pitch_count,
                        source_width=source_width,
                        target_width=target_width,
                        target_offset=target_start_offset,
                        tonic=tonic,
                    )

                total += edge_count

    return total


def count_edges(
    *,
    n: int,
    pitch_count: int,
    max_interval: int,
    tonic: int,
) -> dict[str, int]:
    gaps = gap_vectors(n=n, pitch_count=pitch_count, max_interval=max_interval)
    node_count = count_nodes(gaps=gaps, pitch_count=pitch_count, tonic=tonic)

    counts = {
        "raw_nodes": comb(pitch_count, n),
        "trimmed_nodes": node_count,
        "no_trims": node_count * (node_count - 1),
    }

    for trim_crossing in (False, True):
        for trim_parallel_fifths in (False, True):
            for trim_single_line_interval in (False, True):
                label = trim_label(
                    trim_crossing=trim_crossing,
                    trim_parallel_fifths=trim_parallel_fifths,
                    trim_single_line_interval=trim_single_line_interval,
                )

                if label == "no_trims":
                    continue

                counts[label] = count_edges_for_trim(
                    gaps=gaps,
                    pitch_count=pitch_count,
                    max_interval=max_interval,
                    tonic=tonic,
                    trim_crossing=trim_crossing,
                    trim_parallel_fifths=trim_parallel_fifths,
                    trim_single_line_interval=trim_single_line_interval,
                )

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate formula counts for G(n) edge totals under trims."
    )
    parser.add_argument("n", type=int, help="number of voices")
    parser.add_argument(
        "--pitch-count",
        type=int,
        default=PITCH_COUNT,
        help="number of MIDI pitch values; default uses 0..127",
    )
    parser.add_argument(
        "--max-interval",
        type=int,
        default=DEFAULT_MAX_INTERVAL,
        help=(
            "shared max interval value: adjacent node gaps and active "
            "single-line trim both use this cap"
        ),
    )
    parser.add_argument(
        "--tonic",
        type=int,
        required=True,
        help="tonic MIDI note tau; only tau mod 12 affects counts",
    )
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("n must be at least 1")

    if args.n > args.pitch_count:
        raise SystemExit("n cannot exceed pitch-count")

    counts = count_edges(
        n=args.n,
        pitch_count=args.pitch_count,
        max_interval=args.max_interval,
        tonic=args.tonic,
    )

    print(f"G({args.n}) over pitch values 0..{args.pitch_count - 1}")
    print(f"raw nodes before G(n)_0 trims: {format_int(counts['raw_nodes'])}")
    print(f"trimmed nodes: {format_int(counts['trimmed_nodes'])}")
    print(
        "G(n)_0 trims: "
        f"adjacent gaps <= {args.max_interval}, "
        "adjacent gaps not in "
        f"{sorted(FORBIDDEN_ADJACENT_VERTICAL_INTERVALS)}, "
        "outer interval in "
        f"{sorted(ALLOWED_TONIC_ROOT_INTERVALS)} mod 12, "
        f"total chord width <= {ceil(args.n * 6)}, "
        f"root - tonic in {sorted(ALLOWED_TONIC_ROOT_INTERVALS)} mod 12"
    )
    print(f"G(n)_1 single-line interval trim: mu_i - lambda_i <= {args.max_interval}")
    print()

    labels = [
        "no_trims",
        "max_single_line_interval",
        "no_parallel_fifths",
        "no_parallel_fifths__max_single_line_interval",
        "no_voice_crossing",
        "no_voice_crossing__max_single_line_interval",
        "no_voice_crossing__no_parallel_fifths",
        "no_voice_crossing__no_parallel_fifths__max_single_line_interval",
    ]

    for label in labels:
        print(f"{label}: {format_int(counts[label])}")


if __name__ == "__main__":
    main()
```

## Status

This Markdown document is the preserved design/reference form of the prior counting script.

It should not be treated as the live runtime authority for graph definition.
