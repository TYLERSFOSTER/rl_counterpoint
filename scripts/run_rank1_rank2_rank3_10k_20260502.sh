#!/bin/zsh
set -euo pipefail

LINEAGE=rank1-rank2-rank3-10k-20260502
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"
LOG="$ART/logs/$LINEAGE.log"

mkdir -p "$ART/logs"

cd "$ROOT"

uv run python scripts/tower_train_staged.py \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --stage1-episodes 10000 \
  --stage2-episodes 10000 \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --goal-octave-direction-weight 5.0 \
  --target-root-octave-choices 2,3,4,5 \
  --final-rank 3 \
  --induced-rank3-pitch-min 36 \
  --induced-rank3-pitch-max 84 \
  --induced-rank3-max-step-size 7 \
  --final-inference-sample-target-root-octave \
  --final-inference-sample-initial-state 2>&1 | tee "$LOG"

uv run python scripts/tower_train_rank2.py \
  --episodes 10000 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --parent-top-m 1 \
  --final-rank 3 \
  --sample-initial-state \
  --sample-target-root-octave \
  --target-root-octave-choices 2,3,4,5 \
  --induced-rank3-pitch-min 36 \
  --induced-rank3-pitch-max 84 \
  --induced-rank3-max-step-size 7 \
  --no-log-reward-diagnostics \
  --final-inference-sample-target-root-octave \
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"

uv run python scripts/tower_train_rank3.py \
  --episodes 10000 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --parent-top-m 1 \
  --sample-initial-state \
  --sample-target-root-octave \
  --target-root-octave-choices 2,3,4,5 \
  --no-log-reward-diagnostics \
  --final-inference-sample-target-root-octave \
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"
