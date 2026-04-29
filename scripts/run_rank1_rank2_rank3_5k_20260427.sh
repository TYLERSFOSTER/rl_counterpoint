#!/bin/zsh
set -euo pipefail

LINEAGE=rank1-rank2-rank3-5k-20260427
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"
LOG="$ART/logs/$LINEAGE.log"

cd "$ROOT"

uv run python scripts/tower_train_staged.py \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --stage1-episodes 5000 \
  --stage2-episodes 5000 \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --goal-octave-direction-weight 5.0 \
  --target-root-octave-choices 2,3,4,5 2>&1 | tee "$LOG"

uv run python scripts/tower_train_rank2.py \
  --episodes 5000 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --initial-parent-pitch 62 \
  --initial-child-pitch 69 \
  --parent-top-m 1 \
  --no-log-reward-diagnostics 2>&1 | tee -a "$LOG"

uv run python scripts/tower_train_rank3.py \
  --episodes 5000 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --initial-pedal-pitch 62 \
  --initial-middle-pitch 65 \
  --initial-top-pitch 69 \
  --parent-top-m 1 \
  --no-log-reward-diagnostics 2>&1 | tee -a "$LOG"
