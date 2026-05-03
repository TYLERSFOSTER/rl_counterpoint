#!/bin/zsh
set -euo pipefail

LINEAGE="${1:-rank1-rank2-rank3-10k-sampledtonic}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"

cd "$ROOT"

./.venv/bin/python scripts/tower_train_rank1_stage1.py \
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
  --final-inference-sample-initial-state
