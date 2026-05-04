#!/bin/zsh
set -euo pipefail

LINEAGE="${1:-rank1-rank2-rank3-5k-two-stage}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"

cd "$ROOT"

./.venv/bin/python scripts/tower_train_rank3_stage1.py \
  --episodes 5000 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --parent-top-m 1 \
  --sample-key-pitch-class \
  --key-pitch-class-choices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --sample-target-root-octave \
  --target-root-octave-choices 2,3,4,5 \
  --final-inference-sample-target-root-octave \
  --final-inference-sample-initial-state
