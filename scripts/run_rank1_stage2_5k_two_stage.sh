#!/bin/zsh
set -euo pipefail

LINEAGE="${1:-rank1-rank2-rank3-5k-two-stage}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"

cd "$ROOT"

./.venv/bin/python scripts/resume_rank1_stage2.py \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --stage1-episodes 5000 \
  --stage2-episodes 5000 \
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
