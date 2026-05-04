#!/bin/zsh
set -euo pipefail

LINEAGE="${1:-rank1-rank2-rank3-100-two-stage-smoke}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"
LOG="$ART/logs/$LINEAGE.orchestrator.log"

mkdir -p "$ART/logs"

cd "$ROOT"

echo "lineage: $LINEAGE" | tee "$LOG"

./.venv/bin/python scripts/tower_train_rank1_stage1.py \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --stage1-episodes 100 \
  --stage2-episodes 100 \
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
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"

./.venv/bin/python scripts/resume_rank1_stage2.py \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --stage1-episodes 100 \
  --stage2-episodes 100 \
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
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"

./.venv/bin/python scripts/tower_train_rank2_stage1.py \
  --episodes 100 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --parent-top-m 1 \
  --final-rank 3 \
  --sample-key-pitch-class \
  --key-pitch-class-choices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --sample-target-root-octave \
  --target-root-octave-choices 2,3,4,5 \
  --induced-rank3-pitch-min 36 \
  --induced-rank3-pitch-max 84 \
  --induced-rank3-max-step-size 7 \
  --final-inference-sample-target-root-octave \
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"

./.venv/bin/python scripts/resume_rank2_stage2.py \
  --episodes 100 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --parent-top-m 1 \
  --final-rank 3 \
  --sample-key-pitch-class \
  --key-pitch-class-choices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --sample-target-root-octave \
  --target-root-octave-choices 2,3,4,5 \
  --induced-rank3-pitch-min 36 \
  --induced-rank3-pitch-max 84 \
  --induced-rank3-max-step-size 7 \
  --final-inference-sample-target-root-octave \
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"

./.venv/bin/python scripts/tower_train_rank3_stage1.py \
  --episodes 100 \
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
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"

./.venv/bin/python scripts/resume_rank3_stage2.py \
  --episodes 100 \
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
  --final-inference-sample-initial-state 2>&1 | tee -a "$LOG"
