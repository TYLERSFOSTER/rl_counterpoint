#!/bin/zsh
set -euo pipefail

LINEAGE="${1:-rank1-rank2-rank3-5k-two-stage}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"
LOG="$ART/logs/$LINEAGE.orchestrator.log"

mkdir -p "$ART/logs"

cd "$ROOT"

echo "lineage: $LINEAGE" | tee "$LOG"

./scripts/run_rank1_stage1_5k_two_stage.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank1_stage2_5k_two_stage.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank2_stage1_5k_two_stage.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank2_stage2_5k_two_stage.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank3_stage1_5k_two_stage.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank3_stage2_5k_two_stage.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
