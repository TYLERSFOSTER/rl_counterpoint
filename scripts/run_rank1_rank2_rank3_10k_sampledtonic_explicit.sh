#!/bin/zsh
set -euo pipefail

LINEAGE="${1:-rank1-rank2-rank3-10k-sampledtonic-explicit}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"
LOG="$ART/logs/$LINEAGE.orchestrator.log"

mkdir -p "$ART/logs"

cd "$ROOT"

echo "lineage: $LINEAGE" | tee "$LOG"

./scripts/run_rank1_stage1_10k_sampledtonic.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank1_stage2_10k_sampledtonic.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank2_10k_sampledtonic.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
./scripts/run_rank3_10k_sampledtonic.sh "$LINEAGE" 2>&1 | tee -a "$LOG"
