#!/bin/zsh
set -euo pipefail

LINEAGE="rank1-rank2-diatonic-10k-20260427-110948"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/tower"
EX="$ROOT/docs/design/tower/examples/$LINEAGE"
LOG="$ART/logs/$LINEAGE.log"

uv run python "$ROOT/scripts/tower_train_staged.py" \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --stage1-episodes 10000 \
  --stage2-episodes 10000 \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --goal-octave-direction-weight 5.0 \
  --target-root-octave-choices 2,3,4,5 \
  2>&1 | tee "$LOG"

uv run python "$ROOT/scripts/tower_train_rank2.py" \
  --episodes 10000 \
  --lineage-id "$LINEAGE" \
  --artifact-root "$ART" \
  --max-steps 64 \
  --max-step-size 7 \
  --seed 123 \
  --initial-parent-pitch 64 \
  --initial-child-pitch 68 \
  --no-log-reward-diagnostics \
  2>&1 | tee -a "$LOG"

mkdir -p "$EX"

cp "$ART/${LINEAGE}-stage1/rank_1/example_episode.mid" "$EX/rank1_stage1_example_episode.mid"
cp "$ART/${LINEAGE}-stage1/rank_1/example_episode_1.mid" "$EX/rank1_stage1_example_episode_1.mid"
cp "$ART/${LINEAGE}-stage1/rank_1/example_episode_2.mid" "$EX/rank1_stage1_example_episode_2.mid"
cp "$ART/${LINEAGE}-stage1/rank_1/example_episode_3.mid" "$EX/rank1_stage1_example_episode_3.mid"

cp "$ART/$LINEAGE/rank_1/example_episode.mid" "$EX/rank1_stage2_example_episode.mid"
cp "$ART/$LINEAGE/rank_1/example_episode_1.mid" "$EX/rank1_stage2_example_episode_1.mid"
cp "$ART/$LINEAGE/rank_1/example_episode_2.mid" "$EX/rank1_stage2_example_episode_2.mid"
cp "$ART/$LINEAGE/rank_1/example_episode_3.mid" "$EX/rank1_stage2_example_episode_3.mid"

cp "$ART/$LINEAGE/rank_2/example_episode.mid" "$EX/rank2_example_episode.mid"
cp "$ART/$LINEAGE/rank_2/example_episode_1.mid" "$EX/rank2_example_episode_1.mid"
cp "$ART/$LINEAGE/rank_2/example_episode_2.mid" "$EX/rank2_example_episode_2.mid"
cp "$ART/$LINEAGE/rank_2/example_episode_3.mid" "$EX/rank2_example_episode_3.mid"
