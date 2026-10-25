#!/bin/bash
# triage18 phase 2 (10-23/24), in the triage18 run directory (shared probing ledger), one GPU, one go:
#   1. the controls at the full budget: random_full, cheapest_full;
#   2. the rescaled budget tiers 4000 / 5500 / 7100 s for all six schedulers = 18 arms (at max_fail 128 the
#      700/900 s tiers stop inside the expansion and are all alike);
#   3. one union validation over all 32 arms (validation ledger reused; the 12-arm report kept as *.12arms.*).
# The arms run on a snapshot of the working tree's tracked files (the original run carried the control
# schedulers before they were committed as 2695741). Idempotent. usage: run_triage18_phase2.sh <gpu index>
set -u
. "$(dirname "$0")/_common.sh"
GPU=${1:?gpu index}
TAG=triage18-chartqa-2b-mf128
RUN=$OUT/$MODEL/$DATASET.$TAG
LOG=$SCRATCH/run_${TAG}_phase2.log
CODE=$SCRATCH/code_${TAG}_p2_$(date +%s)
COMMON=$(common_flags "$TAG" 128)
PHASE1="rules:full scored:full scored_x0:full scored_d:full rules:b700 scored:b700 scored_x0:b700 scored_d:b700 rules:b900 scored:b900 scored_x0:b900 scored_d:b900"
ARMS="random:full cheapest:full"
for b in 4000 5500 7100; do for k in rules scored scored_x0 scored_d random cheapest; do ARMS="$ARMS $k:b$b"; done; done

snapshot_code "$CODE" worktree || exit 1
grep -q '"cheapest"' "$CODE/evalrx/eval_agent/probing/controller.py" || { echo "snapshot has no control schedulers" >> "$LOG"; exit 1; }
[ -f "$RUN/probing/triage_report.rules_full.json" ] || { echo "phase 1 arms missing" >> "$LOG"; exit 1; }
prepare_env "$GPU" "$CODE" "${TAG}_phase2"
cd "$CODE" || exit 1
echo "START phase2 on GPU $GPU at $(date '+%m-%d %H:%M') code=$CODE (HEAD $(git -C "$REPO" rev-parse --short HEAD) + $(git -C "$REPO" status --short | wc -l) modified files)" >> "$LOG"
echo "ARMS: $ARMS" >> "$LOG"
keep_aside "$RUN" 12arms
run_arms "$LOG" "$RUN" "$COMMON" $ARMS || exit 1
validate_union "$LOG" "$(arm_names_json $PHASE1 $ARMS)" "$COMMON"
