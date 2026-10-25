#!/bin/bash
# triage18 phase 3 (10-24), in the triage18 run directory (shared probing ledger), one GPU, one go: every one
# of the 32 arms again under the step_cost fix (scoring mvp3-score-5: a zero-call audit is priced at what it
# measured, so an expansion's estimate is no longer 3.5-24x its cost), then one union validation over the 32
# arms. Phase-2 artefacts are kept: arm reports under probing/arms_p2/, the validation as *.32arms_p2.*.
# The arms run on a snapshot of the working tree's tracked files. Idempotent. usage: run_triage18_phase3.sh <gpu index>
set -u
. "$(dirname "$0")/_common.sh"
GPU=${1:?gpu index}
TAG=triage18-chartqa-2b-mf128
RUN=$OUT/$MODEL/$DATASET.$TAG
LOG=$SCRATCH/run_${TAG}_phase3.log
CODE=$SCRATCH/code_${TAG}_p3_$(date +%s)
COMMON=$(common_flags "$TAG" 128)
ARMS=""
for k in rules scored scored_x0 scored_d random cheapest; do ARMS="$ARMS $k:full"; done
for b in 700 900; do for k in rules scored scored_x0 scored_d; do ARMS="$ARMS $k:b$b"; done; done
for b in 4000 5500 7100; do for k in rules scored scored_x0 scored_d random cheapest; do ARMS="$ARMS $k:b$b"; done; done

snapshot_code "$CODE" worktree || exit 1
grep -q 'mvp3-score-5' "$CODE/evalrx/eval_agent/probing/scoring.py" || { echo "snapshot lacks the step_cost fix" >> "$LOG"; exit 1; }
prepare_env "$GPU" "$CODE" "${TAG}_phase3"
cd "$CODE" || exit 1
echo "START phase3 on GPU $GPU at $(date '+%m-%d %H:%M') code=$CODE (HEAD $(git -C "$REPO" rev-parse --short HEAD) + $(git -C "$REPO" status --short | wc -l) modified files)" >> "$LOG"
echo "ARMS: $ARMS" >> "$LOG"
if [ ! -d "$RUN/probing/arms_p2" ]; then   # keep phase 2's arm reports aside, once
  mkdir -p "$RUN/probing/arms_p2"
  for a in $ARMS; do k=${a%:*}; b=${a#*:}; n=$(report_name "$k"); mv "$RUN/probing/$n.${k}_$b.json" "$RUN/probing/$n.${k}_$b.md" "$RUN/probing/arms_p2/" 2>> "$LOG"; done
fi
keep_aside "$RUN" 32arms_p2
run_arms "$LOG" "$RUN" "$COMMON" $ARMS || exit 1
validate_union "$LOG" "$(arm_names_json $ARMS)" "$COMMON"
