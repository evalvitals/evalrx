#!/bin/bash
# triage18 phase 1 (10-22/23): the 12-arm protocol of triage12-16 (rules / scored / scored_x0 / scored_d x
# full / b700 / b900) under rules-5 with --triage-max-fail 128, then the 12 arms validated together on CONFIRM
# plus the validation-only source. The arms run on a code snapshot (the original run: commit 0d01bb6), so the
# working tree can be edited meanwhile; triage17's observation ledgers are copied in for chunk reuse.
# Idempotent: an arm whose report copy exists is skipped.
# usage: run_triage18.sh <gpu index> [git ref to snapshot, default HEAD]
set -u
. "$(dirname "$0")/_common.sh"
GPU=${1:?gpu index}; REF=${2:-HEAD}
TAG=triage18-chartqa-2b-mf128
RUN=$OUT/$MODEL/$DATASET.$TAG
PREV=$OUT/$MODEL/$DATASET.triage17-chartqa-2b-extval
LOG=$SCRATCH/run_$TAG.log
CODE=$SCRATCH/code_${TAG}_$(date +%s)
COMMON=$(common_flags "$TAG" 128)
ARMS="rules:full scored:full scored_x0:full scored_d:full rules:b700 scored:b700 scored_x0:b700 scored_d:b700 rules:b900 scored:b900 scored_x0:b900 scored_d:b900"

snapshot_code "$CODE" "$REF" || exit 1
prepare_env "$GPU" "$CODE" "$TAG"
cd "$CODE" || exit 1
mkdir -p "$RUN/probing" "$RUN/validation"
[ -d "$RUN/probing/observations" ] || { [ -d "$PREV/probing/observations" ] && cp -r "$PREV/probing/observations" "$RUN/probing/"; }
[ -d "$RUN/validation/observations" ] || { [ -d "$PREV/validation/observations" ] && cp -r "$PREV/validation/observations" "$RUN/validation/"; }
echo "START $TAG on GPU $GPU at $(date '+%m-%d %H:%M') code=$CODE ($REF)" >> "$LOG"
run_arms "$LOG" "$RUN" "$COMMON" $ARMS || exit 1
validate_union "$LOG" "$(arm_names_json $ARMS)" "$COMMON"
