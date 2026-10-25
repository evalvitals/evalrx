#!/bin/bash
# triage17 (10-21/22): triage16's stored arms, validated once more on DIAGNOSTIC_VALIDATION extended by the
# validation-only source (the human rows the pool left: the rest of the test split plus the val split). Only
# the validation runs (two Stage 0 passes: the pool's and the source's); the code is the working tree.
# usage: run_triage17.sh <gpu index> [run tag] [source manifest]
set -u
. "$(dirname "$0")/_common.sh"
GPU=${1:?gpu index}
TAG=${2:-triage17-chartqa-2b-extval}
SRC=${3:-$SRC}
LOG=$SCRATCH/validate_$TAG.log
COMMON=$(common_flags "$TAG" 32)
ARMS="rules:full scored:full scored_x0:full scored_d:full rules:b700 scored:b700 scored_x0:b700 scored_d:b700 rules:b900 scored:b900 scored_x0:b900 scored_d:b900"
prepare_env "$GPU" "$REPO" "$TAG"
cd "$REPO" || exit 1
echo "START $TAG validation on GPU $GPU at $(date '+%m-%d %H:%M') source=$SRC" >> "$LOG"
validate_union "$LOG" "$(arm_names_json $ARMS)" "$COMMON"
