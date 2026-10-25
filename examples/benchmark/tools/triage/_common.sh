# Shared settings of the triage protocol scripts (sourced by run_*.sh, not run).
# Machine-specific values come from the environment:
#   PY              python with evalrx and the model stack (default: python; the runs used a vLLM/transformers venv)
#   TRIAGE_SCRATCH  logs, a private TMPDIR and code snapshots (default: ~/.cache/evalrx/triage)
#   TRIAGE_DATA     the frozen ChartQA pool (default: <repo>/examples/benchmark/vlm/_data/triage_pool)
#   TRIAGE_OUT      the run-directory root (default: <repo>/examples/benchmark/vlm/qwen/host_outputs)
#   TRIAGE_SOURCE   the validation-only manifest (default: $TRIAGE_DATA/chartqa_validation/manifest.json)
#   TRIAGE_MODEL / TRIAGE_DATASET   the model spec and task (default: qwen3.5-2b-vl / chartqa)
#   JUDGE_FLAGS     the judge of the scored_d arms (default: claude-opus-5 through the claude CLI, effort high)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
PY=${PY:-python}
SCRATCH=${TRIAGE_SCRATCH:-$HOME/.cache/evalrx/triage}
DATA=${TRIAGE_DATA:-$REPO/examples/benchmark/vlm/_data/triage_pool}
OUT=${TRIAGE_OUT:-$REPO/examples/benchmark/vlm/qwen/host_outputs}
SRC=${TRIAGE_SOURCE:-$DATA/chartqa_validation/manifest.json}
MODEL=${TRIAGE_MODEL:-qwen3.5-2b-vl}
DATASET=${TRIAGE_DATASET:-chartqa}
JUDGE_FLAGS=${JUDGE_FLAGS:---judge-provider claude --judge-model claude-opus-5 --judge-effort high}
mkdir -p "$SCRATCH"

common_flags() {  # <run tag> <max fail>: the flags every arm shares (model, data, protocol constants)
  echo "--modality vlm --model $MODEL --backend hf_local --dataset $DATASET --data-dir $DATA --run-dir $OUT --run-tag $1 --limit 1000 --no-download --device cuda --attn-impl eager --m1-policy triage_active --triage-n-fail 8 --triage-max-fail $2 --triage-max-actions 30 --triage-chunk 8 $JUDGE_FLAGS"
}
report_name() {  # <scheduler kind> -> the report stem run_triage writes
  case $1 in rules) echo triage_report ;; scored|scored_x0) echo triage_report.scored ;; scored_d) echo triage_report.scored_d ;;
               random) echo triage_report.random ;; cheapest) echo triage_report.cheapest ;; esac
}
sched_flags() {  # <scheduler kind> -> its flags
  case $1 in rules) echo "--triage-scheduler rules" ;; scored) echo "--triage-scheduler scored" ;;
             scored_x0) echo "--triage-scheduler scored --triage-explore-share 0" ;;
             scored_d) echo "--triage-scheduler scored --triage-explanations judge" ;;
             random) echo "--triage-scheduler random" ;; cheapest) echo "--triage-scheduler cheapest" ;; esac
}
budget_flags() {  # <full | b<seconds>>
  case $1 in b*) echo "--triage-budget-sec ${1#b}" ;; esac
}
arm_names_json() {  # <kind:budget ...> -> the JSON list validate_arms.py takes
  local s="[" a k b
  for a in "$@"; do k=${a%:*}; b=${a#*:}; s="$s[\"${k}_${b}\",\"$(report_name "$k")\"],"; done
  echo "${s%,}]"
}
snapshot_code() {  # <dir> <git ref | worktree>: the code the arms run on, so the working tree can change meanwhile
  mkdir -p "$1" || return 1
  if [ "$2" = worktree ]; then (cd "$REPO" && git ls-files -z | rsync -a --files-from=- --from0 ./ "$1"/)
  else (cd "$REPO" && git archive "$2") | tar -x -C "$1"; fi
}
prepare_env() {  # <gpu index> <code dir> <tmp name>: one GPU, offline weights, a private TMPDIR
  local v
  for v in $(env | grep -o '^CLAUDE[^=]*'); do unset "$v"; done   # the claude CLI judge must not see a parent session
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 PYTHONPATH=$2
  export TMPDIR=$SCRATCH/tmp_$3; mkdir -p "$TMPDIR"
}
run_arms() {  # <log> <run dir> <common flags> <kind:budget ...>: each arm once; its report copy marks it done
  local log=$1 run=$2 common=$3 a k b arm name flags rc; shift 3
  for a in "$@"; do
    k=${a%:*}; b=${a#*:}; arm=${k}_${b}; name=$(report_name "$k"); flags="$(sched_flags "$k") $(budget_flags "$b")"
    if [ -f "$run/probing/$name.$arm.json" ]; then echo "SKIP $arm (report copy exists)" >> "$log"; continue; fi
    echo "ARM $arm start $(date '+%m-%d %H:%M'): $flags" >> "$log"
    $PY -u -m evalrx.benchmark.run $common $flags >> "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$run/probing/$name.json" ]; then echo "ARM $arm FAILED rc=$rc at $(date '+%m-%d %H:%M')" >> "$log"; return 1; fi
    cp "$run/probing/$name.json" "$run/probing/$name.$arm.json"; cp "$run/probing/$name.md" "$run/probing/$name.$arm.md"
    echo "ARM $arm done $(date '+%m-%d %H:%M')" >> "$log"
  done
}
validate_union() {  # <log> <arm names json> <common flags>: one Stage 0 (plus the source's), then the union validation
  echo "VALIDATE start $(date '+%m-%d %H:%M')" >> "$1"
  $PY -u "$HERE/validate_arms.py" "$2" -- $3 --triage-validate --triage-validation-source "$SRC" >> "$1" 2>&1
  echo "EXIT validate $? at $(date '+%m-%d %H:%M')" >> "$1"
}
keep_aside() {  # <run dir> <suffix>: copy the current validation report and frozen family aside, once
  local v=$1/validation
  if [ -f "$v/validation_report.json" ] && [ ! -f "$v/validation_report.$2.json" ]; then
    cp "$v/validation_report.json" "$v/validation_report.$2.json"; cp "$v/validation_report.md" "$v/validation_report.$2.md"
    cp "$v/frozen.json" "$v/frozen.$2.json"
  fi
}
