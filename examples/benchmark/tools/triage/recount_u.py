"""Re-read a run's stored U evidence under the exact and the tolerant agreement rule (no model, no GPU).

usage: python recount_u.py <run dir>

Reads the self_consistency and prompt_reword observations of the run's probing/ and validation/ ledgers and
re-derives every case's U check with ``triage._u_check`` twice: with key equality (``exact_answer``, the rule up
to rules-4) and with the task's tolerant rule (``tasks.same_answer``, rules-5). Prints the fired/clear/untested
tallies per label and the FAIL cases whose U state flips. The case labels come from the run's baseline.json
(and baseline_validation_source.json for the validation ledger); the task metadata is ChartQA's.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from evalrx.benchmark import tasks
from evalrx.eval_agent.probing.actions import Observation
from evalrx.eval_agent.probing.triage import _u_check, exact_answer

D = Path(sys.argv[1])
META = {"task": "exact_or_numeric", "numeric_tolerance": 0.05}


def cases_of(*files):
    out = {}
    for f in files:
        if not (D / f).is_file():
            continue
        for r in json.load(open(D / f))["cases"]:
            out[r["id"]] = SimpleNamespace(id=r["id"], observed=r["observed"], label=r["label"], metadata=META)
    return out


def chunks(sub):
    root = D / sub / "observations"
    idx = [json.loads(line) for line in open(root / "index.jsonl")]
    by = {"self_consistency": [], "prompt_reword": []}
    for r in dict.fromkeys(x["action_id"] for x in idx):
        o = Observation.from_dict(json.load(open(root / f"{r}.json")))
        if o.status != "ok":
            continue
        if o.analyzer == "self_consistency":
            by["self_consistency"].append(o)
        elif o.analyzer == "prompt_contrast" and "question_frame" in (o.params.get("strategies") or {}):
            by["prompt_reword"].append(o)
    return by


def for_case(objs, cid):
    for o in reversed(objs):
        if cid in o.planned_ids:
            return o
    return None


def recount(sub, cases):
    by = chunks(sub)
    covered = sorted({c for o in by["self_consistency"] for c in o.planned_ids}
                     | {c for o in by["prompt_reword"] for c in o.planned_ids})
    tally = {}
    for rule, agree in (("exact", exact_answer), ("tolerant", tasks.same_answer)):
        t = {lab: Counter() for lab in ("fail", "pass")}
        for cid in covered:
            case = cases.get(cid)
            if case is None:
                continue
            u = _u_check(for_case(by["self_consistency"], cid), for_case(by["prompt_reword"], cid), case,
                         tasks.answer_key, agree)
            lab = t[case.label]
            lab["n"] += 1
            lab[f"U_{u['state']}"] += 1
            for part in u["parts"]:
                lab[f"{part['part']}_{part['state']}"] += 1
        tally[rule] = {k: dict(v) for k, v in t.items()}
    print(sub, json.dumps(tally, indent=1, sort_keys=True))
    flips = Counter()   # which FAIL cases change state between the two rules
    for cid in covered:
        case = cases.get(cid)
        if case is None or case.label != "fail":
            continue
        states = [_u_check(for_case(by["self_consistency"], cid), for_case(by["prompt_reword"], cid), case,
                           tasks.answer_key, agree)["state"] for agree in (exact_answer, tasks.same_answer)]
        flips[f"{states[0]}->{states[1]}"] += 1
    print(sub, "FAIL U transitions", dict(flips))


recount("probing", cases_of("baseline.json"))
recount("validation", cases_of("baseline.json", "baseline_validation_source.json"))
