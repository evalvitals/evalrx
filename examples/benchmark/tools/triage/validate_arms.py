"""One Stage 0 (plus the validation source's), then the stored triage arms of a run directory validated together.

usage: python validate_arms.py '<json list of [arm, report stem]>' -- <evalrx.benchmark.run argv ... --triage-validate>

Every arm of a comparison runs ``evalrx.benchmark.run`` into the same run directory, each copying its report
aside as ``probing/<report stem>.<arm>.json`` (see ``_common.sh``). This driver then runs the benchmark entry
once more with two of the runner's steps replaced: the controller is not run (the first listed arm's stored
report stands in for it), and the runner's ``validate()`` call becomes one over the union of every listed
arm's frozen issues, so the arms share one family, one Bonferroni level and one validation ledger. The
``summary.json`` the runner then overwrites is kept as ``summary.before_validation.json``.
"""
import json
import shutil
import sys

import evalrx.eval_agent.probing as probing
import evalrx.eval_agent.probing.validation as validation
from evalrx.benchmark import runner
from evalrx.benchmark.run import main

ARMS = json.loads(sys.argv[1])
argv = sys.argv[sys.argv.index("--") + 1:]
real_validate = validation.validate
orig = runner._run_triage


def arms(args, task, resolved, spec, model, gen_kwargs, cases, ctx, run_dir, baseline, **kw):
    root = run_dir / "probing"
    reports = {arm: json.loads((root / f"{name}.{arm}.json").read_text()) for arm, name in ARMS}
    first = next(iter(reports.values()))
    if (run_dir / "summary.json").is_file():
        shutil.copy(run_dir / "summary.json", run_dir / "summary.before_validation.json")

    def stored(*_a, **_k):
        return first

    def union(model_, discovery, validation_, families_, reports_, **kw_):
        # the runner's own entry (the first arm under the scheduler's report name) stays, for its summary
        families = {**families_, **{arm: validation.freeze_issues(r) for arm, r in reports.items()}}
        return real_validate(model_, discovery, validation_, families, {**reports_, **reports}, **kw_)

    probing.run_triage = stored
    validation.validate = union
    try:
        return orig(args, task, resolved, spec, model, gen_kwargs, cases, ctx, run_dir, baseline, **kw)
    finally:
        validation.validate = real_validate


runner._run_triage = arms
sys.exit(main(argv))
