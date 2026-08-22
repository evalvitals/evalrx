"""Shared code for ``examples/benchmark`` — one entry point for every
(modality, family, size, dataset) cell.

Layout::

    _common/run.py       CLI: --modality vlm|llm|alm --model <size> --dataset <name>
    _common/models.py    the support matrix: size key -> spec key per modality
    _common/tasks/       datasets: download -> manifest.json, scorer, protocol, pinned M1
    _common/scoring.py   the answer parsers/graders the task kinds share
    _common/runner.py    discovery -> M1..M5 -> M4 -> fix wiring (all modalities)

Run it as a module from ``examples/benchmark`` (``python -m _common.run``) or
with ``PYTHONPATH=/app/examples/benchmark`` inside the images — never as a
script from inside ``_common/`` (that would put ``_common/tasks`` on
``sys.path[0]`` and shadow nothing today, but keep the one rule).
"""
