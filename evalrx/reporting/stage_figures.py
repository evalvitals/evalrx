"""The figures M2 and M3 cite: logged by the run, rendered by the explore step,
or planned in its snapshot report.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _m2_figures(
    raw: Mapping[str, Any], m2: Mapping[str, Any], explore: Mapping[str, Any],
    logs_dir: Path, root: Path,
) -> list[dict[str, Any]]:
    # An explore output keeps its figures in figures/ beside its report.
    explore_dir = raw.get("explore_dir")
    figures = _explore_figures(explore, Path(explore_dir) / "figures", root) if explore_dir else []
    if not figures:
        # A loop run: the logger copied the explorer's figures into
        # M2/artifacts/ under hashed names and kept the original name of each
        # beside the copy; that name is what the takeaways cite, so the join
        # runs on it.
        figures = _snapshot_figures(explore, logs_dir, root)
    if not figures:
        # No explore report to join against: the explore and analysis events
        # still name their figures as paths under logs/.
        return _logged_figures(
            [*(explore.get("figures") or []), *(m2.get("figures") or [])], logs_dir, root)
    # The analysis step's own chart sits beside the explorer's, as it did
    # when both came off the events.
    known = {figure["path"] for figure in figures}
    return figures + [
        figure for figure in _logged_figures(m2.get("figures") or [], logs_dir, root)
        if figure["path"] not in known
    ]


def _visual_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(str(value or "")).stem.lower())


def _logged_figures(paths: Iterable[Any], logs_dir: Path, root: Path) -> list[dict[str, Any]]:
    """Figures the run logged by path (relative to logs/), that exist on disk."""
    result: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for item in paths:
        if not isinstance(item, str) or not item.lower().endswith(".png"):
            continue
        candidate = (logs_dir / item).resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        stem = re.sub(r"_[0-9a-f]{8}$", "", candidate.stem)   # the content-hash suffix V2 adds
        result.append({
            "id": candidate.stem, "title": stem.replace("_", " ").strip().title(),
            "question": "", "path": os.path.relpath(candidate, root), "reading": "",
            "do_not_infer": "No agent-authored interpretation was saved for this figure.",
            "disposition": "supporting", "not_promoted_reason": "",
        })
    return result[:30]


#: The ordering prefix the host puts on the charts it renders
#: (``00_class_balance.png`` for the plan's ``class_balance``).
_ORDER_PREFIX = re.compile(r"^\d+_")


_EXPLORE_FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".svg")


def _explore_figures(explore: Mapping[str, Any], figure_dir: Path, root: Path) -> list[dict[str, Any]]:
    """The explorer's figures from the figures/ directory of an explore output."""
    files = sorted(figure_dir.glob("*.png")) if figure_dir.is_dir() else []
    return _planned_figures(explore, [(path.name, path) for path in files], root)


def _snapshot_figures(explore: Mapping[str, Any], logs_dir: Path, root: Path) -> list[dict[str, Any]]:
    """The explorer's figures as RunLoggerV2 copied them into M2/artifacts/.

    ``media_files`` (see ``RunLoggerV2._inline_workspace``) pairs each sandbox
    file -- ``figures/00_class_balance.png`` -- with its content-hashed copy
    under logs/. The sandbox name is the one the takeaways cite and the visual
    plan is keyed by; the copy is only where the bytes are. So the join runs
    on the name and a figure's id here is a chart name, never a hash stem.
    """
    pairs: list[tuple[str, Path]] = []
    for name, copied in (explore.get("media_files") or {}).items():
        if not isinstance(name, str) or not isinstance(copied, str):
            continue
        if not name.lower().endswith(_EXPLORE_FIGURE_SUFFIXES):
            continue
        candidate = (logs_dir / copied).resolve()
        if candidate.is_file():
            pairs.append((Path(name).name, candidate))
    pairs.sort(key=lambda pair: pair[0])
    return _planned_figures(explore, pairs, root)


def _planned_figures(
    explore: Mapping[str, Any], files: Sequence[tuple[str, Path]], root: Path,
) -> list[dict[str, Any]]:
    """Figure records for *files*, each ``(name the explorer used, path on disk)``.

    A file the visual plan names gets the plan's reader-facing title and
    question and the chart reading written for it; its id is the plan's name,
    which is also what the takeaways cite. A file the plan does not name is
    still shown, under its own stem minus the ordering prefix -- again the
    name a takeaway would cite.
    """
    plans = [item for item in explore.get("visual_plan") or [] if isinstance(item, Mapping)]
    readings = [item for item in explore.get("chart_readings") or [] if isinstance(item, Mapping)]
    result: list[dict[str, Any]] = []
    used: set[Path] = set()
    ids: set[str] = set()

    def unique(candidate: str) -> str:
        chosen, n = candidate, 2
        while chosen in ids:
            chosen = f"{candidate}_{n}"
            n += 1
        ids.add(chosen)
        return chosen

    for plan in plans:
        name = str(plan.get("name") or "")
        key = _visual_key(name)
        if not key:
            continue
        # An exact match after the ordering prefix comes off wins; only then a
        # file whose name merely contains the plan's. Otherwise
        # ``failrate_by_answer_length`` takes ``..._band``'s file whenever that
        # one sorts first.
        match = next((path for fname, path in files
                      if path not in used and _visual_key(_ORDER_PREFIX.sub("", fname)) == key), None)
        if match is None:
            match = next((path for fname, path in files
                          if path not in used and key in _visual_key(fname)), None)
        if match is None:
            continue
        used.add(match)
        reading = next((item for item in readings if _visual_key(item.get("chart")) == key), {})
        result.append({
            "id": unique(name), "title": str(plan.get("display_name") or name.replace("_", " ").title()),
            "question": str(plan.get("question") or ""), "path": os.path.relpath(match, root),
            "reading": str(reading.get("reading") or ""),
            "do_not_infer": str(reading.get("do_not_infer") or ""),
            "disposition": str(plan.get("disposition") or "supporting"),
            "not_promoted_reason": str(plan.get("not_promoted_reason") or ""),
        })
    for fname, path in files:
        if path in used:
            continue
        stem = _ORDER_PREFIX.sub("", Path(fname).stem)
        result.append({
            "id": unique(stem), "title": stem.replace("_", " ").title(),
            "question": "", "path": os.path.relpath(path, root), "reading": "",
            "do_not_infer": "No agent-authored interpretation was saved for this figure.",
            "disposition": "audit", "not_promoted_reason": "Unlinked exploratory artifact",
        })
    return result[:30]
