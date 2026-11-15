"""Generate (or check) the registry-derived tables in docs/analyzers.md.

    python tools/gen_analyzer_docs.py            # rewrite the tables in place
    python tools/gen_analyzer_docs.py --check    # exit 1 if the doc is stale

The page keeps its hand-written prose; only regions between marker comments are
generated::

    <!-- BEGIN GENERATED: analyzer-index -->        category overview with counts
    <!-- BEGIN GENERATED: analyzers/<package> -->   one table per analyzers/ subpackage
    <!-- BEGIN GENERATED: models/<group> -->        one table per model group
    <!-- END GENERATED -->

Inside a table, the columns the code knows are regenerated on every run: an
analyzer's Key, Requires, Modality and Status; a model's Key, Family, Modalities
and Traits.  The columns only a person can write -- an analyzer's display name and
Paper, a model's Notes -- are carried over from the existing row with the same
key, wherever on the page it sits, so rows can be curated and reordered by hand.
A newly registered key gets a placeholder row appended to its section (the run
lists it so the placeholder gets curated); a key that is no longer registered
loses its row.  A registered key whose package or group has no marked section is
an error: where it belongs on the page, and the prose around it, is a decision
this script does not make.

Needs the full install (torch + numpy): without them the attention analyzers do
not register, and a table generated from that registry would silently drop them.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import inspect
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "analyzers.md"
REGEN_CMD = "python tools/gen_analyzer_docs.py"

_BLOCK = re.compile(
    r"(?P<begin><!-- BEGIN GENERATED: (?P<name>[\w./-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- END GENERATED -->)",
    re.DOTALL,
)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_KEY_CELL = re.compile(r"^`([^`]+)`$")

_MODALITY_ORDER = ("text", "image", "audio", "video")
_NONE = "—"

ANALYZER_HEADER = (
    "| Analyzer | Key | Requires | Modality | Paper | Status |\n"
    "|---|---|---|---|---|---|\n"
)
MODEL_HEADER = (
    "| Key | Family | Modalities | Traits | Notes |\n"
    "|---|---|---|---|---|\n"
)
INDEX_HEADER = (
    "| Category | Package | Analyzers | Stubs |\n"
    "|---|---|---|---|\n"
)

#: Model groups, in the order a missing-section error lists them.
MODEL_GROUPS = ("text", "vision", "omni", "api")


# ----------------------------------------------------------------------
# What the code says
# ----------------------------------------------------------------------

def _load_registries():
    """Import evalrx (registers everything) and return (analyzers, model specs)."""
    sys.path.insert(0, str(ROOT))
    import evalrx  # noqa: F401  -- side effect: analyzers + models self-register
    from evalrx import analyzers, registry
    from evalrx.specs import REGISTRY

    if analyzers.attention is None:
        raise SystemExit(
            "the attention analyzers did not register (torch/numpy not importable), so the "
            "generated tables would silently drop them -- run this from the full install"
        )
    return registry.analyzers.all(), dict(REGISTRY)


def analyzer_package(cls) -> str:
    """``evalrx.analyzers.<package>.<module>`` -> ``<package>``."""
    parts = cls.__module__.split(".")
    if parts[:2] != ["evalrx", "analyzers"] or len(parts) < 4:
        raise SystemExit(f"{cls.__name__} lives outside evalrx/analyzers/<package>/: {cls.__module__}")
    return parts[2]


def is_stub(cls) -> bool:
    """True when ``_run`` does nothing but ``raise NotImplementedError`` (Stage-2 stubs)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls._run)))
    body = [
        stmt for stmt in tree.body[0].body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    exc = body[0].exc.func if isinstance(body[0].exc, ast.Call) else body[0].exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def _ordered(modalities) -> list[str]:
    known = [m for m in _MODALITY_ORDER if m in modalities]
    return known + sorted(set(modalities) - set(_MODALITY_ORDER))


def requires_cell(cls) -> str:
    parts = [f"`{cap.name}`" for cap in sorted(cls.requires, key=lambda c: c.name)]
    if getattr(cls, "requires_trajectories", False):
        parts.append("trajectories")
    return " + ".join(parts) or _NONE


def modality_cell(cls) -> str:
    """``applies_to_modalities``; slots in ``requires_modalities`` are bold."""
    required = set(getattr(cls, "requires_modalities", ()) or ())
    return " + ".join(
        f"**{m}**" if m in required else m for m in _ordered(cls.applies_to_modalities)
    )


def default_display_name(cls) -> str:
    """Placeholder name for a new row: ``CoverageVerificationGap`` -> ``Coverage verification gap``."""
    stem = re.sub(r"Analyzer$", "", cls.__name__) or cls.__name__
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", stem)
    if not words:
        return stem
    return " ".join([words[0]] + [w if w.isupper() else w.lower() for w in words[1:]])


def model_group(spec) -> str:
    if spec.api_only:
        return "api"
    if spec.modalities & {"audio", "video"}:
        return "omni"
    if "image" in spec.modalities:
        return "vision"
    return "text"


def model_traits(spec) -> str:
    traits = [
        label for flag, label in (
            (spec.is_moe, "MoE"),
            (spec.is_reasoning, "reasoning"),
            (spec.tool_calling, "tool calling"),
            (spec.jax is not None, "JAX"),
        ) if flag
    ]
    return ", ".join(traits) or _NONE


# ----------------------------------------------------------------------
# What the page says
# ----------------------------------------------------------------------

def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rows(body: str, key_col: int) -> list[tuple[str, list[str]]]:
    """``(key, cells)`` for every data row of the table in *body*."""
    rows = []
    for line in body.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _cells(line)
        match = _KEY_CELL.match(cells[key_col]) if len(cells) > key_col else None
        if match:  # header and separator rows have no backticked key
            rows.append((match.group(1), cells))
    return rows


def _slug(title: str) -> str:
    """Heading anchor as Python-Markdown's toc (MkDocs) and GitHub both build it for plain titles."""
    text = re.sub(r"[^\w\s-]", "", title).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def _heading_before(text: str, pos: int) -> str | None:
    headings = [m.group(1) for m in _HEADING.finditer(text, 0, pos)]
    return headings[-1] if headings else None


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def render(text: str, *, require_all: bool = True) -> tuple[str, list[str]]:
    """Return ``(regenerated page, keys that got a placeholder row)``.

    With *require_all*, a registered analyzer package or model group that has no
    marked section on the page raises ``ValueError`` instead of being skipped.
    """
    analyzers, specs = _load_registries()
    blocks = list(_BLOCK.finditer(text))

    # Curated cells by key, collected page-wide so a row moved between sections keeps them.
    curated_analyzers: dict[str, list[str]] = {}
    curated_models: dict[str, list[str]] = {}
    for block in blocks:
        name = block.group("name")
        if name.startswith("analyzers/"):
            curated_analyzers.update(_rows(block.group("body"), key_col=1))
        elif name.startswith("models/"):
            curated_models.update(_rows(block.group("body"), key_col=0))

    by_package: dict[str, dict[str, type]] = {}
    for key, cls in analyzers.items():
        by_package.setdefault(analyzer_package(cls), {})[key] = cls
    by_group: dict[str, dict[str, object]] = {}
    for key, spec in specs.items():
        by_group.setdefault(model_group(spec), {})[key] = spec

    sections = {b.group("name") for b in blocks}
    missing = [f"analyzers/{p}" for p in sorted(by_package) if f"analyzers/{p}" not in sections]
    missing += [f"models/{g}" for g in MODEL_GROUPS if g in by_group and f"models/{g}" not in sections]
    unknown = [
        s for s in sorted(sections)
        if (s.startswith("analyzers/") and s.split("/", 1)[1] not in by_package)
        or (s.startswith("models/") and s.split("/", 1)[1] not in by_group)
        or not (s.startswith(("analyzers/", "models/")) or s == "analyzer-index")
    ]
    if require_all and (missing or unknown):
        problems = []
        if missing:
            problems.append(
                "registered but no section on the page (add a heading, a sentence, and an empty "
                "'<!-- BEGIN GENERATED: <name> -->' / '<!-- END GENERATED -->' pair): "
                + ", ".join(missing)
            )
        if unknown:
            problems.append("section matches nothing registered: " + ", ".join(unknown))
        raise ValueError("; ".join(problems))

    added: list[str] = []
    index_rows: list[str] = []

    def analyzer_table(package: str, body: str) -> str:
        registered = by_package.get(package, {})
        order = [k for k, _ in _rows(body, key_col=1) if k in registered]
        order += sorted(k for k in registered if k not in order)
        lines = [ANALYZER_HEADER]
        for key in order:
            cls = registered[key]
            if key in curated_analyzers:
                cells = curated_analyzers[key]
                name, paper = cells[0], (cells[4] if len(cells) > 4 else _NONE)
            else:
                name, paper = default_display_name(cls), _NONE
                added.append(f"analyzers/{package}: {key}")
            status = "stub" if is_stub(cls) else "✓"
            lines.append(
                f"| {name} | `{key}` | {requires_cell(cls)} | {modality_cell(cls)} "
                f"| {paper} | {status} |\n"
            )
        return "".join(lines)

    def model_table(group: str, body: str) -> str:
        registered = by_group.get(group, {})
        order = [k for k, _ in _rows(body, key_col=0) if k in registered]
        order += sorted(k for k in registered if k not in order)
        lines = [MODEL_HEADER]
        for key in order:
            spec = registered[key]
            if key in curated_models:
                cells = curated_models[key]
                notes = cells[4] if len(cells) > 4 and cells[4] else _NONE
            else:
                notes = _NONE
                added.append(f"models/{group}: {key}")
            modalities = " + ".join(_ordered(spec.modalities))
            lines.append(
                f"| `{key}` | `{spec.family}` | {modalities} | {model_traits(spec)} | {notes} |\n"
            )
        return "".join(lines)

    # Pass 1: every table except the index (the index counts what pass 1 renders).
    out: list[str] = []
    cursor = 0
    index_at: int | None = None
    for block in blocks:
        name = block.group("name")
        out.append(text[cursor:block.start("body")])
        if name.startswith("analyzers/"):
            package = name.split("/", 1)[1]
            out.append(analyzer_table(package, block.group("body")))
            registered = by_package.get(package, {})
            title = _heading_before(text, block.start()) or package
            stubs = sum(is_stub(cls) for cls in registered.values())
            index_rows.append(
                f"| [{title}](#{_slug(title)}) | `{package}/` | {len(registered)} | {stubs} |\n"
            )
        elif name.startswith("models/"):
            out.append(model_table(name.split("/", 1)[1], block.group("body")))
        else:  # analyzer-index: filled in below
            index_at = len(out)
            out.append("")
        cursor = block.start("end")
    out.append(text[cursor:])

    if index_at is not None:
        documented = [cls for pkg in by_package.values() for cls in pkg.values()]
        total = (
            f"| **Total** | | **{len(documented)}** | "
            f"**{sum(is_stub(cls) for cls in documented)}** |\n"
        )
        out[index_at] = INDEX_HEADER + "".join(index_rows) + total
    return "".join(out), added


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", type=Path, default=DOC, help="page to regenerate (default: %(default)s)")
    ap.add_argument(
        "--check", action="store_true",
        help="write nothing; exit 1 if regenerating would change the page (for CI)",
    )
    args = ap.parse_args(argv)

    text = args.doc.read_text(encoding="utf-8")
    try:
        fresh, added = render(text)
    except ValueError as e:
        print(f"{args.doc}: {e}", file=sys.stderr)
        return 1

    if args.check:
        if fresh == text:
            print(f"{args.doc}: up to date")
            return 0
        diff = difflib.unified_diff(
            text.splitlines(keepends=True), fresh.splitlines(keepends=True),
            fromfile=f"{args.doc} (on disk)", tofile=f"{args.doc} (from registry)",
        )
        sys.stderr.writelines(diff)
        print(f"\n{args.doc} is out of date -- regenerate with: {REGEN_CMD}", file=sys.stderr)
        return 1

    if fresh == text:
        print(f"{args.doc}: up to date")
        return 0
    args.doc.write_text(fresh, encoding="utf-8")
    print(f"{args.doc}: updated")
    if added:
        print("placeholder rows added -- fill in the hand-written cells (name/Paper, or Notes):")
        for entry in added:
            print(f"  {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
