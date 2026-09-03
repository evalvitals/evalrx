"""Operationalization bridge — compile a candidate's signal RECIPE into a frozen,
reproducible per-case signal function (case -> value).

This is the one channel that lets the rich, free-form discoveries of the LAMBDA
explorer enter the validated stats engine. A hypothesis ("small peripheral objects
fail more") is a sentence; it cannot be tested until it is pinned to a concrete,
deterministic per-case signal that lands in ``StatsInput.per_case`` and competes in
M2's e-BH family. ``compile_recipe`` performs that compilation.

Two recipe kinds (DESIGN §5):

- ``kind="expr"`` (PREFERRED): a RESTRICTED DSL evaluated over a row's already-known
  columns — interactions / thresholds / composite predicates reduced to one
  boolean/continuous function of existing signals. No codegen, deterministic,
  auditable. A safe AST walker (no calls except a tiny numeric whitelist, no
  attribute/subscript access, no imports) evaluates it; a case missing any
  referenced column is SKIPPED, never defaulted.
- ``kind="code"`` (FALLBACK): a brand-new continuous estimand that is not yet any
  known column. Deferred to a later Phase B step — it will reuse the leak-1
  sandbox + sufficient-statistics host-check path. Raises ``NotImplementedError``
  for now.

``compile_recipe`` returns ``{case_id -> value}``; the fused pipeline (Phase B2)
wraps that into a synthetic-analyzer ``findings["per_case"]`` entry so
``build_stats_input`` collects it like any analyzer signal.
"""

from __future__ import annotations

import ast
import logging
import operator
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# A recipe expr may NEVER reference the outcome/id columns: a signal that IS the
# label (or trivially derived from it) would perfectly "predict" FAIL — a
# self-prediction leak. records/per_case_to_records may carry these columns, so the
# DSL hard-rejects any expr that names one (defense regardless of what the rows hold).
_RESERVED_RECIPE_NAMES = frozenset({
    "label", "labels", "case_id", "id", "sample_id",
    "is_fail", "is_correct", "outcome", "y_true", "target", "gold",
})

# ---------------------------------------------------------------------------
# Recipe data contract
# ---------------------------------------------------------------------------


class RecipeError(ValueError):
    """A recipe is structurally invalid (bad syntax / disallowed construct)."""


@dataclass
class SignalRecipe:
    """A deterministic recipe for a new per-case signal (carries NO e-value).

    Attributes:
        name:           per-case signal key, e.g. ``"explored.small_and_peripheral"``.
        description:    mechanism language (read by M3), not used for computation.
        kind:           ``"expr"`` | ``"code"``.
        expr:           ``kind="expr"`` — restricted DSL over existing columns,
                        e.g. ``"(obj_size < 40) and (attention_focus_share < 0.3)"``.
        code:           ``kind="code"`` — sandbox source (deferred).
        suggested_test: catalog tool to route to, e.g. ``"signal_label_assoc"``.
    """

    name: str
    description: str = ""
    kind: str = "expr"
    expr: str = ""
    code: str = ""
    suggested_test: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalRecipe":
        """Build from the ``recipe`` dict an explorer candidate may carry."""
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            kind=str(data.get("kind", "expr")),
            expr=str(data.get("expr", "")),
            code=str(data.get("code", "")),
            suggested_test=str(data.get("suggested_test", "")),
        )


# ---------------------------------------------------------------------------
# Safe expression DSL (kind="expr")
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
# The ONLY callables a recipe may invoke. All pure, total, side-effect free.
_FUNCS = {"abs": abs, "min": min, "max": max, "float": float, "int": int, "len": len}

# AST node types the validator permits (anything else is rejected outright).
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.BinOp, ast.UnaryOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Name, ast.Load,
    ast.Constant,
    *(_BIN_OPS), *(_CMP_OPS), *(_UNARY_OPS),
)


def _validate(tree: ast.AST) -> None:
    """Reject any construct outside the safe subset (calls, attributes, imports…)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
                raise RecipeError("only abs/min/max/float/int/len calls are allowed")
            if node.keywords:
                raise RecipeError("keyword arguments are not allowed in a recipe expr")
            continue
        if not isinstance(node, _ALLOWED_NODES):
            raise RecipeError(f"disallowed expression construct: {type(node).__name__}")


def _free_names(tree: ast.AST) -> list[str]:
    """Column identifiers the expr references (Names that are not whitelisted funcs)."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _FUNCS and node.id not in names:
            names.append(node.id)
    return names


def _ev(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _ev(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_ev(node.operand, env))
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](_ev(node.left, env), _ev(node.right, env))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for v in node.values:
                result = _ev(v, env)
                if not result:
                    return result
            return result
        result = False
        for v in node.values:
            result = _ev(v, env)
            if result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, env)
            if not _CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _ev(node.body, env) if _ev(node.test, env) else _ev(node.orelse, env)
    if isinstance(node, ast.Call):
        args = [_ev(a, env) for a in node.args]
        return _FUNCS[node.func.id](*args)  # type: ignore[attr-defined]
    raise RecipeError(f"cannot evaluate node: {type(node).__name__}")


def _to_float(value: Any) -> float | None:
    """Coerce a recipe result to a signal value; None if it is not numeric."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

_ID_KEYS = ("case_id", "id", "sample_id")


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _row_id(row: Any, id_col: str, index: int) -> str:
    for key in (id_col, *_ID_KEYS):
        v = _row_get(row, key)
        if v not in (None, ""):
            return str(v)
    return str(index)


def compile_recipe(
    recipe: SignalRecipe,
    records: Any,
    *,
    id_col: str = "case_id",
) -> dict[str, float]:
    """Compile *recipe* into ``{case_id -> value}`` over *records*.

    Raises :class:`RecipeError` for a structurally invalid ``expr`` (bad syntax or
    a disallowed construct) — fail loud at compile time. A per-row failure (missing
    column, type mismatch, non-numeric result) SKIPS that case rather than aborting,
    so a recipe that only partially applies still yields a usable (smaller) signal;
    an empty result tells the caller the recipe could not be operationalized.
    """
    if recipe.kind == "code":
        raise NotImplementedError(
            "SignalRecipe kind='code' (sandbox codegen) is deferred to a later "
            "Phase B step; use kind='expr' for now"
        )
    if recipe.kind != "expr":
        raise RecipeError(f"unknown recipe kind: {recipe.kind!r}")

    expr = (recipe.expr or "").strip()
    if not expr:
        raise RecipeError("recipe.expr is empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise RecipeError(f"invalid expr syntax: {exc}") from exc
    _validate(tree)
    names = _free_names(tree)

    # Hard leak guard (G4): a recipe must be a function of PREDICTOR signals only,
    # never the outcome/id. Referencing the label column would make the signal the
    # label itself — a guaranteed double-dip self-prediction.
    reserved = _RESERVED_RECIPE_NAMES | {id_col}
    leaked = sorted(n for n in names if n in reserved)
    if leaked:
        raise RecipeError(
            f"recipe expr may not reference reserved outcome/id column(s): {leaked}"
        )

    out: dict[str, float] = {}
    for i, row in enumerate(records or []):
        env: dict[str, Any] = {}
        missing = False
        for n in names:
            v = _row_get(row, n)
            if v is None:
                missing = True
                break
            env[n] = v
        if missing:
            continue
        try:
            value = _ev(tree.body, env)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        fv = _to_float(value)
        if fv is not None:
            out[_row_id(row, id_col, i)] = fv
    return out


def compile_recipes(
    recipes: "list[SignalRecipe]",
    records: Any,
    *,
    id_col: str = "case_id",
) -> dict[str, dict[str, float]]:
    """Compile each expr recipe; skip (do not abort) recipes that fail to compile.

    Returns ``{recipe.name -> {case_id -> value}}`` for the recipes that produced a
    non-empty signal. Recipes that raise :class:`RecipeError` or yield an empty map
    are dropped here; the caller routes them to ``recommended_confirmatory_tests``.
    """
    out: dict[str, dict[str, float]] = {}
    for r in recipes:
        if r.kind != "expr":
            continue
        if r.name in out:
            # Duplicate recipe name would silently clobber the first signal — keep
            # the first, drop the rest, and say so rather than corrupt the family.
            logger.warning(
                "compile_recipes: duplicate recipe name %r — keeping the first, "
                "dropping the duplicate", r.name,
            )
            continue
        try:
            values = compile_recipe(r, records, id_col=id_col)
        except (RecipeError, NotImplementedError):
            continue
        if values:
            out[r.name] = values
    return out


def per_case_finding(value_maps: dict[str, dict[str, float]], *, id_key: str = "case_id") -> list[dict[str, Any]]:
    """Assemble compiled signals into ``findings["per_case"]`` entries.

    Merges several ``{case_id -> value}`` maps into one row-per-case list so a
    synthetic analyzer Result carries the bridged signals; ``build_stats_input``
    then keys them as ``"<analyzer>.<signal>"`` like any other per-case finding.
    """
    rows: dict[str, dict[str, Any]] = {}
    for signal_name, values in value_maps.items():
        for cid, val in values.items():
            rows.setdefault(cid, {id_key: cid})[signal_name] = val
    return list(rows.values())


# ---------------------------------------------------------------------------
# In-loop bridge: existing analyzer per_case signals -> records -> synthetic Result
# ---------------------------------------------------------------------------

def safe_ident(name: str) -> str:
    """Map a per-case signal key (e.g. ``"saliency.obj_size"``) to a DSL identifier
    (``"saliency_obj_size"``). Recipe ``expr`` references these sanitized names, since
    the dotted ``analyzer.metric`` keys are not valid Python identifiers."""
    s = re.sub(r"\W", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s


def per_case_to_records(
    per_case: dict[str, dict[str, float]],
    labels: dict[str, bool] | None = None,
    *,
    id_key: str = "case_id",
    label_key: str = "label",
) -> list[dict[str, Any]]:
    """Transpose ``{signal -> {case_id -> value}}`` into per-case row dicts.

    Signal keys are sanitized via :func:`safe_ident` so recipe exprs can reference
    them. Each row carries only the signals present for that case (missing ones are
    absent, so a recipe referencing a missing signal SKIPS that case). When *labels*
    is given, a ``pass``/``fail`` ``label`` column is added.

    ``safe_ident`` is not injective (``a.b`` and ``a_b`` both map to ``a_b``). When
    two DISTINCT signal keys collide on their sanitized identifier, the mapping is
    ambiguous, so BOTH are EXCLUDED (never first-wins-overwrite, which would silently
    test the wrong estimand) and a warning is logged. A recipe referencing the
    ambiguous name then finds nothing and is dropped — fail-closed, not corrupt.
    """
    # Identify safe_ident collisions among distinct signal keys; exclude them.
    ident_owner: dict[str, str] = {}
    collided: set[str] = set()
    for signal in per_case:
        ident = safe_ident(signal)
        prior = ident_owner.get(ident)
        if prior is not None and prior != signal:
            collided.add(ident)
        else:
            ident_owner[ident] = signal
    if collided:
        clashing = sorted(s for s in per_case if safe_ident(s) in collided)
        logger.warning(
            "per_case_to_records: %d signal key(s) collide under safe_ident and are "
            "EXCLUDED (ambiguous): %s", len(clashing), clashing,
        )

    case_ids: set[str] = set()
    for vals in per_case.values():
        case_ids.update(vals)
    if labels:
        case_ids.update(labels)

    records: list[dict[str, Any]] = []
    for cid in sorted(case_ids):
        row: dict[str, Any] = {id_key: cid}
        for signal, vals in per_case.items():
            ident = safe_ident(signal)
            if ident in collided:
                continue
            if cid in vals:
                row[ident] = vals[cid]
        if labels is not None and cid in labels:
            row[label_key] = "fail" if labels[cid] else "pass"
        records.append(row)
    return records


def bridge_recipes_to_result(
    recipes: "list[SignalRecipe]",
    probe_results: "dict[str, Any]",
    data: "Any | None" = None,
    *,
    model_repr: str = "",
    analyzer_name: str = "explored",
    id_col: str = "case_id",
) -> "Any | None":
    """Compile pre-registered recipes over existing analyzer per_case signals into a
    synthetic analyzer ``Result``, so bridged composite signals enter M2 through the
    standard ``findings["per_case"]`` contract (DESIGN §5.3).

    LEAK-FREE BY CONSTRUCTION: the recipes must be PRE-SPECIFIED (discovered
    out-of-band — e.g. on the fused pipeline's held-out split, or hand-authored —
    NOT chosen by peeking at *these* labels). Testing a frozen extractor here is then
    exactly like testing a pre-registered analyzer; this function does no discovery.

    Returns the synthetic ``Result`` (analyzer=*analyzer_name*), or ``None`` if no
    recipe compiled to a non-empty signal.
    """
    from evalrx.analysis.stats_tools import build_stats_input
    from evalrx.core.result import Result

    inp = build_stats_input(probe_results, data)
    # Compile over PREDICTOR signals only — never expose the label to recipe
    # compilation (a recipe is a function of signals, not the outcome).
    records = per_case_to_records(inp.per_case, labels=None, id_key=id_col)
    compiled = compile_recipes(recipes, records, id_col=id_col)
    if not compiled:
        return None
    return Result(
        analyzer=analyzer_name,
        model=model_repr,
        findings={"per_case": per_case_finding(compiled, id_key=id_col)},
        cases=data,
    )
