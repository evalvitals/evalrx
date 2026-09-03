"""Failure-mode clustering: group FAIL cases into interpretable clusters.

Complements the M2 exploratory agent's per-signal EDA with pattern discovery
over the raw failing cases themselves — the deterministic tier needs no LLM
and no required extra dependency (a pure-numpy fallback always works; install
the ``[cluster]`` extra for TF-IDF + hdbscan/Agglomerative clustering); an
optional LLM tier then names each cluster from its exemplars.
"""

from __future__ import annotations

import logging
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evalrx.agent_runtime.json_shape import validate_json_shape

if TYPE_CHECKING:
    from evalrx.core.model import Model

logger = logging.getLogger(__name__)

_DEFAULT_TEXT_COLS = ("prompt", "question", "input", "text", "output", "response", "observed")
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "to", "of", "in",
    "on", "for", "and", "or", "it", "this", "that", "with", "as", "at", "by",
    "from", "not", "no", "does", "did", "do", "has", "have", "had",
})


@dataclass
class FailureMode:
    """One cluster of FAIL cases."""

    name: str
    description: str
    case_ids: list[str] = field(default_factory=list)
    exemplars: list[dict[str, Any]] = field(default_factory=list)
    size: int = 0
    top_terms: list[str] = field(default_factory=list)
    # Boundary-aware induction (opt-in via cluster_failures(boundary_aware=True)):
    # each entry is {"fail": <row>, "nearest_pass": <row>, "similarity": float} —
    # a peripheral failing case paired with its nearest verified non-failure, so
    # an LLM can be shown *where* the failure boundary sits, not just what the
    # failing cases have in common. Empty when boundary_aware is off.
    boundary_pairs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "case_ids": self.case_ids,
            "exemplars": self.exemplars,
            "size": self.size,
            "top_terms": self.top_terms,
            "boundary_pairs": self.boundary_pairs,
        }


@dataclass
class FailureModeReport:
    """Output of :func:`cluster_failures`."""

    clusters: list[FailureMode] = field(default_factory=list)
    n_fail_cases: int = 0
    unclustered_ids: list[str] = field(default_factory=list)
    method: str = "none"
    named_by: str = "top_terms"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "n_fail_cases": self.n_fail_cases,
            "unclustered_ids": self.unclustered_ids,
            "method": self.method,
            "named_by": self.named_by,
            "params": self.params,
        }

    def as_hypothesis_context(self) -> str:
        """Compact section for the M3 hypothesis-proposal prompt."""
        if not self.clusters:
            return ""
        lines = [f"FAILURE MODES (clustered, method={self.method}):"]
        for c in self.clusters:
            lines.append(f"  - {c.name} (n={c.size}): {c.description}")
        return "\n".join(lines)


def _row_id(row: dict[str, Any], idx: int, id_key: str) -> str:
    cid = row.get(id_key)
    return str(cid) if cid not in (None, "") else f"row{idx}"


def _row_text(row: dict[str, Any], text_cols: "list[str] | None") -> str:
    cols = text_cols or [c for c in _DEFAULT_TEXT_COLS if c in row]
    parts = [str(row[c]) for c in cols if row.get(c) not in (None, "")]
    return " ".join(parts)


def _row_signal_flags(row: dict[str, Any], signal_cols: "list[str] | None", exclude: set) -> str:
    if signal_cols is not None:
        cols = signal_cols
    else:
        cols = [
            k for k, v in row.items()
            if k not in exclude and isinstance(v, (int, float, bool))
        ]
    return " ".join(f"{k}={row[k]}" for k in cols if k in row)


_OUTPUT_COLS = ("output", "response", "observed")


def _row_output(row: dict[str, Any]) -> str:
    for col in _OUTPUT_COLS:
        val = row.get(col)
        if val not in (None, ""):
            return str(val)
    return ""


def _error_signal_deterministic(row: dict[str, Any], expected_col: str) -> str:
    """Failure-aware fallback with no LLM: a crude but always-available mismatch
    signal ("expected X got Y"), so proximity in embedding space reflects *how*
    the model failed a bit more than raw prompt text alone would."""
    expected = row.get(expected_col)
    if expected in (None, ""):
        return ""
    return f"expected={str(expected)[:80]} got={_row_output(row)[:80]}"


def _error_signals_llm(
    fail_rows: list[tuple[int, dict[str, Any]]],
    expected_col: str,
    judge: "Model",
    *,
    max_repairs: int = 1,
) -> "list[str] | None":
    """One batched call describing the failure mechanism for every FAIL row.

    Returns a list aligned 1:1 with *fail_rows*, or None if the judge never
    produced a parseable response (caller falls back to the deterministic
    signal in that case).
    """
    from evalrx.analysis.prompts.failure_modes import ERROR_SIGNAL_PROMPT

    pairs_block = "\n".join(
        f"{i}: expected={str(row.get(expected_col))[:150]!r} got={_row_output(row)[:150]!r}"
        for i, (_, row) in enumerate(fail_rows)
    )
    prompt = ERROR_SIGNAL_PROMPT.format(pairs_block=pairs_block)
    schema = {
        "type": "array",
        "items": {"type": "object", "required": ["index", "error"]},
    }

    for _attempt in range(max_repairs + 1):
        raw = judge.generate(prompt)
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            import json

            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if validate_json_shape(data, schema):
            continue
        by_index = {int(item["index"]): str(item["error"]) for item in data if "index" in item}
        return [by_index.get(i, "") for i in range(len(fail_rows))]
    return None


def _compute_error_signals(
    fail_rows: list[tuple[int, dict[str, Any]]],
    *,
    expected_col: "str | None",
    error_fn: "Callable[[dict[str, Any]], str] | None",
    judge: "Model | None",
) -> list[str]:
    """Per-FAIL-row failure-mechanism strings, aligned with *fail_rows*.

    Dispatch order: injected *error_fn* > batched LLM (if *judge* + *expected_col*
    given) > deterministic expected-vs-output fallback > "" (no signal at all,
    i.e. today's topic-only clustering) when neither is configured.
    """
    if error_fn is not None:
        return [error_fn(row) for _, row in fail_rows]
    if not expected_col:
        return ["" for _ in fail_rows]
    if judge is not None:
        try:
            signals = _error_signals_llm(fail_rows, expected_col, judge)
        except Exception as exc:  # noqa: BLE001 — error-signal generation is best-effort
            logger.warning("cluster_failures: LLM error-signal generation failed: %s", exc)
            signals = None
        if signals is not None:
            return signals
    return [_error_signal_deterministic(row, expected_col) for _, row in fail_rows]


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _top_terms(cluster_texts: list[str], corpus_doc_freq: Counter, n_docs: int, top_k: int = 5) -> list[str]:
    """Rank words by in-cluster frequency weighted by corpus rarity (mini TF-IDF)."""
    tf: Counter = Counter()
    for text in cluster_texts:
        tf.update(set(_tokenize(text)))
    if not tf:
        return []
    import math

    scored = [
        (word, count * math.log((n_docs + 1) / (corpus_doc_freq.get(word, 0) + 1)))
        for word, count in tf.items()
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in scored[:top_k]]


def _hash_vectorize(texts: list[str], n_features: int = 256):
    """Pure-numpy fallback mirroring sklearn's TfidfVectorizer: word 1-2gram
    hashing, smoothed IDF weighting, L2-normalized rows.

    Word-level (2+ char tokens, like sklearn's default ``\\b\\w\\w+\\b``) plus
    IDF is what keeps the two tiers consistent: shared boilerplate ("expected",
    "got", …) is down-weighted so distinguishing content drives similarity, the
    same way the TF-IDF path does. A raw char-3gram count instead lets common
    character runs dominate, collapsing genuinely distinct failures together.
    """
    import numpy as np

    def _terms(text: str) -> list[str]:
        words = re.findall(r"\w\w+", text.lower())
        if not words:
            # No 2+ char word tokens (e.g. "q", punctuation): fall back to the
            # raw text as a single term so identical short texts still map to
            # identical non-zero vectors (and group) rather than empty rows.
            return [text.lower().strip() or " "]
        return words + [f"{words[k]} {words[k + 1]}" for k in range(len(words) - 1)]

    counts = np.zeros((len(texts), n_features), dtype=float)
    for i, text in enumerate(texts):
        for term in _terms(text):
            counts[i, zlib.crc32(term.encode("utf-8")) % n_features] += 1.0
    # Smoothed IDF (sklearn smooth_idf default): terms present in every doc keep
    # weight 1 rather than vanishing; rarer terms are up-weighted.
    df = (counts > 0).sum(axis=0)
    idf = np.log((1.0 + len(texts)) / (1.0 + df)) + 1.0
    X = counts * idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _n_clusters_for(n_items: int, min_cluster_size: int, max_clusters: int) -> int:
    return max(1, min(max_clusters, n_items // max(1, min_cluster_size)))


def _cluster_cosine_greedy(X, max_clusters: int, threshold: float = 0.25) -> list[int]:
    """Deterministic numpy fallback: greedily group by cosine similarity to a
    running cluster centroid, capped at max_clusters (excess assigned to the
    nearest existing cluster instead of growing further)."""
    import numpy as np

    labels: list[int] = []
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    for row in X:
        if centroids:
            sims = [float(row @ c) for c in centroids]
            best = int(np.argmax(sims))
        else:
            sims, best = [], -1
        if best >= 0 and (sims[best] >= threshold or len(centroids) >= max_clusters):
            centroids[best] = (centroids[best] * counts[best] + row) / (counts[best] + 1)
            counts[best] += 1
            labels.append(best)
        else:
            centroids.append(row.copy())
            counts.append(1)
            labels.append(len(centroids) - 1)
    return labels


def _cluster_labels(X, *, method: str, min_cluster_size: int, max_clusters: int) -> tuple[list[int], str]:
    """Return (labels, method_used). Tries the requested/auto method, falling
    back down the chain: hdbscan -> sklearn Agglomerative -> numpy cosine-greedy."""
    tried_auto = method == "auto"

    if method in ("auto", "hdbscan"):
        try:
            import hdbscan as _hdbscan

            labels = _hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster_size)).fit_predict(X)
            return list(int(x) for x in labels), "hdbscan"
        except ImportError:
            if not tried_auto:
                raise

    if method in ("auto", "agglomerative"):
        try:
            from sklearn.cluster import AgglomerativeClustering

            n_clusters = _n_clusters_for(len(X), min_cluster_size, max_clusters)
            model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
            labels = model.fit_predict(X)
            return list(int(x) for x in labels), "agglomerative"
        except ImportError:
            if not tried_auto:
                raise

    labels = _cluster_cosine_greedy(X, max_clusters)
    return labels, "cosine_greedy"


def _vectorize(texts: list[str]) -> tuple[Any, str]:
    """Try sklearn TF-IDF; fall back to the pure-numpy hashing vectorizer.

    Falls back on any failure (not just a missing import) — e.g. degenerate
    all-empty-text input raises sklearn's "empty vocabulary" ValueError,
    which the hashing vectorizer handles fine (an all-zero row).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        X = vec.fit_transform(texts).toarray()
        return X, "tfidf"
    except Exception:  # noqa: BLE001 — any vectorization failure falls back
        return _hash_vectorize(texts), "hashing"


def _rank_by_centroid_distance(
    idxs: list[int], X: Any, n_core: int, n_boundary: int
) -> tuple[list[int], list[int]]:
    """Split cluster member indices (into *X*) into (core, boundary): the
    *n_core* closest to the cluster centroid (most representative) and the
    *n_boundary* farthest (peripheral — nearest the failure/success edge)."""
    import numpy as np

    centroid = X[idxs].mean(axis=0)
    order = sorted(idxs, key=lambda i: float(np.linalg.norm(X[i] - centroid)))
    core = order[:n_core]
    boundary = order[-n_boundary:] if len(order) > n_core else []
    return core, boundary


def _boundary_contrast_pairs(
    boundary_idxs: list[int],
    X: Any,
    fail_rows: list[tuple[int, dict[str, Any]]],
    pass_rows: list[dict[str, Any]],
    X_pass: Any,
) -> list[dict[str, Any]]:
    """For each boundary FAIL index, pair it with its nearest PASS row (cosine
    similarity in the same vector space — rows are L2-normalized by both
    vectorizer tiers, so a dot product approximates cosine similarity)."""
    if not boundary_idxs or not pass_rows or len(X_pass) == 0:
        return []
    import numpy as np

    pairs = []
    for i in boundary_idxs:
        sims = X_pass @ X[i]
        best = int(np.argmax(sims))
        pairs.append({
            "fail": fail_rows[i][1],
            "nearest_pass": pass_rows[best],
            "similarity": float(sims[best]),
        })
    return pairs


def _name_clusters_with_llm(
    judge: "Model", clusters: list[FailureMode], *, max_repairs: int = 1
) -> bool:
    """Best-effort: ask *judge* to name/describe each cluster from its
    exemplars. Returns True on success (clusters updated in place)."""
    from evalrx.analysis.prompts.failure_modes import NAME_CLUSTERS_PROMPT

    blocks = []
    for i, c in enumerate(clusters):
        exemplar_texts = [str(e) for e in c.exemplars[:3]]
        block = (
            f"Cluster {i} (n={c.size}, top terms: {', '.join(c.top_terms)}):\n"
            + "\n".join(f"  - {t[:200]}" for t in exemplar_texts)
        )
        if c.boundary_pairs:
            contrast_lines = [
                f"  - FAIL: {str(bp['fail'])[:150]} | nearest PASS: {str(bp['nearest_pass'])[:150]}"
                for bp in c.boundary_pairs[:3]
            ]
            block += "\n  Boundary contrast (failure vs. nearest success):\n" + "\n".join(contrast_lines)
        blocks.append(block)
    prompt = NAME_CLUSTERS_PROMPT.format(clusters_block="\n\n".join(blocks))

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["cluster_id", "name", "description"],
            "properties": {"name": {"type": "string", "minLength": 1}},
        },
    }

    for _attempt in range(max_repairs + 1):
        raw = judge.generate(prompt)
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            import json

            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if validate_json_shape(data, schema):
            continue
        by_id = {int(item["cluster_id"]): item for item in data if "cluster_id" in item}
        for i, c in enumerate(clusters):
            item = by_id.get(i)
            if item:
                c.name = str(item["name"])
                c.description = str(item.get("description", c.description))
        return True
    return False


def cluster_failures(
    records: list[dict[str, Any]],
    *,
    outcome_col: str = "label",
    fail_value: str = "FAIL",
    text_cols: "list[str] | None" = None,
    signal_cols: "list[str] | None" = None,
    id_key: str = "case_id",
    judge: "Model | None" = None,
    min_cluster_size: int = 3,
    max_clusters: int = 8,
    n_exemplars: int = 3,
    method: str = "auto",
    expected_col: "str | None" = None,
    error_fn: "Callable[[dict[str, Any]], str] | None" = None,
    boundary_aware: bool = False,
    n_boundary: int = 2,
) -> FailureModeReport:
    """Cluster FAIL cases in *records* into interpretable failure modes.

    Two tiers, no required dependency:

    - Deterministic: TF-IDF (sklearn, if installed) or a pure-numpy word-1-2gram
      IDF hashing vectorizer over each case's text + active signal flags, then
      hdbscan / sklearn Agglomerative / a numpy cosine-greedy fallback —
      whichever is available, in that preference order (or force one via
      *method*: ``"hdbscan"``, ``"agglomerative"``, ``"cosine_greedy"``).
    - LLM (optional): when *judge* is given, one call names/describes each
      cluster from its exemplars; falls back to corpus-weighted top terms
      on a malformed response.

    *text_cols*/*signal_cols* default to auto-detected columns (common text
    field names; all numeric/boolean columns for signals). *outcome_col*/
    *fail_value* select which records count as FAIL (case-insensitive).

    Two opt-in, off-by-default extensions (ProbeLLM-style failure-mode
    synthesis) — both no-ops unless explicitly requested, so existing callers
    see byte-identical output:

    - *expected_col*/*error_fn*: fold a failure-*mechanism* signal into each
      FAIL case's vectorized text, so grouping reflects how the model failed
      rather than only what the prompt was about. *error_fn* (row -> str) is
      used verbatim when given; otherwise, if *judge* and *expected_col* are
      both set, one batched LLM call describes each mismatch, falling back to
      a deterministic "expected=... got=..." string on any judge/parse
      failure. With neither, behavior is unchanged (topic-only clustering).
    - *boundary_aware*: also collect non-FAIL rows as a contrast pool. For
      each cluster, the *n_boundary* cases farthest from the cluster centroid
      are paired with their nearest verified non-failure (by cosine
      similarity in the same vector space) into ``FailureMode.boundary_pairs``
      — surfaced to the LLM namer (when *judge* is given) to describe *where*
      the failure boundary sits, not just what the failing cases share.
    """
    params = {
        "outcome_col": outcome_col, "fail_value": fail_value,
        "min_cluster_size": min_cluster_size, "max_clusters": max_clusters,
    }
    fail_rows = [
        (i, r) for i, r in enumerate(records)
        if str(r.get(outcome_col, "")).strip().lower() == fail_value.strip().lower()
    ]
    if not fail_rows:
        return FailureModeReport(n_fail_cases=0, method="none", params=params)

    pass_rows: list[dict[str, Any]] = []
    if boundary_aware:
        pass_rows = [
            r for i, r in enumerate(records)
            if str(r.get(outcome_col, "")).strip().lower() != fail_value.strip().lower()
        ]

    ids = [_row_id(r, i, id_key) for i, r in fail_rows]
    exclude = {outcome_col, id_key, *(text_cols or [])}
    error_signals = _compute_error_signals(
        fail_rows, expected_col=expected_col, error_fn=error_fn, judge=judge,
    )
    texts = [
        (
            _row_text(r, text_cols) + " " + _row_signal_flags(r, signal_cols, exclude)
            + " " + err
        ).strip()
        for (_, r), err in zip(fail_rows, error_signals)
    ]

    if len(fail_rows) < max(2, min_cluster_size):
        exemplars = [r for _, r in fail_rows[:n_exemplars]]
        corpus_freq: Counter = Counter()
        for t in texts:
            corpus_freq.update(set(_tokenize(t)))
        cluster = FailureMode(
            name="all_failures", description="Too few FAIL cases to cluster meaningfully.",
            case_ids=ids, exemplars=exemplars, size=len(ids),
            top_terms=_top_terms(texts, corpus_freq, len(texts)),
        )
        report = FailureModeReport(
            clusters=[cluster], n_fail_cases=len(fail_rows),
            method="single_cluster", params=params,
        )
        if judge is not None:
            try:
                if _name_clusters_with_llm(judge, report.clusters):
                    report.named_by = f"llm:{judge!r}"
            except Exception as exc:  # noqa: BLE001 — naming is best-effort
                logger.warning("cluster_failures: LLM naming failed: %s", exc)
        return report

    pass_texts = [
        (_row_text(r, text_cols) + " " + _row_signal_flags(r, signal_cols, exclude)).strip()
        for r in pass_rows
    ]
    if pass_texts:
        X_combined, vec_method = _vectorize(texts + pass_texts)
        X, X_pass = X_combined[: len(texts)], X_combined[len(texts):]
    else:
        X, vec_method = _vectorize(texts)
        X_pass = X[:0]  # empty, same width — no contrast pool available

    labels, cluster_method = _cluster_labels(
        X, method=method, min_cluster_size=min_cluster_size, max_clusters=max_clusters
    )

    corpus_freq: Counter = Counter()
    for t in texts:
        corpus_freq.update(set(_tokenize(t)))

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)

    clusters: list[FailureMode] = []
    unclustered_ids: list[str] = []
    for label, idxs in sorted(by_label.items()):
        cluster_ids = [ids[i] for i in idxs]
        if label == -1:  # hdbscan noise
            unclustered_ids.extend(cluster_ids)
            continue
        cluster_texts = [texts[i] for i in idxs]
        top_terms = _top_terms(cluster_texts, corpus_freq, len(texts))
        core_idxs, boundary_idxs = _rank_by_centroid_distance(idxs, X, n_exemplars, n_boundary)
        clusters.append(FailureMode(
            name=f"cluster_{label}_" + ("_".join(top_terms[:2]) or "misc"),
            description=(
                f"{len(idxs)} FAIL case(s) sharing: {', '.join(top_terms)}"
                if top_terms else f"{len(idxs)} FAIL case(s) with no distinguishing terms."
            ),
            case_ids=cluster_ids,
            exemplars=[fail_rows[i][1] for i in core_idxs],
            size=len(idxs),
            top_terms=top_terms,
            boundary_pairs=(
                _boundary_contrast_pairs(boundary_idxs, X, fail_rows, pass_rows, X_pass)
                if boundary_aware else []
            ),
        ))

    report = FailureModeReport(
        clusters=clusters, n_fail_cases=len(fail_rows), unclustered_ids=unclustered_ids,
        method=cluster_method, params={**params, "vectorizer": vec_method},
    )

    if judge is not None and clusters:
        try:
            if _name_clusters_with_llm(judge, clusters):
                report.named_by = f"llm:{judge!r}"
        except Exception as exc:  # noqa: BLE001 — naming is best-effort
            logger.warning("cluster_failures: LLM naming failed: %s", exc)

    return report
