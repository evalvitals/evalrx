"""Row fetching for the LLM task slices: the HF datasets-server, with the hub
files as a fallback.

``requests`` and HF ``datasets`` are imported on use, so importing the task
registry needs neither.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from typing import Optional

from .specs import Spec

ROWS_API = "https://datasets-server.huggingface.co/rows"
#: server-side row filtering. A slice holding <1% of a split (SuperGPQA
#: Medicine/hard is 217 of 26,529) would need ~6,000 rows pulled and thrown
#: away to collect 50 through /rows; /filter also reports num_rows_total for
#: the FILTERED set, so the sampling windows spread across the slice itself
#: rather than across the split that contains it.
FILTER_API = "https://datasets-server.huggingface.co/filter"


def _get_rows(params: dict, timeout: int, retries: int = 5,
              url: str = ROWS_API) -> Optional[dict]:
    """One /rows (or /filter) call, retried through the rate limiter.

    The datasets-server 429s readily once a sweep is fetching in parallel with
    anything else. Without a retry a 429 is indistinguishable from an empty
    window: the sample quietly shrinks and the run still prints a band, which is
    the same silent-measurement-artifact failure the tag-rate column exists to
    catch. Returning None says "this window is unknown", not "this window is
    empty", so the caller can count it.
    """
    import requests

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code not in (429, 500, 502, 503, 504):
                return None
        except requests.RequestException:
            pass
        if attempt < retries - 1:
            time.sleep(min(30, 3 * 2 ** attempt))
    return None


def fetch_rows(spec: Spec, want: int, seed: int = 0, timeout: int = 60,
               n_windows: Optional[int] = None) -> list[dict]:
    """Pull rows through the datasets-server from windows spread over the split.

    Many of these splits are ordered by subset, task, or difficulty, so a head
    sample measures one slice while claiming to measure the set. Shuffling the
    order of 100-row pages does NOT fix that on its own: stopping as soon as
    enough rows are collected then takes them all from whichever one or two
    pages came first, which is still a single contiguous block.

    So: draw ``n_windows`` offsets spaced across the whole split (jittered, not
    page-aligned) and take an equal share from each. The result is a stratified
    cluster sample — better than a head, but still clustered, which is why the
    interval this feeds is called approximate.

    ``spec.row_filter`` is applied HERE rather than after the fetch, because the
    early-stop has to count rows that survive it. Counting raw rows instead
    would stop a 1-in-25 slice after four usable items and still report a band.
    """
    params = {
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "offset": 0,
        "length": 1,
    }
    # a server-side `where` makes every count below -- total, windows, stride --
    # refer to the SLICE, which is what the band is being claimed about
    url = ROWS_API
    if spec.where:
        params["where"] = spec.where
        url = FILTER_API
    meta = _get_rows(params, timeout, url=url)
    if meta is None:
        raise RuntimeError(f"datasets-server unavailable for {spec.dataset}")
    total = meta.get("num_rows_total", 0)
    if not total:
        return []

    def get_page(offset: int, length: int):
        payload = _get_rows({**params, "offset": offset, "length": length}, timeout, url=url)
        if payload is None:
            return None
        return [(entry.get("row_idx"), entry["row"]) for entry in payload.get("rows", [])]

    return _draw_windows(spec, want, seed, n_windows, total, get_page)


def _draw_windows(spec: Spec, want: int, seed: int, n_windows: Optional[int],
                  total: int, get_page) -> list[dict]:
    """The seeded window draw behind ``fetch_rows``, over any pager.

    ``get_page(offset, length)`` returns ``[(row_key, row), ...]`` for one
    window or ``None`` when that window is unknown (counted, never treated as
    empty). The datasets-server and the hub-file fallback share this so a
    freeze taken either way is the same draw over the same slice.
    """
    keep = spec.row_filter or (lambda row: True)
    #: windows that came back unknown after all retries — reported alongside the
    #: band so a thin sample is never read as a measured one
    fetch_rows.last_failed_windows = 0
    rng = random.Random(seed)
    # over-fetch: the adapter drops ungradable rows on top of the filter
    target = want * max(1, spec.fetch_multiplier)
    windows = max(1, min(n_windows or spec.n_windows, math.ceil(total / 10)))
    # A filtered spec pays a request per page either way, so take the whole page
    # — a narrow page just means more requests for the same yield.
    per_window = (
        100 if spec.row_filter else min(100, max(1, math.ceil(target / windows)))
    )
    if spec.where and not spec.row_filter:
        # the slice IS the population here, so the default 3x over-fetch is the
        # only headroom needed (for rows the adapter drops), not a 100x sieve
        per_window = min(100, max(1, math.ceil(target / windows)))
    stride = total / windows

    offsets = []
    for i in range(windows):
        base = int(i * stride)
        span = max(1, int(stride) - per_window)
        offsets.append(min(max(0, base + rng.randrange(span)), max(0, total - 1)))
    rng.shuffle(offsets)

    rows: list[dict] = []
    seen_offsets: set[int] = set()
    # Full-split coverage needs windows to abut, and abutting windows overlap by
    # a row or two once the stride is not an exact multiple of the page. Without
    # this the same item can be graded twice inside one n=50 sample.
    seen_rows: set = set()
    for offset in offsets:
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        page = get_page(offset, min(per_window, total - offset))
        if page is None:
            fetch_rows.last_failed_windows += 1
            continue
        for key, row in page:
            if key is None:
                key = json.dumps(row, sort_keys=True, default=str)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            try:
                if keep(row):
                    rows.append(row)
            except Exception:
                continue
        if len(rows) >= target:
            break
    rng.shuffle(rows)
    return rows


_WHERE_TERM = re.compile(r"""^\s*"([^"]+)"\s*=\s*'([^']*)'\s*$""")


def parse_where(where: str) -> list[tuple[str, str]]:
    """``"col"='value' [AND "col"='value' ...]`` -> ``[(col, value), ...]``.

    That is the only shape the specs use. Anything wider (OR, !=, LIKE, an
    unquoted name) raises, so a local filter can never quietly widen a slice
    the server would have narrowed.
    """
    terms = []
    for part in re.split(r"\s+AND\s+", where.strip(), flags=re.IGNORECASE):
        m = _WHERE_TERM.match(part)
        if not m:
            raise ValueError(f"unsupported where clause for a local filter: {where!r}")
        terms.append((m.group(1), m.group(2)))
    return terms


def _load_hub_rows(spec: Spec) -> list[dict]:
    """Every row of the split in file order, from the hub's own files
    (``datasets.load_dataset``; the datasets-server is not involved)."""
    import datasets as hf_datasets  # the HF package; only needed on this path

    config = None if spec.config == "default" else spec.config
    ds = hf_datasets.load_dataset(spec.dataset, config, split=spec.split)
    return [dict(row) for row in ds]


def fetch_rows_hub(spec: Spec, want: int, seed: int = 0,
                   n_windows: Optional[int] = None) -> list[dict]:
    """``fetch_rows`` served from the hub files instead of the datasets-server.

    The /filter index behind a ``where`` spec can answer HTTP 500 ("the dataset
    index is loading") for hours. This loads the split itself, applies the
    where clause locally (``parse_where``) and draws the same seeded windows
    ``fetch_rows`` draws over the slice in file order -- the order /filter pages
    through -- so the freeze is the server's sample, not a different one.
    """
    rows = _load_hub_rows(spec)
    if spec.where:
        terms = parse_where(spec.where)
        rows = [r for r in rows if all(str(r.get(col)) == val for col, val in terms)]
    total = len(rows)
    if not total:
        return []

    def get_page(offset: int, length: int):
        return [(i, rows[i]) for i in range(offset, min(offset + length, total))]

    return _draw_windows(spec, want, seed, n_windows, total, get_page)
