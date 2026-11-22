"""Prompt template for naming failure-mode clusters."""

from __future__ import annotations

NAME_CLUSTERS_PROMPT = """\
You are an ML failure-analysis expert. Below are groups of failing cases that
were clustered by similarity. For each cluster, propose a short, specific
failure-mode name and a one-sentence description of what these cases have in
common. Some clusters include a "boundary contrast" section: a failing case
paired with its nearest successful (non-failing) counterpart. When present,
use these pairs to pinpoint exactly where the model's behavior crosses from
correct to incorrect, rather than just describing the failing topic.

{clusters_block}

Give one entry per cluster listed above: its cluster_id, a short snake_case
name, and a one-sentence description."""

ERROR_SIGNAL_PROMPT = """\
You are an ML failure-analysis expert. Below are (expected, observed) pairs
from cases where a model's output was judged incorrect. For each pair, write
a short phrase (5-10 words) describing the *mechanism* of the mismatch (e.g.
"off-by-one count", "wrong entity substituted", "missing negation"), not just
that it is wrong.

{pairs_block}

Give one entry per pair listed above: its index and the short phrase."""
