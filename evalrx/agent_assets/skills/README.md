# Bundled Agent Skills

Agent Skills vendored into the package so they travel with the repo (both
`git clone` and `pip install`) and are auto-discovered by `evalrx explore`
on the **claude** / **agy** backends. They style the figures the exploratory
coding agent writes under `figures/` — they do **not** affect the host-rendered,
deterministic chart specs (`render_chart_specs`).

Each `explore` run vendors these into its sandbox at
`<workdir>/.claude/skills/<name>/`, which an `--add-dir <workdir>` Claude Code /
agy run discovers automatically.

## nature-figure

Submission-grade Nature/high-impact-journal matplotlib/ggplot figure workflow.

- **Source:** https://github.com/Yuan1z0825/nature-skills (`skills/nature-figure`)
- **License:** Apache-2.0 — see [`nature-figure/LICENSE`](nature-figure/LICENSE).
- **Trimmed:** only the functional core is vendored (SKILL.md, manifest.yaml,
  `static/`, `references/`, ~176 KB). The upstream `assets/` demo galleries
  (~30 MB of example PNGs and figures4papers scripts) are **omitted** to keep
  the repo lean; fetch them from the source repo if you want the demo galleries.

## Disabling / overriding

- `evalrx explore … --no-skills` skips bundled skills for a run.
- `evalrx explore … --skill /path/to/other-skill` adds more skill dirs.
- `--allow-skills` also lets globally-installed `~/.claude/skills` be used.
