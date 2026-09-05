# Local run outputs — index

Not committed content (the directories below are `git`-ignored,
`examples/**/outputs_*/`), just a map for whoever's machine this is. `outputs/`
itself is the live default — the volume Docker mounts and the path every
command in `README.md` uses — never renamed or touched here.

The four directories below were ad hoc side-runs of the TCD audio refactor
(see project memory `project_tcd_audio_refactor.md`) that had accumulated
under inconsistent names (`outputs_r2`, `outputs_agy_r3`, `outputs_codex_terra`,
`outputs_codex_terra_live_20260821`). Renamed to a consistent
`outputs_<date>_<judge>_<role><n_cases>` pattern, sorts chronologically,
nothing deleted or re-run. No script or doc in the repo referenced the old
names (checked before renaming).

| Directory | Date | Judge | Cases | Commit | Outcome |
|---|---|---|---|---|---|
| `outputs_20260820_agy_pilot10` | 2026-08-20 | agy (local binary) | 10 | `92a3dcc` | 1 cycle, `stopped_by=no_hypotheses`, fix not attempted — small pilot to confirm the harness runs end-to-end before the full sweep below |
| `outputs_20260821_agy_full358` | 2026-08-21 | agy (local binary) | 358 | `1d2904a` | Full MMAU sweep, same outcome (`no_hypotheses`, `fixed=False`) — the agy-judged verification pass |
| `outputs_20260821_codexterra_full358` | 2026-08-21 | codex / gpt-5.6-terra | 358 | `62bd943` | Full MMAU sweep, same outcome — the second verification pass, different judge |
| `outputs_20260821_codexterra_live26` | 2026-08-21 | codex / gpt-5.6-terra | 26 | `62bd943` | Smaller live-mode re-run (stopped after diagnosis; `analysis`/`diagnosis`/`stage_skipped`, no `fix` event) |

All four ended `stopped_by: "no_hypotheses"` with `fixed: False` — none of
these produced a confirmed repair. They're evidence the pipeline ran cleanly
end-to-end on real MMAU cases under two different judges, not evidence of a
fix; that distinction is what `project_tcd_audio_refactor.md` means by
"verified twice."

Not evaluated here (out of scope for this pass, nothing deleted or merged):
whether any of these four supersede another, or whether they're all worth
keeping. That's a call for whoever reads this to make deliberately, not one
made by renaming.
