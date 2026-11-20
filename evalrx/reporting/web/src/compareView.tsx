/**
 * This run beside another one the server holds (`#/compare/<id>`).
 *
 * Only on a served report: an export or the static build has one run and no
 * server to open a second. The second run is opened by the id the server
 * handed out; ids do not survive a server restart, so an id it no longer
 * knows falls back to picking the run again.
 */
import { useEffect, useState } from "react";
import { compareRuns } from "./compare";
import { DetailShell } from "./detail";
import { fetchRuns, openRun, type RunListItem } from "./runsPanel";
import { routeHash, withRun, type Navigate } from "./route";
import type { ReportData } from "./types";
import type { UploadedRun } from "./upload";

function describe(data: ReportData): string {
  return data.setting.kind === "explore"
    ? `Exploratory analysis of ${data.setting.dataset}`
    : `${data.setting.model} on ${data.setting.dataset}`;
}

export function CompareView({
  data,
  run,
  otherId,
  back,
  navigate,
  open,
}: {
  data: ReportData;
  run: { id: string; label: string };
  otherId?: string;
  back: () => void;
  navigate: Navigate;
  open: (run: UploadedRun) => void;
}) {
  const [other, setOther] = useState<UploadedRun | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    setOther(null);
    setError("");
    if (!otherId) return;
    openRun(otherId)
      .then(setOther)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [otherId]);
  return (
    <DetailShell
      title="Compare runs"
      subtitle="This run beside another: what each measured, and which shared cases changed outcome."
      back={back}
    >
      {!otherId || error ? (
        <RunPicker
          runId={run.id}
          error={error}
          pick={(id) => navigate({ view: "compare", with: id })}
        />
      ) : !other ? (
        <p className="compare-note">Opening the other run…</p>
      ) : (
        <ComparisonTable a={data} run={run} b={other} open={open} />
      )}
    </DetailShell>
  );
}

function RunPicker({
  runId,
  error,
  pick,
}: {
  runId: string;
  error: string;
  pick: (id: string) => void;
}) {
  const [items, setItems] = useState<RunListItem[] | null>(null);
  const [listError, setListError] = useState("");
  useEffect(() => {
    fetchRuns()
      .then((res) => setItems(res.items.filter((item) => item.id !== runId)))
      .catch((err) => setListError(err instanceof Error ? err.message : String(err)));
  }, [runId]);
  return (
    <section className="compare-picker">
      {error && (
        <p className="compare-note">That run could not be opened ({error}); pick it again.</p>
      )}
      <h2>Compare with</h2>
      {listError && <p className="compare-note">{listError}</p>}
      {items && !items.length && (
        <p className="compare-note">No other run was found beside this one.</p>
      )}
      <ul>
        {items?.map((item) => (
          <li key={item.id}>
            <button type="button" onClick={() => pick(item.id)}>
              <strong>{item.path}</strong>
              <span>{[item.model, item.dataset].filter(Boolean).join(" · ")}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function ComparisonTable({
  a,
  run,
  b,
  open,
}: {
  a: ReportData;
  run: { id: string; label: string };
  b: UploadedRun;
  open: (run: UploadedRun) => void;
}) {
  const result = compareRuns(a, b.data);
  const caseLink = (id: string) => withRun(routeHash({ view: "cases", caseId: id }), run.id);
  const shown = (value: string | number | null) => (value === null ? "—" : String(value));
  return (
    <>
      <section className="compare-head">
        <div>
          <span>This run</span>
          <strong>{describe(a)}</strong>
          {run.label && <code>{run.label}</code>}
        </div>
        <div>
          <span>Compared with</span>
          <strong>{describe(b.data)}</strong>
          {b.label && <code>{b.label}</code>}
          <button type="button" onClick={() => open(b)}>
            Open this run
          </button>
        </div>
      </section>
      {(!result.sameDataset || !result.sameKind) && (
        <p className="compare-note">
          {!result.sameKind
            ? "One is an exploratory analysis and the other a diagnosis run, so only what both recorded lines up."
            : "These runs used different datasets, so their numbers are not measured on the same cases."}
        </p>
      )}
      <div className="record-table-wrap">
        <table className="record-table compare-table">
          <thead>
            <tr>
              <th />
              <th>This run</th>
              <th>Compared with</th>
            </tr>
          </thead>
          <tbody>
            {result.metrics.map((row) => (
              <tr key={row.id}>
                <td>{row.label}</td>
                <td>{shown(row.a)}</td>
                <td>{shown(row.b)}</td>
              </tr>
            ))}
            {result.stages.map((row) => (
              <tr key={row.id}>
                <td>
                  {row.id.toUpperCase()} · {row.title}
                </td>
                <td>{shown(row.a)}</td>
                <td>{shown(row.b)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="compare-cases">
        <h2>Cases both runs evaluated: {result.cases.shared}</h2>
        <p className="compare-note">
          {result.cases.onlyA} only in this run, {result.cases.onlyB} only in the other.
        </p>
        <CaseList title="Failed here, passed there" ids={result.cases.nowPass} link={caseLink} />
        <CaseList title="Passed here, failed there" ids={result.cases.nowFail} link={caseLink} />
      </section>
    </>
  );
}

function CaseList({
  title,
  ids,
  link,
}: {
  title: string;
  ids: string[];
  link: (id: string) => string;
}) {
  return (
    <div className="compare-case-list">
      <h3>
        {title} ({ids.length})
      </h3>
      {ids.map((id) => (
        <a key={id} href={link(id)}>
          {id}
        </a>
      ))}
    </div>
  );
}
