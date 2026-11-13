/**
 * M2 — screen: which measured behaviours line up with the errors (descriptive only).
 *
 * `M2Detail` is the stage's full record (L3); `briefM2` derives its one-screen
 * summary (L2) from the same data.
 */
import {
  headlineSplit,
  leadFrom,
  oddsPhrase,
  plural,
  pointsGap,
  stripMarkdown,
} from "../reportAccess";
import { AlertTriangle } from "lucide-react";
import type { M2StageDetail, ReportData } from "../types";
import { reportChart, BriefChart, BriefImage, type Brief } from "../brief";
import { RecordTable } from "../detail";
import { normalizeKey } from "../format";
import { chartCited, labelledFigures, EvidenceFigure, StatEvidenceChart } from "./evidence";
import { StageBanner, StageKpis } from "./layout";

export function M2Detail({ data }: { data: Partial<M2StageDetail> }) {
  const figures = labelledFigures(data.figures || [], data.stats || []);
  const takeaways = data.takeaways || [];
  const citedKeys = takeaways.flatMap((item) => item.chart_names || []).map(normalizeKey);
  const supporting = figures.filter((figure) => !chartCited(figure.id, citedKeys));
  return (
    <>
      <StageBanner kind="DESCRIPTIVE" title="Exploratory evidence">
        Patterns here were found in the analysis split. They are leads—not validated mechanisms.
        Only M4 can issue a held-out verdict.
      </StageBanner>
      <StageKpis
        items={[
          { label: "Ranked findings", value: takeaways.length },
          { label: "Visual artifacts", value: figures.length },
          { label: "Statistical screens", value: data.stats?.length || 0 },
        ]}
      />
      {data.conclusion && (
        <div className="stage-callout">
          <b>Agent screening conclusion</b>
          <p>{data.conclusion}</p>
        </div>
      )}
      {!!data.stats?.length && (
        <StatEvidenceChart
          stats={data.stats}
          title="Which measured behaviors are most connected to errors?"
          note="Each bar summarizes the difference observed between correct and incorrect cases. It is a pattern, not proof of cause."
        />
      )}
      <div className="finding-stack">
        {takeaways.map((item, index) => {
          const names = (item.chart_names || []).map(normalizeKey);
          const evidence = figures.filter((figure) => chartCited(figure.id, names));
          return (
            <article className="analysis-finding" key={index}>
              <header>
                <span>FINDING {String(index + 1).padStart(2, "0")}</span>
                <h3>{item.plain_title || item.title}</h3>
              </header>
              {evidence.length > 0 ? (
                <div className="analysis-figures">
                  {evidence.map((figure) => (
                    <EvidenceFigure figure={figure} key={figure.id} />
                  ))}
                </div>
              ) : (
                <div className="missing-evidence">
                  <AlertTriangle size={17} /> Referenced visual evidence was not found in this
                  report bundle.
                </div>
              )}
              <details className="finding-details">
                <summary>Interpretation, caveat, and provenance</summary>
                {item.analysis && <p>{item.analysis}</p>}
                {item.caveat && (
                  <div className="evidence-caution">
                    <b>Boundary</b>
                    {item.caveat}
                  </div>
                )}
                {!!item.table_names?.length && (
                  <small>Source tables: {item.table_names.join(", ")}</small>
                )}
              </details>
            </article>
          );
        })}
      </div>
      {supporting.length > 0 && (
        <details className="agent-transcript">
          <summary>
            Supporting exploratory material—not ranked conclusions ({supporting.length})
          </summary>
          <div className="analysis-figures supporting">
            {supporting.map((figure) => (
              <EvidenceFigure figure={figure} key={figure.id} />
            ))}
          </div>
        </details>
      )}
      {!!data.stats?.length && (
        <details className="agent-transcript">
          <summary>Statistical screening records ({data.stats.length})</summary>
          <RecordTable rows={data.stats} />
        </details>
      )}
    </>
  );
}

// ── M2 brief (L2) ──────────────────────────────────────────────────────────
export function briefM2(report: ReportData, detail: Partial<M2StageDetail>): Brief {
  const stats = detail.stats || [];
  const takeaways = detail.takeaways || [];
  const figures = detail.figures || [];
  const conclusion = stripMarkdown(detail.conclusion || detail.narrative || "");
  const [headline, rest] = headlineSplit(conclusion);
  const survived = stats.filter((item) => item.reject);

  const verdict = headline
    ? headline
    : survived.length
      ? `${plural(survived.length, "measured behaviour")} line up with the model's errors ` +
        "more often than chance would explain."
      : "No behaviour stood out strongly enough to separate the right answers from the wrong ones.";

  const points: string[] = [];
  for (const item of takeaways.slice(0, 3)) {
    const line = stripMarkdown(item.plain_title || item.title || "");
    if (line) points.push(line);
  }
  if (!points.length) {
    // Never the tool's own `summary` here: it is the audit line
    // ("[mcnemar + e-value (paired binary)] effect=-0.2917 ... -> REJECT H0"),
    // and pasting it into a summary layer is the exact thing this layer is
    // supposed to spare the reader. It stays reachable one level down.
    for (const item of survived.slice(0, 3)) {
      const label = stripMarkdown(String(item.label || item.raw_signal || "A measured behaviour"));
      const { fail_rate_signal: withIt, fail_rate_control: without } = item;
      if (typeof withIt === "number" && typeof without === "number") {
        points.push(
          `${label}: the model got ${Math.round(withIt * 100)}% of cases ` +
            `wrong when this shows up, against ${Math.round(without * 100)}% when it does not.`,
        );
        continue;
      }
      const gap = pointsGap(item.effect);
      const odds = oddsPhrase(item.e_value);
      points.push(
        `${label}: a gap of ${gap || "some size"} between the two groups compared` +
          (odds ? `, with the evidence against that being luck running ${odds}.` : "."),
      );
    }
  }
  if (!points.length && conclusion) {
    for (const step of (detail.observations || []).slice(0, 3)) {
      points.push(stripMarkdown(String(step)));
    }
  }
  if (survived.length && stats.length > survived.length) {
    points.push(
      `${survived.length} of the ${stats.length} patterns screened are still standing ` +
        "after discounting for how many were tried at once; the rest could be luck.",
    );
  }

  const topFigure = figures[0];
  const effects = reportChart(report, "effects");

  return {
    verdict,
    lead: leadFrom(rest) || undefined,
    kpis: [
      ...(takeaways.length ? [{ label: "Ranked findings", value: takeaways.length }] : []),
      {
        label: "Patterns screened",
        value: stats.length,
        note: "each one compares a measured behaviour against the errors",
      },
      {
        label: "Still standing",
        value: survived.length,
        note: "after discounting for how many were tried at once",
      },
      ...(figures.length ? [{ label: "Figures produced", value: figures.length }] : []),
    ],
    figure: topFigure ? (
      <BriefImage figure={topFigure} />
    ) : effects ? (
      <BriefChart chart={effects} />
    ) : undefined,
    figureNote: topFigure
      ? stripMarkdown(topFigure.reading || topFigure.title || "")
      : "A longer bar means the behaviour separates right from wrong answers more " +
        "sharply. It is a pattern, not a cause.",
    points,
    caveat:
      "These patterns were found on the same half of the data used to look for " +
      "them, so any of them could be a coincidence that happens to fit. Nothing here " +
      "is a verdict — the held-out check is the only step allowed to issue one.",
  };
}
