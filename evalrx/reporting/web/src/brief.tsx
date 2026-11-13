/**
 * L2 — one screen per stage: what this stop did, what came out, what that rests on.
 *
 * The report has three depths. L1 (the overview) shows the M1→M5 path and says
 * nothing about any one stage. L3 (each stage's `MxDetail`) shows everything a
 * stage retained — every probe, every figure, every raw record — and is the
 * audit surface. Between them there was nothing, so clicking a stage on L1
 * dropped the reader straight into `Gold In Answer Region 0`.
 *
 * L2 is that missing layer, and it is a SUMMARY, not a re-skin of L3: one
 * verdict sentence, a handful of numbers that each say what they amount to, one
 * figure, and three to five points that justify the verdict. If it needs a
 * scrollbar to make its point it has stopped being L2.
 *
 * Everything in a brief is DERIVED from what a run already emitted — no field
 * was added to the contract for it. That is deliberate: a "one-line conclusion"
 * field would be filled by five different producers with five different ideas
 * of what belongs in it, and would be empty on every run recorded before it
 * existed. Derivation has the opposite failure mode: it degrades to a shorter
 * sentence instead of a blank one. Each stage's builder (`briefMx` in
 * `stages/mx.tsx`) reads `stage_detail` (present on every run) first and treats
 * `contract` (absent on older runs, and on runs whose emission failed) as
 * enrichment it can do without.
 *
 * This module is the rendering the builders share.
 */
import ReactECharts from "echarts-for-react";
import { tc } from "./theme";
import { ZoomableImage } from "./lightbox";
import { chartValue, outcomeColors } from "./reportAccess";
import { AlertTriangle, ArrowRight, BarChart3 } from "lucide-react";
import type { Chart, FigureView, ReportData } from "./types";

/** What one L2 screen shows. Every field except `verdict` is optional, because
 *  every one of them is missing on some real run. */
export type Brief = {
  /** The answer, as one sentence. Never a number on its own. */
  verdict: string;
  /** One or two sentences of context under the verdict. */
  lead?: string;
  /** At most five numbers. `note` is what the number amounts to. */
  kpis: Array<{ label: string; value: React.ReactNode; note?: string }>;
  /** The single visual for this stage. Omitted rather than faked. */
  figure?: React.ReactNode;
  /** Caption under the figure, saying how to read it. */
  figureNote?: string;
  /** Three to five points the verdict rests on. */
  points: string[];
  /** The checks this step ran, each with the question it asks. Rendered as a
   *  list of its own rather than crammed into one bullet. */
  checks?: Array<{ name: string; question?: string; n?: number | null }>;
  /** The boundary of the claim — rendered as a warning, not as a footnote. */
  caveat?: string;
};

export function StageBrief({ brief, onDeepen }: { brief: Brief; onDeepen: () => void }) {
  // Two analyzers can measure the same thing under different names, and the
  // derived sentences then come out word-for-word identical. Three identical
  // bullets read as a rendering bug and crowd out the points that differ, so
  // the list is deduplicated on the rendered text before it is capped.
  const points = [...new Set(brief.points.map((point) => point.trim()).filter(Boolean))].slice(
    0,
    5,
  );
  return (
    <div className="stage-brief">
      <section className="brief-verdict">
        <span className="section-kicker">WHAT CAME OUT OF THIS STEP</span>
        <h2>{brief.verdict}</h2>
        {brief.lead && <p className="lead-small">{brief.lead}</p>}
      </section>

      {brief.kpis.length > 0 && (
        <div className="brief-kpis">
          {brief.kpis.slice(0, 5).map((item) => (
            <article key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value ?? "—"}</strong>
              {item.note && <small>{item.note}</small>}
            </article>
          ))}
        </div>
      )}

      <div className={`brief-body${brief.figure ? "" : " brief-body-textonly"}`}>
        {brief.figure && (
          <figure className="brief-figure">
            {brief.figure}
            {brief.figureNote && (
              <figcaption>
                <BarChart3 size={13} /> {brief.figureNote}
              </figcaption>
            )}
          </figure>
        )}
        {(points.length > 0 || (brief.checks && brief.checks.length > 0)) && (
          <div className="brief-points">
            {brief.checks && brief.checks.length > 0 && (
              <>
                <h3>Checks run</h3>
                <ul className="brief-checks">
                  {brief.checks.map((check) => (
                    <li key={check.name}>
                      <b>{check.name}</b>
                      {typeof check.n === "number" && check.n > 0 && <small>{check.n} cases</small>}
                      {check.question && <span>{check.question}</span>}
                    </li>
                  ))}
                </ul>
              </>
            )}
            {points.length > 0 && (
              <>
                <h3>What that rests on</h3>
                <ul>
                  {points.map((point, i) => (
                    <li key={i}>{point}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>

      {brief.caveat && (
        <div className="brief-caveat">
          <AlertTriangle size={17} />
          <p>{brief.caveat}</p>
        </div>
      )}

      <button className="brief-deepen" onClick={onDeepen}>
        See everything this step recorded <ArrowRight size={16} />
      </button>
    </div>
  );
}

/** The stage's own chart from the compiled report, by id. Undefined is fine. */
export function reportChart(report: ReportData, id: string): Chart | undefined {
  return (report.charts || []).find((chart) => chart.id === id);
}

/** A compact echarts rendering of one compiled `Chart`, sized for L2. */
export function BriefChart({ chart }: { chart: Chart }) {
  const series = chart.series || [];
  const option =
    chart.kind === "donut"
      ? {
          tooltip: { trigger: "item", valueFormatter: chartValue },
          color: outcomeColors(series.map((item) => item.label)),
          // The L2 column is narrower than the L3 page, and outside labels with
          // leader lines get clipped to "Pas..." there. A legend under the ring
          // carries the same two facts and cannot be cut off by the container.
          legend: {
            bottom: 0,
            textStyle: { color: tc("#b8c9c4"), fontSize: 11 },
            formatter: (name: string) =>
              `${name}  ${series.find((item) => item.label === name)?.value ?? ""}`,
          },
          series: [
            {
              type: "pie",
              radius: ["54%", "76%"],
              center: ["50%", "44%"],
              label: { show: false },
              data: series.map((item) => ({ name: item.label, value: item.value })),
            },
          ],
        }
      : {
          grid: { left: 152, right: 26, top: 10, bottom: 24 },
          tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: chartValue },
          xAxis: {
            type: "value",
            axisLabel: { color: tc("#8da19b") },
            splitLine: { lineStyle: { color: tc("#23332f") } },
          },
          yAxis: {
            type: "category",
            data: series.map((item) => item.label),
            // These labels are analyzer questions, not short keys. Truncating them to
            // "Can repeated attempt..." leaves a bar nobody can identify, so they wrap
            // onto two lines and the row height grows to match.
            axisLabel: {
              color: tc("#b8c9c4"),
              width: 138,
              overflow: "break",
              lineHeight: 13,
              fontSize: 11,
            },
            axisLine: { show: false },
            axisTick: { show: false },
          },
          series: [
            {
              type: "bar",
              barWidth: 12,
              data: series.map((item) => ({
                value: item.value,
                itemStyle: {
                  color: item.highlight ? tc("#6bd8ad") : tc("#586a65"),
                  borderRadius: 4,
                },
              })),
            },
          ],
        };
  const height = chart.kind === "donut" ? 250 : Math.max(200, series.length * 42 + 40);
  // notMerge: moving between stages swaps a bar chart for a donut in the same
  // slot, and echarts merges into the live instance by default — the previous
  // stage's category axis was still drawn behind the new chart.
  return <ReactECharts option={option} notMerge style={{ height }} />;
}

/** A saved PNG a stage produced. `data_uri` on an exported report, path on a served one. */
export function BriefImage({ figure }: { figure: FigureView }) {
  const source = figure.data_uri || `/api/artifact?path=${encodeURIComponent(figure.path)}`;
  return (
    <ZoomableImage
      className="brief-image"
      src={source}
      alt={figure.title || "Stage figure"}
      caption={figure.title || ""}
    />
  );
}

/** `text` guaranteed to end as a sentence, so derived lines can be joined
 *  without producing "…consistency=0.2) No way of testing…" or a double dot. */
export function sentence(text: string): string {
  const clean = String(text || "")
    .trim()
    .replace(/\.{2,}$/, ".");
  if (!clean) return "";
  return /[.!?…]$/.test(clean) ? clean : `${clean}.`;
}
