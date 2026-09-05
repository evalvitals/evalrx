import { useRef, useState } from "react";
import { tc } from "./theme";
import { useVirtualizer } from "@tanstack/react-virtual";
import ReactECharts from "echarts-for-react";
import { AlertTriangle, ArrowLeft, BarChart3, Beaker, Bot, CheckCircle2, ChevronRight, Microscope, Search, ShieldCheck, Wrench, XCircle } from "lucide-react";
import type { AnalyzerSelection, Case, DebugEvent, DiagnosisOutput, FixAttemptWire,
  FixOutput, HypothesisTestOutput, Modality, ProbeOutput, ReportData } from "./types";
import { buildBrief, StageBrief } from "./brief";
import { ZoomableImage } from "./lightbox";
import { chartPercent, chartValue, findContract, stageCode } from "./reportAccess";

/**
 * A stage, at whichever of the two depths the reader asked for.
 *
 * Clicking a stage on the overview used to land on the complete record — every
 * probe, every raw metric name, the model's full output — which is the right
 * page for auditing a conclusion and the wrong one for finding out what the
 * conclusion IS. The brief (L2) answers that in one screen and hands the reader
 * a way down; the full record (L3) is unchanged behind it, because nothing in
 * it was surplus, it was just never the first thing anyone needed.
 *
 * Depth resets when the reader moves to another stage: arriving at M4 already
 * scrolled into its raw event log is a state nobody asked for.
 */
export function EvidenceView({ data, back, navigate, initialStage }: { data: ReportData; back: () => void; navigate?: (view: string) => void; initialStage?: string }) {
  const [selected, setSelected] = useState(initialStage || data.stages[0]?.id);
  const [full, setFull] = useState(false);
  const stage = data.stages.find((item) => item.id === selected);
  const events = data.debug.events.filter((event) => String(event.stage || "").toLowerCase().includes(selected));
  const brief = selected ? buildBrief(selected, data) : null;
  const show = (id: string) => { setSelected(id); setFull(false); };
  return <DetailShell
    title={full ? "Everything this step recorded" : "What this step found"}
    subtitle={full
      ? "The complete record behind the summary: every measurement, artifact and agent event."
      : "One screen per step: the finding, the numbers behind it, and what it does not settle."}
    back={back}
    agent={data.setting.diagnosed_by}
  >
    <div className="evidence-layout"><aside className="stage-list">{data.stages.map((item) => <button className={selected === item.id ? "active" : ""} key={item.id} onClick={() => show(item.id)}><span>{stageCode(item.code)}</span><div><strong>{item.title}</strong><small>{item.status.replaceAll("-", " ")}</small></div><ChevronRight /></button>)}</aside><section className="evidence-detail">
      <header className="evidence-head">
        <div><span className="section-kicker">{stageCode(stage?.code)} · {stage?.status?.replaceAll("-", " ")}</span><h2>{stage?.title}</h2><p className="lead-small">{stage?.purpose}</p></div>
        {brief && <div className="depth-switch" role="tablist" aria-label="Level of detail">
          <button role="tab" aria-selected={!full} className={full ? "" : "active"} onClick={() => setFull(false)}>Summary</button>
          <button role="tab" aria-selected={full} className={full ? "active" : ""} onClick={() => setFull(true)}>Full record</button>
        </div>}
      </header>
      {brief && !full
        ? <StageBrief brief={brief} onDeepen={() => setFull(true)} />
        : <>
          <StageArtifact stage={selected || ""} detail={data.stage_detail || {}} report={data} navigate={navigate} />
          <details className="raw-events"><summary>Raw agent events ({events.length})</summary>{events.length ? events.map((event, index) => <EventRow event={event} key={`${event.event_seq}-${index}`} />) : <p className="empty">No stage-level events were retained.</p>}</details>
        </>}
    </section></div>
  </DetailShell>;
}

function StageArtifact({ stage, detail, report, navigate }: { stage: string; detail: Record<string, any>; report: ReportData; navigate?: (view: string) => void }) {
  if (stage === "m1") return <M1Detail data={detail.m1 || {}} report={report} />;
  if (stage === "m2") return <M2Detail data={detail.m2 || {}} />;
  if (stage === "m3") return <M3Detail data={detail.m3 || {}} report={report} />;
  if (stage === "m4") return <M4Detail data={detail.m4 || {}} report={report} />;
  if (stage === "m5") return <M5Detail data={detail.m5 || {}} report={report} navigate={navigate} />;
  return <EmptyStage title="No stage data" body="This stage did not retain a structured artifact." />;
}

function StageBanner({ kind, title, children }: { kind: string; title: string; children: React.ReactNode }) {
  return <div className={`stage-banner stage-banner-${kind}`}><div><span>{kind}</span><strong>{title}</strong></div><p>{children}</p></div>;
}

function StageKpis({ items }: { items: Array<{ label: string; value: React.ReactNode; note?: string }> }) {
  return <div className="stage-kpis">{items.map((item) => <article key={item.label}><span>{item.label}</span><strong>{item.value ?? "—"}</strong>{item.note && <small>{item.note}</small>}</article>)}</div>;
}

/**
 * Which modality the run was actually about, read from M1's contract payload.
 *
 * The three sets are shown separately because they disagree in exactly the case
 * that used to go wrong silently: an omni model declares four modalities, the
 * benchmark fills one, and analyzer routing must follow the batch. A single
 * "model kind" label could not express that, and rendering only the model's
 * declaration would tell the reader the run probed images when it probed audio.
 */
function ModalityBand({ selection }: { selection?: AnalyzerSelection }) {
  if (!selection) return null;
  const routed = selection.routed_on || [];
  const declared = selection.model_modalities || [];
  const probed = selection.probed_modalities || [];
  const narrowed = declared.length > routed.length;
  return <div className="modality-band">
    <div><small>ROUTED ON</small><b>{routed.map(labelModality).join(" + ") || "text"}</b></div>
    <div><small>MODEL ACCEPTS</small><b>{declared.map(labelModality).join(" + ") || "text"}</b></div>
    <div><small>CASES CARRIED</small><b>{probed.map(labelModality).join(" + ") || "text"}</b></div>
    {selection.is_agent && <div><small>SHAPE</small><b>agent trajectories</b></div>}
    {narrowed && <p className="modality-note">The model accepts more than this benchmark exercises, so the checks were narrowed to what the cases actually contain.</p>}
  </div>;
}

function labelModality(slot: Modality): string {
  return { text: "text", image: "images", audio: "audio", video: "video" }[slot] || slot;
}

function M1Detail({ data, report }: { data: any; report: ReportData }) {
  const probes = data.probes || [];
  const examples = data.examples || [];
  const m1 = findContract<ProbeOutput>(report, "m1");
  return <>
    <StageBanner kind="MEASURE" title="Behavioral checkup">We give the model real tasks, then run several checks on its behavior. This tells us where it struggles; it does not yet tell us why.</StageBanner>
    <ModalityBand selection={m1?.selection} />
    <StageKpis items={[{ label: "Probes run", value: data.n_probes || probes.length }, { label: "Cases measured", value: data.n_measured || "—" }, { label: "Runtime", value: seconds(data.duration) }]} />
    {examples.length > 0 && <ExampleSection eyebrow="A real example" title="What one M1 check looks like" note="Examples make the measurement concrete. The result below is based on all measured cases, not just this one."><div className="example-deck">{examples.map((example: any) => <M1Example example={example} report={report} key={example.id} />)}</div></ExampleSection>}
    {data.operations?.length > 0 && <OperationExamples examples={data.operations} report={report} />}
    <div className="probe-grid">{probes.map((probe: any, index: number) => <details className="probe-card" key={probe.id} open={index === 0}>
      <summary><span className="section-kicker">PROBE {String(index + 1).padStart(2, "0")} · {probe.n_cases || "—"} CASES</span><h3>{probe.title}</h3><p>{probe.question}</p><div className="probe-metrics">{(probe.metrics || []).map((metric: any) => <span key={metric.label}><b>{metric.value ?? "—"}</b>{metric.label}</span>)}</div></summary>
      <div className="probe-expanded"><p>{probe.description}</p>{Object.keys(probe.finding_summary || {}).length > 0 && <KeyValueGrid values={probe.finding_summary} />}{Object.keys(probe.raw_finding_summary || {}).length > 0 && <details className="nested-detail"><summary>Technical measurement names and raw values</summary><KeyValueGrid values={probe.raw_finding_summary} /></details>}{probe.sample_rows?.length > 0 && <details className="nested-detail"><summary>Inspect {probe.sample_rows.length} representative measurement rows</summary><RecordTable rows={probe.sample_rows} /></details>}</div>
    </details>)}</div>
  </>;
}

function ExampleSection({ eyebrow, title, note, children }: { eyebrow: string; title: string; note: string; children: React.ReactNode }) {
  return <section className="example-section"><header><span>{eyebrow}</span><h3>{title}</h3><p>{note}</p></header>{children}</section>;
}

function M1Example({ example, report }: { example: any; report?: ReportData }) {
  const outcome = String(example.outcome || "unknown").toLowerCase();
  return <article className="example-card m1-example"><header><span className="example-step">1 · ORIGINAL TASK</span><em className={`status status-${outcome}`}>{outcome}</em></header><ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} /><div className="example-prompt">{example.input || "The original task text was not retained."}</div><div className="example-flow"><div><small>MODEL ANSWER</small><b>{displayValue(example.baseline_output)}</b></div><div><small>EXPECTED ANSWER</small><b>{displayValue(example.expected)}</b></div><div><small>CHECK RUN</small><b>{example.probe_title || "Behavior check"}</b></div></div><p className="example-question">{example.probe_question}</p><div className="example-check"><span>2 · WHAT THE CHECK RECORDED</span><KeyValueGrid values={example.check_result || {}} /></div><footer>{example.plain_reading}</footer></article>;
}

function ExampleMedia({ mediaIds, report, caseId }: { mediaIds?: string[]; report?: ReportData; caseId?: string }) {
  if (!report || !mediaIds?.length) return null;
  const media = mediaIds.map((id) => report.media.find((item) => item.id === id)).filter(Boolean) as ReportData["media"];
  if (!media.length) return null;
  return <div className="example-media">{media.map((item) => <MediaPreview key={item.id} media={item} caseId={caseId || "example"} />)}</div>;
}

/**
 * Whether a figure is the chart a takeaway cites.
 *
 * The coder agent names its own files, and the two ends of this join disagree
 * in ways seen live: a takeaway cites `failrate_coverage_verification_gap_
 * majority_share` while the file on disk is `02_failrate_coverage_
 * verification_gap_majority_shar.png` — an ordering prefix the citation never
 * had, and a stem the agent cut one letter short. Exact equality told the
 * reader "Referenced visual evidence was not found" with the chart sitting in
 * the bundle, filed under supporting material.
 *
 * So the match strips a leading order prefix and accepts one side being a
 * truncation of the other — with enough shared prefix that unrelated charts
 * cannot collide (every explorer chart name this loose match applies to is
 * far longer than the floor).
 */
function chartCited(figureId: any, citedKeys: string[]): boolean {
  const fig = normalizeKey(figureId).replace(/^\d+/, "");
  return citedKeys.some((cited) =>
    fig === cited
    || (Math.min(fig.length, cited.length) >= 16
        && (fig.startsWith(cited) || cited.startsWith(fig))));
}

/**
 * Figure titles with the run's own reader labels in place of flat identifiers.
 *
 * The explorer titles its charts with the identifiers it computed over —
 * "Failure rate by coverage_verification_gap_majority_share" — because the
 * identifier is all it has. The stats rows already carry the reader-facing
 * name of each signal, so this is a lookup, not a paraphrase: a flat
 * identifier naming a known signal becomes that signal's label, longest first
 * so a name containing another is replaced whole. Whatever stays unmapped is
 * only de-slugged — the same words, minus the underscores. The original title
 * survives on the card as its hover text.
 */
function labelledFigures(figures: any[], stats: any[]): any[] {
  const pairs = (stats || [])
    .filter((row: any) => row?.raw_signal && row?.label && String(row.raw_signal).includes("."))
    .map((row: any) => [String(row.raw_signal).replace(/\./g, "_"), String(row.label)] as const)
    .sort((a, b) => b[0].length - a[0].length);
  const display = (title: string) => {
    let out = title;
    for (const [flat, label] of pairs) out = out.split(flat).join(label);
    return out.replace(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b/g, (m) => m.replace(/_/g, " "));
  };
  return (figures || []).map((figure: any) => {
    const title = display(String(figure.title || ""));
    return title === figure.title ? figure : { ...figure, title, raw_title: figure.title };
  });
}

function M2Detail({ data }: { data: any }) {
  const figures = labelledFigures(data.figures || [], data.stats || []);
  const takeaways = data.takeaways || [];
  const citedKeys = takeaways.flatMap((item: any) => item.chart_names || []).map(normalizeKey);
  const supporting = figures.filter((figure: any) => !chartCited(figure.id, citedKeys));
  return <>
    <StageBanner kind="DESCRIPTIVE" title="Exploratory evidence">Patterns here were found in the analysis split. They are leads—not validated mechanisms. Only M4 can issue a held-out verdict.</StageBanner>
    <StageKpis items={[{ label: "Ranked findings", value: takeaways.length }, { label: "Visual artifacts", value: figures.length }, { label: "Statistical screens", value: data.stats?.length || 0 }]} />
    {data.conclusion && <div className="stage-callout"><b>Agent screening conclusion</b><p>{data.conclusion}</p></div>}
    {data.stats?.length > 0 && <StatEvidenceChart stats={data.stats} title="Which measured behaviors are most connected to errors?" note="Each bar summarizes the difference observed between correct and incorrect cases. It is a pattern, not proof of cause." />}
    <div className="finding-stack">{takeaways.map((item: any, index: number) => {
      const names = (item.chart_names || []).map(normalizeKey);
      const evidence = figures.filter((figure: any) => chartCited(figure.id, names));
      return <article className="analysis-finding" key={index}><header><span>FINDING {String(index + 1).padStart(2, "0")}</span><h3>{item.plain_title || item.title}</h3></header>
        {evidence.length > 0 ? <div className="analysis-figures">{evidence.map((figure: any) => <EvidenceFigure figure={figure} key={figure.id} />)}</div> : <div className="missing-evidence"><AlertTriangle size={17} /> Referenced visual evidence was not found in this report bundle.</div>}
        <details className="finding-details"><summary>Interpretation, caveat, and provenance</summary>{item.analysis && <p>{item.analysis}</p>}{item.caveat && <div className="evidence-caution"><b>Boundary</b>{item.caveat}</div>}{item.table_names?.length > 0 && <small>Source tables: {item.table_names.join(", ")}</small>}</details>
      </article>;
    })}</div>
    {supporting.length > 0 && <details className="agent-transcript"><summary>Supporting exploratory material—not ranked conclusions ({supporting.length})</summary><div className="analysis-figures supporting">{supporting.map((figure: any) => <EvidenceFigure figure={figure} key={figure.id} />)}</div></details>}
    {data.stats?.length > 0 && <details className="agent-transcript"><summary>Statistical screening records ({data.stats.length})</summary><RecordTable rows={data.stats} /></details>}
  </>;
}

function EvidenceFigure({ figure }: { figure: any }) {
  const source = figure.data_uri || `/api/artifact?path=${encodeURIComponent(figure.path)}`;
  return <figure><ZoomableImage src={source} alt={figure.title} caption={figure.title} /><figcaption><b title={figure.raw_title || undefined}>{figure.title}</b>{figure.question && <span>{figure.question}</span>}{figure.reading && <p><BarChart3 size={13} /> {figure.reading}</p>}{figure.do_not_infer && <small>Do not infer: {figure.do_not_infer}</small>}</figcaption></figure>;
}

/**
 * A y-axis of short numbered handles, with the real name on hover.
 *
 * These labels are analyzer questions — "Did the model give a usable answer? —
 * Answer appears cut short" — and no honest amount of gutter fits eight of
 * them. Truncated to one line they collapsed into six identical rows;
 * wrapped, they ate half the chart. Numbering them turns the axis into an
 * index the eye can scan, hands the width back to the bars, and puts the full
 * name one hover away in the tooltip, which is where the numbers already are.
 *
 * The rows are ordered by rank, so Behavior 1 is the strongest — the numbering
 * carries that, it is not decoration.
 */
const AXIS_GUTTER = 96;

function numberedAxis(count: number, noun: string) {
  // echarts draws category index 0 at the BOTTOM of a horizontal bar chart, and
  // every series here is reversed to put rank 1 on top; the labels reverse with
  // them so the numbering runs downwards.
  const labels = Array.from({ length: count }, (_, i) => `${noun} ${i + 1}`).reverse();
  return {
    type: "category", data: labels,
    axisLabel: { color: tc("#b8c9c4"), fontSize: 11 },
    axisTick: { show: false },
  };
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"]/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" } as Record<string, string>)[c]);
}

/**
 * Tooltip whose heading is the full name the axis had to shorten.
 *
 * `names` is in plot order (bottom-first), matching the series data, so the
 * hovered dataIndex reads straight out of it.
 */
function namedTooltip(names: string[], format: (value: unknown) => string) {
  return (params: any) => {
    const series = Array.isArray(params) ? params : [params];
    const first = series[0] || {};
    const head = escapeHtml(String(names[first.dataIndex] ?? first.name ?? ""));
    const lines = series.map((row: any) =>
      `<div style="display:flex;gap:18px;justify-content:space-between;align-items:baseline">`
      + `<span>${row.marker} ${escapeHtml(String(row.seriesName || ""))}</span>`
      + `<b>${escapeHtml(format(row.value))}</b></div>`).join("");
    return `<div style="max-width:300px;white-space:normal;line-height:1.45;margin-bottom:7px">`
      + `<b>${first.axisValue ? escapeHtml(String(first.axisValue)) + " · " : ""}${head}</b></div>${lines}`;
  };
}

/** Hovering a label opens its row's tooltip — the axis alone no longer names it. */
function labelHoverTooltip(rowCount: number) {
  return (chart: any) => {
    const zr = chart.getZr?.();
    if (!zr) return;
    let shown = -1;
    const hide = () => {
      if (shown === -1) return;
      shown = -1;
      chart.dispatchAction({ type: "hideTip" });
    };
    // Two spellings of the same question; which one a build answers depends on
    // the echarts version, so ask both and take whichever returns a number.
    const rowAt = (x: number, y: number) => {
      for (const value of [
        () => chart.convertFromPixel({ gridIndex: 0 }, [x, y])?.[1],
        () => chart.convertFromPixel({ yAxisIndex: 0 }, y),
      ]) {
        try {
          const index = Math.round(Number(value()));
          if (Number.isFinite(index)) return index;
        } catch { /* try the other spelling */ }
      }
      return NaN;
    };
    zr.on("mousemove", (event: any) => {
      if (event.offsetX >= AXIS_GUTTER) return hide();
      const index = rowAt(event.offsetX, event.offsetY);
      if (!Number.isFinite(index) || index < 0 || index >= rowCount) return hide();
      if (index === shown) return;
      shown = index;
      chart.dispatchAction({ type: "showTip", seriesIndex: 0, dataIndex: index });
    });
    zr.on("globalout", hide);
  };
}

function StatEvidenceChart({ stats, title, note }: { stats: any[]; title: string; note: string }) {
  const rows = stats.filter((item) => typeof item.effect === "number").slice(0, 8);
  if (!rows.length) return null;
  const plotted = rows.slice().reverse();
  const effectNames = plotted.map((item) => String(item.label));
  const option = { grid: { left: AXIS_GUTTER, right: 30, top: 30, bottom: 30 }, tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, formatter: namedTooltip(effectNames, (v) => chartValue(v, { signed: true })) }, xAxis: { type: "value", name: "difference in error rate", nameTextStyle: { color: tc("#8fa49d"), fontSize: 10 }, axisLabel: { color: tc("#8fa49d") }, splitLine: { lineStyle: { color: tc("#22332e") } } }, yAxis: numberedAxis(rows.length, "Behavior"), series: [{ name: "Observed difference", type: "bar", data: plotted.map((item) => ({ value: item.effect, itemStyle: { color: item.reject ? tc("#6bd8ad") : tc("#71857f"), borderRadius: 4 } })), markLine: { silent: true, symbol: "none", lineStyle: { color: tc("#f4ca72"), type: "dashed" }, data: [{ xAxis: 0 }] } }] };
  const rateRows = rows.filter((item) => typeof item.fail_rate_signal === "number" && typeof item.fail_rate_control === "number");
  const ratePlotted = rateRows.slice().reverse();
  const rateNames = ratePlotted.map((item) => String(item.label));
  const rateOption = { grid: { left: AXIS_GUTTER, right: 30, top: 24, bottom: 28 }, tooltip: { trigger: "axis", formatter: namedTooltip(rateNames, chartPercent) }, legend: { top: 0, textStyle: { color: tc("#9fb2ac"), fontSize: 10 } }, xAxis: { type: "value", max: 1, axisLabel: { color: tc("#8fa49d"), formatter: (v: number) => `${Math.round(v * 100)}%` }, splitLine: { lineStyle: { color: tc("#22332e") } } }, yAxis: numberedAxis(rateRows.length, "Pattern"), series: [{ name: "Cases with this behavior", type: "bar", data: ratePlotted.map((item) => item.fail_rate_signal), itemStyle: { color: tc("#89a6ff"), borderRadius: 3 } }, { name: "Other cases", type: "bar", data: ratePlotted.map((item) => item.fail_rate_control), itemStyle: { color: tc("#657b74"), borderRadius: 3 } }] };
  return <section className="stat-evidence"><header><span>VISUAL SUMMARY OF M2</span><h3>{title}</h3><p>{note}</p></header><ReactECharts option={option} notMerge onChartReady={labelHoverTooltip(effectNames.length)} style={{ height: Math.max(300, rows.length * 46) }} />{rateRows.length > 0 && <><h4 className="stat-subtitle">What those patterns mean in the cases</h4><p className="stat-caption">For each measured behavior, compare the error rate among cases with that behavior against all other cases. This is an observed comparison, not a causal claim.</p><ReactECharts option={rateOption} notMerge onChartReady={labelHoverTooltip(rateNames.length)} style={{ height: Math.max(280, rateRows.length * 54) }} /></>}<details><summary>Technical measurement names and test records</summary><RecordTable rows={rows.map((item, index) => ({ behavior: `Behavior ${index + 1}`, name: item.label, measurement: item.raw_signal, effect: item.effect, interval: item.ci, error_rate_with_behavior: item.fail_rate_signal, error_rate_other_cases: item.fail_rate_control, passed_screen: item.reject, tool: item.tool }))} /></details></section>;
}

/**
 * Hypotheses M3 proposed with no test_design, from the contract.
 *
 * The legacy card renders "No test design was retained", which says the design
 * existed and was lost. It did not exist: the judge proposed a mechanism and no
 * way to be wrong about it. That distinction decides how to read M4 — such a
 * claim returns INCONCLUSIVE however much evidence the next cycle gathers, and
 * without saying so the reader concludes "needs more data" and runs it again.
 */
function untestableIds(report: ReportData): Set<string> {
  const m3 = findContract<DiagnosisOutput>(report, "m3");
  return new Set((m3?.hypotheses || []).filter((h) => !h.test_design?.trim()).map((h) => h.id));
}

/** A design a person can act on, but that names nothing M4 measured this cycle. */
const SIGNAL_REF = /\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b/;
const DIRECTIVES = ["prompt_contrast", "analyzer_params", "strategy_contrast", "paired_rerun"];
function isRoutable(design?: string): boolean {
  const t = (design || "").trim();
  if (!t) return false;
  return SIGNAL_REF.test(t.toLowerCase()) || DIRECTIVES.some((d) => t.includes(d));
}

function M3Detail({ data, report }: { data: any; report: ReportData }) {
  const accepted = data.hypotheses || [];
  const recovered = data.unparsed_proposals || [];
  const hypotheses = accepted.length ? accepted : recovered;
  const m3 = findContract<DiagnosisOutput>(report, "m3");
  const untestable = m3 ? (m3.hypotheses || []).filter((h) => !h.test_design?.trim()) : [];
  const unroutable = m3
    ? (m3.hypotheses || []).filter((h) => h.test_design?.trim() && !isRoutable(h.test_design))
    : [];
  return <>
    <StageBanner kind="PROPOSAL ONLY" title="Falsifiable mechanisms">M3 turns M2 leads into explanations that could be proven wrong. These cards are proposals; validation status belongs exclusively to M4.</StageBanner>
    {untestable.length > 0 && <div className="parser-warning"><AlertTriangle /><div>
      <b>{untestable.length === 1 ? "One proposal names no test" : `${untestable.length} proposals name no test`}.</b>
      <p>A hypothesis with no test design cannot be decided by any amount of evidence — M4 will return “inconclusive” for it on every cycle. Read that verdict as “this claim was never testable”, not as “not enough data yet”.</p>
    </div></div>}
    {unroutable.length > 0 && <div className="parser-warning"><AlertTriangle /><div>
      <b>{unroutable.length === 1 ? "One proposal names a measurement nobody has taken yet" : `${unroutable.length} proposals name measurements nobody has taken yet`}.</b>
      <p>These do describe an experiment — they just refer to something this cycle did not measure, so M4 has nothing to resolve them against. That is work for the next round of checks, not a claim without a falsifier.</p>
    </div></div>}
    <StageKpis items={[{ label: "Accepted proposals", value: accepted.length }, { label: "Recovered from transcript", value: recovered.length }, { label: "Test designs", value: hypotheses.filter((item: any) => item.test_design).length }]} />
    {!accepted.length && recovered.length > 0 && <div className="parser-warning"><AlertTriangle /><div><b>The AI Doctor proposed hypotheses, but the pipeline parser rejected their format.</b><p>They are shown below for audit only and did not unlock M4 or M5.</p></div></div>}
    {data.evidence_figures?.length > 0 && <section className="m3-evidence"><header><span>THE VISUAL EVIDENCE THIS STEP STARTS FROM</span><h3>Patterns the agent is trying to explain</h3><p>These charts come from M2. They are observations that motivate the ideas below, not confirmation that an idea is true.</p></header><div className="analysis-figures">{labelledFigures(data.evidence_figures, data.evidence_stats || []).map((figure: any) => <EvidenceFigure figure={figure} key={figure.id} />)}</div></section>}
    {data.evidence_stats?.length > 0 && <StatEvidenceChart stats={data.evidence_stats} title="The strongest M2 patterns carried into this step" note="M3 turns these observed patterns into testable ideas. M4 is still needed to decide whether an idea holds up." />}
    {hypotheses.length ? <div className="hypothesis-list">{hypotheses.map((hypothesis: any, index: number) => <article className="hypothesis-card" key={index}><header><span>H{index + 1}</span><em>{accepted.length ? "IDEA TO TEST" : "NOT YET USABLE"}</em></header><h3>{hypothesis.plain_statement || hypothesis.statement || hypothesis.hypothesis}</h3>{hypothesis.statement && hypothesis.plain_statement && hypothesis.statement !== hypothesis.plain_statement && <details><summary>Technical wording</summary><p>{hypothesis.statement}</p></details>}<div className="hypothesis-grid"><div><small>WHAT MAY BE GOING WRONG</small><p>{plainFailureMode(hypothesis.failure_mode)}</p></div><div><small>WHY THIS IS PLAUSIBLE</small><p>{hypothesis.basis || "Based on the patterns found in the previous step."}</p></div><div className="test-design"><small>WHAT WOULD PROVE IT WRONG?</small><p>{hypothesis.test_design || (m3 ? "Nothing — the AI Doctor proposed no test for this idea, so no result can decide it." : "No test design was retained.")}</p>{hypothesis.expected_association && <details><summary>Technical test expression</summary><code>{hypothesis.expected_association}</code></details>}</div></div></article>)}</div> : <EmptyStage title="No formal hypotheses" body="The earlier pattern search did not yield an idea the pipeline could test." />}
    {(data.candidate_signals?.length > 0 || data.recommended_tests?.length > 0) && <details className="agent-transcript"><summary>Candidate signals and suggested follow-ups</summary>{data.candidate_signals?.length > 0 && <RecordTable rows={data.candidate_signals} />}{data.recommended_tests?.map((item: any, i: number) => <p key={i}>• {String(item)}</p>)}</details>}
    {data.agent_response && <details className="agent-transcript"><summary>AI Doctor raw response</summary><pre>{data.agent_response}</pre></details>}
  </>;
}

/**
 * Separates "the evidence was weak" from "the claim was never decidable".
 *
 * Both render as INCONCLUSIVE, and only the M3<->M4 join can tell them apart:
 * match each verdict's hypothesis_id back to the proposal and check whether it
 * carried a test_design. Without this the reader sees "inconclusive", concludes
 * "gather more data", and the next cycle returns the same verdict for the same
 * reason. The join is why the ids on both sides have to agree.
 */
function UndecidableNote({ report }: { report: ReportData }) {
  const m4 = findContract<HypothesisTestOutput>(report, "m4");
  if (!m4) return null;
  const untestable = untestableIds(report);
  const stuck = (m4.results || []).filter(
    (r) => r.status === "inconclusive" && untestable.has(r.hypothesis_id),
  );
  if (!stuck.length) return null;
  return <div className="parser-warning"><AlertTriangle /><div>
    <b>{stuck.length === 1 ? "One “inconclusive” verdict is not a shortage of evidence." : `${stuck.length} “inconclusive” verdicts are not a shortage of evidence.`}</b>
    <p>
      {stuck.map((r) => r.hypothesis_id).join(", ")} came back inconclusive because the
      proposal named no test, so nothing measured here could have decided it either way.
      Re-running with more cases returns the same verdict. The fix belongs in M3.
    </p>
  </div></div>;
}

function M4Detail({ data, report }: { data: any; report: ReportData }) {
  const results = data.results || [];
  if (!data.ran || !results.length) return <><StageBanner kind="CONFIRMATORY" title="Held-out validation">M4 tests frozen hypotheses on evidence not used to propose them.</StageBanner><EmptyStage title="Validation was not reached" body="No accepted M3 hypothesis was available for independent adjudication in this run." /></>;
  const statuses = (name: string) => results.filter((item: any) => String(item.status || "").toLowerCase() === name).length;
  const consistent = results.filter((item: any) => item.protocol_consistent !== false).length;
  const examples = data.examples || [];
  return <>
    <StageBanner kind="CONFIRMATORY" title="Independent check">First we freeze a possible explanation. Then we test it on fresh evidence the agent did not use to invent the explanation. This is the stage allowed to say whether the idea held up.</StageBanner>
    <StageKpis items={[{ label: "Hypotheses checked", value: results.length }, { label: "Supported", value: statuses("supported") }, { label: "Refuted", value: statuses("refuted") }, { label: "Inconclusive", value: statuses("inconclusive") }, { label: "Answered the question asked", value: `${consistent}/${results.length}` }]} />
    <UndecidableNote report={report} />
    {examples.length > 0 && <ExampleSection eyebrow="A validation example" title="How we decide whether an explanation survives" note="This is one recorded validation check. Its conclusion uses the complete independent test set—not a hand-picked example."><div className="example-deck">{examples.map((example: any) => <M4Example example={example} report={report} key={example.id} />)}</div></ExampleSection>}
    <div className="verdict-list">{results.map((result: any, index: number) => <VerdictCard result={result} fallback={data.event} index={index} key={index} />)}</div>
  </>;
}

function M4Example({ example, report }: { example: any; report?: ReportData }) {
  const status = String(example.status || "inconclusive").replaceAll("_", " ");
  const isCase = example.kind === "validation_case";
  return <article className="example-card m4-example"><header><span className="example-step">1 · FROZEN IDEA</span><em className={`status status-${status}`}>{plainStatus(status)}</em></header><h4>{example.hypothesis || "A proposed explanation"}</h4>{isCase && <><span className="example-step">2 · ONE FRESH CASE IN THE TEST POOL</span><ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} /><div className="example-prompt">{example.input || "The original task text was not retained."}</div><div className="example-flow"><div><small>MODEL ANSWER</small><b>{displayValue(example.baseline_output)}</b></div><div><small>EXPECTED ANSWER</small><b>{displayValue(example.expected)}</b></div></div></>}<div className="validation-flow"><div><span>2 · INDEPENDENT TEST</span><b>{plainTest(example.test)}</b></div><div><span>3 · RESULT</span><b>{plainStatus(status)}</b>{example.effect !== undefined && example.effect !== null && <small>Measured difference: {number(example.effect)}</small>}{example.interval && <small>Likely range: {formatInterval(example.interval)}</small>}</div></div>{example.verdict && <p className="example-verdict">{example.verdict}</p>}<footer>{example.plain_reading}</footer></article>;
}

function VerdictCard({ result, fallback, index }: { result: any; fallback: any; index: number }) {
  const evidence = result.evidence || fallback?.evidence || {};
  const status = String(result.status || fallback?.status || "inconclusive").toLowerCase();
  const hypothesis = result.hypothesis || result.statement || fallback?.hypothesis || `Hypothesis ${index + 1}`;
  const icon = status === "supported" ? <CheckCircle2 /> : status === "refuted" ? <XCircle /> : <AlertTriangle />;
  const ci = evidence.ci || result.ci;
  return <article className={`verdict-card verdict-${status}`}><header>{icon}<div><span>H{index + 1} · INDEPENDENT CHECK</span><strong>{plainStatus(status)}</strong></div></header><h3>{hypothesis}</h3><div className="verdict-metrics"><span><small>MEASURED DIFFERENCE</small><b>{number(result.effect_size ?? evidence.effect_size)}</b></span><span><small>CONFIDENCE</small><b>{percent(result.confidence ?? fallback?.confidence_score)}</b></span><span><small>LIKELY RANGE</small><b>{formatInterval(ci)}</b></span><span><small>TYPE OF EVIDENCE</small><b>{plainEvidence(result.evidence_grade || evidence.evidence_grade)}</b></span></div><p className="verdict-reason">{plainVerdict(result.verdict || evidence.m4_verdict || "No validation explanation was retained.")}</p><div className="verdict-foot"><span className={result.protocol_consistent === false ? "bad" : "good"}>{result.protocol_consistent === false ? "Does not match the requested evaluation" : "Matches the requested evaluation"}</span><span>{evidence.fdr?.method ? "Multiple-comparison check applied" : "Independent evidence"}</span></div><details><summary>Technical audit evidence</summary><pre>{JSON.stringify(evidence, null, 2)}</pre></details></article>;
}

/**
 * The candidates a run tried while CHOOSING one, from the contract.
 *
 * These never reached the legacy stage view, which carries only what was
 * confirmed. A run that swept seven candidates across L1 and L2 and confirmed
 * one L1 therefore displayed a single L1 row, and every reader concluded L2 was
 * never attempted. What got ruled out is part of what the run found.
 *
 * Kept visually apart from the confirmation, and labelled, because these
 * numbers CHOSE the candidate and so cannot also test it — presenting them in
 * one table would invite exactly the double-dip the two-stage protocol exists
 * to prevent.
 */
/**
 * Every repair the run tried, keyed by the identifier the producer assigned.
 *
 * `name` is a slug the agent chose while naming its log folder — it is a join
 * key, not language, and putting it on screen made readers try to decode it.
 * The contract carries `ref` ("R3") to point at a repair and `headline` to say
 * what it does; both come from the producer, which is the only place that
 * knows. A candidate with no headline is left blank rather than shown as its
 * slug in title case, which would read as an explanation the run never gave.
 */
function repairRefs(report: ReportData): Map<string, FixAttemptWire> {
  const m5 = findContract<FixOutput>(report, "m5_fix");
  const byName = new Map<string, FixAttemptWire>();
  for (const row of [...(m5?.selection || []), ...(m5?.attempted || [])]) {
    if (!byName.has(row.name)) byName.set(row.name, row);
  }
  return byName;
}

function refFor(report: ReportData, name?: string) {
  if (!name) return "";
  const row = repairRefs(report).get(name);
  if (row?.ref) return row.ref;
  const order = [...repairRefs(report).keys()].indexOf(name);
  return order >= 0 ? `R${order + 1}` : "";
}

function headlineFor(report: ReportData, name?: string) {
  return (name && repairRefs(report).get(name)?.headline) || "";
}

function SelectionSweep({ report }: { report: ReportData }) {
  const m5 = findContract<FixOutput>(report, "m5_fix");
  const rows = m5?.selection || [];
  if (!rows.length) return null;
  const tiers = [...new Set(rows.map((r) => r.tier))].sort();
  const anyHeadline = rows.some((r) => r.headline);
  const chosen = m5?.selected_on_explore;
  const tone = (v?: string) => v === "fixed" ? "ok"
    : v === "unsafe" || v === "regressed" ? "bad"
    : v === "partial" || v === "model_independent" ? "warn" : "";
  // The verdicts are contract enum values, and three of the seven do not
  // survive being read literally: "no effect" invites "so it did nothing",
  // which is the result; "model independent" means the scaffold, not the
  // model, produced the answer.
  const plainVerdictWord = (v?: string) => ({
    fixed: "fixed errors",
    partial: "helped some, hurt some",
    unsafe: "broke more than it fixed",
    regressed: "made things worse",
    no_effect: "changed nothing",
    not_executed: "could not run",
    model_independent: "the scaffold answered, not the model",
  } as Record<string, string>)[String(v || "")] || String(v || "").replace(/_/g, " ");
  return <section className="phase">
    <span className="eyebrow">Stage 1 · choosing — not evidence</span>
    <h3>{rows.length === 1 ? "One repair was tried" : `${rows.length} repairs were tried`}</h3>
    <p className="stat-caption">
      Each was run on the diagnosis cases to pick one worth confirming. These numbers chose
      the repair, so they cannot also test it — read them as what was ruled out, never as
      results.{chosen ? ` ${refFor(report, chosen) || chosen} was taken forward.` : ""}
    </p>
    <div className="scroll"><table>
      <thead><tr>{["", "#", "Tier", ...(anyHeadline ? ["What it changes"] : []),
        "Result", "Errors fixed", "New errors", "Net change"]
        .map((h, i) => <th key={`${h}-${i}`}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={`${r.name}-${i}`}
        className={chosen && r.name === chosen ? "row-chosen" : undefined}>
        <td><span className={`chip ${tone(r.verdict)}`}><span className="dot" /></span></td>
        <td className="repair-ref">{r.ref || `R${i + 1}`}</td>
        <td>{r.tier}</td>
        {anyHeadline && <td className="txt">{r.headline || <em className="muted">not described</em>}</td>}
        <td>{plainVerdictWord(r.verdict)}</td>
        <td>{r.n_fixed}</td>
        <td>{r.n_broken}</td>
        <td>{r.effect === null || r.effect === undefined ? "—"
          : `${r.effect >= 0 ? "+" : ""}${Number(r.effect).toFixed(3)}`}</td>
      </tr>)}</tbody>
    </table></div>
    <p className="mono-sm">
      “Net change” is the share of cases that improved minus the share that got worse, so
      +0.083 means about eight cases in a hundred came out better than before.
      {tiers.length > 1 && ` Tiers reached: ${tiers.join(", ")}. A tier missing here had no
        repair to offer — for L3a that usually means the model exposes no internals to read.`}
    </p>
  </section>;
}

/**
 * The case ids a candidate repaired or broke, as links into the Case Studio.
 *
 * "12 repaired, 1 broken" is two numbers; these are thirteen cases someone can
 * read. A case the report has no record of is shown greyed and unclickable
 * rather than as a dead link — on a run whose held-out split went unlogged that
 * is every one of them, and the reader needs to know that is why.
 */
function CaseLinks({ ids, kind, report, navigate }: {
  ids?: string[]; kind: "fixed" | "broken";
  report: ReportData; navigate?: (view: string) => void;
}) {
  if (!ids?.length) return null;
  const known = new Set(report.cases.map((c) => c.id));
  const missing = ids.filter((id) => !known.has(id)).length;
  return <div className="case-links">
    <small>{kind === "fixed" ? "REPAIRED" : "BROKEN"} ({ids.length})</small>
    <div>{ids.map((id) => known.has(id)
      ? <button className={`case-chip ${kind}`} key={id}
          onClick={() => navigate?.(`cases:${id}`)}>{id}</button>
      : <span className="case-chip missing" key={id} title="No record of this case in the report">{id}</span>)}
    </div>
    {missing > 0 && <em>
      {missing === ids.length ? "None of these cases" : `${missing} of these cases`} were
      recorded in this run, so they cannot be opened. The repair was validated on the
      held-out split, and this run logged only the diagnosis split.
    </em>}
  </div>;
}

function M5Detail({ data, report, navigate }: { data: any; report: ReportData; navigate?: (view: string) => void }) {
  const candidates = data.candidates || [];
  if (!data.ran || !candidates.length) return <><StageBanner kind="INTERVENTION" title="Repair and regression check">M5 compares targeted changes against the same unmodified baseline cases.</StageBanner><SelectionSweep report={report} /><EmptyStage title={data.skipped ? "Repair was deliberately held back" : data.ran ? "No repair candidate was testable" : "Repair was not reached"} body={data.skipped ? (data.skip_detail || "The evidence review did not yet accept a mechanism for repair. The next step is a targeted diagnostic probe, not a failed repair.") : data.ran ? "The stage opened, but no accepted and testable mechanism produced a repair candidate." : "The run stopped before a targeted intervention could be evaluated."} /></>;
  const fixed = candidates.reduce((sum: number, item: any) => sum + Number(item.n_fixed || 0), 0);
  const broken = candidates.reduce((sum: number, item: any) => sum + Number(item.n_broken || 0), 0);
  const winner = candidates.find((item: any) => item.fixed) || candidates.reduce((best: any, item: any) => Number(item.effect || -Infinity) > Number(best?.effect || -Infinity) ? item : best, null);
  const option = { grid: { left: 145, right: 24, top: 18, bottom: 32 }, color: [tc("#6bd8ad"), tc("#f06d5f")], tooltip: { trigger: "axis" }, legend: { textStyle: { color: tc("#9fb2ac") } }, xAxis: { type: "value", axisLabel: { color: tc("#8fa49d") }, splitLine: { lineStyle: { color: tc("#22332e") } } }, yAxis: { type: "category", data: candidates.map((item: any, i: number) => `${refFor(report, item.name) || `R${i + 1}`} · ${item.tier || "?"}`).reverse(), axisLabel: { color: tc("#b8c9c4"), width: 130, overflow: "truncate" } }, series: [{ name: "Repaired", type: "bar", stack: "cases", data: candidates.map((item: any) => item.n_fixed || 0).reverse() }, { name: "Broken", type: "bar", stack: "cases", data: candidates.map((item: any) => item.n_broken || 0).reverse() }] };
  return <>
    <StageBanner kind="INTERVENTION" title="Paired repair sweep">Every candidate is compared case-by-case with the unchanged model. A useful repair must fix failures without breaking cases that were right before, and its lead must hold up after discounting for how many candidates were tried at once.</StageBanner>
    <StageKpis items={[{ label: "Candidates tried", value: candidates.length }, { label: "Repaired flips", value: fixed }, { label: "Broken flips", value: broken }, { label: "Best repair", value: refFor(report, winner?.name) || (winner ? `R${candidates.indexOf(winner) + 1}` : "—"), note: headlineFor(report, winner?.name) || winner?.headline || winner?.tier }]} />
    <div className={`repair-outcome ${data.fixed ? "success" : "neutral"}`}><Wrench /><div><span>FINAL REPAIR OUTCOME</span><h3>{data.fixed ? "A repair was confirmed" : "No candidate passed the repair gate"}</h3>{/* The sentence under the verdict was `candidate.summary` — the audit line,
      "[mcnemar + e-value (paired binary)] ... -> inconclusive". Worse than
      unreadable, it sat beside "Repaired flips: 15" with nothing explaining
      how both are true. The derived sentence answers that: it fixed 15 and
      broke 6, and the evidence against luck is not strong enough to count.
      The audit line stays under the candidate's details toggle. */}
    {(headlineFor(report, winner?.name) || winner?.headline)
      ? <p>{headlineFor(report, winner?.name) || winner?.headline}</p>
      : winner ? <RepairVerdict candidate={winner} /> : <p>Inspect the full repair sweep below.</p>}</div></div>
    {data.examples?.length > 0 && <M5ExampleSection examples={data.examples} report={report} />}
    {data.operation_previews?.length > 0 && <RepairOperationPreviews examples={data.operation_previews} report={report} />}
    <SelectionSweep report={report} />
    <div className="repair-chart"><h3>Confirmed on held-out cases: paired flips vs. baseline</h3><ReactECharts option={option} style={{ height: Math.max(300, candidates.length * 48) }} /></div>
    <div className="candidate-list">{candidates.map((candidate: any, index: number) => <article className={`candidate-card candidate-${candidate.verdict || "unknown"}`} key={`${candidate.name}-${index}`}><header><span>{refFor(report, candidate.name) || `R${index + 1}`}<em>{candidate.tier || "?"}</em></span><div>{/* The producer's own headline, from the contract or the stage row. The
      raw `summary` is the audit line ("[mcnemar + e-value …] -> REJECT H0") and
      set as a card TITLE it is the first thing the reader met; it stays intact
      under the details toggle below. */}
    <h3>{headlineFor(report, candidate.name) || candidate.headline || plainCandidateKind(candidate.kind)}</h3><small className="mono-sm">{candidate.name}</small></div><b>{candidate.fixed ? "helped" : "did not pass"}</b></header><div className="candidate-metrics"><span><b>{candidate.n_fixed ?? 0}</b> errors fixed</span><span><b>{candidate.n_broken ?? 0}</b> new errors</span><span><b>{number(candidate.effect)}</b> net change</span><span><b>{percent(candidate.coverage)}</b> of errors covered</span></div><RepairVerdict candidate={candidate} /><details><summary>Technical repair definition and affected cases</summary>{candidate.summary && <pre className="stat-line">{candidate.summary}</pre>}<KeyValueGrid values={candidate.payload || {}} /></details><CaseLinks ids={candidate.fixed_cases} kind="fixed" report={report} navigate={navigate} /><CaseLinks ids={candidate.broken_cases} kind="broken" report={report} navigate={navigate} /></article>)}</div>
  </>;
}

/**
 * What the paired test found, as a sentence.
 *
 * The producer's own line reads
 * `[mcnemar + e-value (paired binary)] effect=+0.1833 (B>A) CI=+0.0833..+0.3000,
 * e=45.01 -> REJECT H0`. That is the audit record and it stays reachable under
 * the details toggle, but it is not an answer to "did this repair work?" for
 * anyone who has not met an e-value. The numbers here are the same numbers.
 */
function RepairVerdict({ candidate }: { candidate: any }) {
  const fixed = Number(candidate.n_fixed || 0);
  const broke = Number(candidate.n_broken || 0);
  const pairs = Number(candidate.n_pairs || 0);
  const e = typeof candidate.e_value === "number" && Number.isFinite(candidate.e_value)
    ? candidate.e_value : null;
  // An e-value is odds: e=45 means the evidence runs about 45 to 1 against
  // this being luck. "Reject at 0.05" is the same statement in a dialect
  // nobody outside the field speaks.
  const odds = e === null ? null
    : e >= 1000 ? "over 1000 to 1"
    : e >= 10 ? `about ${Math.round(e)} to 1`
    : `about ${e.toFixed(1)} to 1`;
  const counted = pairs > 0
    ? `Across ${pairs} case${pairs === 1 ? "" : "s"} it was tested on, this repair fixed ${fixed} and broke ${broke}.`
    : `This repair fixed ${fixed} case${fixed === 1 ? "" : "s"} and broke ${broke}.`;
  const judged = odds === null
    ? "The run recorded no significance test for it."
    : candidate.reject
      ? `The evidence against that being luck runs ${odds}, which is strong enough to count.`
      : `The evidence against that being luck runs only ${odds}, which is not strong enough to count.`;
  const covered = typeof candidate.coverage === "number" && Number.isFinite(candidate.coverage)
    ? ` It applied to ${percent(candidate.coverage)} of the failures it was aimed at.` : "";
  const independent = Number(candidate.n_model_independent || 0) > 0
    ? ` ${candidate.n_model_independent} case${candidate.n_model_independent === 1 ? " was" : "s were"} solved by the added code rather than by the model, and were left out of the count.`
    : "";
  return <p className="repair-verdict">{counted} {judged}{covered}{independent}</p>;
}

/**
 * The example deck wrapper, worded for whichever case M5's search actually
 * produced. A confirmed repair gets the "accepted" framing it earned; an
 * unconfirmed one is still shown — that is the whole point, a reader
 * debugging a failed search needs a real case, not just a gate that closed —
 * but the copy around it says plainly that nothing here was accepted.
 */
function M5ExampleSection({ examples, report }: { examples: any[]; report: ReportData }) {
  const first = examples[0] || {};
  const confirmed = first.confirmed !== false;
  const broke = !confirmed && first.kind === "broken";
  const eyebrow = confirmed ? "A repaired case" : broke ? "What the search actually broke" : "The search's best attempt";
  const title = confirmed ? "One real before-and-after repair"
    : broke ? "No case was fixed — here is one that broke"
    : "One real attempt, not an accepted repair";
  const note = confirmed
    ? "This is a case counted as fixed. The repair was accepted only after checking every paired case for improvements and regressions."
    : "No candidate cleared the significance bar, so nothing below was accepted as a repair. This is the strongest attempt's own case — shown so the search is debuggable, not just marked failed.";
  return <ExampleSection eyebrow={eyebrow} title={title} note={note}><div className="example-deck">{examples.map((example: any) => <M5Example example={example} report={report} key={example.id} />)}</div></ExampleSection>;
}

function M5Example({ example, report }: { example: any; report: ReportData }) {
  const confirmed = example.confirmed !== false;
  const broke = example.kind === "broken";
  const label = confirmed ? "A CASE THE REPAIR HELPED" : broke ? "A CASE THE ATTEMPT BROKE" : "A CASE THE ATTEMPT HELPED — UNCONFIRMED";
  const status = confirmed ? "fixed" : broke ? "broken" : "unconfirmed";
  const statusLabel = confirmed ? "fixed" : broke ? "broken" : "not confirmed";
  return <article className="example-card m5-example"><header><span className="example-step">{label}</span><em className={`status status-${status}`}>{statusLabel}</em></header><ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} /><div className="example-prompt">{example.input || "The original task text was not retained."}</div><div className="example-flow"><div><small>{broke ? "BEFORE ATTEMPT" : "BEFORE REPAIR"}</small><b>{example.baseline_available ? displayValue(example.baseline_output) : "Not retained"}</b></div><div><small>{broke ? "AFTER ATTEMPT" : "AFTER REPAIR"}</small><b>{displayValue(example.repaired_output)}</b></div><div><small>EXPECTED ANSWER</small><b>{displayValue(example.expected)}</b></div></div><footer>{example.plain_reading}</footer></article>;
}

function OperationExamples({ examples, report }: { examples: any[]; report: ReportData }) {
  return <ExampleSection eyebrow="Recorded agent actions" title="See the intermediate work, case by case" note="These actions come from the saved agent trajectory. They are not a reconstruction from a prompt or a proposed repair."><div className="operation-deck">{examples.map((example) => <article className="operation-card" key={example.id}><header><span>CASE {example.case_id}</span><em className={`status status-${String(example.outcome || "unknown").toLowerCase()}`}>{example.outcome}</em></header><ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} /><p className="operation-prompt">{example.input}</p><div className="operation-steps">{example.steps.map((step: any) => <div key={`${step.order}-${step.raw_action}`}><span>{step.order}</span><section><b>{step.action}</b>{Object.keys(step.parameters || {}).length > 0 && <small>{Object.entries(step.parameters).map(([key, value]) => `${humanize(key)}: ${formatCompact(value)}`).join(" · ")}</small>}{step.thought && <p>{step.thought}</p>}</section></div>)}</div><footer>{example.plain_reading}</footer></article>)}</div></ExampleSection>;
}

function RepairOperationPreviews({ examples, report }: { examples: any[]; report: ReportData }) {
  return <ExampleSection eyebrow="A repair operation, made concrete" title="What the image-aware repair would do to a real case" note="This shows the source case and the operation declared by the repair candidate. It does not claim an altered image or a successful answer unless the run recorded one."><div className="operation-deck">{examples.map((example) => <article className="operation-card repair-operation" key={example.id}><header><span>{example.executed ? "CANDIDATE WAS TESTED" : "CANDIDATE PREVIEW — NOT EXECUTED"}</span><em>{refFor(report, example.candidate) || example.candidate}</em></header><ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} /><p className="operation-prompt">{example.input}</p><div className="example-flow"><div><small>BASELINE ANSWER</small><b>{displayValue(example.observed)}</b></div><div><small>EXPECTED ANSWER</small><b>{displayValue(example.expected)}</b></div></div><div className="operation-steps">{example.operations.map((operation: any, index: number) => <div key={`${operation.raw_action}-${index}`}><span>{index + 1}</span><section><b>{operation.action}</b>{Object.keys(operation.parameters || {}).length > 0 && <small>{Object.entries(operation.parameters).map(([key, value]) => `${humanize(key)}: ${formatCompact(value)}`).join(" · ")}</small>}</section></div>)}</div></article>)}</div></ExampleSection>;
}

function EmptyStage({ title, body }: { title: string; body: string }) { return <div className="empty-stage"><ShieldCheck /><div><h3>{title}</h3><p>{body}</p></div></div>; }

function KeyValueGrid({ values }: { values: Record<string, any> }) { return <div className="kv-grid">{Object.entries(values).map(([key, value]) => <div key={key}><small>{humanize(key)}</small><b>{formatCompact(value)}</b></div>)}</div>; }

function RecordTable({ rows }: { rows: any[] }) {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row || {})))].slice(0, 8);
  return <div className="record-table-wrap"><table className="record-table"><thead><tr>{columns.map((column) => <th key={column}>{humanize(column)}</th>)}</tr></thead><tbody>{rows.slice(0, 16).map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{formatCompact(row?.[column])}</td>)}</tr>)}</tbody></table></div>;
}

function normalizeKey(value: any) { return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, ""); }
function humanize(value: string) { return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase()); }
function plainStatus(value: string) { return ({ supported: "held up", refuted: "did not hold up", inconclusive: "not enough evidence" } as Record<string, string>)[value.toLowerCase()] || humanize(value); }
function plainEvidence(value: any) { return ({ observational: "pattern in the data", intervention: "direct comparison", none: "no usable evidence" } as Record<string, string>)[String(value || "").toLowerCase()] || "recorded evidence"; }
function plainTest(value: any) { return ({ signal_label_assoc: "Compare this behavior between correct and incorrect answers", rank_corr: "Check whether the behavior and errors move together", single_rate_evalue: "Check whether the rate could plausibly be chance" } as Record<string, string>)[String(value || "")] || "Independent statistical check"; }
function plainVerdict(value: string) { return String(value).replace(/signal_label_assoc: signal '[^']+' vs FAIL:\s*/i, "").replace(/\[clustered bootstrap \(unpaired\)\]\s*/i, "").replace(/\s*\[BH:.*?\]/i, ""); }
function plainFailureMode(value: any) { return ({ computation_slip: "The model may be making a calculation mistake", hallucination: "The model may be reading information that is not there", brittleness: "The model may be overly sensitive to a small change", format_error: "The model may understand the task but reply in the wrong format" } as Record<string, string>)[String(value || "").toLowerCase()] || (value ? humanize(String(value)) : "Not specified"); }
function plainCandidateKind(value: any) { return ({ prompt: "A change to the instructions", prompt_template: "A change to the instructions", decoding: "A change to how answers are generated", intervention: "A targeted model intervention" } as Record<string, string>)[String(value || "").toLowerCase()] || "A proposed repair"; }
function plainOperation(value: any) { return ({ imagezoomin: "Zoom in on part of the image", zoomcenter: "Zoom into the chart center", crop: "Crop the visual evidence", cropcasebbox: "Crop the relevant chart region", sharpen: "Sharpen chart details", contrast: "Increase visual contrast", detect: "Locate an object or region" } as Record<string, string>)[normalizeKey(value)] || humanize(String(value || "tool")); }
function seconds(value: any) { return typeof value === "number" ? `${value.toFixed(1)}s` : "—"; }
function number(value: any) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "") : "—"; }
function percent(value: any) { return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—"; }
function formatInterval(value: any) { return Array.isArray(value) && value.length >= 2 ? `[${number(value[0])}, ${number(value[1])}]` : "—"; }
function formatCompact(value: any) { if (value === null || value === undefined || value === "") return "—"; if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, ""); if (typeof value === "object") return JSON.stringify(value).slice(0, 180); return String(value).slice(0, 220); }
function displayValue(value: any) { return formatCompact(value); }

export function CasesView({ data, back, initialCaseId }: { data: ReportData; back: () => void; initialCaseId?: string }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [selected, setSelected] = useState<Case | null>(
    initialCaseId ? data.cases.find((c) => c.id === initialCaseId) || null : null,
  );
  // fixed / broken / unchanged describe what the REPAIR did; pass / fail
  // describe the original run. Filtering on the wrong one silently returns
  // nothing, so each button reads the field it names.
  const repairFilters = new Set(["fixed", "broken", "unchanged"]);
  const rows = data.cases.filter((item) => {
    const matches = repairFilters.has(status)
      ? item.repair?.status === status
      : status === "all" || item.status === status;
    return matches && `${item.id} ${item.prompt} ${item.task}`.toLowerCase().includes(query.toLowerCase());
  });
  const parentRef = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({ count: rows.length, getScrollElement: () => parentRef.current, estimateSize: () => 92, overscan: 8 });
  return <DetailShell title="Case Studio" subtitle="Inspect the actual input, expected answer, model output, and attached media." back={back} agent={data.setting.diagnosed_by}>
    <div className="case-toolbar"><label><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search cases" /></label><div>{["all", "fail", "pass", "fixed", "broken", "unchanged"].map((value) => <button className={status === value ? "active" : ""} onClick={() => setStatus(value)} key={value}>{value}</button>)}</div><span>{rows.length} cases</span></div>
    <div className="case-studio"><div className="virtual-list" ref={parentRef}><div style={{ height: virtual.getTotalSize(), position: "relative" }}>{virtual.getVirtualItems().map((row) => { const item = rows[row.index]; return <button className={`virtual-row ${selected?.id === item.id ? "active" : ""}`} key={item.id} onClick={() => setSelected(item)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: row.size, transform: `translateY(${row.start}px)` }}><span className={`status status-${item.status}`}>{item.status}</span><div><strong>{item.id}</strong><p>{item.prompt || "No prompt retained"}</p></div><ChevronRight /></button>; })}</div></div><CaseDetail item={selected || rows[0]} data={data} /></div>
  </DetailShell>;
}

/**
 * The same case before and after M5's repair, side by side.
 *
 * A repair reported as "12 repaired, 1 broken" is only reviewable if you can
 * read what changed. The gold answer sits between the two so the direction is
 * legible without arithmetic: which side matches it is the whole finding.
 */
function RepairStudy({ item }: { item: Case }) {
  const r = item.repair;
  if (!r?.output) return null;
  const flipped = r.status === "fixed" || r.status === "broken";
  return <div className={`repair-study ${r.status || ""}`}>
    <header>
      <span className="eyebrow">
        {r.status === "fixed" ? "The repair corrected this case"
          : r.status === "broken" ? "The repair broke this case"
          : "The repair did not change this case"}
      </span>
      {r.candidate && <em>{r.tier} · {r.candidate}</em>}
    </header>
    <div className="repair-cols">
      <div><small>BEFORE — UNCHANGED MODEL</small><b>{displayValue(item.observed)}</b></div>
      <div className="gold"><small>CORRECT ANSWER</small><b>{displayValue(item.expected)}</b></div>
      <div><small>AFTER — WITH THE REPAIR</small><b>{displayValue(r.output)}</b></div>
    </div>
    {!flipped && <p>Both answers land on the same side of the gold answer, so this
      case counts toward neither the repairs nor the regressions.</p>}
  </div>;
}

function CaseDetail({ item, data }: { item?: Case; data: ReportData }) {
  if (!item) return <section className="case-detail empty">No cases match this filter.</section>;
  return <section className="case-detail"><div className="case-detail-head"><div><span className={`status status-${item.status}`}>{item.status}</span>{item.repair?.status && item.repair.status !== "unchanged" && <span className={`status status-${item.repair.status}`}>{item.repair.status} by repair</span>}<h2>{item.id}</h2></div><small>{item.task}</small></div><RepairStudy item={item} /><DetailBlock label="MODEL INPUT" value={item.prompt} />{item.choices?.length > 0 && <DetailBlock label="CHOICES" value={item.choices.map(String).join("\n")} />}<div className="io-grid"><DetailBlock label="EXPECTED" value={format(item.expected)} /><DetailBlock label="MODEL OUTPUT" value={format(item.observed)} /></div>{item.media_ids.map((id) => { const media = data.media.find((entry) => entry.id === id); if (!media) return null; return <MediaPreview key={id} media={media} caseId={item.id} />; })}{Boolean(item.trajectory) && <TrajectoryPanel trajectory={item.trajectory} />}{item.tags.length > 0 && <div className="tag-row">{item.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}</section>;
}

function TrajectoryPanel({ trajectory }: { trajectory: any }) {
  const steps = Array.isArray(trajectory?.steps) ? trajectory.steps : [];
  const calls = steps.filter((step: any) => step?.tool_call);
  if (!calls.length) return null;
  return <details className="case-trajectory"><summary>Recorded intermediate agent actions ({calls.length})</summary>{calls.map((step: any, index: number) => <div className="trajectory-row" key={index}><b>{plainOperation(step.tool_call?.name)}</b><small>{formatCompact(step.tool_call?.args)}</small>{step.observation && <p>{formatCompact(step.observation)}</p>}</div>)}</details>;
}

function MediaPreview({ media, caseId }: { media: ReportData["media"][number]; caseId: string }) {
  const source = media.data_uri || `/api/media/${encodeURIComponent(media.id)}`;
  if (media.kind === "image") return <ZoomableImage className="case-media" src={source} alt={`Input for case ${caseId}`} caption={`Input for case ${caseId}`} />;
  if (media.kind === "video") return <video className="case-media" controls preload="metadata" src={source} />;
  return <audio controls preload="metadata" src={source} />;
}

export function DebugView({ data, back }: { data: ReportData; back: () => void }) {
  const [filter, setFilter] = useState("all");
  const rows = filter === "all" ? data.debug.events : data.debug.events.filter((row) => row.event === filter);
  const eventTypes = [...new Set(data.debug.events.map((event) => event.event))];
  const parentRef = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({ count: rows.length, getScrollElement: () => parentRef.current, estimateSize: () => 84, overscan: 10 });
  return <DetailShell title="Agent audit log" subtitle="The most detailed layer: ordered decisions, tools, stage events, and publication provenance." back={back} agent={data.setting.diagnosed_by}><div className="debug-toolbar"><select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">All event types</option>{eventTypes.map((value) => <option key={value}>{value}</option>)}</select><span>{rows.length} / {data.debug.event_count} events</span></div><div className="debug-list" ref={parentRef}><div style={{ height: virtual.getTotalSize(), position: "relative" }}>{virtual.getVirtualItems().map((row) => <div className="debug-row" key={row.key} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: row.size, transform: `translateY(${row.start}px)` }}><EventRow event={rows[row.index]} /></div>)}</div></div></DetailShell>;
}

/**
 * The frame every depth below the overview sits in.
 *
 * `agent` rides along because the overview's topbar — the only other place the
 * run says which agent produced it — is not rendered here, and these are the
 * screens where two runs actually get compared side by side.
 */
function DetailShell({ title, subtitle, back, agent, children }: { title: string; subtitle: string; back: () => void; agent?: string; children: React.ReactNode }) {
  return <main className="detail-page"><nav><button onClick={back}><ArrowLeft size={17} /> Overview</button><div><strong>{title}</strong><span>{subtitle}</span></div>{agent && <span className="shell-agent" title="The agent that drove this run"><Bot size={13} />{agent}</span>}</nav>{children}</main>;
}

function EventRow({ event }: { event: DebugEvent }) {
  return <div className="event-row"><code>{String(event.event_seq || "—").padStart(3, "0")}</code><span className="event-type">{event.event}</span><div><strong>{stageCode(event.stage) || "RUN"}{event.cycle !== undefined && event.cycle !== null ? ` · cycle ${event.cycle}` : ""}</strong><p>{event.summary}</p></div></div>;
}

function DetailBlock({ label, value }: { label: string; value: string }) {
  return <div className="detail-block"><small>{label}</small><pre>{value || "Not retained"}</pre></div>;
}

function format(value: unknown) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "Not retained";
  return JSON.stringify(value, null, 2);
}
