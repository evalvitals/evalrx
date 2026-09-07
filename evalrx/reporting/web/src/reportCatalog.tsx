import { createContext, useContext, useMemo } from "react";
import { tc } from "./theme";
import { defineCatalog } from "@json-render/core";
import { defineRegistry } from "@json-render/react";
import { schema } from "@json-render/react/schema";
import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import ReactECharts from "echarts-for-react";
import { ArrowUpRight, Check, CircleDot, Database, Wrench } from "lucide-react";
import { z } from "zod";
import type { Chart, ReportData, Stage } from "./types";
import { ZoomableImage } from "./lightbox";
import { CaseStudySheet as CaseStudySection } from "./caseStudy";
import { chartValue, outcomeColors, stageCode } from "./reportAccess";

const ids = z.array(z.string()).optional();
export const reportCatalog = defineCatalog(schema, {
  components: {
    ReportPage: { props: z.object({}), description: "The report page container" },
    SettingHero: { props: z.object({}), description: "Model, dataset and diagnostic question" },
    MetricStrip: { props: z.object({}), description: "Compact run metrics" },
    Journey: { props: z.object({}), description: "Visual failure-to-fix M1-M5 journey" },
    FindingGrid: { props: z.object({ findingIds: ids }), description: "At most three evidence findings" },
    ChartGrid: { props: z.object({ chartIds: ids }), description: "At most two evidence charts" },
    OutcomeCard: { props: z.object({}), description: "What was learned and whether repair worked" },
    CasePreview: { props: z.object({ caseIds: ids }), description: "Representative model I/O" },
    CaseStudySheet: { props: z.object({}), description: "The whole run as one failure-to-repair sheet" },
    EvidenceIndex: { props: z.object({}), description: "Progressive disclosure navigation" },
  },
  actions: {},
});

const DataContext = createContext<ReportData | null>(null);
const NavContext = createContext<(view: string) => void>(() => undefined);

export function ReportProviders({ data, navigate, children }: { data: ReportData; navigate: (view: string) => void; children: React.ReactNode }) {
  return <DataContext value={data}><NavContext value={navigate}>{children}</NavContext></DataContext>;
}

function useReport() {
  const data = useContext(DataContext);
  if (!data) throw new Error("Report data is unavailable");
  return data;
}

function StageNode({ data }: NodeProps<Node<{ stage: Stage }>>) {
  const stage = data.stage;
  return <div className={`stage-node stage-${stage.status}`}>
    <Handle type="target" position={Position.Left} />
    <div className="stage-code">{stageCode(stage.code)}</div>
    <strong>{stage.title}</strong>
    <small>{stage.status.replaceAll("-", " ")}</small>
    <Handle type="source" position={Position.Right} />
  </div>;
}

const nodeTypes = { stage: StageNode };

function EvidenceChart({ chart }: { chart: Chart }) {
  const option = chart.kind === "donut" ? {
    tooltip: { trigger: "item", valueFormatter: chartValue },
    // Keyed to what each slice means, not to its position in the series: the
    // compiler emits [Fail, Pass] on one run and [Pass, Fail] on the next, and
    // a positional palette painted the passes red on half the reports.
    color: outcomeColors(chart.series.map((item) => item.label)),
    series: [{ type: "pie", radius: ["54%", "76%"], center: ["50%", "52%"], label: { color: tc("#b8c9c4"), formatter: "{b}  {c}" }, data: chart.series.map((item) => ({ name: item.label, value: item.value })) }],
  } : {
    grid: { left: 168, right: 24, top: 12, bottom: 22 },
    xAxis: { type: "value", splitLine: { lineStyle: { color: tc("#23332f") } }, axisLabel: { color: tc("#8da19b") } },
    // These labels are analyzer questions, not short keys. Truncating eight of
    // them to the same "Did the model g..." left a chart whose bars nobody
    // could tell apart, so they wrap and the chart grows a row at a time.
    yAxis: { type: "category", data: chart.series.map((item) => item.label), axisLabel: { color: tc("#b8c9c4"), width: 154, overflow: "break", lineHeight: 13, fontSize: 11 }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{ type: "bar", data: chart.series.map((item) => ({ value: item.value, itemStyle: { color: item.highlight ? tc("#6bd8ad") : tc("#586a65"), borderRadius: 4 } })), barWidth: 13 }],
  };
  const height = chart.kind === "donut" ? 250 : Math.max(250, chart.series.length * 42 + 40);
  return <article className="chart-card"><h3>{chart.title}</h3>{chart.subtitle && <p>{chart.subtitle}</p>}<ReactECharts option={option} notMerge style={{ height }} />{chart.series.some((item) => item.means || item.raw_label) && <div className="chart-legend">{chart.series.map((item) => <div key={item.raw_label || item.label}><b>{item.label}</b>{item.means ? <span>{item.means}</span> : <em>The analyzer did not document what this measures.</em>}<code>{item.raw_label}</code></div>)}</div>}</article>;
}

export const { registry } = defineRegistry(reportCatalog, {
  components: {
    ReportPage: ({ children }) => <main className="report-page">{children}</main>,
    SettingHero: () => {
      const data = useReport();
      // A run that ships its own cover figure (`evalrx_main.*` beside its
      // baseline.json) gets it rendered between the lead and the route bar,
      // with the whole hero centring around it — the one picture the producer
      // chose to explain the run, ahead of anything derived. Runs without one
      // keep the left-set hero and its decorative rings exactly as they were.
      const hero = data.setting.hero_image;
      return <section className={`setting-hero${hero ? " has-figure" : ""}`}>
        <div className="eyebrow eyebrow-title"><CircleDot size={14} /> {data.setting.model} model's diagnosis and fixing recipe report</div>
        <h1>From model failure<br /><span>to tested repair.</span></h1>
        <p className="lead">{data.setting.question}</p>
        {hero && <div className="hero-figure">
          <ZoomableImage src={hero} alt="The figure this run shipped" caption="evalrx_main — the figure this run shipped" />
        </div>}
        {/* The model is already in the eyebrow, so the route reads as one
            sentence. Two runs of one benchmark against one model differ only
            in the agent that drove them — named when the run recorded it,
            never guessed. */}
        <p className="setting-line">
          Evaluated on dataset <strong>{data.setting.dataset}</strong>.
          {data.setting.diagnosed_by && <> Diagnosed by <strong title={data.setting.diagnosed_by}>{data.setting.diagnosed_by}</strong>.</>}
        </p>
      </section>;
    },
    MetricStrip: () => {
      const data = useReport();
      return <section className="metric-strip">{data.metrics.map((metric) => <div key={metric.id}><strong>{metric.value}</strong><span>{metric.label}</span></div>)}</section>;
    },
    Journey: () => {
      const data = useReport();
      const navigate = useContext(NavContext);
      const nodes: Node<{ stage: Stage }>[] = data.stages.map((stage, index) => ({ id: stage.id, type: "stage", position: { x: index * 205, y: 20 }, data: { stage } }));
      const edges: Edge[] = data.stages.slice(1).map((stage, index) => ({ id: `${data.stages[index].id}-${stage.id}`, source: data.stages[index].id, target: stage.id, animated: !["not-run", "skipped"].includes(stage.status), style: { stroke: tc("#6bd8ad"), strokeWidth: 1.5 } }));
      return <section className="section journey-section"><header><div><span className="section-kicker">THE EVALRX PIPELINE</span><p className="journey-hint">Click any stage to inspect its evidence and the agent events behind it.</p></div></header><div className="journey-canvas"><ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView minZoom={0.6} maxZoom={1.2} nodesDraggable={false} nodesConnectable={false} panOnScroll={false} onNodeClick={(_, node) => navigate(`evidence:${node.id}`)}><Background color={tc("#24332f")} gap={22} size={1} /><Controls showInteractive={false} /></ReactFlow></div></section>;
    },
    FindingGrid: ({ props }) => {
      const data = useReport();
      const selected = data.findings.filter((item) => !props.findingIds || props.findingIds.includes(item.id)).slice(0, 3);
      return <section className="section"><header><div><span className="section-kicker">WHAT THE AGENT FOUND</span><h2>Evidence, distilled</h2></div></header><div className="finding-grid">{selected.map((finding, index) => <article key={finding.id}><span>0{index + 1}</span><div className="evidence-level">{finding.evidence_level || "Observed"}</div><h3>{finding.title}</h3><p>{finding.summary}</p>{finding.limitation && <small>{finding.limitation}</small>}</article>)}</div></section>;
    },
    ChartGrid: ({ props }) => {
      const data = useReport();
      const selected = data.charts.filter((item) => !props.chartIds || props.chartIds.includes(item.id)).slice(0, 2);
      return <section className="chart-grid">{selected.map((chart) => <EvidenceChart key={chart.id} chart={chart} />)}</section>;
    },
    OutcomeCard: () => {
      const data = useReport();
      const repair = data.repairs[0];
      return <section className={`outcome-card ${repair?.fixed ? "success" : "neutral"}`}>
        <div className="outcome-icon">{repair?.fixed ? <Check /> : <Wrench />}</div>
        <div><span className="section-kicker">OUTCOME</span><h2>{data.summary.headline}</h2><p>{data.summary.answer}</p></div>
        <div className="outcome-proof"><small>REPORT CONFIDENCE</small><strong>{data.summary.confidence}</strong>{repair && <span>{repair.fixed_cases} fixed · {repair.broken_cases} regressions</span>}</div>
      </section>;
    },
    CasePreview: ({ props }) => {
      const data = useReport();
      const navigate = useContext(NavContext);
      const selected = data.cases.filter((item) => !props.caseIds || props.caseIds.includes(item.id)).slice(0, 4);
      if (!selected.length) return <></>;
      return <section className="section"><header><div><span className="section-kicker">REAL MODEL I/O</span><h2>Representative cases</h2></div><button className="text-button" onClick={() => navigate("cases")}>Open Case Studio <ArrowUpRight size={15} /></button></header><div className="case-preview">{selected.map((item) => <article key={item.id}><span className={`status status-${item.status}`}>{item.status}</span><h3>{item.id}</h3><p>{item.prompt}</p><PreviewMedia item={item} data={data} /></article>)}</div></section>;
    },
    // The sheet is the only component that renders nothing at all when its data
    // is absent: a run that never probed has no story to tell in this shape,
    // and an empty sheet reads as a run that found nothing.
    CaseStudySheet: () => {
      const data = useReport();
      return data.case_study ? <CaseStudySection sheet={data.case_study} /> : <></>;
    },
    EvidenceIndex: () => {
      const navigate = useContext(NavContext);
      return <section className="evidence-index"><Database /><div><h3>Need to audit the conclusion?</h3><p>Move from stage evidence to individual model I/O, then to raw agent events.</p></div>{["evidence", "cases", "debug"].map((view, i) => <button key={view} onClick={() => navigate(view)}><span>0{i + 2}</span>{view}</button>)}</section>;
    },
  },
});

function PreviewMedia({ item, data }: { item: ReportData["cases"][number]; data: ReportData }) {
  const media = item.media_ids.map((id) => data.media.find((entry) => entry.id === id)).filter(Boolean);
  const first = media[0];
  if (!first) return null;
  const source = first.data_uri || `/api/media/${encodeURIComponent(first.id)}`;
  if (first.kind === "image") return <ZoomableImage className="case-preview-media" src={source} alt={`Input for ${item.id}`} caption={`Input for ${item.id}`} />;
  if (first.kind === "audio") return <audio className="case-preview-audio" controls preload="metadata" src={source} />;
  return <video className="case-preview-media" controls preload="metadata" src={source} />;
}
