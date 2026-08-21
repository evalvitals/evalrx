import { createContext, useContext, useMemo } from "react";
import { defineCatalog } from "@json-render/core";
import { defineRegistry } from "@json-render/react";
import { schema } from "@json-render/react/schema";
import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import ReactECharts from "echarts-for-react";
import { ArrowUpRight, Check, CircleDot, Database, Wrench } from "lucide-react";
import { z } from "zod";
import type { Chart, ReportData, Stage } from "./types";

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
    <div className="stage-code">{stage.code}</div>
    <strong>{stage.title}</strong>
    <small>{stage.status.replaceAll("-", " ")}</small>
    <Handle type="source" position={Position.Right} />
  </div>;
}

const nodeTypes = { stage: StageNode };

function EvidenceChart({ chart }: { chart: Chart }) {
  const option = chart.kind === "donut" ? {
    tooltip: { trigger: "item" },
    color: ["#f06d5f", "#6bd8ad", "#89a6ff", "#f4ca72"],
    series: [{ type: "pie", radius: ["54%", "76%"], center: ["50%", "52%"], label: { color: "#b8c9c4", formatter: "{b}  {c}" }, data: chart.series.map((item) => ({ name: item.label, value: item.value })) }],
  } : {
    grid: { left: 110, right: 24, top: 12, bottom: 22 },
    xAxis: { type: "value", splitLine: { lineStyle: { color: "#23332f" } }, axisLabel: { color: "#8da19b" } },
    yAxis: { type: "category", data: chart.series.map((item) => item.label), axisLabel: { color: "#b8c9c4", width: 96, overflow: "truncate" }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{ type: "bar", data: chart.series.map((item) => ({ value: item.value, itemStyle: { color: item.highlight ? "#6bd8ad" : "#586a65", borderRadius: 4 } })), barWidth: 13 }],
  };
  return <article className="chart-card"><h3>{chart.title}</h3>{chart.subtitle && <p>{chart.subtitle}</p>}<ReactECharts option={option} style={{ height: 250 }} />{chart.series.some((item) => item.raw_label) && <details className="chart-audit"><summary>Technical measurement names</summary>{chart.series.map((item) => item.raw_label && <code key={item.raw_label}>{item.raw_label}</code>)}</details>}</article>;
}

export const { registry } = defineRegistry(reportCatalog, {
  components: {
    ReportPage: ({ children }) => <main className="report-page">{children}</main>,
    SettingHero: () => {
      const data = useReport();
      return <section className="setting-hero">
        <div className="eyebrow"><CircleDot size={14} /> Completed diagnostic run</div>
        <h1>From model failure<br /><span>to tested repair.</span></h1>
        <p className="lead">{data.setting.question}</p>
        <div className="setting-route">
          <div><small>MODEL</small><strong>{data.setting.model}</strong></div>
          <ArrowUpRight size={20} />
          <div><small>EVALUATED ON</small><strong>{data.setting.dataset}</strong></div>
        </div>
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
      const edges: Edge[] = data.stages.slice(1).map((stage, index) => ({ id: `${data.stages[index].id}-${stage.id}`, source: data.stages[index].id, target: stage.id, animated: !["not-run", "skipped"].includes(stage.status), style: { stroke: "#6bd8ad", strokeWidth: 1.5 } }));
      return <section className="section journey-section"><header><div><span className="section-kicker">THE AGENT'S PATH</span><h2>Find the failure. Test the cause. Repair the model.</h2></div><p>Click any stage to inspect its evidence and the agent events behind it.</p></header><div className="journey-canvas"><ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView minZoom={0.6} maxZoom={1.2} nodesDraggable={false} nodesConnectable={false} panOnScroll={false} onNodeClick={(_, node) => navigate(`evidence:${node.id}`)}><Background color="#24332f" gap={22} size={1} /><Controls showInteractive={false} /></ReactFlow></div></section>;
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
  if (first.kind === "image") return <img className="case-preview-media" src={source} alt={`Input for ${item.id}`} />;
  if (first.kind === "audio") return <audio className="case-preview-audio" controls preload="metadata" src={source} />;
  return <video className="case-preview-media" controls preload="metadata" src={source} />;
}
