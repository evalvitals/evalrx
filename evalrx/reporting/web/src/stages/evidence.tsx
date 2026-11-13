/** M2's evidence — its figures and its statistical screens — which M3 shows again as its starting point. */
import ReactECharts from "echarts-for-react";
import type { ECharts, ElementEvent, TooltipComponentFormatterCallbackParams } from "echarts";
import { tc } from "../theme";
import { ZoomableImage } from "../lightbox";
import { useFigureSource } from "../api";
import { chartPercent, chartValue } from "../reportAccess";
import { BarChart3 } from "lucide-react";
import type { FigureView, StatView } from "../types";
import { RecordTable } from "../detail";
import { normalizeKey } from "../format";

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
export function chartCited(figureId: string, citedKeys: string[]): boolean {
  const fig = normalizeKey(figureId).replace(/^\d+/, "");
  return citedKeys.some(
    (cited) =>
      fig === cited ||
      (Math.min(fig.length, cited.length) >= 16 &&
        (fig.startsWith(cited) || cited.startsWith(fig))),
  );
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
/** A figure as shown: `raw_title` keeps the explorer's own title when it was relabelled. */
type LabelledFigure = FigureView & { raw_title?: string };

export function labelledFigures(figures: FigureView[], stats: StatView[]): LabelledFigure[] {
  const pairs = (stats || [])
    .filter((row) => row?.raw_signal && row?.label && String(row.raw_signal).includes("."))
    .map((row) => [String(row.raw_signal).replace(/\./g, "_"), String(row.label)] as const)
    .sort((a, b) => b[0].length - a[0].length);
  const display = (title: string) => {
    let out = title;
    for (const [flat, label] of pairs) out = out.split(flat).join(label);
    return out.replace(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b/g, (m) => m.replace(/_/g, " "));
  };
  return (figures || []).map((figure) => {
    const title = display(String(figure.title || ""));
    return title === figure.title ? figure : { ...figure, title, raw_title: figure.title };
  });
}

export function EvidenceFigure({ figure }: { figure: LabelledFigure }) {
  const source = useFigureSource(figure);
  return (
    <figure>
      <ZoomableImage src={source} alt={figure.title} caption={figure.title} />
      <figcaption>
        <b title={figure.raw_title || undefined}>{figure.title}</b>
        {figure.question && <span>{figure.question}</span>}
        {figure.reading && (
          <p>
            <BarChart3 size={13} /> {figure.reading}
          </p>
        )}
        {figure.do_not_infer && <small>Do not infer: {figure.do_not_infer}</small>}
      </figcaption>
    </figure>
  );
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
    type: "category",
    data: labels,
    axisLabel: { color: tc("#b8c9c4"), fontSize: 11 },
    axisTick: { show: false },
  };
}

function escapeHtml(text: string): string {
  return text.replace(
    /[&<>"]/g,
    (c) =>
      (({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }) as Record<string, string>)[c],
  );
}

/**
 * Tooltip whose heading is the full name the axis had to shorten.
 *
 * `names` is in plot order (bottom-first), matching the series data, so the
 * hovered dataIndex reads straight out of it.
 */
function namedTooltip(names: string[], format: (value: unknown) => string) {
  return (params: TooltipComponentFormatterCallbackParams) => {
    const series = Array.isArray(params) ? params : [params];
    const first: Partial<(typeof series)[number]> & { axisValue?: unknown } = series[0] || {};
    const named = first.dataIndex === undefined ? undefined : names[first.dataIndex];
    const head = escapeHtml(String(named ?? first.name ?? ""));
    const lines = series
      .map(
        (row) =>
          `<div style="display:flex;gap:18px;justify-content:space-between;align-items:baseline">` +
          `<span>${row.marker} ${escapeHtml(String(row.seriesName || ""))}</span>` +
          `<b>${escapeHtml(format(row.value))}</b></div>`,
      )
      .join("");
    return (
      `<div style="max-width:300px;white-space:normal;line-height:1.45;margin-bottom:7px">` +
      `<b>${first.axisValue ? escapeHtml(String(first.axisValue)) + " · " : ""}${head}</b></div>${lines}`
    );
  };
}

/** Hovering a label opens its row's tooltip — the axis alone no longer names it. */
function labelHoverTooltip(rowCount: number) {
  return (chart: ECharts) => {
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
        } catch {
          /* try the other spelling */
        }
      }
      return NaN;
    };
    zr.on("mousemove", (event: ElementEvent) => {
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

export function StatEvidenceChart({
  stats,
  title,
  note,
}: {
  stats: StatView[];
  title: string;
  note: string;
}) {
  const rows = stats.filter((item) => typeof item.effect === "number").slice(0, 8);
  if (!rows.length) return null;
  const plotted = rows.slice().reverse();
  const effectNames = plotted.map((item) => String(item.label));
  const option = {
    grid: { left: AXIS_GUTTER, right: 30, top: 30, bottom: 30 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: namedTooltip(effectNames, (v) => chartValue(v, { signed: true })),
    },
    xAxis: {
      type: "value",
      name: "difference in error rate",
      nameTextStyle: { color: tc("#8fa49d"), fontSize: 10 },
      axisLabel: { color: tc("#8fa49d") },
      splitLine: { lineStyle: { color: tc("#22332e") } },
    },
    yAxis: numberedAxis(rows.length, "Behavior"),
    series: [
      {
        name: "Observed difference",
        type: "bar",
        data: plotted.map((item) => ({
          value: item.effect,
          itemStyle: { color: item.reject ? tc("#6bd8ad") : tc("#71857f"), borderRadius: 4 },
        })),
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: tc("#f4ca72"), type: "dashed" },
          data: [{ xAxis: 0 }],
        },
      },
    ],
  };
  const rateRows = rows.filter(
    (item) =>
      typeof item.fail_rate_signal === "number" && typeof item.fail_rate_control === "number",
  );
  const ratePlotted = rateRows.slice().reverse();
  const rateNames = ratePlotted.map((item) => String(item.label));
  const rateOption = {
    grid: { left: AXIS_GUTTER, right: 30, top: 24, bottom: 28 },
    tooltip: { trigger: "axis", formatter: namedTooltip(rateNames, chartPercent) },
    legend: { top: 0, textStyle: { color: tc("#9fb2ac"), fontSize: 10 } },
    xAxis: {
      type: "value",
      max: 1,
      axisLabel: { color: tc("#8fa49d"), formatter: (v: number) => `${Math.round(v * 100)}%` },
      splitLine: { lineStyle: { color: tc("#22332e") } },
    },
    yAxis: numberedAxis(rateRows.length, "Pattern"),
    series: [
      {
        name: "Cases with this behavior",
        type: "bar",
        data: ratePlotted.map((item) => item.fail_rate_signal),
        itemStyle: { color: tc("#89a6ff"), borderRadius: 3 },
      },
      {
        name: "Other cases",
        type: "bar",
        data: ratePlotted.map((item) => item.fail_rate_control),
        itemStyle: { color: tc("#657b74"), borderRadius: 3 },
      },
    ],
  };
  return (
    <section className="stat-evidence">
      <header>
        <span>VISUAL SUMMARY OF M2</span>
        <h3>{title}</h3>
        <p>{note}</p>
      </header>
      <ReactECharts
        option={option}
        notMerge
        onChartReady={labelHoverTooltip(effectNames.length)}
        style={{ height: Math.max(300, rows.length * 46) }}
      />
      {rateRows.length > 0 && (
        <>
          <h4 className="stat-subtitle">What those patterns mean in the cases</h4>
          <p className="stat-caption">
            For each measured behavior, compare the error rate among cases with that behavior
            against all other cases. This is an observed comparison, not a causal claim.
          </p>
          <ReactECharts
            option={rateOption}
            notMerge
            onChartReady={labelHoverTooltip(rateNames.length)}
            style={{ height: Math.max(280, rateRows.length * 54) }}
          />
        </>
      )}
      <details>
        <summary>Technical measurement names and test records</summary>
        <RecordTable
          rows={rows.map((item, index) => ({
            behavior: `Behavior ${index + 1}`,
            name: item.label,
            measurement: item.raw_signal,
            effect: item.effect,
            interval: item.ci,
            error_rate_with_behavior: item.fail_rate_signal,
            error_rate_other_cases: item.fail_rate_control,
            passed_screen: item.reject,
            tool: item.tool,
          }))}
        />
      </details>
    </section>
  );
}
