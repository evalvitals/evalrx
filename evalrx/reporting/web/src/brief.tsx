/**
 * L2 — one screen per stage: what this stop did, what came out, what that rests on.
 *
 * The report has three depths. L1 (the overview) shows the M1→M5 path and says
 * nothing about any one stage. L3 (`StageArtifact` in views.tsx) shows
 * everything a stage retained — every probe, every figure, every raw record —
 * and is the audit surface. Between them there was nothing, so clicking a stage
 * on L1 dropped the reader straight into `Gold In Answer Region 0`.
 *
 * L2 is that missing layer, and it is a SUMMARY, not a re-skin of L3: one
 * verdict sentence, a handful of numbers that each say what they amount to, one
 * figure, and three to five points that justify the verdict. If it needs a
 * scrollbar to make its point it has stopped being L2.
 *
 * Everything here is DERIVED from what a run already emitted — no field was
 * added to the contract for it. That is deliberate: a "one-line conclusion"
 * field would be filled by five different producers with five different ideas
 * of what belongs in it, and would be empty on every run recorded before it
 * existed. Derivation has the opposite failure mode: it degrades to a shorter
 * sentence instead of a blank one. Each builder below reads `stage_detail`
 * (present on every run) first and treats `contract` (absent on older runs, and
 * on runs whose emission failed) as enrichment it can do without.
 */
import ReactECharts from "echarts-for-react";
import { tc } from "./theme";
import {
  AlertTriangle, ArrowRight, BarChart3, CheckCircle2, HelpCircle, XCircle,
} from "lucide-react";
import type {
  AnalyzerSelection, Chart, DiagnosisOutput, FixOutput, HypothesisTestOutput,
  Modality, ProbeOutput, ReportData,
} from "./types";
import { ZoomableImage } from "./lightbox";
import {
  chartValue, countOf, findContract, headlineSplit, leadFrom, metric, oddsPhrase,
  outcomeColors, plural, pointsGap, stageDetail, stripMarkdown,
} from "./reportAccess";

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

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

export function StageBrief({ brief, onDeepen }: { brief: Brief; onDeepen: () => void }) {
  // Two analyzers can measure the same thing under different names, and the
  // derived sentences then come out word-for-word identical. Three identical
  // bullets read as a rendering bug and crowd out the points that differ, so
  // the list is deduplicated on the rendered text before it is capped.
  const points = [...new Set(brief.points.map((point) => point.trim()).filter(Boolean))].slice(0, 5);
  return <div className="stage-brief">
    <section className="brief-verdict">
      <span className="section-kicker">WHAT CAME OUT OF THIS STEP</span>
      <h2>{brief.verdict}</h2>
      {brief.lead && <p className="lead-small">{brief.lead}</p>}
    </section>

    {brief.kpis.length > 0 && <div className="brief-kpis">
      {brief.kpis.slice(0, 5).map((item) => <article key={item.label}>
        <span>{item.label}</span>
        <strong>{item.value ?? "—"}</strong>
        {item.note && <small>{item.note}</small>}
      </article>)}
    </div>}

    <div className={`brief-body${brief.figure ? "" : " brief-body-textonly"}`}>
      {brief.figure && <figure className="brief-figure">
        {brief.figure}
        {brief.figureNote && <figcaption><BarChart3 size={13} /> {brief.figureNote}</figcaption>}
      </figure>}
      {(points.length > 0 || (brief.checks && brief.checks.length > 0)) && <div className="brief-points">
        {brief.checks && brief.checks.length > 0 && <>
          <h3>Checks run</h3>
          <ul className="brief-checks">{brief.checks.map((check) => <li key={check.name}>
            <b>{check.name}</b>{typeof check.n === "number" && check.n > 0 && <small>{check.n} cases</small>}
            {check.question && <span>{check.question}</span>}
          </li>)}</ul>
        </>}
        {points.length > 0 && <>
          <h3>What that rests on</h3>
          <ul>{points.map((point, i) => <li key={i}>{point}</li>)}</ul>
        </>}
      </div>}
    </div>

    {brief.caveat && <div className="brief-caveat">
      <AlertTriangle size={17} />
      <p>{brief.caveat}</p>
    </div>}

    <button className="brief-deepen" onClick={onDeepen}>
      See everything this step recorded <ArrowRight size={16} />
    </button>
  </div>;
}

/** The stage's own chart from the compiled report, by id. Undefined is fine. */
function reportChart(report: ReportData, id: string): Chart | undefined {
  return (report.charts || []).find((chart) => chart.id === id);
}

/** A compact echarts rendering of one compiled `Chart`, sized for L2. */
function BriefChart({ chart }: { chart: Chart }) {
  const series = chart.series || [];
  const option = chart.kind === "donut" ? {
    tooltip: { trigger: "item", valueFormatter: chartValue },
    color: outcomeColors(series.map((item) => item.label)),
    // The L2 column is narrower than the L3 page, and outside labels with
    // leader lines get clipped to "Pas..." there. A legend under the ring
    // carries the same two facts and cannot be cut off by the container.
    legend: {
      bottom: 0, textStyle: { color: tc("#b8c9c4"), fontSize: 11 },
      formatter: (name: string) =>
        `${name}  ${series.find((item) => item.label === name)?.value ?? ""}`,
    },
    series: [{
      type: "pie", radius: ["54%", "76%"], center: ["50%", "44%"],
      label: { show: false },
      data: series.map((item) => ({ name: item.label, value: item.value })),
    }],
  } : {
    grid: { left: 152, right: 26, top: 10, bottom: 24 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: chartValue },
    xAxis: {
      type: "value", axisLabel: { color: tc("#8da19b") },
      splitLine: { lineStyle: { color: tc("#23332f") } },
    },
    yAxis: {
      type: "category", data: series.map((item) => item.label),
      // These labels are analyzer questions, not short keys. Truncating them to
      // "Can repeated attempt..." leaves a bar nobody can identify, so they wrap
      // onto two lines and the row height grows to match.
      axisLabel: { color: tc("#b8c9c4"), width: 138, overflow: "break", lineHeight: 13, fontSize: 11 },
      axisLine: { show: false }, axisTick: { show: false },
    },
    series: [{
      type: "bar", barWidth: 12,
      data: series.map((item) => ({
        value: item.value,
        itemStyle: { color: item.highlight ? tc("#6bd8ad") : tc("#586a65"), borderRadius: 4 },
      })),
    }],
  };
  const height = chart.kind === "donut" ? 250 : Math.max(200, series.length * 42 + 40);
  // notMerge: moving between stages swaps a bar chart for a donut in the same
  // slot, and echarts merges into the live instance by default — the previous
  // stage's category axis was still drawn behind the new chart.
  return <ReactECharts option={option} notMerge style={{ height }} />;
}

/** A saved PNG a stage produced. `data_uri` on an exported report, path on a served one. */
function BriefImage({ figure }: { figure: any }) {
  const source = figure.data_uri || `/api/artifact?path=${encodeURIComponent(figure.path)}`;
  return <ZoomableImage className="brief-image" src={source}
    alt={figure.title || "Stage figure"} caption={figure.title || ""} />;
}

// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

export function buildBrief(stage: string, report: ReportData): Brief | null {
  const detail = stageDetail(report, stage);
  const status = report.stages.find((item) => item.id === stage)?.status || "";
  if (["not-run", "skipped"].includes(status) && stage !== "m5") {
    return {
      verdict: "This step did not run.",
      lead: "Nothing was measured here, so there is nothing to summarise. The steps "
        + "before it still stand on their own.",
      kpis: [], points: [],
    };
  }
  if (stage === "m1") return briefM1(report, detail);
  if (stage === "m2") return briefM2(report, detail);
  if (stage === "m3") return briefM3(report, detail);
  if (stage === "m4") return briefM4(report, detail);
  if (stage === "m5") return briefM5(report, detail);
  return null;
}

/** `text` guaranteed to end as a sentence, so derived lines can be joined
 *  without producing "…consistency=0.2) No way of testing…" or a double dot. */
function sentence(text: string): string {
  const clean = String(text || "").trim().replace(/\.{2,}$/, ".");
  if (!clean) return "";
  return /[.!?…]$/.test(clean) ? clean : `${clean}.`;
}

const MODALITY_WORD: Record<string, string> = {
  text: "text", image: "images", audio: "audio", video: "video",
};
function modalityWords(slots?: Modality[]): string {
  const words = (slots || []).map((slot) => MODALITY_WORD[slot] || slot);
  return words.length ? words.join(" + ") : "text";
}

// ── M1 ─────────────────────────────────────────────────────────────────────
function briefM1(report: ReportData, detail: Record<string, any>): Brief {
  const probes: any[] = detail.probes || [];
  const measured = Number(detail.n_measured || 0);
  const evaluated = metric(report, "evaluated");
  const failed = metric(report, "failed");
  const m1 = findContract<ProbeOutput>(report, "m1");
  const selection: AnalyzerSelection | undefined = m1?.selection;
  const broken = Object.entries(m1?.failed_analyzers || {});

  const verdict = failed !== undefined && evaluated
    ? `${plural(probes.length, "behavioural check")} ran, and the model got `
      + `${failed} of the ${evaluated} cases wrong.`
    : `${plural(probes.length, "behavioural check")} ran on `
      + `${plural(measured, "case")}.`;

  // Naming the checks beats describing them here. Each probe carries a
  // `question`, but an analyzer the glossary does not know gets a generic
  // sentence ("Measures model behavior across this dimension"), and four of
  // those in a row is filler that pushes the real content off the screen. The
  // full description of every check is one click away, which is where it
  // belongs.
  const points: string[] = [];
  if (selection) {
    const routed = modalityWords(selection.routed_on);
    const probed = modalityWords(selection.probed_modalities);
    points.push(routed === probed
      ? `The checks read ${routed}, which is what these cases actually carry.`
      : `The checks read ${routed}, while the cases themselves carry ${probed} — `
        + "so the parts of the input outside that were never examined.");
  }
  // The checks get a list of their own (`checks`): one bullet that joined
  // five titles with " · and 3 more" mixed questions with Title-Cased ids and
  // read as noise.
  const generic = (text: string) => !text || text.startsWith("Measures model behavior");
  const checks = probes.map((probe) => ({
    name: String(probe.display_name || probe.title || probe.raw_name || ""),
    question: generic(String(probe.question || "")) ? undefined : String(probe.question),
    n: probe.n_cases === undefined || probe.n_cases === null ? null : Number(probe.n_cases),
  })).filter((check) => check.name);

  // A check with `n_cases: 0` and a check that ran on 8 of 12 cases are
  // different facts, and lumping them into one filter once produced "Every
  // check ran on all 12 cases" directly above a caveat saying one had not.
  // `coverage` is the checks that measured something; `zeroed` is the checks
  // that explicitly recorded measuring nothing.
  const coverage = probes.map((probe) => Number(probe.n_cases)).filter((n) => n > 0);
  const zeroed = probes.filter((probe) => Number(probe.n_cases) === 0
    && probe.n_cases !== undefined && probe.n_cases !== null);
  const lowest = coverage.length ? Math.min(...coverage) : 0;
  if (measured && coverage.length) {
    points.push(lowest === measured
      ? (coverage.length === probes.length
          ? `Every check ran on all ${plural(measured, "case")}.`
          : `${plural(coverage.length, "check")} reported a case count, and each ran on all ${measured} cases.`)
      : `Coverage ranged from ${lowest} to ${measured} cases per check — the shorter ones `
        + "were capped to keep the step inside its time budget.");
  }
  const capped = coverage.filter((n) => n < measured).length;
  const caveatParts: string[] = [];
  if (zeroed.length) {
    const names = zeroed.map((probe) => String(probe.title || probe.raw_name || ""))
      .filter(Boolean).join(", ");
    caveatParts.push(`${plural(zeroed.length, "check")}${names ? ` (${names})` : ""} measured `
      + `no cases at all, so ${zeroed.length === 1 ? "its question" : "their questions"} simply `
      + "went unanswered this run.");
  }
  if (broken.length) {
    caveatParts.push(
      `${plural(broken.length, "check")} could not run at all `
      + `(${broken.map(([name]) => name).join(", ")}), so whatever ${broken.length === 1 ? "it" : "they"} `
      + "would have measured is simply absent from everything downstream — not measured and found clean.");
  }
  if (capped && measured) {
    caveatParts.push(`${plural(capped, "check")} ran on fewer than all ${measured} cases, `
      + "which costs statistical power later but does not bias what was measured.");
  }

  return {
    verdict,
    checks,
    lead: "This step only measures. It records what the model does differently on "
      + "the cases it gets right and the ones it gets wrong — it does not yet claim "
      + "to know why anything failed.",
    kpis: [
      { label: "Checks run", value: probes.length || detail.n_probes || 0,
        note: broken.length ? `${broken.length} could not run` : "all completed" },
      { label: "Cases measured", value: measured || "—",
        note: evaluated ? `out of ${evaluated} evaluated` : undefined },
      ...(failed !== undefined && evaluated
        ? [{ label: "Got it wrong", value: countOf(failed, evaluated),
             note: "these are what the rest of the report is about" }]
        : []),
      ...(typeof detail.duration === "number"
        ? [{ label: "Time taken", value: `${detail.duration.toFixed(0)}s` }] : []),
    ],
    figure: reportChart(report, "outcomes")
      ? <BriefChart chart={reportChart(report, "outcomes")!} /> : undefined,
    figureNote: "How the evaluated cases came out. Everything later in the report is "
      + "an attempt to explain the wrong ones.",
    points,
    caveat: caveatParts.join(" ") || undefined,
  };
}

// ── M2 ─────────────────────────────────────────────────────────────────────
function briefM2(report: ReportData, detail: Record<string, any>): Brief {
  const stats: any[] = detail.stats || [];
  const takeaways: any[] = detail.takeaways || [];
  const figures: any[] = detail.figures || [];
  const conclusion = stripMarkdown(detail.conclusion || detail.narrative || "");
  const [headline, rest] = headlineSplit(conclusion);
  const survived = stats.filter((item) => item.reject);

  const verdict = headline
    ? headline
    : survived.length
      ? `${plural(survived.length, "measured behaviour")} line up with the model's errors `
        + "more often than chance would explain."
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
      const hasRates = typeof item.fail_rate_signal === "number"
        && typeof item.fail_rate_control === "number";
      if (hasRates) {
        points.push(`${label}: the model got ${Math.round(item.fail_rate_signal * 100)}% of cases `
          + `wrong when this shows up, against ${Math.round(item.fail_rate_control * 100)}% when it does not.`);
        continue;
      }
      const gap = pointsGap(item.effect);
      const odds = oddsPhrase(item.e_value);
      points.push(`${label}: a gap of ${gap || "some size"} between the two groups compared`
        + (odds ? `, with the evidence against that being luck running ${odds}.` : "."));
    }
  }
  if (!points.length && conclusion) {
    for (const step of (detail.observations || []).slice(0, 3)) {
      points.push(stripMarkdown(String(step)));
    }
  }
  if (survived.length && stats.length > survived.length) {
    points.push(`${survived.length} of the ${stats.length} patterns screened are still standing `
      + "after discounting for how many were tried at once; the rest could be luck.");
  }

  const topFigure = figures[0];
  const effects = reportChart(report, "effects");

  return {
    verdict,
    lead: leadFrom(rest) || undefined,
    kpis: [
      ...(takeaways.length ? [{ label: "Ranked findings", value: takeaways.length }] : []),
      { label: "Patterns screened", value: stats.length,
        note: "each one compares a measured behaviour against the errors" },
      { label: "Still standing", value: survived.length,
        note: "after discounting for how many were tried at once" },
      ...(figures.length ? [{ label: "Figures produced", value: figures.length }] : []),
    ],
    figure: topFigure ? <BriefImage figure={topFigure} />
      : effects ? <BriefChart chart={effects} /> : undefined,
    figureNote: topFigure
      ? stripMarkdown(topFigure.reading || topFigure.title || "")
      : "A longer bar means the behaviour separates right from wrong answers more "
        + "sharply. It is a pattern, not a cause.",
    points,
    caveat: "These patterns were found on the same half of the data used to look for "
      + "them, so any of them could be a coincidence that happens to fit. Nothing here "
      + "is a verdict — the held-out check is the only step allowed to issue one.",
  };
}

// ── M3 ─────────────────────────────────────────────────────────────────────
function briefM3(report: ReportData, detail: Record<string, any>): Brief {
  const accepted: any[] = detail.hypotheses || [];
  const recovered: any[] = detail.unparsed_proposals || [];
  const list = accepted.length ? accepted : recovered;
  const m3 = findContract<DiagnosisOutput>(report, "m3");
  const untestable = (m3?.hypotheses || []).filter((h) => !h.test_design?.trim());
  const withTest = list.filter((h) => String(h.test_design || "").trim()).length;

  const verdict = list.length
    ? `${plural(list.length, "possible explanation")} for the failures ${list.length === 1 ? "was" : "were"} `
      + `put forward, and ${list.length === 1 ? "it is not proven" : "none of them is proven"} yet.`
    : "No explanation could be put forward from the patterns found.";

  // A hypothesis statement can be a 400-character paragraph and its test
  // design another one; pasting both into a bullet rebuilds L3 with bullet
  // glyphs. The claim is cut only at punctuation the author wrote
  // (headlineSplit), and a test design that does not fit a summary line is
  // pointed at rather than quoted — the full wording is one click down.
  const points = list.slice(0, 4).map((h, i) => {
    // `plain_statement` is already the one-sentence rendering the producer was
    // asked for, and its numbers ("sixteen of the twenty-six correct answers
    // were lists of six words or fewer") are the part that makes it worth
    // reading — cutting it at its first dash throws away exactly that and
    // leaves a claim with no evidence. Only the technical `statement`, which
    // can run to a 400-character paragraph, gets shortened.
    const plain = stripMarkdown(h.plain_statement || "");
    const full = plain || stripMarkdown(h.statement || h.hypothesis || "");
    const [head] = plain ? [plain] : headlineSplit(full);
    const claim = sentence(head || full);
    const clipped = (head || full).length < full.length;
    const test = stripMarkdown(String(h.test_design || "").trim());
    const lead = `H${i + 1} — ${claim}`;
    if (!test) return `${lead} No way of testing it was named.`;
    if (!clipped && test.length <= 160) return `${lead} Checked against: ${sentence(test)}`;
    return `${lead} ${clipped ? "Shortened here — the full claim and how it will be checked are"
      : "How it will be checked is spelled out"} one level down.`;
  });

  const caveatParts: string[] = [];
  if (!accepted.length && recovered.length) {
    caveatParts.push("The agent did propose explanations, but the pipeline could not read "
      + "the format it wrote them in, so they never reached the held-out check. They are "
      + "shown for audit only.");
  }
  if (untestable.length) {
    caveatParts.push(`${plural(untestable.length, "proposal")} name${untestable.length === 1 ? "s" : ""} `
      + "no way of being wrong. Those come back \"could not be decided\" however much "
      + "evidence the next step gathers — read that as \"never testable\", not as "
      + "\"not enough data yet\".");
  }

  return {
    verdict,
    lead: "This step turns the patterns from the previous one into explanations that "
      + "could be shown to be wrong. Everything on this screen is a proposal; whether "
      + "any of it survives is decided one step later.",
    kpis: [
      { label: "Explanations proposed", value: list.length },
      { label: "With a way to test them", value: withTest,
        note: withTest < list.length ? `${list.length - withTest} without` : "all of them" },
      ...(recovered.length && !accepted.length
        ? [{ label: "Read for audit only", value: recovered.length,
             note: "the pipeline could not parse these" }] : []),
    ],
    // Only a figure this step produced. Borrowing the previous step's chart
    // would put the same picture on two screens in a row, which is exactly the
    // "L2 is L3 rearranged" failure this layer exists to avoid — and it would
    // imply M3 measured something, which it never does.
    figure: (detail.evidence_figures || [])[0]
      ? <BriefImage figure={detail.evidence_figures[0]} /> : undefined,
    figureNote: "The pattern these explanations are trying to account for. It comes "
      + "from the previous step, and motivates the ideas — it does not confirm them.",
    points,
    caveat: caveatParts.join(" ") || undefined,
  };
}

// ── M4 ─────────────────────────────────────────────────────────────────────
const M4_WORD: Record<string, string> = {
  supported: "the held-out cases agreed with it",
  refuted: "the held-out cases pointed the other way",
  inconclusive: "the held-out cases could not settle it",
};

function briefM4(report: ReportData, detail: Record<string, any>): Brief {
  const results: any[] = detail.results || [];
  if (!detail.ran || !results.length) {
    return {
      verdict: "No explanation reached the independent check.",
      lead: "This step tests a frozen explanation on cases that were not used to invent "
        + "it. Nothing was available to test, so this run has no verdict of any kind.",
      kpis: [], points: [],
    };
  }
  const count = (name: string) =>
    results.filter((item) => String(item.status || "").toLowerCase() === name).length;
  const supported = count("supported");
  const refuted = count("refuted");
  const undecided = count("inconclusive");
  const consistent = results.filter((item) => item.protocol_consistent !== false).length;

  // n=1 needs its own sentence: "Of the 1 explanation checked … while forming
  // them, 1 could not be decided" is what the plural template degrades to.
  const verdict = results.length === 1
    ? "One explanation was checked on cases the agent never saw while forming it, and "
      + `it ${supported ? "held up" : refuted ? "was contradicted" : "could not be decided"}.`
    : `Of the ${plural(results.length, "explanation")} checked on cases the `
      + "agent never saw while forming them, "
      + [
        supported ? `${supported} held up` : "",
        refuted ? `${refuted} ${refuted === 1 ? "was" : "were"} contradicted` : "",
        undecided ? `${undecided} could not be decided` : "",
      ].filter(Boolean).join(", ") + ".";

  const m3 = findContract<DiagnosisOutput>(report, "m3");
  const untestable = new Set((m3?.hypotheses || [])
    .filter((h) => !h.test_design?.trim()).map((h) => h.id));
  const m4 = findContract<HypothesisTestOutput>(report, "m4");
  const stuck = (m4?.results || []).filter(
    (r) => r.status === "inconclusive" && untestable.has(r.hypothesis_id));

  const points = results.slice(0, 4).map((result, i) => {
    const full = stripMarkdown(
      result.plain_statement || result.hypothesis || result.statement || `Explanation ${i + 1}`);
    // The claim is a whole M3 statement on some runs; keep the bullet a
    // summary by cutting only at the author's own punctuation.
    const claim = (headlineSplit(full)[0] || full).replace(/[.]$/, "");
    const status = String(result.status || "inconclusive").toLowerCase();
    return `${claim} — ${M4_WORD[status] || status}.`;
  });
  if (consistent < results.length) {
    const off = results.length - consistent;
    points.push(`${plural(off, "check")} answered a question this run was not actually `
      + `asking, and ${off === 1 ? "was" : "were"} kept out of the conclusions.`);
  }

  const withEffect = results.filter((r) => typeof r.effect_size === "number");
  const option = {
    grid: { left: 130, right: 26, top: 10, bottom: 26 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: unknown) => chartValue(v, { signed: true }) },
    xAxis: {
      type: "value", axisLabel: { color: tc("#8da19b") },
      splitLine: { lineStyle: { color: tc("#23332f") } },
    },
    yAxis: {
      type: "category",
      data: withEffect.map((_, i) => `H${i + 1}`),
      axisLabel: { color: tc("#b8c9c4") }, axisLine: { show: false }, axisTick: { show: false },
    },
    series: [{
      type: "bar", barWidth: 14,
      data: withEffect.map((r) => ({
        value: r.effect_size,
        itemStyle: {
          color: String(r.status).toLowerCase() === "supported" ? tc("#6bd8ad")
            : String(r.status).toLowerCase() === "refuted" ? tc("#f06d5f") : tc("#586a65"),
          borderRadius: 4,
        },
      })),
    }],
  };

  return {
    verdict,
    lead: "First an explanation is frozen. Then it is tested against cases the agent did "
      + "not use to invent it. This is the only step in the report allowed to say whether "
      + "an idea held up.",
    kpis: [
      { label: "Explanations checked", value: results.length },
      { label: "Held up", value: supported, note: supported ? "survived the held-out test" : "none" },
      { label: "Contradicted", value: refuted },
      { label: "Could not be decided", value: undecided },
    ],
    // A single grey bar for a single undecided hypothesis is a chart that
    // conveys nothing and reads as a rendering failure. The picture earns its
    // place once there is either a comparison to make or a decided verdict to
    // show; otherwise the words carry it.
    figure: withEffect.length && (withEffect.length > 1 || supported || refuted)
      ? <ReactECharts option={option} notMerge style={{ height: Math.max(160, withEffect.length * 42 + 50) }} />
      : undefined,
    figureNote: "How large a gap each explanation predicted, and whether it survived. "
      + "Green held up, red pointed the other way, grey could not be decided.",
    points,
    caveat: stuck.length
      ? (stuck.length === 1
        ? "One \"could not be decided\" verdict is not a shortage of evidence: that "
          + "proposal named no test, so nothing measured here could have settled it "
          + "either way. Re-running with more cases returns the same answer."
        : `${stuck.length} "could not be decided" verdicts are not a shortage of evidence: `
          + "those proposals named no test, so nothing measured here could have settled "
          + "them either way. Re-running with more cases returns the same answer.")
      : undefined,
  };
}

// ── M5 ─────────────────────────────────────────────────────────────────────
function briefM5(report: ReportData, detail: Record<string, any>): Brief {
  const candidates: any[] = detail.candidates || [];
  const m5 = findContract<FixOutput>(report, "m5_fix");
  const sweep = m5?.selection || [];

  if (!detail.ran || !candidates.length) {
    const tried = sweep.length;
    return {
      verdict: detail.skipped
        ? "Repair was deliberately held back."
        : tried
          ? `${plural(tried, "repair")} ${tried === 1 ? "was" : "were"} tried while choosing one, `
            + "and none was worth confirming."
          : "No repair was attempted.",
      lead: detail.skip_detail || detail.skip_reason
        || "A repair is only attempted once an explanation has survived the held-out "
           + "check, so that there is something specific to repair.",
      kpis: tried ? [{ label: "Repairs tried", value: tried, note: "none reached confirmation" }] : [],
      points: sweep.slice(0, 4).map((row) =>
        `${row.ref || row.name}${row.headline ? ` — ${row.headline}` : ""}`),
    };
  }

  const winner = candidates.find((item) => item.fixed)
    || candidates.reduce((best: any, item: any) =>
      Number(item.effect ?? -Infinity) > Number(best?.effect ?? -Infinity) ? item : best, null);
  const fixed = Number(winner?.n_fixed || 0);
  const broke = Number(winner?.n_broken || 0);
  const pairs = Number(winner?.n_pairs || 0);
  const odds = oddsPhrase(winner?.e_value);
  // The plain description of a repair can live in either place. A "floor"
  // candidate carries it on the stage row; one the judge authored carries it
  // only in the contract, and reading just the stage row lost the sentence for
  // exactly the runs where it was written by hand. Neither source is
  // guaranteed — a run with neither gets no sentence rather than its slug.
  const headline = String(
    winner?.headline
    || [...(m5?.attempted || []), ...sweep].find((row) => row.name === winner?.name)?.headline
    || "",
  ).trim();

  const held = candidates.filter((item: any) => item.fixed).length;
  const heldWord = held > 1 ? `${held} repairs held up, the strongest of them` : "One repair held up";
  const verdict = detail.fixed
    ? (headline
        ? `${heldWord}: ${headline.charAt(0).toLowerCase()}${headline.slice(1)}`
        : `${heldWord} on cases it had not been tuned on.`)
    : `${plural(candidates.length, "repair")} ${candidates.length === 1 ? "was" : "were"} `
      + "tested against the unchanged model, and none earned its place.";

  const points: string[] = [];
  if (pairs || fixed || broke) {
    points.push(`Across ${plural(pairs || 0, "case")} it was tested on, the best repair fixed `
      + `${fixed} and broke ${broke}.`);
  }
  if (odds) {
    points.push(winner?.reject
      ? `The evidence against that being luck runs ${odds}, which is strong enough to count.`
      : `The evidence against that being luck runs only ${odds}, which is not strong enough to count.`);
  }
  if (typeof winner?.coverage === "number") {
    const pct = Math.round(winner.coverage * 100);
    // At 100% "the rest were out of its reach" describes an empty set.
    points.push(pct >= 100
      ? "It applied to every failure it was aimed at — none was out of its reach."
      : `It applied to ${pct}% of the failures it was aimed at; the rest were out of `
        + "its reach entirely.");
  }
  if (candidates.length > 1) {
    // "Confirmed" would overclaim: every candidate reached the confirmation
    // stage, and most of them lost there.
    points.push(`${candidates.length - 1} other ${candidates.length - 1 === 1 ? "repair was" : "repairs were"} `
      + "tested on the same cases and are listed one level down.");
  }

  const option = {
    grid: { left: 120, right: 24, top: 26, bottom: 28 },
    color: [tc("#6bd8ad"), tc("#f06d5f")],
    tooltip: { trigger: "axis", valueFormatter: chartValue },
    legend: { textStyle: { color: tc("#9fb2ac"), fontSize: 11 }, top: 0 },
    xAxis: {
      type: "value", axisLabel: { color: tc("#8da19b") },
      splitLine: { lineStyle: { color: tc("#23332f") } },
    },
    yAxis: {
      type: "category",
      data: candidates.map((item: any, i: number) => item.ref || `R${i + 1}`),
      axisLabel: { color: tc("#b8c9c4") }, axisLine: { show: false }, axisTick: { show: false },
    },
    series: [
      { name: "errors fixed", type: "bar", barWidth: 11,
        data: candidates.map((item: any) => Number(item.n_fixed || 0)) },
      { name: "new errors", type: "bar", barWidth: 11,
        data: candidates.map((item: any) => Number(item.n_broken || 0)) },
    ],
  };

  const independent = Number(winner?.n_model_independent || 0);
  return {
    verdict,
    lead: "Every repair is run case by case against the unchanged model on cases held "
      + "back from the search, so a repair that only looks good on the cases it was "
      + "designed against does not get through.",
    kpis: [
      // "Confirmed" would be wrong for this count: every candidate here reached
      // the confirmation stage, and most of them lost there.
      { label: "Repairs tested", value: candidates.length,
        note: "on cases held back from the search" },
      { label: "Held up", value: held, note: held ? "cleared the gate" : "none cleared the gate" },
      { label: "Errors fixed", value: fixed, note: pairs ? `out of ${pairs} cases tested` : undefined },
      { label: "New errors caused", value: broke,
        note: broke ? "cases that were right before" : "none" },
      ...(odds ? [{ label: "Odds against luck", value: odds,
                    note: winner?.reject ? "strong enough to count" : "not strong enough" }] : []),
    ],
    figure: <ReactECharts option={option} notMerge style={{ height: Math.max(180, candidates.length * 44 + 60) }} />,
    figureNote: "For each repair, how many cases it turned from wrong to right against "
      + "how many it turned from right to wrong. Both bars matter — a repair that fixes "
      + "ten and breaks nine has done almost nothing.",
    points,
    caveat: independent
      ? `${plural(independent, "case")} ${independent === 1 ? "was" : "were"} solved by the added `
        + "code rather than by the model, and left out of the count — a scaffold that "
        + "answers the question itself has repaired the task, not the model."
      : undefined,
  };
}

/** Status icon for the L2 header, so the verdict reads before the words do. */
export function BriefStatusIcon({ stage, report }: { stage: string; report: ReportData }) {
  const detail = stageDetail(report, stage);
  if (stage === "m5") {
    return detail.fixed ? <CheckCircle2 /> : detail.ran ? <XCircle /> : <HelpCircle />;
  }
  if (stage === "m4") {
    const results: any[] = detail.results || [];
    const supported = results.filter((r) => String(r.status).toLowerCase() === "supported").length;
    if (!results.length) return <HelpCircle />;
    return supported ? <CheckCircle2 /> : <XCircle />;
  }
  return null;
}
