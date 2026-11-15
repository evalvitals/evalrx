/**
 * M4 — verify: the frozen hypotheses tested on held-out cases.
 *
 * `M4Detail` is the stage's full record (L3); `briefM4` derives its one-screen
 * summary (L2) from the same data.
 */
import ReactECharts from "echarts-for-react";
import { tc } from "../theme";
import { chartValue, findContract, headlineSplit, plural, stripMarkdown } from "../reportAccess";
import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import type {
  DiagnosisOutput,
  HypothesisTestOutput,
  M4StageDetail,
  ReportData,
  ValidationEventView,
  ValidationEvidence,
  ValidationExample,
  ValidationResultView,
} from "../types";
import { type Brief } from "../brief";
import { EmptyStage } from "../detail";
import {
  plainStatus,
  plainEvidence,
  plainTest,
  plainVerdict,
  number,
  percent,
  formatInterval,
  displayValue,
} from "../format";
import { StageBanner, StageKpis, ExampleSection, ExampleMedia } from "./layout";

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
  return (
    <div className="parser-warning">
      <AlertTriangle />
      <div>
        <b>
          {stuck.length === 1
            ? "One “inconclusive” verdict is not a shortage of evidence."
            : `${stuck.length} “inconclusive” verdicts are not a shortage of evidence.`}
        </b>
        <p>
          {stuck.map((r) => r.hypothesis_id).join(", ")} came back inconclusive because the proposal
          named no test, so nothing measured here could have decided it either way. Re-running with
          more cases returns the same verdict. The fix belongs in M3.
        </p>
      </div>
    </div>
  );
}

export function M4Detail({ data, report }: { data: Partial<M4StageDetail>; report: ReportData }) {
  const results = data.results || [];
  if (!data.ran || !results.length)
    return (
      <>
        <StageBanner kind="CONFIRMATORY" title="Held-out validation">
          M4 tests frozen hypotheses on evidence not used to propose them.
        </StageBanner>
        <EmptyStage
          title="Validation was not reached"
          body="No accepted M3 hypothesis was available for independent adjudication in this run."
        />
      </>
    );
  const statuses = (name: string) =>
    results.filter((item) => String(item.status || "").toLowerCase() === name).length;
  const consistent = results.filter((item) => item.protocol_consistent !== false).length;
  const examples = data.examples || [];
  return (
    <>
      <StageBanner kind="CONFIRMATORY" title="Independent check">
        First we freeze a possible explanation. Then we test it on fresh evidence the agent did not
        use to invent the explanation. This is the stage allowed to say whether the idea held up.
      </StageBanner>
      <StageKpis
        items={[
          { label: "Hypotheses checked", value: results.length },
          { label: "Supported", value: statuses("supported") },
          { label: "Refuted", value: statuses("refuted") },
          { label: "Inconclusive", value: statuses("inconclusive") },
          { label: "Answered the question asked", value: `${consistent}/${results.length}` },
        ]}
      />
      <UndecidableNote report={report} />
      {examples.length > 0 && (
        <ExampleSection
          eyebrow="A validation example"
          title="How we decide whether an explanation survives"
          note="This is one recorded validation check. Its conclusion uses the complete independent test set—not a hand-picked example."
        >
          <div className="example-deck">
            {examples.map((example) => (
              <M4Example example={example} report={report} key={example.id} />
            ))}
          </div>
        </ExampleSection>
      )}
      <div className="verdict-list">
        {results.map((result, index) => (
          <VerdictCard result={result} fallback={data.event} index={index} key={index} />
        ))}
      </div>
    </>
  );
}

function M4Example({ example, report }: { example: ValidationExample; report?: ReportData }) {
  const status = String(example.status || "inconclusive").replaceAll("_", " ");
  const isCase = example.kind === "validation_case";
  return (
    <article className="example-card m4-example">
      <header>
        <span className="example-step">1 · FROZEN IDEA</span>
        <em className={`status status-${status}`}>{plainStatus(status)}</em>
      </header>
      <h4>{example.hypothesis || "A proposed explanation"}</h4>
      {isCase && (
        <>
          <span className="example-step">2 · ONE FRESH CASE IN THE TEST POOL</span>
          <ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} />
          <div className="example-prompt">
            {example.input || "The original task text was not retained."}
          </div>
          <div className="example-flow">
            <div>
              <small>MODEL ANSWER</small>
              <b>{displayValue(example.baseline_output)}</b>
            </div>
            <div>
              <small>EXPECTED ANSWER</small>
              <b>{displayValue(example.expected)}</b>
            </div>
          </div>
        </>
      )}
      <div className="validation-flow">
        <div>
          <span>2 · INDEPENDENT TEST</span>
          <b>{plainTest(example.test)}</b>
        </div>
        <div>
          <span>3 · RESULT</span>
          <b>{plainStatus(status)}</b>
          {example.effect !== undefined && example.effect !== null && (
            <small>Measured difference: {number(example.effect)}</small>
          )}
          {example.interval && <small>Likely range: {formatInterval(example.interval)}</small>}
        </div>
      </div>
      {example.verdict && <p className="example-verdict">{example.verdict}</p>}
      <footer>{example.plain_reading}</footer>
    </article>
  );
}

function VerdictCard({
  result,
  fallback,
  index,
}: {
  result: ValidationResultView;
  fallback?: ValidationEventView;
  index: number;
}) {
  const evidence: ValidationEvidence = result.evidence || fallback?.evidence || {};
  const status = String(result.status || fallback?.status || "inconclusive").toLowerCase();
  const hypothesis =
    result.hypothesis || result.statement || fallback?.hypothesis || `Hypothesis ${index + 1}`;
  const icon =
    status === "supported" ? (
      <CheckCircle2 />
    ) : status === "refuted" ? (
      <XCircle />
    ) : (
      <AlertTriangle />
    );
  const ci = evidence.ci || result.ci;
  return (
    <article className={`verdict-card verdict-${status}`}>
      <header>
        {icon}
        <div>
          <span>H{index + 1} · INDEPENDENT CHECK</span>
          <strong>{plainStatus(status)}</strong>
        </div>
      </header>
      <h3>{hypothesis}</h3>
      <div className="verdict-metrics">
        <span>
          <small>MEASURED DIFFERENCE</small>
          <b>{number(result.effect_size ?? evidence.effect_size)}</b>
        </span>
        <span>
          <small>CONFIDENCE</small>
          <b>{percent(result.confidence ?? fallback?.confidence_score)}</b>
        </span>
        <span>
          <small>LIKELY RANGE</small>
          <b>{formatInterval(ci)}</b>
        </span>
        <span>
          <small>TYPE OF EVIDENCE</small>
          <b>{plainEvidence(result.evidence_grade || evidence.evidence_grade)}</b>
        </span>
      </div>
      <p className="verdict-reason">
        {plainVerdict(
          result.verdict || evidence.m4_verdict || "No validation explanation was retained.",
        )}
      </p>
      <div className="verdict-foot">
        <span className={result.protocol_consistent === false ? "bad" : "good"}>
          {result.protocol_consistent === false
            ? "Does not match the requested evaluation"
            : "Matches the requested evaluation"}
        </span>
        <span>
          {evidence.fdr?.method ? "Multiple-comparison check applied" : "Independent evidence"}
        </span>
      </div>
      <details>
        <summary>Technical audit evidence</summary>
        <pre>{JSON.stringify(evidence, null, 2)}</pre>
      </details>
    </article>
  );
}

// ── M4 brief (L2) ──────────────────────────────────────────────────────────
const M4_WORD: Record<string, string> = {
  supported: "the held-out cases agreed with it",
  refuted: "the held-out cases pointed the other way",
  inconclusive: "the held-out cases could not settle it",
};

export function briefM4(report: ReportData, detail: Partial<M4StageDetail>): Brief {
  const results = detail.results || [];
  if (!detail.ran || !results.length) {
    return {
      verdict: "No explanation reached the independent check.",
      lead:
        "This step tests a frozen explanation on cases that were not used to invent " +
        "it. Nothing was available to test, so this run has no verdict of any kind.",
      kpis: [],
      points: [],
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
  const verdict =
    results.length === 1
      ? "One explanation was checked on cases the agent never saw while forming it, and " +
        `it ${supported ? "held up" : refuted ? "was contradicted" : "could not be decided"}.`
      : `Of the ${plural(results.length, "explanation")} checked on cases the ` +
        "agent never saw while forming them, " +
        [
          supported ? `${supported} held up` : "",
          refuted ? `${refuted} ${refuted === 1 ? "was" : "were"} contradicted` : "",
          undecided ? `${undecided} could not be decided` : "",
        ]
          .filter(Boolean)
          .join(", ") +
        ".";

  const m3 = findContract<DiagnosisOutput>(report, "m3");
  const untestable = new Set(
    (m3?.hypotheses || []).filter((h) => !h.test_design?.trim()).map((h) => h.id),
  );
  const m4 = findContract<HypothesisTestOutput>(report, "m4");
  const stuck = (m4?.results || []).filter(
    (r) => r.status === "inconclusive" && untestable.has(r.hypothesis_id),
  );

  const points = results.slice(0, 4).map((result, i) => {
    const full = stripMarkdown(
      result.plain_statement || result.hypothesis || result.statement || `Explanation ${i + 1}`,
    );
    // The claim is a whole M3 statement on some runs; keep the bullet a
    // summary by cutting only at the author's own punctuation.
    const claim = (headlineSplit(full)[0] || full).replace(/[.]$/, "");
    const status = String(result.status || "inconclusive").toLowerCase();
    return `${claim} — ${M4_WORD[status] || status}.`;
  });
  if (consistent < results.length) {
    const off = results.length - consistent;
    points.push(
      `${plural(off, "check")} answered a question this run was not actually ` +
        `asking, and ${off === 1 ? "was" : "were"} kept out of the conclusions.`,
    );
  }

  const withEffect = results.filter((r) => typeof r.effect_size === "number");
  const option = {
    grid: { left: 130, right: 26, top: 10, bottom: 26 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      valueFormatter: (v: unknown) => chartValue(v, { signed: true }),
    },
    xAxis: {
      type: "value",
      axisLabel: { color: tc("#8da19b") },
      splitLine: { lineStyle: { color: tc("#23332f") } },
    },
    yAxis: {
      type: "category",
      data: withEffect.map((_, i) => `H${i + 1}`),
      axisLabel: { color: tc("#b8c9c4") },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      {
        type: "bar",
        barWidth: 14,
        data: withEffect.map((r) => ({
          value: r.effect_size,
          itemStyle: {
            color:
              String(r.status).toLowerCase() === "supported"
                ? tc("#6bd8ad")
                : String(r.status).toLowerCase() === "refuted"
                  ? tc("#f06d5f")
                  : tc("#586a65"),
            borderRadius: 4,
          },
        })),
      },
    ],
  };

  return {
    verdict,
    lead:
      "First an explanation is frozen. Then it is tested against cases the agent did " +
      "not use to invent it. This is the only step in the report allowed to say whether " +
      "an idea held up.",
    kpis: [
      { label: "Explanations checked", value: results.length },
      {
        label: "Held up",
        value: supported,
        note: supported ? "survived the held-out test" : "none",
      },
      { label: "Contradicted", value: refuted },
      { label: "Could not be decided", value: undecided },
    ],
    // A single grey bar for a single undecided hypothesis is a chart that
    // conveys nothing and reads as a rendering failure. The picture earns its
    // place once there is either a comparison to make or a decided verdict to
    // show; otherwise the words carry it.
    figure:
      withEffect.length && (withEffect.length > 1 || supported || refuted) ? (
        <ReactECharts
          option={option}
          notMerge
          style={{ height: Math.max(160, withEffect.length * 42 + 50) }}
        />
      ) : undefined,
    figureNote:
      "How large a gap each explanation predicted, and whether it survived. " +
      "Green held up, red pointed the other way, grey could not be decided.",
    points,
    caveat: stuck.length
      ? stuck.length === 1
        ? 'One "could not be decided" verdict is not a shortage of evidence: that ' +
          "proposal named no test, so nothing measured here could have settled it " +
          "either way. Re-running with more cases returns the same answer."
        : `${stuck.length} "could not be decided" verdicts are not a shortage of evidence: ` +
          "those proposals named no test, so nothing measured here could have settled " +
          "them either way. Re-running with more cases returns the same answer."
      : undefined,
  };
}
