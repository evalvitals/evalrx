/**
 * M3 — explain: falsifiable hypotheses proposed from M2's leads (proposals only).
 *
 * `M3Detail` is the stage's full record (L3); `briefM3` derives its one-screen
 * summary (L2) from the same data.
 */
import { findContract, headlineSplit, plural, stripMarkdown } from "../reportAccess";
import { AlertTriangle } from "lucide-react";
import type { DiagnosisOutput, M3StageDetail, Proposal, ReportData } from "../types";
import { BriefImage, sentence, type Brief } from "../brief";
import { EmptyStage, RecordTable } from "../detail";
import { plainFailureMode } from "../format";
import { labelledFigures, EvidenceFigure, StatEvidenceChart } from "./evidence";
import { StageBanner, StageKpis } from "./layout";

/** A design a person can act on, but that names nothing M4 measured this cycle. */
const SIGNAL_REF = /\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b/;

const DIRECTIVES = ["prompt_contrast", "analyzer_params", "strategy_contrast", "paired_rerun"];

function isRoutable(design?: string): boolean {
  const t = (design || "").trim();
  if (!t) return false;
  return SIGNAL_REF.test(t.toLowerCase()) || DIRECTIVES.some((d) => t.includes(d));
}

export function M3Detail({ data, report }: { data: Partial<M3StageDetail>; report: ReportData }) {
  const hypotheses: Proposal[] = data.hypotheses || [];
  const m3 = findContract<DiagnosisOutput>(report, "m3");
  const untestable = m3 ? (m3.hypotheses || []).filter((h) => !h.test_design?.trim()) : [];
  const unroutable = m3
    ? (m3.hypotheses || []).filter((h) => h.test_design?.trim() && !isRoutable(h.test_design))
    : [];
  return (
    <>
      <StageBanner kind="PROPOSAL ONLY" title="Falsifiable mechanisms">
        M3 turns M2 leads into explanations that could be proven wrong. These cards are proposals;
        validation status belongs exclusively to M4.
      </StageBanner>
      {untestable.length > 0 && (
        <div className="parser-warning">
          <AlertTriangle />
          <div>
            <b>
              {untestable.length === 1
                ? "One proposal names no test"
                : `${untestable.length} proposals name no test`}
              .
            </b>
            <p>
              A hypothesis with no test design cannot be decided by any amount of evidence — M4 will
              return “inconclusive” for it on every cycle. Read that verdict as “this claim was
              never testable”, not as “not enough data yet”.
            </p>
          </div>
        </div>
      )}
      {unroutable.length > 0 && (
        <div className="parser-warning">
          <AlertTriangle />
          <div>
            <b>
              {unroutable.length === 1
                ? "One proposal names a measurement nobody has taken yet"
                : `${unroutable.length} proposals name measurements nobody has taken yet`}
              .
            </b>
            <p>
              These do describe an experiment — they just refer to something this cycle did not
              measure, so M4 has nothing to resolve them against. That is work for the next round of
              checks, not a claim without a falsifier.
            </p>
          </div>
        </div>
      )}
      <StageKpis
        items={[
          { label: "Accepted proposals", value: hypotheses.length },
          {
            label: "Test designs",
            value: hypotheses.filter((item) => item.test_design).length,
          },
        ]}
      />
      {!!data.evidence_figures?.length && (
        <section className="m3-evidence">
          <header>
            <span>THE VISUAL EVIDENCE THIS STEP STARTS FROM</span>
            <h3>Patterns the agent is trying to explain</h3>
            <p>
              These charts come from M2. They are observations that motivate the ideas below, not
              confirmation that an idea is true.
            </p>
          </header>
          <div className="analysis-figures">
            {labelledFigures(data.evidence_figures, data.evidence_stats || []).map((figure) => (
              <EvidenceFigure figure={figure} key={figure.id} />
            ))}
          </div>
        </section>
      )}
      {!!data.evidence_stats?.length && (
        <StatEvidenceChart
          stats={data.evidence_stats}
          title="The strongest M2 patterns carried into this step"
          note="M3 turns these observed patterns into testable ideas. M4 is still needed to decide whether an idea holds up."
        />
      )}
      {hypotheses.length ? (
        <div className="hypothesis-list">
          {hypotheses.map((hypothesis, index) => (
            <article className="hypothesis-card" key={index}>
              <header>
                <span>H{index + 1}</span>
                <em>IDEA TO TEST</em>
              </header>
              <h3>{hypothesis.plain_statement || hypothesis.statement || hypothesis.hypothesis}</h3>
              {hypothesis.statement &&
                hypothesis.plain_statement &&
                hypothesis.statement !== hypothesis.plain_statement && (
                  <details>
                    <summary>Technical wording</summary>
                    <p>{hypothesis.statement}</p>
                  </details>
                )}
              <div className="hypothesis-grid">
                <div>
                  <small>WHAT MAY BE GOING WRONG</small>
                  <p>{plainFailureMode(hypothesis.failure_mode)}</p>
                </div>
                <div>
                  <small>WHY THIS IS PLAUSIBLE</small>
                  <p>{hypothesis.basis || "Based on the patterns found in the previous step."}</p>
                </div>
                <div className="test-design">
                  <small>WHAT WOULD PROVE IT WRONG?</small>
                  <p>
                    {hypothesis.test_design ||
                      (m3
                        ? "Nothing — the AI Doctor proposed no test for this idea, so no result can decide it."
                        : "No test design was retained.")}
                  </p>
                  {hypothesis.expected_association && (
                    <details>
                      <summary>Technical test expression</summary>
                      <code>{hypothesis.expected_association}</code>
                    </details>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <EmptyStage
          title="No formal hypotheses"
          body="The earlier pattern search did not yield an idea the pipeline could test."
        />
      )}
      {(!!data.candidate_signals?.length || !!data.recommended_tests?.length) && (
        <details className="agent-transcript">
          <summary>Candidate signals and suggested follow-ups</summary>
          {!!data.candidate_signals?.length && <RecordTable rows={data.candidate_signals} />}
          {data.recommended_tests?.map((item, i) => (
            <p key={i}>• {String(item)}</p>
          ))}
        </details>
      )}
    </>
  );
}

// ── M3 brief (L2) ──────────────────────────────────────────────────────────
export function briefM3(report: ReportData, detail: Partial<M3StageDetail>): Brief {
  const list: Proposal[] = detail.hypotheses || [];
  const m3 = findContract<DiagnosisOutput>(report, "m3");
  const untestable = (m3?.hypotheses || []).filter((h) => !h.test_design?.trim());
  const withTest = list.filter((h) => String(h.test_design || "").trim()).length;
  const evidenceFigure = (detail.evidence_figures || [])[0];

  const verdict = list.length
    ? `${plural(list.length, "possible explanation")} for the failures ${list.length === 1 ? "was" : "were"} ` +
      `put forward, and ${list.length === 1 ? "it is not proven" : "none of them is proven"} yet.`
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
    return `${lead} ${
      clipped
        ? "Shortened here — the full claim and how it will be checked are"
        : "How it will be checked is spelled out"
    } one level down.`;
  });

  const caveatParts: string[] = [];
  if (untestable.length) {
    caveatParts.push(
      `${plural(untestable.length, "proposal")} name${untestable.length === 1 ? "s" : ""} ` +
        'no way of being wrong. Those come back "could not be decided" however much ' +
        'evidence the next step gathers — read that as "never testable", not as ' +
        '"not enough data yet".',
    );
  }

  return {
    verdict,
    lead:
      "This step turns the patterns from the previous one into explanations that " +
      "could be shown to be wrong. Everything on this screen is a proposal; whether " +
      "any of it survives is decided one step later.",
    kpis: [
      { label: "Explanations proposed", value: list.length },
      {
        label: "With a way to test them",
        value: withTest,
        note: withTest < list.length ? `${list.length - withTest} without` : "all of them",
      },
    ],
    // Only a figure this step produced. Borrowing the previous step's chart
    // would put the same picture on two screens in a row, which is exactly the
    // "L2 is L3 rearranged" failure this layer exists to avoid — and it would
    // imply M3 measured something, which it never does.
    figure: evidenceFigure ? <BriefImage figure={evidenceFigure} /> : undefined,
    figureNote:
      "The pattern these explanations are trying to account for. It comes " +
      "from the previous step, and motivates the ideas — it does not confirm them.",
    points,
    caveat: caveatParts.join(" ") || undefined,
  };
}
