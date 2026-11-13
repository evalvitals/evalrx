/**
 * M5 — repair: candidates compared case by case against the unchanged model.
 *
 * `M5Detail` is the stage's full record (L3); `briefM5` derives its one-screen
 * summary (L2) from the same data.
 */
import ReactECharts from "echarts-for-react";
import { tc } from "../theme";
import { chartValue, findContract, oddsPhrase, oddsValue, plural } from "../reportAccess";
import { Wrench } from "lucide-react";
import type {
  FixAttemptWire,
  FixOutput,
  M5StageDetail,
  RepairCandidateView,
  RepairExample,
  RepairOperationPreview,
  ReportData,
} from "../types";
import { type Brief } from "../brief";
import { EmptyStage, KeyValueGrid } from "../detail";
import {
  humanize,
  plainCandidateKind,
  number,
  percent,
  formatCompact,
  displayValue,
} from "../format";
import { StageBanner, StageKpis, ExampleSection, ExampleMedia } from "./layout";

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

function refFor(report: ReportData, name?: string | null) {
  if (!name) return "";
  const row = repairRefs(report).get(name);
  if (row?.ref) return row.ref;
  const order = [...repairRefs(report).keys()].indexOf(name);
  return order >= 0 ? `R${order + 1}` : "";
}

function headlineFor(report: ReportData, name?: string | null) {
  return (name && repairRefs(report).get(name)?.headline) || "";
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
function SelectionSweep({ report }: { report: ReportData }) {
  const m5 = findContract<FixOutput>(report, "m5_fix");
  const rows = m5?.selection || [];
  if (!rows.length) return null;
  const tiers = [...new Set(rows.map((r) => r.tier))].sort();
  const anyHeadline = rows.some((r) => r.headline);
  const chosen = m5?.selected_on_explore;
  const tone = (v?: string) =>
    v === "fixed"
      ? "ok"
      : v === "unsafe" || v === "regressed"
        ? "bad"
        : v === "partial" || v === "model_independent"
          ? "warn"
          : "";
  // The verdicts are contract enum values, and three of the seven do not
  // survive being read literally: "no effect" invites "so it did nothing",
  // which is the result; "model independent" means the scaffold, not the
  // model, produced the answer.
  const plainVerdictWord = (v?: string) =>
    (
      ({
        fixed: "fixed errors",
        partial: "helped some, hurt some",
        unsafe: "broke more than it fixed",
        regressed: "made things worse",
        no_effect: "changed nothing",
        not_executed: "could not run",
        model_independent: "the scaffold answered, not the model",
      }) as Record<string, string>
    )[String(v || "")] || String(v || "").replace(/_/g, " ");
  return (
    <section className="phase">
      <span className="eyebrow">Stage 1 · choosing — not evidence</span>
      <h3>{rows.length === 1 ? "One repair was tried" : `${rows.length} repairs were tried`}</h3>
      <p className="stat-caption">
        Each was run on the diagnosis cases to pick one worth confirming. These numbers chose the
        repair, so they cannot also test it — read them as what was ruled out, never as results.
        {chosen ? ` ${refFor(report, chosen) || chosen} was taken forward.` : ""}
      </p>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              {[
                "",
                "#",
                "Tier",
                ...(anyHeadline ? ["What it changes"] : []),
                "Result",
                "Errors fixed",
                "New errors",
                "Net change",
              ].map((h, i) => (
                <th key={`${h}-${i}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr
                key={`${r.name}-${i}`}
                className={chosen && r.name === chosen ? "row-chosen" : undefined}
              >
                <td>
                  <span className={`chip ${tone(r.verdict)}`}>
                    <span className="dot" />
                  </span>
                </td>
                <td className="repair-ref">{r.ref || `R${i + 1}`}</td>
                <td>{r.tier}</td>
                {anyHeadline && (
                  <td className="txt">{r.headline || <em className="muted">not described</em>}</td>
                )}
                <td>{plainVerdictWord(r.verdict)}</td>
                <td>{r.n_fixed}</td>
                <td>{r.n_broken}</td>
                <td>
                  {r.effect === null || r.effect === undefined
                    ? "—"
                    : `${r.effect >= 0 ? "+" : ""}${Number(r.effect).toFixed(3)}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mono-sm">
        “Net change” is the share of cases that improved minus the share that got worse, so +0.083
        means about eight cases in a hundred came out better than before.
        {tiers.length > 1 &&
          ` Tiers reached: ${tiers.join(", ")}. A tier missing here had no
        repair to offer — for L3a that usually means the model exposes no internals to read.`}
      </p>
    </section>
  );
}

/**
 * The case ids a candidate repaired or broke, as links into the Case Studio.
 *
 * "12 repaired, 1 broken" is two numbers; these are thirteen cases someone can
 * read. A case the report has no record of is shown greyed and unclickable
 * rather than as a dead link — on a run whose held-out split went unlogged that
 * is every one of them, and the reader needs to know that is why.
 */
function CaseLinks({
  ids,
  kind,
  report,
  navigate,
}: {
  ids?: string[];
  kind: "fixed" | "broken";
  report: ReportData;
  navigate?: (view: string) => void;
}) {
  if (!ids?.length) return null;
  const known = new Set(report.cases.map((c) => c.id));
  const missing = ids.filter((id) => !known.has(id)).length;
  return (
    <div className="case-links">
      <small>
        {kind === "fixed" ? "REPAIRED" : "BROKEN"} ({ids.length})
      </small>
      <div>
        {ids.map((id) =>
          known.has(id) ? (
            <button
              className={`case-chip ${kind}`}
              key={id}
              onClick={() => navigate?.(`cases:${id}`)}
            >
              {id}
            </button>
          ) : (
            <span
              className="case-chip missing"
              key={id}
              title="No record of this case in the report"
            >
              {id}
            </span>
          ),
        )}
      </div>
      {missing > 0 && (
        <em>
          {missing === ids.length ? "None of these cases" : `${missing} of these cases`} were
          recorded in this run, so they cannot be opened. The repair was validated on the held-out
          split, and this run logged only the diagnosis split.
        </em>
      )}
    </div>
  );
}

export function M5Detail({
  data,
  report,
  navigate,
}: {
  data: Partial<M5StageDetail>;
  report: ReportData;
  navigate?: (view: string) => void;
}) {
  const candidates = data.candidates || [];
  if (!data.ran || !candidates.length)
    return (
      <>
        <StageBanner kind="INTERVENTION" title="Repair and regression check">
          M5 compares targeted changes against the same unmodified baseline cases.
        </StageBanner>
        <SelectionSweep report={report} />
        <EmptyStage
          title={
            data.skipped
              ? "Repair was deliberately held back"
              : data.ran
                ? "No repair candidate was testable"
                : "Repair was not reached"
          }
          body={
            data.skipped
              ? data.skip_detail ||
                "The evidence review did not yet accept a mechanism for repair. The next step is a targeted diagnostic probe, not a failed repair."
              : data.ran
                ? "The stage opened, but no accepted and testable mechanism produced a repair candidate."
                : "The run stopped before a targeted intervention could be evaluated."
          }
        />
      </>
    );
  const fixed = candidates.reduce((sum, item) => sum + Number(item.n_fixed || 0), 0);
  const broken = candidates.reduce((sum, item) => sum + Number(item.n_broken || 0), 0);
  const winner =
    candidates.find((item) => item.fixed) ||
    candidates.reduce<RepairCandidateView | null>(
      (best, item) =>
        Number(item.effect || -Infinity) > Number(best?.effect || -Infinity) ? item : best,
      null,
    );
  const option = {
    grid: { left: 145, right: 24, top: 18, bottom: 32 },
    color: [tc("#6bd8ad"), tc("#f06d5f")],
    tooltip: { trigger: "axis" },
    legend: { textStyle: { color: tc("#9fb2ac") } },
    xAxis: {
      type: "value",
      axisLabel: { color: tc("#8fa49d") },
      splitLine: { lineStyle: { color: tc("#22332e") } },
    },
    yAxis: {
      type: "category",
      data: candidates
        .map((item, i) => `${refFor(report, item.name) || `R${i + 1}`} · ${item.tier || "?"}`)
        .reverse(),
      axisLabel: { color: tc("#b8c9c4"), width: 130, overflow: "truncate" },
    },
    series: [
      {
        name: "Repaired",
        type: "bar",
        stack: "cases",
        data: candidates.map((item) => item.n_fixed || 0).reverse(),
      },
      {
        name: "Broken",
        type: "bar",
        stack: "cases",
        data: candidates.map((item) => item.n_broken || 0).reverse(),
      },
    ],
  };
  return (
    <>
      <StageBanner kind="INTERVENTION" title="Paired repair sweep">
        Every candidate is compared case-by-case with the unchanged model. A useful repair must fix
        failures without breaking cases that were right before, and its lead must hold up after
        discounting for how many candidates were tried at once.
      </StageBanner>
      <StageKpis
        items={[
          { label: "Candidates tried", value: candidates.length },
          { label: "Repaired flips", value: fixed },
          { label: "Broken flips", value: broken },
          {
            label: "Best repair",
            value:
              refFor(report, winner?.name) || (winner ? `R${candidates.indexOf(winner) + 1}` : "—"),
            note: headlineFor(report, winner?.name) || winner?.headline || winner?.tier,
          },
        ]}
      />
      <div className={`repair-outcome ${data.fixed ? "success" : "neutral"}`}>
        <Wrench />
        <div>
          <span>FINAL REPAIR OUTCOME</span>
          <h3>{data.fixed ? "A repair was confirmed" : "No candidate passed the repair gate"}</h3>
          {/* The sentence under the verdict was `candidate.summary` — the audit line,
      "[mcnemar + e-value (paired binary)] ... -> inconclusive". Worse than
      unreadable, it sat beside "Repaired flips: 15" with nothing explaining
      how both are true. The derived sentence answers that: it fixed 15 and
      broke 6, and the evidence against luck is not strong enough to count.
      The audit line stays under the candidate's details toggle. */}
          {headlineFor(report, winner?.name) || winner?.headline ? (
            <p>{headlineFor(report, winner?.name) || winner?.headline}</p>
          ) : winner ? (
            <RepairVerdict candidate={winner} />
          ) : (
            <p>Inspect the full repair sweep below.</p>
          )}
        </div>
      </div>
      {!!data.examples?.length && <M5ExampleSection examples={data.examples} report={report} />}
      {!!data.operation_previews?.length && (
        <RepairOperationPreviews examples={data.operation_previews} report={report} />
      )}
      <SelectionSweep report={report} />
      <div className="repair-chart">
        <h3>Confirmed on held-out cases: paired flips vs. baseline</h3>
        <ReactECharts option={option} style={{ height: Math.max(300, candidates.length * 48) }} />
      </div>
      <div className="candidate-list">
        {candidates.map((candidate, index) => (
          <article
            className={`candidate-card candidate-${candidate.verdict || "unknown"}`}
            key={`${candidate.name}-${index}`}
          >
            <header>
              <span>
                {refFor(report, candidate.name) || `R${index + 1}`}
                <em>{candidate.tier || "?"}</em>
              </span>
              <div>
                {/* The producer's own headline, from the contract or the stage row. The
      raw `summary` is the audit line ("[mcnemar + e-value …] -> REJECT H0") and
      set as a card TITLE it is the first thing the reader met; it stays intact
      under the details toggle below. */}
                <h3>
                  {headlineFor(report, candidate.name) ||
                    candidate.headline ||
                    plainCandidateKind(candidate.kind)}
                </h3>
                <small className="mono-sm">{candidate.name}</small>
              </div>
              <b>{candidate.fixed ? "helped" : "did not pass"}</b>
            </header>
            <div className="candidate-metrics">
              <span>
                <b>{candidate.n_fixed ?? 0}</b> errors fixed
              </span>
              <span>
                <b>{candidate.n_broken ?? 0}</b> new errors
              </span>
              <span>
                <b>{number(candidate.effect)}</b> net change
              </span>
              <span>
                <b>{percent(candidate.coverage)}</b> of errors covered
              </span>
            </div>
            <RepairVerdict candidate={candidate} />
            <details>
              <summary>Technical repair definition and affected cases</summary>
              {candidate.summary && <pre className="stat-line">{candidate.summary}</pre>}
              <KeyValueGrid values={candidate.payload || {}} />
            </details>
            <CaseLinks
              ids={candidate.fixed_cases}
              kind="fixed"
              report={report}
              navigate={navigate}
            />
            <CaseLinks
              ids={candidate.broken_cases}
              kind="broken"
              report={report}
              navigate={navigate}
            />
          </article>
        ))}
      </div>
    </>
  );
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
function RepairVerdict({ candidate }: { candidate: RepairCandidateView }) {
  const fixed = Number(candidate.n_fixed || 0);
  const broke = Number(candidate.n_broken || 0);
  const pairs = Number(candidate.n_pairs || 0);
  const e =
    typeof candidate.e_value === "number" && Number.isFinite(candidate.e_value)
      ? candidate.e_value
      : null;
  // An e-value is odds: e=45 means the evidence runs about 45 to 1 against
  // this being luck. "Reject at 0.05" is the same statement in a dialect
  // nobody outside the field speaks.
  const odds =
    e === null
      ? null
      : e >= 1000
        ? "over 1000 to 1"
        : e >= 10
          ? `about ${Math.round(e)} to 1`
          : `about ${e.toFixed(1)} to 1`;
  const counted =
    pairs > 0
      ? `Across ${pairs} case${pairs === 1 ? "" : "s"} it was tested on, this repair fixed ${fixed} and broke ${broke}.`
      : `This repair fixed ${fixed} case${fixed === 1 ? "" : "s"} and broke ${broke}.`;
  const judged =
    odds === null
      ? "The run recorded no significance test for it."
      : candidate.reject
        ? `The evidence against that being luck runs ${odds}, which is strong enough to count.`
        : `The evidence against that being luck runs only ${odds}, which is not strong enough to count.`;
  const covered =
    typeof candidate.coverage === "number" && Number.isFinite(candidate.coverage)
      ? ` It applied to ${percent(candidate.coverage)} of the failures it was aimed at.`
      : "";
  const independent =
    Number(candidate.n_model_independent || 0) > 0
      ? ` ${candidate.n_model_independent} case${candidate.n_model_independent === 1 ? " was" : "s were"} solved by the added code rather than by the model, and were left out of the count.`
      : "";
  return (
    <p className="repair-verdict">
      {counted} {judged}
      {covered}
      {independent}
    </p>
  );
}

/**
 * The example deck wrapper, worded for whichever case M5's search actually
 * produced. A confirmed repair gets the "accepted" framing it earned; an
 * unconfirmed one is still shown — that is the whole point, a reader
 * debugging a failed search needs a real case, not just a gate that closed —
 * but the copy around it says plainly that nothing here was accepted.
 */
function M5ExampleSection({ examples, report }: { examples: RepairExample[]; report: ReportData }) {
  const first = examples[0] || {};
  const confirmed = first.confirmed !== false;
  const broke = !confirmed && first.kind === "broken";
  const eyebrow = confirmed
    ? "A repaired case"
    : broke
      ? "What the search actually broke"
      : "The search's best attempt";
  const title = confirmed
    ? "One real before-and-after repair"
    : broke
      ? "No case was fixed — here is one that broke"
      : "One real attempt, not an accepted repair";
  const note = confirmed
    ? "This is a case counted as fixed. The repair was accepted only after checking every paired case for improvements and regressions."
    : "No candidate cleared the significance bar, so nothing below was accepted as a repair. This is the strongest attempt's own case — shown so the search is debuggable, not just marked failed.";
  return (
    <ExampleSection eyebrow={eyebrow} title={title} note={note}>
      <div className="example-deck">
        {examples.map((example) => (
          <M5Example example={example} report={report} key={example.id} />
        ))}
      </div>
    </ExampleSection>
  );
}

function M5Example({ example, report }: { example: RepairExample; report: ReportData }) {
  const confirmed = example.confirmed !== false;
  const broke = example.kind === "broken";
  const label = confirmed
    ? "A CASE THE REPAIR HELPED"
    : broke
      ? "A CASE THE ATTEMPT BROKE"
      : "A CASE THE ATTEMPT HELPED — UNCONFIRMED";
  const status = confirmed ? "fixed" : broke ? "broken" : "unconfirmed";
  const statusLabel = confirmed ? "fixed" : broke ? "broken" : "not confirmed";
  return (
    <article className="example-card m5-example">
      <header>
        <span className="example-step">{label}</span>
        <em className={`status status-${status}`}>{statusLabel}</em>
      </header>
      <ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} />
      <div className="example-prompt">
        {example.input || "The original task text was not retained."}
      </div>
      <div className="example-flow">
        <div>
          <small>{broke ? "BEFORE ATTEMPT" : "BEFORE REPAIR"}</small>
          <b>
            {example.baseline_available ? displayValue(example.baseline_output) : "Not retained"}
          </b>
        </div>
        <div>
          <small>{broke ? "AFTER ATTEMPT" : "AFTER REPAIR"}</small>
          <b>{displayValue(example.repaired_output)}</b>
        </div>
        <div>
          <small>EXPECTED ANSWER</small>
          <b>{displayValue(example.expected)}</b>
        </div>
      </div>
      <footer>{example.plain_reading}</footer>
    </article>
  );
}

function RepairOperationPreviews({
  examples,
  report,
}: {
  examples: RepairOperationPreview[];
  report: ReportData;
}) {
  return (
    <ExampleSection
      eyebrow="A repair operation, made concrete"
      title="What the image-aware repair would do to a real case"
      note="This shows the source case and the operation declared by the repair candidate. It does not claim an altered image or a successful answer unless the run recorded one."
    >
      <div className="operation-deck">
        {examples.map((example) => (
          <article className="operation-card repair-operation" key={example.id}>
            <header>
              <span>
                {example.executed ? "CANDIDATE WAS TESTED" : "CANDIDATE PREVIEW — NOT EXECUTED"}
              </span>
              <em>{refFor(report, example.candidate) || example.candidate}</em>
            </header>
            <ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} />
            <p className="operation-prompt">{example.input}</p>
            <div className="example-flow">
              <div>
                <small>BASELINE ANSWER</small>
                <b>{displayValue(example.observed)}</b>
              </div>
              <div>
                <small>EXPECTED ANSWER</small>
                <b>{displayValue(example.expected)}</b>
              </div>
            </div>
            <div className="operation-steps">
              {example.operations.map((operation, index) => (
                <div key={`${operation.raw_action}-${index}`}>
                  <span>{index + 1}</span>
                  <section>
                    <b>{operation.action}</b>
                    {Object.keys(operation.parameters || {}).length > 0 && (
                      <small>
                        {Object.entries(operation.parameters)
                          .map(([key, value]) => `${humanize(key)}: ${formatCompact(value)}`)
                          .join(" · ")}
                      </small>
                    )}
                  </section>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </ExampleSection>
  );
}

// ── M5 brief (L2) ──────────────────────────────────────────────────────────
export function briefM5(report: ReportData, detail: Partial<M5StageDetail>): Brief {
  const candidates = detail.candidates || [];
  const m5 = findContract<FixOutput>(report, "m5_fix");
  const sweep = m5?.selection || [];

  if (!detail.ran || !candidates.length) {
    const tried = sweep.length;
    return {
      verdict: detail.skipped
        ? "Repair was deliberately held back."
        : tried
          ? `${plural(tried, "repair")} ${tried === 1 ? "was" : "were"} tried while choosing one, ` +
            "and none was worth confirming."
          : "No repair was attempted.",
      lead:
        detail.skip_detail ||
        detail.skip_reason ||
        "A repair is only attempted once an explanation has survived the held-out " +
          "check, so that there is something specific to repair.",
      kpis: tried
        ? [{ label: "Repairs tried", value: tried, note: "none reached confirmation" }]
        : [],
      points: sweep
        .slice(0, 4)
        .map((row) => `${row.ref || row.name}${row.headline ? ` — ${row.headline}` : ""}`),
    };
  }

  const winner =
    candidates.find((item) => item.fixed) ||
    candidates.reduce<RepairCandidateView | null>(
      (best, item) =>
        Number(item.effect ?? -Infinity) > Number(best?.effect ?? -Infinity) ? item : best,
      null,
    );
  const fixed = Number(winner?.n_fixed || 0);
  const broke = Number(winner?.n_broken || 0);
  const pairs = Number(winner?.n_pairs || 0);
  const odds = oddsPhrase(winner?.e_value);
  // The e-value gate this candidate was judged against (1/alpha), when the
  // run recorded it; older runs did not, and the note then names no bar.
  const gate = oddsValue(winner?.e_threshold);
  // The plain description of a repair can live in either place. A "floor"
  // candidate carries it on the stage row; one the judge authored carries it
  // only in the contract, and reading just the stage row lost the sentence for
  // exactly the runs where it was written by hand. Neither source is
  // guaranteed — a run with neither gets no sentence rather than its slug.
  const headline = String(
    winner?.headline ||
      [...(m5?.attempted || []), ...sweep].find((row) => row.name === winner?.name)?.headline ||
      "",
  ).trim();

  const held = candidates.filter((item) => item.fixed).length;
  const ratio = oddsValue(winner?.e_value);
  const bar = gate ? `the ${gate} bar` : "the bar";
  const heldNote = held
    ? `cleared ${bar}${ratio ? ` at ${ratio}` : ""}`
    : `none cleared ${bar}${ratio ? ` · best ran ${ratio}` : ""}`;
  const heldWord =
    held > 1 ? `${held} repairs held up, the strongest of them` : "One repair held up";
  const verdict = detail.fixed
    ? headline
      ? `${heldWord}: ${headline.charAt(0).toLowerCase()}${headline.slice(1)}`
      : `${heldWord} on cases it had not been tuned on.`
    : `${plural(candidates.length, "repair")} ${candidates.length === 1 ? "was" : "were"} ` +
      "tested against the unchanged model, and none earned its place.";

  const points: string[] = [];
  if (pairs || fixed || broke) {
    points.push(
      `Across ${plural(pairs || 0, "case")} it was tested on, the best repair fixed ` +
        `${fixed} and broke ${broke}.`,
    );
  }
  if (odds) {
    points.push(
      winner?.reject
        ? `The evidence against that being luck runs ${odds}, which is strong enough to count.`
        : `The evidence against that being luck runs only ${odds}, which is not strong enough to count.`,
    );
  }
  if (typeof winner?.coverage === "number") {
    const pct = Math.round(winner.coverage * 100);
    // At 100% "the rest were out of its reach" describes an empty set.
    points.push(
      pct >= 100
        ? "It applied to every failure it was aimed at — none was out of its reach."
        : `It applied to ${pct}% of the failures it was aimed at; the rest were out of ` +
            "its reach entirely.",
    );
  }
  if (candidates.length > 1) {
    // "Confirmed" would overclaim: every candidate reached the confirmation
    // stage, and most of them lost there.
    points.push(
      `${candidates.length - 1} other ${candidates.length - 1 === 1 ? "repair was" : "repairs were"} ` +
        "tested on the same cases and are listed one level down.",
    );
  }

  // The best repair's tested cases as one ring: what it moved against what
  // it left alone. Fixed and broken are the two live slices; the two
  // "unchanged" slices are deliberately recessive so the eye lands on the
  // moved ones and reads how small (or large) a share of the pool they are.
  // Counts derive from the paired tally the run recorded, so the four sum to
  // the cases tested; a run that logged no baseline tally shows only the
  // slices it can vouch for.
  const baselineRight = Number(winner?.n_baseline_correct);
  const stillRight = Number.isFinite(baselineRight) ? Math.max(0, baselineRight - broke) : null;
  const stillWrong =
    stillRight !== null && pairs ? Math.max(0, pairs - stillRight - fixed - broke) : null;
  const slices = [
    { name: "errors fixed", value: fixed, color: tc("#6bd8ad") },
    { name: "new errors", value: broke, color: tc("#f06d5f") },
    ...(stillRight !== null
      ? [{ name: "still right", value: stillRight, color: tc("#8fa49d") }]
      : []),
    ...(stillWrong !== null
      ? [{ name: "still wrong", value: stillWrong, color: tc("#4f615b") }]
      : []),
  ];
  const net = fixed - broke;
  const option = {
    // Drawn in one frame: an entry animation only delays the ring, and a
    // capture taken from a background tab (where animation frames never
    // fire) would otherwise show the legend and the centre with no ring.
    animation: false,
    tooltip: { trigger: "item", valueFormatter: chartValue },
    color: slices.map((slice) => slice.color),
    legend: {
      bottom: 0,
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 14,
      textStyle: { color: tc("#b8c9c4"), fontSize: 11 },
      formatter: (name: string) =>
        `${name}  ${slices.find((slice) => slice.name === name)?.value ?? ""}`,
    },
    // Net change in the middle: the one number the whole ring is about.
    graphic: [
      {
        type: "group",
        left: "center",
        top: pairs ? "34%" : "38%",
        children: [
          {
            type: "text",
            style: {
              text: `${net > 0 ? "+" : ""}${net}`,
              fontSize: 30,
              fontFamily: "Newsreader",
              fill: tc(net > 0 ? "#6bd8ad" : net < 0 ? "#f06d5f" : "#b8c9c4"),
              textAlign: "center",
            },
          },
          ...(pairs
            ? [
                {
                  type: "text",
                  top: 38,
                  style: {
                    text: `net of ${pairs} tested`,
                    fontSize: 10,
                    fontFamily: "DM Mono",
                    fill: tc("#8da19b"),
                    textAlign: "center",
                  },
                },
              ]
            : []),
        ],
      },
    ],
    series: [
      {
        type: "pie",
        radius: ["56%", "78%"],
        center: ["50%", "44%"],
        label: { show: false },
        avoidLabelOverlap: false,
        // A 2px surface gap between slices so adjacent fills never touch.
        itemStyle: { borderColor: tc("#0d1a16"), borderWidth: 2 },
        emphasis: { scale: false },
        data: slices.map((slice) => ({ name: slice.name, value: slice.value })),
      },
    ],
  };

  const independent = Number(winner?.n_model_independent || 0);
  return {
    verdict,
    lead:
      "Every repair is run case by case against the unchanged model on cases held " +
      "back from the search, so a repair that only looks good on the cases it was " +
      "designed against does not get through.",
    kpis: [
      // "Confirmed" would be wrong for this count: every candidate here reached
      // the confirmation stage, and most of them lost there.
      {
        label: "Repairs tested",
        value: candidates.length,
        note: "on cases held back from the search",
      },
      // The odds against luck ride on this tile rather than getting one of
      // their own: the kicker, this count and the bullet under the chart all
      // already say whether the repair held up, so a fifth tile repeating
      // "not strong enough" added nothing. The number is what is new here —
      // how far from the bar the best candidate ran.
      { label: "Held up", value: held, note: heldNote },
      {
        label: "Errors fixed",
        value: fixed,
        note: pairs ? `out of ${pairs} cases tested` : undefined,
      },
      {
        label: "New errors caused",
        value: broke,
        note: broke ? "cases that were right before" : "none",
      },
    ],
    figure: <ReactECharts option={option} notMerge style={{ height: 280 }} />,
    figureNote:
      `The ${plural(pairs || fixed + broke, "case")} the best repair` +
      `${winner?.ref ? ` (${winner.ref})` : ""} was tested on: how many it turned from ` +
      "wrong to right, how many from right to wrong, and how many it left where they " +
      "were. Both coloured slices matter — a repair that fixes ten and breaks nine has " +
      "done almost nothing.",
    points,
    caveat: independent
      ? `${plural(independent, "case")} ${independent === 1 ? "was" : "were"} solved by the added ` +
        "code rather than by the model, and left out of the count — a scaffold that " +
        "answers the question itself has repaired the task, not the model."
      : undefined,
  };
}
