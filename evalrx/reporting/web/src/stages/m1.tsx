/**
 * M1 — measure: run the behavioural checks and record what each found.
 *
 * `M1Detail` is the stage's full record (L3); `briefM1` derives its one-screen
 * summary (L2) from the same data.
 */
import { countOf, findContract, metric, plural } from "../reportAccess";
import type {
  AnalyzerSelection,
  M1StageDetail,
  Modality,
  ModelCallView,
  OperationExample,
  ProbeExample,
  ProbeOutput,
  ReportData,
} from "../types";
import { reportChart, BriefChart, type Brief } from "../brief";
import { DetailBlock, KeyValueGrid, RecordTable } from "../detail";
import { humanize, seconds, formatCompact, displayValue } from "../format";
import { StageBanner, StageKpis, ExampleSection, ExampleMedia } from "./layout";

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
  return (
    <div className="modality-band">
      <div>
        <small>ROUTED ON</small>
        <b>{routed.map(labelModality).join(" + ") || "text"}</b>
      </div>
      <div>
        <small>MODEL ACCEPTS</small>
        <b>{declared.map(labelModality).join(" + ") || "text"}</b>
      </div>
      <div>
        <small>CASES CARRIED</small>
        <b>{probed.map(labelModality).join(" + ") || "text"}</b>
      </div>
      {selection.is_agent && (
        <div>
          <small>SHAPE</small>
          <b>agent trajectories</b>
        </div>
      )}
      {narrowed && (
        <p className="modality-note">
          The model accepts more than this benchmark exercises, so the checks were narrowed to what
          the cases actually contain.
        </p>
      )}
    </div>
  );
}

function labelModality(slot: Modality): string {
  return { text: "text", image: "images", audio: "audio", video: "video" }[slot] || slot;
}

export function M1Detail({ data, report }: { data: Partial<M1StageDetail>; report: ReportData }) {
  const probes = data.probes || [];
  const examples = data.examples || [];
  const m1 = findContract<ProbeOutput>(report, "m1");
  return (
    <>
      <StageBanner kind="MEASURE" title="Behavioral checkup">
        We give the model real tasks, then run several checks on its behavior. This tells us where
        it struggles; it does not yet tell us why.
      </StageBanner>
      <ModalityBand selection={m1?.selection} />
      <StageKpis
        items={[
          { label: "Probes run", value: data.n_probes || probes.length },
          { label: "Cases measured", value: data.n_measured || "—" },
          { label: "Runtime", value: seconds(data.duration) },
        ]}
      />
      {examples.length > 0 && (
        <ExampleSection
          eyebrow="A real example"
          title="What one M1 check looks like"
          note="Examples make the measurement concrete. The result below is based on all measured cases, not just this one."
        >
          <div className="example-deck">
            {examples.map((example) => (
              <M1Example example={example} report={report} key={example.id} />
            ))}
          </div>
        </ExampleSection>
      )}
      {!!data.operations?.length && (
        <OperationExamples examples={data.operations} report={report} />
      )}
      <div className="probe-grid">
        {probes.map((probe, index) => (
          <details className="probe-card" key={probe.id} open={index === 0}>
            <summary>
              <span className="section-kicker">
                PROBE {String(index + 1).padStart(2, "0")} · {probe.n_cases || "—"} CASES
              </span>
              <h3>{probe.title}</h3>
              <p>{probe.question}</p>
              <div className="probe-metrics">
                {(probe.metrics || []).map((metric) => (
                  <span key={metric.label}>
                    <b>{metric.value ?? "—"}</b>
                    {metric.label}
                  </span>
                ))}
              </div>
            </summary>
            <div className="probe-expanded">
              <p>{probe.description}</p>
              {Object.keys(probe.finding_summary || {}).length > 0 && (
                <KeyValueGrid values={probe.finding_summary} />
              )}
              {(probe.sample_rows || []).length > 0 && (
                <details className="nested-detail">
                  <summary>
                    Per-case measurements ({probe.n_sample_rows || probe.sample_rows.length}
                    {probe.n_sample_rows > probe.sample_rows.length
                      ? `, first ${probe.sample_rows.length}`
                      : ""}
                    )
                  </summary>
                  <RecordTable rows={probe.sample_rows} />
                </details>
              )}
              <ProbeCalls calls={(data.calls || {})[probe.raw_name] || []} />
              {Object.keys(probe.raw_finding_summary || {}).length > 0 && (
                <details className="nested-detail">
                  <summary>Technical measurement names and raw values</summary>
                  <KeyValueGrid values={probe.raw_finding_summary} />
                </details>
              )}
              {!!probe.sample_rows?.length && (
                <details className="nested-detail">
                  <summary>
                    Inspect {probe.sample_rows.length} representative measurement rows
                  </summary>
                  <RecordTable rows={probe.sample_rows} />
                </details>
              )}
            </div>
          </details>
        ))}
      </div>
    </>
  );
}

/**
 * What one probe asked the model and what came back — the audit trail behind
 * its numbers. Collapsed by default: a probe can make hundreds of calls, and
 * the list is for checking one, not reading all.
 */
function ProbeCalls({ calls }: { calls: ModelCallView[] }) {
  if (!calls.length) return null;
  return (
    <details className="nested-detail probe-calls">
      <summary>Model calls ({calls.length}) — prompt and reply, one per call</summary>
      <div className="call-list">
        {calls.map((call, index) => {
          const scope = call.case_id
            ? `case ${String(call.case_id).slice(0, 12)}`
            : call.batch_case_ids?.length
              ? `${call.batch_case_ids.length} cases`
              : "";
          return (
            <details
              className={`call-row${call.error ? " failed" : ""}`}
              key={`${call.seq}-${index}`}
            >
              <summary>
                <code>{String(call.seq ?? index + 1).padStart(3, "0")}</code>
                <b>{call.method || "call"}</b>
                <span>{scope}</span>
                <em>
                  {typeof call.duration_sec === "number" ? `${call.duration_sec.toFixed(2)}s` : ""}
                  {call.output ? ` · ${String(call.output).length} chars out` : ""}
                  {call.error ? " · error" : ""}
                </em>
              </summary>
              <div className="call-io">
                <DetailBlock label="PROMPT" value={call.prompt} />
                <DetailBlock
                  label={call.error ? "ERROR" : "REPLY"}
                  value={call.error || call.output}
                />
                {!!call.batch_case_ids?.length && (
                  <small className="call-cases">cases: {call.batch_case_ids.join(", ")}</small>
                )}
              </div>
            </details>
          );
        })}
      </div>
    </details>
  );
}

function M1Example({ example, report }: { example: ProbeExample; report?: ReportData }) {
  const outcome = String(example.outcome || "unknown").toLowerCase();
  return (
    <article className="example-card m1-example">
      <header>
        <span className="example-step">1 · ORIGINAL TASK</span>
        <em className={`status status-${outcome}`}>{outcome}</em>
      </header>
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
        <div>
          <small>CHECK RUN</small>
          <b>{example.probe_title || "Behavior check"}</b>
        </div>
      </div>
      <p className="example-question">{example.probe_question}</p>
      <div className="example-check">
        <span>2 · WHAT THE CHECK RECORDED</span>
        <KeyValueGrid values={example.check_result || {}} />
      </div>
      <footer>{example.plain_reading}</footer>
    </article>
  );
}

function OperationExamples({
  examples,
  report,
}: {
  examples: OperationExample[];
  report: ReportData;
}) {
  return (
    <ExampleSection
      eyebrow="Recorded agent actions"
      title="See the intermediate work, case by case"
      note="These actions come from the saved agent trajectory. They are not a reconstruction from a prompt or a proposed repair."
    >
      <div className="operation-deck">
        {examples.map((example) => (
          <article className="operation-card" key={example.id}>
            <header>
              <span>CASE {example.case_id}</span>
              <em className={`status status-${String(example.outcome || "unknown").toLowerCase()}`}>
                {example.outcome}
              </em>
            </header>
            <ExampleMedia mediaIds={example.media_ids} report={report} caseId={example.case_id} />
            <p className="operation-prompt">{example.input}</p>
            <div className="operation-steps">
              {example.steps.map((step) => (
                <div key={`${step.order}-${step.raw_action}`}>
                  <span>{step.order}</span>
                  <section>
                    <b>{step.action}</b>
                    {Object.keys(step.parameters || {}).length > 0 && (
                      <small>
                        {Object.entries(step.parameters)
                          .map(([key, value]) => `${humanize(key)}: ${formatCompact(value)}`)
                          .join(" · ")}
                      </small>
                    )}
                    {step.thought && <p>{step.thought}</p>}
                  </section>
                </div>
              ))}
            </div>
            <footer>{example.plain_reading}</footer>
          </article>
        ))}
      </div>
    </ExampleSection>
  );
}

const MODALITY_WORD: Record<string, string> = {
  text: "text",
  image: "images",
  audio: "audio",
  video: "video",
};

function modalityWords(slots?: Modality[]): string {
  const words = (slots || []).map((slot) => MODALITY_WORD[slot] || slot);
  return words.length ? words.join(" + ") : "text";
}

// ── M1 brief (L2) ──────────────────────────────────────────────────────────
export function briefM1(report: ReportData, detail: Partial<M1StageDetail>): Brief {
  const probes = detail.probes || [];
  const measured = Number(detail.n_measured || 0);
  const evaluated = metric(report, "evaluated");
  const failed = metric(report, "failed");
  const m1 = findContract<ProbeOutput>(report, "m1");
  const selection: AnalyzerSelection | undefined = m1?.selection;
  const broken = Object.entries(m1?.failed_analyzers || {});

  const verdict =
    failed !== undefined && evaluated
      ? `${plural(probes.length, "behavioural check")} ran, and the model got ` +
        `${failed} of the ${evaluated} cases wrong.`
      : `${plural(probes.length, "behavioural check")} ran on ` + `${plural(measured, "case")}.`;

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
    points.push(
      routed === probed
        ? `The checks read ${routed}, which is what these cases actually carry.`
        : `The checks read ${routed}, while the cases themselves carry ${probed} — ` +
            "so the parts of the input outside that were never examined.",
    );
  }
  // The checks get a list of their own (`checks`): one bullet that joined
  // five titles with " · and 3 more" mixed questions with Title-Cased ids and
  // read as noise.
  const generic = (text: string) => !text || text.startsWith("Measures model behavior");
  const checks = probes
    .map((probe) => ({
      name: String(probe.display_name || probe.title || probe.raw_name || ""),
      question: generic(String(probe.question || "")) ? undefined : String(probe.question),
      n: probe.n_cases === undefined || probe.n_cases === null ? null : Number(probe.n_cases),
    }))
    .filter((check) => check.name);

  // A check with `n_cases: 0` and a check that ran on 8 of 12 cases are
  // different facts, and lumping them into one filter once produced "Every
  // check ran on all 12 cases" directly above a caveat saying one had not.
  // `coverage` is the checks that measured something; `zeroed` is the checks
  // that explicitly recorded measuring nothing.
  const coverage = probes.map((probe) => Number(probe.n_cases)).filter((n) => n > 0);
  const zeroed = probes.filter(
    (probe) => Number(probe.n_cases) === 0 && probe.n_cases !== undefined && probe.n_cases !== null,
  );
  const lowest = coverage.length ? Math.min(...coverage) : 0;
  if (measured && coverage.length) {
    points.push(
      lowest === measured
        ? coverage.length === probes.length
          ? `Every check ran on all ${plural(measured, "case")}.`
          : `${plural(coverage.length, "check")} reported a case count, and each ran on all ${measured} cases.`
        : `Coverage ranged from ${lowest} to ${measured} cases per check — the shorter ones ` +
            "were capped to keep the step inside its time budget.",
    );
  }
  const capped = coverage.filter((n) => n < measured).length;
  const caveatParts: string[] = [];
  if (zeroed.length) {
    const names = zeroed
      .map((probe) => String(probe.title || probe.raw_name || ""))
      .filter(Boolean)
      .join(", ");
    caveatParts.push(
      `${plural(zeroed.length, "check")}${names ? ` (${names})` : ""} measured ` +
        `no cases at all, so ${zeroed.length === 1 ? "its question" : "their questions"} simply ` +
        "went unanswered this run.",
    );
  }
  if (broken.length) {
    caveatParts.push(
      `${plural(broken.length, "check")} could not run at all ` +
        `(${broken.map(([name]) => name).join(", ")}), so whatever ${broken.length === 1 ? "it" : "they"} ` +
        "would have measured is simply absent from everything downstream — not measured and found clean.",
    );
  }
  if (capped && measured) {
    caveatParts.push(
      `${plural(capped, "check")} ran on fewer than all ${measured} cases, ` +
        "which costs statistical power later but does not bias what was measured.",
    );
  }

  return {
    verdict,
    checks,
    lead:
      "This step only measures. It records what the model does differently on " +
      "the cases it gets right and the ones it gets wrong — it does not yet claim " +
      "to know why anything failed.",
    kpis: [
      {
        label: "Checks run",
        value: probes.length || detail.n_probes || 0,
        note: broken.length ? `${broken.length} could not run` : "all completed",
      },
      {
        label: "Cases measured",
        value: measured || "—",
        note: evaluated ? `out of ${evaluated} evaluated` : undefined,
      },
      ...(failed !== undefined && evaluated
        ? [
            {
              label: "Got it wrong",
              value: countOf(failed, evaluated),
              note: "these are what the rest of the report is about",
            },
          ]
        : []),
      ...(typeof detail.duration === "number"
        ? [{ label: "Time taken", value: `${detail.duration.toFixed(0)}s` }]
        : []),
    ],
    figure: reportChart(report, "outcomes") ? (
      <BriefChart chart={reportChart(report, "outcomes")!} />
    ) : undefined,
    figureNote:
      "How the evaluated cases came out. Everything later in the report is " +
      "an attempt to explain the wrong ones.",
    points,
    caveat: caveatParts.join(" ") || undefined,
  };
}
