/**
 * The case-study sheet: the whole run as one failure-to-repair story.
 *
 * The five stage views answer "what happened at M2"; this answers "what
 * happened", in the order a reader meets it — what the probes asked, which
 * signals survived correction, which hypotheses held up on cases the model
 * never saw, which rung of the repair ladder held, and what that cost.
 *
 * Every number is read from `data.case_study`, compiled server-side by
 * `evalvitals/reporting/case_study.py`. Nothing here computes a statistic, and
 * a block whose data is missing renders nothing rather than a zero: a run that
 * stopped at M2 has no ladder, and an empty ladder is not a ladder of failures.
 */
import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { CaseStudy, CaseStudyPhase, CaseStudyVerdict } from "./types";

const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
const signed = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined ? "—" : `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(digits)}`;
/** Never print p as 0: below the resolution of the test, say so instead. */
const pValue = (value: number | null | undefined) =>
  value === null || value === undefined ? "" : value < 0.001 ? "p ≪ .001" : `p = ${value.toFixed(3)}`;
/** `working_memory_capacity_limit` -> `Working memory capacity limit`. */
const humanise = (value: string | null | undefined) => {
  const text = String(value || "").replace(/[_`]+/g, " ").trim();
  return text ? text[0].toUpperCase() + text.slice(1) : "";
};

function Forest({ tests, method }: { tests: CaseStudyPhase["tests"]; method?: string | null }) {
  const rows = tests.filter((test) => !test.degenerate && typeof test.effect === "number");
  if (!rows.length) return null;
  // One shared scale, so two rows of the same length mean the same thing. Laid
  // out in HTML rather than SVG: a viewBox stretched to the card's width turned
  // every point estimate into an oval.
  const span = Math.max(0.2, ...rows.flatMap((row) => [Math.abs(row.effect ?? 0), ...(row.ci || []).map(Math.abs)]));
  const at = (value: number) => `${50 + (value / span) * 46}%`;
  return <div className="cs-forest">
    {rows.map((row, index) => {
      const ci = row.ci || [row.effect ?? 0, row.effect ?? 0];
      const on = row.survives_correction;
      return <div className={`cs-frow${on ? " on" : ""}`} key={`${row.signal}-${index}`}
        title={`${row.signal}: ${signed(row.effect)}${row.ci ? ` CI [${signed(ci[0])}, ${signed(ci[1])}]` : ""}`}>
        <i className="cs-fzero" />
        <b className="cs-fci" style={{ left: at(ci[0]), width: `${((ci[1] - ci[0]) / (span * 2)) * 92}%` }} />
        <em className="cs-fdot" style={{ left: at(row.effect ?? 0) }} />
      </div>;
    })}
    <div className="cs-forest-axis"><span>−{span.toFixed(1)}</span><span>0</span><span>+{span.toFixed(1)}</span></div>
    <p className="cs-note">One row per candidate signal: the extra failure rate when that signal is
      high, with its confidence interval. Filled = survived {method || "multiplicity"} correction.</p>
  </div>;
}

/**
 * One hypothesis: the label on the card, the argument on demand.
 *
 * M3 statements are a paragraph each — three of them turn the card into a wall
 * nobody reads, and clamping them to five lines cuts the qualifier that makes a
 * hypothesis falsifiable. So the card carries the claim's name and the full
 * text opens over it, scrolling rather than truncating.
 */
/** A popover anchored to a row, closing on an outside click or Escape. */
function useDismiss<E extends HTMLElement>(open: boolean, close: () => void) {
  const box = useRef<E>(null);
  useEffect(() => {
    if (!open) return;
    const away = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) close();
    };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") close(); };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", escape);
    };
  }, [open, close]);
  return box;
}

function Hypothesis({ hypothesis }: { hypothesis: CaseStudy["m3"][number] }) {
  const [open, setOpen] = useState(false);
  const box = useDismiss<HTMLDivElement>(open, () => setOpen(false));
  return <div className="cs-hyp-wrap" ref={box}>
    <button type="button" className={`cs-hyp${open ? " open" : ""}`} onClick={() => setOpen((was) => !was)}
      aria-expanded={open}>
      <span className="cs-hyp-id">{hypothesis.id} · {humanise(hypothesis.failure_mode)}</span>
      <ChevronDown size={13} />
    </button>
    {open && <div className="cs-hyp-pop" role="dialog">
      <div className="cs-hyp-pop-head">{hypothesis.id} · {humanise(hypothesis.failure_mode)}</div>
      <p>{hypothesis.statement}</p>
      {hypothesis.expected_direction &&
        <small>Expected direction: {hypothesis.expected_direction}</small>}
    </div>}
  </div>;
}


function Verdict({ verdict }: { verdict: CaseStudyVerdict }) {
  const [open, setOpen] = useState(false);
  const box = useDismiss<HTMLDivElement>(open, () => setOpen(false));
  const supported = verdict.status === "supported";
  return <div className="cs-hyp-wrap" ref={box}>
    <button type="button" className={`cs-verdict${supported ? " sup" : ""}${open ? " open" : ""}`}
      onClick={() => setOpen((was) => !was)} aria-expanded={open}>
      <span className={`cs-pill${supported ? "" : " no"}`}>{String(verdict.status || "untested").toUpperCase()}</span>
      <span className="cs-verdict-name">{verdict.id} · {humanise(verdict.failure_mode)}</span>
      <ChevronDown size={13} />
    </button>
    {open && <div className="cs-hyp-pop" role="dialog">
      <div className="cs-hyp-pop-head">{verdict.id} · {humanise(verdict.failure_mode)}</div>
      {verdict.statement && <p>{verdict.statement}</p>}
      <dl className="cs-verdict-facts">
        {typeof verdict.effect === "number" && <><dt>effect</dt><dd>{signed(verdict.effect)}</dd></>}
        {verdict.ci && <><dt>95% CI</dt><dd>[{signed(verdict.ci[0])}, {signed(verdict.ci[1])}]</dd></>}
        {verdict.test_name && <><dt>test</dt><dd>{verdict.test_name}</dd></>}
        {verdict.evidence_grade && <><dt>evidence</dt><dd>{verdict.evidence_grade}</dd></>}
        {verdict.underpowered ? <><dt>power</dt><dd>underpowered</dd></> : null}
      </dl>
      <small>Adjudicated on the held-out split.</small>
    </div>}
  </div>;
}

export function CaseStudySheet({ sheet }: { sheet: CaseStudy }) {
  const [askOpen, setAskOpen] = useState(false);
  const { headline, m1, m2, m3, m5, m4, repair, validation, example_case: example } = sheet;
  // The held-out pass is what the verdicts were adjudicated on; a run that
  // never got one is screened on explore, and the sheet says which it is.
  const phase = m2?.heldout ? "heldout" : "explore";
  const stats = m2?.[phase];
  const modules = Object.fromEntries(sheet.modules.map((module) => [module.code, module]));

  return <section className="section cs">
    <header>
      <div>
        <span className="section-kicker">THE RUN AS ONE SHEET</span>
        <h2>From model failure to tested repair.</h2>
      </div>
    </header>

    <div className="cs-facts">
      <div><small>MODEL</small><strong>{headline.model}</strong></div>
      <div><small>DATASET</small><strong>{headline.dataset}</strong>
        {headline.n_cases ? <span>{headline.n_cases} cases</span> : null}</div>
      <div><small>BASELINE</small><strong>{pct(headline.baseline_accuracy)}</strong></div>
      <div className="win"><small>AFTER REPAIR</small>
        <strong>{pct(validation?.candidate_rate)}</strong>
        {typeof headline.delta === "number" &&
          <span>{signed(headline.delta, 3)} on {validation?.n_pairs} held-out pairs</span>}</div>
    </div>

    <div className="cs-tracks">
      <div className="cs-track explore"><h3>Explore / Discovery</h3><span>probe freely · inspect many signals · form hypotheses</span></div>
      <div className="cs-track holdout"><h3>Held-out / Validation</h3><span>{validation ? `${validation.n_pairs} paired cases the repair never saw` : "unseen cases only"}</span></div>
    </div>

    <div className="cs-modules">
      {m1 && <article className="cs-mod">
        <div className="cs-tag">M1</div>
        <h3>{modules.M1?.name}</h3>
        <p className="cs-sub">{modules.M1?.subtitle}</p>
        {m1.families.map((family) => <div
          key={family.family} className={`cs-group${family.selected ? "" : " off"}`}>
          <div className="cs-group-head">
            <span>{family.label}</span>
            {family.selected ? <b className="cs-badge">SELECTED</b> : <em>not used</em>}
          </div>
          <ul>{family.probes.map((probe) => <li key={probe.phrase}
            className={probe.confirmed ? "on confirmed" : probe.used ? "on" : ""}
            title={probe.analyzers.length ? probe.analyzers.join(", ") : "not run"}>{probe.phrase}</li>)}</ul>
        </div>)}
        {!!m1.questions.length && <div className="cs-probe-block">
          <button type="button" className={`cs-lbl cs-toggle${askOpen ? " open" : ""}`}
            onClick={() => setAskOpen((was) => !was)} aria-expanded={askOpen}>
            What the probes ask <span>({m1.questions.length})</span>
            <ChevronDown size={12} />
          </button>
          {askOpen && <ul className="cs-probes">{m1.questions.map((item) =>
            <li key={item.question} title={`${item.analyzers.join(", ")} · ${item.n_measurements} measurements, ${item.n_candidates} usable`}>
              {item.question}</li>)}</ul>}
        </div>}
        <div className="cs-meter">
          <div className="cs-meter-bar">
            <b style={{ width: `${((m1.n_forwarded ?? 0) / Math.max(m1.n_measured, 1)) * 100}%` }} />
          </div>
          <div className="cs-meter-cap">{m1.n_measured} measurements · <em>{m1.n_forwarded ?? "—"} forwarded to M2</em></div>
        </div>
      </article>}

      {stats && <article className="cs-mod">
        <div className="cs-tag">M2</div>
        <h3>{modules.M2?.name}</h3>
        <p className="cs-sub">{modules.M2?.subtitle}</p>
        <Forest tests={stats.tests} method={stats.correction_method} />
        <div className="cs-lbl">Statistical tests</div>
        <p className="cs-survive">
          <b>{stats.survivors.length} of {stats.n_in_correction_family}</b> signals survive
          {" "}{stats.correction_method === "BH" ? "Benjamini–Hochberg (BH)" : stats.correction_method || "correction"},
          {" "}on the {phase} pass.
        </p>
        {(() => {
          const best = stats.tests
            .filter((test) => test.survives_correction && typeof test.effect === "number")
            .sort((a, b) => Math.abs(b.effect ?? 0) - Math.abs(a.effect ?? 0))[0];
          // the labels move off the number line so the numbers keep one line
          return best ? <div className="cs-stat" title={best.signal || undefined}>
            <small>effect · 95% CI · p</small>
            <b>{signed(best.effect)} [{signed(best.ci?.[0])},{signed(best.ci?.[1])}] {
              typeof best.p_value === "number" && best.p_value < 0.001
                ? "≪.001" : pValue(best.p_value).replace("p = ", "")}</b>
          </div> : null;
        })()}
      </article>}

      {!!m3.length && <article className="cs-mod">
        <div className="cs-tag">M3</div>
        <h3>{modules.M3?.name}</h3>
        <p className="cs-sub">{modules.M3?.subtitle}</p>
        <div className="cs-lbl">The {["no", "one", "two", "three", "four", "five"][m3.length] || m3.length} hypotheses</div>
        {m3.map((hypothesis) => <Hypothesis key={hypothesis.id} hypothesis={hypothesis} />)}
      </article>}

      {!!m5.length && <article className="cs-mod val">
        <div className="cs-tag">M5</div>
        <h3>{modules.M5?.name}</h3>
        <p className="cs-sub">{modules.M5?.subtitle}</p>
        {m5.map((verdict) => <Verdict key={verdict.id} verdict={verdict} />)}
      </article>}

      {m4 && <article className="cs-mod val cs-wide">
        <div className="cs-tag">M4</div>
        <h3>{modules.M4?.name}</h3>
        <p className="cs-sub">{modules.M4?.subtitle}</p>
        <div className="cs-ladder">
          {m4.ladder.map((rung) => <div key={rung.tier}
            className={`cs-rung${rung.status === "accepted" ? " ok" : rung.status === "regressed" ? " bad" : ""}${rung.status === "untouched" ? " off" : ""}`}>
            <span className="cs-lv">{rung.tier}</span>
            <span className="cs-nm">{rung.label}</span>
            <span className="cs-st">{rung.status === "accepted" ? `Accepted ${signed(rung.best_effect)}`
              : rung.status === "regressed" ? `Regressed ${signed(rung.best_effect)}`
                : rung.status === "untouched" ? (rung.within_cap ? "Untouched" : `Beyond the ${rung.tier_cap} cap`)
                  : `Tried ${rung.n_candidates}, not selected`}</span>
          </div>)}
        </div>
        {!!m4.candidates.length && (() => {
          const reach = Math.max(0.05, ...m4.candidates.map((candidate) => Math.abs(candidate.effect ?? 0)));
          return <>
            <div className="cs-lbl">{m4.candidates.length} candidates tried on explore</div>
            <div className="cs-cands">{m4.candidates.map((candidate, index) => {
              const effect = candidate.effect ?? 0;
              const width = `${(Math.abs(effect) / reach) * 46}%`;
              return <div className={`cs-crow${candidate.selected ? " win" : effect < 0 ? " neg" : ""}`} key={`${candidate.name}-${index}`}>
                <span className="cs-k" title={`${candidate.tier} ${candidate.name}`}>{candidate.tier} {candidate.name}</span>
                <div className="cs-track2">
                  <i className="cs-zero" />
                  <b style={effect < 0 ? { right: "50%", width } : { left: "50%", width }} />
                </div>
                <em>{signed(effect, 3)}</em>
              </div>;
            })}</div>
            <p className="cs-note">Explore selection only; the winner is re-validated on held-out cases below.</p>
          </>;
        })()}
      </article>}
    </div>

    {validation && <div className="cs-repair">
      <div className="cs-repair-head">
        <h3>The accepted repair</h3>
        <code>{repair?.name || validation.candidate}</code>
        <span>{repair?.tier || validation.tier} scaffold</span>
      </div>
      <div className="cs-flow">
        {example && <div className="cs-step cs-case">
          <div className="cs-n">ONE CASE · {example.case_id}</div>
          <h4>Input</h4>
          <p className="cs-case-text">{String(example.question || "").split("\n")[0]}</p>
        </div>}
        {(repair?.steps || []).map((step, index) => <div className="cs-step" key={index}>
          <div className="cs-n">STEP {index + 1}</div>
          <h4>{step.title}</h4>
          <ul className={step.mono ? "mono" : ""}>{step.lines.map((line, i) => <li key={i}>{line}</li>)}</ul>
        </div>)}
        {example && <div className="cs-step cs-final">
          <div className="cs-n">ANSWER</div>
          <h4>Correct answer</h4>
          <code>{String(example.gold ?? "—")}</code>
          {example.baseline_answer && <div className="cs-case-kv">
            <small>{String(example.label || "").toLowerCase() === "fail"
              ? "WHAT THE MODEL ANSWERED — WRONG"
              : "WHAT THE MODEL ANSWERED"}</small>
            <code className="bad">{example.baseline_answer}</code></div>}
        </div>}
        <div className="cs-results">
          <div className="cs-rbar">
            <span className="cs-k">BASELINE</span>
            <div className="cs-t"><b style={{ width: pct(validation.baseline_rate, 0) }} /></div>
            <em>{pct(validation.baseline_rate)}</em>
          </div>
          <div className="cs-rbar win">
            <span className="cs-k">REPAIRED</span>
            <div className="cs-t"><b style={{ width: pct(validation.candidate_rate, 0) }} /></div>
            <em>{pct(validation.candidate_rate)}</em>
          </div>
          <p className="cs-note">
            {validation.n_fixed} fixed, {validation.n_broken} broken over {validation.n_pairs} paired
            held-out cases ({signed(validation.effect, 3)}
            {validation.ci ? `, CI [${signed(validation.ci[0], 3)}, ${signed(validation.ci[1], 3)}]` : ""}).
          </p>
          {example?.split === "explore" && <p className="cs-note">
            The case shown is from the explore split, answered by the model before the repair:
            the repaired output for it was not measured.</p>}
        </div>
      </div>
    </div>}

  </section>;
}
