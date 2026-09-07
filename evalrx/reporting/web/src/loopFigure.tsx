/**
 * The loop figure: the whole run as the paper's "closing the loop" diagram.
 *
 *   INPUTS → [ EXPLORE / DISCOVERY  M1 · M2 · M3 ] → [ M4 held-out ] → [ M5 repair ] → OUTPUT health card
 *
 * It merges the pipeline strip (which stage ran, click to inspect) with the
 * Takeaway sheet (what each stage actually produced) into one picture, in the
 * report's own dark style. Every number is read from `data.case_study`
 * (compiled by `evalrx/reporting/case_study.py`) and `data.stages`; nothing is
 * computed here, and a block with no data renders its empty state rather than
 * a zero. Clicking a stage opens its evidence view; hovering a stage shows what
 * went in and what came out of it.
 */
import { Check, Database, Lock, Trophy, X } from "lucide-react";
import type { CaseStudy, ReportData, Stage } from "./types";
import { Hypothesis, Verdict, humanise, pct, signed } from "./caseStudy";

type Nav = (view: string) => void;

const STATUS_WORD: Record<string, string> = {
  completed: "completed", "not-run": "not run", skipped: "skipped",
  improved: "improved", "no-improvement": "no improvement",
};

/** One hover tooltip: what entered the stage and what left it. */
function IO({ into, out }: { into: string[]; out: string[] }) {
  return <div className="lf-tip" role="tooltip">
    <div><b>IN</b>{into.length ? into.map((line, i) => <span key={i}>{line}</span>) : <span>—</span>}</div>
    <div><b>OUT</b>{out.length ? out.map((line, i) => <span key={i}>{line}</span>) : <span>—</span>}</div>
    <small>click to open the stage evidence</small>
  </div>;
}

function StageCard({ stage, code, title, subtitle, into, out, onClick, lane, children }: {
  stage?: Stage; code: string; title: string; subtitle: string;
  into: string[]; out: string[]; onClick: () => void; lane: "explore" | "holdout" | "repair";
  children?: React.ReactNode;
}) {
  const status = stage?.status ?? "not-run";
  return <article className={`lf-card lf-${lane} lf-s-${status}`} tabIndex={0} onClick={onClick}
    onKeyDown={(event) => { if (event.key === "Enter") onClick(); }}
    aria-label={`${code} ${title}, ${STATUS_WORD[status] ?? status}`}>
    <header>
      <span className="lf-code">{code}</span>
      <span className={`lf-status lf-status-${status}`}>{STATUS_WORD[status] ?? status}</span>
    </header>
    <h3>{title}</h3>
    <p className="lf-sub">{subtitle}</p>
    <div className="lf-body" onClick={(event) => event.stopPropagation()}>{children}</div>
    <IO into={into} out={out} />
  </article>;
}

/** Top few M2 rows: signal, effect with CI, and whether it survived correction. */
function SignalRows({ stats }: { stats: NonNullable<CaseStudy["m2"]>[string] }) {
  const rows = stats.tests
    .filter((test) => !test.degenerate && typeof test.effect === "number")
    .sort((a, b) => Number(b.survives_correction) - Number(a.survives_correction) || Math.abs(b.effect ?? 0) - Math.abs(a.effect ?? 0))
    .slice(0, 4);
  if (!rows.length) return <p className="lf-empty">no measurable signal</p>;
  const span = Math.max(0.2, ...rows.flatMap((row) => [Math.abs(row.effect ?? 0), ...(row.ci || []).map(Math.abs)]));
  const at = (value: number) => `${50 + (value / span) * 46}%`;
  return <div className="lf-signals">
    <div className="lf-sighead">effect ± CI · <Check size={9} /> survived correction</div>
    {rows.map((row, index) => {
      const ci = row.ci || [row.effect ?? 0, row.effect ?? 0];
      return <div key={`${row.signal}-${index}`} className={`lf-sig${row.survives_correction ? " on" : ""}`}
        title={`${row.signal}: ${signed(row.effect)}${row.ci ? ` CI [${signed(ci[0])}, ${signed(ci[1])}]` : ""}${typeof row.p_value === "number" ? ` p=${row.p_value < 0.001 ? "≪.001" : row.p_value.toFixed(3)}` : ""}`}>
        <span className="lf-sname">s<sub>{index + 1}</sub></span>
        <span className="lf-strack"><i /><b style={{ left: at(ci[0]), width: `${((ci[1] - ci[0]) / (span * 2)) * 92}%` }} /><em style={{ left: at(row.effect ?? 0) }} /></span>
        <span className="lf-smark">{row.survives_correction ? <Check size={12} /> : <X size={12} />}</span>
      </div>;
    })}
  </div>;
}

export function LoopFigureView({ data, navigate }: { data: ReportData; navigate: Nav }) {
  const cs = data.case_study ?? null;
  const stage = Object.fromEntries(data.stages.map((item) => [item.code, item])) as Record<string, Stage>;
  const modules = Object.fromEntries((cs?.modules ?? []).map((module) => [module.code, module]));
  const modality = data.media.some((m) => m.kind === "audio") ? "alm"
    : data.media.some((m) => m.kind === "image" || m.kind === "video") ? "vlm" : "llm";

  // `setting.n_cases` is run_start's count, which the loop records for the
  // explore partition it actually diagnosed on; everything else in the case
  // list was withheld from M1-M3. The confirm pairs are known only once the
  // fix stage measured a candidate on them.
  const nTotal = data.cases.length;
  const nExplore = data.setting.n_cases || cs?.headline.n_explore || nTotal;
  const nHeldout = nTotal > nExplore ? nTotal - nExplore : (cs?.headline.n_heldout ?? null);
  const nConfirm = cs?.validation?.n_pairs ?? null;
  const m1 = cs?.m1 ?? null;
  const phase = cs?.m2?.heldout ? "heldout" : "explore";
  const stats = cs?.m2?.[phase] ?? null;
  const m3 = cs?.m3 ?? [];
  const m4 = cs?.m4 ?? [];
  const m5 = cs?.m5 ?? null;
  const validation = cs?.validation ?? null;
  const example = cs?.example_case ?? null;
  // The sheet's example case may not carry answers (a run that never got a
  // validated repair records only its prompt); the case list always does.
  const exampleCase = example ? data.cases.find((c) => c.id === example.case_id) ?? null : null;
  const shown = (value: unknown) => { const text = String(value ?? "").trim(); return text ? text : null; };
  const exampleModel = shown(example?.baseline_answer) ?? shown(example?.baseline_output) ?? shown(exampleCase?.observed) ?? "—";
  const exampleGold = shown(example?.gold) ?? shown(exampleCase?.expected) ?? "—";
  const repair = data.repairs[0];

  const verdictCount = (status: string) => m4.filter((v) => String(v.status || "").toLowerCase() === status).length;
  const supported = verdictCount("supported"), refuted = verdictCount("refuted");
  const inconclusive = m4.length - supported - refuted;
  const selectedFamilies = (m1?.families ?? []).filter((f) => f.selected);
  const usedProbes = (m1?.families ?? []).flatMap((f) => f.probes.filter((p) => p.used).map((p) => p.phrase));
  const accepted = Boolean(repair?.fixed || (validation && String(validation.verdict || "").toLowerCase().includes("accept")));
  const ladder = m5?.ladder ?? [];
  const tried = m5?.candidates.length ?? 0;
  const best = m5?.candidates.find((c) => c.selected) ?? null;
  const go = (id: string) => () => navigate(`evidence:${id}`);
  const casesLabel = (n: number | null, what: string) => n === null ? what : `${n} ${what}`;

  return <section className="section lf-section">
    <header>
      <div>
        <span className="section-kicker">THE EVALRX PIPELINE</span>
        <h2 className="lf-title">Closing the loop from model evaluation to an auditable repair recipe.</h2>
        <p className="journey-hint">Click any stage to inspect its evidence and the agent events behind it — hover to see what went in and what came out.</p>
      </div>
    </header>

    <div className="lf">
      {/* ── INPUTS ─────────────────────────────────────────────────────── */}
      <aside className="lf-inputs" onClick={() => navigate("cases")} role="button" tabIndex={0}>
        <span className="lf-lane-kicker">INPUTS</span>
        <div className="lf-box">
          <small>TARGET MODEL</small>
          <strong title={data.setting.model}>{data.setting.model}</strong>
          <div className="lf-chips">
            {(["llm", "vlm", "alm"] as const).map((kind) => <span key={kind} className={`lf-chip${modality === kind ? " on" : ""}`}>
              {kind === "llm" ? "LLM" : kind === "vlm" ? "VLM" : "Audio-LM"}</span>)}
          </div>
        </div>
        <div className="lf-plus">+</div>
        <div className="lf-box">
          <small>FROZEN CASE BATCH</small>
          <code className="lf-math">D = {"{"}(x, y, ŷ, z, m){"}"}</code>
          <ul className="lf-splits">
            <li className="on"><b>D<sub>E</sub></b><span>{casesLabel(nExplore, "explore")}</span></li>
            <li className={nHeldout ? "on" : ""}><b>D<sub>H</sub></b><span>{casesLabel(nHeldout, "held-out")}</span></li>
            <li className={nConfirm ? "on" : ""}><b>D<sub>C</sub></b><span>{casesLabel(nConfirm, "confirm")}</span></li>
          </ul>
          <em className="lf-dataset" title={data.setting.dataset}>{data.setting.dataset}</em>
        </div>
        <div className="lf-tip"><div><b>IN</b><span>{data.setting.dataset}</span></div><div><b>OUT</b><span>{data.setting.n_cases} cases, split before anything ran</span></div><small>click to open the case studio</small></div>
      </aside>

      <div className="lf-arrow" aria-hidden="true" />

      {/* ── THE LOOP ───────────────────────────────────────────────────── */}
      <div className="lf-loop">
        <div className="lf-lane lf-lane-explore">
          <div className="lf-lane-head"><span>EXPLORE / DISCOVERY</span><code>D<sub>E</sub></code><em>probe freely · inspect many signals · form hypotheses</em></div>
          <div className="lf-lane-cards">
            <StageCard stage={stage.M1} code="M1" lane="explore" onClick={go("m1")}
              title={modules.M1?.name?.split(" ")[0] === "Suspicious" ? "Probe" : modules.M1?.name || "Probe"}
              subtitle="cluster failures into modes, route analyzers"
              into={[casesLabel(nExplore, "explore cases"), selectedFamilies.length ? `${selectedFamilies.length} analyzer families selected` : "analyzer families"]}
              out={[m1 ? `${m1.n_measured} measurements` : "—", m1?.n_forwarded != null ? `${m1.n_forwarded} signals forwarded to M2` : "—"]}>
              {m1 ? <>
                <ul className="lf-list">
                  {m1.families.map((family) => <li key={family.family} className={family.selected ? "on" : ""}>{family.label.toLowerCase().includes("black") ? "Black-box behavior" : family.label.toLowerCase().includes("white") || family.label.toLowerCase().includes("internal") ? "White-box internals" : family.label.toLowerCase().includes("multi") ? "Multi-modal grounding" : family.label}</li>)}
                  <li className="on">Agent trajectories</li>
                </ul>
                <div className="lf-vitals">
                  <small>VITALS · PROBES RUN</small>
                  {usedProbes.slice(0, 4).map((phrase) => <div key={phrase}><span>{phrase}</span><b>OK</b></div>)}
                  {usedProbes.length > 4 && <em>+{usedProbes.length - 4} more</em>}
                </div>
              </> : <p className="lf-empty">no probe record</p>}
            </StageCard>

            <StageCard stage={stage.M2} code="M2" lane="explore" onClick={go("m2")}
              title="Analyze" subtitle="statistical attribution, multiplicity control"
              into={[m1?.n_forwarded != null ? `${m1.n_forwarded} candidate signals` : "candidate signals", `${phase} pass`]}
              out={stats ? [`${stats.survivors.length} of ${stats.n_in_correction_family} survive ${stats.correction_method || "correction"}`] : ["no statistics"]}>
              {stats ? <>
                <SignalRows stats={stats} />
                <div className="lf-foot">paired e-values, {stats.correction_method === "BH" ? "e-BH" : stats.correction_method || "correction"}: <b>{stats.survivors.length} of {stats.n_in_correction_family}</b> signals survive</div>
              </> : <p className="lf-empty">no statistics</p>}
            </StageCard>

            <StageCard stage={stage.M3} code="M3" lane="explore" onClick={go("m3")}
              title="Hypothesize" subtitle="freeze 1–3 falsifiable hypothesis contracts"
              into={stats ? [`${stats.survivors.length} surviving signals`, "M2 tables + explore figures"] : ["M2 result"]}
              out={[m3.length ? `${m3.length} frozen hypotheses` : "no hypothesis"]}>
              <code className="lf-math">h = ⟨claim, slice, assoc, test, grade⟩</code>
              {m3.length ? <>
                <div className="lf-lock"><Lock size={11} /> FROZEN CONTRACT</div>
                {m3.map((hypothesis) => <Hypothesis key={hypothesis.id} hypothesis={hypothesis} />)}
              </> : <p className="lf-empty">no hypothesis was frozen</p>}
            </StageCard>
          </div>
        </div>

        <div className="lf-lane lf-lane-holdout">
          <div className="lf-lane-head"><code>D<sub>H</sub></code><em>held-out</em></div>
          <div className="lf-lane-cards">
            <StageCard stage={stage.M4} code="M4" lane="holdout" onClick={go("m4")}
              title="Validate" subtitle="re-test & filter the frozen hypotheses"
              into={[`${m3.length} hypotheses`, casesLabel(nHeldout, "cases the model never saw")]}
              out={m4.length ? [`${supported} supported · ${refuted} refuted · ${inconclusive} inconclusive`] : ["not adjudicated"]}>
              <div className="lf-pills">
                <span className={`lf-pill ok${supported ? " lit" : ""}`}>SUPPORTED {supported}</span>
                <span className={`lf-pill bad${refuted ? " lit" : ""}`}>REFUTED {refuted}</span>
                <span className={`lf-pill${inconclusive ? " lit" : ""}`}>INCONCLUSIVE {inconclusive}</span>
              </div>
              {m4.map((verdict) => <Verdict key={verdict.id} verdict={verdict} />)}
              {!m4.length && <p className="lf-empty">no verdict recorded</p>}
            </StageCard>
          </div>
        </div>

        <div className="lf-lane lf-lane-repair">
          <div className="lf-lane-head"><span>REPAIR</span><em>confirm on</em><code>D<sub>C</sub></code></div>
          <div className="lf-lane-cards">
            <StageCard stage={stage.M5} code="M5" lane="repair" onClick={go("m5")}
              title="Repair" subtitle="cheapest repair that fixes the most cases"
              into={[best ? `best lead: ${best.tier} ${best.name}` : "best unverified lead", casesLabel(nConfirm, "confirm pairs")]}
              out={validation ? [`${validation.n_fixed} fixed · ${validation.n_broken} broken (${signed(validation.effect, 3)})`] : [accepted ? "repair accepted" : "no candidate passed the gate"]}>
              {ladder.length ? <div className="lf-ladder">
                {ladder.map((rung) => <div key={rung.tier} className={`lf-rung${rung.status === "accepted" ? " ok" : rung.status === "regressed" ? " bad" : ""}${rung.status === "untouched" ? " off" : ""}`}
                  title={rung.best_candidate ? `${rung.best_candidate} ${signed(rung.best_effect, 3)}` : undefined}>
                  <span className="lf-lv">{rung.tier}</span>
                  <span className="lf-nm">{rung.label}</span>
                  <span className="lf-st">{rung.status === "accepted" ? `accepted ${signed(rung.best_effect, 3)}`
                    : rung.status === "regressed" ? `regressed ${signed(rung.best_effect, 3)}`
                      : rung.status === "untouched" ? (rung.within_cap ? "untouched" : `beyond ${rung.tier_cap}`)
                        : `tried ${rung.n_candidates}`}</span>
                </div>)}
              </div> : <p className="lf-empty">no repair attempted</p>}
              {tried > 0 && <div className="lf-foot">{tried} candidates tried on explore{best ? <>; selected <b>{best.name}</b></> : "; none selected"}</div>}
            </StageCard>
          </div>
        </div>

        <div className="lf-iter"><span>ITERATION · v<sub>0</sub> → v<sub>1</sub></span><em>only the frozen contract crosses from explore to held-out</em></div>
      </div>

      <div className="lf-arrow" aria-hidden="true" />

      {/* ── OUTPUT ─────────────────────────────────────────────────────── */}
      <aside className={`lf-output${accepted ? " ok" : ""}`} onClick={() => navigate("evidence")} role="button" tabIndex={0}>
        <span className="lf-lane-kicker">OUTPUT <Trophy size={13} /></span>
        <h3>Health card</h3>
        <p className="lf-sub">An auditable record, not merely a final score</p>
        <ol className="lf-health">
          <li><b>1</b><span>baseline + failure slice</span><code>{pct(cs?.headline.baseline_accuracy)}{cs?.headline.n_cases ? ` · ${cs.headline.n_cases} cases` : ""}</code></li>
          <li><b>2</b><span>frozen hypothesis contract</span><code>{m3.length ? `${m3.length} hypotheses` : "none"}</code></li>
          <li><b>3</b><span>held-out verdict</span><code>{m4.length ? `${supported}/${m4.length} supported` : "—"}</code></li>
          <li><b>4</b><span>repair + side effects</span><code>{validation ? `${validation.n_fixed} fixed · ${validation.n_broken} broken` : repair ? `${repair.fixed_cases} fixed · ${repair.broken_cases} broken` : "—"}</code></li>
        </ol>
        <div className="lf-gate">
          <small>PROMOTION GATE</small>
          <code>paired Δ &gt; δ<sub>min</sub> &amp; regression ≤ ρ<sub>max</sub></code>
          <div className="lf-version">
            <span>v<sub>t</sub></span>
            <i className={accepted ? "go" : "stop"}>{accepted ? <Check size={14} /> : <X size={14} />}</i>
            <span>v<sub>t+1</sub></span>
          </div>
          <strong className={accepted ? "ok" : "no"}>{accepted ? "Accepted → promote v" : "Not promoted · v"}<sub>{accepted ? "t+1" : "t"}</sub>{accepted ? "" : " kept"}</strong>
        </div>
        <div className="lf-tip">
          {example ? <>
            <div><b>IN</b><span>{example.case_id} · {String(example.question || "").split("\n")[0].slice(0, 160)}</span></div>
            <div><b>OUT</b><span>model: {exampleModel.slice(0, 120)}</span><span>gold: {exampleGold.slice(0, 80)}</span></div>
          </> : <>
            <div><b>IN</b><span>{m3.length} hypotheses, {m4.length} verdicts</span></div>
            <div><b>OUT</b><span>{data.summary.headline}</span></div>
          </>}
          <small>click to open the evidence index</small>
        </div>
      </aside>
    </div>

    {!cs && <p className="lf-note"><Database size={12} /> This run recorded no probe or statistics artifacts, so the stage cards show status only.</p>}
    {cs?.qa_flags?.length ? <ul className="lf-flags">{cs.qa_flags.map((flag) => <li key={flag.code} className={`lf-flag-${flag.level}`}><b>{humanise(flag.code)}</b> {flag.detail}</li>)}</ul> : null}
  </section>;
}
