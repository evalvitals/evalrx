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
import { Check, Database, Lock, Search, Trophy, X } from "lucide-react";
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
    {/* The IN/OUT tip belongs to the stage, so it answers to the stage's own
        chrome only; the body holds buttons with their own hover and popovers. */}
    <div className="lf-head">
      <header>
        <span className="lf-code">{code}</span>
        <span className={`lf-status lf-status-${status}`}>{STATUS_WORD[status] ?? status}</span>
      </header>
      <h3>{title}</h3>
      <p className="lf-sub">{subtitle}</p>
    </div>
    <div className="lf-body" onClick={(event) => event.stopPropagation()}>{children}</div>
    <IO into={into} out={out} />
  </article>;
}

/**
 * The frozen case batch as a stacked cylinder: D_E on top, D_H, D_C at the
 * bottom, each band as tall as its share of the batch. A partition whose size
 * is not yet known (confirm pairs before a repair was validated) keeps a
 * minimum band, dashed, rather than being drawn as empty.
 */
function BatchCylinder({ explore, heldout, confirm, caption, onBand }: { explore: number | null; heldout: number | null; confirm: number | null; caption?: string; onBand?: (key: "E" | "H" | "C") => void }) {
  const bands = [
    { key: "E", label: "Explore", n: explore, icon: <Search size={12} />, cls: "explore" },
    { key: "H", label: "Held-out", n: heldout, icon: <Lock size={12} />, cls: "heldout" },
    { key: "C", label: "Confirm", n: confirm, icon: <Check size={12} />, cls: "confirm" },
  ];
  const known = bands.map((b) => b.n ?? 0);
  const total = Math.max(1, known.reduce((a, b) => a + b, 0));
  const width = 74, rx = 33, ry = 6, cx = width / 2, left = cx - rx, right = cx + rx;
  const usable = 92, minBand = 18;
  const heights = bands.map((b) => Math.max(minBand, ((b.n ?? 0) / total) * usable));
  let y = ry + 2;
  const drawn = bands.map((band, i) => { const top = y; y += heights[i]; return { ...band, top, bottom: y }; });
  const height = y + ry + 2;
  return <div className="lf-cyl">
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden="true">
      {drawn.map((band) => <g key={band.key} className={`lf-band lf-band-${band.cls}${band.n === null ? " unknown" : ""}${onBand ? " clickable" : ""}`}
        onClick={onBand ? (event) => { event.stopPropagation(); onBand(band.key as "E" | "H" | "C"); } : undefined}>
        {onBand && <title>{`open the ${band.label.toLowerCase()} cases`}</title>}
        <path d={`M${left} ${band.top} A${rx} ${ry} 0 0 0 ${right} ${band.top} L${right} ${band.bottom} A${rx} ${ry} 0 0 1 ${left} ${band.bottom} Z`} />
        <text x={cx} y={(band.top + band.bottom) / 2 + ry / 2 + 3} textAnchor="middle">D<tspan baselineShift="sub" fontSize="7">{band.key}</tspan></text>
      </g>)}
      <ellipse className="lf-cap" cx={cx} cy={drawn[0].top} rx={rx} ry={ry} />
    </svg>
    {caption && <em className="lf-dataset" title={caption}>{caption}</em>}
    <ul>
      {drawn.map((band) => <li key={band.key} className={`lf-band-${band.cls}${band.n === null ? " unknown" : ""}${onBand ? " clickable" : ""}`}
        onClick={onBand ? (event) => { event.stopPropagation(); onBand(band.key as "E" | "H" | "C"); } : undefined}>
        <i>{band.icon}</i><span>{band.label}</span><b>{band.n === null ? "—" : band.n}</b>
      </li>)}
    </ul>
  </div>;
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
    <div className="lf-sighead"><span>effect ± CI</span><span title="survived multiplicity correction">BH</span></div>
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
  const partitions = data.setting.partitions ?? [];
  const nExplore = data.setting.n_cases || cs?.headline.n_explore || nTotal;
  const nWithheld = nTotal > nExplore ? nTotal - nExplore : (cs?.headline.n_heldout ?? null);
  const nConfirm = cs?.validation?.n_pairs ?? null;
  // The withheld cases split into the held-out pool M4 adjudicates on and the
  // confirm pairs the repair is scored on; the latter is known only after a
  // repair was validated, so until then the whole remainder is "held-out".
  const nHeldout = nWithheld === null ? null : nConfirm !== null && nConfirm < nWithheld ? nWithheld - nConfirm : nWithheld;
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
  const repair = Array.isArray(data.repairs) ? data.repairs[0] : undefined;
  // The FixAgent's own verdict on the run: what to do next and why. This is
  // the sentence that makes the gate auditable ("only 4 failing cases…").
  const m5Detail = (data.stage_detail?.m5 ?? {}) as { recommendation?: { action?: string | null; reason?: string | null } | null };
  const recommendation = m5Detail.recommendation && (m5Detail.recommendation.action || m5Detail.recommendation.reason) ? m5Detail.recommendation : null;

  const verdictCount = (status: string) => m4.filter((v) => String(v.status || "").toLowerCase() === status).length;
  const supported = verdictCount("supported"), refuted = verdictCount("refuted");
  const inconclusive = m4.length - supported - refuted;
  const selectedFamilies = (m1?.families ?? []).filter((f) => f.selected);
  // Only the probes this run actually ran; `confirmed` marks the one whose
  // signal survived M2's correction — that is a flag, "ran" is not a verdict.
  const usedProbes = (m1?.families ?? []).flatMap((f) => f.probes.filter((p) => p.used));
  const flagged = usedProbes.filter((p) => p.confirmed).length;
  // Per-analyzer question + headline numbers, compiled by extract_run_data
  // (stage_detail.m1.probes) — the "what did it find" behind each phrase.
  type ProbeDetail = { raw_name?: string; question?: string; description?: string; n_cases?: number | null; metrics?: Array<{ label: string; value: string | number | null }> };
  const probeDetail: Record<string, ProbeDetail> = Object.fromEntries(
    (((data.stage_detail?.m1 ?? {}) as { probes?: ProbeDetail[] }).probes ?? []).map((item) => [String(item.raw_name || ""), item]));
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
          <BatchCylinder explore={nExplore} heldout={nHeldout} confirm={nConfirm} caption={data.setting.dataset}
            onBand={partitions.length ? (key) => {
              // A band opens the studio on its own partition. The cylinder's
              // three letters map onto whatever the run actually recorded: on
              // a two-way run H and C are one pool, coded "H/C".
              const hit = partitions.find((p) => p.code.split("/").includes(key));
              navigate(hit ? `cases:split=${hit.split}` : "cases");
            } : undefined} />
        </div>
        <div className="lf-tip"><div><b>IN</b><span>{data.setting.dataset}</span></div><div><b>OUT</b><span>{data.setting.n_cases} cases, split before anything ran</span></div><small>click to open the case studio</small></div>
      </aside>

      <div className="lf-arrow" aria-hidden="true" />

      {/* ── THE LOOP ───────────────────────────────────────────────────── */}
      <div className="lf-loop">
        <div className="lf-lane lf-lane-explore">
          <div className="lf-lane-head"><span>EXPLORE / DISCOVERY</span><code>D<sub>E</sub></code></div>
          <div className="lf-lane-cards">
            <StageCard stage={stage.M1} code="M1" lane="explore" onClick={go("m1")}
              title={modules.M1?.name?.split(" ")[0] === "Suspicious" ? "Probe" : modules.M1?.name || "Probe"}
              subtitle="cluster failures into modes, route analyzers"
              into={[casesLabel(nExplore, "explore cases"), selectedFamilies.length ? `${selectedFamilies.length} analyzer families selected` : "analyzer families"]}
              out={[m1 ? `${m1.n_measured} measurements` : "—", m1?.n_forwarded != null ? `${m1.n_forwarded} signals forwarded to M2` : "—"]}>
              {m1 ? <>
                <ul className="lf-list">
                  {m1.families.map((family) => <li key={family.family} className={family.selected ? "on" : ""}
                    title={family.selected ? family.analyzers.join(", ") : "not used on this run"}>
                    {family.label.toLowerCase().includes("black") ? "Black-box behavior" : family.label.toLowerCase().includes("white") || family.label.toLowerCase().includes("internal") ? "White-box internals" : family.label.toLowerCase().includes("multi") ? "Multi-modal grounding" : family.label}
                    {!family.selected && <em> · not used</em>}
                  </li>)}
                </ul>
                {/* Flagged probes stay visible — they are findings; the full
                    list of what ran is one hover away. */}
                <div className="lf-vitals" tabIndex={0}>
                  <small>PROBES RUN · {usedProbes.length}{flagged ? ` · ${flagged} FLAG` : ""}{usedProbes.length > flagged ? <em>hover for the list</em> : null}</small>
                  {usedProbes.filter((probe) => probe.confirmed).map((probe) => <div key={probe.phrase} title={probe.analyzers.join(", ")}>
                    <span>{probe.phrase}</span><b>FLAG</b>
                  </div>)}
                  {!usedProbes.length && <em>none recorded</em>}
                  {usedProbes.length > 0 && <div className="lf-vitals-pop">
                    <small>PROBES RUN · {usedProbes.length}</small>
                    {usedProbes.map((probe) => <div key={probe.phrase} className={probe.confirmed ? "flag" : ""}>
                      <span>{probe.phrase}</span>
                      {probe.confirmed ? <b>FLAG</b> : <b className="ran">ran</b>}
                      {/* What each analyzer behind the phrase asks, and what it measured. */}
                      {probe.analyzers.map((analyzer) => {
                        const detail = probeDetail[analyzer];
                        const metrics = (detail?.metrics ?? []).filter((m) => m.value !== null && m.value !== undefined);
                        return <div className="lf-probe-detail" key={analyzer}>
                          <i>{analyzer}{detail?.n_cases ? ` · ${detail.n_cases} cases` : ""}</i>
                          {detail?.question && !detail.question.startsWith("Measures model behavior") && <p>{detail.question}</p>}
                          {metrics.length > 0 && <dl>{metrics.map((m) => <div key={m.label}><dt>{m.label}</dt><dd>{m.value}</dd></div>)}</dl>}
                        </div>;
                      })}
                    </div>)}
                  </div>}
                </div>
                {m1.n_measured > 0 ? <div className="lf-meter" title={`${m1.n_measured} measurements, ${m1.n_forwarded ?? "—"} entered M2's correction family`}>
                  <div className="lf-meter-bar"><b style={{ width: `${Math.min(100, ((m1.n_forwarded ?? 0) / m1.n_measured) * 100)}%` }} /></div>
                  <small>{m1.n_measured} measurements → <em>{m1.n_forwarded ?? "—"}</em> forwarded to M2</small>
                </div> : <div className="lf-meter"><small>measurements not recorded{m1.n_forwarded != null ? ` · ${m1.n_forwarded} forwarded to M2` : ""}</small></div>}
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
      {/* The health card is the audit record: each line is the receipt for one
          stage, so each line opens that stage. The card itself is not a target
          — a click that only leads to the evidence index taught readers that
          the card was broken. */}
      <aside className={`lf-output${accepted ? " ok" : ""}`}>
        <span className="lf-lane-kicker">OUTPUT <Trophy size={13} /></span>
        <h3>Health card</h3>
        <p className="lf-sub">An auditable record, not merely a final score</p>
        <ol className="lf-health">
          <li><button type="button" onClick={() => navigate("cases")}
            title={example ? `${example.case_id} · ${String(example.question || "").split("\n")[0].slice(0, 140)}\nmodel: ${exampleModel.slice(0, 100)}\ngold: ${exampleGold.slice(0, 60)}` : "open the case studio"}>
            <b>1</b><span>baseline + failure slice</span><code>{pct(cs?.headline.baseline_accuracy)}{cs?.headline.n_cases ? ` · ${cs.headline.n_cases} cases` : ""}</code></button></li>
          <li><button type="button" onClick={go("m3")} title="open M3: the frozen hypotheses">
            <b>2</b><span>frozen hypothesis contract</span><code>{m3.length ? `${m3.length} hypotheses` : "none"}</code></button></li>
          <li><button type="button" onClick={go("m4")} title="open M4: the held-out verdicts">
            <b>3</b><span>held-out verdict</span><code>{m4.length ? `${supported}/${m4.length} supported` : "—"}</code></button></li>
          <li><button type="button" onClick={go("m5")} title="open M5: the repair and its side effects">
            <b>4</b><span>repair + side effects</span><code>{validation ? `${validation.n_fixed} fixed · ${validation.n_broken} broken` : repair ? `${repair.fixed_cases} fixed · ${repair.broken_cases} broken` : "—"}</code></button></li>
        </ol>
        <div className="lf-gate">
          <small>PROMOTION GATE</small>
          <code>paired Δ &gt; δ<sub>min</sub> &amp; regression ≤ ρ<sub>max</sub></code>
          {validation && <dl className="lf-gate-facts">
            <dt>Δ</dt><dd>{signed(validation.effect, 3)} on {validation.n_pairs} pairs{typeof validation.e_value === "number" ? ` · e = ${validation.e_value.toFixed(1)}` : ""}</dd>
            <dt>side</dt><dd>{validation.n_fixed} fixed · {validation.n_broken} broken</dd>
          </dl>}
          <div className="lf-version">
            <span>v<sub>t</sub></span>
            <i className={accepted ? "go" : "stop"}>{accepted ? <Check size={14} /> : <X size={14} />}</i>
            <span>v<sub>t+1</sub></span>
          </div>
          <strong className={accepted ? "ok" : "no"}>{accepted ? "Accepted → promote v" : "Not promoted · v"}<sub>{accepted ? "t+1" : "t"}</sub>{accepted ? "" : " kept"}</strong>
          {recommendation && <p className="lf-why">
            {recommendation.action && <b>{humanise(recommendation.action)}.</b>} {recommendation.reason}
          </p>}
        </div>
      </aside>
    </div>

    {!cs && <p className="lf-note"><Database size={12} /> This run recorded no probe or statistics artifacts, so the stage cards show status only.</p>}
    {cs?.qa_flags?.length ? <ul className="lf-flags">{cs.qa_flags.map((flag) => <li key={flag.code} className={`lf-flag-${flag.level}`}><b>{humanise(flag.code)}</b> {flag.detail}</li>)}</ul> : null}
  </section>;
}
