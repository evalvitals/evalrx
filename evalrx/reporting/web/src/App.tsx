import { useEffect, useState } from "react";
import { JSONUIProvider, Renderer } from "@json-render/react";
import { Activity, Bot, FolderClock, Moon, Sun } from "lucide-react";
import { applyTheme, initialTheme, type Theme } from "./theme";
import { registry, ReportProviders } from "./reportCatalog";
import type { LayoutEnvelope, ReportData } from "./types";
import { Lightbox } from "./lightbox";
import { RunDrop, RunSwitch, type UploadedRun } from "./upload";
import { RunsPanel } from "./runsPanel";
import { CasesView, DebugView, EvidenceView } from "./views";

type Payload = { data: ReportData; layout: LayoutEnvelope };

declare global {
  interface Window { __EVALRX_REPORT__?: Payload }
}

export function App() {
  // A portable export ships its own payload and has no server behind it, so the
  // upload affordances only belong to the served app.
  const embedded = Boolean(window.__EVALRX_REPORT__);
  const [payload, setPayload] = useState<Payload | null>(window.__EVALRX_REPORT__ || null);
  const [error, setError] = useState("");
  const [needsRun, setNeedsRun] = useState(false);
  const [view, setView] = useState("overview");
  // Charts read their colors through theme.tc at render time, so a switch
  // remounts the view tree (key={theme}) rather than restyling in place.
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [runsOpen, setRunsOpen] = useState(false);
  // The resolved root of whatever is on screen, so the runs panel can
  // highlight it against its own listing (both key entries by this same
  // path). Null for an uploaded/embedded run: it isn't one of the scanned
  // experiments, so nothing in the list should read as "this one".
  const [activeRoot, setActiveRoot] = useState<string | null>(null);
  useEffect(() => applyTheme(theme), [theme]);
  useEffect(() => {
    if (embedded) return;
    fetch("/api/report").then(async (res) => {
      // The server starts empty when `evalrx serve` was given no run.
      if (res.status === 404) { setNeedsRun(true); return; }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPayload(await res.json());
    }).catch((err) => setError(String(err)));
    fetch("/api/session").then((res) => res.ok ? res.json() : null)
      .then((session) => { if (session?.root) setActiveRoot(session.root); })
      .catch(() => undefined); // the runs panel just starts with nothing highlighted
  }, []);
  const load = (run: UploadedRun, root: string | null = null) => {
    setPayload({ data: run.data, layout: run.layout });
    setNeedsRun(false);
    setError("");
    setView("overview");
    setRunsOpen(false);
    setActiveRoot(root);
  };
  const body = needsRun
    ? <RunDrop onLoaded={load} />
    : error
      ? <div className="load-state"><Activity /><h1>Could not load this report</h1><p>{error}</p></div>
      : !payload
        ? <div className="load-state"><Activity className="pulse" /><p>Composing the failure-to-fix story…</p></div>
        // Which view the reader is on is chosen here rather than returned from
        // here, so the one figure overlay can sit above all of them — a chart
        // opens the same way whether it was found on the overview, a stage
        // brief or the full record, and the detail views all return past the
        // topbar below.
        : renderReport(payload, view, setView, embedded, load);
  return <>
    <div key={theme}>{body}</div>
    {!embedded && <>
      <button
        type="button"
        className="runs-toggle"
        onClick={() => setRunsOpen((prev) => !prev)}
        title="Browse experiments launched before"
        aria-label="Browse experiments launched before"
      ><FolderClock size={15} /></button>
      <RunsPanel open={runsOpen} onClose={() => setRunsOpen(false)} onOpened={load} activeRoot={activeRoot} />
    </>}
    <button
      className="theme-toggle"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
    >{theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}</button>
    <Lightbox />
  </>;
}

function renderReport(
  payload: Payload, view: string, setView: (view: string) => void, embedded: boolean,
  load: (run: UploadedRun, root?: string | null) => void,
) {
  const back = () => setView("overview");
  if (view === "evidence" || view.startsWith("evidence:")) {
    return <EvidenceView data={payload.data} back={back} navigate={setView} initialStage={view.split(":")[1]} />;
  }
  // "cases:<id>" opens the studio focused on one case, which is how M5's
  // repaired/broken chips link into it; "cases:split=<partition>" opens it
  // filtered to one band of the batch cylinder.
  if (view === "cases" || view.startsWith("cases:")) {
    return <CasesView data={payload.data} back={back}
      initialCaseId={view.split(":")[1]?.startsWith("split=") ? undefined : view.split(":")[1]}
      initialSplit={view.split(":")[1]?.startsWith("split=") ? view.split(":")[1].slice("split=".length) : undefined} />;
  }
  if (view === "debug") return <DebugView data={payload.data} back={back} />;
  return <>
    <div className="topbar"><a href="#"><span className="logo-mark">EV</span><strong>EvalRX</strong></a>{!embedded && <RunSwitch onLoaded={load} />}{payload.data.setting.diagnosed_by && <div className="run-agent" title="The agent that drove this run"><Bot size={13} />{payload.data.setting.diagnosed_by}</div>}<div><span className="live-dot" /> report complete</div><code>{payload.data.trace_id.slice(0, 8)}</code></div>
    <ReportProviders data={payload.data} navigate={setView}><JSONUIProvider registry={registry} initialState={{}}><Renderer spec={payload.layout.spec} registry={registry} /></JSONUIProvider></ReportProviders>
    <footer><span>EvalRX</span><p>Evidence is progressively disclosed from overview to raw audit logs.</p><small>Layout: {payload.layout.generated_by.mode}</small></footer>
  </>;
}
