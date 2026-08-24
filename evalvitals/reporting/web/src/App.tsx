import { useEffect, useState } from "react";
import { JSONUIProvider, Renderer } from "@json-render/react";
import { Activity, Bot } from "lucide-react";
import { registry, ReportProviders } from "./reportCatalog";
import type { LayoutEnvelope, ReportData } from "./types";
import { RunDrop, RunSwitch, type UploadedRun } from "./upload";
import { CasesView, DebugView, EvidenceView } from "./views";

type Payload = { data: ReportData; layout: LayoutEnvelope };

declare global {
  interface Window { __EVALVITALS_REPORT__?: Payload }
}

export function App() {
  // A portable export ships its own payload and has no server behind it, so the
  // upload affordances only belong to the served app.
  const embedded = Boolean(window.__EVALVITALS_REPORT__);
  const [payload, setPayload] = useState<Payload | null>(window.__EVALVITALS_REPORT__ || null);
  const [error, setError] = useState("");
  const [needsRun, setNeedsRun] = useState(false);
  const [view, setView] = useState("overview");
  useEffect(() => {
    if (embedded) return;
    fetch("/api/report").then(async (res) => {
      // The server starts empty when `evalvitals serve` was given no run.
      if (res.status === 404) { setNeedsRun(true); return; }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPayload(await res.json());
    }).catch((err) => setError(String(err)));
  }, []);
  const accept = (run: UploadedRun) => {
    setPayload({ data: run.data, layout: run.layout });
    setNeedsRun(false);
    setError("");
    setView("overview");
  };
  if (needsRun) return <RunDrop onLoaded={accept} />;
  if (error) return <div className="load-state"><Activity /><h1>Could not load this report</h1><p>{error}</p></div>;
  if (!payload) return <div className="load-state"><Activity className="pulse" /><p>Composing the failure-to-fix story…</p></div>;
  const back = () => setView("overview");
  if (view === "evidence" || view.startsWith("evidence:")) return <EvidenceView data={payload.data} back={back} navigate={setView} initialStage={view.split(":")[1]} />;
  // "cases:<id>" opens the studio focused on one case, which is how M4's
  // repaired/broken chips link into it.
  if (view === "cases" || view.startsWith("cases:"))
    return <CasesView data={payload.data} back={back} initialCaseId={view.split(":")[1]} />;
  if (view === "debug") return <DebugView data={payload.data} back={back} />;
  return <>
    <div className="topbar"><a href="#"><span className="logo-mark">EV</span><strong>EvalVitals</strong></a>{!embedded && <RunSwitch onLoaded={accept} />}{payload.data.setting.diagnosed_by && <div className="run-agent" title="The agent that drove this run"><Bot size={13} />{payload.data.setting.diagnosed_by}</div>}<div><span className="live-dot" /> report complete</div><code>{payload.data.trace_id.slice(0, 8)}</code></div>
    <ReportProviders data={payload.data} navigate={setView}><JSONUIProvider registry={registry} initialState={{}}><Renderer spec={payload.layout.spec} registry={registry} /></JSONUIProvider></ReportProviders>
    <footer><span>EvalVitals</span><p>Evidence is progressively disclosed from overview to raw audit logs.</p><small>Layout: {payload.layout.generated_by.mode}</small></footer>
  </>;
}
