import { useEffect, useState } from "react";
import { JSONUIProvider, Renderer } from "@json-render/react";
import { Activity, Bot, Columns2, FolderClock, Moon, Sun } from "lucide-react";
import { applyTheme, initialTheme, type Theme } from "./theme";
import { registry, ReportProviders } from "./reportCatalog";
import type { LayoutEnvelope, ReportData } from "./types";
import { Lightbox } from "./lightbox";
import { RunDrop, RunSwitch, STATIC_MODE, type UploadedRun } from "./upload";
import { StaticLanding } from "./landing";
import { RunsPanel } from "./runsPanel";
import { CasesView } from "./casesView";
import { DebugView } from "./debugView";
import { EvidenceView } from "./evidenceView";
import { CompareView } from "./compareView";
import { navigate, parseRoute, runFromHash, withRun, type Route } from "./route";
import { api, RunIdContext } from "./api";
import { runStatus } from "./runStatus";

type Payload = { data: ReportData; layout: LayoutEnvelope };

declare global {
  interface Window {
    __EVALRX_REPORT__?: Payload;
  }
}

export function App() {
  // A portable export ships its own payload and has no server behind it, so the
  // upload affordances only belong to the served app.
  const embedded = Boolean(window.__EVALRX_REPORT__);
  const [payload, setPayload] = useState<Payload | null>(window.__EVALRX_REPORT__ || null);
  const [error, setError] = useState("");
  const [needsRun, setNeedsRun] = useState(false);
  // The view on screen is the URL hash, so a refresh or a shared link lands on it.
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  // Charts read their colors through theme.tc at render time, so a switch
  // remounts the view tree (key={theme}) rather than restyling in place.
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [runsOpen, setRunsOpen] = useState(false);
  // The resolved root of whatever is on screen, so the runs panel can
  // highlight it against its own listing (both key entries by this same
  // path). Null for an uploaded/embedded run: it isn't one of the scanned
  // experiments, so nothing in the list should read as "this one".
  const [activeRoot, setActiveRoot] = useState<string | null>(null);
  // The server's id for the run on screen: every report request names it.
  // Empty when no server holds the run (an export, or an archive read in the
  // static build).
  const [runId, setRunId] = useState("");
  // The server's name for that run (its folder), to tell two runs of one
  // model on one dataset apart.
  const [runLabel, setRunLabel] = useState("");
  useEffect(() => applyTheme(theme), [theme]);
  // A run still being written is followed: the server streams an event each
  // time its log grows, and the report on screen is swapped for the new one
  // in place, on whatever view the reader is on.
  const following = Boolean(runId && payload && payload.data.setting.finished === false);
  useEffect(() => {
    if (!following) return;
    const source = new EventSource(api.live(runId));
    const refresh = () =>
      fetch(api.report(runId))
        .then((res) => (res.ok ? res.json() : null))
        .then((run) => run && setPayload({ data: run.data, layout: run.layout }))
        .catch(() => undefined);
    source.addEventListener("update", refresh);
    source.addEventListener("done", () => {
      source.close();
      refresh();
    });
    return () => source.close();
  }, [following, runId]);
  useEffect(() => {
    const follow = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);
  useEffect(() => {
    if (embedded) return;
    // The static build has no server: it opens on the landing page and reads archives itself.
    if (STATIC_MODE) {
      setNeedsRun(true);
      return;
    }
    const open = async () => {
      const session = await fetch(api.session).then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      });
      // A reload or a shared link names the run that was on screen; the
      // server's start-up run is the fallback — and the only choice after a
      // restart, since a server knows only the run ids it handed out itself.
      const named = runFromHash(window.location.hash);
      for (const id of [named, session.run?.id]) {
        if (!id) continue;
        const res = await fetch(api.report(id));
        if (res.status === 404) continue;
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const run = await res.json();
        setPayload({ data: run.data, layout: run.layout });
        setRunId(id);
        setRunLabel(run.label ?? "");
        setActiveRoot(run.root ?? null);
        if (id !== named) {
          // Another run than the hash named: its view of that run means nothing here.
          const here = named
            ? "#/"
            : window.location.hash.startsWith("#/")
              ? window.location.hash
              : "#/";
          window.location.replace(withRun(here, id));
        }
        return;
      }
      // The server starts empty when `evalrx serve` was given no run.
      setNeedsRun(true);
    };
    open().catch((err) => setError(String(err)));
  }, [embedded]);
  const load = (run: UploadedRun, root: string | null = null) => {
    setPayload({ data: run.data, layout: run.layout });
    setRunId(run.id || "");
    setRunLabel(run.label || "");
    setNeedsRun(false);
    setError("");
    navigate({ view: "overview" }, { run: run.id ?? null });
    setRunsOpen(false);
    setActiveRoot(root);
  };
  const body = needsRun ? (
    STATIC_MODE ? (
      <StaticLanding onLoaded={load} />
    ) : (
      <RunDrop onLoaded={load} />
    )
  ) : error ? (
    <div className="load-state">
      <Activity />
      <h1>Could not load this report</h1>
      <p>{error}</p>
    </div>
  ) : !payload ? (
    <div className="load-state">
      <Activity className="pulse" />
      <p>Composing the failure-to-fix story…</p>
    </div>
  ) : (
    // Which view the reader is on is chosen here rather than returned from
    // here, so the one figure overlay can sit above all of them — a chart
    // opens the same way whether it was found on the overview, a stage
    // brief or the full record, and the detail views all return past the
    // topbar below.
    renderReport(payload, route, embedded, load, { id: runId, label: runLabel })
  );
  return (
    <>
      <RunIdContext value={runId}>
        <div key={theme}>{body}</div>
      </RunIdContext>
      {!embedded && !STATIC_MODE && (
        <>
          <button
            type="button"
            className="runs-toggle"
            onClick={() => setRunsOpen((prev) => !prev)}
            title="Browse experiments launched before"
            aria-label="Browse experiments launched before"
          >
            <FolderClock size={15} />
          </button>
          <RunsPanel
            open={runsOpen}
            onClose={() => setRunsOpen(false)}
            onOpened={load}
            activeRoot={activeRoot}
          />
        </>
      )}
      <button
        className="theme-toggle"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      >
        {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
      </button>
      <Lightbox />
    </>
  );
}

function renderReport(
  payload: Payload,
  route: Route,
  embedded: boolean,
  load: (run: UploadedRun, root?: string | null) => void,
  run: { id: string; label: string },
) {
  const runId = run.id;
  const back = () => navigate({ view: "overview" });
  if (route.view === "compare" && runId) {
    return (
      <CompareView
        data={payload.data}
        run={run}
        otherId={route.with}
        back={back}
        navigate={navigate}
        open={load}
      />
    );
  }
  if (route.view === "evidence") {
    return <EvidenceView data={payload.data} route={route} back={back} navigate={navigate} />;
  }
  // A case id opens the studio focused on one case, which is how M5's
  // repaired/broken chips link into it; a split opens it filtered to one band
  // of the batch cylinder.
  if (route.view === "cases") {
    return <CasesView data={payload.data} route={route} back={back} navigate={navigate} />;
  }
  if (route.view === "debug") return <DebugView data={payload.data} back={back} />;
  return (
    <>
      <div className="topbar">
        <a href={withRun("#/", runFromHash(window.location.hash))}>
          <span className="logo-mark">EV</span>
          <strong>EvalRX</strong>
        </a>
        {!embedded && <RunSwitch onLoaded={load} />}
        {runId && (
          <a
            className="compare-toggle"
            href={withRun("#/compare", runId)}
            title="Compare this run with another"
          >
            <Columns2 size={13} /> Compare
          </a>
        )}
        {payload.data.setting.diagnosed_by && (
          <div className="run-agent" title="The agent that drove this run">
            <Bot size={13} />
            {payload.data.setting.diagnosed_by}
          </div>
        )}
        <RunStatusBadge setting={payload.data.setting} />
        <code>{payload.data.trace_id.slice(0, 8)}</code>
      </div>
      <ReportProviders data={payload.data} navigate={navigate}>
        <JSONUIProvider registry={registry} initialState={{}}>
          <Renderer spec={payload.layout.spec} registry={registry} />
        </JSONUIProvider>
      </ReportProviders>
      <footer>
        <span>EvalRX</span>
        <p>Evidence is progressively disclosed from overview to raw audit logs.</p>
        <small>Layout: {payload.layout.generated_by.mode}</small>
      </footer>
    </>
  );
}

/** The run's state in the top bar, re-read each minute so "quiet" can arrive. */
function RunStatusBadge({ setting }: { setting: ReportData["setting"] }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (setting.finished !== false) return;
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [setting.finished]);
  const status = runStatus(setting, now);
  return (
    <div className={`run-status run-status-${status.state}`}>
      <span className={`live-dot${status.state === "live" ? " pulse" : ""}`} /> {status.label}
    </div>
  );
}
