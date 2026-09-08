/**
 * Browsing experiments launched earlier, without a .zip in hand.
 *
 * The server scans the run's neighborhood for anything else holding a
 * run_log.jsonl (see `discover_runs` in server.py) and this panel lists what
 * it found. Picking one asks the server to open it by id — the panel never
 * sends a filesystem path back, only an id the server itself handed out.
 */
import { useCallback, useEffect, useState } from "react";
import { FolderClock, RefreshCw, X } from "lucide-react";
import type { UploadedRun } from "./upload";

export type RunListItem = {
  id: string;
  /** Relative to the listing's `root` — the display label. Two sibling
   *  experiments both nested under an `outputs/logs` child would otherwise
   *  read as identical rows; this is what tells them apart. */
  path: string;
  root: string;
  dataset: string | null;
  model: string | null;
  published: boolean;
  modified_at: string | null;
};

async function fetchRuns(): Promise<{ root: string; items: RunListItem[]; truncated: boolean }> {
  const res = await fetch("/api/runs");
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail || `HTTP ${res.status}`);
  return res.json();
}

async function openRun(id: string): Promise<UploadedRun> {
  const res = await fetch(`/api/runs/${encodeURIComponent(id)}/open`, { method: "POST" });
  const text = await res.text();
  let parsed: any = null;
  try { parsed = JSON.parse(text); } catch { /* an error page, not JSON */ }
  if (!res.ok) throw new Error(parsed?.detail || `HTTP ${res.status}`);
  return parsed as UploadedRun;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "";
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function RunsPanel({
  open, onClose, onOpened, activeRoot,
}: {
  open: boolean;
  onClose: () => void;
  onOpened: (run: UploadedRun, root: string) => void;
  activeRoot: string | null;
}) {
  const [items, setItems] = useState<RunListItem[] | null>(null);
  const [root, setRoot] = useState("");
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError("");
    fetchRuns()
      .then((res) => { setItems(res.items); setRoot(res.root); setTruncated(res.truncated); })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  // Rescan every time the panel opens rather than once at mount, so a run
  // launched after the server started still shows up.
  useEffect(() => { if (open) refresh(); }, [open, refresh]);

  const pick = async (item: RunListItem) => {
    if (busyId) return;
    setBusyId(item.id);
    setError("");
    try {
      onOpened(await openRun(item.id), item.root);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  };

  return <>
    {/* Closes on an outside click; Escape is handled by the panel itself. */}
    {open && <div className="runs-backdrop" onClick={onClose} />}
    <aside className={`runs-panel${open ? " open" : ""}`} aria-hidden={!open}
      onKeyDown={(event) => { if (event.key === "Escape") onClose(); }}>
      <header>
        <FolderClock size={15} /><h3>Experiments</h3>
        <button type="button" className="runs-icon-btn" onClick={refresh} disabled={loading} title="Rescan for runs">
          <RefreshCw size={13} className={loading ? "spin" : ""} />
        </button>
        <button type="button" className="runs-icon-btn" onClick={onClose} title="Close">
          <X size={15} />
        </button>
      </header>
      {root && <p className="runs-root" title={root}>{root}{truncated && ` — showing the ${items?.length ?? 0} most recent`}</p>}
      {error && <p className="runs-error">{error}</p>}
      {!error && !loading && items?.length === 0 && (
        <p className="runs-empty">No experiments found near this run yet.</p>
      )}
      <ul className="runs-list">
        {(items || []).map((item) => (
          <li key={item.id}>
            <button
              type="button"
              className={`runs-item${item.root === activeRoot ? " active" : ""}${busyId === item.id ? " busy" : ""}`}
              disabled={busyId !== null}
              onClick={() => pick(item)}
              title={item.root}
            >
              <span className="runs-item-label">{item.path}</span>
              <span className="runs-item-meta">
                {item.model && <span>{item.model}</span>}
                {item.dataset && <span>{item.dataset}</span>}
                {!item.published && <em>compiles on open</em>}
              </span>
              <span className="runs-item-time">{relativeTime(item.modified_at)}</span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  </>;
}
