/**
 * Opening a run that was produced somewhere else.
 *
 * The server can start with no run loaded, so the first thing the page may have
 * to render is a way to hand it one: a zipped run directory, dropped or picked.
 * The same POST backs the topbar switcher, so a reader can move between runs
 * without restarting the server.
 */
import { useRef, useState } from "react";
import { FolderUp, Upload } from "lucide-react";

export type UploadedRun = { label?: string; data: any; layout: any };

async function postRun(file: File): Promise<UploadedRun> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body });
  const text = await res.text();
  let parsed: any = null;
  try { parsed = JSON.parse(text); } catch { /* an error page, not JSON */ }
  if (!res.ok) throw new Error(parsed?.detail || `HTTP ${res.status}`);
  return parsed as UploadedRun;
}

function useRunUpload(onLoaded: (run: UploadedRun) => void) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const take = async (file?: File | null) => {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    try {
      onLoaded(await postRun(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, take };
}

export function RunDrop({ onLoaded }: { onLoaded: (run: UploadedRun) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const { busy, error, take } = useRunUpload(onLoaded);
  const [over, setOver] = useState(false);
  return <div className="load-state">
    <div
      className={`run-drop${over ? " over" : ""}${busy ? " busy" : ""}`}
      onClick={() => !busy && input.current?.click()}
      onDragOver={(event) => { event.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => { event.preventDefault(); setOver(false); take(event.dataTransfer.files?.[0]); }}
    >
      <Upload className={busy ? "pulse" : ""} />
      <h2>Drop a run archive</h2>
      <p>{busy
        ? "Unpacking the archive and composing the report…"
        : "A .zip of a run directory — the folder holding run.json + M1..M5 (or the legacy run_log.jsonl), or the example folder around it."}</p>
      <span className="run-drop-cta">Choose a .zip</span>
      <input ref={input} type="file" accept=".zip,application/zip" hidden
        onChange={(event) => { take(event.target.files?.[0]); event.target.value = ""; }} />
      {error && <small className="run-drop-error">{error}</small>}
    </div>
  </div>;
}

export function RunSwitch({ onLoaded }: { onLoaded: (run: UploadedRun) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const { busy, error, take } = useRunUpload(onLoaded);
  return <button type="button" className="run-switch" disabled={busy} title={error || "Open another zipped run"}
    onClick={() => input.current?.click()}>
    <FolderUp size={13} />{busy ? "opening…" : error ? "upload failed" : "open .zip"}
    <input ref={input} type="file" accept=".zip,application/zip" hidden
      onChange={(event) => { take(event.target.files?.[0]); event.target.value = ""; }} />
  </button>;
}
