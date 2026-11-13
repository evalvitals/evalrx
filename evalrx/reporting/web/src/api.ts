/**
 * The local report server's API (`evalrx serve`, evalrx/reporting/server.py).
 *
 * Every report request names the run it is about — the server keeps no
 * "current run" — so the media and figures on screen are fetched through the
 * id of the run they belong to, which `RunIdContext` carries down the tree.
 */
import { createContext, useContext } from "react";

const part = encodeURIComponent;

export const api = {
  session: "/api/session",
  runs: "/api/runs",
  upload: "/api/upload",
  open: (runId: string) => `/api/runs/${part(runId)}/open`,
  report: (runId: string) => `/api/runs/${part(runId)}/report`,
  live: (runId: string) => `/api/runs/${part(runId)}/live`,
  media: (runId: string, mediaId: string) => `/api/runs/${part(runId)}/media/${part(mediaId)}`,
  artifact: (runId: string, path: string) => `/api/runs/${part(runId)}/artifact?path=${part(path)}`,
};

/** POST to the report server and return its JSON; a failure throws the server's `detail`. */
export async function postJson<T>(url: string, body?: BodyInit): Promise<T> {
  const res = await fetch(url, { method: "POST", body });
  const text = await res.text();
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    /* an error page, not JSON */
  }
  if (!res.ok)
    throw new Error((parsed as { detail?: string } | null)?.detail || `HTTP ${res.status}`);
  return parsed as T;
}

/**
 * The server-side id of the run on screen. Empty for a report with no server
 * behind it (a portable export, or an archive read in the static build): its
 * media is embedded as data URIs, and anything not embedded cannot be fetched.
 */
export const RunIdContext = createContext("");

/** URL of a report media item: inlined when the payload carries it, else from the server. */
export function useMediaSource(media: { id: string; data_uri?: string }): string {
  const runId = useContext(RunIdContext);
  return media.data_uri || api.media(runId, media.id);
}

/** URL of a figure the run saved: inlined on an export, else served from the run. */
export function useFigureSource(figure: { path: string; data_uri?: string | null }): string {
  const runId = useContext(RunIdContext);
  return figure.data_uri || api.artifact(runId, figure.path);
}
