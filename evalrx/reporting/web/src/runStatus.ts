/**
 * What the top bar says about a run: complete, still being written, or
 * unfinished and quiet. Read off the report alone, so an export says the same.
 *
 * A run that stops logging is not known to have crashed (a long M5 step can
 * be quiet for minutes), so a quiet run is described, not judged.
 */
import type { ReportData } from "./types";

/** How long a run may go without logging before it reads as quiet. */
export const QUIET_AFTER_MS = 5 * 60_000;

export type RunStatus = { state: "complete" | "live" | "quiet"; label: string };

export function runStatus(setting: ReportData["setting"], now: number): RunStatus {
  if (setting.finished !== false) return { state: "complete", label: "report complete" };
  const last = setting.last_event_at ? Date.parse(setting.last_event_at) : NaN;
  if (Number.isNaN(last) || now - last < QUIET_AFTER_MS) {
    return { state: "live", label: "run in progress" };
  }
  return { state: "quiet", label: `unfinished · no new events for ${age(now - last)}` };
}

function age(ms: number): string {
  const minutes = Math.round(ms / 60_000);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(minutes / 60);
  return hours < 48 ? `${hours}h` : `${Math.round(hours / 24)}d`;
}
