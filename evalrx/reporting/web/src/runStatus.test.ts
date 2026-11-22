import { describe, expect, it } from "vitest";
import { QUIET_AFTER_MS, runStatus } from "./runStatus";
import type { ReportData } from "./types";

const at = (fields: Partial<ReportData["setting"]>) => fields as ReportData["setting"];
const now = Date.parse("2026-11-13T12:00:00Z");

describe("the run status", () => {
  it("is complete once the run logged its end", () => {
    expect(runStatus(at({ finished: true }), now).state).toBe("complete");
  });

  it("is live while an unfinished run keeps logging", () => {
    const recent = new Date(now - QUIET_AFTER_MS + 1000).toISOString();
    expect(runStatus(at({ finished: false, last_event_at: recent }), now)).toEqual({
      state: "live",
      label: "run in progress",
    });
    expect(runStatus(at({ finished: false, last_event_at: null }), now).state).toBe("live");
  });

  it("describes a quiet unfinished run without calling it crashed", () => {
    const old = new Date(now - 3 * 3600_000).toISOString();
    expect(runStatus(at({ finished: false, last_event_at: old }), now)).toEqual({
      state: "quiet",
      label: "unfinished · no new events for 3h",
    });
  });
});
