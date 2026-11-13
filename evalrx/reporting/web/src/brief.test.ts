import { isValidElement, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import fixture from "../fixtures/chartqa-gemini.json";
import { buildBrief, type Brief } from "./brief";
import type { ReportData } from "./types";

// Chart options resolve theme colours from the stylesheet (theme.tc). There is
// no stylesheet here, so every colour falls back to its dark-theme literal.
vi.stubGlobal("document", { documentElement: {} });
vi.stubGlobal("getComputedStyle", () => ({ getPropertyValue: () => "" }));

const real = fixture.data as unknown as ReportData;
const STAGES = ["m1", "m2", "m3", "m4", "m5"];

/** A brief with its rendered parts (charts, figures) reduced to a marker. */
function plain(brief: Brief | null) {
  if (!brief) return brief;
  const node = (value: ReactNode) => (isValidElement(value) ? "<element>" : value);
  return {
    ...brief,
    kpis: brief.kpis.map((kpi) => ({ ...kpi, value: node(kpi.value) })),
    figure: brief.figure === undefined ? undefined : node(brief.figure),
  };
}

describe("buildBrief on a real run", () => {
  it.each(STAGES)("summarises %s", (stage) => {
    const brief = buildBrief(stage, real);
    expect(brief?.verdict).toBeTruthy();
    expect(plain(brief)).toMatchSnapshot();
  });
});

describe("buildBrief on a sparse run", () => {
  const empty = {
    trace_id: "t",
    metrics: [],
    charts: [],
    stages: STAGES.map((id) => ({ id, code: id, title: id, status: "completed" })),
  } as unknown as ReportData;

  it.each(STAGES)("degrades to a sentence for %s instead of throwing", (stage) => {
    const brief = buildBrief(stage, empty);
    expect(typeof brief?.verdict).toBe("string");
    expect(brief?.verdict.length).toBeGreaterThan(0);
  });

  it("says a skipped stage did not run", () => {
    const skipped = { ...empty, stages: [{ id: "m3", code: "m3", title: "m3", status: "skipped" }] } as ReportData;
    expect(buildBrief("m3", skipped)?.verdict).toBe("This step did not run.");
  });

  it("has no brief for an unknown stage", () => {
    expect(buildBrief("m9", empty)).toBeNull();
  });
});
