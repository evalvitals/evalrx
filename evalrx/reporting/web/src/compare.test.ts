import { describe, expect, it } from "vitest";
import { compareRuns } from "./compare";
import type { ReportData } from "./types";

function run(overrides: {
  dataset?: string;
  kind?: "loop" | "explore";
  metrics?: Array<[string, number]>;
  stages?: Array<[string, string]>;
  cases?: Array<[string, string]>;
}): ReportData {
  return {
    setting: { dataset: overrides.dataset ?? "chartqa", kind: overrides.kind ?? "loop" },
    metrics: (overrides.metrics ?? []).map(([id, value]) => ({ id, label: id, value })),
    stages: (overrides.stages ?? []).map(([id, status]) => ({ id, title: id, status })),
    cases: (overrides.cases ?? []).map(([id, status]) => ({ id, status })),
  } as unknown as ReportData;
}

describe("comparing two runs", () => {
  it("joins cases by id and lists the ones whose outcome changed", () => {
    const a = run({
      cases: [
        ["c1", "fail"],
        ["c2", "pass"],
        ["c3", "fail"],
        ["only-a", "fail"],
      ],
    });
    const b = run({
      cases: [
        ["c1", "pass"],
        ["c2", "fail"],
        ["c3", "fail"],
        ["only-b", "pass"],
      ],
    });
    expect(compareRuns(a, b).cases).toEqual({
      shared: 3,
      onlyA: 1,
      onlyB: 1,
      nowPass: ["c1"],
      nowFail: ["c2"],
    });
  });

  it("does not read a status other than pass or fail as a change", () => {
    const a = run({ cases: [["c1", "unchanged"]] });
    const b = run({ cases: [["c1", "pass"]] });
    expect(compareRuns(a, b).cases).toMatchObject({ shared: 1, nowPass: [], nowFail: [] });
  });

  it("joins metrics by id and keeps the ones only one run has", () => {
    const a = run({
      metrics: [
        ["evaluated", 32],
        ["fixed", 3],
      ],
    });
    const b = run({
      kind: "explore",
      metrics: [
        ["evaluated", 30],
        ["takeaways", 7],
      ],
    });
    const result = compareRuns(a, b);
    expect(result.metrics.map((m) => [m.id, m.a, m.b])).toEqual([
      ["evaluated", 32, 30],
      ["fixed", 3, null],
      ["takeaways", null, 7],
    ]);
    expect(result.sameKind).toBe(false);
    expect(result.sameDataset).toBe(true);
  });

  it("lines up each stage's status", () => {
    const a = run({
      stages: [
        ["m1", "completed"],
        ["m5", "improved"],
      ],
    });
    const b = run({ dataset: "other", stages: [["m1", "completed"]] });
    const result = compareRuns(a, b);
    expect(result.stages.map((s) => [s.id, s.a, s.b])).toEqual([
      ["m1", "completed", "completed"],
      ["m5", "improved", null],
    ]);
    expect(result.sameDataset).toBe(false);
  });
});
