import { describe, expect, it } from "vitest";
import { parseRoute, routeHash, runFromHash, withRun, type Route } from "./route";

describe("routes", () => {
  const cases: Array<[Route, string]> = [
    [{ view: "overview" }, "#/"],
    [{ view: "evidence" }, "#/evidence"],
    [{ view: "evidence", stage: "m3" }, "#/evidence/m3"],
    [{ view: "evidence", stage: "m3", full: true }, "#/evidence/m3/full"],
    [{ view: "cases" }, "#/cases"],
    [{ view: "cases", caseId: "chartqa-0019" }, "#/cases/chartqa-0019"],
    [{ view: "cases", split: "heldout" }, "#/cases?split=heldout"],
    [{ view: "cases", caseId: "a/b c", split: "" }, "#/cases/a%2Fb%20c?split="],
    [{ view: "debug" }, "#/debug"],
  ];

  it.each(cases)("round-trips %j", (route, hash) => {
    expect(routeHash(route)).toBe(hash);
    expect(parseRoute(hash)).toEqual(route);
  });

  it("opens the overview for an empty or unknown hash", () => {
    for (const hash of ["", "#", "#/", "#/nowhere", "#evidence-old"]) {
      expect(parseRoute(hash)).toEqual({ view: "overview" });
    }
  });
});

describe("the run a hash names", () => {
  it("is read and written beside the route", () => {
    expect(withRun("#/cases/c1?split=E", "r9")).toBe("#/cases/c1?split=E&run=r9");
    expect(withRun("#/debug", "r9")).toBe("#/debug?run=r9");
    expect(withRun("#/debug?run=r1", null)).toBe("#/debug");
    expect(runFromHash("#/cases/c1?split=E&run=r9")).toBe("r9");
    expect(runFromHash("#/evidence/m3")).toBeNull();
  });

  it("does not disturb the route itself", () => {
    expect(parseRoute("#/cases/c1?split=E&run=r9")).toEqual({
      view: "cases",
      caseId: "c1",
      split: "E",
    });
    expect(parseRoute("#/evidence/m3/full?run=r9")).toEqual({
      view: "evidence",
      stage: "m3",
      full: true,
    });
  });
});
