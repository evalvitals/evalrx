import { describe, expect, it } from "vitest";
import { parseRoute, routeHash, type Route } from "./route";

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
