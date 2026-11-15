import { describe, expect, it } from "vitest";
import {
  chartPercent,
  chartValue,
  countOf,
  findContract,
  firstSentence,
  headlineSplit,
  leadFrom,
  metric,
  oddsPhrase,
  oddsValue,
  outcomeColors,
  plural,
  pointsGap,
  restAfterFirstSentence,
  stageCode,
  stageDetail,
  stripMarkdown,
} from "./reportAccess";
import type { ReportData } from "./types";

const report = (fields: Partial<ReportData>) =>
  ({ metrics: [], stages: [], ...fields }) as ReportData;

describe("findContract", () => {
  it("returns the newest cycle's payload for a stage", () => {
    const data = report({
      contract: {
        "c0.m1": { stage: "m1", n: 1 },
        "c1.m1": { stage: "m1", n: 2 },
        "c0.m2": { stage: "m2" },
      } as unknown as ReportData["contract"],
    });
    expect(findContract<{ n: number }>(data, "m1")?.n).toBe(2);
  });

  it("matches a bare span id and skips emission errors", () => {
    const data = report({
      contract: {
        m5_fix: { stage: "m5" },
        "c0.m3": { stage: "m3", error: "boom" },
      } as unknown as ReportData["contract"],
    });
    expect(findContract(data, "m5_fix")).toEqual({ stage: "m5" });
    expect(findContract(data, "m3")).toBeUndefined();
  });

  it("is undefined for a run that predates contract emission", () => {
    expect(findContract(report({}), "m1")).toBeUndefined();
  });
});

describe("stageDetail and metric", () => {
  it("returns {} for a stage the run never reached", () => {
    expect(stageDetail(report({}), "m4")).toEqual({});
    expect(stageDetail(report({ stage_detail: { m4: null } } as never), "m4")).toEqual({});
    expect(stageDetail(report({ stage_detail: { m1: { n_probes: 3 } } } as never), "m1")).toEqual({
      n_probes: 3,
    });
  });

  it("reads numeric metrics only", () => {
    const data = report({
      metrics: [
        { id: "failed", label: "", value: 4 },
        { id: "rate", label: "", value: "n/a" },
      ],
    });
    expect(metric(data, "failed")).toBe(4);
    expect(metric(data, "rate")).toBeUndefined();
    expect(metric(data, "missing")).toBeUndefined();
  });
});

describe("odds", () => {
  it("phrases an e-value as betting odds", () => {
    expect(oddsPhrase(45.4)).toBe("about 45 to 1");
    expect(oddsPhrase(1.53)).toBe("about 1.5 to 1");
    expect(oddsPhrase(5000)).toBe("over 1000 to 1");
    expect(oddsPhrase(0)).toBe("");
    expect(oddsPhrase("x")).toBe("");
  });

  it("sets the same odds as a one-line tile figure", () => {
    expect(oddsValue(45.4)).toBe("45 : 1");
    expect(oddsValue(1.53)).toBe("1.5 : 1");
    expect(oddsValue(2000)).toBe("> 1000 : 1");
    expect(oddsValue(null)).toBe("");
  });
});

describe("sentences", () => {
  it("does not split on abbreviations or decimals", () => {
    const text = "The gap is +0.244 on held-out cases, e.g. charts. The rest follows.";
    expect(firstSentence(text)).toBe("The gap is +0.244 on held-out cases, e.g. charts.");
    expect(restAfterFirstSentence(text)).toBe("The rest follows.");
    expect(firstSentence("no break here")).toBe("no break here");
    expect(restAfterFirstSentence("One sentence only.")).toBe("");
    expect(firstSentence("")).toBe("");
  });

  it("cuts an over-long headline at the first clause break the author wrote", () => {
    const head = `${"The model misreads stacked bars ".repeat(3)}— ${"which costs it accuracy on every chart ".repeat(3)}.`;
    const [headline, body] = headlineSplit(`${head} Second sentence.`);
    expect(headline).toBe(
      "The model misreads stacked bars The model misreads stacked bars The model misreads stacked bars.",
    );
    expect(body.startsWith("Which costs it accuracy")).toBe(true);
    expect(body.endsWith("Second sentence.")).toBe(true);
  });

  it("keeps a short headline whole and strips markdown", () => {
    expect(headlineSplit("**Bold** claim with `code` inside it. Then more.")).toEqual([
      "Bold claim with code inside it.",
      "Then more.",
    ]);
    expect(headlineSplit("")).toEqual(["", ""]);
    expect(stripMarkdown("*a* **b** `c`")).toBe("a b c");
  });

  it("trims a lead only at a sentence end", () => {
    const text = `${"A sentence of moderate length. ".repeat(20)}`;
    const lead = leadFrom(text, 100);
    expect(lead.endsWith(".")).toBe(true);
    expect(lead.length).toBeLessThanOrEqual(160);
    expect(leadFrom("Short.")).toBe("Short.");
  });
});

describe("numbers", () => {
  it("formats counts with their share", () => {
    expect(countOf(45, 125)).toBe("45 of 125 (36%)");
    expect(countOf(3, 0)).toBe("3");
    expect(countOf("x", 10)).toBe("—");
    expect(plural(1, "case")).toBe("1 case");
    expect(plural(3, "case")).toBe("3 cases");
    expect(plural(2, "analysis", "analyses")).toBe("2 analyses");
  });

  it("formats effects and chart values", () => {
    expect(pointsGap(-0.29)).toBe("29 percentage points");
    expect(pointsGap(0.004)).toBe("0.4 percentage points");
    expect(pointsGap("x")).toBe("");
    expect(chartValue(0.5995475113122173)).toBe("0.600");
    expect(chartValue(0.25, { signed: true })).toBe("+0.250");
    expect(chartValue(7)).toBe("7");
    expect(chartValue(undefined)).toBe("—");
    expect(chartPercent(0.256)).toBe("25.6%");
    expect(chartPercent(null)).toBe("0.0%");
    expect(chartPercent("x")).toBe("—");
    expect(stageCode("m4")).toBe("m4");
    expect(stageCode(undefined)).toBe("");
  });

  it("colours outcome slices by meaning, not position", () => {
    expect(outcomeColors(["Fail", "Pass"])).toEqual(["#f06d5f", "#6bd8ad"]);
    expect(outcomeColors(["Pass", "Fail"])).toEqual(["#6bd8ad", "#f06d5f"]);
    expect(outcomeColors(["alpha", "Unknown", "beta"])).toEqual(["#89a6ff", "#586a65", "#f4ca72"]);
  });
});
