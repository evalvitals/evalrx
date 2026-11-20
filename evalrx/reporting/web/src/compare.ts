/**
 * Two runs side by side: what each measured, how far each got, and which of
 * the cases they share changed outcome. Pure, so the joins are testable.
 *
 * Runs are joined on what they have in common — metrics by id, stages by id,
 * cases by id — and the comparison says how much was joined, so a reader can
 * tell two runs of one benchmark from two runs that merely sit side by side.
 */
import type { ReportData } from "./types";

type Value = string | number | null;

export type Comparison = {
  sameDataset: boolean;
  sameKind: boolean;
  metrics: Array<{ id: string; label: string; a: Value; b: Value }>;
  stages: Array<{ id: string; title: string; a: string | null; b: string | null }>;
  cases: {
    /** Case ids both runs evaluated. */
    shared: number;
    onlyA: number;
    onlyB: number;
    /** Shared cases that failed in A and passed in B, and the reverse. */
    nowPass: string[];
    nowFail: string[];
  };
};

const OUTCOMES = new Set(["pass", "fail"]);

export function compareRuns(a: ReportData, b: ReportData): Comparison {
  const metricsB = new Map(b.metrics.map((m) => [m.id, m]));
  const metrics = [
    ...a.metrics.map((m) => ({
      id: m.id,
      label: m.label,
      a: m.value,
      b: metricsB.get(m.id)?.value ?? null,
    })),
    ...b.metrics
      .filter((m) => !a.metrics.some((other) => other.id === m.id))
      .map((m) => ({ id: m.id, label: m.label, a: null, b: m.value })),
  ];
  const stagesB = new Map(b.stages.map((s) => [s.id, s]));
  const stages = a.stages.map((s) => ({
    id: s.id,
    title: s.title,
    a: s.status,
    b: stagesB.get(s.id)?.status ?? null,
  }));
  const statusB = new Map(b.cases.map((c) => [c.id, c.status]));
  const idsA = new Set(a.cases.map((c) => c.id));
  const nowPass: string[] = [];
  const nowFail: string[] = [];
  let shared = 0;
  for (const c of a.cases) {
    const other = statusB.get(c.id);
    if (other === undefined) continue;
    shared += 1;
    if (!OUTCOMES.has(c.status) || !OUTCOMES.has(other) || c.status === other) continue;
    (other === "pass" ? nowPass : nowFail).push(c.id);
  }
  return {
    sameDataset: a.setting.dataset === b.setting.dataset,
    sameKind: a.setting.kind === b.setting.kind,
    metrics,
    stages,
    cases: {
      shared,
      onlyA: a.cases.length - shared,
      onlyB: b.cases.filter((c) => !idsA.has(c.id)).length,
      nowPass,
      nowFail,
    },
  };
}
