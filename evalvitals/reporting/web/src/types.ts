import type {
  DiagnosisOutput,
  FixOutput,
  HypothesisTestOutput,
  ProbeOutput,
  StatsReportWire,
} from "./contract/contract";

/**
 * Payloads the pipeline validated against the wire contract on the way out,
 * keyed by span id ("c0.m1", "c0.m2", ... / "m4_fix").
 *
 * These are typed; `stage_detail` below is not, and cannot be — it is compiled
 * from the run log by a chain of defensive lookups, so its shape is whatever
 * that run happened to write. Prefer these wherever a view can use them, and
 * treat a missing key as "this run emitted no contract payload for that stage",
 * never as "the stage did not run".
 */
export type ContractPayloads = {
  [span: string]:
    | ProbeOutput
    | StatsReportWire
    | DiagnosisOutput
    | HypothesisTestOutput
    | FixOutput
    | { stage: string; error: string }
    | undefined;
};

export type {
  AnalyzerSelection,
  FixAttemptWire,
  DiagnosisOutput,
  FixOutput,
  HypothesisTestOutput,
  Modality,
  ProbeOutput,
  StatsReportWire,
} from "./contract/contract";

export type Stage = {
  id: string;
  code: string;
  title: string;
  purpose: string;
  status: string;
  evidence_count: number;
};

export type Finding = {
  id: string;
  title: string;
  summary: string;
  why_it_matters?: string;
  evidence_level?: string;
  limitation?: string;
};

export type Chart = {
  id: string;
  kind: "donut" | "bar";
  title: string;
  subtitle?: string;
  series: Array<{
    label: string;
    /** The producing analyzer's own sentence for what this measures. Empty when
     *  it documented none — say so, never paraphrase the identifier. */
    means?: string;
    raw_label?: string;
    value: number;
    highlight?: boolean;
  }>;
};

export type Case = {
  id: string;
  status: string;
  prompt: string;
  expected: unknown;
  observed: unknown;
  choices: unknown[];
  tags: string[];
  task: string;
  media_ids: string[];
  trajectory?: unknown;
  /** What M4's confirmed repair answered on this case, when it was one of the
   *  held-out cases the repair was validated on. */
  repair?: {
    candidate?: string;
    tier?: string;
    /** fixed = was wrong, became right. broken = was right, became wrong. */
    status?: "fixed" | "broken" | "unchanged" | string;
    output?: string;
  };
};

export type ReportData = {
  trace_id: string;
  setting: { model: string; dataset: string; question: string; protocol: string; n_cases: number };
  summary: { headline: string; answer: string; confidence: string; stopped_by: string };
  metrics: Array<{ id: string; label: string; value: string | number }>;
  stages: Stage[];
  findings: Finding[];
  charts: Chart[];
  repairs: Array<{ id: string; fixed: boolean; title: string; effect?: number; fixed_cases: number; broken_cases: number }>;
  /** Legacy hand-built per-stage view. Untyped by nature — see ContractPayloads. */
  stage_detail?: Record<string, any>;
  /** Contract-validated stage payloads. Absent on runs from before emission existed. */
  contract?: ContractPayloads;
  cases: Case[];
  media: Array<{ id: string; kind: string; path: string; data_uri?: string }>;
  debug: { event_count: number; events: DebugEvent[] };
};

export type DebugEvent = {
  event: string;
  stage?: string;
  cycle?: number;
  event_seq?: number;
  ts?: string;
  span_id?: string;
  summary: string;
};

export type LayoutEnvelope = {
  generated_by: { mode: string; model?: string };
  spec: { root: string; elements: Record<string, LayoutElement> };
};

export type LayoutElement = {
  type: string;
  props: Record<string, unknown>;
  children?: string[];
};
