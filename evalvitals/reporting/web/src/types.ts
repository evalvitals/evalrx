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
  series: Array<{ label: string; value: number; highlight?: boolean }>;
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
  stage_detail?: Record<string, any>;
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
