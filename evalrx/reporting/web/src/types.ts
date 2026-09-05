import type {
  DiagnosisOutput,
  FixOutput,
  HypothesisTestOutput,
  ProbeOutput,
  StatsReportWire,
} from "./contract/contract";

/**
 * Payloads the pipeline validated against the wire contract on the way out,
 * keyed by span id ("c0.m1", "c0.m2", ... / "m5_fix").
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
  /** What M5's confirmed repair answered on this case, when it was one of the
   *  held-out cases the repair was validated on. */
  repair?: {
    candidate?: string;
    tier?: string;
    /** fixed = was wrong, became right. broken = was right, became wrong. */
    status?: "fixed" | "broken" | "unchanged" | string;
    output?: string;
  };
};

/** One module's label on the case-study sheet. */
export type CaseStudyModule = { code: string; name: string; subtitle: string };

export type CaseStudyVerdict = {
  id: string;
  failure_mode: string;
  status?: string | null;
  statement?: string | null;
  test_name?: string | null;
  effect?: number | null;
  ci?: number[] | null;
  evidence_grade?: string | null;
  underpowered?: boolean | null;
};

export type CaseStudyPhase = {
  tests: Array<{
    signal?: string | null;
    effect?: number | null;
    ci?: number[] | null;
    p_value?: number | null;
    survives_correction: boolean;
    degenerate: boolean;
    in_correction_family: boolean;
  }>;
  n_in_correction_family: number;
  n_degenerate: number;
  correction_method?: string | null;
  survivors: string[];
};

/**
 * The whole run as one failure-to-repair sheet, compiled by
 * `evalrx/reporting/case_study.py`. Absent on a run with no probe or stats
 * artifacts — the section is dropped rather than rendered empty, so every
 * optional block here means "this run did not get that far", never "zero".
 */
export type CaseStudy = {
  modules: CaseStudyModule[];
  headline: {
    model?: string | null;
    dataset?: string | null;
    n_cases?: number | null;
    baseline_accuracy?: number | null;
    n_explore?: number | null;
    n_heldout?: number | null;
    judge?: string | null;
    repair?: string | null;
    repair_tier?: string | null;
    /** candidate_rate − baseline_rate on the held-out pairs. */
    delta?: number | null;
  };
  m1: {
    selection_mode?: string | null;
    selected_analyzers: string[];
    families: Array<{
      family: string;
      label: string;
      selected: boolean;
      analyzers: string[];
      /** The family's fixed probe menu, in words. `used` marks the rows this
       *  run actually ran, `confirmed` the row that produced the surviving
       *  signal; the rest stay greyed, so the card shows the whole menu rather
       *  than only the order. */
      probes: Array<{ phrase: string; used: boolean; analyzers: string[]; confirmed?: boolean }>;
    }>;
    questions: Array<{ question: string; analyzers: string[]; n_measurements: number; n_candidates: number }>;
    n_analyzers: number;
    n_measured: number;
    /** Signals that entered the correction family: the BH denominator. */
    n_forwarded?: number | null;
    dropped: { saw_the_answer_key: number; never_varied: number; partial_coverage: number };
    note: string;
    signal_curve?: {
      signal: string;
      analyzer: string;
      field: string;
      binning: "levels" | "quartiles" | string;
      bins: Array<{ label: string; value?: unknown; n_cases: number; n_fail: number; failure_rate?: number | null }>;
      n_cases: number;
      all_surviving_signals: string[];
      n_surviving_signals: number;
    } | null;
  } | null;
  m2: Record<string, CaseStudyPhase> | null;
  m3: Array<{ id: string; failure_mode: string; statement?: string | null; expected_direction?: string | null }>;
  m4: CaseStudyVerdict[];
  m5: {
    ladder: Array<{
      tier: string; label: string; n_candidates: number; best_effect?: number | null;
      best_candidate?: string | null; status: string; within_cap: boolean; tier_cap?: string | null;
    }>;
    candidates: Array<{ name?: string | null; tier: string; effect?: number | null; selected: boolean }>;
    confirmed: Array<{ name?: string | null; tier: string; effect?: number | null; selected: boolean }>;
  } | null;
  repair: {
    name?: string | null; tier?: string | null; strategy?: string | null; n_samples?: number | null;
    steps: Array<{ title: string; lines: string[]; mono?: boolean }>;
  } | null;
  validation: {
    candidate?: string | null; tier?: string | null; n_pairs: number;
    baseline_rate?: number | null; candidate_rate?: number | null;
    n_fixed: number; n_broken: number; both_correct: number; both_wrong: number;
    effect?: number | null; ci?: number[] | null; e_value?: number | null; verdict?: string | null;
  } | null;
  example_case: {
    case_id: string; split: string; question?: string | null; gold?: unknown;
    /** The case's own outcome. The sheet picks a failing case, so this is how
     *  the answer below is labelled without the reader having to infer it. */
    label?: string | null;
    baseline_output?: unknown;
    /** The answer the generation ended on — the head of a chain of thought is
     *  the least useful part of it on a card this size. */
    baseline_answer?: string | null;
    signal?: string | null; media_paths?: string[];
  } | null;
  /** The caveats the sheet must carry, derived from the run's own numbers. */
  qa_flags: Array<{ level: string; code: string; detail: string }>;
};

export type ReportData = {
  trace_id: string;
  /** `model` is what was diagnosed; `diagnosed_by` is the agent that did the
   *  diagnosing. Empty on a run that recorded neither a manifest nor a coder. */
  setting: { model: string; dataset: string; question: string; protocol: string; n_cases: number; diagnosed_by?: string;
    /** The cover figure the run shipped (`evalrx_main.*` at its root), as a data URI. */
    hero_image?: string };
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
  /** The run as one sheet. Null when it produced no probe or stats artifacts. */
  case_study?: CaseStudy | null;
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
