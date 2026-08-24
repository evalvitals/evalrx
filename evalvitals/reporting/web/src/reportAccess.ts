/**
 * Reading a report payload without assuming what a particular run produced.
 *
 * Every helper here takes the whole `ReportData` and returns something safe for
 * an empty run: a `ReportData` is the union of what five stages MIGHT emit, and
 * any given run fills a different subset of it. An `alm` run has no image
 * figures, a run that stopped at M3 has no verdicts, a run that predates
 * contract emission has no `contract` at all — and every one of those must
 * render as an honest blank rather than throwing.
 */
import type { ReportData } from "./types";

/**
 * The newest contract payload for a stage, across cycles.
 *
 * Spans are "c<cycle>.<stage>", so the last key in sort order is the last cycle
 * that ran. Returns undefined for a run that predates contract emission — the
 * caller must render without it rather than showing an error, since those runs
 * are still perfectly readable through the legacy stage_detail.
 */
export function findContract<T>(report: ReportData, stage: string): T | undefined {
  const payloads = report.contract;
  if (!payloads) return undefined;
  const keys = Object.keys(payloads).filter((k) => k.endsWith(`.${stage}`) || k === stage);
  if (!keys.length) return undefined;
  const payload = payloads[keys.sort()[keys.length - 1]];
  if (!payload || "error" in payload) return undefined;
  return payload as T;
}

/** A `stage_detail` entry, or `{}` for a stage this run never reached. */
export function stageDetail(report: ReportData, stage: string): Record<string, any> {
  const detail = (report.stage_detail || {}) as Record<string, any>;
  const value = detail[stage];
  return value && typeof value === "object" ? value : {};
}

/** A run-level metric by id (the compiler emits the same four for every run). */
export function metric(report: ReportData, id: string): number | undefined {
  const found = (report.metrics || []).find((item) => item.id === id);
  return typeof found?.value === "number" ? found.value : undefined;
}

/**
 * An e-value as betting odds, or "" when there is no usable number.
 *
 * An e-value IS odds against the null: e=45 means the evidence runs about 45 to
 * 1 against this being luck. "Reject at 0.05" is the same statement in a
 * dialect nobody outside the field speaks. The console narration
 * (`run_logger._odds_phrase`) renders the same number the same way, so the two
 * surfaces can never quietly disagree about what one number means.
 */
export function oddsPhrase(value: any): string {
  const e = Number(value);
  if (!Number.isFinite(e) || e <= 0) return "";
  if (e >= 1000) return "over 1000 to 1";
  return e >= 10 ? `about ${Math.round(e)} to 1` : `about ${e.toFixed(1)} to 1`;
}

/**
 * The first sentence of a paragraph, for use as a headline.
 *
 * Abbreviations ("e.g.", "i.e.", "vs.") and decimals ("+0.244") both put a dot
 * mid-sentence, so a naive split on "." beheads the conclusion after three
 * words. Only a dot followed by whitespace and a capital letter counts, and a
 * paragraph with no such break is returned whole rather than truncated — a
 * headline that stops mid-clause is worse than a long one.
 */
export function firstSentence(text: string): string {
  const clean = String(text || "").trim();
  if (!clean) return "";
  const match = clean.match(/^([\s\S]{20,}?[.!?])\s+[A-Z(`"']/);
  return (match ? match[1] : clean).trim();
}

/** Everything after `firstSentence`, or "" when the text was one sentence. */
export function restAfterFirstSentence(text: string): string {
  const clean = String(text || "").trim();
  const head = firstSentence(clean);
  return clean.length > head.length ? clean.slice(head.length).trim() : "";
}

// Past this, a headline has stopped being one and the reader is back to
// reading a paragraph set in display type.
const HEADLINE_LIMIT = 175;

/**
 * A paragraph split into [headline, everything else].
 *
 * `firstSentence` alone is not enough: an agent will happily write one
 * 300-character sentence joined by em dashes and semicolons, and setting that
 * in 34px type fills the screen before the reader learns anything. So an
 * over-long first sentence is cut at its first CLAUSE break instead, and the
 * remainder is pushed down into the body where it reads fine.
 *
 * The cut only ever happens at punctuation the author wrote. A sentence with no
 * clause break is returned whole — a headline that stops mid-thought misstates
 * the finding, which is worse than a long one.
 */
export function headlineSplit(text: string): [string, string] {
  const clean = stripMarkdown(String(text || "").trim());
  if (!clean) return ["", ""];
  const head = firstSentence(clean);
  const tail = clean.length > head.length ? clean.slice(head.length).trim() : "";
  if (head.length <= HEADLINE_LIMIT) return [head, tail];
  const breaks = [...head.matchAll(/\s*[—–;:]\s*|\s-\s/g)]
    .filter((m) => (m.index ?? 0) > 40 && (m.index ?? 0) < HEADLINE_LIMIT + 60);
  if (!breaks.length) return [head, tail];
  const at = breaks[0].index ?? 0;
  const cut = head.slice(0, at).trim().replace(/[,;:—–-]$/, "");
  const moved = head.slice(at).replace(/^\s*[—–;:-]\s*/, "").trim();
  const rejoined = [moved.charAt(0).toUpperCase() + moved.slice(1), tail]
    .filter(Boolean).join(" ");
  return [`${cut}.`, rejoined];
}

/**
 * "45 of 125 (36%)" — a count with the share it amounts to.
 *
 * A bare rate is the single most common way these reports lose a reader: 0.36
 * is not a quantity anyone can picture, and the denominator is the part that
 * says whether to believe it.
 */
export function countOf(n: any, total: any): string {
  const count = Number(n);
  const whole = Number(total);
  if (!Number.isFinite(count)) return "—";
  if (!Number.isFinite(whole) || whole <= 0) return String(count);
  return `${count} of ${whole} (${Math.round((count / whole) * 100)}%)`;
}

/** `n` with a noun that agrees with it: "1 case" / "3 cases". */
export function plural(n: number, one: string, many: string = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** Markdown emphasis stripped, so agent prose renders as prose, not as syntax. */
export function stripMarkdown(text: string): string {
  return String(text || "")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\*\*([^*]*)\*\*/g, "$1")
    .replace(/\*([^*]*)\*/g, "$1")
    .trim();
}

/**
 * Colours for an outcome donut, keyed to what each slice MEANS.
 *
 * The palette used to be a positional array, and the compiler does not
 * guarantee slice order: one run emitted [Fail, Pass] and got a red Fail, the
 * next emitted [Pass, Fail] and got a red Pass. A chart whose colours invert
 * between runs is worse than a chart with no colours — a reader who has seen
 * one report now misreads the next at a glance. Anything the vocabulary does
 * not recognise falls back to the neutral sequence, which is honest: an
 * unknown category gets no implied verdict.
 */
const OUTCOME_COLOR: Array<[RegExp, string]> = [
  [/^(fail|wrong|incorrect|error)/i, "#f06d5f"],
  [/^(pass|right|correct|ok)/i, "#6bd8ad"],
  [/^(unknown|unscored|skipped|n\/a)/i, "#586a65"],
];
const NEUTRAL_SEQUENCE = ["#89a6ff", "#f4ca72", "#b48ce0", "#7fc7d9"];

export function outcomeColors(labels: string[]): string[] {
  let spare = 0;
  return labels.map((label) => {
    const hit = OUTCOME_COLOR.find(([pattern]) => pattern.test(String(label).trim()));
    return hit ? hit[1] : NEUTRAL_SEQUENCE[spare++ % NEUTRAL_SEQUENCE.length];
  });
}

/**
 * A paragraph trimmed to at most *maxChars*, cut only at a sentence end.
 *
 * The L2 lead exists to add one or two sentences under the verdict. An agent's
 * full conclusion can run to fifteen lines, and dropping all of it there
 * rebuilds the wall of text this layer was meant to replace — while cutting it
 * mid-sentence would misquote the agent. So the cut lands on a full stop, or
 * does not happen at all.
 */
export function leadFrom(text: string, maxChars = 340): string {
  const clean = stripMarkdown(String(text || "").trim());
  if (clean.length <= maxChars) return clean;
  const ends = [...clean.slice(0, maxChars + 80).matchAll(/[.!?](?=\s|$)/g)]
    .map((m) => (m.index ?? 0) + 1)
    .filter((at) => at <= maxChars + 60);
  return ends.length ? clean.slice(0, ends[ends.length - 1]).trim() : clean.slice(0, maxChars).trim();
}

/** A signed effect as "29 percentage points", or "" when there is no number. */
export function pointsGap(effect: any): string {
  const value = Number(effect);
  if (!Number.isFinite(value)) return "";
  const pp = Math.abs(value) * 100;
  return `${pp < 1 ? pp.toFixed(1) : Math.round(pp)} percentage points`;
}
