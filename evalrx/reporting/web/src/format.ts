/** Values and identifiers, worded for a reader. */

export function normalizeKey(value: unknown) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

export function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function plainStatus(value: string) {
  return (
    (
      {
        supported: "held up",
        refuted: "did not hold up",
        inconclusive: "not enough evidence",
      } as Record<string, string>
    )[value.toLowerCase()] || humanize(value)
  );
}

export function plainEvidence(value: unknown) {
  return (
    (
      {
        observational: "pattern in the data",
        intervention: "direct comparison",
        none: "no usable evidence",
      } as Record<string, string>
    )[String(value || "").toLowerCase()] || "recorded evidence"
  );
}

export function plainTest(value: unknown) {
  return (
    (
      {
        signal_label_assoc: "Compare this behavior between correct and incorrect answers",
        rank_corr: "Check whether the behavior and errors move together",
        single_rate_evalue: "Check whether the rate could plausibly be chance",
      } as Record<string, string>
    )[String(value || "")] || "Independent statistical check"
  );
}

export function plainVerdict(value: string) {
  return String(value)
    .replace(/signal_label_assoc: signal '[^']+' vs FAIL:\s*/i, "")
    .replace(/\[clustered bootstrap \(unpaired\)\]\s*/i, "")
    .replace(/\s*\[BH:.*?\]/i, "");
}

export function plainFailureMode(value: unknown) {
  return (
    (
      {
        computation_slip: "The model may be making a calculation mistake",
        hallucination: "The model may be reading information that is not there",
        brittleness: "The model may be overly sensitive to a small change",
        format_error: "The model may understand the task but reply in the wrong format",
      } as Record<string, string>
    )[String(value || "").toLowerCase()] || (value ? humanize(String(value)) : "Not specified")
  );
}

export function plainCandidateKind(value: unknown) {
  return (
    (
      {
        prompt: "A change to the instructions",
        prompt_template: "A change to the instructions",
        decoding: "A change to how answers are generated",
        intervention: "A targeted model intervention",
      } as Record<string, string>
    )[String(value || "").toLowerCase()] || "A proposed repair"
  );
}

export function plainOperation(value: unknown) {
  return (
    (
      {
        imagezoomin: "Zoom in on part of the image",
        zoomcenter: "Zoom into the chart center",
        crop: "Crop the visual evidence",
        cropcasebbox: "Crop the relevant chart region",
        sharpen: "Sharpen chart details",
        contrast: "Increase visual contrast",
        detect: "Locate an object or region",
      } as Record<string, string>
    )[normalizeKey(value)] || humanize(String(value || "tool"))
  );
}

export function seconds(value: unknown) {
  return typeof value === "number" ? `${value.toFixed(1)}s` : "—";
}

export function number(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")
    : "—";
}

export function percent(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

export function formatInterval(value: unknown) {
  return Array.isArray(value) && value.length >= 2
    ? `[${number(value[0])}, ${number(value[1])}]`
    : "—";
}

export function formatCompact(value: unknown) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number")
    return Number.isInteger(value)
      ? String(value)
      : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  if (typeof value === "object") return JSON.stringify(value).slice(0, 180);
  return String(value).slice(0, 220);
}

export function displayValue(value: unknown) {
  return formatCompact(value);
}

export function format(value: unknown) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "Not retained";
  return JSON.stringify(value, null, 2);
}
