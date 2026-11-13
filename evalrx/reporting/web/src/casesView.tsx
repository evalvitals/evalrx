/** The Case Studio: every case's model I/O, filterable by outcome and partition. */
import { useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ChevronRight, Search } from "lucide-react";
import type { Case, Partition, ReportData } from "./types";
import { DetailShell, DetailBlock, MediaPreview } from "./detail";
import { plainOperation, formatCompact, displayValue, format } from "./format";

/** One line of the studio's list: a partition heading or a case. */
type StudioRow = { kind: "group"; partition: Partition } | { kind: "case"; item: Case };

/** The partition a case belongs to, or the "unrecorded" row when none. */
function partitionOf(item: Case, partitions: Partition[]): Partition | undefined {
  return partitions.find((p) => p.split === (item.split ?? ""));
}

/** The E / H / C subscript beside a case, coloured like its band on the cylinder. */
function SplitChip({ partition, title }: { partition?: Partition; title?: boolean }) {
  if (!partition || !partition.split) return null;
  return (
    <span
      className={`split-chip split-${partition.split}`}
      title={`${partition.label} split — ${partition.role}`}
    >
      D<sub>{partition.code}</sub>
      {title && <em>{partition.label}</em>}
    </span>
  );
}

export function CasesView({
  data,
  back,
  initialCaseId,
  initialSplit,
}: {
  data: ReportData;
  back: () => void;
  initialCaseId?: string;
  initialSplit?: string;
}) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const partitions = data.setting.partitions ?? [];
  // Which band of the batch cylinder the list shows. "all" keeps every case
  // and draws the bands as headed groups instead — the batch was split before
  // anything ran, and a flat list hid that.
  const [split, setSplit] = useState(
    initialSplit && partitions.some((p) => p.split === initialSplit) ? initialSplit : "all",
  );
  const [selected, setSelected] = useState<Case | null>(
    initialCaseId ? data.cases.find((c) => c.id === initialCaseId) || null : null,
  );
  // fixed / broken / unchanged describe what the REPAIR did; pass / fail
  // describe the original run. Filtering on the wrong one silently returns
  // nothing, so each button reads the field it names.
  const repairFilters = new Set(["fixed", "broken", "unchanged"]);
  // Status and search narrow every partition alike; the partition buttons
  // count under that narrowing so switching between them never reads "0".
  const narrowed = data.cases.filter((item) => {
    const matches = repairFilters.has(status)
      ? item.repair?.status === status
      : status === "all" || item.status === status;
    return (
      matches &&
      `${item.id} ${item.prompt} ${item.task}`.toLowerCase().includes(query.toLowerCase())
    );
  });
  const matching =
    split === "all" ? narrowed : narrowed.filter((item) => (item.split ?? "") === split);
  // Grouped in the cylinder's order (explore on top, then what was withheld)
  // whenever the run has more than one partition to show; a heading carries
  // the group's count under the current filter, not the partition's size.
  const rows: StudioRow[] = [];
  if (split === "all" && partitions.length > 1) {
    for (const partition of partitions) {
      const members = matching.filter((item) => (item.split ?? "") === partition.split);
      if (!members.length) continue;
      rows.push({ kind: "group", partition });
      for (const item of members) rows.push({ kind: "case", item });
    }
  } else {
    for (const item of matching) rows.push({ kind: "case", item });
  }
  const firstCase = rows.find(
    (row): row is Extract<StudioRow, { kind: "case" }> => row.kind === "case",
  )?.item;
  const countIn = (partition: Partition) =>
    narrowed.filter((item) => (item.split ?? "") === partition.split).length;
  const parentRef = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => (rows[index]?.kind === "group" ? 58 : 92),
    overscan: 8,
  });
  return (
    <DetailShell
      title="Case Studio"
      subtitle="Inspect the actual input, expected answer, model output, and attached media."
      back={back}
      agent={data.setting.diagnosed_by}
    >
      <div className="case-toolbar">
        <label>
          <Search size={16} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search cases"
          />
        </label>
        <div>
          {["all", "fail", "pass", "fixed", "broken", "unchanged"].map((value) => (
            <button
              className={status === value ? "active" : ""}
              onClick={() => setStatus(value)}
              key={value}
            >
              {value}
            </button>
          ))}
        </div>
        <span>{matching.length} cases</span>
      </div>
      {partitions.length > 0 && (
        <div className="case-splits" role="tablist" aria-label="Batch partition">
          <small>FROZEN BATCH</small>
          <button className={split === "all" ? "active" : ""} onClick={() => setSplit("all")}>
            All partitions
            <b>
              {narrowed.length}
              {narrowed.length !== data.cases.length ? ` / ${data.cases.length}` : ""}
            </b>
          </button>
          {partitions.map((partition) => (
            <button
              key={partition.split || "unrecorded"}
              className={`split-${partition.split || "unrecorded"}${split === partition.split ? " active" : ""}`}
              onClick={() => setSplit(partition.split)}
              title={partition.role}
            >
              <span>
                D<sub>{partition.code}</sub>
              </span>
              {partition.label}
              <b>
                {countIn(partition)}
                {countIn(partition) !== partition.n ? ` / ${partition.n}` : ""}
              </b>
            </button>
          ))}
          {partitions.some((p) => p.inferred) && (
            <em title="This run predates the split tag on case records; the partition is read from run_start's explore count and the order the cases were logged in.">
              split inferred from the record order
            </em>
          )}
        </div>
      )}
      <div className="case-studio">
        <div className="virtual-list" ref={parentRef}>
          <div style={{ height: virtual.getTotalSize(), position: "relative" }}>
            {virtual.getVirtualItems().map((row) => {
              const entry = rows[row.index];
              const style = {
                position: "absolute" as const,
                top: 0,
                left: 0,
                width: "100%",
                height: row.size,
                transform: `translateY(${row.start}px)`,
              };
              if (entry.kind === "group") {
                const { partition } = entry;
                return (
                  <div
                    className={`virtual-group split-${partition.split || "unrecorded"}`}
                    key={`group-${partition.split}`}
                    style={style}
                    title={partition.role}
                  >
                    <span>
                      D<sub>{partition.code}</sub>
                    </span>
                    <div>
                      <strong>{partition.label}</strong>
                      <p>{partition.role}</p>
                    </div>
                    <b>{countIn(partition)}</b>
                  </div>
                );
              }
              const { item } = entry;
              return (
                <button
                  className={`virtual-row ${selected?.id === item.id ? "active" : ""}`}
                  key={item.id}
                  onClick={() => setSelected(item)}
                  style={style}
                >
                  <span className={`status status-${item.status}`}>{item.status}</span>
                  <div>
                    <strong>
                      {item.id}
                      {split === "all" && partitions.length > 1 ? null : (
                        <SplitChip partition={partitionOf(item, partitions)} />
                      )}
                    </strong>
                    <p>{item.prompt || "No prompt retained"}</p>
                  </div>
                  <ChevronRight />
                </button>
              );
            })}
          </div>
        </div>
        <CaseDetail item={selected || firstCase} data={data} />
      </div>
    </DetailShell>
  );
}

/**
 * The same case before and after M5's repair, side by side.
 *
 * A repair reported as "12 repaired, 1 broken" is only reviewable if you can
 * read what changed. The gold answer sits between the two so the direction is
 * legible without arithmetic: which side matches it is the whole finding.
 */
function RepairStudy({ item }: { item: Case }) {
  const r = item.repair;
  if (!r?.output) return null;
  const flipped = r.status === "fixed" || r.status === "broken";
  return (
    <div className={`repair-study ${r.status || ""}`}>
      <header>
        <span className="eyebrow">
          {r.status === "fixed"
            ? "The repair corrected this case"
            : r.status === "broken"
              ? "The repair broke this case"
              : "The repair did not change this case"}
        </span>
        {r.candidate && (
          <em>
            {r.tier} · {r.candidate}
          </em>
        )}
      </header>
      <div className="repair-cols">
        <div>
          <small>BEFORE — UNCHANGED MODEL</small>
          <b>{displayValue(item.observed)}</b>
        </div>
        <div className="gold">
          <small>CORRECT ANSWER</small>
          <b>{displayValue(item.expected)}</b>
        </div>
        <div>
          <small>AFTER — WITH THE REPAIR</small>
          <b>{displayValue(r.output)}</b>
        </div>
      </div>
      {!flipped && (
        <p>
          Both answers land on the same side of the gold answer, so this case counts toward neither
          the repairs nor the regressions.
        </p>
      )}
    </div>
  );
}

function CaseDetail({ item, data }: { item?: Case; data: ReportData }) {
  if (!item) return <section className="case-detail empty">No cases match this filter.</section>;
  const partition = partitionOf(item, data.setting.partitions ?? []);
  return (
    <section className="case-detail">
      <div className="case-detail-head">
        <div>
          <span className={`status status-${item.status}`}>{item.status}</span>
          {item.repair?.status && item.repair.status !== "unchanged" && (
            <span className={`status status-${item.repair.status}`}>
              {item.repair.status} by repair
            </span>
          )}
          <SplitChip partition={partition} title />
          <h2>{item.id}</h2>
          {partition?.split && <p className="split-note">{partition.role}</p>}
        </div>
        <small>{item.task}</small>
      </div>
      <RepairStudy item={item} />
      <DetailBlock label="MODEL INPUT" value={item.prompt} />
      {item.choices?.length > 0 && (
        <DetailBlock label="CHOICES" value={item.choices.map(String).join("\n")} />
      )}
      <div className="io-grid">
        <DetailBlock label="EXPECTED" value={format(item.expected)} />
        <DetailBlock label="MODEL OUTPUT" value={format(item.observed)} />
      </div>
      {item.media_ids.map((id) => {
        const media = data.media.find((entry) => entry.id === id);
        if (!media) return null;
        return <MediaPreview key={id} media={media} caseId={item.id} />;
      })}
      {Boolean(item.trajectory) && <TrajectoryPanel trajectory={item.trajectory} />}
      {item.tags.length > 0 && (
        <div className="tag-row">
          {item.tags.map((tag) => (
            <span key={tag}>{tag}</span>
          ))}
        </div>
      )}
    </section>
  );
}

function TrajectoryPanel({ trajectory }: { trajectory: any }) {
  const steps = Array.isArray(trajectory?.steps) ? trajectory.steps : [];
  const calls = steps.filter((step: any) => step?.tool_call);
  if (!calls.length) return null;
  return (
    <details className="case-trajectory">
      <summary>Recorded intermediate agent actions ({calls.length})</summary>
      {calls.map((step: any, index: number) => (
        <div className="trajectory-row" key={index}>
          <b>{plainOperation(step.tool_call?.name)}</b>
          <small>{formatCompact(step.tool_call?.args)}</small>
          {step.observation && <p>{formatCompact(step.observation)}</p>}
        </div>
      ))}
    </details>
  );
}
