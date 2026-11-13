/** The raw agent event stream, for auditing a run step by step. */
import { useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { stageCode } from "./reportAccess";
import type { DebugEvent, ReportData } from "./types";
import { DetailShell } from "./detail";

export function DebugView({ data, back }: { data: ReportData; back: () => void }) {
  const [filter, setFilter] = useState("all");
  const rows =
    filter === "all" ? data.debug.events : data.debug.events.filter((row) => row.event === filter);
  const eventTypes = [...new Set(data.debug.events.map((event) => event.event))];
  const parentRef = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 84,
    overscan: 10,
  });
  return (
    <DetailShell
      title="Agent audit log"
      subtitle="The most detailed layer: ordered decisions, tools, stage events, and publication provenance."
      back={back}
      agent={data.setting.diagnosed_by}
    >
      <div className="debug-toolbar">
        <select value={filter} onChange={(event) => setFilter(event.target.value)}>
          <option value="all">All event types</option>
          {eventTypes.map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
        <span>
          {rows.length} / {data.debug.event_count} events
        </span>
      </div>
      <div className="debug-list" ref={parentRef}>
        <div style={{ height: virtual.getTotalSize(), position: "relative" }}>
          {virtual.getVirtualItems().map((row) => (
            <div
              className="debug-row"
              key={row.key}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: row.size,
                transform: `translateY(${row.start}px)`,
              }}
            >
              <EventRow event={rows[row.index]} />
            </div>
          ))}
        </div>
      </div>
    </DetailShell>
  );
}

function EventRow({ event }: { event: DebugEvent }) {
  return (
    <div className="event-row">
      <code>{String(event.event_seq || "—").padStart(3, "0")}</code>
      <span className="event-type">{event.event}</span>
      <div>
        <strong>
          {stageCode(event.stage) || "RUN"}
          {event.cycle !== undefined && event.cycle !== null ? ` · cycle ${event.cycle}` : ""}
        </strong>
        <p>{event.summary}</p>
      </div>
    </div>
  );
}
