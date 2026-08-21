import { useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowLeft, ChevronRight, Search } from "lucide-react";
import type { Case, DebugEvent, ReportData } from "./types";

export function EvidenceView({ data, back }: { data: ReportData; back: () => void }) {
  const [selected, setSelected] = useState(data.stages[0]?.id);
  const stage = data.stages.find((item) => item.id === selected);
  const events = data.debug.events.filter((event) => String(event.stage || "").toLowerCase().includes(selected));
  return <DetailShell title="Stage evidence" subtitle="How the agent moved from measurements to a tested intervention." back={back}>
    <div className="evidence-layout"><aside className="stage-list">{data.stages.map((item) => <button className={selected === item.id ? "active" : ""} key={item.id} onClick={() => setSelected(item.id)}><span>{item.code}</span><div><strong>{item.title}</strong><small>{item.status.replaceAll("-", " ")}</small></div><ChevronRight /></button>)}</aside><section className="evidence-detail"><span className="section-kicker">{stage?.code} · {stage?.status}</span><h2>{stage?.title}</h2><p className="lead-small">{stage?.purpose}</p><div className="evidence-stat"><strong>{stage?.evidence_count || 0}</strong><span>structured evidence records</span></div><h3>Recorded events</h3>{events.length ? events.map((event, index) => <EventRow event={event} key={`${event.event_seq}-${index}`} />) : <p className="empty">No stage-level debug events were retained for this legacy run.</p>}</section></div>
  </DetailShell>;
}

export function CasesView({ data, back }: { data: ReportData; back: () => void }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [selected, setSelected] = useState<Case | null>(null);
  const rows = data.cases.filter((item) => (status === "all" || item.status === status) && `${item.id} ${item.prompt} ${item.task}`.toLowerCase().includes(query.toLowerCase()));
  const parentRef = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({ count: rows.length, getScrollElement: () => parentRef.current, estimateSize: () => 92, overscan: 8 });
  return <DetailShell title="Case Studio" subtitle="Inspect the actual input, expected answer, model output, and attached media." back={back}>
    <div className="case-toolbar"><label><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search cases" /></label><div>{["all", "fail", "pass", "fixed", "broken", "unchanged"].map((value) => <button className={status === value ? "active" : ""} onClick={() => setStatus(value)} key={value}>{value}</button>)}</div><span>{rows.length} cases</span></div>
    <div className="case-studio"><div className="virtual-list" ref={parentRef}><div style={{ height: virtual.getTotalSize(), position: "relative" }}>{virtual.getVirtualItems().map((row) => { const item = rows[row.index]; return <button className={`virtual-row ${selected?.id === item.id ? "active" : ""}`} key={item.id} onClick={() => setSelected(item)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: row.size, transform: `translateY(${row.start}px)` }}><span className={`status status-${item.status}`}>{item.status}</span><div><strong>{item.id}</strong><p>{item.prompt || "No prompt retained"}</p></div><ChevronRight /></button>; })}</div></div><CaseDetail item={selected || rows[0]} data={data} /></div>
  </DetailShell>;
}

function CaseDetail({ item, data }: { item?: Case; data: ReportData }) {
  if (!item) return <section className="case-detail empty">No cases match this filter.</section>;
  return <section className="case-detail"><div className="case-detail-head"><div><span className={`status status-${item.status}`}>{item.status}</span><h2>{item.id}</h2></div><small>{item.task}</small></div><DetailBlock label="MODEL INPUT" value={item.prompt} />{item.choices?.length > 0 && <DetailBlock label="CHOICES" value={item.choices.map(String).join("\n")} />}<div className="io-grid"><DetailBlock label="EXPECTED" value={format(item.expected)} /><DetailBlock label="MODEL OUTPUT" value={format(item.observed)} /></div>{item.media_ids.map((id) => { const media = data.media.find((entry) => entry.id === id); if (!media) return null; return <MediaPreview key={id} media={media} caseId={item.id} />; })}{item.tags.length > 0 && <div className="tag-row">{item.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}</section>;
}

function MediaPreview({ media, caseId }: { media: ReportData["media"][number]; caseId: string }) {
  const source = media.data_uri || `/api/media/${encodeURIComponent(media.id)}`;
  if (media.kind === "image") return <img className="case-media" src={source} alt={`Input for case ${caseId}`} />;
  if (media.kind === "video") return <video className="case-media" controls preload="metadata" src={source} />;
  return <audio controls preload="metadata" src={source} />;
}

export function DebugView({ data, back }: { data: ReportData; back: () => void }) {
  const [filter, setFilter] = useState("all");
  const rows = filter === "all" ? data.debug.events : data.debug.events.filter((row) => row.event === filter);
  const eventTypes = [...new Set(data.debug.events.map((event) => event.event))];
  const parentRef = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({ count: rows.length, getScrollElement: () => parentRef.current, estimateSize: () => 84, overscan: 10 });
  return <DetailShell title="Agent audit log" subtitle="The most detailed layer: ordered decisions, tools, stage events, and publication provenance." back={back}><div className="debug-toolbar"><select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">All event types</option>{eventTypes.map((value) => <option key={value}>{value}</option>)}</select><span>{rows.length} / {data.debug.event_count} events</span></div><div className="debug-list" ref={parentRef}><div style={{ height: virtual.getTotalSize(), position: "relative" }}>{virtual.getVirtualItems().map((row) => <div className="debug-row" key={row.key} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: row.size, transform: `translateY(${row.start}px)` }}><EventRow event={rows[row.index]} /></div>)}</div></div></DetailShell>;
}

function DetailShell({ title, subtitle, back, children }: { title: string; subtitle: string; back: () => void; children: React.ReactNode }) {
  return <main className="detail-page"><nav><button onClick={back}><ArrowLeft size={17} /> Overview</button><div><strong>{title}</strong><span>{subtitle}</span></div></nav>{children}</main>;
}

function EventRow({ event }: { event: DebugEvent }) {
  return <div className="event-row"><code>{String(event.event_seq || "—").padStart(3, "0")}</code><span className="event-type">{event.event}</span><div><strong>{event.stage || "RUN"}{event.cycle !== undefined && event.cycle !== null ? ` · cycle ${event.cycle}` : ""}</strong><p>{event.summary}</p></div></div>;
}

function DetailBlock({ label, value }: { label: string; value: string }) {
  return <div className="detail-block"><small>{label}</small><pre>{value || "Not retained"}</pre></div>;
}

function format(value: unknown) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "Not retained";
  return JSON.stringify(value, null, 2);
}
