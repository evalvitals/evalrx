/** The detail pages' shell and the generic record renderers they share. */
import { ZoomableImage } from "./lightbox";
import { ArrowLeft, Bot, ShieldCheck } from "lucide-react";
import type { ReportData } from "./types";
import { humanize, formatCompact } from "./format";

/**
 * The frame every depth below the overview sits in.
 *
 * `agent` rides along because the overview's topbar — the only other place the
 * run says which agent produced it — is not rendered here, and these are the
 * screens where two runs actually get compared side by side.
 */
export function DetailShell({
  title,
  subtitle,
  back,
  agent,
  children,
}: {
  title: string;
  subtitle: string;
  back: () => void;
  agent?: string;
  children: React.ReactNode;
}) {
  return (
    <main className="detail-page">
      <nav>
        <button onClick={back}>
          <ArrowLeft size={17} /> Overview
        </button>
        <div>
          <strong>{title}</strong>
          <span>{subtitle}</span>
        </div>
        {agent && (
          <span className="shell-agent" title="The agent that drove this run">
            <Bot size={13} />
            {agent}
          </span>
        )}
      </nav>
      {children}
    </main>
  );
}

export function DetailBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-block">
      <small>{label}</small>
      <pre>{value || "Not retained"}</pre>
    </div>
  );
}

export function EmptyStage({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty-stage">
      <ShieldCheck />
      <div>
        <h3>{title}</h3>
        <p>{body}</p>
      </div>
    </div>
  );
}

export function KeyValueGrid({ values }: { values: Record<string, unknown> }) {
  return (
    <div className="kv-grid">
      {Object.entries(values).map(([key, value]) => (
        <div key={key}>
          <small>{humanize(key)}</small>
          <b>{formatCompact(value)}</b>
        </div>
      ))}
    </div>
  );
}

export function RecordTable({ rows }: { rows: Array<object | null> }) {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row || {})))].slice(0, 8);
  return (
    <div className="record-table-wrap">
      <table className="record-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{humanize(column)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 16).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>
                  {formatCompact((row as Record<string, unknown> | null)?.[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MediaPreview({
  media,
  caseId,
}: {
  media: ReportData["media"][number];
  caseId: string;
}) {
  const source = media.data_uri || `/api/media/${encodeURIComponent(media.id)}`;
  if (media.kind === "image")
    return (
      <ZoomableImage
        className="case-media"
        src={source}
        alt={`Input for case ${caseId}`}
        caption={`Input for case ${caseId}`}
      />
    );
  if (media.kind === "video")
    return <video className="case-media" controls preload="metadata" src={source} />;
  return <audio controls preload="metadata" src={source} />;
}
