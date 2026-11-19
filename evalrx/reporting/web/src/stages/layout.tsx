/** The frame every stage's full record is set in: banner, numbers, worked examples. */
import type { ReportData } from "../types";
import { MediaPreview } from "../detail";

export function StageBanner({
  kind,
  title,
  children,
}: {
  kind: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`stage-banner stage-banner-${kind}`}>
      <div>
        <span>{kind}</span>
        <strong>{title}</strong>
      </div>
      <p>{children}</p>
    </div>
  );
}

export function StageKpis({
  items,
}: {
  items: Array<{ label: string; value: React.ReactNode; note?: string | null }>;
}) {
  return (
    <div className="stage-kpis">
      {items.map((item) => (
        <article key={item.label}>
          <span>{item.label}</span>
          <strong>{item.value ?? "—"}</strong>
          {item.note && <small>{item.note}</small>}
        </article>
      ))}
    </div>
  );
}

export function ExampleSection({
  eyebrow,
  title,
  note,
  children,
}: {
  eyebrow: string;
  title: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <section className="example-section">
      <header>
        <span>{eyebrow}</span>
        <h3>{title}</h3>
        <p>{note}</p>
      </header>
      {children}
    </section>
  );
}

export function ExampleMedia({
  mediaIds,
  report,
  caseId,
}: {
  mediaIds?: string[] | null;
  report?: ReportData;
  caseId?: string | null;
}) {
  if (!report || !mediaIds?.length) return null;
  const media = mediaIds
    .map((id) => report.media.find((item) => item.id === id))
    .filter(Boolean) as ReportData["media"];
  if (!media.length) return null;
  return (
    <div className="example-media">
      {media.map((item) => (
        <MediaPreview key={item.id} media={item} caseId={caseId || "example"} />
      ))}
    </div>
  );
}
