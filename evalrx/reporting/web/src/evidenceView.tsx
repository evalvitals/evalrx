/** A stage's evidence: the stage list beside the selected stage's brief or full record. */
import { useState } from "react";
import { stageCode } from "./reportAccess";
import { ChevronRight } from "lucide-react";
import type { ReportData } from "./types";
import { StageBrief } from "./brief";
import { DetailShell } from "./detail";
import { StageArtifact, buildBrief } from "./stages";

/**
 * A stage, at whichever of the two depths the reader asked for.
 *
 * Clicking a stage on the overview used to land on the complete record — every
 * probe, every raw metric name, the model's full output — which is the right
 * page for auditing a conclusion and the wrong one for finding out what the
 * conclusion IS. The brief (L2) answers that in one screen and hands the reader
 * a way down; the full record (L3) is unchanged behind it, because nothing in
 * it was surplus, it was just never the first thing anyone needed.
 *
 * Depth resets when the reader moves to another stage: arriving at M4 already
 * scrolled into its raw event log is a state nobody asked for.
 */
export function EvidenceView({
  data,
  back,
  navigate,
  initialStage,
}: {
  data: ReportData;
  back: () => void;
  navigate?: (view: string) => void;
  initialStage?: string;
}) {
  const [selected, setSelected] = useState(initialStage || data.stages[0]?.id);
  const [full, setFull] = useState(false);
  const stage = data.stages.find((item) => item.id === selected);
  const brief = selected ? buildBrief(selected, data) : null;
  const show = (id: string) => {
    setSelected(id);
    setFull(false);
  };
  return (
    <DetailShell
      title={full ? "Everything this step recorded" : "What this step found"}
      subtitle={
        full
          ? "The complete record behind the summary: every measurement, artifact and agent event."
          : "One screen per step: the finding, the numbers behind it, and what it does not settle."
      }
      back={back}
      agent={data.setting.diagnosed_by}
    >
      <div className="evidence-layout">
        <aside className="stage-list">
          {data.stages.map((item) => (
            <button
              className={selected === item.id ? "active" : ""}
              key={item.id}
              onClick={() => show(item.id)}
            >
              <span>{stageCode(item.code)}</span>
              <div>
                <strong>{item.title}</strong>
                <small>{item.status.replaceAll("-", " ")}</small>
              </div>
              <ChevronRight />
            </button>
          ))}
        </aside>
        <section className="evidence-detail">
          <header className="evidence-head">
            <div>
              <span className="section-kicker">
                {stageCode(stage?.code)} · {stage?.status?.replaceAll("-", " ")}
              </span>
              <h2>{stage?.title}</h2>
              <p className="lead-small">{stage?.purpose}</p>
            </div>
            {brief && (
              <div className="depth-switch" role="tablist" aria-label="Level of detail">
                <button
                  role="tab"
                  aria-selected={!full}
                  className={full ? "" : "active"}
                  onClick={() => setFull(false)}
                >
                  Summary
                </button>
                <button
                  role="tab"
                  aria-selected={full}
                  className={full ? "active" : ""}
                  onClick={() => setFull(true)}
                >
                  Full record
                </button>
              </div>
            )}
          </header>
          {/* The full record is the stage's own artifact and nothing else: the raw
          event stream is Debug's view, and each stage's record already carries
          the events that matter in a shape a reader can use. */}
          {brief && !full ? (
            <StageBrief brief={brief} onDeepen={() => setFull(true)} />
          ) : (
            <StageArtifact stage={selected || ""} report={data} navigate={navigate} />
          )}
        </section>
      </div>
    </DetailShell>
  );
}
