/** One stage at either depth: its full record (L3) or its brief (L2). */
import { stageDetail } from "../reportAccess";
import type { ReportData } from "../types";
import type { Navigate } from "../route";
import { type Brief } from "../brief";
import { EmptyStage } from "../detail";
import { M1Detail, briefM1 } from "./m1";
import { M2Detail, briefM2 } from "./m2";
import { M3Detail, briefM3 } from "./m3";
import { M4Detail, briefM4 } from "./m4";
import { M5Detail, briefM5 } from "./m5";

export function StageArtifact({
  stage,
  report,
  navigate,
}: {
  stage: string;
  report: ReportData;
  navigate?: Navigate;
}) {
  if (stage === "m1") return <M1Detail data={stageDetail(report, "m1")} report={report} />;
  if (stage === "m2") return <M2Detail data={stageDetail(report, "m2")} />;
  if (stage === "m3") return <M3Detail data={stageDetail(report, "m3")} report={report} />;
  if (stage === "m4") return <M4Detail data={stageDetail(report, "m4")} report={report} />;
  if (stage === "m5")
    return <M5Detail data={stageDetail(report, "m5")} report={report} navigate={navigate} />;
  return (
    <EmptyStage title="No stage data" body="This stage did not retain a structured artifact." />
  );
}

export function buildBrief(stage: string, report: ReportData): Brief | null {
  const status = report.stages.find((item) => item.id === stage)?.status || "";
  if (["not-run", "skipped"].includes(status) && stage !== "m5") {
    return {
      verdict: "This step did not run.",
      lead:
        "Nothing was measured here, so there is nothing to summarise. The steps " +
        "before it still stand on their own.",
      kpis: [],
      points: [],
    };
  }
  if (stage === "m1") return briefM1(report, stageDetail(report, "m1"));
  if (stage === "m2") return briefM2(report, stageDetail(report, "m2"));
  if (stage === "m3") return briefM3(report, stageDetail(report, "m3"));
  if (stage === "m4") return briefM4(report, stageDetail(report, "m4"));
  if (stage === "m5") return briefM5(report, stageDetail(report, "m5"));
  return null;
}
