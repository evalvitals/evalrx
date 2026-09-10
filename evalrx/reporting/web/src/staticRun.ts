/**
 * Opening a zipped run with no server behind the page.
 *
 * The GitHub Pages build of this app (`vite build --mode static`) has no
 * /api/upload to hand an archive to, so the archive is read here, in the
 * browser: the published report (report/report_data.json + report_spec.json,
 * what `evalrx publish` writes and `evalrx serve` produces on first open) is
 * the payload, and every media file or stage figure the report cites is
 * resolved inside the archive to an object URL and attached as `data_uri` —
 * the same field a portable export fills, so nothing downstream changes.
 *
 * Not covered: a raw run that was never published. Compiling one needs the
 * Python side (evalrx.reporting.dynamic), so the drop zone says so instead of
 * guessing.
 */
import JSZip from "jszip";
import type { UploadedRun } from "./upload";

const FIGURE_SUFFIXES = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"];
const MIME: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", webp: "image/webp",
  svg: "image/svg+xml", wav: "audio/wav", mp3: "audio/mpeg", flac: "audio/flac", ogg: "audio/ogg",
  m4a: "audio/mp4", mp4: "video/mp4", webm: "video/webm",
};

const join = (dir: string, rel: string) => (dir ? `${dir}/${rel}` : rel);
const parentOf = (path: string) => (path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "");

export async function readRunArchive(file: File): Promise<UploadedRun> {
  const zip = await JSZip.loadAsync(file);
  const files = zip.files;
  const dataEntries = Object.keys(files)
    .filter((name) => !files[name].dir && /(^|\/)report\/report_data\.json$/.test(name))
    .sort((a, b) => a.length - b.length); // the top-most published report wins
  if (dataEntries.length === 0) {
    throw new Error(
      "This archive holds no published report (no report/report_data.json). "
      + "Run `evalrx publish <run-dir>` — or open the run once with `evalrx serve` — then zip it again.",
    );
  }
  const dataPath = dataEntries[0];
  const reportDir = parentOf(dataPath);        // <root>/report
  const root = parentOf(reportDir);            // the run root ("" when zipped from inside it)
  const specPath = `${reportDir}/report_spec.json`;
  if (!files[specPath]) throw new Error(`The archive has ${dataPath} but no ${specPath} beside it.`);

  const data = JSON.parse(await files[dataPath].async("string"));
  const layout = JSON.parse(await files[specPath].async("string"));
  // Reports published before the explicit-props contract omitted `props` for
  // simple catalog elements (mirrors dynamic.load_published_report).
  const elements = layout?.spec?.elements;
  if (elements && typeof elements === "object") {
    for (const element of Object.values(elements) as any[]) {
      if (element && typeof element === "object" && !("props" in element)) element.props = {};
    }
  }

  // Same lookup order as the server's _resolve_media / _resolve_indexed_media:
  // the run root, its parent (the example dir), and the example's data/.
  const parent = parentOf(root);
  const grand = parentOf(parent);
  const candidates = (raw: string) => {
    const rel = raw.replace(/^\.\//, "");
    return [join(root, rel), join(parent, rel), join(join(parent, "data"), rel), join(join(grand, "data"), rel), rel];
  };
  const urls = new Map<string, string>();
  const resolve = async (raw: unknown): Promise<string | undefined> => {
    if (typeof raw !== "string" || !raw || /^[a-z][a-z0-9+.-]*:/i.test(raw) || raw.startsWith("/")) return undefined;
    for (const name of candidates(raw)) {
      const entry = files[name];
      if (!entry || entry.dir) continue;
      if (!urls.has(name)) {
        const ext = name.slice(name.lastIndexOf(".") + 1).toLowerCase();
        const bytes = await entry.async("arraybuffer");
        urls.set(name, URL.createObjectURL(new Blob([bytes], { type: MIME[ext] || "application/octet-stream" })));
      }
      return urls.get(name);
    }
    return undefined;
  };

  for (const item of data.media || []) {
    if (item && !item.data_uri) {
      const url = await resolve(item.path);
      if (url) item.data_uri = url;
    }
  }
  const attachFigures = async (node: any): Promise<void> => {
    if (Array.isArray(node)) { for (const value of node) await attachFigures(value); return; }
    if (!node || typeof node !== "object") return;
    const raw = node.path;
    if (typeof raw === "string" && !node.data_uri && FIGURE_SUFFIXES.some((s) => raw.toLowerCase().endsWith(s))) {
      const url = await resolve(raw);
      if (url) node.data_uri = url;
    }
    for (const value of Object.values(node)) await attachFigures(value);
  };
  // Figures are cited from several trees (stage briefs, findings, charts);
  // cases/media/debug are large and carry none, so they are skipped.
  for (const key of Object.keys(data)) {
    if (key === "cases" || key === "media" || key === "debug") continue;
    await attachFigures(data[key]);
  }

  const label = root.split("/").filter(Boolean).slice(-2).join("/") || file.name.replace(/\.zip$/i, "");
  return { label, data, layout };
}
