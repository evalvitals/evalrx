/**
 * Which view of the report is on screen, kept in the URL hash.
 *
 * The hash rather than the path: the report is also a single HTML file opened
 * from disk (`evalrx report`), where there is no server to answer a path, and
 * a hash survives both. So a refresh keeps the reader where they were, and a
 * copied link opens the same stage or case.
 *
 *   #/                      overview
 *   #/evidence/m3           a stage's brief;  #/evidence/m3/full  its full record
 *   #/cases/<id>?split=E    the case studio, on one case and/or one partition
 *   #/debug                 the raw event log
 *   #/compare/<id>          this run beside another one the server holds
 *
 * On a served report the hash also names the run (`?run=<id>`), so a reload
 * or a link reopens the run that was on screen, not the server's start-up run.
 */
export type Route =
  | { view: "overview" }
  | { view: "evidence"; stage?: string; full?: boolean }
  | { view: "cases"; caseId?: string; split?: string }
  | { view: "debug" }
  | { view: "compare"; with?: string };

export type Navigate = (route: Route, options?: { replace?: boolean; run?: string | null }) => void;

export function parseRoute(hash: string): Route {
  const [path, query = ""] = hash.replace(/^#\/?/, "").split("?");
  const [view, ...rest] = path.split("/").map(decodeURIComponent);
  if (view === "evidence") {
    return {
      view,
      ...(rest[0] ? { stage: rest[0] } : {}),
      ...(rest[1] === "full" ? { full: true } : {}),
    };
  }
  if (view === "cases") {
    const split = new URLSearchParams(query).get("split");
    return { view, ...(rest[0] ? { caseId: rest[0] } : {}), ...(split !== null ? { split } : {}) };
  }
  if (view === "debug") return { view };
  if (view === "compare") return { view, ...(rest[0] ? { with: rest[0] } : {}) };
  return { view: "overview" };
}

export function routeHash(route: Route): string {
  const part = encodeURIComponent;
  switch (route.view) {
    case "evidence":
      return `#/evidence${route.stage ? `/${part(route.stage)}${route.full ? "/full" : ""}` : ""}`;
    case "cases": {
      const query =
        route.split !== undefined ? `?${new URLSearchParams({ split: route.split })}` : "";
      return `#/cases${route.caseId ? `/${part(route.caseId)}` : ""}${query}`;
    }
    case "debug":
      return "#/debug";
    case "compare":
      return `#/compare${route.with ? `/${part(route.with)}` : ""}`;
    default:
      return "#/";
  }
}

/** The server-side run id a hash names, or null. */
export function runFromHash(hash: string): string | null {
  return new URLSearchParams(hash.split("?")[1] ?? "").get("run") || null;
}

/** *hash* naming run *runId* (or no run, for null). */
export function withRun(hash: string, runId: string | null): string {
  const [path, query = ""] = hash.split("?");
  const params = new URLSearchParams(query);
  params.delete("run");
  if (runId) params.set("run", runId);
  const rest = params.toString();
  return rest ? `${path}?${rest}` : path;
}

/**
 * Move to `route`. A move between views adds a history entry, so Back returns
 * to the previous view; `replace` is for a selection within a view (another
 * stage, case or partition), which should not leave a trail of entries.
 * The run the hash names is kept unless `run` says otherwise.
 */
export const navigate: Navigate = (route, { replace = false, run } = {}) => {
  const hash = withRun(
    routeHash(route),
    run === undefined ? runFromHash(window.location.hash) : run,
  );
  if (replace) window.location.replace(hash);
  else window.location.hash = hash;
};
