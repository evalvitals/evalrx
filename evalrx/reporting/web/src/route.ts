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
 */
export type Route =
  | { view: "overview" }
  | { view: "evidence"; stage?: string; full?: boolean }
  | { view: "cases"; caseId?: string; split?: string }
  | { view: "debug" };

export type Navigate = (route: Route, options?: { replace?: boolean }) => void;

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
    default:
      return "#/";
  }
}

/**
 * Move to `route`. A move between views adds a history entry, so Back returns
 * to the previous view; `replace` is for a selection within a view (another
 * stage, case or partition), which should not leave a trail of entries.
 */
export const navigate: Navigate = (route, { replace = false } = {}) => {
  const hash = routeHash(route);
  if (replace) window.location.replace(hash);
  else window.location.hash = hash;
};
