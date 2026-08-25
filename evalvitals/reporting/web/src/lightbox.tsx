/**
 * Opening any figure full-screen, from anywhere in the report.
 *
 * Every image here is evidence: an analyzer's chart with seven axis labels and
 * a value beside each point, or the actual input a case was scored on. At the
 * size they sit on the page — two to a row, a third of the column — none of
 * that is legible, and the reader's only recourse was the browser's own
 * zoom, which enlarges the layout rather than the picture.
 *
 * One click, one size: the figure fills the viewport, centred. There is no
 * zoom control — these are a few hundred kilobytes of chart at a size the
 * screen can hold, and a fit/actual toggle was a second decision to make
 * before reading the thing.
 *
 * The store is module-level rather than a context because the three places that
 * render images (the overview catalog, the stage brief, the full record) do not
 * share a parent below `App`, and threading a callback through all of them
 * would put the plumbing in every signature for the sake of one overlay.
 */
import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import { X } from "lucide-react";

type Shot = { src: string; caption: string } | null;

let current: Shot = null;
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

/** Show *src* full-screen. `caption` names what the reader is looking at. */
export function openLightbox(src: string, caption = "") {
  current = { src, caption };
  emit();
}

export function closeLightbox() {
  if (!current) return;
  current = null;
  emit();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * An image that opens itself. Drop-in for `<img>` — same props, plus the
 * caption the overlay should carry.
 */
export function ZoomableImage(
  { src, alt, caption, className }:
  { src: string; alt: string; caption?: string; className?: string },
) {
  return <img
    className={`zoomable ${className || ""}`.trim()}
    src={src}
    alt={alt}
    // A picture that opens on click has to say so to a keyboard as well: this
    // is the control, so it takes the role and the tab stop rather than being
    // wrapped in a button that would inherit neither the sizing nor the caption.
    role="button"
    tabIndex={0}
    title="Open full size"
    onClick={() => openLightbox(src, caption || alt)}
    onKeyDown={(event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      openLightbox(src, caption || alt);
    }}
  />;
}

/** The overlay itself. Mounted once, near the root. */
export function Lightbox() {
  const shot = useSyncExternalStore(subscribe, () => current, () => null);
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!shot) return;
    closeButton.current?.focus();
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") closeLightbox(); };
    // The page behind must not scroll away under the overlay.
    const scrollLock = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = scrollLock;
    };
  }, [shot]);

  const stop = useCallback((event: React.MouseEvent) => event.stopPropagation(), []);
  if (!shot) return null;

  return <div
    className="lightbox"
    role="dialog"
    aria-modal="true"
    aria-label={shot.caption || "Full size figure"}
    onClick={closeLightbox}
  >
    <div className="lightbox-bar" onClick={stop}>
      <span>{shot.caption}</span>
      <button type="button" ref={closeButton} onClick={closeLightbox} title="Close (Esc)">
        <X size={15} /> Close
      </button>
    </div>
    <div className="lightbox-frame" onClick={stop}>
      <img src={shot.src} alt={shot.caption || "Figure"} />
    </div>
  </div>;
}
