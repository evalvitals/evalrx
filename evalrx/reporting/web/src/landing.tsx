/**
 * The front page of the static (GitHub Pages) build: the demo reports this
 * site ships, and the drop zone for a reader's own zipped run.
 *
 * The demo list is read from ./demos.json beside the page rather than built
 * in, so adding a run is a JSON edit and a new exported HTML, not a rebuild.
 */
import { useEffect, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { RunDrop, type UploadedRun } from "./upload";

type Demo = { tag: string; title: string; blurb: string; href: string };

export function StaticLanding({ onLoaded }: { onLoaded: (run: UploadedRun) => void }) {
  const [demos, setDemos] = useState<Demo[]>([]);
  useEffect(() => {
    fetch("./demos.json", { cache: "no-cache" })
      .then((res) => (res.ok ? res.json() : []))
      .then((list) => setDemos(Array.isArray(list) ? list : []))
      .catch(() => setDemos([]));
  }, []);
  return <div className="landing">
    <header className="landing-head">
      <div className="landing-brand"><span className="logo-mark">EV</span><strong>EvalRX</strong></div>
      <h1>From failure to fix, with the evidence attached.</h1>
      <p>One loop diagnoses a model's failures (M1–M3), verifies each hypothesis on held-out cases (M4) and validates a repair on untouched ones (M5). These reports are its output.</p>
    </header>
    {demos.length > 0 && <section className="demo-grid" aria-label="Demo reports">
      {demos.map((demo) => <a key={demo.href} className="demo-card" href={demo.href}>
        <small>{demo.tag}</small>
        <b>{demo.title}<ArrowUpRight size={14} /></b>
        <span>{demo.blurb}</span>
      </a>)}
    </section>}
    <section className="landing-drop">
      <RunDrop onLoaded={onLoaded} />
    </section>
  </div>;
}
