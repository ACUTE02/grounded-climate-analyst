import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getExample } from "../api/client";
import type { ReportResponse } from "../api/types";
import HorizonBadges, { badgeKind } from "../components/HorizonBadges";
import { N } from "../components/Num";
import { HeroScene, IconBadge, Icons } from "../components/Art";
import { SkillChart, SkillLegend } from "../components/SkillChart";
import checkerJson from "../data/checker.json";
import droughtJson from "../data/drought.json";
import heatJson from "../data/heat.json";
import type { Checker, Drought, Heat } from "../data/types";

const drought = droughtJson as unknown as Drought;
const heat = heatJson as unknown as Heat;
const checker = checkerJson as unknown as Checker;

const PREVIEW_ID = "drought_one_month";

/** A real saved answer from the backend, so the first screen shows what the product returns (not a mock-up). */
function Preview() {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "unavailable">("loading");

  useEffect(() => {
    getExample(PREVIEW_ID)
      .then((r) => {
        // Show the preview only for a well-formed saved answer; anything else is treated as unavailable.
        if (typeof r?.request === "string" && Array.isArray(r.horizon_confidence)) {
          setReport(r);
          setState("ready");
        } else {
          setState("unavailable");
        }
      })
      .catch(() => setState("unavailable"));
  }, []);

  const months: Record<string, string> = {};
  for (const h of (report?.tool_outputs?.["forecast_drought_risk"]?.horizon_confidence ?? []) as { horizon?: string; month?: string }[]) {
    if (h.horizon && h.month) months[h.horizon] = h.month;
  }

  return (
    <aside className="card preview" aria-label="Preview of a saved answer" data-testid="landing-preview">
      <div className="preview-bar">
        <span>Saved example, written by the live pipeline</span>
        {report && <span className={`status ${report.grounding.status}`}>number check: {report.grounding.status}</span>}
      </div>
      <div className="preview-body">
        {state === "loading" && <div className="skeleton" aria-busy="true" aria-label="Loading the saved example" style={{ minHeight: 190 }} />}
        {state === "unavailable" && (
          <p className="muted small" style={{ margin: 0 }}>
            The preview is a real saved answer from the backend, and the backend is not reachable. Start it (see the README) to see it
            here. Everything else on this page works without it.
          </p>
        )}
        {report && (
          <>
            <p className="q">{report.request}</p>
            <HorizonBadges horizons={report.horizon_confidence} months={months} compact />
            <p style={{ margin: "16px 0 0" }}>
              <Link to={`/analyst/report/${PREVIEW_ID}`}>Open the full report</Link>
            </p>
          </>
        )}
      </div>
    </aside>
  );
}

export default function Landing() {
  const rows = drought.regions["rajasthan"] ?? [];
  const first = rows[0];
  const last = rows[rows.length - 1];

  return (
    <>
      <section className="hero" aria-labelledby="h-hero">
        <div>
          <p className="eyebrow">Rajasthan and Barmer, India</p>
          <h1 id="h-hero">A climate-risk analyst that shows how far to trust each number.</h1>
          <p className="lede">
            It forecasts drought, reports observed heat and assesses crop impact. Every forecast month carries the
            test result it earned, including the months where the answer is "no skill", and every number in a written
            report is checked against a source by plain code.
          </p>
          <p className="small">
            For analysts, extension staff and students who need a reading and its reliability. A research and
            demonstration project, not an early-warning service. The demo runs with no key and no network and serves six
            saved examples; asking your own question needs live mode and your own Gemini key.
          </p>
          <div className="actions">
            <Link className="button" to="/analyst">Open the analyst</Link>
            <Link className="button ghost" to="/evidence">See the evidence</Link>
          </div>
        </div>
        <Preview />
        <HeroScene />
      </section>

      <section className="block" aria-labelledby="h-how">
        <div className="section-head"><h2 id="h-how">How it works</h2></div>
        <ol className="steps">
          <li>
            <IconBadge>{Icons.gauge}</IconBadge>
            <h3>Tools produce the numbers</h3>
            <p className="muted">Forecast, heat and crop tools compute figures from the data. A language model does not invent them.</p>
          </li>
          <li>
            <IconBadge>{Icons.report}</IconBadge>
            <h3>A report is written from those outputs</h3>
            <p className="muted">The model writes only from the tool outputs and the retrieved source passages.</p>
          </li>
          <li>
            <IconBadge>{Icons.check}</IconBadge>
            <h3>Every number is checked</h3>
            <p className="muted">A checker looks each number up in the sources and flags the ones it cannot find. A flagged report gets one rewrite, then a warning.</p>
          </li>
        </ol>
      </section>

      <section className="block" aria-labelledby="h-proof">
        <div className="section-head">
          <h2 id="h-proof">What was measured</h2>
          <Link to="/evidence" className="small">All evidence, with sources</Link>
        </div>
        <div className="card">
          <div className="card-head">
            <h3>Forecast skill by horizon</h3>
            <span className="small faint">0 = no better than predicting the seasonal normal</span>
          </div>
          <div className="chart-set">
            {Object.entries(drought.regions).map(([region, r]) => (
              <SkillChart key={region} region={region} rows={r} />
            ))}
          </div>
          <SkillLegend />
        </div>
        <div className="proofs">
          {first && (
            <div className="proof">
              <span className="big"><N n={first.published_skill} /></span>
              <span className="cap">Rajasthan skill at {first.horizon}: {badgeKind(first.label).title}</span>
            </div>
          )}
          {last && (
            <div className="proof">
              <span className="big"><N n={last.published_skill} /></span>
              <span className="cap">Rajasthan skill at {last.horizon}: {badgeKind(last.label).title}</span>
            </div>
          )}
          <div className="proof">
            <span className="big"><N n={checker.precision} /></span>
            <span className="cap">Number-checker precision on <N n={checker.cases} /> hand-labelled cases (recall <N n={checker.recall} />)</span>
          </div>
          <div className="proof">
            <span className="big"><N n={heat.extended_search.cells_tried} /></span>
            <span className="cap">heat-forecast combinations tried; anything cleared the bar: {heat.extended_search.anything_clears_bar ? "yes" : "no"}</span>
          </div>
        </div>
      </section>

      <section className="block" aria-labelledby="h-products">
        <div className="section-head"><h2 id="h-products">Two products</h2></div>
        <div className="grid two">
          <Link to="/analyst" className="product">
            <span className="head"><IconBadge>{Icons.sun}</IconBadge><span className="eyebrow">Climate Analyst</span></span>
            <h3>Forecasts with an honest label on every month</h3>
            <p className="muted">
              Drought forecast per horizon, observed heat, crop impact, a written report whose numbers are traced to
              their sources, and a plain list of what the system does not have.
            </p>
            <span className="go">Open the analyst</span>
          </Link>
          <Link to="/slm" className="product">
            <span className="head"><IconBadge>{Icons.chat}</IconBadge><span className="eyebrow">Climate SLM</span></span>
            <h3>A small model that answers only from the text it is given</h3>
            <p className="muted">
              Ask about the project's own report and source passages, or paste a passage. It answers from that text, or
              says it cannot. Experimental, and separate from the analyst.
            </p>
            <span className="go">Open the SLM chat</span>
          </Link>
        </div>
      </section>
    </>
  );
}
