import { Link } from "react-router-dom";
import { N } from "../components/Num";
import { badgeKind } from "../components/HorizonBadges";
import checkerJson from "../data/checker.json";
import droughtJson from "../data/drought.json";
import heatJson from "../data/heat.json";
import type { Checker, Drought, Heat } from "../data/types";

const drought = droughtJson as unknown as Drought;
const heat = heatJson as unknown as Heat;
const checker = checkerJson as unknown as Checker;

export default function Landing() {
  const rows = drought.regions["rajasthan"] ?? [];
  const first = rows[0];
  const last = rows[rows.length - 1];

  return (
    <>
      <section className="hero" aria-labelledby="h-hero">
        <p className="eyebrow">Rajasthan and Barmer, India</p>
        <h1 id="h-hero">A climate-risk analyst that shows how far to trust each number.</h1>
        <p className="lede">
          It forecasts drought, reports observed heat and assesses crop impact. Every number in a written report is
          checked mechanically against a source, and every forecast month carries the test result it earned, including
          the months where the answer is "no skill".
        </p>
        <p className="muted">
          The demo runs with no key and no network: it serves six saved examples. Asking your own question needs live
          mode and your own Gemini key.
        </p>
        <div className="actions">
          <Link className="button" to="/analyst">Open the analyst</Link>
          <Link className="button ghost" to="/evidence">See the evidence</Link>
        </div>
      </section>

      <section className="block" aria-labelledby="h-products">
        <h2 id="h-products">Two products</h2>
        <div className="grid two">
          <Link to="/analyst" className="product">
            <span className="eyebrow">Climate Analyst</span>
            <h3>Forecasts with an honest label on every month</h3>
            <p className="muted">
              Drought forecast per horizon, observed heat, crop impact, a written report whose numbers are traced to
              their sources, and a plain list of what the system does not have.
            </p>
          </Link>
          <Link to="/slm" className="product">
            <span className="eyebrow">Climate SLM</span>
            <h3>A small model that answers only from the text it is given</h3>
            <p className="muted">
              Ask about the project's own report and source passages, or paste a passage. It answers from that text, or says
              it cannot. Experimental, and separate from the analyst.
            </p>
          </Link>
        </div>
      </section>

      <section className="block" aria-labelledby="h-how">
        <h2 id="h-how">How it works</h2>
        <ol className="steps">
          <li>
            <h3>Tools produce the numbers</h3>
            <p className="muted">Forecast, heat and crop tools compute the figures from the data. A model does not invent them.</p>
          </li>
          <li>
            <h3>A report is written from those outputs</h3>
            <p className="muted">The language model writes only from the tool outputs and the retrieved passages.</p>
          </li>
          <li>
            <h3>Every number is checked</h3>
            <p className="muted">A mechanical checker looks each number up in the sources and flags the ones it cannot find.</p>
          </li>
        </ol>
      </section>

      <section className="block" aria-labelledby="h-proof">
        <h2 id="h-proof">What was measured</h2>
        <p className="muted">Each figure is read from a saved result file; hover it for the file and place.</p>
        <div className="proofs">
          {first && (
            <div className="proof">
              <span className="big"><N n={first.published_skill} /></span>
              Rajasthan skill at {first.horizon}: {badgeKind(first.label).title}
            </div>
          )}
          {last && (
            <div className="proof">
              <span className="big"><N n={last.published_skill} /></span>
              Rajasthan skill at {last.horizon}: {badgeKind(last.label).title}
            </div>
          )}
          <div className="proof">
            <span className="big"><N n={checker.precision} /></span>
            Number-checker precision on <N n={checker.cases} /> hand-labelled cases (recall <N n={checker.recall} />)
          </div>
          <div className="proof">
            <span className="big"><N n={heat.extended_search.cells_tried} /></span>
            heat-forecast combinations tried; anything cleared the bar: {heat.extended_search.anything_clears_bar ? "yes" : "no"}
          </div>
        </div>
        <p style={{ marginTop: 24 }}>
          <Link to="/evidence">All the evidence, with sources</Link>
        </p>
      </section>
    </>
  );
}
