import checkerJson from "../data/checker.json";
import droughtJson from "../data/drought.json";
import heatJson from "../data/heat.json";
import trackaJson from "../data/tracka.json";
import type { Checker, Drought, Heat, TrackA } from "../data/types";
import { badgeKind } from "../components/HorizonBadges";
import { N, SrcLine } from "../components/Num";
import { SkillChart, SkillLegend } from "../components/SkillChart";
import { Table, slm } from "../components/SlmTables";
import { useState } from "react";

const drought = droughtJson as unknown as Drought;
const heat = heatJson as unknown as Heat;
const checker = checkerJson as unknown as Checker;
const tracka = trackaJson as unknown as TrackA;

const TABS = [
  { id: "drought", label: "Drought forecasting" },
  { id: "heat", label: "Heat (null result)" },
  { id: "retrieval", label: "Retrieval (Track A)" },
  { id: "checker", label: "Grounding checker" },
  { id: "slm", label: "SLM" },
] as const;
type TabId = (typeof TABS)[number]["id"];

export default function Evidence() {
  const [tab, setTab] = useState<TabId>("drought");
  const noReproduction = drought.reproduced_file === null;
  return (
    <>
      <div className="page-head">
        <p className="eyebrow">Evidence</p>
        <h1>What was measured, and where it was read from.</h1>
        <p className="lede">
          Every number on this page is read from a saved result file by a script, and the file is named next
          to it. Hover a number to see the exact place it came from.
        </p>
      </div>

      <div className="tabs" role="tablist" aria-label="Evidence sections">
        {TABS.map((x) => (
          <button key={x.id} role="tab" id={`tab-${x.id}`} aria-selected={tab === x.id} aria-controls="evidence-panel"
            onClick={() => setTab(x.id)}>
            {x.label}
          </button>
        ))}
      </div>
      <div id="evidence-panel" role="tabpanel" aria-labelledby={`tab-${tab}`}>
      {tab === "drought" && (
      <section className="card" aria-labelledby="e-drought">
        <h2 id="e-drought">Drought forecast: skill per horizon</h2>
        <p className="muted">
          Skill score is 1 minus the model's error divided by the error of always predicting the seasonal
          normal: 0 means no better than the normal, higher is better. The right-hand columns are the
          values reproduced on this clone by <code>python -m scripts.setup</code>.
        </p>
        <div className="chart-set" style={{ margin: "16px 0 4px" }}>
          {Object.entries(drought.regions).map(([region, rows]) => (
            <SkillChart key={region} region={region} rows={rows} />
          ))}
        </div>
        <SkillLegend />
        {Object.entries(drought.regions).map(([region, rows]) => (
          <div key={region} style={{ marginBottom: 16 }}>
            <h3 style={{ textTransform: "capitalize", marginTop: 20 }}>{region}</h3>
            <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Horizon</th>
                  <th>Verdict</th>
                  <th className="num">Published skill</th>
                  <th className="num">Published error</th>
                  <th className="num">Reproduced skill</th>
                  <th className="num">Reproduced error</th>
                  <th>Same?</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.horizon}>
                    <td>{r.horizon}</td>
                    <td>{badgeKind(r.label).title}</td>
                    <td className="num"><N n={r.published_skill} /></td>
                    <td className="num"><N n={r.published_rmse} /></td>
                    <td className="num"><N n={r.reproduced_skill} /></td>
                    <td className="num"><N n={r.reproduced_rmse} /></td>
                    <td>
                      {r.identical_to_published === null
                        ? "not reproduced yet"
                        : r.identical_to_published
                          ? "identical"
                          : "differs"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        ))}
        <SrcLine file={drought.published_file} />
        {drought.reproduced_file ? (
          <SrcLine file={drought.reproduced_file} />
        ) : (
          <p className="calm">
            {noReproduction && "No reproduced values on this clone yet: run python -m scripts.setup, then npm run sync-evidence."}
          </p>
        )}
      </section>

      )}
      {tab === "heat" && (
      <section className="card" aria-labelledby="e-heat">
        <h2 id="e-heat">Heat stress: a null result</h2>
        <p>
          Forecasting heat stress was tested and found to have no usable skill, so the system reports what
          was observed and offers no forecast.
        </p>
        {Object.entries(heat.regions).map(([region, rows]) => (
          <div key={region}>
            <h3 style={{ textTransform: "capitalize" }}>{region}</h3>
            <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Horizon</th><th>Verdict</th><th className="num">Skill</th></tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.horizon}>
                    <td>{r.horizon}</td>
                    <td>{badgeKind(r.label).title}</td>
                    <td className="num"><N n={r.skill} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        ))}
        <SrcLine file={heat.manifest_file} />
        <p style={{ marginTop: 12 }}>
          A wider search tried <N n={heat.extended_search.cells_tried} /> combinations. The highest-scoring one
          ({heat.extended_search.best_cell.region}, {heat.extended_search.best_cell.target},{" "}
          {heat.extended_search.best_cell.feature_set}, {heat.extended_search.best_cell.horizon}) reached a
          skill of <N n={heat.extended_search.best_cell.skill} />. Anything cleared the bar:{" "}
          <strong>{heat.extended_search.anything_clears_bar ? "yes" : "no"}</strong>.
        </p>
        <SrcLine file={heat.extended_search.file} />
      </section>

      )}
      {tab === "checker" && (
      <section className="card" aria-labelledby="e-check">
        <h2 id="e-check">The number checker</h2>
        <p className="muted">{checker.what_this_measures}</p>
        <div className="table-wrap">
          <table>
            <tbody>
              <tr><th scope="row">Test cases</th><td className="num"><N n={checker.cases} /></td></tr>
              <tr><th scope="row">Numbers labelled</th><td className="num"><N n={checker.numbers_labelled} /></td></tr>
              <tr><th scope="row">Invented numbers caught</th><td className="num"><N n={checker.true_positives} /></td></tr>
              <tr><th scope="row">Invented numbers missed</th><td className="num"><N n={checker.false_negatives} /></td></tr>
              <tr><th scope="row">Good numbers wrongly flagged</th><td className="num"><N n={checker.false_positives} /></td></tr>
              <tr><th scope="row">Precision</th><td className="num"><N n={checker.precision} /></td></tr>
              <tr><th scope="row">Recall</th><td className="num"><N n={checker.recall} /></td></tr>
              <tr><th scope="row">F1</th><td className="num"><N n={checker.f1} /></td></tr>
            </tbody>
          </table>
        </div>
        <SrcLine file={checker.file} />
        <div className="calm" style={{ marginTop: 12 }}>
          <strong>Known defect.</strong> {checker.known_defect.text}
          <p className="src" style={{ margin: "6px 0 0" }}>
            source: {checker.known_defect.source_file}, line {checker.known_defect.line.value}
          </p>
        </div>
      </section>

      )}
      {tab === "retrieval" && (
      <section className="card" aria-labelledby="e-tracka">
        <h2 id="e-tracka">Track A: finding the right passage</h2>
        <p className="calm">All Track A numbers were {tracka.corpus_note}.</p>
        <h3>Easy test (document level)</h3>
        <p>
          <N n={tracka.easy.queries} /> questions, top <N n={tracka.easy.k} />: precision{" "}
          <N n={tracka.easy.precision_at_k} />, mean reciprocal rank <N n={tracka.easy.mrr} />{" "}
          <span className="muted">({tracka.easy.note})</span>
        </p>
        <SrcLine file={tracka.easy.file} />
        <h3>Strict test (chunk level, human-labelled passages)</h3>
        <p className="muted small">{tracka.strict.metric_note}</p>
        {(["soft", "hard"] as const).map((set) => (
          <div key={set} style={{ marginBottom: 12 }}>
            <h3>{set === "soft" ? "Regular question set" : "Hard question set"}</h3>
            <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Search</th><th className="num">Questions</th><th className="num">Precision at 5</th><th className="num">MRR</th></tr>
              </thead>
              <tbody>
                {Object.entries(tracka.strict[set]).map(([cfg, a]) => (
                  <tr key={cfg}>
                    <td>{cfg}</td>
                    <td className="num"><N n={a.queries} /></td>
                    <td className="num"><N n={a.precision_at_5} /></td>
                    <td className="num"><N n={a.mrr} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        ))}
        <p className="small muted">{tracka.strict.pooling_assumption}</p>
        <SrcLine file={tracka.strict.file} />
        <p>
          <strong>{tracka.default_search.statement}</strong>{" "}
          <span className="src">
            ({tracka.default_search.source_file}, line {tracka.default_search.line.value})
          </span>
        </p>
      </section>
      )}
      {tab === "slm" && (
        <>
          <p className="calm">
            Quoted from the SLM repository's README with the file and line next to each row. {slm.test_split.text}{" "}
            <span className="src">({slm.test_split.source_file}, line {slm.test_split.line.value})</span>
          </p>
          <p className="small muted">
            The README does not say which saved checkpoint produced each row, so a row is not claimed to be exactly the
            published adapter's file. Only the 1.5B SFT adapter is published.
          </p>
          <Table title="Model quality" t={slm.model_quality} notes />
          <Table title="Does the service answer like the evaluator?" t={slm.parity} notes />
          <Table title="Serving speed: vLLM against the hf backend" t={slm.benchmark} notes />
          <Table title="Start-up time" t={slm.startup} />
        </>
      )}
      </div>
    </>
  );
}
