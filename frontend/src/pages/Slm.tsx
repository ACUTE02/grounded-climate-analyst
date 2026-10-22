import SlmTryIt from "../components/SlmTryIt";
import slmJson from "../data/slm.json";
import type { Cite, Slm as SlmData, SlmTable } from "../data/types";

const slm = slmJson as unknown as SlmData;

function Quote({ c }: { c: Cite }) {
  return (
    <li>
      {c.text}{" "}
      <span className="src">
        ({c.source_file}, line {c.line.value})
      </span>
    </li>
  );
}

// Which rows have a published adapter: only the 1.5B SFT one. Everything else is quoted for context.
function adapterNote(model: string): string {
  if (model.startsWith("1.5B SFT")) return "adapter published";
  if (model.includes("untuned")) return "no adapter (base model)";
  if (model.includes("0.5B")) return "adapter not in this release";
  return "";
}

function Table({ title, t, notes = false }: { title: string; t: SlmTable; notes?: boolean }) {
  return (
    <section className="card" aria-label={title}>
      <h2>{title}</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {t.columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
              {notes && <th>Adapter</th>}
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {t.rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell) => (
                  <td key={cell.column} className={cell.column === row[0].column ? "" : "num"}>
                    {cell.text.replace(/\*\*/g, "")}
                  </td>
                ))}
                {notes && <td className="small muted">{adapterNote(row[0].text.replace(/\*\*/g, ""))}</td>}
                <td className="src">
                  {slm.source_file}:{row[0].line.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function Slm() {
  return (
    <>
      <h1>The small language model</h1>
      <div className="calm" data-testid="slm-status">
        <strong>{slm.status}</strong> It is a separate piece of work that answers a question from a given
        passage or says it cannot. Nothing on this page claims that it improves the climate analyst.
      </div>
      <p className="muted" style={{ marginTop: 16 }}>
        These results are quoted from the SLM repository's README, with the file and line next to each row.
      </p>
      <p>{slm.test_split.text}{" "}
        <span className="src">({slm.test_split.source_file}, line {slm.test_split.line.value})</span>
      </p>
      <section className="card" aria-labelledby="slm-adapter">
        <h2 id="slm-adapter">The published adapter</h2>
        <p>
          <a href={slm.adapter.url}>{slm.adapter.id}</a> on Hugging Face ({slm.adapter.licence}), a LoRA
          adapter for {slm.adapter.base_model}. {slm.adapter.statement}
        </p>
        <p className="small muted">
          Saved examples: there are none yet, because a real model run has not been saved. The results below
          are quoted evaluation results; the card underneath calls the live service only when you press Ask.
        </p>
        <p className="small muted">
          The README does not say which saved checkpoint produced each row of the table below, so a row is not
          claimed to be exactly the published adapter's file.
        </p>
      </section>
      <SlmTryIt />
      <Table title="Model quality" t={slm.model_quality} notes />
      <Table title="Does the service answer like the evaluator?" t={slm.parity} notes />
      <Table title="Serving speed: vLLM against the hf backend" t={slm.benchmark} notes />
      <Table title="Start-up time" t={slm.startup} />
      <section className="card" aria-labelledby="slm-limits">
        <h2 id="slm-limits">Limits, in the README's words</h2>
        <ul>
          {slm.limits.map((c) => (
            <Quote key={c.line.value} c={c} />
          ))}
          <Quote c={slm.laptop_smoke_test} />
        </ul>
      </section>
    </>
  );
}
