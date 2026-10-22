import { Link } from "react-router-dom";
import SlmChat from "../components/SlmChat";
import { Quote, slm } from "../components/SlmTables";

export default function Slm() {
  return (
    <>
      <p className="eyebrow">Climate SLM</p>
      <h1>A small model that answers from the passage you give it.</h1>
      <div className="calm" data-testid="slm-status">
        <strong>{slm.status}</strong> It is a separate piece of work that answers a question from a given passage or
        says it cannot. Nothing on this page claims that it improves the climate analyst.
      </div>
      <p className="lede">
        Passage-grounded question answering: you supply the passage and the question, and the answer may use only
        that passage. If the passage does not contain the answer, the model is trained to refuse with one fixed
        sentence.
      </p>

      <section className="block">
        <SlmChat />
      </section>

      <section className="card" aria-labelledby="slm-adapter">
        <h2 id="slm-adapter">The published adapter</h2>
        <p>
          <a href={slm.adapter.url}>{slm.adapter.id}</a> on Hugging Face ({slm.adapter.licence}), a LoRA adapter for{" "}
          {slm.adapter.base_model}. {slm.adapter.statement}
        </p>
        <p className="small muted">
          Its evaluation results are quoted, with file and line, on the <Link to="/evidence">Evidence page</Link>.
        </p>
      </section>

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
