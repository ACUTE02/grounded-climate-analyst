import { useEffect, useState, type FormEvent } from "react";
import { getSlmStatus, postSlmAnswer } from "../api/client";
import type { SlmAnswer, SlmStatus } from "../api/types";

export const MAX_CONTEXT = 4000;
export const MAX_QUESTION = 500;

// Written for this card; it is not taken from the project's data or sources.
const EXAMPLE_PASSAGE =
  "Barmer is a district in the west of Rajasthan, on the border with Pakistan. " +
  "Much of the district lies in the Thar desert, and most farms depend on rain.";
const EXAMPLE_QUESTION = "In which state is Barmer?";

export default function SlmTryIt() {
  const [status, setStatus] = useState<SlmStatus | null>(null);
  const [context, setContext] = useState(EXAMPLE_PASSAGE);
  const [question, setQuestion] = useState(EXAMPLE_QUESTION);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<SlmAnswer | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSlmStatus()
      .then(setStatus)
      .catch(() => setStatus({ configured: false, available: false, model: null, message: "" }));
  }, []);

  const available = status?.available === true;
  const tooLong = context.length > MAX_CONTEXT || question.length > MAX_QUESTION;
  const ready = available && !busy && !tooLong && context.trim() !== "" && question.trim() !== "";

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!ready) return;
    setBusy(true);
    setResult(null);
    setError(null);
    try {
      setResult(await postSlmAnswer(context, question));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card" aria-labelledby="slm-try" data-testid="slm-tryit">
      <h2 id="slm-try">Try it</h2>
      <p className="small muted">
        A small model (experimental). It answers only from the passage you give it, or says it cannot.
        It can be wrong, and nothing here measures how often.
      </p>
      {status && !available && (
        <div className="calm" data-testid="slm-unavailable">
          <p>
            <strong>The SLM service is not available, so this card is off.</strong>{" "}
            {status.configured ? status.message : "It is optional and not started by default."}
          </p>
          <p className="small" style={{ margin: 0 }}>
            To use it: start the SLM service on port 8001 (<code>PHASE5_PORT=8001</code>), then start the
            backend with <code>SLM_URL=http://127.0.0.1:8001</code>.
          </p>
        </div>
      )}
      <form onSubmit={ask}>
        <label htmlFor="slm-context" className="muted small">
          Passage ({context.length}/{MAX_CONTEXT} characters)
        </label>
        <textarea
          id="slm-context"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          disabled={!available || busy}
        />
        <label htmlFor="slm-question" className="muted small">
          Question ({question.length}/{MAX_QUESTION} characters)
        </label>
        <input
          id="slm-question"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={!available || busy}
          style={{ width: "100%" }}
        />
        {tooLong && (
          <div className="error" role="alert">
            The passage may have at most {MAX_CONTEXT} characters and the question {MAX_QUESTION}.
          </div>
        )}
        <p style={{ marginTop: 12 }}>
          <button type="submit" disabled={!ready}>
            {busy ? "Working…" : "Ask"}
          </button>
        </p>
      </form>
      {result && (
        <div className="calm" data-testid="slm-result">
          <p style={{ marginTop: 0 }}>{result.answer}</p>
          <p className="small muted" style={{ margin: 0 }}>
            refused: {String(result.refused)} · model: {result.model ?? "not reported"}
          </p>
        </div>
      )}
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}
