import { useEffect, useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError, getExamples, postReport } from "../api/client";
import type { DemoRefusal, ExampleItem } from "../api/types";
import { useShell } from "../components/context";

const KIND_TEXT: Record<string, string> = {
  live_generation: "written by the live pipeline",
  tools_only: "tool output, no written report",
  constructed_fixture: "built to show the number checker",
  validators_only: "refusals from the system's own checks",
};

export default function Home() {
  const { health, offline } = useShell();
  const navigate = useNavigate();
  const [examples, setExamples] = useState<ExampleItem[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState<DemoRefusal | null>(null);
  const [askError, setAskError] = useState<string | null>(null);

  useEffect(() => {
    getExamples()
      .then((r) => setExamples(r.examples))
      .catch((e: unknown) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, []);

  const demo = health?.mode !== "live";

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setRefusal(null);
    setAskError(null);
    try {
      const report = await postReport({ request: question.trim() });
      navigate("/report/live", { state: { report } });
    } catch (e) {
      if (e instanceof ApiError && e.demoRefusal) setRefusal(e.demoRefusal);
      else setAskError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section style={{ marginBottom: "var(--gap)" }}>
        <h1>How far can a climate forecast be trusted?</h1>
        <p className="muted" style={{ maxWidth: "62ch" }}>
          This analyst covers drought, heat stress and crop impact for Rajasthan and Barmer. Every
          forecast month carries its own measured test result, every number in a written report is
          checked against a source, and what the system does not know is stated, not hidden.
        </p>
      </section>

      <section aria-labelledby="h-examples">
        <h2 id="h-examples">Saved examples</h2>
        {loadError && (
          <div className="error" role="alert">
            Could not load the examples: {loadError}
          </div>
        )}
        {!examples && !loadError && !offline && <p className="muted">Loading…</p>}
        <div className="grid">
          {examples?.map((ex) => (
            <Link key={ex.id} to={`/report/${ex.id}`} className="example-card">
              <h3>{ex.label}</h3>
              <span className="muted small">{ex.shows}</span>
              {ex.kind && (
                <span className="pill">{KIND_TEXT[ex.kind] ?? ex.kind}</span>
              )}
            </Link>
          ))}
        </div>
      </section>

      <section className="card" style={{ marginTop: "var(--gap)" }} aria-labelledby="h-ask">
        <h2 id="h-ask">Ask your own question</h2>
        <form onSubmit={ask}>
          <label htmlFor="q" className="muted small">
            Your question
          </label>
          <textarea
            id="q"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={demo || busy}
            placeholder={
              demo
                ? "Free-text questions are off in demo mode."
                : "For example: what is the drought risk for Barmer?"
            }
          />
          <p style={{ marginTop: 12 }}>
            <button type="submit" disabled={demo || busy || !question.trim()}>
              {busy ? "Working…" : "Ask"}
            </button>
          </p>
        </form>
        {demo && (
          <div className="calm" data-testid="demo-explanation">
            Demo mode answers only the saved examples above, and makes no key or network call. To ask
            your own question, run the backend in live mode with your own Gemini key.
          </div>
        )}
        {refusal && (
          <div className="calm" role="status" data-testid="refusal">
            <p>{refusal.message}</p>
            <p className="small muted" style={{ margin: 0 }}>
              Try one of the saved examples:{" "}
              {refusal.examples.map((id, i) => (
                <span key={id}>
                  {i > 0 && ", "}
                  <Link to={`/report/${id}`}>{id}</Link>
                </span>
              ))}
            </p>
          </div>
        )}
        {askError && (
          <div className="error" role="alert">
            {askError}
          </div>
        )}
      </section>
    </>
  );
}
