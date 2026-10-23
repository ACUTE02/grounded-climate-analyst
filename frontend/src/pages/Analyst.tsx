import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError, getExamples, postReport } from "../api/client";
import type { DemoRefusal, ExampleItem, ExamplesResponse, Health } from "../api/types";
import { IconBadge, Icons } from "../components/Art";
import { useShell } from "../components/context";

const KIND_ICON = {
  live_generation: Icons.report,
  tools_only: Icons.bars,
  constructed_fixture: Icons.check,
  validators_only: Icons.check,
} as const;

const KIND_TEXT: Record<string, string> = {
  live_generation: "written by the live pipeline",
  tools_only: "tool output, no written report",
  constructed_fixture: "built to show the number checker",
  validators_only: "refusals from the system's own checks",
};

function dataThrough(health: Health | null): string {
  const values = Object.values(health?.data_currency ?? {})
    .map((v) => v?.data_current_through)
    .filter((v): v is string => typeof v === "string");
  return values.length ? [...new Set(values)].join(", ") : "not reported";
}

/** What the running backend says about itself: real values from /health and /examples, nothing assumed. */
function StatusPanel({ health, meta }: { health: Health | null; meta: ExamplesResponse | null }) {
  const quota = health?.quota ?? meta?.quota;
  return (
    <aside className="card" aria-labelledby="h-status" data-testid="status-panel">
      <div className="card-head">
        <h2 id="h-status" style={{ fontSize: "1rem" }}>System status</h2>
        {health && <span className={`pill ${health.status === "ok" ? "ok" : "warm"}`}>{health.status}</span>}
      </div>
      {!health ? (
        <p className="muted small">Waiting for the backend. If this stays empty, the backend is not running (see the README).</p>
      ) : (
        <dl className="kv small">
          <dt>Mode</dt><dd>{health.mode}</dd>
          <dt>Gemini key</dt><dd>{health.api_key_present ? "present" : "not set"}</dd>
          <dt>Data through</dt><dd className="num">{dataThrough(health)}</dd>
          <dt>Quota today</dt>
          <dd>
            {quota?.calls_remaining_today !== undefined && quota?.daily_call_budget !== undefined
              ? `${quota.calls_remaining_today} of ${quota.daily_call_budget} calls left`
              : "not reported"}
          </dd>
          {meta && (<><dt>Regions</dt><dd>{meta.supported_regions.join(", ")}</dd><dt>Crops</dt><dd>{meta.supported_crops.join(", ")}</dd></>)}
        </dl>
      )}
    </aside>
  );
}

export default function Analyst() {
  const { health, offline } = useShell();
  const navigate = useNavigate();
  const [meta, setMeta] = useState<ExamplesResponse | null>(null);
  const [examples, setExamples] = useState<ExampleItem[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState<DemoRefusal | null>(null);
  const [askError, setAskError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoadError(null);
    setExamples(null);
    getExamples()
      .then((r) => {
        setMeta(r);
        setExamples(r.examples);
      })
      .catch((e: unknown) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(load, [load]);

  const demo = health?.mode !== "live";

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setRefusal(null);
    setAskError(null);
    try {
      const report = await postReport({ request: question.trim() });
      navigate("/analyst/report/live", { state: { report } });
    } catch (e) {
      if (e instanceof ApiError && e.demoRefusal) setRefusal(e.demoRefusal);
      else setAskError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <p className="eyebrow">Climate Analyst</p>
        <h1>Ask about drought, heat and crops.</h1>
        <p className="lede">
          Covers Rajasthan and Barmer. Each forecast month shows its own measured test result, and a written report
          lists the numbers it could not trace.
        </p>
      </div>

      <div className="analyst-grid">
        <div>
          <section className="card" aria-labelledby="h-ask">
            <h2 id="h-ask">Ask your own question</h2>
            <form onSubmit={ask}>
              <label htmlFor="q" style={{ marginTop: 0 }}>Your question</label>
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
              <p style={{ margin: "12px 0 0" }}>
                <button type="submit" disabled={demo || busy || !question.trim()}>
                  {busy && <span className="spinner" aria-hidden="true" />}
                  {busy ? "Working…" : "Ask"}
                </button>
              </p>
            </form>
            <div aria-live="polite">
              {demo && (
                <div className="calm" data-testid="demo-explanation" style={{ marginTop: 16 }}>
                  Demo mode answers only the saved examples below, and makes no key or network call. To ask your own
                  question, run the backend in live mode with your own Gemini key (see the README).
                </div>
              )}
              {refusal && (
                <div className="calm" role="status" data-testid="refusal" style={{ marginTop: 16 }}>
                  <p>{refusal.message}</p>
                  <p className="small muted" style={{ margin: 0 }}>
                    Try one of the saved examples:{" "}
                    {refusal.examples.map((id, i) => (
                      <span key={id}>
                        {i > 0 && ", "}
                        <Link to={`/analyst/report/${id}`}>{id}</Link>
                      </span>
                    ))}
                  </p>
                </div>
              )}
              {askError && (
                <div className="error" role="alert" style={{ marginTop: 16 }}>
                  {askError}
                </div>
              )}
            </div>
          </section>

          <section aria-labelledby="h-examples" className="block">
            <div className="section-head">
              <h2 id="h-examples">Saved examples</h2>
              <span className="small faint">Real outputs of the pipeline, kept as they were saved</span>
            </div>
            {loadError && (
              <div className="error" role="alert">
                <p>Could not load the examples: {loadError}</p>
                <button type="button" className="ghost" onClick={load}>Try again</button>
              </div>
            )}
            {!examples && !loadError && !offline && (
              <div className="grid" aria-busy="true" aria-label="Loading the examples">
                {[0, 1, 2].map((i) => <div key={i} className="skeleton" />)}
              </div>
            )}
            <div className="grid">
              {examples?.map((ex) => (
                <Link key={ex.id} to={`/analyst/report/${ex.id}`} className={`example-card k-${ex.kind ?? "other"}`}>
                  <span className="head">
                    <IconBadge>{KIND_ICON[(ex.kind ?? "") as keyof typeof KIND_ICON] ?? Icons.bars}</IconBadge>
                    <h3>{ex.label}</h3>
                  </span>
                  <span className="muted small">{ex.shows}</span>
                  <span className="meta">
                    {ex.kind && <span className="pill">{KIND_TEXT[ex.kind] ?? ex.kind}</span>}
                  </span>
                </Link>
              ))}
            </div>
          </section>
        </div>
        <StatusPanel health={health} meta={meta} />
      </div>
    </>
  );
}
