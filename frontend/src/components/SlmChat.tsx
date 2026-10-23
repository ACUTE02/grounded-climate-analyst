import { useEffect, useRef, useState, type FormEvent } from "react";
import { getSlmReports, getSlmStatus, postSlmAnswer, postSlmAsk } from "../api/client";
import type { SlmStatus } from "../api/types";

export const MAX_CONTEXT = 4000;
export const MAX_QUESTION = 500;
export const REFUSAL_SENTENCE = "I cannot answer this from the provided context.";

// Written for this page; it is not taken from the project's data or sources.
const EXAMPLE_PASSAGE =
  "Barmer is a district in the west of Rajasthan, on the border with Pakistan. " +
  "Much of the district lies in the Thar desert, and most farms depend on rain.";

type Source = "project" | "passage";

interface Turn {
  id: number;
  question: string;
  source: Source;
  passage: string;
  reportId: string | null;
  answer?: string;
  refused?: boolean;
  model?: string | null;
  used?: { kind: string; name: string }[];
  truncated?: boolean;
  error?: string;
}

/** Passage-grounded Q&A shown as a conversation. Every turn sends the passage above with the new question. */
export default function SlmChat() {
  const [status, setStatus] = useState<SlmStatus | null>(null);
  const [source, setSource] = useState<Source>("project");
  const [reports, setReports] = useState<{ id: string; label: string }[]>([]);
  const [reportId, setReportId] = useState("");
  const [passage, setPassage] = useState(EXAMPLE_PASSAGE);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const next = useRef(1);

  useEffect(() => {
    getSlmStatus()
      .then(setStatus)
      .catch(() => setStatus({ configured: false, available: false, model: null, message: "" }));
    // The saved reports, in any mode (in live mode /examples lists request templates, not saved reports).
    getSlmReports()
      .then((r) => setReports(Array.isArray(r.reports) ? r.reports : []))
      .catch(() => setReports([]));
  }, []);

  const available = status?.available === true;
  const tooLong = (source === "passage" && passage.length > MAX_CONTEXT) || question.length > MAX_QUESTION;
  const ready = available && !busy && !tooLong && (source === "project" || passage.trim() !== "") && question.trim() !== "";

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!ready) return;
    const turn: Turn = { id: next.current++, question: question.trim(), source, passage, reportId: reportId || null };
    setTurns((t) => [...t, turn]);
    setQuestion("");
    setBusy(true);
    try {
      const r =
        turn.source === "project"
          ? await postSlmAsk(turn.question, turn.reportId)
          : await postSlmAnswer(turn.passage, turn.question);
      setTurns((t) =>
        t.map((x) =>
          x.id === turn.id
            ? { ...x, answer: r.answer, refused: r.refused, model: r.model, used: r.sources, truncated: r.truncated }
            : x,
        ),
      );
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setTurns((t) => t.map((x) => (x.id === turn.id ? { ...x, error: message } : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="slm-chat" data-testid="slm-tryit">
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
        <h2 id="slm-chat" style={{ margin: 0 }}>Ask the model</h2>
        {available ? (
          <span className="pill ok" data-testid="slm-state">service running</span>
        ) : (
          <span className="pill warm" data-testid="slm-state">service not running</span>
        )}
        {status?.model && <span className="pill" data-testid="slm-model-badge">{status.model}</span>}
      </div>
      <p className="small muted">
        This is not a general chatbot. It answers only from the text it is given, or replies "{REFUSAL_SENTENCE}" It can be wrong.
      </p>

      {status && !available && (
        <div className="calm" data-testid="slm-unavailable">
          <p>
            <strong>The SLM service is not available, so the chat is off.</strong>{" "}
            {status.configured ? status.message : "It is optional and not started by default."}
          </p>
          <p className="small" style={{ margin: 0 }}>
            To use it: start the SLM service on port 8001 (<code>PHASE5_PORT=8001</code>), then start the backend
            with <code>SLM_URL=http://127.0.0.1:8001</code>.
          </p>
        </div>
      )}

      <div className="chat">
        <fieldset className="passage" style={{ margin: 0 }}>
          <legend className="small muted">Answer from</legend>
          <label style={{ display: "inline-flex", gap: 6, marginRight: 16 }}>
            <input type="radio" name="slm-source" checked={source === "project"} onChange={() => setSource("project")} />
            The project's report and source corpus
          </label>
          <label style={{ display: "inline-flex", gap: 6 }}>
            <input type="radio" name="slm-source" checked={source === "passage"} onChange={() => setSource("passage")} />
            A passage I paste
          </label>
          {source === "project" ? (
            <>
              <label htmlFor="slm-report">Report to include</label>
              <select id="slm-report" value={reportId} onChange={(e) => setReportId(e.target.value)} disabled={!available || busy}>
                <option value="">No report (source corpus only)</option>
                {reports.map((r) => (
                  <option key={r.id} value={r.id}>{r.label}</option>
                ))}
              </select>
              <p className="small muted" style={{ margin: "8px 0 0" }}>
                The backend gives the model the chosen report, its forecast labels and the best-matching passages of the
                source corpus (at most {MAX_CONTEXT} characters in all). The raw weather tables are not given to it.
              </p>
            </>
          ) : (
            <>
              <label htmlFor="slm-context">
                Passage the answers must come from ({passage.length}/{MAX_CONTEXT} characters). You can edit it between questions.
              </label>
              <textarea id="slm-context" value={passage} onChange={(e) => setPassage(e.target.value)} disabled={!available || busy} />
            </>
          )}
        </fieldset>

        <div className="thread" role="log" aria-live="polite" aria-label="Conversation">
          {turns.length === 0 && <p className="muted small">No questions yet.</p>}
          {turns.map((t) => (
            <div key={t.id} style={{ display: "contents" }}>
              <div className="bubble q">{t.question}</div>
              {t.error ? (
                <div className="bubble a" role="alert">{t.error}</div>
              ) : t.answer === undefined ? (
                <div className="bubble a muted">Working…</div>
              ) : (
                <div className={`bubble a${t.refused ? " refusal" : ""}`} data-testid="slm-result">
                  {t.answer}
                  <span className="tag">
                    refused: {String(t.refused)} · model: {t.model ?? "not reported"}
                  </span>
                  {t.used && t.used.length > 0 && (
                    <span className="tag" data-testid="slm-sources">
                      from: {t.used.map((s) => `${s.kind}: ${s.name}`).join("; ")}
                      {t.truncated ? " (some text was cut to fit)" : ""}
                    </span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        <form onSubmit={ask}>
          <label htmlFor="slm-question">Question ({question.length}/{MAX_QUESTION} characters)</label>
          <div className="asker">
            <input
              id="slm-question"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={!available || busy}
            />
            <button type="submit" disabled={!ready}>
              {busy ? "Working…" : "Ask"}
            </button>
          </div>
          {tooLong && (
            <div className="error" role="alert" style={{ marginTop: 8 }}>
              The passage may have at most {MAX_CONTEXT} characters and the question {MAX_QUESTION}.
            </div>
          )}
        </form>
      </div>
    </section>
  );
}
