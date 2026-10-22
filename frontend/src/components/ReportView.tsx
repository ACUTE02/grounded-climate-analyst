import type { ReactNode } from "react";
import type { ReportResponse } from "../api/types";
import HorizonBadges from "./HorizonBadges";
import Markdown from "./Markdown";

const GROUNDING_MEANING: Record<string, string> = {
  clean: "Every number in the written report was traced to a tool output or a source.",
  warning:
    "At least one number in the written report could not be traced to a source. Do not rely on the flagged values.",
  not_generated:
    "There is no written report in this answer, so nothing was checked. Zero numbers checked is not a pass.",
};

function Fact({ name, value }: { name: string; value: ReactNode }) {
  return (
    <tr>
      <th scope="row">{name}</th>
      <td>{value}</td>
    </tr>
  );
}

const show = (v: unknown): string =>
  v === null || v === undefined ? "none" : typeof v === "object" ? JSON.stringify(v) : String(v);

export default function ReportView({ report }: { report: ReportResponse }) {
  const { meta, grounding } = report;
  const drought = report.tool_outputs["forecast_drought_risk"];
  const heat = report.tool_outputs["forecast_heat_stress_risk"];
  const crop = report.tool_outputs["assess_crop_impact"];
  const months: Record<string, string> = {};
  for (const h of drought?.horizon_confidence ?? []) {
    if (h.horizon && h.month) months[h.horizon] = h.month;
  }
  const hasReport = report.report.trim().length > 0;

  return (
    <article>
      <div className="banner" role="status" data-testid="banner">
        {meta?.banner ??
          (meta?.mode === "live"
            ? "Live answer. Written just now by the live pipeline; asking again can give different text."
            : "The backend sent no banner for this answer.")}
        {meta && (
          <div className="meta-row">
            <span className="pill">mode: {meta.mode}</span>
            <span className="pill">how made: {meta.how_made}</span>
            <span className="pill">generated: {meta.generated_at}</span>
            {meta.chat_model && <span className="pill">model: {meta.chat_model}</span>}
            {meta.data_source && <span className="pill">data: {meta.data_source}</span>}
            {meta.horizon_manifest_source && (
              <span className="pill">manifest: {meta.horizon_manifest_source}</span>
            )}
            {meta.data_current_through &&
              Object.entries(meta.data_current_through).map(([region, month]) => (
                <span key={region} className="pill">
                  {region} data through {month}
                </span>
              ))}
          </div>
        )}
      </div>

      <h1>{report.request}</h1>

      {report.horizon_confidence.length > 0 && (
        <section className="card" aria-labelledby="h-horizons">
          <h2 id="h-horizons">How far each forecast can be trusted</h2>
          <p className="muted">
            Each month carries its own measured test result. The labels are the backend's verdicts and are
            never merged into one confidence number.
          </p>
          <HorizonBadges horizons={report.horizon_confidence} months={months} />
          {drought && (
            <p className="small muted" style={{ marginTop: 14 }}>
              Forecast anchored to {String(drought.forecast_anchor_month)}; predicted index values{" "}
              <span className="num">{(drought.predicted_values ?? []).map(String).join(", ")}</span>.
            </p>
          )}
        </section>
      )}

      {(heat || crop) && (
        <section className="card" aria-labelledby="h-obs">
          <h2 id="h-obs">What the tools found</h2>
          <div className="table-wrap">
            <table>
              <tbody>
                {heat && (
                  <>
                    <Fact name="Heat stress: month" value={show(heat.month)} />
                    <Fact name="Heat wave days observed" value={show(heat.heatwave_days)} />
                    <Fact name="Severe heat wave days" value={show(heat.severe_heatwave_days)} />
                    <Fact name="Highest daily maximum (°C)" value={show(heat.max_tmax_c)} />
                    <Fact
                      name="Forecast available"
                      value={show(heat.forecast_available) + " — " + show(heat.note)}
                    />
                  </>
                )}
                {crop && (
                  <>
                    <Fact name="Crop impact: crop, month" value={`${show(crop.crop)}, ${show(crop.month)}`} />
                    <Fact name="Risk that binds yield" value={show(crop.dominant_risk)} />
                    <Fact name="Why" value={show(crop.risk_reasoning)} />
                    <Fact name="Yield impact (%)" value={show(crop.yield_impact_pct)} />
                    <Fact name="Yield impact status" value={show(crop.yield_impact_status)} />
                    {crop.yield_impact_source && (
                      <Fact name="Source of the coefficient" value={show(crop.yield_impact_source)} />
                    )}
                    {crop.yield_impact_caveat && (
                      <Fact name="Caveat" value={show(crop.yield_impact_caveat)} />
                    )}
                    <Fact name="Confidence" value={show(crop.confidence_label)} />
                  </>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="card" aria-labelledby="h-data">
        <h2 id="h-data">What this system does not have</h2>
        {report.missing_data.length === 0 ? (
          <p className="muted">The backend reported no missing data for this answer.</p>
        ) : (
          <ul>
            {report.missing_data.map((m, i) => (
              <li key={i}>
                <strong>{m.what}</strong> — {m.reason}
              </li>
            ))}
          </ul>
        )}
        <h3>Outside sources fetched for this answer</h3>
        {report.external_sources.length === 0 ? (
          <p className="muted small">None are part of this answer.</p>
        ) : (
          <ul>
            {report.external_sources.map((s, i) => (
              <li key={i}>
                <strong>{s.title ?? s.id}</strong> ({s.publisher}) —{" "}
                {s.available ? `available, fetched ${s.fetched_at}` : `unavailable: ${s.reason}`}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card" aria-labelledby="h-ground">
        <h2 id="h-ground">Number check</h2>
        <p>
          <span className={`status ${grounding.status}`} data-testid="grounding-status">
            {grounding.status}
          </span>{" "}
          <span className="muted">
            {GROUNDING_MEANING[grounding.status] ?? "The backend sent a status this page does not know."}
          </span>
        </p>
        <p className="small muted">{grounding.explanation}</p>
        {grounding.unverified_numbers.length > 0 && (
          <p>
            Flagged: <span className="num">{grounding.unverified_numbers.join(", ")}</span>
          </p>
        )}
      </section>

      <section className="card" aria-labelledby="h-report">
        <h2 id="h-report">Written report</h2>
        {hasReport ? (
          <div className="report-text">
            <Markdown text={report.report} />
          </div>
        ) : (
          <div className="calm">
            Not generated in this saved example. {grounding.explanation}
          </div>
        )}
      </section>

      {report.slm && report.slm.enabled && (
        <section className="card" aria-labelledby="h-slm" data-testid="slm-section">
          <h2 id="h-slm">Passage check (experimental small language model)</h2>
          <p className="muted small">
            Optional and off by default. Its effect on the quality of the report has not been measured, and it
            is not used to write the report. {report.slm.note}
          </p>
          {report.slm.skipped_reason && (
            <div className="calm">Skipped: {report.slm.skipped_reason}</div>
          )}
          <ul>
            {report.slm.answers.map((a, i) => (
              <li key={i}>
                <strong>{a.source ?? "passage"}</strong>
                {a.citation ? <span className="src"> {a.citation}</span> : null}
                <br />
                {a.error ? (
                  <span className="muted">{a.error}</span>
                ) : a.refused ? (
                  <span className="muted">The model says this passage cannot answer the question.</span>
                ) : (
                  <span>{a.answer}</span>
                )}
              </li>
            ))}
          </ul>
          {report.slm.model && <p className="src">model: {report.slm.model}</p>}
        </section>
      )}

      <section className="card quiet" aria-labelledby="h-src">
        <h2 id="h-src">Sources and run details</h2>
        <p className="small muted">
          Tools called: {report.tools_called.length ? report.tools_called.join(", ") : "none"}
        </p>
        {report.retrieved_sources.length > 0 ? (
          <ul className="small">
            {report.retrieved_sources.map((s, i) => (
              <li key={i}>
                {s.source} — <span className="muted">{s.citation}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="small muted">No documents were retrieved for this answer.</p>
        )}
        {report.warnings.length > 0 && (
          <>
            <h3>Warnings</h3>
            <ul className="small">
              {report.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </>
        )}
      </section>
    </article>
  );
}
