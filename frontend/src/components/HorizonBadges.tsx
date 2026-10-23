import type { HorizonConfidence } from "../api/types";

// The backend's label is the verdict. The browser only chooses how to *display* it;
// it never decides a label, never derives one from a score and never changes a score.
export function badgeKind(label: string): { cls: string; title: string } {
  if (label === "validated") return { cls: "validated", title: "Validated" };
  if (label === "weak/directional") return { cls: "weak", title: "Weak / Directional" };
  if (label.startsWith("no skill")) return { cls: "noskill", title: "No skill" };
  return { cls: "", title: label };
}

interface Props {
  horizons: HorizonConfidence[];
  /** fewer details per badge (landing preview) */
  compact?: boolean;
  /** calendar month per horizon, as sent inside tool_outputs.forecast_drought_risk */
  months?: Record<string, string>;
}

export default function HorizonBadges({ horizons, months = {}, compact = false }: Props) {
  if (horizons.length === 0) return null;
  return (
    <div className="badges" role="list" aria-label="Forecast horizons">
      {horizons.map((h) => {
        const kind = badgeKind(h.label);
        return (
          <div key={h.horizon} className={`badge ${kind.cls}`} role="listitem">
            <span className="muted small">
              {h.horizon}
              {months[h.horizon] ? ` · ${months[h.horizon]}` : ""}
            </span>
            <span className="kind">{kind.title}</span>
            <span className="score" aria-label="skill score">
              {String(h.skill_score)}
            </span>
            {!compact && <span className="small muted">{h.label}</span>}
            {!compact && (
              <span className="small muted">
                method {h.method} · reliable: {h.reliable ? "yes" : "no"}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
