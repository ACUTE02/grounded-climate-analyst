import { badgeKind } from "./HorizonBadges";
import type { DroughtRow } from "../data/types";

const W = 320;
const H = 150;
const TOP = 22;      // room for the value labels
const BOTTOM = 26;   // room for the horizon labels
const SCALE_MAX = 0.3;

const CLASS: Record<string, string> = { validated: "bar-validated", weak: "bar-weak", noskill: "bar-noskill" };

/** Published skill per horizon for one region, as bars around a zero line (zero = no better than the seasonal normal).
 *  The colour is the backend's label (via badgeKind); the bar height is the measured skill read from the saved file. */
export function SkillChart({ region, rows }: { region: string; rows: DroughtRow[] }) {
  const mid = TOP + (H - TOP - BOTTOM) * 0.62;               // zero line, lower than centre: positive skill is the story
  const unitUp = (mid - TOP) / SCALE_MAX;
  const unitDown = (H - BOTTOM - mid) / SCALE_MAX;
  const slot = W / Math.max(rows.length, 1);
  const summary = rows
    .map((r) => `${r.horizon} ${badgeKind(r.label).title} ${r.published_skill.value}`)
    .join("; ");

  return (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ textTransform: "capitalize" }}>{region}</figcaption>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Skill by horizon for ${region}: ${summary}`}>
        <line className="grid-line" x1="0" x2={W} y1={mid - 0.1 * unitUp} y2={mid - 0.1 * unitUp} />
        <text x={W - 2} y={mid - 0.1 * unitUp - 3} textAnchor="end">bar +0.10</text>
        <line className="axis" x1="0" x2={W} y1={mid} y2={mid} />
        {rows.map((r, i) => {
          const v = r.published_skill.value;
          const h = Math.max(Math.abs(v) * (v >= 0 ? unitUp : unitDown), 1.5);
          const x = i * slot + slot * 0.25;
          const w = slot * 0.5;
          const y = v >= 0 ? mid - h : mid;
          const kind = badgeKind(r.label).cls;
          return (
            <g key={r.horizon}>
              <rect className={CLASS[kind] ?? "bar-other"} x={x} y={y} width={w} height={h} rx="2" />
              <text x={x + w / 2} y={v >= 0 ? y - 4 : y + h + 11} textAnchor="middle">{v > 0 ? "+" : ""}{v.toFixed(4)}</text>
              <text x={x + w / 2} y={H - 8} textAnchor="middle">{r.horizon}</text>
            </g>
          );
        })}
      </svg>
      <table className="sr-only">
        <caption>Skill by horizon for {region}</caption>
        <thead><tr><th>Horizon</th><th>Label</th><th>Skill</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.horizon}><td>{r.horizon}</td><td>{badgeKind(r.label).title}</td><td>{r.published_skill.value}</td></tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

export function SkillLegend() {
  return (
    <div className="legend" aria-hidden="true">
      <span><i style={{ background: "var(--ok)" }} />Validated</span>
      <span><i style={{ background: "var(--warn)" }} />Weak / directional</span>
      <span><i style={{ background: "var(--bad)" }} />No skill</span>
    </div>
  );
}
