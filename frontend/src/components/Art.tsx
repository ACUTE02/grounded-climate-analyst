import type { ReactNode } from "react";

/** The Thar at dusk, along the foot of the home band: a low sun behind three layers of sand dunes. Decorative (aria-hidden). */
export function HeroScene() {
  return (
    <div className="scene" aria-hidden="true">
      <svg viewBox="0 0 1440 130" preserveAspectRatio="xMidYMax slice" focusable="false">
        <defs>
          <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor="#ffd08a" stopOpacity=".55" />
            <stop offset="1" stopColor="#ffd08a" stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle cx="1120" cy="92" r="86" fill="url(#glow)" />
        <circle cx="1120" cy="92" r="30" fill="#f7b866" />
        <g fill="#ffffff" opacity=".22">
          <ellipse cx="220" cy="34" rx="90" ry="9" />
          <ellipse cx="290" cy="24" rx="52" ry="8" />
          <ellipse cx="700" cy="44" rx="70" ry="8" />
        </g>
        <path d="M0 96 C180 70 320 82 480 92 S760 108 940 88 S1250 66 1440 90 V130 H0Z" fill="#c9b27a" />
        <path d="M0 110 C200 90 380 106 560 110 S900 118 1100 102 S1340 96 1440 108 V130 H0Z" fill="#dcc690" />
        <path d="M0 124 C220 112 420 124 640 124 S1060 128 1440 118 V130 H0Z" fill="#f1e4c2" />
      </svg>
    </div>
  );
}

const base = { width: 20, height: 20, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true, focusable: false };

export const Icons = {
  /** a dial: the tools that produce the numbers */
  gauge: (
    <svg {...base}><path d="M4 16a8 8 0 1 1 16 0" /><path d="M12 16l4-5" /><circle cx="12" cy="16" r="1.2" /></svg>
  ),
  /** a page of text: the written report */
  report: (
    <svg {...base}><path d="M7 3h7l4 4v14H7z" /><path d="M14 3v4h4" /><path d="M10 12h5M10 16h5" /></svg>
  ),
  /** a tick in a shield: the number check */
  check: (
    <svg {...base}><path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z" /><path d="M9 12l2 2 4-4" /></svg>
  ),
  /** sun over a horizon: the analyst */
  sun: (
    <svg {...base}><path d="M4 17h16" /><path d="M7 17a5 5 0 0 1 10 0" /><path d="M12 6v2M5.6 9.6l1.4 1.4M18.4 9.6L17 11" /></svg>
  ),
  /** a speech bubble: the SLM chat */
  chat: (
    <svg {...base}><path d="M5 5h14v10H10l-4 4v-4H5z" /><path d="M9 9h6M9 12h4" /></svg>
  ),
  /** three bars: a forecast */
  bars: (
    <svg {...base}><path d="M5 20V12M12 20V6M19 20v-5" /></svg>
  ),
} satisfies Record<string, ReactNode>;

export function IconBadge({ children }: { children: ReactNode }) {
  return <span className="icon-badge">{children}</span>;
}
