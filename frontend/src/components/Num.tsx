import type { Num } from "../data/types";

/** A number with the file it was read from; hover shows the exact place. */
export function N({ n }: { n: Num | null }) {
  if (!n) return <span className="muted">not available</span>;
  return (
    <span title={`${n.source_file} · ${n.pointer ?? ""} · read on ${n.read_on}`}>
      <span className="num">{String(n.value)}</span>
    </span>
  );
}

export function SrcLine({ file, pointer }: { file: string; pointer?: string }) {
  return (
    <p className="src">
      source: {file}
      {pointer ? ` · ${pointer}` : ""}
    </p>
  );
}
