import slmJson from "../data/slm.json";
import type { Cite, Slm as SlmData, SlmTable } from "../data/types";

export const slm = slmJson as unknown as SlmData;

export function Quote({ c }: { c: Cite }) {
  return (
    <li>
      {c.text}{" "}
      <span className="src">
        ({c.source_file}, line {c.line.value})
      </span>
    </li>
  );
}

// Which rows have a published adapter: only the 1.5B SFT one. Everything else is quoted for context.
function adapterNote(model: string): string {
  if (model.startsWith("1.5B SFT")) return "adapter published";
  if (model.includes("untuned")) return "no adapter (base model)";
  if (model.includes("0.5B")) return "adapter not in this release";
  return "";
}

export function Table({ title, t, notes = false }: { title: string; t: SlmTable; notes?: boolean }) {
  return (
    <section className="card" aria-label={title}>
      <h3>{title}</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {t.columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
              {notes && <th>Adapter</th>}
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {t.rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell) => (
                  <td key={cell.column} className={cell.column === row[0].column ? "" : "num"}>
                    {cell.text.replace(/\*\*/g, "")}
                  </td>
                ))}
                {notes && <td className="small muted">{adapterNote(row[0].text.replace(/\*\*/g, ""))}</td>}
                <td className="src">
                  {slm.source_file}:{row[0].line.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
