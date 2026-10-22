import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// Every number in src/data must come with the file it was read from. The data files are
// written by scripts/sync-evidence.mjs; a number typed by hand would fail here.
const dir = path.resolve(__dirname, "..", "data");
const files = readdirSync(dir).filter((f) => f.endsWith(".json"));

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };

function walk(node: Json, trail: string, parent: Json | undefined, problems: string[]) {
  if (typeof node === "number") {
    problems.push(`${trail}: a bare number (every number must be {value, source_file, ...})`);
  } else if (Array.isArray(node)) {
    node.forEach((child, i) => walk(child, `${trail}[${i}]`, node, problems));
  } else if (node && typeof node === "object") {
    for (const [key, child] of Object.entries(node)) {
      if (typeof child === "number") {
        if (key !== "value") {
          problems.push(`${trail}.${key}: a bare number`);
        } else {
          const holder = (o: any) => typeof o?.source_file === "string" && typeof o?.read_on === "string";
          if (!holder(node) && !holder(parent)) problems.push(`${trail}: a number without source_file and read_on`);
        }
      } else {
        walk(child, `${trail}.${key}`, node, problems);
      }
    }
  }
}

describe("src/data", () => {
  it("has the evidence files (five synced, plus the about numbers)", () => {
    expect(files.sort()).toEqual(["about.json", "checker.json", "drought.json", "heat.json", "slm.json", "tracka.json"]);
  });

  for (const file of files) {
    it(`${file}: every number carries source_file and read_on`, () => {
      const data = JSON.parse(readFileSync(path.join(dir, file), "utf8")) as Json;
      const problems: string[] = [];
      walk(data, file, undefined, problems);
      expect(problems).toEqual([]);
    });
  }

  it("the SLM numbers carry a file and a line", () => {
    const slm = JSON.parse(readFileSync(path.join(dir, "slm.json"), "utf8"));
    for (const table of [slm.model_quality, slm.parity, slm.benchmark, slm.startup]) {
      for (const row of table.rows) {
        for (const cell of row) {
          expect(cell.source_file).toMatch(/README\.md$/);
          expect(cell.line.value).toBeGreaterThan(0);
        }
      }
    }
  });

  it("Track A numbers sit next to the corpus note", () => {
    const a = JSON.parse(readFileSync(path.join(dir, "tracka.json"), "utf8"));
    expect(a.corpus_note).toBe("measured on the original 224-chunk corpus");
    expect(a.easy.note).toBe(a.corpus_note);
    expect(a.strict.note).toBe(a.corpus_note);
  });

  it("the published and reproduced drought values are compared by the script, not the browser", () => {
    const d = JSON.parse(readFileSync(path.join(dir, "drought.json"), "utf8"));
    for (const rows of Object.values(d.regions) as any[]) {
      for (const r of rows) expect(typeof r.identical_to_published === "boolean" || r.identical_to_published === null).toBe(true);
    }
  });
});
