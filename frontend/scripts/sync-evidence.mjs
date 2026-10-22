// Writes frontend/src/data/*.json from the saved result files. Nothing is typed by hand:
// every number is written as { value, source_file, pointer | line, read_on }.
//
//   node scripts/sync-evidence.mjs
//
// Sources (read-only): backend/models/*.json, backend/retrieval/eval_results*.json,
// backend/evaluation/checker_eval_results.json, and the SLM repo's README
// (SLM_REPO, default E:\Climate-Domain-SLM-publish).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const outDir = path.resolve(here, "..", "src", "data");
const slmRepo = process.env.SLM_REPO ?? "E:\\Climate-Domain-SLM-publish";
const readOn = new Date().toISOString().slice(0, 10);

const readJson = (rel) => JSON.parse(readFileSync(path.join(root, rel), "utf8"));
const readLines = (abs) => readFileSync(abs, "utf8").split(/\r?\n/);
const lineOf = (lines, needle, from = 0) => {
  const i = lines.findIndex((l, k) => k >= from && l.includes(needle));
  if (i < 0) throw new Error(`not found: ${needle}`);
  return i + 1;
};

/** A number with its source. `pointer` is a path inside a JSON file. */
const num = (value, file, pointer) => {
  if (typeof value !== "number") throw new Error(`${file} ${pointer}: not a number (${value})`);
  return { value, source_file: file, pointer, read_on: readOn };
};
const write = (name, data) => {
  mkdirSync(outDir, { recursive: true });
  writeFileSync(path.join(outDir, name), JSON.stringify(data, null, 2) + "\n");
  console.log(`wrote src/data/${name}`);
};

// ---- Drought: published (frozen) vs reproduced on this clone ----------------
const PUB = "backend/models/horizon_manifest.json";
const GEN = "backend/models/generated/horizon_manifest.json";
const published = readJson(PUB);
let reproduced = null;
try {
  reproduced = readJson(GEN);
} catch {
  console.warn(`(no ${GEN}: run python -m scripts.setup; the page will say so)`);
}
const drought = { published_file: PUB, reproduced_file: reproduced ? GEN : null, regions: {} };
for (const [region, rows] of Object.entries(published.regions)) {
  drought.regions[region] = rows.map((row, i) => {
    const re = reproduced?.regions?.[region]?.[i];
    const p = (k, doc, file) => num(doc[k], file, `regions.${region}[${i}].${k}`);
    return {
      horizon: row.horizon,
      method: row.method,
      label: row.label,
      lookback: p("lookback", row, PUB),
      published_skill: p("skill_score", row, PUB),
      published_rmse: p("rmse_window_a", row, PUB),
      reproduced_skill: re ? p("skill_score", re, GEN) : null,
      reproduced_rmse: re ? p("rmse_window_a", re, GEN) : null,
      // decided here, not in the browser
      identical_to_published: re
        ? re.skill_score === row.skill_score &&
          re.rmse_window_a === row.rmse_window_a &&
          re.label === row.label &&
          re.lookback === row.lookback
        : null,
    };
  });
}
write("drought.json", drought);

// ---- Heat: the null result ---------------------------------------------------
const HM = "backend/models/heat_manifest.json";
const HP = "backend/models/heat_phase11.json";
const heatManifest = readJson(HM);
const phase11 = readJson(HP);
write("heat.json", {
  manifest_file: HM,
  regions: Object.fromEntries(
    Object.entries(heatManifest.regions).map(([region, doc]) => [
      region,
      doc.horizons.map((h, i) => ({
        horizon: h.horizon,
        label: h.label,
        skill: num(h.skill_score, HM, `regions.${region}.horizons[${i}].skill_score`),
      })),
    ]),
  ),
  extended_search: {
    file: HP,
    cells_tried: num(phase11.results.length, HP, "results (length)"),
    best_cell: {
      region: phase11.best_cell.region,
      target: phase11.best_cell.target,
      feature_set: phase11.best_cell.feature_set,
      horizon: phase11.best_cell.horizon,
      skill: num(phase11.best_cell.skill_score, HP, "best_cell.skill_score"),
    },
    anything_clears_bar: phase11.anything_clears_bar,
    anything_clears_bar_pointer: "anything_clears_bar",
    read_on: readOn,
  },
});

// ---- The number checker -------------------------------------------------------
const CK = "backend/evaluation/checker_eval_results.json";
const checker = readJson(CK);
const defectLine = (() => {
  const lines = readLines(path.join(root, "backend/api/app.py"));
  return lineOf(lines, '"known_defect"');
})();
write("checker.json", {
  file: CK,
  generated_at: checker.generated_at,
  what_this_measures: checker.what_this_measures,
  cases: num(checker.cases, CK, "cases"),
  numbers_labelled: num(checker.numbers_labelled, CK, "numbers_labelled"),
  true_positives: num(checker.true_positives, CK, "true_positives"),
  false_positives: num(checker.false_positives, CK, "false_positives"),
  false_negatives: num(checker.false_negatives, CK, "false_negatives"),
  true_negatives: num(checker.true_negatives, CK, "true_negatives"),
  precision: num(checker.precision, CK, "precision"),
  recall: num(checker.recall, CK, "recall"),
  f1: num(checker.f1, CK, "f1"),
  known_defect: {
    text: "An invented integer percentage below 50% is currently accepted when the sources contain a zero. Measured, documented, not yet fixed.",
    source_file: "backend/api/app.py",
    line: { value: defectLine },
    read_on: readOn,
  },
});

// ---- Track A: retrieval ---------------------------------------------------------
const EASY = "backend/retrieval/eval_results.json";
const STRICT = "backend/retrieval/eval_results_hybrid.json";
const CORPUS_NOTE = "measured on the original 224-chunk corpus";
const easy = readJson(EASY);
const strict = readJson(STRICT);
const configLines = readLines(path.join(root, "backend/retrieval/config.py"));
const modeLine = lineOf(configLines, "RETRIEVAL_MODE =");
const modeValue = /RETRIEVAL_MODE\s*=\s*"(\w+)"/.exec(configLines[modeLine - 1])?.[1];
const strictSet = (name) => {
  const agg = strict.sets[name].aggregate;
  return Object.fromEntries(
    Object.entries(agg).map(([cfg, a]) => [
      cfg,
      {
        queries: num(a.n_queries, STRICT, `sets.${name}.aggregate.${cfg}.n_queries`),
        precision_at_5: num(a.chunk_precision_at_5, STRICT, `sets.${name}.aggregate.${cfg}.chunk_precision_at_5`),
        mrr: num(a.mrr, STRICT, `sets.${name}.aggregate.${cfg}.mrr`),
      },
    ]),
  );
};
write("tracka.json", {
  corpus_note: CORPUS_NOTE,
  easy: {
    file: EASY,
    note: CORPUS_NOTE,
    queries: num(easy.results.length, EASY, "results (length)"),
    k: num(easy.k, EASY, "k"),
    precision_at_k: num(easy.precision_at_k, EASY, "precision_at_k"),
    mrr: num(easy.mrr, EASY, "mrr"),
    generated_at: easy.generated_at,
  },
  strict: {
    file: STRICT,
    note: CORPUS_NOTE,
    metric_note: strict.metric_note,
    pooling_assumption: strict.pooling_assumption,
    generated_at: strict.generated_at,
    soft: strictSet("soft"),
    hard: strictSet("hard"),
  },
  default_search: {
    mode: modeValue,
    statement: "Plain dense search stays the default retrieval mode.",
    source_file: "backend/retrieval/config.py",
    line: { value: modeLine },
    read_on: readOn,
  },
});

// ---- The SLM: results quoted from its README ---------------------------------------
const SLM_FILE = "Climate-Domain-SLM-publish/README.md";
const slmLines = readLines(path.join(slmRepo, "README.md"));
const cite = (line, text) => ({ text, source_file: SLM_FILE, line: { value: line }, read_on: readOn });
const table = (header) => {
  const start = lineOf(slmLines, header);
  let i = start; // 1-based line of the header; table begins after a blank line
  while (!slmLines[i]?.startsWith("|")) i++;
  const head = slmLines[i].split("|").slice(1, -1).map((s) => s.trim());
  const rows = [];
  for (let j = i + 2; slmLines[j]?.startsWith("|"); j++) {
    const cells = slmLines[j].split("|").slice(1, -1).map((s) => s.trim());
    rows.push(cells.map((c, k) => ({ column: head[k], ...cite(j + 1, c) })));
  }
  return { header, columns: head, rows };
};
const quote = (needle) => {
  const n = lineOf(slmLines, needle);
  return cite(n, slmLines[n - 1].replace(/^-\s*/, "").trim());
};
write("slm.json", {
  source_file: SLM_FILE,
  status: "Evaluation results. The SLM service is optional and is not started by default.",
  adapter: {
    id: "GHOSTBHAIYA/climate-slm-1.5b-sft",
    url: "https://huggingface.co/GHOSTBHAIYA/climate-slm-1.5b-sft",
    base_model: "Qwen/Qwen2.5-1.5B-Instruct",
    licence: "Apache-2.0",
    statement: "Only the 1.5B SFT adapter is published. The 0.5B and DPO adapters are not part of this release.",
  },
  model_quality: table("### Model quality"),
  parity: table("### Serving: does the service answer like the evaluator?"),
  benchmark: table("### Serving: vLLM against the `hf` backend"),
  startup: table("**Start-up time**"),
  test_split: quote("All numbers are on the **test split**"),
  limits: [
    quote("The dataset is small."),
    quote("**0.5B SFT vs 0.5B DPO v2**"),
    quote("**The data was written and checked by the same model family.**"),
    quote("**The benchmark is narrow.**"),
    quote("**DPO v1 failed**"),
  ],
  laptop_smoke_test: quote("On the author's laptop"),
});
