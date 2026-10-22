"""Phase 4b, Stage A: the preference pairs of DPO v2. Pure Python, no model and no GPU.

    python -m climate_slm.training.dpo2_pairs [--show]

Inputs: sft_train.jsonl (the ONLY source of rows), eval_base_train.json (untuned 0.5B greedy, made in v1) and
outputs/dpo2_candidates.json (untuned sampled, SFT greedy, SFT sampled; made on Colab by dpo2_generate, cell P4b-0).

Refusal-chosen pairs: for each of the 21 refusal rows, chosen = the exact refusal text, rejected = each distinct generated
text that is not empty and not a refusal by the strict AND the loose detector of the evaluator. Identical texts of one row
are kept once (the first source in the order of `generation.sources`).
Two sets:  v2-main = refusal-chosen pairs only.  v2-mix = the same + type 1 pairs (answer chosen, refusal text rejected)
sampled down (seeded) so that answer-chosen lines == refusal-chosen lines.  Type 2 stays dropped.
The pair files are written only when every source is present; with a missing source the run is PARTIAL: counts and
samples are printed and a summary (partial: true) is written, but no pair file."""
import argparse
import json
import random
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase4_config, load_phase4b_config
from climate_slm.data.sft import parse_example
from climate_slm.evaluation import metrics as M
from climate_slm.training import dpo_pairs as D1

SET_FILES = {"v2-main": ("main_pairs", "main_sha256"), "v2-mix": ("mix_pairs", "mix_sha256")}
SKIP_KEYS = ("empty", "strict_refusal", "loose_refusal_only", "duplicate")


def copies_context(text: str, context: str) -> bool:
    """True if the (normalised) text is a verbatim piece of the context passage (v1 models were seen copying passages)."""
    t = M.normalize(text)
    return bool(t) and t in M.normalize(context)


def classify(text: str, refusal_text: str, loose_phrases: list[str]) -> str:
    """'empty' | 'strict_refusal' | 'loose_refusal_only' | 'ok'."""
    if not text.strip():
        return "empty"
    if M.is_refusal_strict(text, refusal_text):
        return "strict_refusal"
    if M.is_refusal_loose(text, refusal_text, loose_phrases):
        return "loose_refusal_only"
    return "ok"


# ---- inputs -----------------------------------------------------------------------------------------------------------
def untuned_greedy_texts(base_eval: dict, refusal_ids: list[str]) -> dict[str, list[str]]:
    """The untuned greedy output of each refusal row, from the v1 evaluator file."""
    items = {it["id"]: it for it in base_eval["items"]}
    return {rid: [items[rid]["output"]] for rid in refusal_ids}


def check_candidates(cand: dict, train_rows: list[dict], train_sha: str, cfg4b: dict, adapter_sha: str | None = None) -> list[str]:
    """Problems with the Colab candidate file (empty list = usable)."""
    m, g, problems = cand.get("meta", {}), cfg4b["generation"], []
    if m.get("data_sha256") != train_sha:
        problems.append("data_sha256 does not match the current sft_train.jsonl")
    if adapter_sha is not None and m.get("sft_adapter_sha256") != adapter_sha:
        problems.append("the SFT adapter hash differs from configs/phase4.yaml")
    if m.get("n_samples") != g["n_samples"] or m.get("sample_seeds") != [g["sample_seed"] + k for k in range(g["n_samples"])]:
        problems.append("n_samples or sample seeds differ from configs/phase4b.yaml")
    ids = sorted(r["meta"]["id"] for r in train_rows if r["meta"]["type"] == "unanswerable")
    if sorted(cand.get("rows", {})) != ids:
        problems.append("the rows of the candidate file are not exactly the refusal rows of sft_train")
    expect = {"untuned_sampled": g["n_samples"], "sft_greedy": 1, "sft_sampled": g["n_samples"]}
    for rid, d in cand.get("rows", {}).items():
        for src, n in expect.items():
            if len(d.get(src, [])) != n:
                problems.append(f"{rid}: {src} has {len(d.get(src, []))} texts, expected {n}")
                break
    return problems


def collect_texts(refusal_ids: list[str], base_eval: dict, cand: dict | None, sources: list[str]) -> tuple[dict, list[str]]:
    """{row id: [(source, k, text), ...]} in source order, and the sources that are missing."""
    by_source = {"untuned_greedy": untuned_greedy_texts(base_eval, refusal_ids)}
    missing = []
    for s in sources:
        if s == "untuned_greedy":
            continue
        if cand is None:
            missing.append(s)
        else:
            by_source[s] = {rid: cand["rows"][rid].get(s, []) for rid in refusal_ids}
    out = {rid: [] for rid in refusal_ids}
    for s in sources:
        for rid in refusal_ids:
            for k, t in enumerate(by_source.get(s, {}).get(rid, []), 1):
                out[rid].append((s, k, t))
    return out, missing


# ---- pairs ----------------------------------------------------------------------------------------------------------------
def refusal_pair(row: dict, text: str, source: str, k: int, refusal_text: str) -> dict:
    p = D1._pair(row, 3, refusal_text, text, source, None)
    p["pair_id"] = p["meta"]["distinct_pair_id"] = f"{row['meta']['id']}:t3:{source}:{k}"
    p["meta"]["rejected_k"] = k
    return p


def build_refusal_pairs(train_rows: list[dict], texts: dict, refusal_text: str, loose_phrases: list[str], sources: list[str],
                        exclude_ids: set[str] | None = None) -> tuple[list[dict], dict]:
    """(pairs, stats). stats per source: generated, empty, strict_refusal, loose_refusal_only, duplicate, kept."""
    exclude_ids = exclude_ids or set()
    stats = {s: {"generated": 0, **{k: 0 for k in SKIP_KEYS}, "kept": 0, "kept_copied_from_context": 0} for s in sources}
    pairs, rows_with_pairs = [], 0
    for row in train_rows:
        if row["meta"]["type"] != "unanswerable" or row["meta"]["id"] in exclude_ids:
            continue
        seen, n_before = set(), len(pairs)
        for source, k, text in texts.get(row["meta"]["id"], []):
            st = stats[source]
            st["generated"] += 1
            verdict = classify(text, refusal_text, loose_phrases)
            key = M.normalize(text)
            if verdict != "ok":
                st[verdict] += 1
            elif key in seen:
                st["duplicate"] += 1
            else:
                seen.add(key)
                st["kept"] += 1
                pair = refusal_pair(row, text.strip(), source, k, refusal_text)
                pair["meta"]["copies_context"] = copies_context(text, parse_example(row)[0])
                st["kept_copied_from_context"] += pair["meta"]["copies_context"]
                pairs.append(pair)
        rows_with_pairs += len(pairs) > n_before
    stats["_rows_with_at_least_one_pair"] = rows_with_pairs
    return pairs, stats


def build_type1(train_rows: list[dict], refusal_text: str, exclude_ids: set[str] | None = None) -> list[dict]:
    exclude_ids = exclude_ids or set()
    out = []
    for row in train_rows:
        if row["meta"]["type"] != "unanswerable" and row["meta"]["id"] not in exclude_ids:
            out.append(D1._pair(row, 1, parse_example(row)[2], refusal_text, "refusal_text", None))
    return out


def sample_equal(type1: list[dict], n: int, seed: int) -> list[dict]:
    """n type 1 pairs (all of them if fewer), seeded, in file order."""
    rng = random.Random(seed)
    return [type1[i] for i in sorted(rng.sample(range(len(type1)), min(n, len(type1))))]


def build_sets(refusal_pairs: list[dict], type1: list[dict], mix_seed: int) -> dict[str, list[dict]]:
    """v2-main = refusal-chosen pairs; v2-mix = those + as many sampled type 1 pairs (answer chosen)."""
    return {"v2-main": list(refusal_pairs), "v2-mix": sample_equal(type1, len(refusal_pairs), mix_seed) + list(refusal_pairs)}


def set_counts(lines: list[dict]) -> dict:
    return {"lines": len(lines), "answer_chosen_lines": sum(1 for p in lines if p["pair_type"] == 1),
            "refusal_chosen_lines": sum(1 for p in lines if p["pair_type"] == 3),
            "distinct_pairs": len({p["meta"]["distinct_pair_id"] for p in lines})}


def pick_rejected_samples(pairs: list[dict], per_source: int, seed: int, sources: list[str]) -> dict[str, list[dict]]:
    """Random rejected texts per source (all of them if fewer), seeded; shown in file order."""
    rng, out = random.Random(seed), {}
    for s in sources:
        of = [p for p in pairs if p["meta"]["rejected_source"] == s]
        out[s] = [of[i] for i in sorted(rng.sample(range(len(of)), min(per_source, len(of))))]
    return out


def format_rejected(p: dict) -> str:
    _, question, _ = parse_example({"prompt": p["prompt"], "completion": [{"role": "assistant", "content": ""}]})
    note = " | copies the context verbatim" if p["meta"].get("copies_context") else ""
    return f"[{p['pair_id']}] chunk {p['meta']['chunk_id']}{note}\nQUESTION: {question}\nREJECTED: {p['rejected']}"


def summarize(stats: dict, sets: dict, sources: list[str], missing: list[str], train_rows: list[dict], hashes: dict,
              problems: list[str], type1_available: int) -> dict:
    n_ref = sum(1 for r in train_rows if r["meta"]["type"] == "unanswerable")
    totals = {k: sum(stats[s][k] for s in sources) for k in ("generated", *SKIP_KEYS, "kept", "kept_copied_from_context")}
    return {"partial": bool(missing), "missing_sources": missing, "sources": sources, "refusal_rows": n_ref,
            "refusal_rows_with_at_least_one_pair": stats["_rows_with_at_least_one_pair"],
            "per_source": {s: stats[s] for s in sources}, "totals": totals,
            "type1_pairs_available": type1_available, "sets": {k: set_counts(v) for k, v in sets.items()},
            "leakage": {"problems": problems, "clean": not problems}, "input_sha256": hashes}


# ---- command line ---------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the DPO v2 pairs from sft_train.jsonl, the v1 untuned file and the Colab candidates.")
    ap.add_argument("--show", action="store_true", help="print random rejected texts per source in full")
    ap.add_argument("--no-write", action="store_true",
                    help="look first: print counts and samples, write NO pair file (only the summary, pair_files_written: false)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg4b, cfg4, ecfg = load_phase4b_config(), load_phase4_config(), load_eval_config()
    refusal, loose = ecfg["refusal_text"], ecfg["eval"]["loose_refusal_phrases"]
    P = {k: REPO_ROOT / v for k, v in cfg4b["paths"].items()}
    train, val, test = (D1.read_jsonl(P[k]) for k in ("sft_train", "sft_val", "sft_test"))
    if not P["base_train_eval"].exists():
        print(f"{P['base_train_eval']} is missing (v1 cell P4-0).")
        return 1
    base = json.loads(P["base_train_eval"].read_text(encoding="utf-8"))
    train_sha = D1.file_sha256(P["sft_train"])
    D1.check_base_eval(base, train, train_sha)
    sources = cfg4b["generation"]["sources"]
    cand = json.loads(P["candidates"].read_text(encoding="utf-8")) if P["candidates"].exists() else None
    if cand is not None:
        bad = check_candidates(cand, train, train_sha, cfg4b, cfg4["start"]["seeds"][cfg4b["generation"]["sft_seed"]]["sha256"])
        if bad:
            print("the candidate file cannot be used: " + "; ".join(bad))
            return 1
    refusal_ids = [r["meta"]["id"] for r in train if r["meta"]["type"] == "unanswerable"]
    texts, missing = collect_texts(refusal_ids, base, cand, sources)
    excluded = D1.rows_to_exclude(train, val, test)
    rpairs, stats = build_refusal_pairs(train, texts, refusal, loose, sources, excluded)
    type1 = build_type1(train, refusal, excluded)
    sets = build_sets(rpairs, type1, cfg4b["pairs"]["mix_seed"])
    problems = D1.leakage_problems(rpairs + type1, val, test)
    hashes = {"sft_train": train_sha, "sft_val": D1.file_sha256(P["sft_val"]), "sft_test": D1.file_sha256(P["sft_test"]),
              "eval_base_train": D1.file_sha256(P["base_train_eval"]),
              "candidates": D1.file_sha256(P["candidates"]) if cand is not None else None}
    if problems:
        print("LEAKAGE, nothing written:\n" + "\n".join(problems))
        return 1
    written = not missing and not a.no_write
    if written:
        for name, lines in sets.items():
            pf, sf = (P[k] for k in SET_FILES[name])
            pf.write_text(D1.pair_lines(lines), encoding="utf-8")
            sha = D1.file_sha256(pf)
            sf.write_text(f"{sha}  {pf.name}\n", encoding="utf-8")
            hashes[name] = sha
    summary = summarize(stats, sets, sources, missing, train, hashes, problems, len(type1))
    summary["pair_files_written"] = written
    P["summary"].write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("partial", "missing_sources", "refusal_rows", "refusal_rows_with_at_least_one_pair",
                                              "per_source", "totals", "type1_pairs_available", "sets")}, indent=1))
    print("PARTIAL: no pair file written (sources missing: run cell P4b-0 on Colab)." if missing
          else "pair files written: " + ", ".join(f"{n} sha256 {hashes[n][:16]}..." for n in sets) if written
          else "--no-write: NO pair file written.")
    if a.show:
        pc = cfg4b["pairs"]
        for s, items in pick_rejected_samples(rpairs, pc["samples_per_source"], pc["sample_seed"], sources).items():
            n_src = sum(1 for p in rpairs if p["meta"]["rejected_source"] == s)
            print(f"\n===== REJECTED TEXTS FROM {s} ({len(items)} shown of {n_src} kept) =====")
            for p in items:
                print(format_rejected(p) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
