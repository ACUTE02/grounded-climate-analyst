"""Phase 4, Stage A: the preference pairs for DPO. Pure Python, no model and no GPU.

Inputs: data/instruction/sft_train.jsonl (the ONLY source of rows) and outputs/eval_base_train.json (the untuned 0.5B,
greedy, default prompt, on the train prompts; made by cell P4-0). Output: data/instruction/dpo_train.jsonl, its sha256
and a summary json.

Pair types (prompt = the SFT prompt of the row). Type 2 is DROPPED by the reviewer (amendment 2026-10-20): its code stays
here behind `pairs.include_type2` (default false) so the choice is visible and testable.
  1. answerable row: chosen = the reference answer, rejected = the exact refusal text.
  2. answerable row: chosen = the reference answer, rejected = the untuned output, if its token F1 against the
     reference is below `rejected_f1_below` AND (key_recall is below `rejected_key_recall_below` OR the output has a
     number that the context does not support, i.e. number faithfulness < 1). A low F1 alone is not enough: a correct
     answer in other words has a low F1 too. An output that is a strict refusal or empty is not used. If fewer than
     `type2_min_pairs` pairs are left, type 2 is dropped completely.
  3. refusal row: chosen = the exact refusal text, rejected = the untuned output (an answer), if the evaluator's LOOSE
     refusal detector does not call it a refusal (and it is not empty).
Type 3 pairs are written `refusal_pair_repeat` times with the pair ids <row id>:t3:r1, :r2, ... (the distinct pair is
<row id>:t3, the same for all repeats). Stage B: compute the reference log-probabilities ONCE per distinct pair.

The rejected outputs come from the UNTUNED base (off-policy): they are not what the SFT model would write.
Run:  python -m climate_slm.training.dpo_pairs [--show]"""
import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase4_config
from climate_slm.data.sft import parse_example
from climate_slm.evaluation import metrics as M

TYPE_NAMES = {1: "answerable: reference vs refusal", 2: "answerable: reference vs wrong untuned answer",
              3: "refusal row: refusal vs untuned answer"}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_refusal_row(row: dict) -> bool:
    return row["meta"]["type"] == "unanswerable"


# ---------------------------------------------------------------- checks on the inputs

def check_base_eval(base: dict, train_rows: list[dict], train_sha256: str) -> None:
    """The base eval file must be the plain untuned base on exactly this train file. Raises ValueError otherwise."""
    problems = []
    if base.get("split") != "train":
        problems.append(f"split is {base.get('split')!r}, not 'train'")
    if base.get("limit"):
        problems.append(f"smoke run (limit={base['limit']})")
    if base.get("adapter"):
        problems.append("an adapter was loaded: only the untuned base is allowed")
    if base.get("load_4bit"):
        problems.append("the base was loaded in 4-bit")
    if base.get("prompt_variant", "default") != "default":
        problems.append(f"prompt variant {base.get('prompt_variant')!r}, not the default prompt")
    if base.get("decoding_mode", "greedy") != "greedy":
        problems.append(f"decoding mode {base.get('decoding_mode')!r}, not greedy")
    if base.get("data_sha256") != train_sha256:
        problems.append("data_sha256 does not match the current sft_train.jsonl")
    ids_rows = [r["meta"]["id"] for r in train_rows]
    ids_eval = [it["id"] for it in base.get("items", [])]
    if sorted(ids_rows) != sorted(ids_eval):
        problems.append(f"item ids differ from the train rows ({len(ids_eval)} items vs {len(ids_rows)} rows)")
    if problems:
        raise ValueError("eval_base_train.json cannot be used: " + "; ".join(problems))


def split_overlap(train_rows: list[dict], other_rows: list[dict]) -> dict:
    """Train rows that share an id or a chunk_id with the other split: {'ids': [...], 'chunk_rows': [train ids]}."""
    other_ids = {r["meta"]["id"] for r in other_rows}
    other_chunks = {r["meta"]["chunk_id"] for r in other_rows}
    return {"ids": sorted(r["meta"]["id"] for r in train_rows if r["meta"]["id"] in other_ids),
            "chunk_rows": sorted(r["meta"]["id"] for r in train_rows if r["meta"]["chunk_id"] in other_chunks)}


def rows_to_exclude(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]) -> set[str]:
    """Ids of train rows that must not enter a pair because the row or its chunk also occurs in val or test."""
    out = set()
    for other in (val_rows, test_rows):
        o = split_overlap(train_rows, other)
        out.update(o["ids"])
        out.update(o["chunk_rows"])
    return out


def leakage_problems(pairs: list[dict], val_rows: list[dict], test_rows: list[dict]) -> list[str]:
    """Empty list = clean. Checks the finished pairs against the ids and chunk_ids of val and test."""
    bad_ids = {r["meta"]["id"] for r in val_rows + test_rows}
    bad_chunks = {r["meta"]["chunk_id"] for r in val_rows + test_rows}
    problems = []
    for p in pairs:
        if p["meta"]["row_id"] in bad_ids:
            problems.append(f"{p['pair_id']}: row id {p['meta']['row_id']} occurs in val or test")
        if p["meta"]["chunk_id"] in bad_chunks:
            problems.append(f"{p['pair_id']}: chunk {p['meta']['chunk_id']} occurs in val or test")
    return problems


# ---------------------------------------------------------------- the pairs

def _pair(row: dict, ptype: int, chosen: str, rejected: str, rejected_source: str, base_item: dict | None,
          repeat_no: int | None = None) -> dict:
    """distinct_pair_id is shared by the repeats of one pair (Stage B: one reference log-prob per distinct pair)."""
    m = row["meta"]
    distinct = f"{m['id']}:t{ptype}"
    return {"pair_id": distinct if repeat_no is None else f"{distinct}:r{repeat_no}", "pair_type": ptype,
            "prompt": row["prompt"], "chosen": chosen, "rejected": rejected,
            "meta": {"row_id": m["id"], "chunk_id": m["chunk_id"], "source_id": m["source_id"],
                     "row_type": m["type"], "rejected_source": rejected_source, "distinct_pair_id": distinct,
                     "repeat_no": repeat_no or 1,
                     "base_token_f1": base_item["token_f1"] if base_item else None}}


TYPE2_DROPPED_REASON = ("dropped by reviewer: proxy cannot separate wrong from terse-correct "
                       "(all 54 read by hand)")
ACCEPTED_BY = {("key_recall_low",): "key_recall_low_only", ("unsupported_number",): "unsupported_number_only",
               ("key_recall_low", "unsupported_number"): "both"}


def untuned_scores(row: dict, output: str, refusal_text: str) -> dict:
    """key_recall and number_faithfulness of the untuned output, from the evaluator's own metric functions (the same
    ones that wrote them into eval_base_train.json). None = the metric does not apply to this item."""
    context, question, reference = parse_example(row)
    return {"key_recall": M.key_recall(output, reference, question, refusal_text),
            "number_faithfulness": M.number_faithfulness(output, context)}


def type2_reasons(scores: dict, cfg4: dict) -> list[str]:
    """Why a low-F1 untuned output counts as a wrong answer: [] = it does not (key words found, numbers supported).
    A score that does not apply (None) never counts as a reason."""
    reasons = []
    if scores["key_recall"] is not None and scores["key_recall"] < cfg4["pairs"]["rejected_key_recall_below"]:
        reasons.append("key_recall_low")
    if scores["number_faithfulness"] is not None and scores["number_faithfulness"] < 1:
        reasons.append("unsupported_number")
    return reasons


def build_pairs(train_rows: list[dict], base_items: dict[str, dict], refusal_text: str, cfg4: dict,
                exclude_ids: set[str] | None = None, loose_phrases: list[str] | None = None
                ) -> tuple[list[dict], dict, list[dict]]:
    """(pairs, counts, excluded_type2). Order: all type 1 pairs, then type 2, then type 3, each in the order of the
    train file (deterministic). `base_items` maps row id -> the evaluator record of the untuned output.
    excluded_type2 = rows with a low F1 that the key / number rule did NOT accept (shown in the report)."""
    p = cfg4["pairs"]
    repeat, loose_phrases = p["refusal_pair_repeat"], loose_phrases or []
    exclude_ids = exclude_ids or set()
    t1, t2, t3, excluded = [], [], [], []
    c = {"rows_in_file": len(train_rows), "rows_excluded_val_test_overlap": 0,
         "type2_skipped_base_refusal_or_empty": 0, "type2_excluded_f1_not_below_threshold": 0,
         "type2_f1_below_threshold": 0,                  # rows that pass the F1 test (accepted or not)
         "type2_accepted_by": {"key_recall_low_only": 0, "unsupported_number_only": 0, "both": 0},
         "type2_excluded_key_ok_and_numbers_ok": 0,      # F1 low, but key words found and no unsupported number
         "type2_excluded_key_recall_not_applicable": 0,  # of those: key_recall did not apply (None)
         "type3_skipped_base_strict_refusal": 0, "type3_skipped_base_loose_refusal_only": 0, "type3_skipped_empty": 0}
    for row in train_rows:
        rid = row["meta"]["id"]
        if rid in exclude_ids:
            c["rows_excluded_val_test_overlap"] += 1
            continue
        item = base_items[rid]
        out = item["output"].strip()
        _, _, reference = parse_example(row)
        if is_refusal_row(row):
            if not out:
                c["type3_skipped_empty"] += 1
            elif M.is_refusal_strict(out, refusal_text):
                c["type3_skipped_base_strict_refusal"] += 1
            elif M.is_refusal_loose(out, refusal_text, loose_phrases):
                c["type3_skipped_base_loose_refusal_only"] += 1
            else:
                t3.extend(_pair(row, 3, refusal_text, out, "untuned_base_greedy", item, k) for k in range(1, repeat + 1))
            continue
        t1.append(_pair(row, 1, reference, refusal_text, "refusal_text", None))
        if not p["include_type2"]:
            continue
        if not out or M.is_refusal_strict(out, refusal_text):
            c["type2_skipped_base_refusal_or_empty"] += 1
        elif item["token_f1"] >= p["rejected_f1_below"]:
            c["type2_excluded_f1_not_below_threshold"] += 1
        else:
            c["type2_f1_below_threshold"] += 1
            scores = untuned_scores(row, out, refusal_text)
            reasons = type2_reasons(scores, cfg4)
            if reasons:
                c["type2_accepted_by"][ACCEPTED_BY[tuple(reasons)]] += 1
                pair = _pair(row, 2, reference, out, "untuned_base_greedy", item)
                pair["meta"].update({"why_rejected": reasons, **scores})
                t2.append(pair)
            else:
                c["type2_excluded_key_ok_and_numbers_ok"] += 1
                c["type2_excluded_key_recall_not_applicable"] += scores["key_recall"] is None
                excluded.append({"row_id": rid, "question": parse_example(row)[1], "reference": reference,
                                 "output": out, "token_f1": item["token_f1"], **scores})
    c["type2_included"] = p["include_type2"]
    c["type2_status"] = (TYPE2_DROPPED_REASON if not p["include_type2"] else "included")
    c["type2_dropped_too_few"] = p["include_type2"] and len(t2) < p["type2_min_pairs"]
    if c["type2_dropped_too_few"]:                       # too few to be a real part of the data: type 2 is removed
        c["type2_pairs_before_drop"], t2 = len(t2), []
    c.update({"type1": len(t1), "type2": len(t2), "type3_distinct": len(t3) // repeat, "type3_weighted": len(t3)})
    c["distinct_pairs"] = len(t1) + len(t2) + c["type3_distinct"]
    c["weighted_total"] = len(t1) + len(t2) + len(t3)
    return t1 + t2 + t3, c, excluded


# ---------------------------------------------------------------- statistics for the report

def histogram(values: list[float], bins: list[float]) -> list[dict]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        out.append({"from": lo, "to": min(hi, 1.0), "count": sum(1 for v in values if lo <= v < hi)})
    return out


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def summarize(pairs: list[dict], counts: dict, train_rows: list[dict], base_items: dict[str, dict], cfg4: dict,
              refusal_text: str, input_hashes: dict, overlap_problems: list[str],
              excluded_type2: list[dict] | None = None) -> dict:
    p = cfg4["pairs"]
    answerable_ids = [r["meta"]["id"] for r in train_rows if not is_refusal_row(r)]
    f1_all = [base_items[i]["token_f1"] for i in answerable_ids]
    by_type = {t: [x for x in pairs if x["pair_type"] == t] for t in (1, 2, 3)}
    return {
        "counts": counts,
        "rejected_f1_below": p["rejected_f1_below"], "rejected_key_recall_below": p["rejected_key_recall_below"],
        "include_type2": p["include_type2"], "type2_status": counts["type2_status"],
        "type2_min_pairs": p["type2_min_pairs"], "refusal_pair_repeat": p["refusal_pair_repeat"],
        "distinct_pairs": counts["distinct_pairs"], "weighted_total": counts["weighted_total"],
        "stage_b_note": "reference log-probabilities: ONE computation per distinct pair (meta.distinct_pair_id); the "
                        "type 3 repeats reuse it",
        "base_f1_all_answerable_train_rows": {**describe(f1_all), "histogram": histogram(f1_all, p["f1_bins"]),
                                              "below_threshold": sum(1 for v in f1_all if v < p["rejected_f1_below"])},
        "base_f1_type2_excluded_by_key_and_number_rule": describe([x["token_f1"] for x in excluded_type2 or []]),
        "base_f1_type2_rejected": {**describe([x["meta"]["base_token_f1"] for x in by_type[2]]),
                                   "histogram": histogram([x["meta"]["base_token_f1"] for x in by_type[2]], p["f1_bins"])},
        "base_strict_refusals_on_train_answerable": sum(1 for i in answerable_ids if base_items[i]["refusal_strict"]),
        "base_strict_refusals_on_train_refusal_rows": sum(
            1 for r in train_rows if is_refusal_row(r) and base_items[r["meta"]["id"]]["refusal_strict"]),
        "length_chars_median": {str(t): {"chosen": statistics.median(len(x["chosen"]) for x in by_type[t]),
                                         "rejected": statistics.median(len(x["rejected"]) for x in by_type[t])}
                                for t in (1, 2, 3) if by_type[t]},
        "refusal_text": refusal_text,
        "type2_excluded_example_ids": [x["row_id"] for x in excluded_type2 or []],
        "leakage": {"problems": overlap_problems, "clean": not overlap_problems},
        "input_sha256": input_hashes,
    }


def pick_samples(pairs: list[dict], per_type: int, seed: int) -> dict[int, list[dict]]:
    """Seeded, so the same pairs are shown every time; shown in the order of the file."""
    rng, out = random.Random(seed), {}
    for t in (1, 2, 3):
        of_type = [x for x in pairs if x["pair_type"] == t and x["meta"]["repeat_no"] == 1]   # repeats are copies
        n = per_type[t] if isinstance(per_type, dict) else per_type
        chosen = sorted(rng.sample(range(len(of_type)), min(n, len(of_type))))
        out[t] = [of_type[i] for i in chosen]
    return out


def pick_excluded(excluded: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [excluded[i] for i in sorted(rng.sample(range(len(excluded)), min(n, len(excluded))))]


def format_excluded(x: dict) -> str:
    kr = "n/a" if x["key_recall"] is None else f"{x['key_recall']:.3f}"
    nf = "n/a (no number)" if x["number_faithfulness"] is None else f"{x['number_faithfulness']:.3f}"
    return (f"[{x['row_id']}] untuned F1 {x['token_f1']:.3f} | key_recall {kr} | number faithfulness {nf}"
            f"\nQUESTION:  {x['question']}\nREFERENCE: {x['reference']}\nUNTUNED:   {x['output']}")


def format_pair(pair: dict) -> str:
    _, question, _ = parse_example({"prompt": pair["prompt"], "completion": [{"role": "assistant", "content": ""}]})
    f1 = pair["meta"]["base_token_f1"]
    head = (f"[{pair['pair_id']}] chunk {pair['meta']['chunk_id']} | rejected from: {pair['meta']['rejected_source']}"
            + (f" | untuned F1 vs reference {f1:.3f}" if f1 is not None else "")
            + (f" | why rejected: {', '.join(pair['meta']['why_rejected'])} (key_recall {pair['meta']['key_recall']}, "
               f"number faithfulness {pair['meta']['number_faithfulness']})" if "why_rejected" in pair["meta"] else ""))
    return f"{head}\nQUESTION: {question}\nCHOSEN:   {pair['chosen']}\nREJECTED: {pair['rejected']}"


def pair_lines(pairs: list[dict]) -> str:
    return "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)


# ---------------------------------------------------------------- command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the DPO preference pairs from sft_train.jsonl and eval_base_train.json.")
    ap.add_argument("--show", action="store_true",
                    help="print sample pairs of type 2 and 3 and the excluded type 2 examples, in full")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):            # the samples contain characters a Windows console cannot print
        sys.stdout.reconfigure(encoding="utf-8")
    cfg4, ecfg = load_phase4_config(), load_eval_config()
    refusal = ecfg["refusal_text"]
    P = {k: REPO_ROOT / v for k, v in cfg4["paths"].items()}
    train_rows, val_rows, test_rows = read_jsonl(P["sft_train"]), read_jsonl(P["sft_val"]), read_jsonl(P["sft_test"])
    if not P["base_train_eval"].exists():
        print(f"{P['base_train_eval']} is missing. Run cell P4-0 on Colab and put the file into outputs/.")
        return 1
    base = json.loads(P["base_train_eval"].read_text(encoding="utf-8"))
    check_base_eval(base, train_rows, file_sha256(P["sft_train"]))
    base_items = {it["id"]: it for it in base["items"]}
    excluded = rows_to_exclude(train_rows, val_rows, test_rows)
    pairs, counts, excluded2 = build_pairs(train_rows, base_items, refusal, cfg4, excluded,
                                           ecfg["eval"]["loose_refusal_phrases"])
    problems = leakage_problems(pairs, val_rows, test_rows)
    if problems:
        print("LEAKAGE, nothing written:\n" + "\n".join(problems))
        return 1
    P["pairs"].write_text(pair_lines(pairs), encoding="utf-8")
    sha = file_sha256(P["pairs"])
    P["pairs_sha256"].write_text(f"{sha}  {P['pairs'].name}\n", encoding="utf-8")
    hashes = {"sft_train": file_sha256(P["sft_train"]), "sft_val": file_sha256(P["sft_val"]),
              "sft_test": file_sha256(P["sft_test"]), "eval_base_train": file_sha256(P["base_train_eval"]),
              "dpo_train": sha}
    summary = summarize(pairs, counts, train_rows, base_items, cfg4, refusal, hashes, problems, excluded2)
    P["pairs_summary"].write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1))
    print("dpo_train.jsonl sha256", sha)
    if a.show:
        sizes, seed = cfg4["pairs"]["samples"], cfg4["pairs"]["sample_seed"]
        for t, items in pick_samples(pairs, sizes, seed).items():
            if not items:
                continue
            print(f"\n===== TYPE {t}: {TYPE_NAMES[t]} ({len(items)} samples) =====")
            for x in items:
                print(format_pair(x) + "\n")
        if cfg4["pairs"]["include_type2"]:
            shown = pick_excluded(excluded2, sizes[2], seed)
            print(f"\n===== TYPE 2 EXCLUDED: low F1, but key words found and no unsupported number "
                  f"({len(shown)} of {len(excluded2)}) =====")
            for x in shown:
                print(format_excluded(x) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
