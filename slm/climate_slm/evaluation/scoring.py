"""Turn raw outputs into per-item records and a summary. Shared by the baselines and the model evaluator.
Answerable and refusal items are always summarised separately."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from climate_slm.data.sft import parse_example
from climate_slm.evaluation import metrics as M

REVIEW_COLUMNS = ["id", "kind", "why_listed", "strict_refusal", "loose_refusal", "question", "reference", "output"]


def load_items(path: Path, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return items[:limit] if limit else items


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kind_of(item: dict) -> str:
    return "refusal" if item["meta"]["type"] == "unanswerable" else "answerable"


def score_record(item: dict, gen: dict, nll: tuple[float, int] | None, cfg: dict,
                 prompt: list[dict] | None = None) -> dict:
    """gen: {'text', 'n_new_tokens', 'hit_max_new_tokens'}; nll: (sum of NLL, completion tokens) or None.
    prompt: the messages that were really sent (a prompt variant); default = the stored prompt of the item."""
    e, refusal = cfg["eval"], cfg["refusal_text"]
    context, question, reference = parse_example(item)
    out = gen["text"]
    return {
        "id": item["meta"]["id"], "kind": kind_of(item), "source_id": item["meta"]["source_id"],
        "prompt_chars": sum(len(m["content"]) for m in (prompt or item["prompt"])),
        "question": question, "reference": reference, "output": out, "output_chars": len(out),
        "n_new_tokens": gen.get("n_new_tokens"), "hit_max_new_tokens": gen.get("hit_max_new_tokens"),
        "refusal_strict": M.is_refusal_strict(out, refusal),
        "refusal_loose": M.is_refusal_loose(out, refusal, e["loose_refusal_phrases"]),
        "token_f1": M.token_f1(out, reference), "rouge_l": M.rouge_l(out, reference),
        "number_faithfulness": M.number_faithfulness(out, context),
        "key_recall": M.key_recall(out, reference, question, refusal),
        "key_support": M.key_support(out, reference, question, refusal),
        "completion_nll_sum": nll[0] if nll else None, "completion_tokens": nll[1] if nll else None,
    }


def _ci(pair):
    return [pair[0], pair[1]] if pair else None


def _mean_block(values: list[float], e: dict) -> dict:
    return {"n": len(values), "mean": M.mean(values),
            "ci95": _ci(M.bootstrap_ci(values, e["n_boot"], e["boot_seed"], e["ci_level"]))}


def _rate_block(flags: list[bool], e: dict) -> dict:
    k, n = sum(flags), len(flags)
    return {"count": k, "n": n, "rate": k / n if n else None,
            "ci95": _ci(M.bootstrap_ci([int(f) for f in flags], e["n_boot"], e["boot_seed"], e["ci_level"])),
            "wilson95": _ci(M.wilson_ci(k, n))}


def key_blocks(answerable: list[dict], e: dict) -> dict:
    """key_recall and key_support (answerable items only): mean, interval, and on how many items they applied."""
    out = {}
    for name in ("key_recall", "key_support"):
        vals = [r[name] for r in answerable if r.get(name) is not None]
        out[name] = dict(_mean_block(vals, e), n_applied=len(vals), n_answerable=len(answerable))
    return out


def add_key_scores(ev: dict, cfg: dict) -> bool:
    """Fill key_recall / key_support into an eval dict that was written before Phase 2c (in memory only; the
    file on disk is not touched). Uses only the saved outputs. Returns True if anything was added."""
    if all("key_recall" in r for r in ev["items"]) and "key_recall" in ev["summary"]["answerable"]:
        return False
    refusal = cfg["refusal_text"]
    for r in ev["items"]:
        r["key_recall"] = M.key_recall(r["output"], r["reference"], r["question"], refusal)
        r["key_support"] = M.key_support(r["output"], r["reference"], r["question"], refusal)
    ev["summary"]["answerable"].update(key_blocks([r for r in ev["items"] if r["kind"] == "answerable"], cfg["eval"]))
    ev["key_scores_added_at_report_time"] = True
    return True


def bare_answer_table(records: list[dict], max_chars: int) -> dict:
    """Mean token F1, key_recall and key_support of the answerable items, split into short ("bare") outputs
    (fewer than max_chars characters) and longer ones. Shows how much of the F1 gap is style."""
    ans = [r for r in records if r["kind"] == "answerable"]
    out = {}
    for name, rows in (("bare", [r for r in ans if r["output_chars"] < max_chars]),
                       ("longer", [r for r in ans if r["output_chars"] >= max_chars])):
        def avg(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return M.mean(vals)
        out[name] = {"n": len(rows), "token_f1": avg("token_f1"), "key_recall": avg("key_recall"),
                     "key_support": avg("key_support")}
    return out


def _ppl(records: list[dict]) -> float | None:
    rows = [r for r in records if r["completion_tokens"]]
    tok = sum(r["completion_tokens"] for r in rows)
    return math.exp(sum(r["completion_nll_sum"] for r in rows) / tok) if tok else None


def _ppl_block(records: list[dict], e: dict) -> dict | None:
    rows = [r for r in records if r["completion_tokens"]]
    if not rows:
        return None
    return {"n": len(rows), "n_tokens": sum(r["completion_tokens"] for r in rows), "perplexity": _ppl(rows),
            "ci95": _ci(M.bootstrap_stat(rows, _ppl, e["n_boot"], e["boot_seed"], e["ci_level"]))}


def summarize(records: list[dict], cfg: dict) -> dict:
    e = cfg["eval"]
    ans = [r for r in records if r["kind"] == "answerable"]
    ref = [r for r in records if r["kind"] == "refusal"]
    nf = [r["number_faithfulness"] for r in ans if r["number_faithfulness"] is not None]
    lens = [r["output_chars"] for r in records]
    hit = [r["hit_max_new_tokens"] for r in records if r["hit_max_new_tokens"] is not None]
    return {
        "n_items": len(records),
        "answerable": {
            "n": len(ans),
            "token_f1": _mean_block([r["token_f1"] for r in ans], e),
            "rouge_l": _mean_block([r["rouge_l"] for r in ans], e),
            "number_faithfulness": dict(_mean_block(nf, e), n_applied=len(nf), n_answerable=len(ans)),
            **key_blocks(ans, e),
            "false_refusal_strict": _rate_block([r["refusal_strict"] for r in ans], e),
            "false_refusal_loose": _rate_block([r["refusal_loose"] for r in ans], e),
        },
        "refusal": {
            "n": len(ref),
            "correct_refusal_strict": _rate_block([r["refusal_strict"] for r in ref], e),
            "correct_refusal_loose": _rate_block([r["refusal_loose"] for r in ref], e),
            # "answered" = not even the loose match saw a refusal
            "answered_when_should_refuse": _rate_block([not r["refusal_loose"] for r in ref], e),
        },
        "perplexity": {"overall": _ppl_block(records, e), "answerable": _ppl_block(ans, e),
                       "refusal": _ppl_block(ref, e)},
        "output_length_chars": ({"min": min(lens), "median": statistics.median(lens), "mean": M.mean(lens),
                                 "max": max(lens)} if lens else None),
        "hit_max_new_tokens": {"count": sum(hit), "n": len(hit)} if hit else None,
    }


def review_rows(records: list[dict]) -> list[dict]:
    """Outputs a person should read: loose says refusal but strict does not, and refusal items that were answered."""
    rows = []
    for r in records:
        why = []
        if r["refusal_loose"] and not r["refusal_strict"]:
            why.append("loose_refusal_but_not_strict")
        if r["kind"] == "refusal" and not r["refusal_loose"]:
            why.append("refusal_item_was_answered")
        if why:
            rows.append({"id": r["id"], "kind": r["kind"], "why_listed": ";".join(why),
                         "strict_refusal": r["refusal_strict"], "loose_refusal": r["refusal_loose"],
                         "question": r["question"], "reference": r["reference"], "output": r["output"]})
    return rows


def write_review_csv(path: Path, records: list[dict]) -> int:
    rows = review_rows(records)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
