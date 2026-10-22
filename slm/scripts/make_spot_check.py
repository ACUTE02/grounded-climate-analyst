"""A2 F5: blind owner spot-check CSVs (UTF-8 with BOM, quoted, for Excel) + a separate key file."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.sft import parse_example
from climate_slm.data.spotcheck import CSV_COLUMNS, KEY_COLUMNS, sample_spot_check, shuffled, stratum_of
from climate_slm.data.textutil import read_jsonl


def to_row(ex: dict) -> dict:
    chunk, q, a = parse_example(ex)
    is_ref = ex["meta"]["type"] == "unanswerable"
    reason = ex.get("ai_reason", "")
    if ex.get("ai_fix"):
        reason += " [AI changed the item]"
    return {"id": ex["id"], "split": ex["meta"]["split"], "source_id": ex["meta"]["source_id"],
            "chunk_text": chunk, "question": q, "answer": a,
            "evidence_quote": "" if is_ref else " | ".join(ex.get("evidence_quotes") or []),
            "owner_verdict": "", "owner_note": "", "is_refusal": is_ref,
            "ai_verdict": ex.get("ai_verdict", ""), "ai_reason": reason}


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    cfg = load_config()
    s = cfg["spot_check"]
    pool_tr = [to_row(e) for e in read_jsonl(resolve(cfg, "candidates")) if e.get("dropped_by") is None]
    pool_te = [to_row(e) for e in read_jsonl(resolve(cfg, "test_candidates")) if e.get("dropped_by") is None]
    tr = sample_spot_check(pool_tr, s["spot_check_train_n"], s["seed"], s["spot_min_refusal"],
                           s["spot_min_ai_fixed_or_failed"])
    te = sample_spot_check(pool_te, s["spot_check_test_n"], s["seed"], s["spot_min_refusal_test"], 0)
    tr, te = shuffled(tr, s["seed"]), shuffled(te, s["seed"] + 1)
    write_csv(resolve(cfg, "spot_check_train"), tr, CSV_COLUMNS)
    write_csv(resolve(cfg, "spot_check_test"), te, CSV_COLUMNS)
    key = [{"id": r["id"], "stratum": stratum_of(r), "ai_verdict": r["ai_verdict"], "ai_reason": r["ai_reason"]}
           for r in tr + te]
    write_csv(resolve(cfg, "spot_check_key"), key, KEY_COLUMNS)
    summary = {
        "train_rows": len(tr), "train_strata": dict(Counter(stratum_of(r) for r in tr)),
        "train_refusals": sum(r["is_refusal"] for r in tr),
        "test_rows": len(te), "test_strata": dict(Counter(stratum_of(r) for r in te)),
        "test_refusals": sum(r["is_refusal"] for r in te),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
