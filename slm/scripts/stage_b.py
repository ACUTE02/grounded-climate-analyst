"""Stage B: housekeeping on the filled spot-check files, disagreement rate and stop rule, final SFT files.

Steps: `housekeeping`, `disagreement`, `dry-run` (final sets in memory only, nothing written but the summary),
`finalize` (refuses to run when the stop rule says stop), `all` (the three real steps in order)."""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_verification import write_review_log
from climate_slm.config import REPO_ROOT, load_config, resolve
from climate_slm.data import stageb
from climate_slm.data.blind import token_f1
from climate_slm.data.sft import parse_example
from climate_slm.data.textutil import read_jsonl, write_jsonl

SPOT_FILES = {"train": ("spot_check_train", "candidates"), "test": ("spot_check_test", "test_candidates")}


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_spot(cfg: dict) -> dict[str, list[dict]]:
    """Spot-check rows joined with the key file (AI verdict, stratum) and the candidate kind."""
    key = {r["id"]: r for r in read_csv(resolve(cfg, "spot_check_key"))}
    out = {}
    for name, (spot_key, cand_key) in SPOT_FILES.items():
        cand = {r["id"]: r for r in read_jsonl(resolve(cfg, cand_key))}
        rows = read_csv(resolve(cfg, spot_key))
        for r in rows:
            k, c = key[r["id"]], cand[r["id"]]
            r.update(ai_verdict=k["ai_verdict"], ai_reason=k["ai_reason"], stratum=k["stratum"],
                     kind="refusal" if c["meta"]["type"] == "unanswerable" else "answerable",
                     ai_verdict_now=c.get("ai_verdict"), dropped_by_now=c["dropped_by"])
        out[name] = rows
    return out


def save_summary(cfg: dict, key: str, value) -> None:
    p = REPO_ROOT / cfg["stage_b"]["summary"]
    prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    prev[key] = value
    p.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")


def housekeeping(cfg: dict) -> dict:
    b = cfg["stage_b"]
    mismatches, bad_verdicts = [], []
    for name, (spot_key, cand_key) in SPOT_FILES.items():
        cand = {r["id"]: r for r in read_jsonl(resolve(cfg, cand_key))}
        for row in read_csv(resolve(cfg, spot_key)):
            ex = cand.get(row["id"])
            if ex is None:
                mismatches.append({"id": row["id"], "columns": ["id"]})
                continue
            chunk, q, a = parse_example(ex)
            is_ref = ex["meta"]["type"] == "unanswerable"
            want = {"id": ex["id"], "split": ex["meta"]["split"], "source_id": ex["meta"]["source_id"],
                    "chunk_text": chunk, "question": q, "answer": a,
                    "evidence_quote": "" if is_ref else " | ".join(ex.get("evidence_quotes") or [])}
            diff = [c for c in b["frozen_columns"] if row[c] != want[c]]
            if diff:
                mismatches.append({"id": row["id"], "columns": diff})
            if row["owner_verdict"] not in (b["owner_good"], b["owner_bad"]):
                bad_verdicts.append({"id": row["id"], "owner_verdict": row["owner_verdict"]})
    out = {"frozen_column_mismatches": mismatches, "rows_without_good_or_bad": bad_verdicts}
    if mismatches or bad_verdicts:
        print(json.dumps(out, indent=1))
        raise SystemExit("STOP: the spot-check files do not pass housekeeping. Hashes were not written.")

    hp = REPO_ROOT / cfg["a3"]["spot_check_hashes"]
    idir = resolve(cfg, "candidates").parent
    stored = json.loads(hp.read_text(encoding="utf-8"))
    before = stored.pop(b["hashes_before_key"], None)
    now = {n: hashlib.sha256((idir / n).read_bytes()).hexdigest() for n in stored}
    if before is None:  # first run after the owner filled the files
        hp.write_text(json.dumps({**now, b["hashes_before_key"]: stored}, indent=1), encoding="utf-8")
        out["hashes"] = "written (old values kept under %s)" % b["hashes_before_key"]
    elif now != stored:
        raise SystemExit("STOP: spot-check files changed after they were frozen: "
                         + ", ".join(n for n in now if now[n] != stored[n]))
    else:
        out["hashes"] = "unchanged since frozen"
    out["files_changed_by_owner_fill"] = [n for n in now if (before or stored)[n] != now[n]]
    print(json.dumps(out, indent=1))
    save_summary(cfg, "housekeeping", out)
    return out


def disagreement(cfg: dict) -> dict:
    spot = load_spot(cfg)
    tables = {name: stageb.disagreement_table(rows, cfg) for name, rows in spot.items()}
    decision = stageb.stop_decision(tables, cfg)
    listing = [{"id": r["id"], "file": name, "kind": r["kind"], "stratum": r["stratum"],
                "decided_by": "reviewer" if stageb.is_reviewer_decided(r["owner_note"], cfg) else "human",
                "owner_verdict": r["owner_verdict"], "owner_note": r["owner_note"],
                "ai_verdict": r["ai_verdict"], "ai_reason": r["ai_reason"]}
               for name, rows in spot.items() for r in rows if r["id"] in tables[name]["disagreeing_ids"]]
    key_vs_now = [{"id": r["id"], "key": r["ai_verdict"], "now": r["ai_verdict_now"]}
                  for rows in spot.values() for r in rows if r["ai_verdict"] != r["ai_verdict_now"]]
    out = {"tables": tables, "decision": decision, "disagreeing_rows": listing,
           "key_verdict_differs_from_candidates": key_vs_now}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("DECISION:", "STOP" if decision["stop"] else "continue")
    save_summary(cfg, "disagreement", out)
    return out


def blind_table(cfg: dict) -> list[dict]:
    out = []
    for r in read_csv(REPO_ROOT / cfg["a3"]["blind_report"]):
        row = {"id": r["id"], "split": r["split"], "flags": r["flags"], "reference_answer": r["reference_answer"],
               "blind_answer": r["blind_answer"]}
        if r["kind"] == "answerable":
            row["f1"] = round(token_f1(r["blind_answer"], r["reference_answer"]), 3)
        out.append(row)
    return out


def finalize(cfg: dict, dry_run: bool) -> dict:
    b = cfg["stage_b"]
    decision = stageb.stop_decision({n: stageb.disagreement_table(rows, cfg) for n, rows in load_spot(cfg).items()}, cfg)
    if decision["stop"] and not dry_run and not b["allow_finalize_after_stop"]:
        raise SystemExit("STOP RULE: final files are NOT written.\n" + "\n".join(decision["reasons"]))
    spot_rows = load_spot(cfg)
    spot = {r["id"]: r for rows in spot_rows.values() for r in rows}
    cand = read_jsonl(resolve(cfg, "candidates"))
    test = read_jsonl(resolve(cfg, "test_candidates"))
    for r in cand + test:  # a second run starts from the same state as the first one
        if r["dropped_by"] == b["owner_drop_label"]:
            r["dropped_by"] = None
    before = {"train": sum(stageb.is_final(r, {}, cfg) and r["meta"]["split"] == "train" for r in cand),
              "val": sum(stageb.is_final(r, {}, cfg) and r["meta"]["split"] == "val" for r in cand),
              "test": sum(stageb.is_final(r, {}, cfg) for r in test)}
    if dry_run:
        cand, test = copy.deepcopy(cand), copy.deepcopy(test)
    owner_bad = stageb.mark_owner_drops(cand, spot, cfg) + stageb.mark_owner_drops(test, spot, cfg)
    final = {"train": stageb.final_items([r for r in cand if r["meta"]["split"] == "train"], spot, cfg),
             "val": stageb.final_items([r for r in cand if r["meta"]["split"] == "val"], spot, cfg),
             "test": stageb.final_items(test, spot, cfg)}
    leak = stageb.final_leakage(final["train"], final["val"], final["test"], cfg)
    auto = {s: stageb.recheck(items, cfg) for s, items in final.items()}
    overridden = decision["stop"] and not dry_run
    out = {"dry_run": dry_run, "stop_rule": decision, "stop_rule_overridden": overridden,
           "override": ({"passed_on_purpose": True, "date": datetime.date.today().isoformat(),
                         "decided_in": b["override_source"], "reason": b["override_reason"],
                         "reviewer_adjudication": b["reviewer_adjudication"]} if overridden else None),
           "items_before_owner_drops": before, "owner_bad_ids": owner_bad,
           "owner_bad_removed_from_final": [i for i in owner_bad
                                            if spot[i]["dropped_by_now"] in (None, b["owner_drop_label"])
                                            and spot[i]["ai_verdict_now"] in b["ai_ok_verdicts"]],
           "counts": {s: stageb.counts(items) for s, items in final.items()},
           "val_minimums": stageb.val_minimums(final["val"], cfg),
           "auto_flags_on_final": auto, "leakage": leak,
           "length_stats": {s: stageb.length_stats(items, cfg) for s, items in final.items()},
           "dropped_refusals": stageb.refusal_siblings(cand, owner_bad) + stageb.refusal_siblings(test, owner_bad),
           "blind_flag_table": blind_table(cfg)}
    problems = [k for k, v in leak.items() if v] + [f"auto_flags_{s}" for s, v in auto.items() if v]
    out["problems"] = problems
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if not out["val_minimums"]["ok"]:
        print("WARNING: validation minimums not met (val_min_items / val_min_refusals). Nothing was padded.")
    save_summary(cfg, "dry_run_final" if dry_run else "final", out)
    if dry_run:
        print("DRY RUN: no final file and no candidate file was written.")
        return out
    if problems:
        raise SystemExit(f"STOP: problems on the final sets, files NOT written: {problems}")
    write_jsonl(resolve(cfg, "candidates"), cand)
    write_jsonl(resolve(cfg, "test_candidates"), test)
    write_review_log(resolve(cfg, "review_log"), {"train": cand, "test": test})
    for s, items in final.items():
        write_jsonl(REPO_ROOT / b[f"sft_{s}"], items)
        print("wrote", b[f"sft_{s}"], len(items))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["housekeeping", "disagreement", "dry-run", "finalize", "all"])
    step = ap.parse_args().step
    cfg = load_config()
    if step in ("housekeeping", "all"):
        housekeeping(cfg)
    if step in ("disagreement", "all"):
        disagreement(cfg)
    if step in ("finalize", "all"):
        finalize(cfg, dry_run=False)
    if step == "dry-run":
        finalize(cfg, dry_run=True)
