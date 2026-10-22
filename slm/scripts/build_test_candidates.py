"""A2 build of the test candidates (test chunks only): answerable drafts, three refusal strategies,
leakage filter against the training/validation items. Nothing is deleted."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.a2 import INSTR, boiler_for, load_chunks, load_test_drafts
from climate_slm.data.a2build import assemble, flag_leakage
from climate_slm.data.build import run_checks
from climate_slm.data.textutil import read_jsonl, write_jsonl
from build_candidates import load_can_answer


def main():
    cfg = load_config()
    chunks = load_chunks(cfg)
    boiler = boiler_for(chunks, cfg)
    drafts, near = load_test_drafts(cfg, chunks)
    plan = json.loads(resolve(cfg, "refusal_plan_test").read_text(encoding="utf-8"))
    leaked_ans, leaked_near = set(plan["leaked_answerable_idx"]), set(plan["leaked_near_idx"])
    can_answer = load_can_answer(resolve(cfg, "verify_dir") / "verdicts_partner_test.jsonl")
    assert can_answer, "partner verdicts missing: run the partner verifier first"

    keep_d = [d for d in drafts if d["idx"] not in leaked_ans]
    keep_n = [n for i, n in enumerate(near) if i not in leaked_near]
    rows, info = assemble(cfg, chunks, boiler, keep_d, keep_n, plan, can_answer, "llm_drafted_test", "te2-")
    # leaked drafts stay in the file, flagged, with the matching training id
    n0 = len(rows)
    by_id = {c["id"]: c for c in chunks}
    from climate_slm.data.sft import make_example
    extra = []
    for d in drafts:
        if d["idx"] in leaked_ans:
            meta = {"chunk_id": d["chunk_id"], "source_id": d["source_id"], "split": "test", "type": d["type"],
                    "generator": "llm_drafted_test", "verification": "unverified"}
            ex = make_example(cfg, by_id[d["chunk_id"]]["text"], d["question"], d["answer"], meta)
            ex["dropped_by"] = "leakage_filter"
            ex["leakage"] = plan["leakage_hits"][f"a{d['idx']}"]
            extra.append(ex)
    for i, n in enumerate(near):
        if i in leaked_near:
            meta = {"chunk_id": n["chunk_id"], "source_id": by_id[n["chunk_id"]]["source_id"], "split": "test",
                    "type": "unanswerable", "generator": "llm_drafted_test", "verification": "unverified",
                    "question_chunk_id": n["chunk_id"], "partner_chunk_id": n["chunk_id"],
                    "unanswerable_strategy": "near_miss", "needs_check": True}
            ex = make_example(cfg, by_id[n["chunk_id"]]["text"], n["question"], cfg["refusal_text"], meta)
            ex["dropped_by"] = "leakage_filter"
            ex["leakage"] = plan["leakage_hits"][f"n{i}"]
            extra.append(ex)
    for j, ex in enumerate(extra):
        ex["id"] = f"te2-{n0 + j + 1:04d}"
    rows += extra
    for r in rows:
        r.pop("_draft_idx", None)
        r["verdict_history"] = []
        r["from_extension_round"] = False

    # second look with the full filter on every active test row (catches new refusal questions too)
    ref = [r for r in read_jsonl(resolve(cfg, "candidates")) if r["dropped_by"] is None]
    flag_leakage(rows, ref, cfg)
    run_checks(rows, cfg)
    write_jsonl(resolve(cfg, "test_candidates"), rows)

    act = [r for r in rows if r["dropped_by"] is None]
    def kind(r): return "refusal" if r["meta"]["type"] == "unanswerable" else "answerable"
    summary = {
        "test_target_items": cfg["generator"]["test_target_items"],
        "drafted_answerable": len(drafts), "drafted_near_miss": len(near),
        "candidate_rows": len(rows), "active_rows": len(act),
        "dropped_by": dict(Counter(r["dropped_by"] for r in rows if r["dropped_by"])),
        "leakage_rejected": {r["id"]: r["leakage"] for r in rows if r["dropped_by"] == "leakage_filter"},
        "active_answerable": sum(1 for r in act if kind(r) == "answerable"),
        "active_refusals": sum(1 for r in act if kind(r) == "refusal"),
        "refusals_by_strategy": info["refusals_by_strategy"],
        "active_answerable_per_source": dict(Counter(r["meta"]["source_id"] for r in act if kind(r) == "answerable")),
        "distinct_partner_chunks": len(info["partner_use"]),
        "max_partner_reuse": max(info["partner_use"].values()) if info["partner_use"] else 0,
        "auto_flag_counts": dict(Counter(f for r in act for f in r["auto_flags"])),
    }
    print(json.dumps(summary, indent=1))
    (INSTR / "build_summary_test.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
