"""A2 build of the training / validation candidates: answerable drafts, three refusal strategies,
per-source cap, leakage filter (val against train), automatic checks. Nothing is deleted."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.a2 import INSTR, NEARMISS_TRAIN, boiler_for, load_chunks, load_nearmiss, load_train_drafts
from climate_slm.data.a2build import assemble, flag_leakage, source_val_share
from climate_slm.data.build import run_checks
from climate_slm.data.caps import apply_source_cap
from climate_slm.data.textutil import norm_ws, read_jsonl, write_jsonl


def load_can_answer(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                for cid, v in d["candidates"].items():
                    out[(d["id"], cid)] = v["can_answer"].strip().lower() != "no"
    return out


def history_map(cfg) -> dict:
    hist = {}
    for r in read_jsonl(resolve(cfg, "verdict_history")):
        if r["type"] == "answerable":
            hist[(r["chunk_id"], norm_ws(r["question"]))] = {
                "stage": "A", "ai_verdict": r["ai_verdict"], "ai_reason": r["ai_reason"],
                "evidence_quote": r["evidence_quote"]}
    return hist


def main():
    cfg = load_config()
    chunks = load_chunks(cfg)
    boiler = boiler_for(chunks, cfg)
    drafts = load_train_drafts(cfg, chunks)
    near = load_nearmiss(NEARMISS_TRAIN, [c["id"] for c in chunks], full_ids=False)
    plan = json.loads(resolve(cfg, "refusal_plan_train").read_text(encoding="utf-8"))
    can_answer = load_can_answer(resolve(cfg, "verify_dir") / "verdicts_partner_train.jsonl")
    assert can_answer, "partner verdicts missing: run the partner verifier first"

    rows, info = assemble(cfg, chunks, boiler, drafts, near, plan, can_answer, "llm_drafted", "tr2-")
    dmap = {d["idx"]: d for d in drafts}
    hist = history_map(cfg)
    worklist = {c for ids in json.loads(resolve(cfg, "worklist").read_text(encoding="utf-8")).values() for c in ids}

    # per-source cap on answerable items (train + val together)
    dropped_ids: set[str] = set()
    for src, cap in cfg["generator"]["answerable_cap_per_source"].items():
        items = []
        for r in rows:
            if r["meta"]["type"] != "unanswerable" and r["meta"]["source_id"] == src:
                d = dmap[r["_draft_idx"]]
                items.append({"id": r["id"], "chunk_id": d["chunk_id"], "split": d["split"], "style": d["style"],
                              "prior_pass": d["prior_verdict"] == "pass"})
        dropped_ids |= apply_source_cap(items, cap, source_val_share(chunks, src), cfg["generator"]["cap_seed_final"])
    if not cfg["a3"]["cap_applies_to_val"]:  # A3: validation is exempt from the cap
        dropped_ids -= {r["id"] for r in rows if r["meta"]["split"] == "val"}
    for r in rows:
        r["verdict_history"] = []
        if r["meta"]["type"] != "unanswerable":
            d = dmap[r.pop("_draft_idx")]
            r["from_extension_round"] = d["chunk_id"] not in worklist
            h = hist.get((d["chunk_id"], norm_ws(d["question"])))
            if h:
                r["verdict_history"].append(h)
        if r["id"] in dropped_ids:
            r["dropped_by"] = "source_cap"

    # leakage: validation items against training items (train split, still active)
    train_ref = [r for r in rows if r["meta"]["split"] == "train" and r["dropped_by"] is None]
    flag_leakage(rows, train_ref, cfg, only_split="val")
    run_checks(rows, cfg)
    write_jsonl(resolve(cfg, "candidates"), rows)

    act = [r for r in rows if r["dropped_by"] is None]
    def kind(r): return "refusal" if r["meta"]["type"] == "unanswerable" else "answerable"
    summary = {
        "drafted_answerable_questions": len(drafts),
        "candidate_rows": len(rows), "active_rows": len(act),
        "dropped_by": dict(Counter(r["dropped_by"] for r in rows if r["dropped_by"])),
        "active_by_split_and_kind": {f"{s}/{k}": sum(1 for r in act if r["meta"]["split"] == s and kind(r) == k)
                                     for s in ("train", "val") for k in ("answerable", "refusal")},
        "active_answerable_per_source": dict(Counter(r["meta"]["source_id"] for r in act if kind(r) == "answerable")),
        "active_refusal_per_source_of_partner": dict(Counter(r["meta"]["source_id"] for r in act if kind(r) == "refusal")),
        "refusal_share_of_active": round(sum(1 for r in act if kind(r) == "refusal") / len(act), 3),
        "refusals_by_strategy": info["refusals_by_strategy"],
        "distinct_partner_chunks": len(info["partner_use"]),
        "max_partner_reuse": max(info["partner_use"].values()) if info["partner_use"] else 0,
        "same_topic_without_valid_partner": info["same_topic_no_valid_partner"],
        "leakage_rejected_val": {r["id"]: r["leakage"] for r in rows if r["dropped_by"] == "leakage_filter"},
        "items_from_extension_round": sum(1 for r in rows if r.get("from_extension_round")),
        "active_items_from_extension_round": sum(1 for r in act if r.get("from_extension_round")),
        "auto_flag_counts": dict(Counter(f for r in act for f in r["auto_flags"])),
    }
    print(json.dumps(summary, indent=1))
    (INSTR / "build_summary_train.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
