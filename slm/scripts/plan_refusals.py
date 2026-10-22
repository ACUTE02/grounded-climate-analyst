"""A2 F1/F2: decide which questions become refusals, list partner candidates, write the input for the
independent partner verifier. Usage: plan_refusals.py --set train | test"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.a2 import NEARMISS_TRAIN, boiler_for, load_chunks, load_nearmiss, load_test_drafts, load_train_drafts
from climate_slm.data.a2build import cap_dropped, plan_refusals
from climate_slm.data.leakage import leakage_hits
from climate_slm.data.sft import make_example
from climate_slm.data.textutil import read_jsonl


def _ex(cfg, chunk_by_id, cid, question, answer, i, kind):
    meta = {"chunk_id": cid, "source_id": chunk_by_id[cid]["source_id"], "split": chunk_by_id[cid]["split"],
            "type": kind}
    ex = make_example(cfg, chunk_by_id[cid]["text"], question, answer, meta)
    ex["id"] = f"x{i}"
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["train", "test"])
    a = ap.parse_args()
    cfg = load_config()
    chunks = load_chunks(cfg)
    by_id = {c["id"]: c for c in chunks}
    boiler = boiler_for(chunks, cfg)
    g, u = cfg["generator"], cfg["unanswerable"]

    if a.set == "train":
        drafts = load_train_drafts(cfg, chunks)
        near = load_nearmiss(NEARMISS_TRAIN, [c["id"] for c in chunks], full_ids=False)
        spare = cap_dropped(drafts, chunks, cfg)
        a0 = len(drafts) - len(spare)
        total = round(u["ratio"] / (1 - u["ratio"]) * a0)
        plan = plan_refusals(cfg, chunks, boiler, drafts, near, a0, total, spare)
        plan["spare_draft_idx_count"] = len(spare)
    else:
        drafts, near = load_test_drafts(cfg, chunks)
        ref = [r for r in read_jsonl(resolve(cfg, "candidates")) if r.get("dropped_by") is None]
        ans_ex = [_ex(cfg, by_id, d["chunk_id"], d["question"], d["answer"], d["idx"], d["type"]) for d in drafts]
        for e, d in zip(ans_ex, drafts):
            e["id"] = f"a{d['idx']}"
        near_ex = [_ex(cfg, by_id, n["chunk_id"], n["question"], cfg["refusal_text"], i, "unanswerable")
                   for i, n in enumerate(near)]
        for i, e in enumerate(near_ex):
            e["id"] = f"n{i}"
        hits = leakage_hits(ans_ex + near_ex, ref, cfg)
        leaked_ans = sorted(int(k[1:]) for k in hits if k.startswith("a"))
        leaked_near = sorted(int(k[1:]) for k in hits if k.startswith("n"))
        keep_d = [d for d in drafts if d["idx"] not in leaked_ans]
        keep_n = [n for i, n in enumerate(near) if i not in leaked_near]
        r_target = max(g["test_min_refusals"], round(g["test_refusal_ratio"] * g["test_target_items"]))
        total = math.ceil(r_target * g["test_refusal_oversample"])
        plan = plan_refusals(cfg, chunks, boiler, keep_d, keep_n, len(keep_d), total, set())
        plan["refusal_target_valid"] = r_target
        plan["leaked_answerable_idx"] = leaked_ans
        plan["leaked_near_idx"] = leaked_near
        plan["leakage_hits"] = {k: v for k, v in hits.items()}
        plan["near_miss_available_after_filter"] = len(keep_n)

    out = resolve(cfg, f"refusal_plan_{a.set}")
    out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")

    need = sorted({cid for it in plan["same_topic"] for cid in it["pool"]})
    vdir = resolve(cfg, "verify_dir")
    vdir.mkdir(parents=True, exist_ok=True)
    vin = {"chunks": {cid: by_id[cid]["text"] for cid in need},
           "items": [{"id": it["key"], "question": it["question"], "candidates": it["pool"]} for it in plan["same_topic"]]}
    (vdir / f"input_partner_{a.set}.json").write_text(json.dumps(vin, ensure_ascii=False, indent=1), encoding="utf-8")
    show = {k: v for k, v in plan.items() if k not in ("same_topic", "random", "leakage_hits")}
    show["same_topic_items"] = len(plan["same_topic"])
    show["random_items"] = len(plan["random"])
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
