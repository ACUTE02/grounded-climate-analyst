"""A3 G1 + G3: reviewer drops, val exempt from the cap, new val items. Only ADDS rows or flags; no id is
renumbered and nothing is deleted. Phases: plan (writes the partner-check input) and build."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_config, resolve
from climate_slm.data.a2 import boiler_for, load_chunks
from climate_slm.data.a2build import flag_leakage, make_refusal
from climate_slm.data.build import resolve_chunk_id, run_checks
from climate_slm.data.refusals import assign_partners, candidate_pool
from climate_slm.data.sft import make_example
from climate_slm.data.textutil import read_jsonl, write_jsonl
from build_candidates import load_can_answer

GEN = "llm_drafted"


def read_lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def g1_and_readmit(rows: list[dict], cfg: dict) -> dict:
    a3 = cfg["a3"]
    n_drop = n_readmit = 0
    for r in rows:
        if r["id"] in a3["reviewer_drop_ids"] and r["dropped_by"] != "reviewer_override":
            assert r["dropped_by"] is None and r["meta"]["type"] == "unanswerable", r["id"]
            r["dropped_by"] = "reviewer_override"
            r["dropped_reason"] = a3["reviewer_drop_reason"].strip()
            n_drop += 1
        if (not a3["cap_applies_to_val"] and r["dropped_by"] == "source_cap" and r["meta"]["split"] == "val"):
            r["dropped_by"] = None
            r["readmitted_by"] = "a3_val_exempt_from_cap"
            n_readmit += 1
    return {"reviewer_dropped_now": n_drop, "val_readmitted_now": n_readmit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["plan", "build"])
    a = ap.parse_args()
    cfg = load_config()
    a3, u = cfg["a3"], cfg["unanswerable"]
    chunks = load_chunks(cfg)
    by_id = {c["id"]: c for c in chunks}
    ids = list(by_id)
    boiler = boiler_for(chunks, cfg)
    extra = REPO_ROOT / a3["val_extra_dir"]
    vdir = resolve(cfg, "verify_dir")
    rows = read_jsonl(resolve(cfg, "candidates"))
    info = g1_and_readmit(rows, cfg)

    drafts = []
    for d in read_lines(extra / "val_answerable.jsonl"):
        cid = resolve_chunk_id(d["c"], ids)
        assert by_id[cid]["split"] == "val" and not boiler[cid], cid
        drafts.append({"key": len(drafts), "chunk_id": cid, "question": d["q"], "answer": d["a"]})
    near = []
    for d in read_lines(extra / "val_nearmiss.jsonl"):
        cid = resolve_chunk_id(d["c"], ids)
        assert by_id[cid]["split"] == "val" and not boiler[cid], cid
        near.append({"chunk_id": cid, "question": d["q"]})

    rng = random.Random(u["seed"] + 3)
    n_same = a3["val_new_refusals_same_topic"]
    sel = list(range(len(drafts)))
    rng.shuffle(sel)
    same_keys = sel[: n_same + a3["val_same_topic_oversample"]]
    rand_keys = sel[len(same_keys): len(same_keys) + a3["val_new_refusals_random"]]

    if a.phase == "plan":
        items = []
        for k in same_keys:
            d = drafts[k]
            pool = candidate_pool(d["question"], by_id[d["chunk_id"]], chunks, boiler, u["partner_pool_k"])
            items.append({"key": f"v3st-{k}", "home": d["chunk_id"], "question": d["question"], "pool": pool})
        need = sorted({c for it in items for c in it["pool"]})
        (vdir / "input_partner_val3.json").write_text(json.dumps({
            "chunks": {c: by_id[c]["text"] for c in need},
            "items": [{"id": it["key"], "question": it["question"], "candidates": it["pool"]} for it in items]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        (extra / "plan_val3.json").write_text(json.dumps({"same": items, "rand_keys": rand_keys}, indent=1), encoding="utf-8")
        write_jsonl(resolve(cfg, "candidates"), rows)  # G1 flags + readmit
        print(json.dumps({**info, "new_val_answerable_drafts": len(drafts), "near_miss": len(near),
                          "same_topic_sent_to_verifier": len(items), "random": len(rand_keys)}, indent=1))
        return

    # ---- build ----
    if any(r["id"].startswith("v3-") for r in rows):
        write_jsonl(resolve(cfg, "candidates"), rows)
        print("v3 rows already exist; only G1/readmit flags refreshed", info)
        return
    plan = json.loads((extra / "plan_val3.json").read_text(encoding="utf-8"))
    can = load_can_answer(vdir / "verdicts_partner_val3.jsonl")
    assert can, "run the partner verifier for val first"
    used = Counter()
    for r in rows:
        if r["meta"]["type"] == "unanswerable" and r["dropped_by"] is None and r["meta"]["split"] == "val":
            used[r["meta"]["partner_chunk_id"]] += 1
    new: list[dict] = []
    for n in near:
        cid = n["chunk_id"]
        if used[cid] < u["max_partner_reuse"]:
            new.append(make_refusal(cfg, by_id[cid], cid, n["question"], "near_miss", "val", GEN))
            used[cid] += 1
    res = assign_partners([{"key": s["key"], "home": s["home"], "pool": s["pool"]} for s in plan["same"]],
                          [{"key": f"v3rn-{k}", "home": drafts[k]["chunk_id"]} for k in plan["rand_keys"]],
                          chunks, boiler, can, u["max_partner_reuse"], random.Random(u["seed"] + 4), used)
    converted, got = set(), 0
    for s in plan["same"]:
        pick = res["same_topic"].get(s["key"])
        k = int(s["key"].split("-")[1])
        if pick and got < a3["val_new_refusals_same_topic"]:
            new.append(make_refusal(cfg, by_id[pick], s["home"], s["question"], "same_topic_missing", "val", GEN))
            converted.add(k)
            got += 1
    for k in plan["rand_keys"]:
        pick = res["random"].get(f"v3rn-{k}")
        if pick:
            new.append(make_refusal(cfg, by_id[pick], drafts[k]["chunk_id"], drafts[k]["question"],
                                    "random_other_source", "val", GEN))
            converted.add(k)
    for d in drafts:
        if d["key"] in converted:
            continue
        meta = {"chunk_id": d["chunk_id"], "source_id": by_id[d["chunk_id"]]["source_id"], "split": "val",
                "type": "definition", "generator": GEN, "verification": "unverified"}
        new.append(make_example(cfg, by_id[d["chunk_id"]]["text"], d["question"], d["answer"], meta))
    for i, ex in enumerate(new):
        ex["id"] = f"v3-{i + 1:04d}"
        ex["dropped_by"] = None
        ex["verdict_history"] = []
        ex["from_extension_round"] = False
        ex["added_by"] = "a3_val_extension"
    rows_all = rows + new
    # leakage filter against training for every val item that was never verified (readmitted + new)
    train_ref = [r for r in rows if r["meta"]["split"] == "train" and r["dropped_by"] is None]
    todo = [r for r in rows_all if r["meta"]["split"] == "val" and r["dropped_by"] is None and "ai_verdict" not in r]
    hits = flag_leakage(todo, train_ref, cfg)
    run_checks(rows_all, cfg)
    write_jsonl(resolve(cfg, "candidates"), rows_all)
    print(json.dumps({**info, "new_rows": len(new),
                      "new_refusals_by_strategy": dict(Counter(r["meta"]["unanswerable_strategy"] for r in new if r["meta"]["type"] == "unanswerable")),
                      "new_answerable": sum(1 for r in new if r["meta"]["type"] != "unanswerable"),
                      "leakage_rejected_now": {k: v for k, v in hits.items()},
                      "to_verify": sum(1 for r in rows_all if r["dropped_by"] is None and r["meta"]["split"] == "val" and "ai_verdict" not in r)},
                     indent=1))


if __name__ == "__main__":
    main()
