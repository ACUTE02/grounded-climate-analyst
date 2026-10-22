"""A2 build: refusals from three strategies, answerable examples, source cap, leakage flags."""
import math
import random
from collections import Counter

from climate_slm.data.caps import apply_source_cap
from climate_slm.data.leakage import leakage_hits
from climate_slm.data.refusals import allocate, assign_partners, candidate_pool
from climate_slm.data.sft import make_example


def source_val_share(chunks: list[dict], source_id: str) -> float:
    tv = [c for c in chunks if c["source_id"] == source_id and c["split"] in ("train", "val")]
    return (sum(1 for c in tv if c["split"] == "val") / len(tv)) if tv else 0.0


def cap_dropped(drafts: list[dict], chunks: list[dict], cfg: dict, exclude: set[int] = frozenset()) -> set[int]:
    """Draft idx dropped by the per-source cap (train + val together)."""
    dropped: set[int] = set()
    seed = cfg["generator"]["cap_seed_final"]
    for src, cap in cfg["generator"]["answerable_cap_per_source"].items():
        items = [{"id": str(d["idx"]), "chunk_id": d["chunk_id"], "split": d["split"], "style": d["style"],
                  "prior_pass": d["prior_verdict"] == "pass"}
                 for d in drafts if d["source_id"] == src and d["idx"] not in exclude]
        dropped |= {int(i) for i in apply_source_cap(items, cap, source_val_share(chunks, src), seed)}
    return dropped


def plan_refusals(cfg: dict, chunks: list[dict], boiler: dict, drafts: list[dict], near: list[dict],
                  n_answerable_est: int, refusal_total: int, spare: set[int]) -> dict:
    """Pick which drafted questions become refusals and which chunks are the candidate partners."""
    u = cfg["unanswerable"]
    by_id = {c["id"]: c for c in chunks}
    alloc = allocate(refusal_total, u["mix"])
    rng = random.Random(u["seed"])
    # near_miss: use the near-miss drafts that exist (test) - or ask for bases (train, near == [])
    if near:
        usable = [n for n in near if not boiler.get(n["chunk_id"])]
        n_near = min(len(usable), alloc["near_miss"])
        alloc["same_topic_missing"] += alloc["near_miss"] - n_near
        alloc["near_miss"] = n_near
        near_bases = []
    else:
        cand = sorted({d["chunk_id"] for d in drafts if not boiler.get(d["chunk_id"])})
        by_src: dict[str, list[str]] = {}
        for cid in cand:
            by_src.setdefault(by_id[cid]["source_id"], []).append(cid)
        for lst in by_src.values():
            rng.shuffle(lst)
        near_bases = []
        while len(near_bases) < alloc["near_miss"] and any(by_src.values()):
            for s in sorted(by_src):
                if by_src[s] and len(near_bases) < alloc["near_miss"]:
                    near_bases.append(by_src[s].pop())
    n_same = math.ceil(alloc["same_topic_missing"] * u["same_topic_oversample"])
    n_rand = alloc["random_other_source"]

    convertible = [d for d in drafts if not boiler.get(d["chunk_id"])]
    first = sorted((d for d in convertible if d["idx"] in spare), key=lambda d: d["idx"])
    rest = sorted((d for d in convertible if d["idx"] not in spare), key=lambda d: d["idx"])
    rng.shuffle(first)
    rng.shuffle(rest)
    order = first + rest
    same_items, rand_items = [], []
    for d in order:
        if len(same_items) < n_same:
            pool = candidate_pool(d["question"], by_id[d["chunk_id"]], chunks, boiler, u["partner_pool_k"])
            if pool:
                same_items.append({"key": f"st-{d['idx']}", "draft_idx": d["idx"], "home": d["chunk_id"],
                                   "question": d["question"], "pool": pool})
        elif len(rand_items) < n_rand:
            rand_items.append({"key": f"rn-{d['idx']}", "draft_idx": d["idx"], "home": d["chunk_id"],
                               "question": d["question"]})
        else:
            break
    return {"answerable_estimate": n_answerable_est, "refusal_total": refusal_total, "alloc": alloc,
            "same_topic_sent_to_verifier": n_same, "near_bases": near_bases,
            "same_topic": same_items, "random": rand_items}


def make_refusal(cfg, ctx: dict, home_id: str, question: str, strategy: str, split: str, generator: str) -> dict:
    meta = {"chunk_id": ctx["id"], "source_id": ctx["source_id"], "split": split, "type": "unanswerable",
            "generator": generator, "verification": "unverified", "question_chunk_id": home_id,
            "partner_chunk_id": ctx["id"], "unanswerable_strategy": strategy, "needs_check": True}
    return make_example(cfg, ctx["text"], question, cfg["refusal_text"], meta)


def assemble(cfg: dict, chunks: list[dict], boiler: dict, drafts: list[dict], near: list[dict], plan: dict,
             can_answer: dict, generator: str, id_prefix: str) -> tuple[list[dict], dict]:
    """Build all examples (answerable + refusals). Cap / leakage flags are added by the caller."""
    by_id = {c["id"]: c for c in chunks}
    u = cfg["unanswerable"]
    rng = random.Random(u["seed"] + 1)
    used = Counter()
    refusals: list[dict] = []
    converted: set[int] = set()

    # 1. near_miss: context is the same chunk
    n_near_target = plan["alloc"]["near_miss"]
    for n in near:
        if len([r for r in refusals if r["meta"]["unanswerable_strategy"] == "near_miss"]) >= n_near_target:
            break
        cid = n["chunk_id"]
        if boiler.get(cid) or used[cid] >= u["max_partner_reuse"]:
            continue
        refusals.append(make_refusal(cfg, by_id[cid], cid, n["question"], "near_miss", by_id[cid]["split"], generator))
        used[cid] += 1

    # 2 + 3. partners for same_topic_missing and random_other_source
    res = assign_partners(plan["same_topic"], plan["random"], chunks, boiler, can_answer,
                          u["max_partner_reuse"], rng, used)
    n_same_target = plan["alloc"]["same_topic_missing"]
    got_same = 0
    dmap = {d["idx"]: d for d in drafts}
    for it in plan["same_topic"]:
        pick = res["same_topic"].get(it["key"])
        if pick and got_same < n_same_target:
            d = dmap[it["draft_idx"]]
            refusals.append(make_refusal(cfg, by_id[pick], it["home"], it["question"], "same_topic_missing",
                                         d["split"], generator))
            converted.add(it["draft_idx"])
            got_same += 1
        elif pick:  # verifier cleared it but we already have enough: give the partner back
            res["used"][pick] -= 1
    got_rand = 0
    for it in plan["random"]:
        pick = res["random"].get(it["key"])
        if pick:
            d = dmap[it["draft_idx"]]
            refusals.append(make_refusal(cfg, by_id[pick], it["home"], it["question"], "random_other_source",
                                         d["split"], generator))
            converted.add(it["draft_idx"])
            got_rand += 1
    answerable = []
    for d in drafts:
        if d["idx"] in converted:
            continue
        meta = {"chunk_id": d["chunk_id"], "source_id": d["source_id"], "split": d["split"], "type": d["type"],
                "generator": generator, "verification": "unverified"}
        ex = make_example(cfg, by_id[d["chunk_id"]]["text"], d["question"], d["answer"], meta)
        ex["_draft_idx"] = d["idx"]
        answerable.append(ex)
    rows = answerable + refusals
    for i, ex in enumerate(rows):
        ex["id"] = f"{id_prefix}{i + 1:04d}"
        ex["dropped_by"] = None
    info = {"refusals_by_strategy": dict(Counter(r["meta"]["unanswerable_strategy"] for r in refusals)),
            "converted_draft_idx": sorted(converted),
            "same_topic_no_valid_partner": sum(1 for it in plan["same_topic"] if not res["same_topic"].get(it["key"])),
            "partner_use": dict(Counter(r["meta"]["partner_chunk_id"] for r in refusals))}
    return rows, info


def flag_leakage(rows: list[dict], reference: list[dict], cfg: dict, only_split: str | None = None) -> dict:
    """Mark rows dropped_by='leakage_filter' (rows already dropped stay as they are)."""
    cand = [r for r in rows if r["dropped_by"] is None and (only_split is None or r["meta"]["split"] == only_split)]
    hits = leakage_hits(cand, reference, cfg)
    for r in cand:
        h = hits.get(r["id"])
        if h:
            r["dropped_by"] = "leakage_filter"
            r["leakage"] = h
    return hits
