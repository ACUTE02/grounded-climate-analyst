"""Refusal (unanswerable) examples: strategy allocation, partner pools and partner assignment."""
import random
from collections import Counter

from climate_slm.data.lexical import BM25


def allocate(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Split `total` over the strategies by largest remainder (deterministic)."""
    s = sum(mix.values())
    raw = {k: total * v / s for k, v in mix.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def candidate_pool(question: str, home: dict, chunks: list[dict], boiler: dict[str, list[str]], k: int) -> list[str]:
    """Top-k most lexically similar usable chunks of the SAME split (home chunk excluded)."""
    pool = [c for c in chunks if c["split"] == home["split"] and c["id"] != home["id"] and not boiler.get(c["id"])]
    if not pool:
        return []
    bm = BM25([c["text"] for c in pool])
    order = sorted(range(len(pool)), key=lambda i: (-bm.score(question, i), pool[i]["id"]))
    return [pool[i]["id"] for i in order[:k]]


def assign_partners(same_topic: list[dict], random_items: list[dict], chunks: list[dict],
                    boiler: dict[str, list[str]], can_answer: dict[tuple[str, str], bool],
                    max_reuse: int, rng: random.Random, used: Counter | None = None) -> dict:
    """Choose partner chunks.

    same_topic items: {"key", "home", "pool": [chunk ids by similarity]}; the partner is the most similar
    chunk that the verifier says cannot answer the question and that is used < max_reuse times.
    random items: {"key", "home"}; partner is a random usable chunk of another source, same split.
    Returns {"same_topic": {key: chunk_id or None}, "random": {key: chunk_id or None}, "used": Counter}."""
    used = Counter(used or {})
    by_id = {c["id"]: c for c in chunks}
    res = {"same_topic": {}, "random": {}}
    for it in same_topic:
        pick = None
        for cid in it["pool"]:
            if boiler.get(cid) or used[cid] >= max_reuse:
                continue
            if by_id[cid]["split"] != by_id[it["home"]]["split"]:
                continue
            if can_answer.get((it["key"], cid)) is not False:  # only chunks the verifier cleared
                continue
            pick = cid
            break
        res["same_topic"][it["key"]] = pick
        if pick:
            used[pick] += 1
    for it in random_items:
        home = by_id[it["home"]]
        opts = [c["id"] for c in chunks if c["split"] == home["split"] and c["source_id"] != home["source_id"]
                and not boiler.get(c["id"]) and used[c["id"]] < max_reuse]
        pick = rng.choice(sorted(opts)) if opts else None
        res["random"][it["key"]] = pick
        if pick:
            used[pick] += 1
    res["used"] = used
    return res
