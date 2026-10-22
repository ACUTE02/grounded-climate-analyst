"""Deterministic per-source cap on answerable items (train + val together)."""
import random
from collections import defaultdict


def apply_source_cap(items: list[dict], cap: int, val_share: float, seed: int) -> set[str]:
    """items: dicts with id, chunk_id, split ('train'|'val'), style, prior_pass (bool).

    Returns the ids to DROP. Selection per split: AI-passed items first, then as many different chunks
    as possible (max 2 per chunk), alternating question styles. Val quota follows the val share."""
    if len(items) <= cap:
        return set()
    rng = random.Random(seed)
    quota_val = min(round(cap * val_share), sum(1 for i in items if i["split"] == "val"))
    quota = {"val": quota_val, "train": cap - quota_val}
    keep: set[str] = set()
    for split in ("train", "val"):
        pool = sorted((i for i in items if i["split"] == split), key=lambda i: i["id"])
        by_chunk = defaultdict(list)
        for it in pool:
            by_chunk[it["chunk_id"]].append(it)
        for lst in by_chunk.values():
            rng.shuffle(lst)
            lst.sort(key=lambda i: not i["prior_pass"])  # stable: AI-passed first
        chunk_order = sorted(by_chunk)
        rng.shuffle(chunk_order)
        chosen: list[dict] = []
        for rnd in range(2):  # max 2 per chunk
            layer = [by_chunk[c][rnd] for c in chunk_order if len(by_chunk[c]) > rnd]
            layer.sort(key=lambda i: not i["prior_pass"])  # AI-passed first inside the round
            # alternate question styles: round-robin over styles
            styles = defaultdict(list)
            for it in layer:
                styles[it["style"]].append(it)
            order = []
            while any(styles.values()):
                for s in sorted(styles):
                    if styles[s]:
                        order.append(styles[s].pop(0))
            chosen.extend(order)
        # AI-passed first overall (stable), so a failed item is only kept when nothing better is left
        chosen.sort(key=lambda i: not i["prior_pass"])
        for it in chosen[: quota[split]]:
            keep.add(it["id"])
    # if a split had fewer items than its quota, top up from the other split
    short = cap - len(keep)
    if short > 0:
        rest = sorted((i for i in items if i["id"] not in keep), key=lambda i: (not i["prior_pass"], i["id"]))
        keep.update(i["id"] for i in rest[:short])
    return {i["id"] for i in items} - keep
