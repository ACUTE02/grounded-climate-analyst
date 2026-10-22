"""Chunk-level split in blocks of consecutive chunks, with buffer chunks dropped."""
import random


def make_splits(chunks: list[dict], cfg: dict) -> dict:
    """chunks: cleaned chunk dicts in file order. Returns {chunk_id: split or 'buffer'}."""
    s = cfg["split"]
    ratios = {"train": s["train"], "val": s["val"], "test": s["test"]}
    bs, buf = s["block_size"], s["buffer_chunks"]

    # blocks of consecutive chunks inside one source
    by_source: dict[str, list[dict]] = {}
    for ch in chunks:
        by_source.setdefault(ch["source_id"], []).append(ch)
    blocks = []
    for sid, lst in by_source.items():
        for i in range(0, len(lst), bs):
            blocks.append((sid, i // bs, lst[i:i + bs]))

    rng = random.Random(s["seed"])
    rng.shuffle(blocks)
    total = sum(len(b[2]) for b in blocks)
    counts = {k: 0 for k in ratios}
    assign: dict[tuple[str, int], str] = {}
    for sid, bi, lst in blocks:
        # give the block to the split furthest below its target share
        deficit = {k: ratios[k] * total - counts[k] for k in ratios}
        pick = max(deficit, key=deficit.get)
        assign[(sid, bi)] = pick
        counts[pick] += len(lst)

    result = {}
    for sid, bi, lst in blocks:
        for ch in lst:
            result[ch["id"]] = assign[(sid, bi)]

    # buffer: at each boundary between two blocks of different splits, drop chunk(s)
    for sid, lst in by_source.items():
        nblocks = (len(lst) + bs - 1) // bs
        for bi in range(1, nblocks):
            a, b = assign[(sid, bi - 1)], assign[(sid, bi)]
            if a == b:
                continue
            prev_block = lst[(bi - 1) * bs: bi * bs]
            next_block = lst[bi * bs: (bi + 1) * bs]
            # never sacrifice test/val if train can pay; otherwise drop from the later block
            if b == "train" or (a != "train" and b != "test"):
                for ch in next_block[:buf]:
                    result[ch["id"]] = "buffer"
            else:
                for ch in prev_block[-buf:]:
                    result[ch["id"]] = "buffer"
    return result
