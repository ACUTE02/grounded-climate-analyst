"""Leakage filter (question Jaccard, answer n-gram overlap) and cross-split overlap report."""
import re

from climate_slm.data.dedup import jaccard
from climate_slm.data.sft import parse_example
from climate_slm.data.textutil import words

_TOK = re.compile(r"[a-z0-9]+")


def norm_tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower().replace("’", "'"))


def ngrams(text: str, n: int) -> set[tuple]:
    t = norm_tokens(text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def leakage_hits(items: list[dict], reference: list[dict], cfg: dict) -> dict[str, dict]:
    """items / reference are SFT examples (with 'id').

    An item is rejected if its question token Jaccard with any reference question is >= max, or if
    its answer shares an n-word sequence with any context that appears in a reference prompt.
    Returns {item id: {"reason": ..., "matches": reference id, ...}}."""
    lk = cfg["leakage"]
    n, jmax = lk["answer_ngram_n"], lk["question_jaccard_max"]
    ref_q = []
    ref_ctx_ngrams: list[tuple[str, set]] = []
    seen_ctx: dict[str, str] = {}
    for r in reference:
        chunk, q, _ = parse_example(r)
        ref_q.append((r["id"], set(words(q, 2))))
        if chunk not in seen_ctx:
            seen_ctx[chunk] = r["id"]
            ref_ctx_ngrams.append((r["id"], ngrams(chunk, n)))
    out: dict[str, dict] = {}
    for it in items:
        _, q, a = parse_example(it)
        qt = set(words(q, 2))
        best = max(((jaccard(qt, t), rid) for rid, t in ref_q), default=(0.0, None))
        if best[0] >= jmax:
            out[it["id"]] = {"reason": "question_jaccard", "matches": best[1], "score": round(best[0], 3)}
            continue
        if it["meta"].get("type") == "unanswerable":
            continue
        ag = ngrams(a, n)
        for rid, cg in ref_ctx_ngrams:
            common = ag & cg
            if common:
                out[it["id"]] = {"reason": f"answer_{n}gram", "matches": rid,
                                 "example": " ".join(sorted(common)[0])}
                break
    return out


def cross_split_overlap(chunks: list[dict], n: int, top: int) -> list[dict]:
    """Chunk pairs in different splits or different sources, ranked by shared n-grams (buffer excluded)."""
    use = [c for c in chunks if c["split"] != "buffer"]
    grams = [ngrams(c["text"], n) for c in use]
    pairs = []
    for i in range(len(use)):
        for j in range(i + 1, len(use)):
            a, b = use[i], use[j]
            if a["split"] == b["split"] and a["source_id"] == b["source_id"]:
                continue
            common = grams[i] & grams[j]
            if common:
                pairs.append({"shared_ngrams": len(common),
                              "share_of_smaller": round(len(common) / max(1, min(len(grams[i]), len(grams[j]))), 3),
                              "chunk_a": a["id"], "split_a": a["split"], "source_a": a["source_id"],
                              "chunk_b": b["id"], "split_b": b["split"], "source_b": b["source_id"],
                              "example": " ".join(sorted(common)[0])})
    pairs.sort(key=lambda p: (-p["shared_ngrams"], p["chunk_a"], p["chunk_b"]))
    return pairs[:top]
