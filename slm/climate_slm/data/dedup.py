"""Exact and near-duplicate detection on questions."""
from climate_slm.data.sft import parse_example
from climate_slm.data.textutil import norm_ws, words


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def find_duplicates(examples: list[dict], threshold: float) -> dict[int, str]:
    """Map example index -> flag for the LATER example of each duplicate pair.

    Compared inside the same split and the same label (answerable / unanswerable)."""
    seen: list[tuple[str, set, tuple]] = []
    out: dict[int, str] = {}
    for i, ex in enumerate(examples):
        _, q, _ = parse_example(ex)
        qn = norm_ws(q).lower()
        toks = set(words(q, 2))
        label = (ex["meta"]["split"], ex["meta"].get("type") == "unanswerable")
        for qn2, toks2, label2 in seen:
            if label != label2:
                continue
            if qn == qn2:
                out[i] = "duplicate_exact"
                break
            if jaccard(toks, toks2) >= threshold:
                out[i] = "near_duplicate"
                break
        seen.append((qn, toks, label))
    return out
