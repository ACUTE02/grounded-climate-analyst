"""Stratified, blind random sample for the owner's spot check."""
import random

# columns the owner sees: no AI verdict, no AI reason
CSV_COLUMNS = ["id", "split", "source_id", "chunk_text", "question", "answer", "evidence_quote",
               "owner_verdict", "owner_note"]
KEY_COLUMNS = ["id", "stratum", "ai_verdict", "ai_reason"]


def stratum_of(row: dict) -> str:
    if row["ai_verdict"] in ("fix", "fail"):
        return "fix_or_fail"
    return "refusal" if row["is_refusal"] else "pass"


def sample_spot_check(rows: list[dict], n: int, seed: int, min_refusal: int, min_fixed_or_failed: int) -> list[dict]:
    """rows need keys: id, is_refusal, ai_verdict. Returns n rows (or all rows if fewer), sorted by id."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["id"])
    if len(rows) <= n:
        return rows
    refusals = [r for r in rows if r["is_refusal"]]
    bad = [r for r in rows if r["ai_verdict"] in ("fix", "fail")]
    chosen: dict[str, dict] = {}
    for r in rng.sample(refusals, min(min_refusal, len(refusals))):
        chosen[r["id"]] = r
    # refusals that are also fix/fail count toward the second quota
    have_bad = sum(1 for r in chosen.values() if r["ai_verdict"] in ("fix", "fail"))
    pool = [r for r in bad if r["id"] not in chosen]
    need = max(min_fixed_or_failed - have_bad, 0)
    for r in rng.sample(pool, min(need, len(pool))):
        chosen[r["id"]] = r
    rest = [r for r in rows if r["id"] not in chosen and r["ai_verdict"] == "pass" and not r["is_refusal"]]
    if len(rest) < n - len(chosen):  # not enough clean rows, fall back to anything left
        rest = [r for r in rows if r["id"] not in chosen]
    for r in rng.sample(rest, min(n - len(chosen), len(rest))):
        chosen[r["id"]] = r
    return sorted(chosen.values(), key=lambda r: r["id"])


def shuffled(rows: list[dict], seed: int) -> list[dict]:
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out
