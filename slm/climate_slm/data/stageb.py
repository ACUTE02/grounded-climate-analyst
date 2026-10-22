"""Stage B: owner spot-check disagreement, stop rule, final SFT files, leakage re-check, length stats."""
import copy
import math
from collections import Counter, defaultdict
from fractions import Fraction

from climate_slm.data.build import run_checks
from climate_slm.data.leakage import leakage_hits
from climate_slm.data.sft import parse_example

STRATA = ("refusal", "fix_or_fail", "pass")


def ai_ok(ai_verdict: str, cfg: dict) -> bool:
    return ai_verdict in cfg["stage_b"]["ai_ok_verdicts"]


def disagrees(owner_verdict: str, ai_verdict: str, cfg: dict) -> bool:
    """(owner bad and AI OK) or (owner good and AI not OK)."""
    b = cfg["stage_b"]
    if owner_verdict not in (b["owner_good"], b["owner_bad"]):
        raise ValueError(f"owner verdict must be good or bad, got {owner_verdict!r}")
    return (owner_verdict == b["owner_bad"]) == ai_ok(ai_verdict, cfg)


def is_reviewer_decided(note: str, cfg: dict) -> bool:
    return (note or "").startswith(cfg["stage_b"]["reviewer_note_prefix"])


def _rate(rows: list[dict], cfg: dict) -> dict:
    d = sum(disagrees(r["owner_verdict"], r["ai_verdict"], cfg) for r in rows)
    return {"rows": len(rows), "disagree": d, "rate": round(d / len(rows), 4) if rows else None}


def disagreement_table(rows: list[dict], cfg: dict) -> dict:
    """rows need: id, owner_verdict, owner_note, ai_verdict, stratum."""
    human = [r for r in rows if not is_reviewer_decided(r["owner_note"], cfg)]
    reviewer = [r for r in rows if is_reviewer_decided(r["owner_note"], cfg)]
    return {"all": _rate(rows, cfg), "human_only": _rate(human, cfg), "reviewer_decided": _rate(reviewer, cfg),
            "per_stratum": {s: _rate([r for r in rows if r["stratum"] == s], cfg) for s in STRATA},
            "disagreeing_ids": [r["id"] for r in rows if disagrees(r["owner_verdict"], r["ai_verdict"], cfg)]}


def above(rate: dict, threshold: float) -> bool:
    """True if disagree / rows is above the threshold (exact arithmetic; no rows = not above)."""
    if not rate["rows"]:
        return False
    return Fraction(rate["disagree"], rate["rows"]) > Fraction(str(threshold))


def stop_decision(tables: dict[str, dict], cfg: dict) -> dict:
    """tables: {'train': disagreement_table, 'test': disagreement_table}."""
    thr = cfg["spot_check"]["spot_check_max_disagreement"]
    reasons = []
    for name, t in tables.items():
        a, h = above(t["all"], thr), above(t["human_only"], thr)
        if a:
            reasons.append(f"{name}: all-rows rate {t['all']['disagree']}/{t['all']['rows']} is above {thr}")
        if a != h:
            reasons.append(f"{name}: all-rows rate {t['all']['disagree']}/{t['all']['rows']} and human-only rate "
                           f"{t['human_only']['disagree']}/{t['human_only']['rows']} are on different sides of {thr}")
    return {"stop": bool(reasons), "threshold": thr, "reasons": reasons}


def spot_label(spot_row: dict | None, cfg: dict) -> str:
    if spot_row is None:
        return "none"
    if is_reviewer_decided(spot_row["owner_note"], cfg):
        return "reviewer"
    return "human" if spot_row["owner_verdict"] == cfg["stage_b"]["owner_good"] else "none"


def mark_owner_drops(rows: list[dict], spot: dict[str, dict], cfg: dict) -> list[str]:
    """Flag owner-'bad' rows in place. Rows already dropped keep their dropped_by. Returns the flagged ids."""
    b = cfg["stage_b"]
    out = []
    for r in rows:
        s = spot.get(r["id"])
        if s is None:
            continue
        r["owner_verdict"] = s["owner_verdict"]
        if s["owner_verdict"] == b["owner_bad"]:
            out.append(r["id"])
            if r["dropped_by"] is None:
                r["dropped_by"] = b["owner_drop_label"]
                r["dropped_reason"] = "owner spot-check verdict: bad"
    return out


def is_final(r: dict, spot: dict[str, dict], cfg: dict) -> bool:
    if r["dropped_by"] is not None or r["meta"]["verification"] != "llm_verified":
        return False
    if not (r.get("ai_verdict") == "pass" or (r.get("ai_verdict") == "fix" and r.get("ai_fix"))):
        return False
    s = spot.get(r["id"])
    return not (s and s["owner_verdict"] == cfg["stage_b"]["owner_bad"])


def final_items(rows: list[dict], spot: dict[str, dict], cfg: dict) -> list[dict]:
    """Final SFT examples (prompt, completion, meta). The candidate rows are not changed."""
    out = []
    for r in rows:
        if not is_final(r, spot, cfg):
            continue
        meta = dict(copy.deepcopy(r["meta"]), id=r["id"], verification="llm_verified",
                    spot_check=spot_label(spot.get(r["id"]), cfg),
                    blind_flag=[f for f in (r.get("blind_flag") or "").split(";") if f])
        out.append({"prompt": copy.deepcopy(r["prompt"]), "completion": copy.deepcopy(r["completion"]), "meta": meta})
    return out


def _with_id(items: list[dict]) -> list[dict]:
    return [dict(i, id=i["meta"]["id"]) for i in items]


def recheck(items: list[dict], cfg: dict) -> dict[str, list[str]]:
    """Automatic checks and dedup on one final file. Returns {id: flags} for flagged items only."""
    rows = copy.deepcopy(_with_id(items))
    run_checks(rows, cfg)
    return {r["id"]: r["auto_flags"] for r in rows if r["auto_flags"]}


def _chunk_ids(items: list[dict]) -> set[str]:
    out = set()
    for i in items:
        out.add(i["meta"]["chunk_id"])
        if i["meta"].get("question_chunk_id"):
            out.add(i["meta"]["question_chunk_id"])
    return out


def final_leakage(train: list[dict], val: list[dict], test: list[dict], cfg: dict) -> dict:
    """Leakage between the final files. Every value must be empty."""
    ct, cv, cs = _chunk_ids(train), _chunk_ids(val), _chunk_ids(test)
    return {"chunk_in_two_splits": sorted((ct & cv) | (ct & cs) | (cv & cs)),
            "test_vs_train_val": leakage_hits(_with_id(test), _with_id(train + val), cfg),
            "val_vs_train": leakage_hits(_with_id(val), _with_id(train), cfg)}


def is_refusal(item: dict) -> bool:
    return item["meta"]["type"] == "unanswerable"


def counts(items: list[dict]) -> dict:
    ref = sum(is_refusal(i) for i in items)
    return {"items": len(items), "answerable": len(items) - ref, "refusal": ref,
            "per_source": dict(sorted(Counter(i["meta"]["source_id"] for i in items).items())),
            "spot_check": dict(sorted(Counter(i["meta"]["spot_check"] for i in items).items())),
            "blind_flagged": sum(bool(i["meta"]["blind_flag"]) for i in items)}


def val_minimums(val: list[dict], cfg: dict) -> dict:
    a3 = cfg["a3"]
    ref = sum(is_refusal(i) for i in val)
    return {"val_items": len(val), "val_refusals": ref,
            "ok": len(val) >= a3["val_min_items"] and ref >= a3["val_min_refusals"]}


def _stats(values: list[int], pct: int) -> dict:
    if not values:
        return {}
    v = sorted(values)
    mid = len(v) // 2
    median = v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2
    return {"min": v[0], "median": median, f"p{pct}": v[max(math.ceil(pct / 100 * len(v)) - 1, 0)], "max": v[-1]}


def length_stats(items: list[dict], cfg: dict) -> dict:
    """Character lengths. Prompt = system + user message text. Percentile by nearest rank."""
    pct = cfg["stage_b"]["length_percentile"]
    return {"prompt_chars": _stats([sum(len(m["content"]) for m in i["prompt"]) for i in items], pct),
            "completion_chars": _stats([sum(len(m["content"]) for m in i["completion"]) for i in items], pct)}


def refusal_siblings(rows: list[dict], dropped_ids: list[str]) -> list[dict]:
    """For every dropped refusal: strategy, question, partner chunk, other refusals with the same partner chunk."""
    by_ctx = defaultdict(list)
    for r in rows:
        if r["meta"]["type"] == "unanswerable":
            by_ctx[r["meta"]["chunk_id"]].append(r)
    out = []
    for r in rows:
        if r["id"] in dropped_ids and r["meta"]["type"] == "unanswerable":
            sib = [{"id": s["id"], "split": s["meta"]["split"], "strategy": s["meta"].get("unanswerable_strategy"),
                    "question": parse_example(s)[1], "dropped_by": s["dropped_by"], "ai_verdict": s.get("ai_verdict")}
                   for s in by_ctx[r["meta"]["chunk_id"]] if s["id"] != r["id"]]
            out.append({"id": r["id"], "strategy": r["meta"].get("unanswerable_strategy"),
                        "question": parse_example(r)[1], "partner_chunk_id": r["meta"]["chunk_id"],
                        "dropped_by": r["dropped_by"], "other_refusals_same_partner": sib})
    return out
