"""Tests for the A2 fixes (refusal partners, leakage, cap, quotes, blind spot-check, test refusals)."""
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from apply_verification import check_min_refusals, trim_test
from climate_slm.config import load_config, resolve
from climate_slm.data.boilerplate import boilerplate_reasons
from climate_slm.data.caps import apply_source_cap
from climate_slm.data.leakage import leakage_hits
from climate_slm.data.refusals import allocate, assign_partners, candidate_pool
from climate_slm.data.sft import make_example
from climate_slm.data.spotcheck import CSV_COLUMNS, KEY_COLUMNS, sample_spot_check, shuffled, stratum_of
from climate_slm.data.textutil import read_jsonl
from climate_slm.data.verify import coverage_ok, fold, quote_match
from conftest import needs

CFG = load_config()


def _chunks():
    out = []
    for i in range(6):
        out.append({"id": f"s{i % 3}::{i}", "source_id": f"s{i % 3}", "split": "train", "text": f"text {i}"})
    out.append({"id": "s9::t", "source_id": "s9", "split": "test", "text": "test text"})
    return out


def _ex(i, q, a, split="train", typ="numeric", chunk="Heat wave threshold is 40 degrees for plains areas today."):
    meta = {"chunk_id": f"c{i}", "source_id": "s", "split": split, "type": typ, "generator": "g",
            "verification": "unverified"}
    ex = make_example(CFG, chunk, q, a, meta)
    ex["id"] = f"x{i}"
    return ex


# ---- refusal partners -------------------------------------------------------
def test_partner_reuse_cap_and_boilerplate_never_partner():
    chunks = _chunks()[:6]
    boiler = {"s0::3": ["skip_log"]}
    can = {}
    items = []
    for k in range(6):
        home = "s1::1"
        pool = ["s0::3", "s0::0", "s2::2", "s2::5", "s1::4"]
        items.append({"key": f"q{k}", "home": home, "pool": pool})
        for c in pool:
            can[(f"q{k}", c)] = False  # verifier says none can answer
    res = assign_partners(items, [], chunks, boiler, can, max_reuse=2, rng=random.Random(1))
    picks = [v for v in res["same_topic"].values() if v]
    assert "s0::3" not in picks                      # boilerplate is never a partner
    assert max(Counter(picks).values()) <= 2         # reuse cap
    assert len(picks) == 6                           # pool is big enough to place every question


def test_partner_needs_verifier_clearance_and_same_split():
    chunks = _chunks()
    can = {("q", "s0::0"): True, ("q", "s2::2"): False, ("q", "s9::t"): False}
    it = [{"key": "q", "home": "s1::1", "pool": ["s0::0", "s9::t", "s2::2"]}]
    res = assign_partners(it, [], chunks, {}, can, 2, random.Random(1))
    assert res["same_topic"]["q"] == "s2::2"         # not the one that can answer, not the test-split chunk
    rnd = assign_partners([], [{"key": "r", "home": "s1::1"}], chunks, {}, {}, 2, random.Random(3))
    p = rnd["random"]["r"]
    assert p and p.split("::")[0] != "s1" and p != "s9::t"


def test_candidate_pool_same_split_and_no_boilerplate():
    chunks = _chunks()
    home = chunks[0]
    pool = candidate_pool("text 3 question", home, chunks, {"s0::3": ["x"]}, 5)
    assert "s9::t" not in pool and "s0::3" not in pool and home["id"] not in pool


def test_allocation_sums_and_follows_mix():
    a = allocate(29, CFG["unanswerable"]["mix"])
    assert sum(a.values()) == 29
    assert a["near_miss"] >= a["random_other_source"]


@needs("data/processed/chunks_clean.jsonl", "data/instruction/candidates.jsonl", "data/instruction/test_candidates.jsonl")
def test_real_refusals_partner_rules():
    chunks = {c["id"]: c for c in read_jsonl(resolve(CFG, "chunks_clean"))}
    from climate_slm.data.a2 import boiler_for
    boiler = boiler_for(list(chunks.values()), CFG)
    for f in ("candidates", "test_candidates"):
        use = Counter()
        for r in read_jsonl(resolve(CFG, f)):
            if r["meta"]["type"] != "unanswerable" or r["dropped_by"] is not None:
                continue
            m = r["meta"]
            assert chunks[m["partner_chunk_id"]]["split"] == chunks[m["question_chunk_id"]]["split"] == m["split"]
            assert not boiler[m["partner_chunk_id"]], (f, r["id"])
            assert m["needs_check"] is True and m["unanswerable_strategy"] in CFG["unanswerable"]["mix"]
            use[m["partner_chunk_id"]] += 1
        assert use and max(use.values()) <= CFG["unanswerable"]["max_partner_reuse"]


def test_boilerplate_heuristic_examples():
    assert "dotted_leaders" in boilerplate_reasons("Contents\n" + "Foreword ........ 5\n" * 6, CFG)
    menu = "Skip to main content\n" + "\n".join(["Home", "About", "News", "Contact", "Login", "Help", "Maps", "FAQ",
                                                  "Media", "Blog", "Jobs", "Terms", "Sitemap"]) + "\n"
    r = boilerplate_reasons(menu, CFG)
    assert "mostly_short_lines" in r or any(x.startswith("head_phrase") for x in r)
    prose = ("Heat wave is considered if the maximum temperature of a station reaches at least forty degrees for "
             "plains and at least thirty degrees for hilly regions. ") * 6
    assert boilerplate_reasons(prose, CFG) == []


# ---- leakage filter ---------------------------------------------------------
TRAIN_CTX = ("Heat wave is considered if maximum temperature of a station reaches at least 40 degrees for Plains "
             "and at least 30 degrees for Hilly regions during the season.")


def test_leakage_filter_rejects_dup_question_and_ngram_and_passes_clean():
    train = [_ex(1, "What temperature does a station need to reach for a heat wave in the plains?", "40 degrees.",
                 chunk=TRAIN_CTX)]
    near_dup = _ex(2, "What temperature does a station need to reach for a heat wave in the plains region?",
                   "At least 40 degrees.", split="test")
    ngram = _ex(3, "Which regions are covered by the two thresholds given for the season?",
                "Maximum temperature of a station reaches at least 40 degrees for Plains and at least 30 degrees",
                split="test")
    clean = _ex(4, "How often are cyclone bulletins issued by the regional centres?",
                "Cyclone bulletins are issued every three hours.", split="test",
                chunk="Cyclone bulletins are issued every three hours by regional centres.")
    hits = leakage_hits([near_dup, ngram, clean], train, CFG)
    assert hits["x2"]["reason"] == "question_jaccard" and hits["x2"]["matches"] == "x1"
    assert hits["x3"]["reason"] == "answer_8gram"
    assert "x4" not in hits


# ---- per-source cap ---------------------------------------------------------
def _cap_items(n=100):
    items = []
    for i in range(n):
        items.append({"id": f"d{i}", "chunk_id": f"c{i // 2}", "split": "val" if i % 5 == 0 else "train",
                      "style": ["number", "procedure", "definition"][i % 3], "prior_pass": i % 7 != 0})
    return items


def test_cap_is_deterministic_and_respects_cap_of_70():
    items = _cap_items(100)
    a = apply_source_cap(items, 70, 0.2, 42)
    b = apply_source_cap(items, 70, 0.2, 42)
    c = apply_source_cap(items, 70, 0.2, 7)
    assert a == b and a != c
    assert len(items) - len(a) == 70
    kept = [i for i in items if i["id"] not in a]
    assert max(Counter(i["chunk_id"] for i in kept).values()) <= 2
    assert sum(1 for i in kept if i["split"] == "val") in range(12, 17)  # about the val share
    assert sum(1 for i in kept if i["prior_pass"]) >= 60                 # AI-passed items first


@needs("data/instruction/candidates.jsonl")
def test_real_candidates_keep_source_cap_and_dropped_items():
    """Cap of 70 counted over train + val before A3; from A3 on val is exempt (cap_applies_to_val false)."""
    rows = read_jsonl(resolve(CFG, "candidates"))
    cap = CFG["generator"]["answerable_cap_per_source"]["ndma_drought_guidelines"]
    ndma = [r for r in rows if r["meta"]["source_id"] == "ndma_drought_guidelines" and r["meta"]["type"] != "unanswerable"]
    dropped_cap = [r for r in ndma if r["dropped_by"] == "source_cap"]
    assert dropped_cap and all(r["meta"]["split"] == "train" for r in dropped_cap)   # val is exempt now
    assert not CFG["a3"]["cap_applies_to_val"]
    train_kept = [r for r in ndma if r["meta"]["split"] == "train" and r["dropped_by"] in (None, "leakage_filter")]
    assert len(train_kept) <= cap
    assert all("dropped_by" in r for r in rows)
    assert {r["dropped_by"] for r in rows} <= {None, "source_cap", "leakage_filter", "reviewer_override",
                                                CFG["stage_b"]["owner_drop_label"]}


# ---- quotes -----------------------------------------------------------------
def test_quote_normalisation_curly_apostrophe_and_hyphen_artifacts():
    chunk = "under the control of State Disaster Management Authorities (SDMA’s). Next inter -agency work."
    assert fold("SDMA's") == fold("SDMA’s")
    assert quote_match("under the control of State Disaster Management Authorities (SDMA's).", chunk, CFG) == "exact"
    assert quote_match("Next inter-agency work.", chunk, CFG) == "exact"


def test_quote_fuzzy_and_fake():
    chunk = "Lack of check dams in the rainfed areas results in inadequate storage-water in times of vi. need or drought."
    q = "Lack of check dams in the rainfed areas results in inadequate storage-water in times of need or drought."
    assert quote_match(q, chunk, CFG) == "fuzzy"
    assert quote_match("Lack of check dams causes flooding in coastal cities every single monsoon season.", chunk, CFG) is None


def test_coverage_positive_and_negative():
    answer = "Heat wave is declared when the maximum temperature reaches 40 degrees for Plains."
    good = ["Heat wave is declared when the maximum temperature reaches 40 degrees for Plains."]
    ok, _ = coverage_ok(answer, good, CFG)
    assert ok
    bad_number = ["Heat wave is declared when the maximum temperature reaches 30 degrees for Hilly regions."]
    ok2, d2 = coverage_ok(answer, bad_number, CFG)
    assert not ok2 and "40" in d2["missing_numbers"]
    partial = ["Heat wave is declared"]
    ok3, d3 = coverage_ok("Heat wave is declared when the maximum temperature reaches a level for Plains regions.",
                          partial, CFG)
    assert not ok3 and d3["word_coverage"] < CFG["verification"]["evidence_min_coverage"]


# ---- blind spot-check -------------------------------------------------------
def _pool(n=120):
    rows = []
    for i in range(n):
        v = "fix" if i % 17 == 0 else ("fail" if i % 31 == 0 else "pass")
        rows.append({"id": f"r{i:03d}", "is_refusal": i % 6 == 0, "ai_verdict": v})
    return rows


def test_spot_check_strata_shuffle_and_blindness():
    s = CFG["spot_check"]
    a = sample_spot_check(_pool(), s["spot_check_train_n"], s["seed"], s["spot_min_refusal"],
                          s["spot_min_ai_fixed_or_failed"])
    assert len(a) == 30
    assert sum(r["is_refusal"] for r in a) >= 8
    assert sum(r["ai_verdict"] in ("fix", "fail") for r in a) >= 5
    b = shuffled(a, s["seed"])
    assert [r["id"] for r in b] != [r["id"] for r in a]
    assert [r["id"] for r in shuffled(a, s["seed"])] == [r["id"] for r in b]      # same seed, same order
    strata = [stratum_of(r) for r in b]
    assert strata[:10] != sorted(strata)[:10]                                    # not grouped
    t = sample_spot_check(_pool(), s["spot_check_test_n"], s["seed"], s["spot_min_refusal_test"], 0)
    assert len(t) == 12 and sum(r["is_refusal"] for r in t) >= 5
    assert "ai_verdict" not in CSV_COLUMNS and "ai_reason" not in CSV_COLUMNS
    assert set(KEY_COLUMNS) == {"id", "stratum", "ai_verdict", "ai_reason"}


def test_spot_check_files_are_blind_and_shuffled():
    ptr, pk = resolve(CFG, "spot_check_train"), resolve(CFG, "spot_check_key")
    if not pk.exists():
        pytest.skip("blind spot-check files not built yet")
    with open(ptr, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == CSV_COLUMNS
    ids = [r["id"] for r in rows]
    assert ids != sorted(ids)
    with open(pk, encoding="utf-8-sig", newline="") as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    tr_strata = Counter(key[i]["stratum"] for i in ids)
    assert tr_strata["fix_or_fail"] >= CFG["spot_check"]["spot_min_ai_fixed_or_failed"]
    refusals = sum(1 for r in rows if r["answer"] == CFG["refusal_text"])
    assert refusals >= CFG["spot_check"]["spot_min_refusal"]
    with open(resolve(CFG, "spot_check_test"), encoding="utf-8-sig", newline="") as f:
        te = list(csv.DictReader(f))
    assert sum(1 for r in te if r["answer"] == CFG["refusal_text"]) >= CFG["spot_check"]["spot_min_refusal_test"]


# ---- test set refusals ------------------------------------------------------
def test_min_refusals_check_fails_loudly():
    with pytest.raises(SystemExit):
        check_min_refusals(CFG["generator"]["test_min_refusals"] - 1, CFG)
    check_min_refusals(CFG["generator"]["test_min_refusals"], CFG)


def test_trim_test_keeps_all_valid_refusals_and_target():
    rows = []
    for i in range(60):
        meta = {"chunk_id": f"c{i // 2}", "source_id": "s", "split": "test", "type": "numeric",
                "verification": "llm_verified"}
        rows.append({"id": f"a{i:02d}", "meta": meta, "dropped_by": None})
    for i in range(14):
        meta = {"chunk_id": f"r{i}", "source_id": "s", "split": "test", "type": "unanswerable",
                "unanswerable_strategy": ["near_miss", "same_topic_missing", "random_other_source"][i % 3],
                "verification": "llm_verified"}
        rows.append({"id": f"n{i:02d}", "meta": meta, "dropped_by": None})
    out = trim_test(rows, CFG)
    active = [r for r in rows if r["dropped_by"] is None]
    assert len(active) == CFG["generator"]["test_target_items"]
    assert out["kept_refusals"] == min(14, CFG["generator"]["test_max_refusals"])
    assert {r["dropped_by"] for r in rows} <= {None, "over_target"}


@needs("data/instruction/test_candidates.jsonl")
def test_real_test_set_has_enough_valid_refusals_if_verified():
    rows = read_jsonl(resolve(CFG, "test_candidates"))
    if not any(r["meta"]["verification"] == "llm_verified" for r in rows):
        pytest.skip("test set not verified yet")
    final = [r for r in rows if r["dropped_by"] is None and r["meta"]["verification"] == "llm_verified"]
    refusals = [r for r in final if r["meta"]["type"] == "unanswerable"]
    assert len(refusals) >= CFG["generator"]["test_min_refusals"]
    assert len(final) <= CFG["generator"]["test_target_items"]
