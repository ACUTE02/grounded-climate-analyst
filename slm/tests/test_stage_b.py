"""Tests for Stage B (spot-check disagreement, stop rule, final files)."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage_b
from climate_slm.config import REPO_ROOT, load_config, resolve
from climate_slm.data import stageb
from climate_slm.data.sft import make_example
from climate_slm.data.textutil import read_jsonl
from conftest import needs

CFG = load_config()
B = CFG["stage_b"]
REF = CFG["refusal_text"]
NOTE_R = B["reviewer_note_prefix"] + " (owner unsure). reason"


def _row(i, owner, ai, stratum="pass", note="ok"):
    return {"id": f"r{i}", "owner_verdict": owner, "owner_note": note, "ai_verdict": ai, "stratum": stratum}


def _cand(i, split="train", typ="numeric", verdict="pass", dropped=None, chunk=None, q=None, a=None, **extra):
    chunk = chunk or f"Station number {i} reports a heat wave when the maximum temperature reaches 4{i} degrees."
    q = q or f"What maximum temperature does station number {i} need?"
    a = REF if typ == "unanswerable" else (a or f"The maximum temperature reaches 4{i} degrees.")
    meta = {"chunk_id": f"c{i}", "source_id": "s", "split": split, "type": typ, "generator": "g",
            "verification": "llm_verified" if verdict in ("pass", "fix") else "unverified"}
    ex = make_example(CFG, chunk, q, a, meta)
    ex.update(id=f"x{i}", dropped_by=dropped, ai_verdict=verdict, blind_flag="", **extra)
    return ex


def _spot(owner, note="it can answer"):
    return {"owner_verdict": owner, "owner_note": note}


# ---- disagreement definition ------------------------------------------------
def test_config_loads_stage_b_keys():
    assert CFG["spot_check"]["spot_check_max_disagreement"] == 0.20
    assert B["ai_ok_verdicts"] == ["pass", "fix"] and (B["owner_good"], B["owner_bad"]) == ("good", "bad")
    assert isinstance(B["allow_finalize_after_stop"], bool)
    if B["allow_finalize_after_stop"]:                       # an override must come with its reason
        assert B["override_reason"].strip() and B["override_source"] and B["reviewer_adjudication"]
    for k in ("sft_train", "sft_val", "sft_test", "summary"):
        assert B[k].startswith("data/")


def test_disagreement_definition_small_table():
    assert not stageb.disagrees("good", "pass", CFG) and not stageb.disagrees("good", "fix", CFG)
    assert stageb.disagrees("good", "fail", CFG)
    assert stageb.disagrees("bad", "pass", CFG) and stageb.disagrees("bad", "fix", CFG)
    assert not stageb.disagrees("bad", "fail", CFG)
    with pytest.raises(ValueError):
        stageb.disagrees("average", "pass", CFG)
    rows = [_row(1, "good", "pass"), _row(2, "bad", "pass", "refusal"), _row(3, "good", "fail", "fix_or_fail"),
            _row(4, "bad", "fail", "fix_or_fail", NOTE_R), _row(5, "good", "fix", "fix_or_fail", NOTE_R),
            _row(6, "bad", "fix", "refusal", NOTE_R)]
    t = stageb.disagreement_table(rows, CFG)
    assert t["all"] == {"rows": 6, "disagree": 3, "rate": 0.5}
    assert t["human_only"] == {"rows": 3, "disagree": 2, "rate": 0.6667}
    assert t["reviewer_decided"] == {"rows": 3, "disagree": 1, "rate": 0.3333}
    assert t["per_stratum"]["refusal"] == {"rows": 2, "disagree": 2, "rate": 1.0}
    assert t["per_stratum"]["fix_or_fail"] == {"rows": 3, "disagree": 1, "rate": 0.3333}
    assert t["per_stratum"]["pass"] == {"rows": 1, "disagree": 0, "rate": 0.0}
    assert t["disagreeing_ids"] == ["r2", "r3", "r6"]


# ---- stop rule ---------------------------------------------------------------
def _table(n, d, n_human, d_human):
    rows = [_row(i, "bad" if i < d_human else "good", "pass") for i in range(n_human)]
    rows += [_row(100 + i, "bad" if i < d - d_human else "good", "pass", note=NOTE_R) for i in range(n - n_human)]
    return stageb.disagreement_table(rows, CFG)


def test_stop_rule_behaviour():
    ok = _table(30, 3, 20, 2)
    assert not stageb.stop_decision({"train": ok, "test": ok}, CFG)["stop"]
    at_threshold = _table(30, 6, 30, 6)                      # exactly 0.20 is not above
    assert not stageb.stop_decision({"train": at_threshold, "test": ok}, CFG)["stop"]
    over = _table(30, 7, 30, 7)
    assert stageb.stop_decision({"train": over, "test": ok}, CFG)["stop"]
    assert stageb.stop_decision({"train": ok, "test": _table(12, 3, 12, 3)}, CFG)["stop"]
    mixed = _table(30, 6, 23, 6)                             # all rows 0.20, human-only 0.26
    d = stageb.stop_decision({"train": mixed, "test": ok}, CFG)
    assert d["stop"] and "different sides" in d["reasons"][0]
    assert not stageb.above({"rows": 0, "disagree": 0, "rate": None}, 0.2)


@needs("data/instruction/spot_check_train.csv", "data/instruction/spot_check_test.csv", "data/instruction/spot_check_key.csv")
def test_stop_rule_on_real_files_blocks_final_files():
    tables = {n: stageb.disagreement_table(rows, CFG) for n, rows in stage_b.load_spot(CFG).items()}
    assert tables["train"]["all"]["rows"] == CFG["spot_check"]["spot_check_train_n"]
    assert tables["test"]["all"]["rows"] == CFG["spot_check"]["spot_check_test_n"]
    stop = stageb.stop_decision(tables, CFG)["stop"]
    exist = [(REPO_ROOT / B[f"sft_{s}"]).exists() for s in ("train", "val", "test")]
    if stop and not B["allow_finalize_after_stop"]:
        assert not any(exist), "the stop rule says stop, final files must not exist"
        with pytest.raises(SystemExit):
            stage_b.finalize(CFG, dry_run=False)
    else:
        assert all(exist)


# ---- final sets (made-up data) -----------------------------------------------
def test_final_items_drop_rules_and_labels():
    rows = [_cand(1), _cand(2), _cand(3), _cand(4, verdict="fail"), _cand(5, dropped="reviewer_override"),
            _cand(6, dropped="source_cap"), _cand(7, verdict="fix", ai_fix={"fix_answer": "x"}),
            _cand(8, verdict="fix"), _cand(9, typ="unanswerable")]
    rows[0]["blind_flag"] = "low_overlap;number_not_in_context"
    spot = {"x1": _spot("good"), "x2": _spot("bad"), "x3": _spot("good", NOTE_R), "x5": _spot("bad")}
    before = copy.deepcopy(rows)
    final = stageb.final_items(rows, spot, CFG)
    assert rows == before                                              # candidates untouched
    by = {f["meta"]["id"]: f for f in final}
    assert set(by) == {"x1", "x3", "x7", "x9"}                         # owner-bad, fail, dropped, fix-without-fix are out
    assert all(set(f) == {"prompt", "completion", "meta"} for f in final)
    assert all(f["meta"]["verification"] == "llm_verified" for f in final)
    assert by["x1"]["meta"]["spot_check"] == "human" and by["x3"]["meta"]["spot_check"] == "reviewer"
    assert by["x7"]["meta"]["spot_check"] == "none"
    assert by["x1"]["meta"]["blind_flag"] == ["low_overlap", "number_not_in_context"]
    assert by["x7"]["meta"]["blind_flag"] == []
    assert by["x9"]["completion"][0]["content"] == REF
    assert "reviewed" not in str(final)
    assert stageb.spot_label(_spot("bad"), CFG) == "none" and stageb.spot_label(None, CFG) == "none"

    flagged = stageb.mark_owner_drops(rows, spot, CFG)
    assert flagged == ["x2", "x5"]
    assert rows[1]["dropped_by"] == B["owner_drop_label"]              # owner-bad row stays, with the flag
    assert rows[4]["dropped_by"] == "reviewer_override"                # an earlier drop reason is kept
    assert {f["meta"]["id"] for f in stageb.final_items(rows, spot, CFG)} == set(by)


def test_final_leakage_detects_each_kind():
    tr = stageb.final_items([_cand(1), _cand(2, q="Which crop is sown in June?", a="Maize.", chunk="Maize in June.")],
                            {}, CFG)
    va = stageb.final_items([_cand(3, "val", q="Who issues the bulletin?", a="The agency.", chunk="The agency.")], {}, CFG)
    te = stageb.final_items([_cand(4, "test", q="Where is the rain gauge kept?", a="On open ground.",
                                   chunk="On open ground.")], {}, CFG)
    assert not any(stageb.final_leakage(tr, va, te, CFG).values())
    same_chunk = copy.deepcopy(te)
    same_chunk[0]["meta"]["chunk_id"] = "c1"
    assert stageb.final_leakage(tr, va, same_chunk, CFG)["chunk_in_two_splits"] == ["c1"]
    dup_q = stageb.final_items([_cand(5, "test", q="What maximum temperature does station number 1 need?")], {}, CFG)
    assert stageb.final_leakage(tr, va, dup_q, CFG)["test_vs_train_val"]["x5"]["reason"] == "question_jaccard"
    long_ctx = "Farmers in the western zone should sow the pearl millet crop after the first good rain of July."
    tr2 = tr + stageb.final_items([_cand(6, chunk=long_ctx)], {}, CFG)
    ngram = stageb.final_items([_cand(7, "val", chunk=long_ctx + " More text here.", q="When is pearl millet sown?",
                                      a="Sow the pearl millet crop after the first good rain of July.")], {}, CFG)
    assert stageb.final_leakage(tr2, ngram, te, CFG)["val_vs_train"]["x7"]["reason"].startswith("answer_")


def test_length_stats_and_val_minimums():
    items = stageb.final_items([_cand(i, "val", typ="unanswerable" if i < 8 else "numeric") for i in range(50)], {}, CFG)
    assert stageb.val_minimums(items, CFG)["ok"]
    assert not stageb.val_minimums(items[:49], CFG)["ok"]            # too few items
    assert not stageb.val_minimums(items[1:] + items[10:11], CFG)["ok"]  # too few refusals
    s = stageb._stats(list(range(1, 101)), 95)
    assert s == {"min": 1, "median": 50.5, "p95": 95, "max": 100}
    ls = stageb.length_stats(items[:1], CFG)
    assert ls["completion_chars"]["min"] == ls["completion_chars"]["max"] == len(REF)


# ---- final sets (real data, built in memory; also the files when they exist) ----
def _real_final():
    spot = {r["id"]: r for rows in stage_b.load_spot(CFG).values() for r in rows}
    cand, test = read_jsonl(resolve(CFG, "candidates")), read_jsonl(resolve(CFG, "test_candidates"))
    final = {"train": stageb.final_items([r for r in cand if r["meta"]["split"] == "train"], spot, CFG),
             "val": stageb.final_items([r for r in cand if r["meta"]["split"] == "val"], spot, CFG),
             "test": stageb.final_items(test, spot, CFG)}
    return spot, final


@needs("data/instruction/spot_check_train.csv", "data/instruction/spot_check_test.csv", "data/instruction/spot_check_key.csv", "data/instruction/candidates.jsonl", "data/instruction/test_candidates.jsonl")
def test_real_final_sets_respect_owner_and_reviewer_drops():
    spot, final = _real_final()
    ids = {i["meta"]["id"] for items in final.values() for i in items}
    bad = {i for i, r in spot.items() if r["owner_verdict"] == B["owner_bad"]}
    assert bad and not (ids & bad)                                     # owner-bad ids come from the CSVs
    assert not (ids & set(CFG["a3"]["reviewer_drop_ids"]))
    for s, items in final.items():
        assert all(i["meta"]["split"] == s for i in items)
        for i in items:
            m = i["meta"]
            assert m["verification"] == "llm_verified" and isinstance(m["blind_flag"], list)
            assert m["spot_check"] == stageb.spot_label(spot.get(m["id"]), CFG)
            if m["type"] == "unanswerable":
                assert i["completion"][0]["content"] == REF
            else:
                assert i["completion"][0]["content"] != REF
    labels = [i["meta"]["spot_check"] for items in final.values() for i in items]
    good = [r for r in spot.values() if r["owner_verdict"] == B["owner_good"] and r["id"] in ids]
    assert labels.count("human") + labels.count("reviewer") == len(good)


@needs("data/instruction/spot_check_train.csv", "data/instruction/spot_check_test.csv", "data/instruction/spot_check_key.csv", "data/instruction/candidates.jsonl", "data/instruction/test_candidates.jsonl")
def test_real_final_sets_leakage_checks_and_val_minimums():
    _, final = _real_final()
    assert not any(stageb.final_leakage(final["train"], final["val"], final["test"], CFG).values())
    assert not any(stageb.recheck(items, CFG) for items in final.values())
    vm = stageb.val_minimums(final["val"], CFG)
    assert vm["ok"], f"validation minimums not met: {vm}"


def _files():
    paths = {s: REPO_ROOT / B[f"sft_{s}"] for s in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("final files not written (stop rule)")
    return {s: read_jsonl(p) for s, p in paths.items()}


@needs("data/instruction/spot_check_train.csv", "data/instruction/spot_check_test.csv", "data/instruction/spot_check_key.csv", "data/instruction/candidates.jsonl", "data/instruction/test_candidates.jsonl")
def test_final_files_match_in_memory_build_if_present():
    files = _files()
    _, final = _real_final()
    for s in files:
        assert files[s] == final[s]


@needs("data/instruction/spot_check_train.csv", "data/instruction/spot_check_test.csv", "data/instruction/spot_check_key.csv", "data/instruction/candidates.jsonl", "data/instruction/test_candidates.jsonl")
def test_written_final_files_leakage_drops_and_labels():
    files = _files()
    assert not any(stageb.final_leakage(files["train"], files["val"], files["test"], CFG).values())
    assert not any(stageb.recheck(items, CFG) for items in files.values())
    assert stageb.val_minimums(files["val"], CFG)["ok"]
    spot = {r["id"]: r for rows in stage_b.load_spot(CFG).values() for r in rows}
    bad = {i for i, r in spot.items() if r["owner_verdict"] == B["owner_bad"]}
    ids = [i["meta"]["id"] for items in files.values() for i in items]
    assert len(ids) == len(set(ids))
    assert not (set(ids) & bad) and not (set(ids) & set(CFG["a3"]["reviewer_drop_ids"]))
    for s, items in files.items():
        for i in items:
            assert set(i) == {"prompt", "completion", "meta"} and i["meta"]["split"] == s
            assert i["meta"]["verification"] == "llm_verified"
            assert i["meta"]["spot_check"] == stageb.spot_label(spot.get(i["meta"]["id"]), CFG)
            assert (i["completion"][0]["content"] == REF) == (i["meta"]["type"] == "unanswerable")
    cand = {r["id"]: r for k in ("candidates", "test_candidates") for r in read_jsonl(resolve(CFG, k))}
    for i in bad:                                                      # kept in the candidate files, with a flag
        assert cand[i]["dropped_by"] in (B["owner_drop_label"], "reviewer_override") or cand[i]["ai_verdict"] == "fail"
        assert cand[i]["dropped_by"] is not None
