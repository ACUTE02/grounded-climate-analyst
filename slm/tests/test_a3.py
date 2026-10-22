"""Tests for the A3 fixes."""
import hashlib
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from apply_verification import check_val_minimums
from climate_slm.config import REPO_ROOT, load_config, resolve
from climate_slm.data.blind import score_item, token_f1
from climate_slm.data.sft import parse_example
from climate_slm.data.textutil import read_jsonl
from conftest import needs

CFG = load_config()
REF = CFG["refusal_text"]
CTX = "Heat wave is considered if maximum temperature reaches at least 40 degrees for Plains and 30 for Hilly regions."


def _final(rows):
    return [r for r in rows if r["dropped_by"] is None and r["meta"]["verification"] == "llm_verified"]


@needs("data/instruction/candidates.jsonl")
def test_reviewer_dropped_ids_are_out_of_final_pool_but_kept():
    rows = read_jsonl(resolve(CFG, "candidates"))
    by_id = {r["id"]: r for r in rows}
    ids = CFG["a3"]["reviewer_drop_ids"]
    assert set(ids) == {"tr2-0216", "tr2-0214"}
    for i in ids:
        assert i in by_id                                       # not deleted
        assert by_id[i]["dropped_by"] == "reviewer_override"
        assert by_id[i]["dropped_reason"]
    assert not ({r["id"] for r in _final(rows)} & set(ids))


def test_blind_scoring_refusal_item():
    assert score_item(True, REF, REF, CTX, CFG) == (True, [])
    assert score_item(True, REF, "  i CANNOT answer this   from the provided context. ", CTX, CFG) == (True, [])
    ok, f = score_item(True, REF, "Plains need 40 degrees.", CTX, CFG)
    assert not ok and f == ["answered_a_refusal_item"]


def test_blind_scoring_answerable_item_flags():
    ref = "Heat wave is considered if maximum temperature reaches at least 40 degrees for Plains."
    ok, f = score_item(False, ref, "Maximum temperature must reach at least 40 degrees for Plains.", CTX, CFG)
    assert ok and f == []
    ok, f = score_item(False, ref, REF, CTX, CFG)
    assert not ok and f == ["refused_an_answerable_item"]
    ok, f = score_item(False, ref, "It depends on the season.", CTX, CFG)
    assert not ok and f == ["low_overlap"]
    ok, f = score_item(False, ref, "Maximum temperature must reach at least 45 degrees for Plains.", CTX, CFG)
    assert not ok and f == ["number_not_in_context"]


def test_token_f1_threshold_is_config_value():
    a, b = "one two three four", "one two five six"
    assert abs(token_f1(a, b) - 0.5) < 1e-9
    assert CFG["a3"]["blind_f1_min"] == 0.5
    assert score_item(False, b, a, "one two three four five six", CFG)[0]  # exactly at the threshold is ok


def test_val_minimums_check_logic():
    def r(i, ref, ver="llm_verified"):
        return {"id": f"v{i}", "dropped_by": None, "meta": {"split": "val", "verification": ver,
                                                           "type": "unanswerable" if ref else "x"}}
    good = [r(i, i < 8) for i in range(50)]
    assert check_val_minimums(good, CFG)["ok"]
    assert not check_val_minimums(good[:49], CFG)["ok"]
    few_refusals = [r(i, i < 7) for i in range(50)]
    assert not check_val_minimums(few_refusals, CFG)["ok"]
    unverified = [r(i, i < 8, "unverified") for i in range(60)]
    assert check_val_minimums(unverified, CFG)["val_items"] == 0


@needs("data/instruction/candidates.jsonl")
def test_val_minimums_real_data():
    rows = read_jsonl(resolve(CFG, "candidates"))
    vm = check_val_minimums(rows, CFG)
    assert vm["ok"], f"validation minimums not met: {vm}"
    assert all(r["meta"]["split"] == "val" for r in rows if r["id"].startswith("v3-"))


@needs("data/instruction/spot_check_hashes.json", "data/instruction/spot_check_train.csv", "data/instruction/spot_check_test.csv", "data/instruction/spot_check_key.csv")
def test_spot_check_files_unchanged():
    h = json.loads(resolve(CFG, "cross_split_overlap").parent.joinpath("..", "instruction", "spot_check_hashes.json")
                   .resolve().read_text(encoding="utf-8"))
    # Stage B: the owner filled the two CSVs on purpose. The file holds the new hashes (frozen again)
    # and the old ones under `before_owner_fill`. The key file was never edited.
    before = h.pop(CFG["stage_b"]["hashes_before_key"])
    assert set(h) == set(before) == {"spot_check_train.csv", "spot_check_test.csv", "spot_check_key.csv"}
    assert [n for n in h if h[n] != before[n]] == ["spot_check_train.csv", "spot_check_test.csv"]
    for name, digest in h.items():
        p = resolve(CFG, "candidates").parent / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f"{name} was changed"


@needs("data/instruction/a3_id_manifest.json", "data/instruction/candidates.jsonl", "data/instruction/test_candidates.jsonl")
def test_no_existing_id_renumbered_or_removed():
    """Ids are never renumbered or removed. The only allowed text change: a val item that was dropped by the
    cap before A3, was re-admitted, and got an AI fix (never verified and never in the spot-check)."""
    import csv
    man = json.loads((REPO_ROOT / CFG["a3"]["id_manifest"]).read_text(encoding="utf-8"))
    now, fixed = {}, set()
    for f in ("candidates", "test_candidates"):
        for r in read_jsonl(resolve(CFG, f)):
            c, q, a = parse_example(r)
            now[r["id"]] = hashlib.sha256((c + "\x00" + q + "\x00" + a).encode()).hexdigest()
            if r.get("ai_fix") and man.get(r["id"], {}).get("dropped_by") == "source_cap" and r["meta"]["split"] == "val":
                fixed.add(r["id"])
    idir = resolve(CFG, "candidates").parent
    spot = set()
    for name in ("spot_check_train.csv", "spot_check_test.csv"):
        with open(idir / name, encoding="utf-8-sig", newline="") as fh:
            spot |= {row["id"] for row in csv.DictReader(fh)}
    missing = [i for i in man if i not in now]
    changed = [i for i in man if i in now and now[i] != man[i]["sha"] and i not in fixed]
    assert not missing, missing
    assert not changed, changed
    assert not (fixed & spot)                                   # nothing the owner sees was touched
    assert len(now) >= len(man)
