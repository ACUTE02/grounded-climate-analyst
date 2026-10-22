"""Phase 1 tests. Run with: .venv\\Scripts\\python.exe -m pytest tests -q"""
import csv
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.build import build_candidates, pick_context_chunk
from climate_slm.data.clean import clean_text
from climate_slm.data.dedup import find_duplicates
from climate_slm.data.lexical import BM25
from climate_slm.data.sft import REQUIRED_KEYS, make_example, parse_example
from climate_slm.data.spotcheck import CSV_COLUMNS, sample_spot_check
from climate_slm.data.split import make_splits
from climate_slm.data.textutil import numbers, read_jsonl
from climate_slm.data.validate import check_example
from climate_slm.data.verify import evidence_ok
from conftest import needs

CFG = load_config()
CHUNK = ("Heat wave is considered if maximum temperature of a station reaches at least 400C or more "
         "for Plains and at least 300C or more for Hilly regions.")


def _ex(question, answer, meta_extra=None, chunk=CHUNK, split="train"):
    meta = {"chunk_id": "src::a", "source_id": "src", "split": split, "type": "numeric",
            "generator": "llm_drafted", "verification": "unverified"}
    meta.update(meta_extra or {})
    return make_example(CFG, chunk, question, answer, meta)


def _synthetic_chunks(n=40, source="s1"):
    return [{"id": f"{source}::{i:03d}", "source_id": source, "text": f"text {i}"} for i in range(n)]


# ---- config -----------------------------------------------------------------
def test_config_loads_and_split_ratios_sum_to_one():
    s = CFG["split"]
    assert abs(s["train"] + s["val"] + s["test"] - 1.0) < 1e-9
    assert CFG["refusal_text"] and "{refusal_text}" in CFG["system_prompt"]


# ---- split / leakage --------------------------------------------------------
def test_synthetic_split_blocks_and_buffer():
    chunks = _synthetic_chunks(60)
    sp = make_splits(chunks, CFG)
    bs = CFG["split"]["block_size"]
    assert set(sp.values()) <= {"train", "val", "test", "buffer"}
    assert "buffer" in sp.values()
    # a train chunk directly next to a test chunk must not exist (buffer sits between)
    ids = [c["id"] for c in chunks]
    for a, b in zip(ids, ids[1:]):
        assert {sp[a], sp[b]} != {"train", "test"}
        assert {sp[a], sp[b]} != {"val", "test"}
        assert {sp[a], sp[b]} != {"train", "val"}
    assert bs == 4


def test_split_is_deterministic():
    chunks = _synthetic_chunks(60)
    assert make_splits(chunks, CFG) == make_splits(chunks, CFG)


@needs("data/processed/chunks_clean.jsonl", "data/processed/chunk_splits.json")
def test_no_chunk_id_in_two_splits():
    rows = read_jsonl(resolve(CFG, "chunks_clean"))
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    splits = json.loads(resolve(CFG, "chunk_splits").read_text(encoding="utf-8"))
    assert set(splits) == set(ids)
    assert all(splits[r["id"]] == r["split"] for r in rows)


@needs("data/processed/chunks_clean.jsonl")
def test_no_train_chunk_text_equals_test_chunk_text():
    rows = read_jsonl(resolve(CFG, "chunks_clean"))
    test_texts = {r["text"] for r in rows if r["split"] == "test"}
    train_texts = {r["text"] for r in rows if r["split"] in ("train", "val")}
    assert test_texts and not (test_texts & train_texts)


@needs("data/processed/chunks_clean.jsonl", "data/instruction/candidates.jsonl")
def test_test_chunks_never_in_training_files():
    rows = {r["id"]: r for r in read_jsonl(resolve(CFG, "chunks_clean"))}
    test_texts = [r["text"] for r in rows.values() if r["split"] == "test"]
    for ex in read_jsonl(resolve(CFG, "candidates")):
        assert ex["meta"]["split"] in ("train", "val")
        assert rows[ex["meta"]["chunk_id"]]["split"] == ex["meta"]["split"]
        if ex["meta"]["type"] == "unanswerable":
            assert rows[ex["meta"]["question_chunk_id"]]["split"] == ex["meta"]["split"]
        prompt = ex["prompt"][1]["content"]
        assert not any(t in prompt for t in test_texts)
    for ex in read_jsonl(resolve(CFG, "test_candidates")):
        assert ex["meta"]["split"] == "test"
        assert rows[ex["meta"]["chunk_id"]]["split"] == "test"


# ---- cleaning ---------------------------------------------------------------
def test_cleaning_removes_headers_and_joins_hyphens_but_keeps_numbers():
    raw = "1 | P a g e\n\nMax temp of 400C is reached in the plains and te-\nmperature is 4.50C to 6.4 above normal.\n"
    out, stats = clean_text(raw, CFG)
    assert stats["page_header_removed"] == 1
    assert stats["hyphen_split_joined"] == 1
    assert "P a g e" not in out
    body = raw.replace("1 | P a g e", "")
    assert numbers(out) == numbers(body)
    assert "400C" in out and "4.50C" in out


@needs("data/raw/chunks_snapshot.jsonl")
def test_cleaning_never_changes_numbers_on_real_chunks():
    raw = read_jsonl(resolve(CFG, "raw_chunks"))
    checked = 0
    for r in raw:
        if r["source_type"] != CFG["data"]["source_type_keep"]:
            continue
        out, _ = clean_text(r["text"], CFG)
        import re
        hdr = re.compile(CFG["cleaning"]["page_header_regex"])
        no_hdr = "\n".join(line for line in r["text"].split("\n") if not hdr.match(line))
        # hyphen joins only glue letters, so digit runs never change
        assert numbers(out) == numbers(no_hdr), r["id"]
        checked += 1
    assert checked == CFG["data"]["expected_kept_chunks"]


# ---- SFT format and validation ---------------------------------------------
def test_sft_example_format():
    ex = _ex("What is the plains temperature for a heat wave?", "At least 400C or more for Plains.")
    for k in REQUIRED_KEYS:
        assert ex[k]
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["completion"][0]["role"] == "assistant" and ex["completion"][0]["content"]
    assert ex["prompt"][1]["content"].startswith("Context:\n")
    chunk, q, a = parse_example(ex)
    assert chunk == CHUNK and q.endswith("?") and a


@needs("data/instruction/candidates.jsonl", "data/instruction/test_candidates.jsonl")
def test_all_candidates_have_required_keys_and_exact_refusal():
    for f in ("candidates", "test_candidates"):
        rows = read_jsonl(resolve(CFG, f))
        assert rows
        for ex in rows:
            for k in REQUIRED_KEYS:
                assert ex[k], (f, ex["id"], k)
            answer = ex["completion"][0]["content"]
            assert answer.strip()
            if ex["meta"]["type"] == "unanswerable":
                assert answer == CFG["refusal_text"]
            else:
                assert answer != CFG["refusal_text"]
            assert ex["meta"]["generator"] in ("llm_drafted", "llm_drafted_test")
            assert ex["meta"]["verification"] in ("unverified", "llm_verified")


@needs("data/instruction/candidates.jsonl")
def test_unanswerable_share_is_close_to_config_ratio():
    rows = [r for r in read_jsonl(resolve(CFG, "candidates")) if r["dropped_by"] is None]
    share = sum(1 for r in rows if r["meta"]["type"] == "unanswerable") / len(rows)
    assert abs(share - CFG["unanswerable"]["ratio"]) < 0.03


def test_validation_flags():
    good = _ex("What is the plains temperature for a heat wave?", "At least 400C or more for Plains.")
    assert check_example(good, CFG) == []
    bad_num = _ex("What is the plains temperature for a heat wave?", "At least 450C or more for Plains.")
    assert "number_not_in_chunk" in check_example(bad_num, CFG)
    banned = _ex("According to the passage, what is the plains temperature?", "At least 400C or more for Plains.")
    assert "question_banned_phrase" in check_example(banned, CFG)
    no_q = _ex("What is the plains temperature for a heat wave", "At least 400C or more for Plains.")
    assert "question_no_question_mark" in check_example(no_q, CFG)
    empty = _ex("What is the plains temperature for a heat wave?", "  ")
    assert "empty_answer" in check_example(empty, CFG)
    unsupported = _ex("What is the plains temperature for a heat wave?", "Cyclones form over warm ocean water only.")
    assert "low_support_overlap" in check_example(unsupported, CFG)
    wrong_refusal = _ex("What is the cyclone wind speed limit?", "I do not know.", {"type": "unanswerable"})
    assert "refusal_not_exact" in check_example(wrong_refusal, CFG)
    ok_refusal = _ex("What is the cyclone wind speed limit?", CFG["refusal_text"], {"type": "unanswerable"})
    assert check_example(ok_refusal, CFG) == []


def test_dedup_exact_and_near():
    a = _ex("What is the plains temperature for a heat wave?", "At least 400C or more for Plains.")
    b = _ex("What is the plains temperature for a heat wave?", "At least 400C or more for Plains.")
    c = _ex("What is the plains temperature for a heat wave now?", "At least 400C or more for Plains.")
    d = _ex("Which sensor measures soil moisture in agromet stations?", "Agro-AWS data.")
    dups = find_duplicates([a, b, c, d], CFG["checks"]["near_dup_jaccard"])
    assert dups[1] == "duplicate_exact"
    assert dups[2] in ("near_duplicate", "duplicate_exact")
    assert 3 not in dups and 0 not in dups


def test_dedup_compares_only_inside_same_split():
    a = _ex("What is the plains temperature for a heat wave?", "At least 400C or more for Plains.", split="train")
    b = _ex("What is the plains temperature for a heat wave?", "At least 400C or more for Plains.", split="test")
    assert find_duplicates([a, b], 0.85) == {}


# ---- evidence quote ---------------------------------------------------------
def test_evidence_quote_substring_check():
    chunk = "Heat wave is considered if  maximum temperature\nof a station reaches at least 400C or more."
    assert evidence_ok("maximum temperature of a station reaches at least 400C", chunk)   # whitespace normalised
    assert not evidence_ok("maximum temperature of a station reaches at least 450C", chunk)  # invented number
    assert not evidence_ok("NONE", chunk)
    assert not evidence_ok("", chunk)


# ---- unanswerable pairing ---------------------------------------------------
def test_bm25_finds_lexically_close_chunk_and_pairing_uses_other_source():
    docs = ["heat wave temperature plains hilly regions", "reservoir storage bulletin weekly", "seed subsidy fodder banks"]
    bm = BM25(docs)
    scores = [bm.score("what temperature defines a heat wave in plains", i) for i in range(3)]
    assert scores.index(max(scores)) == 0
    pool = [{"id": "a::1", "source_id": "a", "split": "train", "text": docs[0]},
            {"id": "b::1", "source_id": "b", "split": "train", "text": docs[0] + " heat wave"},
            {"id": "c::1", "source_id": "c", "split": "test", "text": docs[0]}]
    home = pool[0]
    ctx = pick_context_chunk("heat wave temperature plains", home, pool, "lexical_hard", random.Random(1))
    assert ctx["source_id"] != "a" and ctx["split"] == "train"


def test_build_candidates_marks_unanswerable_and_needs_check():
    chunks = [{"id": "a::1", "source_id": "a", "split": "train", "text": CHUNK},
              {"id": "b::1", "source_id": "b", "split": "train", "text": "Reservoir storage bulletin is issued weekly by the commission."}]
    drafts = [{"chunk_id": "a::1", "type": "numeric", "question": f"What is plains temperature number {i} for a heat wave?",
               "answer": "At least 400C or more for Plains."} for i in range(10)]
    cfg2 = json.loads(json.dumps(CFG))
    cfg2["unanswerable"]["strategy"] = "lexical_hard"  # v2 helper needs the old key; A2 build uses the mix
    rows = build_candidates(drafts, chunks, cfg2, "llm_drafted", "x-")
    un = [r for r in rows if r["meta"]["type"] == "unanswerable"]
    assert len(un) == round(CFG["unanswerable"]["ratio"] * 10)  # helper still follows unanswerable.ratio
    for r in un:
        assert r["meta"]["needs_check"] is True and r["meta"]["unanswerable_strategy"] == "lexical_hard"  # old helper
        assert r["completion"][0]["content"] == CFG["refusal_text"]
        assert r["meta"]["chunk_id"] == "b::1" and r["meta"]["question_chunk_id"] == "a::1"


# ---- spot check -------------------------------------------------------------
def _rows(n=100):
    rows = []
    for i in range(n):
        rows.append({"id": f"r{i:03d}", "is_refusal": i % 8 == 0,
                     "ai_verdict": "fix" if i % 13 == 0 else ("fail" if i % 29 == 0 else "pass")})
    return rows


def test_spot_check_sample_respects_seed_size_and_minimums():
    rows = _rows()
    a = sample_spot_check(rows, 30, 42, 8, 5)
    b = sample_spot_check(rows, 30, 42, 8, 5)
    c = sample_spot_check(rows, 30, 7, 8, 5)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]
    assert len(a) == 30 and len({r["id"] for r in a}) == 30
    assert sum(r["is_refusal"] for r in a) >= 8
    assert sum(r["ai_verdict"] in ("fix", "fail") for r in a) >= 5


def test_spot_check_csv_files_if_present():
    for key, n in (("spot_check_train", CFG["spot_check"]["spot_check_train_n"]),
                   ("spot_check_test", CFG["spot_check"]["spot_check_test_n"])):
        p = resolve(CFG, key)
        if not p.exists():
            pytest.skip("spot-check CSV not built yet")
        assert p.read_bytes()[:3] == bytes([0xEF, 0xBB, 0xBF])
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n
        assert list(rows[0].keys()) == CSV_COLUMNS
        assert "ai_verdict" not in rows[0] and "ai_reason" not in rows[0]
        # the owner has filled the files (Stage B): every row is good or bad
        ok = (CFG["stage_b"]["owner_good"], CFG["stage_b"]["owner_bad"])
        assert all(r["chunk_text"].strip() and r["owner_verdict"] in ok for r in rows)
