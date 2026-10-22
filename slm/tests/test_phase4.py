"""Tests for Phase 4, stage A: evaluator --split train (untuned base only), preference pair builder, P4-0 cell.
CPU only. No torch, transformers, peft."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_config, load_eval_config, load_phase4_config
from climate_slm.data.sft import make_example
from climate_slm.evaluation import evaluate, report
from climate_slm.training import dpo_pairs as D
from _release import needs_training_data

ECFG = load_eval_config()
CFG4 = load_phase4_config()
P1 = load_config()
REF = ECFG["refusal_text"]


def _real(split):
    return D.read_jsonl(REPO_ROOT / CFG4["paths"][f"sft_{split}"])


def _row(rid, chunk, rtype="definition", answer="The answer is cotton.", question="What is it?"):
    meta = {"id": rid, "chunk_id": chunk, "source_id": "src", "type": rtype}
    return make_example(P1, f"context of {chunk}", question, REF if rtype == "unanswerable" else answer, meta)


def _item(rid, output, f1, refusal=False):
    return {"id": rid, "output": output, "token_f1": f1, "refusal_strict": refusal}


LOOSE = ECFG["eval"]["loose_refusal_phrases"]


def _cfg(**over):
    cfg = copy.deepcopy(CFG4)
    cfg["pairs"].update({"include_type2": True, **over})          # the type 2 code is tested with the flag on
    return cfg


def _build(rows, base, cfg=None, exclude=None):
    return D.build_pairs(rows, base, REF, cfg or _cfg(type2_min_pairs=1), exclude, LOOSE)


def test_type2_is_off_by_default_and_the_summary_says_why():
    assert CFG4["pairs"]["include_type2"] is False
    rows, base = _fake_train()
    cfg = _cfg(include_type2=False, refusal_pair_repeat=3)
    pairs, counts, excluded = D.build_pairs(rows, base, REF, cfg, None, LOOSE)
    assert not [p for p in pairs if p["pair_type"] == 2] and excluded == []
    assert counts["type2"] == 0 and counts["type2_included"] is False and counts["type2_dropped_too_few"] is False
    assert counts["type2_f1_below_threshold"] == 0 and counts["type2_accepted_by"] == {
        "key_recall_low_only": 0, "unsupported_number_only": 0, "both": 0}
    assert counts["type1"] == 8 and counts["type3_distinct"] == 2 and counts["distinct_pairs"] == 10
    assert counts["weighted_total"] == 14
    s = D.summarize(pairs, counts, rows, base, cfg, REF, {}, [], excluded)
    assert s["include_type2"] is False and s["counts"]["type2"] == 0
    assert s["type2_status"] == "dropped by reviewer: proxy cannot separate wrong from terse-correct (all 54 read by hand)"
    # turning the flag on brings type 2 back, same code
    assert D.build_pairs(rows, base, REF, _cfg(type2_min_pairs=1), None, LOOSE)[1]["type2"] == 3


def _fake_train():
    rows = [_row(f"a{i}", f"c{i}") for i in range(1, 9)] + [_row(f"u{i}", f"c{i + 8}", "unanswerable") for i in range(1, 6)]
    base = {i["id"]: i for i in [
        _item("a1", "wrong stuff", 0.10),            # F1 low, key_recall 0            -> type 2 (key_recall_low)
        _item("a2", "cotton", 0.30),                 # F1 low, key_recall 0.5, no number -> excluded (not wrong enough)
        _item("a3", "cotton 99", 0.30),              # F1 low, key_recall 0.5, number 99 not in context -> type 2
        _item("a4", REF, 0.0, True),                 # strict refusal: no type 2 (type 1 already says this)
        _item("a5", "", 0.0),                        # empty output: no type 2
        _item("a6", "The answer is cotton.", 1.0),   # F1 high: no type 2
        _item("a7", "nothing 99", 0.45),             # key_recall 0 and unsupported number -> type 2 (both)
        _item("a8", "wrong", 0.50),                  # exactly the F1 limit: NOT below
        _item("u1", "It is blue.", 0.0),             # type 3
        _item("u2", REF, 1.0, True),                 # strict refusal: no type 3
        _item("u3", "I cannot answer that.", 0.0),   # loose refusal only: no type 3
        _item("u4", "Something.", 0.0),              # type 3
        _item("u5", "", 0.0)]}                       # empty: no type 3
    return rows, base


# ---- evaluator: --split train only for the untuned base ---------------------------------------------------------
def test_train_split_is_known_to_the_config_and_the_cli():
    assert "train" in ECFG["paths"]["splits"]
    src = (REPO_ROOT / "climate_slm" / "evaluation" / "evaluate.py").read_text(encoding="utf-8")
    assert 'choices=["val", "test", "train"]' in src


def test_check_train_split_accepts_only_the_plain_base():
    evaluate.check_train_split("base", None, "default", "greedy", False)
    for args in [("outputs/adapters/x/e2", None, "default", "greedy", False),
                 ("base", "Qwen/Qwen2.5-1.5B-Instruct", "default", "greedy", False),
                 ("base", None, "fewshot", "greedy", False),
                 ("base", None, "default", "qwen_default", False),
                 ("base", None, "default", "greedy", True)]:
        with pytest.raises(ValueError, match="train"):
            evaluate.check_train_split(*args)


class _Fake:
    def __init__(self, adapter=None):
        self.adapter = adapter

    def info(self):
        return {"model_id": "fake/model", "revision": "r", "adapter": self.adapter, "dtype": "float16", "device": "none"}

    def generate(self, prompts):
        return [{"text": "cotton", "n_new_tokens": 1, "hit_max_new_tokens": False} for _ in prompts]

    def completion_nll(self, prompts, completions):
        return [(1.0, 1) for _ in completions]


@needs_training_data
def test_run_eval_on_train_works_for_the_base_and_refuses_a_tuned_model(tmp_path):
    out = tmp_path / "eval_base_train.json"
    d = evaluate.run_eval(_Fake(), ECFG, "train", "base", out_path=out)
    assert d["split"] == "train" and d["summary"]["n_items"] == 156 and d["data_file"].endswith("sft_train.jsonl")
    assert d["data_sha256"] == D.file_sha256(REPO_ROOT / CFG4["paths"]["sft_train"])
    assert d["adapter"] is None and out.exists()
    with pytest.raises(ValueError, match="never be used for a tuned model"):
        evaluate.run_eval(_Fake(adapter="outputs/adapters/x/e2"), ECFG, "train", "x", out_path=tmp_path / "y.json")
    assert not (tmp_path / "y.json").exists()
    evaluate.run_eval(_Fake(adapter="a"), ECFG, "val", "x", limit=2)               # val is unchanged for tuned models


def test_cli_refuses_train_for_an_adapter_before_loading_anything(tmp_path, monkeypatch):
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate, "HFBackend", lambda *a, **k: pytest.fail("a model was loaded"))
    with pytest.raises(SystemExit) as e:
        evaluate.main(["--model", str(adapter), "--split", "train"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        evaluate.main(["--model", "base", "--split", "train", "--prompt-variant", "fewshot"])


def test_default_output_name_for_the_train_run():
    assert evaluate.default_out(ECFG, "base", "train").name == "eval_base_train.json"


@needs_training_data
def test_report_ignores_the_train_eval_file(tmp_path):
    evaluate.run_eval(_Fake(), ECFG, "train", "base", out_path=tmp_path / "eval_base_train.json", write_review=False)
    evals, skipped = report.load_evals(tmp_path, ECFG)
    assert evals == [] and len(skipped) == 1 and "train split" in skipped[0]


# ---- pair builder ------------------------------------------------------------------------------------------------
def test_pair_types_and_the_new_type2_rule():
    rows, base = _fake_train()
    pairs, counts, excluded = _build(rows, base, _cfg(type2_min_pairs=1, refusal_pair_repeat=1))
    by = {t: [p for p in pairs if p["pair_type"] == t] for t in (1, 2, 3)}
    assert [len(by[t]) for t in (1, 2, 3)] == [8, 3, 2]
    assert all(p["rejected"] == REF and p["meta"]["rejected_source"] == "refusal_text" for p in by[1])
    assert all(p["chosen"] == "The answer is cotton." for p in by[1] + by[2])
    assert [p["meta"]["row_id"] for p in by[2]] == ["a1", "a3", "a7"]
    assert [p["meta"]["why_rejected"] for p in by[2]] == [["key_recall_low"], ["unsupported_number"],
                                                          ["key_recall_low", "unsupported_number"]]
    assert by[2][1]["meta"]["key_recall"] == 0.5 and by[2][1]["meta"]["number_faithfulness"] == 0.0
    assert counts["type2_accepted_by"] == {"key_recall_low_only": 1, "unsupported_number_only": 1, "both": 1}
    assert counts["type2_f1_below_threshold"] == 4 and counts["type2_excluded_key_ok_and_numbers_ok"] == 1
    assert counts["type2_excluded_f1_not_below_threshold"] == 2 and counts["type2_skipped_base_refusal_or_empty"] == 2
    assert [x["row_id"] for x in excluded] == ["a2"] and excluded[0]["output"] == "cotton" and excluded[0]["key_recall"] == 0.5
    assert all(p["chosen"] == REF for p in by[3]) and [p["rejected"] for p in by[3]] == ["It is blue.", "Something."]
    assert [p["pair_type"] for p in pairs] == sorted(p["pair_type"] for p in pairs)         # order: type 1, 2, 3


def test_key_scores_come_from_the_evaluator_functions():
    rows, _ = _fake_train()
    s = D.untuned_scores(rows[0], "cotton 99", REF)
    from climate_slm.evaluation import metrics as M
    assert s == {"key_recall": M.key_recall("cotton 99", "The answer is cotton.", "What is it?", REF),
                 "number_faithfulness": M.number_faithfulness("cotton 99", "context of c1")}
    # a score that does not apply (None) is never a reason
    assert D.type2_reasons({"key_recall": None, "number_faithfulness": None}, CFG4) == []
    assert D.type2_reasons({"key_recall": 0.49, "number_faithfulness": 1.0}, CFG4) == ["key_recall_low"]
    assert D.type2_reasons({"key_recall": 0.5, "number_faithfulness": 1.0}, CFG4) == []         # exactly 0.5 is not below
    assert D.type2_reasons({"key_recall": 0.5, "number_faithfulness": 0.5}, CFG4) == ["unsupported_number"]


def test_real_eval_file_scores_match_the_recomputed_ones():
    path = REPO_ROOT / CFG4["paths"]["base_train_eval"]
    if not path.exists():
        pytest.skip("eval_base_train.json is not on this machine")
    ev, rows = json.loads(path.read_text(encoding="utf-8")), {r["meta"]["id"]: r for r in _real("train")}
    for it in ev["items"]:
        if it["kind"] == "answerable":
            s = D.untuned_scores(rows[it["id"]], it["output"], REF)
            assert s["key_recall"] == it["key_recall"] and s["number_faithfulness"] == it["number_faithfulness"]


def test_type2_is_removed_when_too_few_pairs_are_left():
    rows, base = _fake_train()
    pairs, counts, _ = _build(rows, base, _cfg(type2_min_pairs=4))                              # only 3 qualify
    assert counts["type2_dropped_too_few"] is True and counts["type2_pairs_before_drop"] == 3 and counts["type2"] == 0
    assert not [p for p in pairs if p["pair_type"] == 2]
    assert counts["type1"] == 8 and counts["type3_distinct"] == 2
    assert _build(rows, base, _cfg(type2_min_pairs=3))[1]["type2_dropped_too_few"] is False       # at the limit: kept
    assert CFG4["pairs"]["type2_min_pairs"] == 20


def test_type3_skips_strict_and_loose_refusals_and_empty_outputs():
    rows, base = _fake_train()
    _, counts, _ = _build(rows, base)
    assert counts["type3_skipped_base_strict_refusal"] == 1          # u2
    assert counts["type3_skipped_base_loose_refusal_only"] == 1      # u3: "I cannot answer that."
    assert counts["type3_skipped_empty"] == 1                        # u5
    pairs = _build(rows, base)[0]
    assert {p["meta"]["row_id"] for p in pairs if p["pair_type"] == 3} == {"u1", "u4"}


def test_refusal_pairs_are_repeated_with_unique_ids_and_a_shared_distinct_id():
    rows, base = _fake_train()
    pairs, counts, _ = _build(rows, base, _cfg(type2_min_pairs=1, refusal_pair_repeat=3))
    t3 = [p for p in pairs if p["pair_type"] == 3]
    assert [p["pair_id"] for p in t3] == ["u1:t3:r1", "u1:t3:r2", "u1:t3:r3", "u4:t3:r1", "u4:t3:r2", "u4:t3:r3"]
    assert {p["meta"]["distinct_pair_id"] for p in t3} == {"u1:t3", "u4:t3"}
    assert [p["meta"]["repeat_no"] for p in t3] == [1, 2, 3, 1, 2, 3]
    assert len({p["pair_id"] for p in pairs}) == len(pairs)                                        # all ids unique
    assert len({p["meta"]["distinct_pair_id"] for p in pairs}) == counts["distinct_pairs"] == 8 + 3 + 2
    assert counts["weighted_total"] == len(pairs) == 8 + 3 + 6 and counts["type3_weighted"] == 6
    t1 = next(p for p in pairs if p["pair_type"] == 1)
    assert t1["pair_id"] == t1["meta"]["distinct_pair_id"] == "a1:t1"                              # no suffix off type 3
    assert CFG4["pairs"]["refusal_pair_repeat"] == 3 and CFG4["pairs"]["rejected_f1_below"] == 0.5
    assert CFG4["pairs"]["rejected_key_recall_below"] == 0.5


def test_a_pair_never_has_chosen_equal_to_rejected_and_keeps_the_sft_prompt():
    rows, base = _fake_train()
    pairs, _, _ = _build(rows, base)
    prompts = {r["meta"]["id"]: r["prompt"] for r in rows}
    for p in pairs:
        assert p["chosen"] != p["rejected"] and p["prompt"] == prompts[p["meta"]["row_id"]]


def test_builder_is_deterministic():
    rows, base = _fake_train()
    a = D.pair_lines(_build(rows, base)[0])
    b = D.pair_lines(_build(copy.deepcopy(rows), copy.deepcopy(base))[0])
    assert a == b
    pairs, _, excluded = _build(rows, base)
    assert D.pick_samples(pairs, 2, 42) == D.pick_samples(pairs, 2, 42)
    assert all(len(v) == 2 for t, v in D.pick_samples(pairs, 2, 42).items())
    assert all(x["meta"]["repeat_no"] == 1 for x in D.pick_samples(pairs, 10, 42)[3])             # repeats are not shown twice
    assert len(D.pick_samples(pairs, 10, 42)[3]) == 2
    assert D.pick_excluded(excluded, 10, 42) == D.pick_excluded(excluded, 10, 42) == excluded


def test_rows_sharing_an_id_or_a_chunk_with_val_or_test_are_excluded():
    rows, base = _fake_train()
    val = [_row("v1", "c2")]                      # same chunk as a2
    test = [_row("a3", "cX")]                     # same id as a3
    excluded = D.rows_to_exclude(rows, val, test)
    assert excluded == {"a2", "a3"}
    pairs, counts, _ = _build(rows, base, exclude=excluded)
    assert counts["rows_excluded_val_test_overlap"] == 2
    assert not {"a2", "a3"} & {p["meta"]["row_id"] for p in pairs}
    assert D.leakage_problems(pairs, val, test) == []
    leaky, _, _ = _build(rows, base)                                                              # without the exclusion
    problems = D.leakage_problems(leaky, val, test)
    assert any("chunk c2" in p for p in problems) and any("row id a3" in p for p in problems)


@needs_training_data
def test_real_train_rows_have_no_id_or_chunk_in_val_or_test():
    tr, va, te = _real("train"), _real("val"), _real("test")
    assert (len(tr), len(va), len(te)) == (156, 53, 43)
    assert D.rows_to_exclude(tr, va, te) == set()
    assert sum(1 for r in tr if not D.is_refusal_row(r)) == 135 and sum(1 for r in tr if D.is_refusal_row(r)) == 21
    # the finished pairs, built from every real train row, are clean too
    base = {r["meta"]["id"]: _item(r["meta"]["id"], "x", 0.0) for r in tr}
    pairs, counts, _ = D.build_pairs(tr, base, REF, CFG4, D.rows_to_exclude(tr, va, te), LOOSE)      # shipped config: no type 2
    assert counts["type1"] == 135 and counts["type3_distinct"] == 21 and counts["type3_weighted"] == 63 and counts["type2"] == 0
    pairs2, counts2, _ = D.build_pairs(tr, base, REF, _cfg(), D.rows_to_exclude(tr, va, te), LOOSE)   # flag on
    assert counts2["type2"] == 135 and D.leakage_problems(pairs2, va, te) == []
    assert D.leakage_problems(pairs, va, te) == []
    assert {p["meta"]["row_id"] for p in pairs} <= {r["meta"]["id"] for r in tr}


@needs_training_data
def test_base_eval_file_is_checked_before_use():
    tr = _real("train")
    sha = D.file_sha256(REPO_ROOT / CFG4["paths"]["sft_train"])
    good = {"split": "train", "limit": None, "adapter": None, "load_4bit": False, "prompt_variant": "default",
            "decoding_mode": "greedy", "data_sha256": sha, "items": [{"id": r["meta"]["id"]} for r in tr]}
    D.check_base_eval(good, tr, sha)
    for key, value in [("split", "val"), ("limit", 3), ("adapter", "outputs/adapters/x"), ("load_4bit", True),
                       ("prompt_variant", "fewshot"), ("decoding_mode", "qwen_default"), ("data_sha256", "0" * 64),
                       ("items", good["items"][:-1])]:
        with pytest.raises(ValueError, match="cannot be used"):
            D.check_base_eval({**good, key: value}, tr, sha)


def test_summary_has_distinct_and_weighted_totals_and_the_exclusion_counts():
    rows, base = _fake_train()
    pairs, counts, excluded = _build(rows, base, _cfg(type2_min_pairs=1, refusal_pair_repeat=3))
    s = D.summarize(pairs, counts, rows, base, _cfg(type2_min_pairs=1, refusal_pair_repeat=3), REF, {"x": "y"}, [], excluded)
    assert s["distinct_pairs"] == 13 and s["weighted_total"] == 17 and s["refusal_pair_repeat"] == 3
    assert "ONE computation per distinct pair" in s["stage_b_note"]
    c = s["counts"]
    assert c["type2_excluded_key_ok_and_numbers_ok"] == 1 and c["type3_skipped_base_loose_refusal_only"] == 1
    assert s["type2_excluded_example_ids"] == ["a2"] and s["base_f1_type2_excluded_by_key_and_number_rule"]["n"] == 1
    assert s["base_f1_all_answerable_train_rows"]["below_threshold"] == 6
    assert sum(b["count"] for b in s["base_f1_all_answerable_train_rows"]["histogram"]) == 8
    assert s["base_f1_type2_rejected"]["n"] == 3 and s["base_f1_type2_rejected"]["max"] == 0.45
    assert s["base_strict_refusals_on_train_answerable"] == 1 and s["base_strict_refusals_on_train_refusal_rows"] == 1
    assert s["leakage"]["clean"] is True
    assert D.summarize(pairs, counts, rows, base, CFG4, REF, {}, ["bad"])["leakage"]["clean"] is False
    assert D.histogram([0.0, 0.05, 0.5, 1.0], [0.0, 0.5, 1.0001]) == [{"from": 0.0, "to": 0.5, "count": 2}, {"from": 0.5, "to": 1.0, "count": 2}]


def test_format_pair_and_excluded_example_show_everything_in_full():
    rows, base = _fake_train()
    pairs, _, excluded = _build(rows, base)
    text = D.format_pair(next(p for p in pairs if p["pair_id"] == "a1:t2"))
    assert "QUESTION: What is it?" in text and "CHOSEN:   The answer is cotton." in text and "REJECTED: wrong stuff" in text
    assert "0.100" in text and "key_recall_low" in text
    t = D.format_excluded(excluded[0])
    assert "[a2]" in t and "REFERENCE: The answer is cotton." in t and "UNTUNED:   cotton" in t and "key_recall 0.500" in t
    assert "n/a (no number)" in t


@needs_training_data
def test_main_writes_files_and_refuses_a_missing_eval_file(tmp_path, monkeypatch, capsys):
    cfg = copy.deepcopy(CFG4)
    monkeypatch.setattr(D, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(D, "load_phase4_config", lambda: cfg)
    for k in ("sft_train", "sft_val", "sft_test"):
        (tmp_path / "data" / "instruction").mkdir(parents=True, exist_ok=True)
        (tmp_path / cfg["paths"][k]).write_bytes((REPO_ROOT / CFG4["paths"][k]).read_bytes())
    (tmp_path / "outputs").mkdir()
    assert D.main([]) == 1 and "P4-0" in capsys.readouterr().out
    tr = _real("train")
    items = [{**_item(r["meta"]["id"], "an answer", 0.2), "kind": "x"} for r in tr]
    (tmp_path / cfg["paths"]["base_train_eval"]).write_text(json.dumps({
        "split": "train", "limit": None, "adapter": None, "load_4bit": False, "prompt_variant": "default",
        "decoding_mode": "greedy", "data_sha256": D.file_sha256(tmp_path / cfg["paths"]["sft_train"]), "items": items}),
        encoding="utf-8")
    assert D.main(["--show"]) == 0
    out = capsys.readouterr().out
    assert "TYPE 1" in out and "TYPE 3" in out and "TYPE 2" not in out
    assert out.count("[tr2-") == 5 + 10                                                           # 5 type 1 + 10 type 3 samples
    written = D.read_jsonl(tmp_path / cfg["paths"]["pairs"])
    n2 = sum(1 for p in written if p["pair_type"] == 2)
    assert n2 == 0 and len(written) == 135 + 21 * 3 and len({p["pair_id"] for p in written}) == len(written)
    sha_line = (tmp_path / cfg["paths"]["pairs_sha256"]).read_text(encoding="utf-8").split()
    assert sha_line[0] == D.file_sha256(tmp_path / cfg["paths"]["pairs"])
    summary = json.loads((tmp_path / cfg["paths"]["pairs_summary"]).read_text(encoding="utf-8"))
    assert summary["leakage"]["clean"] and summary["weighted_total"] == len(written)
    assert summary["distinct_pairs"] == 135 + n2 + 21
    first = (tmp_path / cfg["paths"]["pairs"]).read_bytes()
    D.main([])
    assert (tmp_path / cfg["paths"]["pairs"]).read_bytes() == first                       # same bytes again


def test_pair_builder_needs_no_gpu_packages():
    src = (REPO_ROOT / "climate_slm" / "training" / "dpo_pairs.py").read_text(encoding="utf-8")
    for lib in ("torch", "transformers", "peft", "bitsandbytes"):
        assert f"import {lib}" not in src and f"from {lib}" not in src


# ---- notebook cell P4-0 ------------------------------------------------------------------------------------------
def test_notebook_cell_p4_0():
    nb = json.loads((REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    p4 = [c for c in code if c.lstrip().startswith("# P4-")]
    assert p4[0].startswith("# P4-0.") and len(p4[0].splitlines()) <= 15
    assert "--model base" in p4[0] and "--split train" in p4[0] and "eval_base_train.json" in p4[0]
    assert "adapter" not in p4[0].lower() and "test" not in p4[0].lower()
    assert "dpo" not in " ".join(c for c in code if not c.startswith(("# P4-", "# P4b-"))).lower()          # no DPO code in the old cells


def test_config_has_the_fixed_pair_settings_and_the_bundle_takes_it():
    assert CFG4["pairs"]["rejected_f1_below"] == 0.5 and CFG4["pairs"]["refusal_pair_repeat"] == 3
    assert CFG4["paths"]["base_train_eval"] == "outputs/eval_base_train.json"
    assert CFG4["paths"]["pairs"] == "data/instruction/dpo_train.jsonl"
    cfg = ECFG["bundle"]
    assert {"path": "configs", "endings": [".yaml"]} in cfg["folders"] and {"path": "climate_slm", "endings": [".py"]} in cfg["folders"]
