"""Tests for Phase 4b stage A (DPO v2 pairs): config, candidate checks, refusal-chosen pairs, the two sets, leakage, the Colab
generation plan and cell. CPU only. No torch, transformers, peft."""
import copy
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_config, load_eval_config, load_phase4_config, load_phase4b_config
from climate_slm.data.sft import make_example
from climate_slm.training import dpo2_generate, dpo2_pairs as P
from climate_slm.training import dpo_pairs as D1
from _release import needs_training_data

CFG, CFG4, ECFG, P1 = load_phase4b_config(), load_phase4_config(), load_eval_config(), load_config()
REF, LOOSE = ECFG["refusal_text"], ECFG["eval"]["loose_refusal_phrases"]
SOURCES = CFG["generation"]["sources"]


def _real(split):
    return D1.read_jsonl(REPO_ROOT / CFG["paths"][f"sft_{split}"])


def _row(rid, chunk, rtype="definition", context=None, question="What is it?"):
    meta = {"id": rid, "chunk_id": chunk, "source_id": "src", "type": rtype}
    return make_example(P1, context or f"context of {chunk}", question, REF if rtype == "unanswerable" else f"answer {rid}", meta)


def _fake_rows():
    return [_row("a1", "c1"), _row("a2", "c2"), _row("a3", "c3"), _row("u1", "c4", "unanswerable", "The sky is blue today. Water is wet."),
            _row("u2", "c5", "unanswerable"), _row("u3", "c6", "unanswerable")]


def _texts():
    """Per refusal row, (source, k, text) in source order."""
    return {"u1": [("untuned_greedy", 1, "It is blue."), ("untuned_sampled", 1, "It is blue."),            # duplicate of the first
                   ("untuned_sampled", 2, "Water is wet."),                                              # copies the context
                   ("sft_greedy", 1, REF), ("sft_sampled", 1, "I cannot answer that."),                  # strict, loose-only
                   ("sft_sampled", 2, "   ")],                                                           # empty
            "u2": [("untuned_greedy", 1, "Something invented."), ("sft_sampled", 1, "Another invention.")],
            "u3": [("untuned_greedy", 1, REF)]}                                                          # only a refusal


# ---- config ------------------------------------------------------------------------------------------------------------
def test_config_fixes_the_v2_design():
    g, d = CFG["generation"], CFG["design"]
    assert g["sources"] == ["untuned_greedy", "untuned_sampled", "sft_greedy", "sft_sampled"] and g["n_samples"] == 4
    assert g["sample_seed"] == 42 and g["sampling_mode"] == "qwen_default" and g["sft_seed"] == 42
    assert g["sampling_mode"] in ECFG["eval"]["decoding_modes"] and ECFG["eval"]["decoding_modes"][g["sampling_mode"]]["do_sample"]
    assert d["loss"]["alphas"] == [0.5, 1.0] and d["beta"] == 0.1 and d["learning_rates"] == [1e-5, 2e-5]
    assert d["epochs"] == 3 and d["micro_batch_size"] == 1 and d["effective_batch_size"] == 8 and d["seed"] == 42   # amendment 2026-10-20: 8, not 16
    gr = d["grid"]
    assert gr["pair_sets"] == ["v2-main", "v2-mix"] and gr["alphas"] == [0.5, 1.0] and gr["learning_rates"] == [1e-5, 2e-5]
    n_runs = len(gr["pair_sets"]) * len(gr["alphas"]) * len(gr["learning_rates"])
    assert n_runs == 8 and n_runs * d["epochs"] == 24 and d["grid_reduction"] == "none"
    assert d["val_candidates_tried"] == {"v1": 12, "v2": 24, "total": 36}
    s = d["selection"]
    assert (s["min_correct_refusals_val"], s["max_median_length_ratio"], s["max_hit_max_new_tokens"], s["must_beat_control_false_refusals"],
            s["min_token_f1_val"]) == (7, 1.5, 1, 3, 0.575)
    assert set(CFG["sets"]) == {"v2-main", "v2-mix"} and CFG["sets"]["v2-main"]["type1"] is False


def test_v1_config_is_untouched():
    assert CFG4["sweep"]["betas"] == [0.1, 0.3] and CFG4["pairs"]["include_type2"] is False
    assert CFG["paths"]["main_pairs"] != CFG4["paths"]["pairs"] and CFG["paths"]["summary"] != CFG4["paths"]["pairs_summary"]


# ---- classification and pairs -------------------------------------------------------------------------------------------------
def test_classify_uses_the_strict_and_the_loose_detector():
    assert P.classify("", REF, LOOSE) == "empty" and P.classify("  \n", REF, LOOSE) == "empty"
    assert P.classify(REF, REF, LOOSE) == "strict_refusal"
    assert P.classify("I cannot answer that from the text.", REF, LOOSE) == "loose_refusal_only"
    assert P.classify("The context does not provide the funding.", REF, LOOSE) == "loose_refusal_only"
    assert P.classify("It is blue.", REF, LOOSE) == "ok"


def test_refusal_pairs_stats_dedupe_and_order():
    rows = _fake_rows()
    pairs, st = P.build_refusal_pairs(rows, _texts(), REF, LOOSE, SOURCES)
    assert [p["pair_id"] for p in pairs] == ["u1:t3:untuned_greedy:1", "u1:t3:untuned_sampled:2", "u2:t3:untuned_greedy:1", "u2:t3:sft_sampled:1"]
    assert all(p["chosen"] == REF and p["pair_type"] == 3 and p["rejected"] != REF for p in pairs)
    assert all(p["meta"]["distinct_pair_id"] == p["pair_id"] and p["meta"]["repeat_no"] == 1 for p in pairs)       # no repeats in v2
    assert {k: st["untuned_greedy"][k] for k in ("generated", "kept", "strict_refusal")} == {"generated": 3, "kept": 2, "strict_refusal": 1}
    assert st["untuned_sampled"]["duplicate"] == 1 and st["untuned_sampled"]["kept"] == 1
    assert st["sft_greedy"]["strict_refusal"] == 1 and st["sft_greedy"]["kept"] == 0
    assert st["sft_sampled"]["loose_refusal_only"] == 1 and st["sft_sampled"]["empty"] == 1 and st["sft_sampled"]["kept"] == 1
    assert st["_rows_with_at_least_one_pair"] == 2                                                              # u3 only refused
    total = {k: sum(st[s][k] for s in SOURCES) for k in ("generated", "kept", "duplicate", "empty", "strict_refusal", "loose_refusal_only")}
    assert total["generated"] == total["kept"] + total["duplicate"] + total["empty"] + total["strict_refusal"] + total["loose_refusal_only"]


def test_copies_context_flag():
    pairs, st = P.build_refusal_pairs(_fake_rows(), _texts(), REF, LOOSE, SOURCES)
    by = {p["pair_id"]: p["meta"]["copies_context"] for p in pairs}
    assert by["u1:t3:untuned_sampled:2"] is True and by["u1:t3:untuned_greedy:1"] is False        # "Water is wet." is in the context
    assert st["untuned_sampled"]["kept_copied_from_context"] == 1 and st["untuned_greedy"]["kept_copied_from_context"] == 0
    assert P.copies_context("water IS wet", "The sky. Water is wet.") and not P.copies_context("", "x")


def test_only_refusal_rows_and_no_excluded_rows_get_pairs():
    rows = _fake_rows()
    pairs, _ = P.build_refusal_pairs(rows, _texts(), REF, LOOSE, SOURCES, exclude_ids={"u1"})
    assert {p["meta"]["row_id"] for p in pairs} == {"u2"}
    assert all(p["meta"]["row_type"] == "unanswerable" for p in pairs)
    t1 = P.build_type1(rows, REF, exclude_ids={"a2"})
    assert [p["meta"]["row_id"] for p in t1] == ["a1", "a3"] and all(p["pair_type"] == 1 and p["rejected"] == REF for p in t1)
    assert t1[0]["chosen"] == "answer a1" and t1[0]["pair_id"] == "a1:t1"


# ---- the two sets --------------------------------------------------------------------------------------------------------------
def test_main_and_mix_sets():
    rows = _fake_rows()
    refusal_pairs, _ = P.build_refusal_pairs(rows, _texts(), REF, LOOSE, SOURCES)
    type1 = P.build_type1(rows, REF)
    sets = P.build_sets(refusal_pairs, type1, 42)
    main, mix = P.set_counts(sets["v2-main"]), P.set_counts(sets["v2-mix"])
    assert main == {"lines": 4, "answer_chosen_lines": 0, "refusal_chosen_lines": 4, "distinct_pairs": 4}
    assert mix["refusal_chosen_lines"] == 4 and mix["answer_chosen_lines"] == 3 and mix["distinct_pairs"] == 7     # only 3 type 1 pairs exist
    many = [dict(type1[0], pair_id=f"x{i}:t1", meta={**type1[0]["meta"], "distinct_pair_id": f"x{i}:t1"}) for i in range(10)]
    mix2 = P.build_sets(refusal_pairs, many, 42)["v2-mix"]
    assert P.set_counts(mix2)["answer_chosen_lines"] == P.set_counts(mix2)["refusal_chosen_lines"] == 4              # equal in number
    assert mix2 == P.build_sets(refusal_pairs, many, 42)["v2-mix"]                                                  # seeded
    assert all(p["pair_type"] == 3 for p in sets["v2-main"]) and not [p for p in mix2 + sets["v2-main"] if p["pair_type"] == 2]


def test_rejected_samples_are_seeded_per_source_and_capped():
    rows = _fake_rows()
    pairs, _ = P.build_refusal_pairs(rows, _texts(), REF, LOOSE, SOURCES)
    a = P.pick_rejected_samples(pairs, 1, 42, SOURCES)
    assert a == P.pick_rejected_samples(pairs, 1, 42, SOURCES) and set(a) == set(SOURCES)
    assert len(a["untuned_greedy"]) == 1 and a["sft_greedy"] == [] and len(P.pick_rejected_samples(pairs, 10, 42, SOURCES)["untuned_greedy"]) == 2
    txt = P.format_rejected(next(p for p in pairs if p["pair_id"] == "u1:t3:untuned_sampled:2"))
    assert "copies the context verbatim" in txt and "REJECTED: Water is wet." in txt and "QUESTION: What is it?" in txt


# ---- candidate file ------------------------------------------------------------------------------------------------------------------
def _cand(rows, sha="s" * 64, adapter="a" * 64):
    n = CFG["generation"]["n_samples"]
    ids = [r["meta"]["id"] for r in rows if r["meta"]["type"] == "unanswerable"]
    return {"meta": {"data_sha256": sha, "sft_adapter_sha256": adapter, "n_samples": n, "sample_seeds": [42 + k for k in range(n)]},
            "rows": {i: {"untuned_sampled": ["x"] * n, "sft_greedy": ["y"], "sft_sampled": ["z"] * n} for i in ids}}


def test_check_candidates():
    rows = _fake_rows()
    assert P.check_candidates(_cand(rows), rows, "s" * 64, CFG, "a" * 64) == []
    assert P.check_candidates(_cand(rows, sha="q"), rows, "s" * 64, CFG)                                     # other data file
    assert P.check_candidates(_cand(rows), rows, "s" * 64, CFG, "b" * 64)                                    # other adapter
    bad = _cand(rows)
    bad["meta"]["sample_seeds"] = [1, 2, 3, 4]
    assert P.check_candidates(bad, rows, "s" * 64, CFG)
    bad = _cand(rows)
    bad["rows"].pop("u3")
    assert any("exactly the refusal rows" in x for x in P.check_candidates(bad, rows, "s" * 64, CFG))
    bad = _cand(rows)
    bad["rows"]["u1"]["sft_sampled"] = ["z"]
    assert any("u1: sft_sampled has 1 texts, expected 4" in x for x in P.check_candidates(bad, rows, "s" * 64, CFG))


def test_collect_texts_orders_sources_and_reports_missing():
    base = {"items": [{"id": "u1", "output": "base 1"}, {"id": "u2", "output": "base 2"}]}
    texts, missing = P.collect_texts(["u1", "u2"], base, None, SOURCES)
    assert missing == ["untuned_sampled", "sft_greedy", "sft_sampled"] and texts["u1"] == [("untuned_greedy", 1, "base 1")]
    cand = {"rows": {"u1": {"untuned_sampled": ["s1", "s2"], "sft_greedy": ["g"], "sft_sampled": ["t1"]},
                     "u2": {"untuned_sampled": [], "sft_greedy": ["g2"], "sft_sampled": []}}}
    texts, missing = P.collect_texts(["u1", "u2"], base, cand, SOURCES)
    assert missing == [] and [(s, k) for s, k, _ in texts["u1"]] == [("untuned_greedy", 1), ("untuned_sampled", 1), ("untuned_sampled", 2),
                                                                    ("sft_greedy", 1), ("sft_sampled", 1)]


# ---- real data ------------------------------------------------------------------------------------------------------------------
@needs_training_data
def test_real_train_has_21_refusal_rows_and_no_leak_into_val_or_test():
    tr, va, te = _real("train"), _real("val"), _real("test")
    assert sum(1 for r in tr if r["meta"]["type"] == "unanswerable") == 21 and D1.rows_to_exclude(tr, va, te) == set()
    type1 = P.build_type1(tr, REF)
    assert len(type1) == 135 and D1.leakage_problems(type1, va, te) == []


def test_partial_build_on_the_real_files(tmp_path, monkeypatch, capsys):
    """With only the v1 untuned file: counts and samples are printed, no pair file is written, the summary says partial."""
    if not (REPO_ROOT / CFG["paths"]["base_train_eval"]).exists():
        pytest.skip("eval_base_train.json is not on this machine")
    cfg = copy.deepcopy(CFG)
    for k in ("sft_train", "sft_val", "sft_test"):
        (tmp_path / "data" / "instruction").mkdir(parents=True, exist_ok=True)
        (tmp_path / cfg["paths"][k]).write_bytes((REPO_ROOT / CFG["paths"][k]).read_bytes())
    (tmp_path / "outputs").mkdir()
    (tmp_path / cfg["paths"]["base_train_eval"]).write_bytes((REPO_ROOT / CFG["paths"]["base_train_eval"]).read_bytes())
    monkeypatch.setattr(P, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(P, "load_phase4b_config", lambda: cfg)
    assert P.main(["--show"]) == 0
    out = capsys.readouterr().out
    assert "PARTIAL" in out and "REJECTED TEXTS FROM untuned_greedy (10 shown of 18 kept)" in out
    s = json.loads((tmp_path / cfg["paths"]["summary"]).read_text(encoding="utf-8"))
    assert s["partial"] is True and s["missing_sources"] == ["untuned_sampled", "sft_greedy", "sft_sampled"]
    assert s["per_source"]["untuned_greedy"]["kept"] == 18 and s["per_source"]["untuned_greedy"]["loose_refusal_only"] == 3
    assert s["sets"]["v2-main"]["lines"] == 18 and s["sets"]["v2-mix"]["answer_chosen_lines"] == 18 and s["type1_pairs_available"] == 135
    assert s["leakage"]["clean"] and not (tmp_path / cfg["paths"]["main_pairs"]).exists() and not (tmp_path / cfg["paths"]["mix_pairs"]).exists()


def test_complete_build_with_a_candidate_file_writes_both_sets(tmp_path, monkeypatch, capsys):
    if not (REPO_ROOT / CFG["paths"]["base_train_eval"]).exists():
        pytest.skip("eval_base_train.json is not on this machine")
    cfg = copy.deepcopy(CFG)
    (tmp_path / "data" / "instruction").mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    for k in ("sft_train", "sft_val", "sft_test", "base_train_eval"):
        (tmp_path / cfg["paths"][k]).write_bytes((REPO_ROOT / CFG["paths"][k]).read_bytes())
    tr = _real("train")
    n = cfg["generation"]["n_samples"]
    ids = [r["meta"]["id"] for r in tr if r["meta"]["type"] == "unanswerable"]
    cand = {"meta": {"data_sha256": D1.file_sha256(tmp_path / cfg["paths"]["sft_train"]),
                     "sft_adapter_sha256": CFG4["start"]["seeds"][42]["sha256"], "n_samples": n,
                     "sample_seeds": [42 + k for k in range(n)]},
            "rows": {i: {"untuned_sampled": [f"invented {i} u{k}" for k in range(n)], "sft_greedy": [REF],
                         "sft_sampled": [REF, f"invented {i} s1", f"invented {i} s1", "I cannot answer."]} for i in ids}}
    (tmp_path / cfg["paths"]["candidates"]).write_text(json.dumps(cand), encoding="utf-8")
    monkeypatch.setattr(P, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(P, "load_phase4b_config", lambda: cfg)
    assert P.main(["--no-write"]) == 0                                                    # look first: no pair file
    assert not (tmp_path / cfg["paths"]["main_pairs"]).exists() and not (tmp_path / cfg["paths"]["mix_sha256"]).exists()
    assert "--no-write: NO pair file written." in capsys.readouterr().out
    assert json.loads((tmp_path / cfg["paths"]["summary"]).read_text(encoding="utf-8"))["pair_files_written"] is False
    assert P.main([]) == 0
    s = json.loads((tmp_path / cfg["paths"]["summary"]).read_text(encoding="utf-8"))
    assert s["pair_files_written"] is True and s["partial"] is False and s["missing_sources"] == []
    assert s["per_source"]["sft_greedy"]["strict_refusal"] == 21 and s["per_source"]["sft_sampled"]["duplicate"] == 21
    assert s["per_source"]["sft_sampled"]["strict_refusal"] == 21 and s["per_source"]["sft_sampled"]["loose_refusal_only"] == 21
    assert s["per_source"]["untuned_sampled"]["kept"] == 21 * n and s["refusal_rows_with_at_least_one_pair"] == 21
    n_main = s["sets"]["v2-main"]["lines"]
    assert n_main == s["totals"]["kept"] == 18 + 21 * n + 21
    assert s["sets"]["v2-mix"]["refusal_chosen_lines"] == s["sets"]["v2-mix"]["answer_chosen_lines"] == n_main == 123
    for name in ("v2-main", "v2-mix"):
        pf, sf = (tmp_path / cfg["paths"][k] for k in P.SET_FILES[name])
        lines = D1.read_jsonl(pf)
        assert sf.read_text(encoding="utf-8").split()[0] == D1.file_sha256(pf) and len(lines) == s["sets"][name]["lines"]
        assert len({x["pair_id"] for x in lines}) == len(lines) and D1.leakage_problems(lines, _real("val"), _real("test")) == []
    assert all(x["pair_type"] == 3 for x in D1.read_jsonl(tmp_path / cfg["paths"]["main_pairs"]))
    first = (tmp_path / cfg["paths"]["mix_pairs"]).read_bytes()
    P.main([])
    assert (tmp_path / cfg["paths"]["mix_pairs"]).read_bytes() == first                                              # deterministic


def test_main_refuses_a_candidate_file_for_other_data(tmp_path, monkeypatch, capsys):
    if not (REPO_ROOT / CFG["paths"]["base_train_eval"]).exists():
        pytest.skip("eval_base_train.json is not on this machine")
    cfg = copy.deepcopy(CFG)
    (tmp_path / "data" / "instruction").mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    for k in ("sft_train", "sft_val", "sft_test", "base_train_eval"):
        (tmp_path / cfg["paths"][k]).write_bytes((REPO_ROOT / CFG["paths"][k]).read_bytes())
    (tmp_path / cfg["paths"]["candidates"]).write_text(json.dumps({"meta": {"data_sha256": "x"}, "rows": {}}), encoding="utf-8")
    monkeypatch.setattr(P, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(P, "load_phase4b_config", lambda: cfg)
    assert P.main([]) == 1 and "cannot be used" in capsys.readouterr().out
    assert not (tmp_path / cfg["paths"]["main_pairs"]).exists()


# ---- generation (Colab) ----------------------------------------------------------------------------------------------------------------
@needs_training_data
def test_generation_plan():
    plan = dpo2_generate.generation_plan(CFG)
    assert [(p["source"], p["seed"]) for p in plan] == [("untuned_sampled", s) for s in (42, 43, 44, 45)] + [("sft_greedy", None)] + \
        [("sft_sampled", s) for s in (42, 43, 44, 45)]
    assert all(p["decoding"] == "qwen_default" for p in plan if p["seed"] is not None) and plan[4]["decoding"] == "greedy"
    assert all(p["source"] in SOURCES and p["source"] != "untuned_greedy" for p in plan)
    assert dpo2_generate.sample_seeds(CFG) == [42, 43, 44, 45]
    rows = dpo2_generate.refusal_rows(_real("train"))
    assert len(rows) == 21 and all(r["meta"]["split"] == "train" for r in rows)


def test_generation_module_structure_without_gpu():
    src = (REPO_ROOT / "climate_slm" / "training" / "dpo2_generate.py").read_text(encoding="utf-8")
    assert "check_start_adapter" in src and "sft_val" not in src and "sft_test" not in src and '"test"' not in src
    for lib in ("torch", "transformers", "peft"):
        assert f"\nimport {lib}" not in src.split("def run(")[0] and f"\nfrom {lib}" not in src.split("def run(")[0]
    assert "HFBackend" in src and "backend.seed" in src and "backend.params" in src
    pairs_src = (REPO_ROOT / "climate_slm" / "training" / "dpo2_pairs.py").read_text(encoding="utf-8")
    for lib in ("torch", "transformers", "peft", "bitsandbytes"):
        assert f"import {lib}" not in pairs_src and f"from {lib}" not in pairs_src


def test_notebook_cell_p4b_0_and_bundle():
    nb = json.loads((REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    p4b = [c for c in code if c.lstrip().startswith("# P4b-")]
    assert p4b[0].startswith("# P4b-0.") and len(p4b[0].splitlines()) <= 15
    assert "dpo2_generate" in p4b[0] and "dpo2_candidates.json" in p4b[0] and "test" not in p4b[0].lower()
    assert [c for c in code if "dpo2_generate" in c] == [p4b[0]]
    assert not any(c.startswith("# P4-") and "dpo2" in c for c in code)
    z = REPO_ROOT / ECFG["paths"]["colab_bundle"]
    if z.exists():
        names = zipfile.ZipFile(z).namelist()
        for n in ("climate_slm/training/dpo2_pairs.py", "climate_slm/training/dpo2_generate.py", "configs/phase4b.yaml"):
            assert n in names, n
