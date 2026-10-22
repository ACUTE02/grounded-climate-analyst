"""Tests for Phase 2c: key-answer metrics, prompt variants, decoding modes, report sections, refusal sheet, collector.
CPU only. The model is a fake stub; HFBackend.generate is checked with a fake torch / tokenizer / model."""
import fnmatch
import hashlib
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_refusal_review
from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.data.sft import parse_example
from climate_slm.evaluation import collect, evaluate, metrics as M, prompts as P, report, review_sheet, scoring
from _release import needs_training_data

CFG = load_eval_config()
E = CFG["eval"]
REF = CFG["refusal_text"]


def _items(split):
    return scoring.load_items(REPO_ROOT / CFG["paths"]["splits"][split])


def _close(a, b):
    """Equal, with floats compared to 1e-12. (Colab ran Python 3.12, whose sum() is compensated, so the last
    digit of a bootstrap bound can differ from the same code on Python 3.11.)"""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == pytest.approx(b, rel=1e-12, abs=1e-12)
    return a == b


def _real(split, item_id):
    it = next(i for i in _items(split) if i["meta"]["id"] == item_id)
    context, question, reference = parse_example(it)
    return question, reference


# ---- key_recall / key_support ---------------------------------------------------------------
def test_stop_words_and_content_words():
    assert {"the", "of", "which", "not", "kg", "ha"} <= M.STOP_WORDS
    assert M.content_words("The Rice yield, of 24.8% of 2037 farmers; rice at Rs 5") == ["rice", "yield", "farmers"]
    assert M.word_in("adopter", {"adopters"}) and M.word_in("adopters", {"adopter"}) and not M.word_in("adopt", {"adopters"})
    assert not M.word_in("is", {"i"})                                          # no stripping of very short words


def test_key_recall_bare_value_on_short_reference_is_high():
    q = "How much higher was the rice yield of adopters compared with non-adopters?"
    ref = "The rice yield of adopters was 24.8% higher than that of non-adopters."
    assert M.key_tokens(ref, q) == ([{"24.8"}], [])
    assert M.key_recall("24.8%", ref, q, REF) == 1.0
    assert M.key_recall("The yield was 24.8% higher.", ref, q, REF) == 1.0
    assert M.key_support("24.8%", ref, q, REF) == 1.0


def test_key_recall_real_rice_item_documents_the_dilution():
    """Real item te2-0036: the reference also states 6741 and 5403. Bare 24.8% finds 1 of 7 key tokens, so
    key_recall is LOW although the answer is right. key_support (output side) is 1.0. See the Phase 2c report."""
    q, ref = _real("test", "te2-0036")
    nums, words = M.key_tokens(ref, q)
    assert nums == [{"6741"}, {"5403"}, {"24.8"}] and words == ["average", "reported", "against", "translated"]
    assert M.key_recall("24.8%", ref, q, REF) == pytest.approx(1 / 7)
    assert M.key_support("24.8%", ref, q, REF) == 1.0
    assert M.key_recall(ref, ref, q, REF) == 1.0                               # the full reference finds everything


def test_key_recall_partial_wrong_number_and_refusal():
    q, ref = _real("test", "te2-0044")                                         # cotton AND rice are in the reference
    cotton = M.key_recall("cotton", ref, q, REF)
    both = M.key_recall("cotton and rice", ref, q, REF)
    assert 0 < cotton < both < 1                                               # partial
    assert M.key_support("cotton", ref, q, REF) == 1.0                         # what it said is backed by the reference
    q2, ref2 = _real("test", "te2-0036")
    assert M.key_recall("22.4%", ref2, q2, REF) == 0.0                         # wrong number: nothing found
    assert M.key_support("22.4%", ref2, q2, REF) == 0.0                        # and not backed
    for fn in (M.key_recall, M.key_support):
        assert fn(REF, ref2, q2, REF) == 0.0                                   # strict refusal scores 0 on both
        assert fn("  i CANNOT answer this from the provided context. ", ref2, q2, REF) == 0.0


def test_key_numbers_lost_degree_sign_question_numbers_and_plural():
    ref = "Heat Wave may be described if the departure is 4.50C or more and the actual maximum temperature is 370C or more."
    q = "What is the criterion for describing a heat wave at coastal stations?"
    assert M.key_tokens(ref, q)[0] == [{"4.5"}, {"370", "37"}]
    words = M.key_tokens(ref, q)[1]
    assert M.key_recall("4.5 and 37", ref, q) == pytest.approx(2 / (2 + len(words)))    # both numbers found (37 = 370C)
    assert M.key_recall("4.5 and 99", ref, q) == pytest.approx(1 / (2 + len(words)))
    # a number that is also in the question is not a key token
    assert M.key_tokens("It is 45 degrees in 2037 for farmers.", "What happens at 45 degrees?")[0] == [{"2037"}]
    # plural tolerance, both directions
    assert M.key_recall("adopters", "adopter", "x?") == 1.0 and M.key_recall("adopter", "adopters", "x?") == 1.0


def test_key_scores_none_cases():
    assert M.key_recall("anything", "It is the one of them.", "What is it?") is None          # no key token in reference
    assert M.key_support("the of and", "reference with words", "q?") is None                   # nothing key-like in output
    assert M.key_support("", "reference with words", "q?") is None
    assert M.key_recall("", "Cotton lost income.", "What is it?") == 0.0                       # empty output finds nothing


# ---- scoring: records, summary, old files -----------------------------------------------------
class _Fake:
    """Answers the reference words 'cotton' to answerable-looking prompts; refuses every 4th item."""

    def __init__(self):
        self.prompts = []
        self.nll_prompts = None

    def info(self):
        return {"model_id": "fake/model", "revision": "r1", "adapter": None, "dtype": "float16", "device": "none"}

    def generate(self, prompts):
        self.prompts = prompts
        return [{"text": REF if i % 4 == 0 else "cotton 37", "n_new_tokens": 3, "hit_max_new_tokens": False}
                for i, _ in enumerate(prompts)]

    def completion_nll(self, prompts, completions):
        self.nll_prompts = prompts
        return [(1.0 * len(c.split()), len(c.split())) for c in completions]


def test_records_and_summary_have_key_scores(tmp_path):
    d = evaluate.run_eval(_Fake(), CFG, "test", "fake", out_path=tmp_path / "eval_fake_test.json")
    r = d["items"][0]
    assert {"key_recall", "key_support"} <= set(r)
    a = d["summary"]["answerable"]
    for k in ("key_recall", "key_support"):
        b = a[k]
        assert b["n_answerable"] == 32 and 0 < b["n_applied"] <= 32 and b["n"] == b["n_applied"]
        assert b["ci95"][0] <= b["mean"] <= b["ci95"][1]
    refused = [x for x in d["items"] if x["kind"] == "answerable" and x["refusal_strict"]]
    assert refused and all(x["key_recall"] == 0.0 and x["key_support"] == 0.0 for x in refused)


def test_old_style_eval_file_gets_the_same_key_scores(tmp_path):
    d = evaluate.run_eval(_Fake(), CFG, "test", "fake", out_path=tmp_path / "eval_fake_test.json")
    old = json.loads(json.dumps(d))
    for r in old["items"]:
        r.pop("key_recall"), r.pop("key_support")
    old["summary"]["answerable"].pop("key_recall"), old["summary"]["answerable"].pop("key_support")
    for k in ("prompt_variant", "decoding_mode", "seed", "fewshot_ids", "system_prompt"):
        old.pop(k)
    assert scoring.add_key_scores(old, CFG) is True
    assert old["summary"]["answerable"]["key_recall"] == d["summary"]["answerable"]["key_recall"]
    assert old["summary"]["answerable"]["key_support"] == d["summary"]["answerable"]["key_support"]
    assert [r["key_recall"] for r in old["items"]] == [r["key_recall"] for r in d["items"]]
    assert scoring.add_key_scores(old, CFG) is False                                  # second call changes nothing
    assert scoring.add_key_scores(d, CFG) is False                                    # new files are left alone


def test_summarize_reproduces_saved_base_numbers_if_present():
    """The scoring code gives the same numbers as the saved Colab files (default run is unchanged)."""
    for split in ("test", "val"):
        p = REPO_ROOT / "outputs" / f"eval_base_{split}.json"
        if not p.exists():
            pytest.skip("saved base eval files not here")
        saved = json.loads(p.read_text(encoding="utf-8"))
        again = scoring.summarize(saved["items"], CFG)
        for part in ("answerable", "refusal"):
            for k, v in saved["summary"][part].items():
                assert _close(again[part][k], v), (split, part, k)
        assert _close(again["perplexity"], saved["summary"]["perplexity"])
        assert saved["decoding"] == {**{k: E[k] for k in ("do_sample", "max_new_tokens", "repetition_penalty",
                                                          "padding_side", "batch_size")}}


# ---- prompt variants ----------------------------------------------------------------------------
@pytest.mark.parametrize("split", ["val", "test"])
def test_default_variant_is_the_stored_prompt(split):
    items = _items(split)
    prompts, info = P.build_prompts(items, "default", CFG)
    assert prompts == [it["prompt"] for it in items]
    assert info["fewshot_ids"] is None and info["prompt_variant"] == "default"


def test_strict_sentence_changes_only_the_system_message():
    items = _items("test")
    prompts, info = P.build_prompts(items, "strict_sentence", CFG)
    for it, m in zip(items, prompts):
        assert len(m) == 2 and m[1] == it["prompt"][1]                              # same context and question
        assert m[0]["content"].startswith(it["prompt"][0]["content"])
        assert m[0]["content"] != it["prompt"][0]["content"]
    sys_text = prompts[0][0]["content"]
    assert "one complete sentence" in sys_text and sys_text.count(REF) == 2          # old sentence + the new one
    assert sys_text.endswith(REF) and "{" not in sys_text
    assert info["system_prompt"] == sys_text and info["fewshot_ids"] is None


@needs_training_data
def test_fewshot_prompt_structure():
    items = _items("test")
    shots = P.load_fewshot(CFG)
    prompts, info = P.build_prompts(items, "fewshot", CFG)
    strict, _ = P.build_prompts(items, "strict_sentence", CFG)
    assert info["fewshot_ids"] == [s["meta"]["id"] for s in shots]
    assert [kind for kind in E["fewshot"]["order"]] == [scoring.kind_of(s) for s in shots]
    for it, m, st in zip(items, prompts, strict):
        assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]
        assert m[0] == st[0] and m[-1] == it["prompt"][1]                           # same system as strict, real item last
        for k, sh in enumerate(shots):
            assert m[1 + 2 * k]["content"] == sh["prompt"][1]["content"]
            assert m[2 + 2 * k]["content"] == sh["completion"][0]["content"]
    assert shots[[scoring.kind_of(s) for s in shots].index("refusal")]["completion"][0]["content"] == REF


@needs_training_data
def test_fewshot_examples_are_train_only_and_follow_the_rule():
    ids = E["fewshot"]["ids"]
    train, val, test = _items("train"), _items("val"), _items("test")
    by_id = {i["meta"]["id"]: i for i in train}
    held_out_ids = {i["meta"]["id"] for i in val + test}
    held_out_chunks = {c for i in val + test for c in (i["meta"]["chunk_id"], i["meta"].get("question_chunk_id")) if c}
    held_out_prompts = {i["prompt"][1]["content"] for i in val + test}
    held_out_contexts = {parse_example(i)[0] for i in val + test}
    assert E["fewshot"]["split"] == "train"
    assert ids == P.pick_fewshot_ids(train, CFG)                                    # config == the seeded rule
    for kind, i in ids.items():
        it = by_id[i]                                                              # KeyError = not in train
        assert i not in held_out_ids and scoring.kind_of(it) == kind
        assert it["meta"]["split"] == "train"
        assert not ({it["meta"]["chunk_id"], it["meta"].get("question_chunk_id")} & held_out_chunks)
        assert it["prompt"][1]["content"] not in held_out_prompts and parse_example(it)[0] not in held_out_contexts
        assert it["meta"]["spot_check"] in E["fewshot"]["require_spot_check"] and not it["meta"]["blind_flag"]
        assert len(parse_example(it)[0]) <= E["fewshot"]["max_context_chars"]


@needs_training_data
def test_fewshot_rejects_wrong_ids(monkeypatch):
    bad = json.loads(json.dumps(CFG))
    bad["eval"]["fewshot"]["ids"]["answerable"] = "te2-0001"                         # a test id
    with pytest.raises(ValueError):
        P.load_fewshot(bad)
    bad["eval"]["fewshot"]["ids"]["answerable"] = CFG["eval"]["fewshot"]["ids"]["refusal"]   # wrong kind
    with pytest.raises(ValueError):
        P.load_fewshot(bad)
    with pytest.raises(ValueError):
        P.variant_config(CFG, "nonsense")


def test_run_names():
    n = lambda *a: P.run_name(*a, CFG)
    assert n("base", "default", "greedy", None) == "base"
    assert n("base", "strict_sentence", "greedy", None) == "base_p_sentence"
    assert n("base", "fewshot", "greedy", None) == "base_p_fewshot"
    assert n("base", "default", "qwen_default", 1) == "base_qwen_s1"
    assert n("base", "fewshot", "qwen_default", 3) == "base_p_fewshot_qwen_s3"
    assert n("qlora", "default", "greedy", None) == "qlora"


# ---- evaluator: variants, decoding ------------------------------------------------------------------
@needs_training_data
def test_run_eval_variant_uses_the_same_prompt_for_generation_and_perplexity(tmp_path):
    fake = _Fake()
    d = evaluate.run_eval(fake, CFG, "val", "base_p_fewshot", out_path=tmp_path / "x.json", prompt_variant="fewshot",
                          limit=5)
    assert fake.prompts == fake.nll_prompts and len(fake.prompts[0]) == 6
    assert d["prompt_variant"] == "fewshot" and d["fewshot_ids"] == [E["fewshot"]["ids"][k] for k in E["fewshot"]["order"]]
    assert d["system_prompt"] == fake.prompts[0][0]["content"]
    plain = evaluate.run_eval(_Fake(), CFG, "val", "base", limit=5)
    assert all(a["prompt_chars"] > b["prompt_chars"] for a, b in zip(d["items"], plain["items"]))   # real prompt length
    assert plain["prompt_variant"] == "default" and plain["fewshot_ids"] is None
    assert plain["decoding_mode"] == "greedy" and plain["seed"] is None


def test_default_run_writes_the_same_decoding_block_as_before():
    d = evaluate.run_eval(_Fake(), CFG, "test", "base", limit=3)
    assert d["decoding"] == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0,
                             "padding_side": "left", "batch_size": E["batch_size"]}
    assert evaluate.decoding_params(CFG, "greedy") == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0}


def test_sampling_mode_needs_a_seed_and_is_recorded():
    with pytest.raises(ValueError):
        evaluate.run_eval(_Fake(), CFG, "test", "base_qwen_s1", limit=3, decoding="qwen_default")
    d = evaluate.run_eval(_Fake(), CFG, "test", "base_qwen_s2", limit=3, decoding="qwen_default", seed=2)
    assert d["decoding_mode"] == "qwen_default" and d["seed"] == 2
    assert d["decoding"]["do_sample"] is True and d["decoding"]["temperature"] == 0.7
    assert (d["decoding"]["top_p"], d["decoding"]["top_k"], d["decoding"]["repetition_penalty"]) == (0.8, 20, 1.1)
    with pytest.raises(ValueError):
        evaluate.decoding_params(CFG, "nonsense")


def test_command_line_checks_happen_before_any_model_is_loaded():
    for argv in (["--split", "test", "--decoding", "qwen_default"],               # no seed
                 ["--split", "test", "--decoding", "beam"],
                 ["--split", "test", "--prompt-variant", "nonsense"]):
        with pytest.raises(SystemExit):
            evaluate.main(argv)
    assert "torch" not in sys.modules and "transformers" not in sys.modules


class _Ten:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, idx):
        rs, cs = idx
        return _Ten([r[cs] for r in self.rows[rs]])

    def tolist(self):
        return [list(r) for r in self.rows]


class _Enc(dict):
    def to(self, device):
        return self


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTorch:
    def __init__(self):
        self.seeds = []
        self.cuda = types.SimpleNamespace(manual_seed_all=lambda s: self.seeds.append(("cuda", s)))

    def inference_mode(self):
        return _Ctx()

    def manual_seed(self, s):
        self.seeds.append(("cpu", s))


class _FakeTok:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "|".join(m["content"] for m in messages)

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        return _Enc(input_ids=_Ten([[1, 2, 3] for _ in texts]))

    def decode(self, ids, skip_special_tokens):
        return " ".join(map(str, ids))


class _FakeLM:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        return _Ten([[1, 2, 3, 7, 8, 99, 5, 5] for _ in kw["input_ids"].rows])


def _backend(decoding, seed):
    b = object.__new__(evaluate.HFBackend)
    b.e, b.torch, b.params, b.seed = E, _FakeTorch(), evaluate.decoding_params(CFG, decoding), seed
    b.tok, b.model, b.eos_ids = _FakeTok(), _FakeLM(), {99}
    return b


def test_hf_generate_arguments_greedy_are_unchanged_and_sampling_is_seeded():
    msgs = _items("test")[0]["prompt"]
    g = _backend("greedy", None)
    out = g.generate([msgs] * 3)
    kw = {k: v for k, v in g.model.calls[0].items() if k != "input_ids"}
    assert kw == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0, "temperature": None,
                  "top_p": None, "top_k": None, "pad_token_id": 0}                 # exactly the old call
    assert g.torch.seeds == []
    assert [o["text"] for o in out] == ["7 8"] * 3 and out[0]["n_new_tokens"] == 2 and out[0]["hit_max_new_tokens"] is False
    s = _backend("qwen_default", 3)
    s.generate([msgs] * 3)
    kw = {k: v for k, v in s.model.calls[0].items() if k != "input_ids"}
    assert kw == {"do_sample": True, "max_new_tokens": 256, "repetition_penalty": 1.1, "temperature": 0.7,
                  "top_p": 0.8, "top_k": 20, "pad_token_id": 0}
    assert s.torch.seeds == [("cpu", 3), ("cuda", 3)]


# ---- report: key columns, variant rows, decoding section -------------------------------------------
class _Seeded(_Fake):
    """Refuses (exact text) on the LAST n items, answers 'cotton' on the rest. (The 11 refusal items are the last 11.)"""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def generate(self, prompts):
        self.prompts = prompts
        return [{"text": REF if i >= len(prompts) - self.n else "cotton", "n_new_tokens": 2, "hit_max_new_tokens": False}
                for i, _ in enumerate(prompts)]


def _write_run(tmp_path, backend, name, split="test", **kw):
    evaluate.run_eval(backend, CFG, split, name, out_path=tmp_path / f"eval_{name}_{split}.json", **kw)


def test_report_has_key_columns_variant_rows_and_decoding_section(tmp_path):
    kinds = [scoring.kind_of(i) for i in _items("test")]
    ns = {1: 3, 2: 7, 3: 12}
    _write_run(tmp_path, _Fake(), "base")
    _write_run(tmp_path, _Fake(), "base_p_sentence", prompt_variant="strict_sentence")
    for seed, n in ns.items():
        _write_run(tmp_path, _Seeded(n), f"base_qwen_s{seed}", decoding="qwen_default", seed=seed)
    report.main(["--outputs", str(tmp_path)])
    table = (tmp_path / "baseline_table.md").read_text(encoding="utf-8")
    main_part, dec_part = table.split("## Decoding sensitivity (base only)")
    assert "key recall" in main_part and "key support" in main_part
    assert "base Qwen, strict_sentence prompt" in main_part and "strict_sentence" in main_part.split("### Source files")[1]
    assert "base Qwen, fewshot prompt | not yet measured" in main_part                      # missing run is shown as such
    assert "base_qwen_s" not in main_part and "qwen_default" not in main_part.split("### Source files")[0]
    # decoding section: counts of strict correct refusals per seed are known from the fake
    correct = {s: sum(k == "refusal" for k in kinds[-n:]) for s, n in ns.items()}
    test_part = dec_part.split("### Split: val")[0]
    fmt = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.1f}"
    rows = {l.split("|")[1].strip(): l for l in test_part.splitlines() if l.startswith("| qwen_default, ")}
    assert f"| {fmt(min(correct.values()))} of 11 |" in rows["qwen_default, min over 3 seeds"]
    assert f"| {fmt(max(correct.values()))} of 11 |" in rows["qwen_default, max over 3 seeds"]
    assert f"| {fmt(sum(correct.values()) / 3)} of 11 |" in rows["qwen_default, mean over 3 seeds"]
    assert len(set(correct.values())) == 3                                                  # the three seeds really differ
    assert "greedy (base)" in test_part and "seed 2 (eval_base_qwen_s2_test.json)" in test_part
    assert "not yet measured" in dec_part.split("### Split: val")[1]                            # val has no runs here
    assert not (tmp_path / "qualitative_base_qwen_s1.md").exists()                             # sampled runs: no qualitative
    assert (tmp_path / "qualitative_base_p_sentence.md").exists()
    assert (tmp_path / "refusal_detection_review_base_qwen_s1_test.csv").exists()


def test_report_on_old_style_files_still_gets_key_columns(tmp_path):
    _write_run(tmp_path, _Fake(), "base")
    p = tmp_path / "eval_base_test.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d["items"]:
        r.pop("key_recall"), r.pop("key_support")
    d["summary"]["answerable"].pop("key_recall"), d["summary"]["answerable"].pop("key_support")
    for k in ("prompt_variant", "decoding_mode", "seed", "fewshot_ids", "system_prompt"):
        d.pop(k)
    p.write_text(json.dumps(d), encoding="utf-8")
    before = p.read_bytes()
    report.main(["--outputs", str(tmp_path)])
    assert p.read_bytes() == before                                                          # the file is not changed
    row = next(l for l in (tmp_path / "baseline_table.md").read_text(encoding="utf-8").splitlines()
               if l.startswith("| base Qwen2.5-0.5B-Instruct |") and "of 32" in l)
    assert "n/a" not in row.split("|")[3]                                                   # key recall column is filled


def test_is_sampled():
    assert report.is_sampled({"decoding_mode": "qwen_default"}) and not report.is_sampled({"decoding_mode": "greedy"})
    assert not report.is_sampled({})                                                         # old files


def test_report_config_lists_the_new_runs():
    names = [r["name"] for r in CFG["report"]["expected_runs"]]
    assert {"base", "base_p_sentence", "base_p_fewshot"} <= set(names)
    assert CFG["report"]["decoding_group"] == "base" and E["decoding_seeds"] == [1, 2, 3]


# ---- collector ------------------------------------------------------------------------------------------
def test_collect_takes_only_the_new_runs(tmp_path):
    new = ["eval_base_p_sentence_val.json", "eval_base_p_fewshot_test.json", "eval_base_qwen_s1_test.json",
           "refusal_detection_review_base_p_fewshot_test.csv", "refusal_detection_review_base_qwen_s3_val.csv"]
    old = ["eval_base_test.json", "eval_base_val.json", "eval_best_sentence_test.json", "eval_always_refuse_val.json",
           "refusal_detection_review_base_test.csv", "smoke_base_test.json", "baseline_table.md"]
    for n in new + old:
        (tmp_path / n).write_text("x", encoding="utf-8")
    files = collect.result_files(tmp_path, CFG)
    assert sorted(f.name for f in files) == sorted(new)
    z = collect.make_zip(files, tmp_path / "out" / "r.zip")
    assert sorted(zipfile.ZipFile(z).namelist()) == sorted(new)
    with pytest.raises(SystemExit):
        collect.main(["--outputs", str(tmp_path / "empty"), "--out", str(tmp_path / "e.zip")])


def test_every_planned_run_name_is_collected_and_default_runs_are_not():
    pats = CFG["collect"]["patterns"]
    wanted = []
    for variant in ("strict_sentence", "fewshot"):
        for split in ("val", "test"):
            n = P.run_name("base", variant, "greedy", None, CFG)
            wanted += [f"eval_{n}_{split}.json", f"refusal_detection_review_{n}_{split}.csv"]
    for seed in E["decoding_seeds"]:
        for split in ("val", "test"):
            n = P.run_name("base", "default", "qwen_default", seed, CFG)
            wanted += [f"eval_{n}_{split}.json", f"refusal_detection_review_{n}_{split}.csv"]
    assert len(wanted) == 20
    assert all(any(fnmatch.fnmatch(w, p) for p in pats) for w in wanted)
    for n in ("base", "best_sentence", "always_refuse"):
        for split in ("val", "test"):
            assert not any(fnmatch.fnmatch(f"eval_{n}_{split}.json", p) for p in pats)


# ---- refusal review sheet ------------------------------------------------------------------------------------
def _refusal_items():
    return [i for s in ("test", "val") for i in _items(s) if scoring.kind_of(i) == "refusal"]


def test_refusal_sheet_lists_all_22_items_with_context_only():
    before = {s: hashlib.sha256((REPO_ROOT / CFG["paths"]["splits"][s]).read_bytes()).hexdigest() for s in ("val", "test")}
    sheet = review_sheet.build_sheet(CFG)
    items = _refusal_items()
    assert len(items) == 22 and sheet.count("\nITEM ") == 22 and sheet.count("\nCHOICE: ") == 22
    for it in items:
        context, question, _ = parse_example(it)
        assert f"ID: {it['meta']['id']} " in sheet and f"QUESTION: {question}" in sheet and context.strip() in sheet
    assert sheet.count(REF) == 1                                                   # only inside the instructions
    for word in ("near_miss", "same_topic_missing", "random_other_source", "unanswerable_strategy", "ai_verdict",
                 "blind_flag", "spot_check"):
        assert not re.search(rf"\b{word}\b", sheet.split("\nITEM 1/")[1], re.I), word   # no labels next to items
    assert sheet == review_sheet.build_sheet(CFG)                                  # seeded, same every time
    after = {s: hashlib.sha256((REPO_ROOT / CFG["paths"]["splits"][s]).read_bytes()).hexdigest() for s in ("val", "test")}
    assert before == after                                                         # no data file touched


def test_refusal_sheet_reads_only_the_data_files_and_has_no_generated_text(monkeypatch):
    seen = []
    real_load = review_sheet.load_items
    monkeypatch.setattr(review_sheet, "load_items", lambda path, limit=None: (seen.append(Path(path)), real_load(path, limit))[1])
    sheet = review_sheet.build_sheet(CFG)
    assert {p.name for p in seen} == {"sft_test.jsonl", "sft_val.jsonl"}           # no eval output file is ever read
    # extra safeguard on saved base outputs: text a model generated (not copied from the contexts) is not in the sheet
    contexts = " ".join(parse_example(i)[0] for i in _refusal_items())
    for split in ("val", "test"):
        p = REPO_ROOT / "outputs" / f"eval_base_{split}.json"
        if p.exists():
            outs = [r["output"] for r in json.loads(p.read_text(encoding="utf-8"))["items"] if r["kind"] == "refusal"]
            assert all(o not in sheet for o in outs if len(o) > 40 and o not in contexts)


def test_refusal_sheet_script_does_not_overwrite_a_filled_sheet(tmp_path):
    p = tmp_path / "sheet.txt"
    p.write_text("ITEM 1/2\nCHOICE: \nNOTE: \nITEM 2/2\nCHOICE: \nNOTE: \n", encoding="utf-8")
    assert not make_refusal_review.filled(p)
    p.write_text("ITEM 1/2\nCHOICE: A\nNOTE: \nITEM 2/2\nCHOICE: \n", encoding="utf-8")
    assert make_refusal_review.filled(p)


# ---- bundle -----------------------------------------------------------------------------------------------------
@needs_training_data
def test_bundle_has_the_new_modules_and_the_train_file():
    from make_colab_bundle import bundle_files
    names = {p.relative_to(REPO_ROOT).as_posix() for p in bundle_files(CFG)}
    for must in ("climate_slm/evaluation/prompts.py", "climate_slm/evaluation/collect.py", "climate_slm/evaluation/scoring.py",
                 "configs/phase2.yaml", "data/instruction/sft_train.jsonl"):
        assert must in names, must
    assert not any(n.startswith(("docs/", "outputs/")) for n in names)


# ---- bare-answer table ------------------------------------------------------------------------------------------
def test_bare_answer_table_splits_by_output_length():
    recs = [{"kind": "answerable", "output_chars": 5, "token_f1": 0.1, "key_recall": 0.2, "key_support": 1.0},
            {"kind": "answerable", "output_chars": 10, "token_f1": 0.3, "key_recall": None, "key_support": 0.5},
            {"kind": "answerable", "output_chars": 80, "token_f1": 0.9, "key_recall": 0.8, "key_support": 0.7},
            {"kind": "refusal", "output_chars": 3, "token_f1": 0.0, "key_recall": None, "key_support": None}]
    t = scoring.bare_answer_table(recs, 40)
    assert t["bare"]["n"] == 2 and t["longer"]["n"] == 1                       # refusal items are left out
    assert t["bare"]["token_f1"] == pytest.approx(0.2) and t["bare"]["key_recall"] == pytest.approx(0.2)
    assert t["bare"]["key_support"] == pytest.approx(0.75) and t["longer"]["key_recall"] == pytest.approx(0.8)
    assert scoring.bare_answer_table([], 40)["bare"] == {"n": 0, "token_f1": None, "key_recall": None, "key_support": None}
    assert CFG["report"]["bare_max_chars"] == 40
