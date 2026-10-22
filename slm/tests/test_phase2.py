"""Tests for Phase 2 (evaluator, baselines, report). CPU only, no torch: the model is a fake stub."""
import csv
import json
import sys
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.evaluation import metrics as M
from climate_slm.evaluation import report, scoring
from climate_slm.evaluation.baselines import AlwaysRefuse, BestSentence, split_user_message
from climate_slm.evaluation.evaluate import run_eval
from _release import needs_training_data

CFG = load_eval_config()
E = CFG["eval"]
REF = CFG["refusal_text"]
PHRASES = E["loose_refusal_phrases"]


class FakeModel:
    """Stands in for the real model: answers '37 degrees' to everything, refuses every third item."""

    def __init__(self):
        self.calls = 0

    def info(self):
        return {"model_id": "fake/model", "revision": "abc123", "adapter": None, "dtype": "float16", "device": "none"}

    def generate(self, prompts):
        out = []
        for i, _ in enumerate(prompts):
            text = "Sorry, the context does not contain that." if i % 3 == 0 else "It is 37 degrees and 999 units."
            out.append({"text": text, "n_new_tokens": 9, "hit_max_new_tokens": i == 1})
        self.calls += 1
        return out

    def completion_nll(self, prompts, completions):
        return [(2.0 * len(c.split()), len(c.split())) for c in completions]


# ---- metrics -----------------------------------------------------------------
def test_normalisation_known_cases():
    assert M.normalize("The Heat-Wave, is at 4.5 °C!") == "heat wave is at 4.5 c"
    assert M.normalize("An answer:  the  A  an end.") == "answer end"
    assert M.normalize("I can’t answer; 1,000 farmers") == "i cant answer 1000 farmers"
    assert M.normalize("") == "" and M.normalize(None) == ""


def test_token_f1_known_cases():
    assert M.token_f1("The cat sat.", "cat sat") == 1.0
    assert M.token_f1("dog", "cat") == 0.0
    assert M.token_f1("", "") == 1.0 and M.token_f1("", "cat") == 0.0
    assert abs(M.token_f1("one two three four", "one two five six") - 0.5) < 1e-9
    assert abs(M.token_f1("cat cat dog", "cat") - 0.5) < 1e-9          # precision 1/3, recall 1


def test_rouge_l_known_cases():
    assert M.rouge_l("The cat sat on the mat.", "cat sat on mat") == 1.0
    assert M.rouge_l("dog", "cat") == 0.0 and M.rouge_l("", "cat") == 0.0
    # LCS of "one two three four" and "one three two four" is 3 -> P = R = F = 0.75
    assert abs(M.rouge_l("one two three four", "one three two four") - 0.75) < 1e-9


def test_number_faithfulness():
    ctx = "Heat Wave when departure is 4.50C or more, provided actual maximum temperature is 370C or more. About 1,200 farmers."
    assert M.number_faithfulness("Departure of 4.5 °C and at least 37°C.", ctx) == 1.0     # lost degree sign
    assert M.number_faithfulness("Departure of 4.50C.", ctx) == 1.0                         # copied as written
    assert M.number_faithfulness("1200 farmers", ctx) == 1.0 and M.number_faithfulness("1,200 farmers", ctx) == 1.0
    assert M.number_faithfulness("No numbers here.", ctx) is None
    assert M.number_faithfulness("It is 45 degrees.", ctx) == 0.0                           # invented number
    assert M.number_faithfulness("4.5 and 99", ctx) == 0.5
    assert M.context_numbers("remains 40 0C or more") >= {"40"}
    assert M.numbers("CO2 rose 3.0 times in 2037") == ["3", "2037"]                         # digit inside a word is skipped


def test_refusal_detection_strict_and_loose():
    assert M.is_refusal_strict(REF, REF)
    assert M.is_refusal_strict("  i CANNOT answer this   from the provided context ", REF)
    assert not M.is_refusal_strict("I cannot answer this.", REF)
    assert not M.is_refusal_strict(REF + " But maybe 40 degrees.", REF)
    for text in (REF, "I cannot answer this.", "Sorry, I can't answer that.", "The context does not contain the answer.",
                 "There is no information about livestock."):
        assert M.is_refusal_loose(text, REF, PHRASES), text
    for text in ("Heat wave is declared at 40 degrees.", "Farmers cannot sow early.", ""):
        assert not M.is_refusal_loose(text, REF, PHRASES), text
    assert not M.is_refusal_loose("the weather can notanswer", REF, ["not answer"])         # whole words only


def test_bootstrap_ci_seeded_and_contains_mean():
    vals = [0.1, 0.4, 0.5, 0.9, 1.0, 0.3, 0.7, 0.2, 0.6, 0.8]
    a = M.bootstrap_ci(vals, 500, 7)
    assert a == M.bootstrap_ci(vals, 500, 7)
    assert a != M.bootstrap_ci(vals, 500, 8)
    assert a[0] <= M.mean(vals) <= a[1] and a[0] < a[1]
    lo, hi = M.bootstrap_ci([1, 0, 0, 1, 1, 0, 1, 1], 500, 7)
    assert 0 <= lo <= 5 / 8 <= hi <= 1
    assert M.bootstrap_ci([], 100, 1) is None
    assert M.bootstrap_ci([1, 1, 1], 100, 1) == (1.0, 1.0)             # zero width, hence the Wilson interval
    w = M.wilson_ci(11, 11)
    assert w[1] == 1.0 and 0.70 < w[0] < 0.75
    assert M.wilson_ci(0, 0) is None


# ---- baselines ---------------------------------------------------------------
REQUIRED_KEYS = {"run_name", "split", "limit", "model_id", "model_revision", "dtype", "decoding", "data_file",
                 "data_sha256", "library_versions", "timestamp_utc", "items", "summary"}
ITEM_KEYS = {"id", "kind", "prompt_chars", "output", "reference", "refusal_strict", "refusal_loose", "token_f1",
             "rouge_l", "number_faithfulness"}


def test_always_refuse_on_test(tmp_path):
    out = tmp_path / "eval_always_refuse_test.json"
    run_eval(AlwaysRefuse(CFG), CFG, "test", "always_refuse", out_path=out)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert REQUIRED_KEYS <= set(d) and ITEM_KEYS <= set(d["items"][0])
    s = d["summary"]
    assert (s["n_items"], s["answerable"]["n"], s["refusal"]["n"]) == (43, 32, 11)
    assert s["refusal"]["correct_refusal_strict"]["count"] == 11        # all refusals right
    assert s["answerable"]["false_refusal_strict"]["count"] == 32       # all answerable wrong
    assert s["refusal"]["answered_when_should_refuse"]["count"] == 0
    assert s["perplexity"] == {"overall": None, "answerable": None, "refusal": None}
    assert d["data_sha256"] == scoring.file_sha256(REPO_ROOT / CFG["paths"]["splits"]["test"])


def test_best_sentence_on_test(tmp_path):
    out = tmp_path / "eval_best_sentence_test.json"
    d = run_eval(BestSentence(CFG), CFG, "test", "best_sentence", out_path=out)
    assert REQUIRED_KEYS <= set(json.loads(out.read_text(encoding="utf-8")))
    s = d["summary"]
    assert s["answerable"]["false_refusal_strict"]["count"] == 0 and s["refusal"]["correct_refusal_strict"]["count"] == 0
    assert s["answerable"]["number_faithfulness"]["mean"] == 1.0        # it only copies the context
    assert all(r["output"] for r in d["items"])
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "Context:\nRice needs water. Heat wave starts at 40 degrees in plains.\n\n"
                                        "Question: When does a heat wave start in plains?"}]
    assert split_user_message(msgs)[1] == "When does a heat wave start in plains?"
    assert BestSentence(CFG).answer(msgs) == "Heat wave starts at 40 degrees in plains."


# ---- evaluator with a fake model ----------------------------------------------
def test_evaluator_with_fake_model(tmp_path):
    out = tmp_path / "eval_fake_test.json"
    d = run_eval(FakeModel(), CFG, "test", "fake", out_path=out)
    assert json.loads(out.read_text(encoding="utf-8"))["summary"] == d["summary"]
    assert REQUIRED_KEYS <= set(d) and ITEM_KEYS <= set(d["items"][0])
    assert (d["model_id"], d["model_revision"], d["dtype"]) == ("fake/model", "abc123", "float16")
    assert d["decoding"] == {"do_sample": False, "max_new_tokens": E["max_new_tokens"],
                             "repetition_penalty": E["repetition_penalty"], "padding_side": "left",
                             "batch_size": E["batch_size"]}
    assert {"python", "transformers", "torch", "rouge-score"} <= set(d["library_versions"])
    s = d["summary"]
    assert set(s["answerable"]) == {"n", "token_f1", "rouge_l", "number_faithfulness", "false_refusal_strict",
                                    "false_refusal_loose", "key_recall", "key_support"}
    assert set(s["refusal"]) == {"n", "correct_refusal_strict", "correct_refusal_loose", "answered_when_should_refuse"}
    ans = [r for r in d["items"] if r["kind"] == "answerable"]
    ref = [r for r in d["items"] if r["kind"] == "refusal"]
    assert (len(ans), len(ref)) == (32, 11)
    assert s["answerable"]["false_refusal_loose"]["count"] == sum(r["refusal_loose"] for r in ans)
    assert s["answerable"]["false_refusal_strict"]["count"] == 0        # the fake never says the exact text
    rf = s["refusal"]
    assert rf["correct_refusal_loose"]["count"] + rf["answered_when_should_refuse"]["count"] == 11
    nf = s["answerable"]["number_faithfulness"]
    assert nf["n_applied"] == sum(r["number_faithfulness"] is not None for r in ans) and nf["n_answerable"] == 32
    assert nf["mean"] < 1.0                                             # 999 is never in the context
    for k in ("overall", "answerable", "refusal"):                      # NLL is 2.0 per token in the fake
        assert abs(s["perplexity"][k]["perplexity"] - 7.389056) < 1e-4
    assert s["perplexity"]["overall"]["n"] == 43
    assert s["hit_max_new_tokens"] == {"count": 1, "n": 43}
    assert s["output_length_chars"]["min"] > 0
    for b in (s["answerable"]["token_f1"], rf["correct_refusal_loose"]):
        assert b["ci95"][0] <= (b.get("mean", b.get("rate"))) <= b["ci95"][1]
    review = tmp_path / "refusal_detection_review_fake_test.csv"
    with open(review, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(r["refusal_loose"] and not r["refusal_strict"] for r in d["items"]) \
        + rf["answered_when_should_refuse"]["count"]
    assert {r["why_listed"] for r in rows} == {"loose_refusal_but_not_strict", "refusal_item_was_answered"}


def test_evaluator_limit(tmp_path):
    out = tmp_path / "smoke.json"
    d = run_eval(FakeModel(), CFG, "val", "fake", limit=3, out_path=out)
    assert d["limit"] == 3 and len(d["items"]) == 3 and d["summary"]["n_items"] == 3
    assert [r["id"] for r in d["items"]] == [i["meta"]["id"] for i in scoring.load_items(
        REPO_ROOT / CFG["paths"]["splits"]["val"])[:3]]
    assert not list(tmp_path.glob("refusal_detection_review_*"))        # no review file for a smoke run


def test_evaluate_module_imports_without_torch():
    import climate_slm.evaluation.evaluate  # noqa: F401
    assert "torch" not in sys.modules and "transformers" not in sys.modules


# ---- report --------------------------------------------------------------------
def test_report_builds_table_from_two_fake_eval_files(tmp_path):
    run_eval(FakeModel(), CFG, "test", "base", out_path=tmp_path / "eval_base_test.json")
    run_eval(AlwaysRefuse(CFG), CFG, "test", "always_refuse", out_path=tmp_path / "eval_always_refuse_test.json")
    run_eval(FakeModel(), CFG, "test", "base", limit=2, out_path=tmp_path / "eval_base_test_smoke.json")
    report.main(["--outputs", str(tmp_path)])
    table = (tmp_path / "baseline_table.md").read_text(encoding="utf-8")
    assert "## Split: test" in table and "### Answerable items (32)" in table and "### Refusal items (11)" in table
    rows = {line.split("|")[1].strip(): line for line in table.split("## Split: val")[0].splitlines()
            if line.startswith("| ") and "Wilson" in line and "=" in line and "/11" in line}
    assert "11/11 = 1.000" in rows["always_refuse (no model)"]
    assert "best_sentence (no model) | not yet measured" in table       # expected run without a file
    assert "base Qwen2.5-0.5B-Instruct | not yet measured" in table.split("## Split: val")[1]
    assert "7.389" in table and "fake/model" in table and "eval_base_test_smoke.json (smoke run, limit=2)" in table
    q = (tmp_path / "qualitative_base.md").read_text(encoding="utf-8")
    assert all(f"## {i} " in q for i in E["qualitative_ids"]) and "**Model output:**" in q
    assert (tmp_path / "refusal_detection_review_base_test.csv").exists()


# ---- config, qualitative ids, bundle ---------------------------------------------
def test_config_and_qualitative_ids():
    assert E["dtype"] == "float16" and E["do_sample"] is False and E["max_new_tokens"] == 256
    assert E["padding_side"] == "left" and REF == "I cannot answer this from the provided context."
    items = scoring.load_items(REPO_ROOT / CFG["paths"]["splits"]["test"])
    kinds = {i["meta"]["id"]: scoring.kind_of(i) for i in items}
    ids = E["qualitative_ids"]
    assert len(ids) == len(set(ids)) == 10 and all(i in kinds for i in ids)
    assert sum(kinds[i] == "answerable" for i in ids) == 6 and sum(kinds[i] == "refusal" for i in ids) == 4
    assert ids == report.pick_qualitative_ids(items, E["qualitative_seed"], E["qualitative_n_answerable"],
                                              E["qualitative_n_refusal"])


def test_requirements_keep_gpu_packages_off_the_laptop():
    cpu = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    gpu = (REPO_ROOT / "requirements-gpu.txt").read_text(encoding="utf-8").lower()
    gpu_pkgs = [line.split(">")[0].split("=")[0].strip() for line in gpu.splitlines() if line and not line.startswith("#")]
    assert not any(p in cpu for p in ("torch", "transformers", "peft", "bitsandbytes"))
    assert "rouge-score" in cpu
    assert {"transformers", "accelerate", "datasets", "rouge-score", "huggingface_hub"} <= set(gpu_pkgs)
    assert not {"torch", "trl"} & set(gpu_pkgs)                                  # torch stays the Colab one
    assert {"peft", "bitsandbytes", "mlflow"} <= set(gpu_pkgs)                    # added in Phase 3


@needs_training_data
def test_colab_bundle_contents(tmp_path):
    from make_colab_bundle import make_bundle
    path, names = make_bundle(CFG, tmp_path / "bundle.zip")
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == sorted(names)
    for must in ("climate_slm/evaluation/evaluate.py", "climate_slm/evaluation/metrics.py", "climate_slm/config.py",
                 "climate_slm/data/sft.py", "configs/phase1.yaml", "configs/phase2.yaml", "requirements-gpu.txt",
                 "data/instruction/sft_train.jsonl", "data/instruction/sft_val.jsonl", "data/instruction/sft_test.jsonl"):
        assert must in names, must
    assert not any("__pycache__" in n or n.startswith(("docs/", "outputs/", "tests/")) for n in names)
    assert not any("spot_check" in n or "candidates" in n for n in names)


def test_notebook_is_valid_and_thin():
    nb = json.loads((REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # one-line comment on top of each cell (the Phase 5 cells have `%cd /content/climate_slm_project` as their very first line)
    first = lambda c: next(x for x in c.splitlines() if not x.startswith("%cd "))                # noqa: E731
    assert len(code) >= 8 and all(first(c).lstrip().startswith("#") for c in code)
    code = code[:8]                                                            # the evaluation part (Phase 3 cells: test_phase3)
    assert "torch.cuda.is_available()" in code[0] and "colab_bundle.zip" in code[1]
    assert "requirements-gpu.txt" in code[2] and "--limit 3" in code[3]
    assert "--split val" in code[4] and "--split test" in code[4]              # default base runs, untouched
    assert "--prompt-variant" not in code[4] and "--decoding" not in code[4]
    assert code[5].count("--prompt-variant") == 4 and "strict_sentence" in code[5] and "fewshot" in code[5]
    assert code[6].count("--decoding qwen_default --seed") == 6 and all(f"--seed {k}" in code[6] for k in (1, 2, 3))
    assert "climate_slm.evaluation.collect" in code[7] and "files.download" in code[7]
    assert all(len(c.splitlines()) <= 15 for c in code)
