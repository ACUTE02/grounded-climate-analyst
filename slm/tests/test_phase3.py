"""Tests for Phase 3 (training code, selection rule, test gating). CPU only. No torch, transformers, peft, mlflow."""
import copy
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from climate_slm import chat
from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config
from climate_slm.evaluation import collect, evaluate, report, scoring
from climate_slm.evaluation import metrics as M
from climate_slm.evaluation.evaluate import default_out
from climate_slm.training import (eval_adapters, final_test, grid, masking, mlflow_log, prep, schedule, select_best,
                                  selection_io, summary, train)
from _release import needs_training_data

CFG3 = load_phase3_config()
ECFG = load_eval_config()
REF = ECFG["refusal_text"]


def _rows(split):
    return scoring.load_items(REPO_ROOT / ECFG["paths"]["splits"][split])


# ---- config and grid ---------------------------------------------------------------------------------------------------
def test_config_matches_the_plan_written_before_the_runs():
    s, lo, t, c, sel = CFG3["sweep"], CFG3["lora"], CFG3["train"], CFG3["criteria"], CFG3["selection"]
    assert s["methods"] == ["lora_fp16", "qlora_4bit"] and s["learning_rates"] == [1e-4, 2e-4]
    assert s["seed"] == 42 and s["repeat_seeds"] == [43, 44]
    assert (lo["r"], lo["alpha"], lo["dropout"]) == (16, 32, 0.05)
    assert set(lo["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert (t["epochs"], t["effective_batch_size"], t["warmup_ratio"], t["lr_schedule"]) == (3, 16, 0.1, "cosine")
    assert (t["weight_decay"], t["max_grad_norm"], t["refusal_repeat"], t["dtype"]) == (0.0, 1.0, 1, "float16")
    q = CFG3["methods"]["qlora_4bit"]
    assert (q["quantize"], q["double_quant"], q["compute_dtype"], q["prepare_kbit"]) == ("nf4", True, "float16", True)
    assert CFG3["methods"]["lora_fp16"]["quantize"] is None
    assert CFG3["arms"]["main"]["model_id"] == ECFG["eval"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert CFG3["arms"]["q15"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct" and CFG3["arms"]["q15"]["gradient_checkpointing"]
    assert (c["min_correct_refusals"], c["max_false_refusals"], c["f1_target_test"]) == (7, {"val": 4, "test": 3}, 0.60)
    assert (c["min_number_faithfulness"], c["max_hit_max_new_tokens"]) == (0.95, 1)
    assert sel["max_false_refusals_val"] == 4
    assert sel["order"][:3] == ["correct_refusals_strict", "token_f1", "number_faithfulness"]
    assert sel["order"][3:] == ["peak_gpu_mem", "trainable_params", "train_seconds", "epoch"]
    assert CFG3["tables"]["prompted_baselines"] == ["base_p_sentence", "base_p_fewshot"]


def test_grid_has_four_runs_and_the_expected_names():
    runs = grid.grid_runs(CFG3)
    assert [r["name"] for r in runs] == ["lora_fp16_lr1e-4", "lora_fp16_lr2e-4", "qlora_4bit_lr1e-4", "qlora_4bit_lr2e-4"]
    assert all(r["seed"] == 42 and r["arm"] == "main" and not r["gradient_checkpointing"] for r in runs)
    assert grid.lr_tag(1e-4) == "1e-4" and grid.lr_tag(2e-4) == "2e-4"
    assert [grid.candidate_name("lora_fp16_lr1e-4", e) for e in grid.epochs(CFG3)] == [
        "lora_fp16_lr1e-4_e1", "lora_fp16_lr1e-4_e2", "lora_fp16_lr1e-4_e3"]
    assert grid.validate(CFG3) == []


def test_repeat_and_other_size_runs():
    w = grid.grid_runs(CFG3)[1]
    assert [r["name"] for r in grid.repeat_runs(CFG3, w)] == ["lora_fp16_lr2e-4_s43", "lora_fp16_lr2e-4_s44"]
    assert [r["seed"] for r in grid.repeat_runs(CFG3, w)] == [43, 44]
    q = grid.q15_run(CFG3, grid.grid_runs(CFG3)[3])
    assert q["name"] == "q15_qlora_4bit" and q["lr"] == 2e-4 and q["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert q["gradient_checkpointing"] and q["seed"] == 42 and q["method"] == "qlora_4bit"


def test_validate_catches_broken_configs():
    def broken(path, value):
        c = copy.deepcopy(CFG3)
        d = c
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        return grid.validate(c)

    assert broken(["sweep", "methods"], ["lora_fp16"])                                  # only 2 runs
    assert broken(["train", "micro_batch_size"], 3)                                      # 16 is not a multiple of 3
    assert broken(["sweep", "repeat_seeds"], [42, 44])                                   # repeat seed equals sweep seed
    assert broken(["sweep", "repeat_seeds"], [43])
    assert broken(["train", "epochs"], 0)
    assert broken(["selection", "order"], ["token_f1", "correct_refusals_strict", "number_faithfulness"])


# ---- masking, preparation --------------------------------------------------------------------------------------------------
def test_build_example_masks_the_prompt_and_never_cuts():
    ex = masking.build_example([1, 2, 3], [7, 8], max_len=10)
    assert ex["input_ids"] == [1, 2, 3, 7, 8] and ex["labels"] == [-100, -100, -100, 7, 8]
    assert (ex["n_prompt"], ex["n_completion"]) == (3, 2)
    assert masking.build_example([1, 2, 3], [7, 8], max_len=5) is not None             # exactly fits
    assert masking.build_example([1, 2, 3], [7, 8], max_len=4) is None                 # one token too long: dropped, not cut
    assert masking.build_example([1, 2, 3], [], max_len=10) is None                    # nothing to learn from
    assert masking.loss_token_count(ex["labels"]) == 2 and masking.loss_token_count([5, -100, -100]) == 0


def test_pad_batch_pads_right_and_masks_padding():
    a = masking.build_example([1, 2], [9], 10)
    b = masking.build_example([1, 2, 3, 4], [9, 9], 10)
    out = masking.pad_batch([a, b], pad_id=0)
    assert out["input_ids"] == [[1, 2, 9, 0, 0, 0], [1, 2, 3, 4, 9, 9]]
    assert out["attention_mask"] == [[1, 1, 1, 0, 0, 0], [1] * 6]
    assert out["labels"] == [[-100, -100, 9, -100, -100, -100], [-100, -100, -100, -100, 9, 9]]


def test_length_stats():
    s = masking.length_stats([10, 20, 30, 40, 50, 3000], 2048)
    assert s == {"n": 6, "min": 10, "median": 35.0, "p95": 3000, "max": 3000, "over_limit": 1}
    assert masking.length_stats([], 10)["n"] == 0


def _fake_template(messages, add_generation_prompt):
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _fake_tokenize(text):
    return [ord(ch) for ch in text]                 # one id per character: easy to read back


def test_encode_rows_with_a_fake_tokenizer():
    rows = _rows("val")[:6]
    out = prep.encode_rows(rows, _fake_template, _fake_tokenize, max_len=100000)
    assert out["dropped_ids"] == [] and len(out["examples"]) == 6 and out["stats"]["n"] == 6
    for row, ex in zip(rows, out["examples"]):
        n = ex["n_prompt"]
        prompt_part = "".join(map(chr, ex["input_ids"][:n]))
        completion_part = "".join(map(chr, ex["input_ids"][n:]))
        assert prompt_part == _fake_template(row["prompt"], True) and prompt_part.endswith("<|im_start|>assistant\n")
        assert completion_part == row["completion"][0]["content"] + "<|im_end|>"      # answer + end-of-turn token
        assert set(ex["labels"][:n]) == {-100} and ex["labels"][n:] == ex["input_ids"][n:]
        assert ex["id"] == row["meta"]["id"]
    small = prep.encode_rows(rows, _fake_template, _fake_tokenize, max_len=500)
    assert small["dropped_ids"] and len(small["examples"]) + len(small["dropped_ids"]) == 6   # dropped, never truncated
    assert all(len(e["input_ids"]) <= 500 for e in small["examples"])
    assert small["stats"]["over_limit"] == len(small["dropped_ids"])


def test_training_text_is_the_text_the_evaluator_builds():
    class Tok:
        eos_token = "<|im_end|>"

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return _fake_template(messages, add_generation_prompt)

    b = object.__new__(evaluate.HFBackend)
    b.tok = Tok()
    for row in _rows("test")[:5]:
        msgs, answer = row["prompt"], row["completion"][0]["content"]
        p_eval = b._prompt_text(msgs)
        c_eval = b._completion_text(msgs, answer, p_eval)
        p_text = chat.prompt_text(_fake_template, msgs)
        c_text = chat.completion_text(_fake_template, msgs, answer, p_text, "<|im_end|>")
        assert (p_eval, c_eval) == (p_text, c_text) and c_text == answer + "<|im_end|>"
    # a template that does not start with the prompt text falls back to answer + eos token
    assert chat.completion_text(lambda m, g: "XYZ", [{"role": "user", "content": "q"}], "a", "PROMPT", "<eos>") == "a<eos>"


# ---- schedule, batching ------------------------------------------------------------------------------------------------------
@needs_training_data
def test_steps_and_learning_rate_schedule():
    n_train = len(_rows("train"))
    spe = schedule.steps_per_epoch(n_train, 16)
    assert (n_train, spe, spe * 3) == (156, 10, 30)
    assert schedule.warmup_steps(30, 0.1) == 3
    lrs = [schedule.lr_at_step(i, 30, 2e-4, 0.1) for i in range(30)]
    assert lrs[0] == pytest.approx(2e-4 / 3) and lrs[2] == pytest.approx(2e-4)          # linear warm-up to the full rate
    assert max(lrs) == pytest.approx(2e-4) and lrs[2] >= lrs[3] > lrs[10] > lrs[20] > lrs[29] > 0   # then cosine down
    assert lrs[29] < 0.01 * 2e-4
    assert schedule.lr_at_step(15, 30, 1.0, 0.1) == pytest.approx(0.5 * (1 + __import__("math").cos(__import__("math").pi * 12 / 27)))


def test_epoch_order_is_pinned_by_the_seed():
    a, b = schedule.epoch_order(156, 42, 1), schedule.epoch_order(156, 42, 1)
    assert a == b and sorted(a) == list(range(156))
    assert a != schedule.epoch_order(156, 42, 2) and a != schedule.epoch_order(156, 43, 1)
    assert a != list(range(156))


def test_optimizer_steps_cover_every_row_once():
    order = schedule.epoch_order(156, 42, 1)
    steps = schedule.optimizer_steps(order, 16, 2)
    assert len(steps) == 10 and [sum(len(mb) for mb in s) for s in steps] == [16] * 9 + [12]
    assert all(len(mb) <= 2 for s in steps for mb in s) and len(steps[0]) == 8            # 16 / 2 = 8 accumulation steps
    assert sorted(i for s in steps for mb in s for i in mb) == list(range(156))


@needs_training_data
def test_refusal_repeat_option():
    rows = _rows("train")
    assert schedule.repeat_refusals(rows, 1) == rows                                     # default: unchanged
    twice = schedule.repeat_refusals(rows, 2)
    n_ref = sum(r["meta"]["type"] == "unanswerable" for r in rows)
    assert (len(rows), n_ref, len(twice)) == (156, 21, 177)
    assert sum(r["meta"]["type"] == "unanswerable" for r in twice) == 42
    with pytest.raises(ValueError):
        schedule.repeat_refusals(rows, 0)


# ---- summary helpers -----------------------------------------------------------------------------------------------------------
def test_repetition_loop_detection():
    assert summary.has_repetition_loop("the rate is 5 percent " * 4, 4, 4)
    assert not summary.has_repetition_loop("the rate is 5 percent " * 3, 4, 4)
    assert not summary.has_repetition_loop("Heat wave is declared when the maximum temperature reaches 40 degrees.", 4, 4)
    assert not summary.has_repetition_loop("short", 4, 4)


def test_compare_directions_and_overlap():
    assert summary.compare(0.7, [0.6, 0.8], 0.5, [0.4, 0.55]) == {"direction": "higher", "overlap": False}
    assert summary.compare(0.7, [0.6, 0.8], 0.65, [0.55, 0.75]) == {"direction": "higher", "overlap": True}
    assert summary.compare(0.4, [0.3, 0.5], 0.7, [0.6, 0.8]) == {"direction": "lower", "overlap": False}
    assert summary.compare(0.5, [0.4, 0.6], 0.5, [0.1, 0.9])["direction"] == "equal"
    assert summary.compare(None, None, 0.5, [0.4, 0.6]) == {"direction": None, "overlap": None}


# ---- fake eval files ---------------------------------------------------------------------------------------------------------------
class _Scripted:
    """Gives `fr` false refusals on answerable items, `cr` correct refusals, and answers the rest with the
    reference (quality 1.0) or its first half (quality 0.5)."""

    def __init__(self, items, fr, cr, quality=1.0):
        self.items, self.fr, self.cr, self.quality = items, fr, cr, quality

    def info(self):
        return {"model_id": "fake", "revision": None, "adapter": None, "dtype": "float16", "device": "none"}

    def generate(self, prompts):
        out, a, r = [], 0, 0
        for it in self.items:
            ref = it["completion"][0]["content"]
            if scoring.kind_of(it) == "answerable":
                toks = ref.split()
                text = REF if a < self.fr else " ".join(toks[:max(1, int(len(toks) * self.quality))])
                a += 1
            else:
                text = REF if r < self.cr else "It is 37 degrees."
                r += 1
            out.append({"text": text, "n_new_tokens": 5, "hit_max_new_tokens": False})
        return out

    def completion_nll(self, prompts, completions):
        return [(float(len(c.split())), len(c.split())) for c in completions]


def _world(tmp_path):
    """Config copies that point at a temp folder, with a few cheap bootstrap resamples."""
    c3, ec = copy.deepcopy(CFG3), copy.deepcopy(ECFG)
    for k in c3["paths"]:
        c3["paths"][k] = str(tmp_path / c3["paths"][k])
    ec["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    ec["eval"]["n_boot"] = 40
    return c3, ec


def _write_eval(ec, name, split, fr, cr, quality=1.0):
    items = _rows(split)
    evaluate.run_eval(_Scripted(items, fr, cr, quality), ec, split, name, out_path=default_out(ec, name, split),
                      write_review=False)


def _write_log(c3, run, peak, params=1000, secs=100.0, complete=True, vloss=(1.0, 0.8, 0.7)):
    p = grid.run_log_path(c3, run)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "run_name": run, "complete": complete, "peak_gpu_mem_mb": peak, "train_seconds": secs, "total_seconds": secs + 30,
        "trainable_params": params, "total_params": params * 50, "trainable_share": 0.02, "model_id": "m",
        "params": {"gradient_checkpointing": False}, "dropped_train_ids": [], "dropped_val_ids": [],
        "epochs": [{"epoch": i + 1, "val_loss": v, "adapter_mb": 35.0} for i, v in enumerate(vloss)]}), encoding="utf-8")


# (false refusals on val, correct refusals on val, quality) per run and epoch
GRID_SCRIPT = {
    "lora_fp16_lr1e-4": [(0, 2, 1.0), (0, 5, 1.0), (1, 6, 1.0)],
    "lora_fp16_lr2e-4": [(0, 6, 1.0), (2, 8, 1.0), (5, 9, 1.0)],
    "qlora_4bit_lr1e-4": [(0, 1, 1.0), (0, 4, 1.0), (0, 6, 1.0)],
    "qlora_4bit_lr2e-4": [(0, 7, 1.0), (1, 8, 1.0), (3, 8, 1.0)],
}
PEAK = {"lora_fp16": 5000.0, "qlora_4bit": 3500.0}


def _fill_world(tmp_path, with_repeats=True, with_q15=True):
    c3, ec = _world(tmp_path)
    for run in grid.grid_runs(c3):
        _write_log(c3, run["name"], PEAK[run["method"]])
        for e, (fr, cr, q) in enumerate(GRID_SCRIPT[run["name"]], 1):
            _write_eval(ec, grid.candidate_name(run["name"], e), "val", fr, cr, q)
    for name, (fr, cr, q) in {"base": (0, 0, 0.5), "base_p_sentence": (0, 0, 0.7), "base_p_fewshot": (0, 1, 0.8),
                              "base15": (0, 0, 0.6)}.items():
        for split in ("val", "test"):
            _write_eval(ec, name, split, fr, cr, q)
    if with_repeats:
        for seed, (fr, cr) in {43: (1, 7), 44: (2, 9)}.items():
            run = f"qlora_4bit_lr2e-4_s{seed}"
            _write_log(c3, run, 3500.0)
            _write_eval(ec, f"{run}_e2", "val", fr, cr)
    if with_q15:
        _write_log(c3, "q15_qlora_4bit", 9000.0, params=4000, secs=400.0)
        for e, (fr, cr) in enumerate([(0, 6), (1, 9), (2, 9)], 1):
            _write_eval(ec, f"q15_qlora_4bit_e{e}", "val", fr, cr)
    return c3, ec


# ---- extraction ----------------------------------------------------------------------------------------------------------------------
def test_extract_metrics_and_flat_metrics(tmp_path):
    c3, ec = _world(tmp_path)
    _write_eval(ec, "x", "val", fr=2, cr=7, quality=1.0)
    m = select_best.load_metrics(ec, c3, "x", "val")
    assert (m["n_answerable"], m["n_refusal"]) == (42, 11)
    assert m["false_refusals"]["count"] == 2 and m["correct_strict"]["count"] == 7 and m["correct_loose"]["count"] == 7
    assert m["answered_when_should_refuse"]["count"] == 4 and m["hit_max_new_tokens"] == 0 and m["repetition_loops"] == 0
    assert m["token_f1"]["mean"] == pytest.approx(40 / 42, abs=0.02) and m["key_recall"] and m["key_support"]
    assert m["correct_strict"]["wilson95"][0] < 7 / 11 < m["correct_strict"]["wilson95"][1]
    flat = summary.flat_metrics(m, "val_")
    assert flat["val_correct_refusals_strict"] == 7.0 and flat["val_false_refusals_strict"] == 2.0
    assert all(k.startswith("val_") and isinstance(v, float) for k, v in flat.items())
    assert select_best.load_metrics(ec, c3, "missing", "val") is None


# ---- the selection rule ----------------------------------------------------------------------------------------------------------------
def _cand(name, epoch=1, correct=7, fr=0, f1=0.7, nf=1.0, peak=1000.0, params=100, secs=10.0, run=None):
    def rate(k, n):
        return {"count": k, "n": n, "rate": k / n, "wilson95": list(M.wilson_ci(k, n))}

    metrics = {"n_answerable": 42, "n_refusal": 11, "token_f1": {"mean": f1, "ci95": [f1 - 0.05, f1 + 0.05], "n": 42},
               "key_recall": None, "key_support": None,
               "number_faithfulness": {"mean": nf, "ci95": [nf, nf] if nf is not None else None, "n_applied": 14},
               "false_refusals": rate(fr, 42), "correct_strict": rate(correct, 11), "correct_loose": rate(correct, 11),
               "answered_when_should_refuse": rate(11 - correct, 11), "refusal_perplexity": 2.0, "hit_max_new_tokens": 0,
               "repetition_loops": 0, "output_chars": {"median": 100, "max": 200}}
    return {"candidate": name, "run": run or name, "epoch": epoch, "method": "lora_fp16", "lr": 2e-4, "seed": 42, "arm": "main",
            "metrics": metrics, "cost": {"peak_gpu_mem_mb": peak, "trainable_params": params, "train_seconds": secs}}


def test_selection_filters_false_refusals_then_ranks():
    cands = [_cand("a", correct=9, fr=5), _cand("b", correct=8, fr=4), _cand("c", correct=7, fr=0), _cand("d", correct=8, fr=3)]
    r = select_best.select_winner(cands, CFG3)
    assert [e["candidate"] for e in r["excluded"]] == ["a"]                       # 5 of 42 > 4 even with 9 correct refusals
    assert set(r["ranked"][:2]) == {"b", "d"} and r["ranked"][-1] == "c"
    assert r["winner"]["candidate"] == "b" and r["deciding_step"] is None             # b and d tie on every step: input order
    none = select_best.select_winner([_cand("a", fr=5), _cand("b", fr=6)], CFG3)
    assert none["winner"] is None and none["ranked"] == [] and len(none["excluded"]) == 2


@pytest.mark.parametrize("better,worse,step", [
    (dict(correct=9), dict(correct=8), "correct_refusals_strict"),
    (dict(f1=0.80), dict(f1=0.70), "token_f1"),
    (dict(nf=1.0), dict(nf=0.96), "number_faithfulness"),
    (dict(peak=900.0), dict(peak=1000.0), "peak_gpu_mem"),
    (dict(params=50), dict(params=100), "trainable_params"),
    (dict(secs=5.0), dict(secs=10.0), "train_seconds"),
    (dict(epoch=1), dict(epoch=2), "epoch"),
])
def test_each_step_of_the_rule_decides_in_order(better, worse, step):
    for first, second in ((better, worse), (worse, better)):                      # the order of the input must not matter
        cands = [_cand("good", **first) if first is better else _cand("bad", **first),
                 _cand("bad", **second) if first is better else _cand("good", **second)]
        r = select_best.select_winner(cands, CFG3)
        assert r["winner"]["candidate"] == "good" and r["deciding_step"] == step


def test_higher_steps_beat_lower_steps_and_none_faithfulness_is_last():
    # more correct refusals beats better F1 and lower cost
    r = select_best.select_winner([_cand("x", correct=8, f1=0.5, peak=9000.0), _cand("y", correct=7, f1=0.9, peak=1.0)], CFG3)
    assert r["winner"]["candidate"] == "x"
    # F1 differences below the rounding (4 decimals) count as a tie and fall through to the next step
    r = select_best.select_winner([_cand("x", f1=0.70001, peak=9.0), _cand("y", f1=0.70002, peak=1.0)], CFG3)
    assert r["winner"]["candidate"] == "y" and r["deciding_step"] == "peak_gpu_mem"
    # a candidate whose number faithfulness is None (no numbers) ranks after one that has a value
    r = select_best.select_winner([_cand("x", nf=None), _cand("y", nf=0.5)], CFG3)
    assert r["winner"]["candidate"] == "y"
    # missing cost information never beats known cost
    c = _cand("x")
    c["cost"] = None
    assert select_best.select_winner([c, _cand("y")], CFG3)["winner"]["candidate"] == "y"


def test_closeness_says_when_the_lead_is_inside_the_intervals():
    near = select_best.closeness(_cand("w", correct=9), _cand("r", correct=8))
    assert any("NOT established" in x and "9/11" in x and "8/11" in x for x in near)
    far = select_best.closeness(_cand("w", correct=11, f1=0.9), _cand("r", correct=1, f1=0.4))
    assert any("do not overlap" in x for x in far) and not any("NOT established" in x for x in far)
    assert any("equal" in x for x in select_best.closeness(_cand("w", correct=8), _cand("r", correct=8)))


def test_pass_criteria_rows():
    base = _cand("base", f1=0.575)["metrics"]
    good = _cand("g", correct=8, fr=3, f1=0.70)["metrics"]
    rows = {r["id"]: r for r in select_best.criteria_rows(good, "val", base, CFG3)}
    assert [rows[i]["ok"] for i in (1, 2, 3, 4, 5, 6)] == [True] * 6 and rows[7]["ok"] is None
    bad = _cand("b", correct=6, fr=5, f1=0.50, nf=0.9)["metrics"]
    bad["hit_max_new_tokens"], bad["repetition_loops"] = 2, 1
    rows = {r["id"]: r for r in select_best.criteria_rows(bad, "val", base, CFG3)}
    assert [rows[i]["ok"] for i in (1, 2, 3, 4, 5)] == [False] * 5 and rows[7]["ok"] is False and "no DPO" in rows[7]["value"]
    # test thresholds: at most 3 false refusals, F1 target reported, and exactly one test run recorded
    t = _cand("t", correct=7, fr=4, f1=0.62)["metrics"]
    rows = {r["id"]: r for r in select_best.criteria_rows(t, "test", _cand("base", f1=0.542)["metrics"], CFG3, test_runs=1)}
    assert rows[2]["ok"] is False and rows[3]["ok"] is True and "target 0.60 reached" in rows[3]["value"] and rows[6]["ok"] is True
    assert select_best.criteria_rows(t, "test", None, CFG3, test_runs=2)[5]["ok"] is False
    nf_none = select_best.criteria_rows(_cand("n", nf=None)["metrics"], "val", base, CFG3)
    assert nf_none[3]["ok"] is None and "n/a" in nf_none[3]["value"]


def test_statements_against_the_best_prompted_baseline():
    mine = _cand("m", correct=8, f1=0.90)["metrics"]
    mine["key_recall"] = {"mean": 0.9, "ci95": [0.85, 0.95], "n": 42}
    mine["key_support"] = {"mean": 0.5, "ci95": [0.45, 0.55], "n": 42}
    p1, p2 = _cand("p1", correct=0, f1=0.60)["metrics"], _cand("p2", correct=1, f1=0.88)["metrics"]
    for p in (p1, p2):
        p["key_recall"] = {"mean": 0.6, "ci95": [0.5, 0.7], "n": 42}
        p["key_support"] = {"mean": 0.7, "ci95": [0.6, 0.8], "n": 42}
    out = select_best.prompted_statements(mine, "val", {"base_p_sentence": p1, "base_p_fewshot": p2}, "tuned")
    text = "\n".join(out)
    assert "token F1 0.900" in text and "base_p_fewshot 0.880" in text and "intervals overlap" in text   # 0.90 vs 0.88: overlap
    assert "key recall 0.900" in text and "is higher than the best prompted baseline" in text
    assert "key support 0.500" in text and "is lower than" in text
    assert "strict correct refusals 8/11 is higher than the best prompted baseline base_p_fewshot 1/11" in text
    assert select_best.prompted_statements(mine, "val", {}, "tuned") == []


# ---- end to end on fake saved files ----------------------------------------------------------------------------------------------------------
def test_collect_selection_and_markdown_from_saved_files(tmp_path):
    c3, ec = _fill_world(tmp_path)
    data = select_best.collect(c3, ec)
    g = data["grid"]
    assert len(g["candidates"]) == 12 and g["missing_runs"] == []
    assert [e["candidate"] for e in g["excluded"]] == ["lora_fp16_lr2e-4_e3"]               # 5 false refusals
    assert g["winner"]["candidate"] == "qlora_4bit_lr2e-4_e2" and g["deciding_step"] == "token_f1"
    assert g["winner"]["method"] == "qlora_4bit" and g["winner"]["lr"] == 2e-4 and g["winner"]["epoch"] == 2
    rp = data["repeats"]
    assert rp["seeds_found"] == [42, 43, 44] and rp["stats"]["correct_refusals_strict"] == {"mean": pytest.approx(8.0), "min": 7, "max": 9, "n": 3}
    assert data["q15"]["winner"]["candidate"] == "q15_qlora_4bit_e2" and data["q15"]["winner"]["arm"] == "q15"
    assert set(data["baselines"]["test"]) == {"base", "base_p_sentence", "base_p_fewshot", "base15"}
    assert data["tuned"]["main"]["test"] is None and data["tuned"]["q15"]["test"] is None        # test not run yet
    md = select_best.selection_md(data, c3)
    assert md.count("| lora_fp16_lr1e-4_e1 |") == 1 and "**qlora_4bit_lr2e-4_e2**" in md and "**WINNER**" in md
    assert sum(1 for l in md.splitlines() if re.match(r"\| \*{0,2}(lora_fp16|qlora_4bit)_lr[0-9e-]+_e[123]\*{0,2} \|", l)) == 12
    assert "excluded (false refusals > 4)" in md and "eligible, rank 2" in md
    for needle in ("base 0.5B, strict_sentence prompt", "base 0.5B, fewshot prompt", "base 1.5B (default prompt)",
                   "Model size comparison (val, then test once)", "#### val", "#### test", "not yet measured",
                   "Seed repeat of the winner", "Against the prompted baselines", "The 7 pass criteria", "Training cost per run"):
        assert needle in md, needle
    assert "| qlora_4bit_lr2e-4_s43 |" in md or "qlora_4bit_lr2e-4_s43" in md                  # seed repeat runs in the cost table


def test_payload_is_frozen_without_tuned_test_results_and_is_stable(tmp_path):
    c3, ec = _fill_world(tmp_path)
    before = json.dumps(select_best.payload(select_best.collect(c3, ec)), sort_keys=True)
    _write_eval(ec, "qlora_4bit_lr2e-4_e2", "test", 0, 9)                                     # a test run appears later
    data = select_best.collect(c3, ec)
    assert data["tuned"]["main"]["test"] is not None
    assert json.dumps(select_best.payload(data), sort_keys=True) == before                    # the frozen part does not change


def test_main_writes_freezes_and_refuses_to_rewrite(tmp_path, monkeypatch, capsys):
    c3, ec = _fill_world(tmp_path)
    monkeypatch.setattr(select_best, "load_phase3_config", lambda: c3)
    monkeypatch.setattr(select_best, "load_eval_config", lambda: ec)
    assert select_best.main([]) == 0
    sp, mp = selection_io.selection_path(c3), Path(c3["paths"]["selection_md"])
    assert sp.exists() and mp.exists() and not selection_io.is_frozen(c3)
    out = capsys.readouterr().out
    assert "0.5B winner: qlora_4bit_lr2e-4_e2" in out and "final_test --size 0.5b" in out and "1.5B winner: q15_qlora_4bit_e2" in out
    with pytest.raises(SystemExit):
        select_best.main(["--final-tables"])                                                   # not frozen yet
    select_best.main(["--freeze"])
    assert selection_io.is_frozen(c3)
    sha = selection_io.sha_path(c3).read_text(encoding="utf-8").split()[0]
    assert sha == selection_io.file_sha256(sp)
    frozen_bytes = sp.read_bytes()
    _write_eval(ec, "lora_fp16_lr1e-4_e1", "val", 0, 11)                                      # a val file changes afterwards
    select_best.main([])
    assert sp.read_bytes() == frozen_bytes and "frozen" in capsys.readouterr().out            # the frozen file is not rewritten
    sp.write_text(sp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert not selection_io.is_frozen(c3)                                                     # any edit breaks the freeze


def test_winner_run_and_find_run_read_the_saved_selection(tmp_path, monkeypatch):
    c3, ec = _fill_world(tmp_path)
    with pytest.raises(SystemExit):
        selection_io.winner_run(c3)                                                           # no selection file yet
    with pytest.raises(SystemExit):
        train.resolve_runs("repeats", c3)
    monkeypatch.setattr(select_best, "load_phase3_config", lambda: c3)
    monkeypatch.setattr(select_best, "load_eval_config", lambda: ec)
    select_best.main([])
    w = selection_io.winner_run(c3)
    assert (w["method"], w["lr"], w["name"]) == ("qlora_4bit", 2e-4, "qlora_4bit_lr2e-4")
    assert [r["name"] for r in train.resolve_runs("repeats", c3)] == ["qlora_4bit_lr2e-4_s43", "qlora_4bit_lr2e-4_s44"]
    assert [r["name"] for r in train.resolve_runs("q15", c3)] == ["q15_qlora_4bit"]
    assert train.find_run("q15_qlora_4bit", c3)["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert train.find_run("lora_fp16_lr1e-4", c3)["method"] == "lora_fp16"
    with pytest.raises(SystemExit):
        train.find_run("nonsense", c3)
    # no winner at all: nothing to repeat
    bad = {"grid": {"winner": None}}
    with pytest.raises(SystemExit):
        selection_io.winner_run(c3, bad)


def test_incomplete_runs_are_reported_not_hidden(tmp_path):
    c3, ec = _world(tmp_path)
    run = grid.grid_runs(c3)[0]
    _write_log(c3, run["name"], 5000.0)
    _write_eval(ec, grid.candidate_name(run["name"], 1), "val", 0, 5)
    data = select_best.collect(c3, ec)
    assert len(data["grid"]["candidates"]) == 1 and data["grid"]["missing_runs"][0]["run"] == run["name"]
    md = select_best.selection_md(data, c3)
    assert "Runs with fewer than 3 evaluated epochs" in md and "not run" in md
    empty = select_best.collect(*_world(tmp_path / "nothing"))
    assert empty["grid"]["winner"] is None
    assert "No candidate passed the false-refusal filter" in select_best.selection_md(empty, _world(tmp_path / "nothing")[0])


# ---- the single test run ------------------------------------------------------------------------------------------------------------------
def test_final_test_is_gated_by_the_frozen_selection_and_runs_once(tmp_path, monkeypatch):
    c3, ec = _fill_world(tmp_path)
    monkeypatch.setattr(select_best, "load_phase3_config", lambda: c3)
    monkeypatch.setattr(select_best, "load_eval_config", lambda: ec)
    select_best.main([])
    with pytest.raises(SystemExit, match="not frozen"):
        final_test.check_can_run(c3, ec, "0.5b")
    select_best.main(["--freeze"])
    w = final_test.check_can_run(c3, ec, "0.5b")
    assert w["candidate"] == "qlora_4bit_lr2e-4_e2"
    cmd = final_test.test_command(c3, w, "0.5b")
    assert cmd[cmd.index("--split") + 1] == "test" and cmd[cmd.index("--name") + 1] == w["candidate"]
    assert cmd[cmd.index("--base-model-id") + 1] == "Qwen/Qwen2.5-0.5B-Instruct" and cmd[cmd.index("--model") + 1].endswith("e2")
    w15 = final_test.check_can_run(c3, ec, "1.5b")
    assert final_test.test_command(c3, w15, "1.5b")[-1] == "Qwen/Qwen2.5-1.5B-Instruct"
    final_test.record(c3, w)
    log = json.loads((tmp_path / c3["paths"]["test_log"]).read_text(encoding="utf-8"))
    assert log[0]["candidate"] == w["candidate"] and log[0]["selection_sha256"] == selection_io.file_sha256(selection_io.selection_path(c3))
    assert select_best.read_test_runs(c3) == {w["candidate"]: 1}
    _write_eval(ec, w["candidate"], "test", 0, 8)                                              # the test file now exists
    with pytest.raises(SystemExit, match="already evaluated on test"):
        final_test.check_can_run(c3, ec, "0.5b")
    # final tables: only after the freeze, and with the test column filled for the tested model
    select_best.main(["--final-tables"])
    final = Path(c3["paths"]["final_md"]).read_text(encoding="utf-8")
    assert "Selection file sha256" in final and "not yet measured" in final                    # 1.5B not tested yet
    assert "test runs recorded: 1" in final


def test_final_test_without_a_winner_refuses(tmp_path, monkeypatch):
    c3, ec = _world(tmp_path)
    monkeypatch.setattr(select_best, "load_phase3_config", lambda: c3)
    monkeypatch.setattr(select_best, "load_eval_config", lambda: ec)
    select_best.main(["--freeze"])
    with pytest.raises(SystemExit, match="no winner"):
        final_test.check_can_run(c3, ec, "0.5b")


# ---- the sweep cannot touch the test split ------------------------------------------------------------------------------------------------------
def test_adapter_evaluation_loop_accepts_val_only():
    with pytest.raises(SystemExit):
        eval_adapters.main(["--group", "grid", "--split", "test"])
    assert eval_adapters.ALLOWED_SPLITS == ("val",)


def test_adapter_plan_and_command(tmp_path):
    c3, ec = _world(tmp_path)
    plan = eval_adapters.plan(grid.grid_runs(c3), c3, ec, "val")
    assert len(plan) == 12 and plan[0]["name"] == "lora_fp16_lr1e-4_e1" and plan[-1]["name"] == "qlora_4bit_lr2e-4_e3"
    assert all(p["eval_file"].name == f"eval_{p['name']}_val.json" for p in plan)
    cmd = eval_adapters.eval_command(plan[0]["adapter"], plan[0]["name"], "val", plan[0]["base_model_id"])
    assert cmd[cmd.index("--split") + 1] == "val" and "--base-model-id" in cmd and cmd.count("--split") == 1
    q = eval_adapters.plan([grid.q15_run(c3, grid.grid_runs(c3)[0])], c3, ec, "val")
    assert q[0]["base_model_id"] == "Qwen/Qwen2.5-1.5B-Instruct" and q[0]["name"] == "q15_lora_fp16_e1"


def test_training_code_never_names_the_test_split():
    for mod in (train, eval_adapters, prep, masking, schedule, grid, mlflow_log):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert '"test"' not in src and "'test'" not in src and "['test']" not in src, mod.__name__
    train_src = Path(train.__file__).read_text(encoding="utf-8")
    assert 'paths["train"]' in train_src and 'paths["val"]' in train_src


def test_training_modules_import_without_torch():
    for name in ("torch", "transformers", "peft", "bitsandbytes", "mlflow"):
        assert name not in sys.modules, name


def test_notebook_phase3_section():
    nb = json.loads((REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    p3 = [c for c in code if c.lstrip().startswith("# P3-")]
    ids = [re.match(r"# P3-(\d+b?)\.", c).group(1) for c in p3]
    assert ids == ["1", "2", "3", "3b", "4", "5", "6", "7", "8", "9", "10"]
    assert all(len(c.splitlines()) <= 15 for c in code)
    by = dict(zip(ids, p3))
    touches_test = {i for i, c in by.items() if "test" in c.lower()}
    assert touches_test == {"7", "9"}                                                           # only these two cells mention test
    assert "untuned" in by["7"].lower() and "--name base15" in by["7"] and "FINAL TEST" in by["9"]
    for i in ("3", "3b", "5", "6"):
        assert "--split val" in by[i] and "--split test" not in by[i]
    assert "--on-4bit-base" in by["3b"] and "--group grid" in by["3b"] and "--on-4bit-base" not in by["3"]
    assert "--group grid" in by["2"] and "--group repeats" in by["5"] and "--group q15" in by["6"]
    assert "--freeze" in by["8"] and "final_test --size 0.5b" in by["9"] and "final_test --size 1.5b" in by["9"]
    assert "phase3-results" in by["10"] and "phase3-adapters" in by["10"] and "phase3_results.zip" in by["10"]
    assert "requirements-gpu.txt" in by["1"] and "bitsandbytes" in by["1"] and "mlflow" in by["1"]
    assert code.index(by["3b"]) == code.index(by["3"]) + 1 and code.index(by["4"]) > code.index(by["3b"])   # after P3-3, before P3-4
    assert code.index(by["9"]) > code.index(by["8"]) > code.index(by["6"])                      # test comes after the freeze
    assert [c for c in code if "training.final_test" in c] == [by["9"]]


# ---- evaluator changes ----------------------------------------------------------------------------------------------------------------------
def test_adapter_base_check(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}), encoding="utf-8")
    assert evaluate.check_adapter_base(d, "Qwen/Qwen2.5-1.5B-Instruct") == "Qwen/Qwen2.5-1.5B-Instruct"
    with pytest.raises(ValueError, match="--base-model-id"):
        evaluate.check_adapter_base(d, "Qwen/Qwen2.5-0.5B-Instruct")                          # wrong base: no silent wrong numbers
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert evaluate.check_adapter_base(d, "anything") is None
    evaluate.validate_model_args("base", "X")
    evaluate.validate_model_args(str(d), "X")
    with pytest.raises(ValueError):
        evaluate.validate_model_args("some/full-model", "X")
    evaluate.validate_model_args("some/full-model", None)
    with pytest.raises(SystemExit):
        evaluate.main(["--split", "val", "--model", "some/full-model", "--base-model-id", "X"])


def test_eval_file_names_the_base_model(tmp_path):
    class B(_Scripted):
        def info(self):
            return {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": None, "adapter": None, "dtype": "float16", "device": "n"}

    d = evaluate.run_eval(B(_rows("val")[:5], 0, 0), ECFG, "val", "base15", limit=5)
    assert d["base_model_id"] == d["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_default_decoding_block_is_unchanged_by_phase3():
    d = evaluate.run_eval(_Scripted(_rows("val")[:3], 0, 0), ECFG, "val", "base", limit=3)
    assert d["decoding"] == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0, "padding_side": "left",
                             "batch_size": ECFG["eval"]["batch_size"]}


# ---- MLflow helpers, packaging, report -----------------------------------------------------------------------------------------------------------
def test_mlflow_helpers_need_no_mlflow():
    assert mlflow_log.clean_params({"a": 1, "b": "x" * 900})["b"] == "x" * 490 and mlflow_log.clean_params({"a": 1})["a"] == "1"
    uri = mlflow_log.tracking_uri(CFG3)
    assert uri.startswith("file:") and uri.endswith("outputs/mlruns")


def test_collect_phase3_files(tmp_path):
    out = tmp_path / "outputs"
    names = ["phase3_selection.md", "phase3_selection.json", "phase3_selection.sha256", "phase3_test_runs.json",
             "eval_lora_fp16_lr1e-4_e1_val.json", "eval_qlora_4bit_lr2e-4_s43_e2_val.json", "eval_q15_qlora_4bit_e1_val.json",
             "eval_base15_val.json", "refusal_detection_review_lora_fp16_lr1e-4_e1_val.csv",
             "runs/lora_fp16_lr1e-4/train_log.json", "mlruns/0/abc/metrics/train_loss", "adapters/lora_fp16_lr1e-4/e1/adapter_model.safetensors"]
    other = ["eval_base_test.json", "eval_base_p_fewshot_val.json", "baseline_table.md", "colab_bundle.zip"]
    for n in names + other:
        (out / n).parent.mkdir(parents=True, exist_ok=True)
        (out / n).write_text("x", encoding="utf-8")
    res = {p.relative_to(out).as_posix() for p in collect.phase3_result_files(out, ECFG)}
    assert res == set(names) - {"adapters/lora_fp16_lr1e-4/e1/adapter_model.safetensors"}
    ad = {p.relative_to(out).as_posix() for p in collect.phase3_adapter_files(out, ECFG)}
    assert ad == {"adapters/lora_fp16_lr1e-4/e1/adapter_model.safetensors"}
    z = collect.make_zip(collect.phase3_result_files(out, ECFG), tmp_path / "r.zip", root=out)
    assert "runs/lora_fp16_lr1e-4/train_log.json" in zipfile.ZipFile(z).namelist()
    with pytest.raises(SystemExit):
        collect.main(["--outputs", str(tmp_path / "empty"), "--set", "phase3-adapters", "--out", str(tmp_path / "e.zip")])
    # the Phase 2c selection is unchanged
    assert {p.name for p in collect.result_files(out, ECFG)} == {"eval_base_p_fewshot_val.json"}


def test_sweep_runs_stay_out_of_the_phase2_table(tmp_path):
    ec = copy.deepcopy(ECFG)
    ec["eval"]["n_boot"] = 30
    for name in ("base", "lora_fp16_lr1e-4_e1", "q15_qlora_4bit_e2"):
        evaluate.run_eval(_Scripted(_rows("val"), 0, 0), ec, "val", name, out_path=tmp_path / f"eval_{name}_val.json")
    evaluate.run_eval(_Scripted(_rows("val"), 0, 0), ec, "val", "base15", out_path=tmp_path / "eval_base15_val.json")
    report.main(["--outputs", str(tmp_path)])
    table = (tmp_path / "baseline_table.md").read_text(encoding="utf-8")
    assert "lora_fp16_lr1e-4_e1" not in table and "q15_qlora_4bit_e2" not in table
    assert "base Qwen2.5-1.5B-Instruct (Phase 3 arm)" in table and "base Qwen2.5-0.5B-Instruct" in table


@needs_training_data
def test_requirements_pins_and_bundle():
    gpu = (REPO_ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    lines = [l.strip() for l in gpu.splitlines() if l.strip() and not l.startswith("#")]
    assert {"transformers==5.18.0", "accelerate==1.15.0", "peft==0.21.1", "huggingface_hub==1.33.0"} <= set(lines)
    assert any(l.startswith("bitsandbytes") for l in lines) and any(l.startswith("mlflow") for l in lines)
    assert not any(l.startswith(("torch", "trl")) for l in lines)
    cpu = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert not any(x in cpu for x in ("torch", "transformers", "peft", "bitsandbytes", "mlflow"))
    from make_colab_bundle import bundle_files
    names = {p.relative_to(REPO_ROOT).as_posix() for p in bundle_files(ECFG)}
    for must in ("configs/phase3.yaml", "climate_slm/chat.py", "climate_slm/training/train.py", "climate_slm/training/select_best.py",
                 "climate_slm/training/final_test.py", "climate_slm/training/eval_adapters.py", "climate_slm/training/masking.py",
                 "data/instruction/sft_train.jsonl", "data/instruction/sft_val.jsonl"):
        assert must in names, must
    assert not any(n.startswith(("docs/", "outputs/")) for n in names)


# ---- fixes after the first Colab run: MLflow file store, torchao ------------------------------------------------------------------------
def _fake_mlflow(monkeypatch, fail=()):
    """A stand-in `mlflow` module. Functions named in `fail` raise."""
    import types
    calls = []
    m = types.ModuleType("mlflow")

    def make(name, result=None):
        def fn(*a, **k):
            calls.append(name)
            if name in fail:
                raise RuntimeError(f"{name} broke")
            return result
        return fn

    class Run:
        info = types.SimpleNamespace(run_id="rid123")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    for n in ("set_tracking_uri", "set_experiment", "log_params", "set_tags", "log_metrics", "end_run"):
        setattr(m, n, make(n))

    def start_run(*a, **k):
        calls.append("start_run")
        if "start_run" in fail:
            raise RuntimeError("start_run broke")
        return Run()

    m.start_run = start_run
    monkeypatch.setitem(sys.modules, "mlflow", m)
    return calls


def test_mlflow_file_store_env_default_is_set_before_mlflow_is_imported():
    import os
    import subprocess
    code = "import os, climate_slm.training.mlflow_log; print(os.environ.get('MLFLOW_ALLOW_FILE_STORE'))"
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_ALLOW_FILE_STORE"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert out.stdout.strip() == "true", out.stderr
    env["MLFLOW_ALLOW_FILE_STORE"] = "false"                                      # a value set by the user is kept
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert out.stdout.strip() == "false"


def test_mlflow_start_failure_does_not_stop_training(monkeypatch, capsys):
    _fake_mlflow(monkeypatch, fail=("start_run",))
    t = mlflow_log.start_tracker(CFG3, "run", {"a": 1}, {"phase": "3"})             # must not raise
    assert not t.active and t.run_id is None
    t.log_metrics({"x": 1.0}, step=1)                                              # switched-off tracker: does nothing
    t.end("FAILED")
    assert "WARNING: MLflow start failed" in capsys.readouterr().out


def test_mlflow_package_missing_does_not_stop_training(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlflow", None)                              # `import mlflow` raises ImportError
    t = mlflow_log.start_tracker(CFG3, "run", {}, {})
    assert not t.active and "WARNING" in capsys.readouterr().out
    assert mlflow_log.log_metrics_into_run(CFG3, "rid", {"x": 1.0}, 1) is False


def test_mlflow_errors_during_training_are_swallowed(monkeypatch, capsys):
    calls = _fake_mlflow(monkeypatch, fail=("log_metrics", "end_run", "log_params"))
    t = mlflow_log.start_tracker(CFG3, "run", {"a": 1}, {})
    assert t.active and t.run_id == "rid123"                                       # the run exists, only params failed
    t.log_metrics({"train_loss": 1.0}, step=3)
    t.log_metrics({"peak": 5.0})
    t.end()
    out = capsys.readouterr().out
    assert out.count("WARNING: MLflow") == 4 and "log_params" in out and "end_run" in out
    assert calls.count("log_metrics") == 2
    assert mlflow_log.log_metrics_into_run(CFG3, "rid", {"x": 1.0}, 1) is False


def test_mlflow_working_case_logs_and_resumes(monkeypatch, capsys):
    calls = _fake_mlflow(monkeypatch)
    t = mlflow_log.start_tracker(CFG3, "run", {"a": 1}, {"phase": "3"})
    t.log_metrics({"x": 1.0}, step=1)
    t.end()
    assert t.active and calls == ["set_tracking_uri", "set_experiment", "start_run", "log_params", "set_tags", "log_metrics", "end_run"]
    assert mlflow_log.log_metrics_into_run(CFG3, "rid", {"x": 1.0}, 2) is True
    assert "WARNING" not in capsys.readouterr().out


def test_training_script_logs_only_through_the_safe_tracker():
    src = Path(train.__file__).read_text(encoding="utf-8")
    assert "mlflow_log.start_tracker(" in src and "mlflow_log.start_run(" not in src
    assert "mlflow.log_metrics(" not in src.replace("tracker.log_metrics(", "") and "mlflow.end_run(" not in src


def test_torchao_is_removed_before_the_install_and_documented():
    nb = json.loads((REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    p31 = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith("# P3-1."))
    assert "!pip uninstall -y -q torchao" in p31 and p31.index("uninstall") < p31.index("pip install")
    assert len(p31.splitlines()) <= 15
    req = (REPO_ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "torchao" in req and not any(l.strip().startswith("torchao") for l in req.splitlines())   # only a comment
    guide = REPO_ROOT / "notebooks" / "README_COLAB.md"                         # the notebook guide is optional in a copy
    if guide.exists():
        assert "torchao" in guide.read_text(encoding="utf-8")


# ---- QLoRA adapters on the 4-bit base (Phase 3b) ------------------------------------------------------------------------------------------------------
def _fake_stack(monkeypatch, calls):
    """Fake torch / transformers / peft, enough to run HFBackend.__init__ without a GPU."""
    import types
    fp16 = object()

    class Param:
        dtype = fp16

    class Model:
        def __init__(self, quantized):
            self.quantized = quantized
            self.config = types.SimpleNamespace(_commit_hash="abc")
            self.generation_config = types.SimpleNamespace(eos_token_id=[7])

        def parameters(self):
            return iter([Param()])

        def to(self, x):
            calls.append(("to", x))
            if self.quantized:
                raise AssertionError("a 4-bit model cannot be moved with .to()")
            return self

        def eval(self):
            return self

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kw):
            calls.append(("kw", kw))
            return Model("quantization_config" in kw)

    class Tok:
        pad_token_id, eos_token_id, padding_side = 0, 7, "right"

        @staticmethod
        def from_pretrained(model_id, **kw):
            return Tok()

    class Peft:
        @staticmethod
        def from_pretrained(model, adapter):
            calls.append(("peft", adapter))
            return model

    class BnB:
        def __init__(self, **kw):
            self.kw = kw

    torch = types.SimpleNamespace(float16=fp16, cuda=types.SimpleNamespace(is_available=lambda: False))
    tf = types.SimpleNamespace(AutoModelForCausalLM=AutoModel, AutoTokenizer=Tok, BitsAndBytesConfig=BnB)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=Peft))
    return BnB


def _adapter_dir(tmp_path, base="Qwen/Qwen2.5-0.5B-Instruct"):
    d = tmp_path / "ad"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}), encoding="utf-8")
    return str(d)


def test_backend_loads_the_base_in_4bit_with_the_training_config(monkeypatch, tmp_path):
    calls = []
    BnB = _fake_stack(monkeypatch, calls)
    b = evaluate.HFBackend(ECFG, _adapter_dir(tmp_path), load_4bit=True)
    kw = next(c[1] for c in calls if c[0] == "kw")
    assert isinstance(kw["quantization_config"], BnB) and kw["device_map"] == {"": 0}
    q = kw["quantization_config"].kw
    assert q["load_in_4bit"] is True and q["bnb_4bit_quant_type"] == "nf4" and q["bnb_4bit_use_double_quant"] is True
    assert q["bnb_4bit_compute_dtype"] is sys.modules["torch"].float16                      # as in training: float16 compute
    assert not any(c[0] == "to" for c in calls)                                              # never .to() on a 4-bit model
    assert ("peft", str(tmp_path / "ad")) in calls and b.adapter and b.info()["load_4bit"] is True
    assert b.info()["quantization"] == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                                        "bnb_4bit_compute_dtype": "float16"}


def test_backend_default_still_loads_fp16_and_moves_to_the_device(monkeypatch, tmp_path):
    calls = []
    _fake_stack(monkeypatch, calls)
    b = evaluate.HFBackend(ECFG, _adapter_dir(tmp_path))
    kw = next(c[1] for c in calls if c[0] == "kw")
    assert "quantization_config" not in kw and "device_map" not in kw and set(kw) == {"revision", "dtype"}
    assert ("to", ECFG["eval"]["device"]) in calls
    assert b.info()["load_4bit"] is False and b.info()["quantization"] is None


def test_training_and_evaluation_share_one_quantisation_config():
    import climate_slm.quant as quant
    method = CFG3["methods"]["qlora_4bit"]
    assert quant.quant_settings(method) == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                            "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float16"}
    for mod in (train, evaluate):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "bnb_config(" in src and "BitsAndBytesConfig(" not in src, mod.__name__   # only climate_slm/quant.py builds it
    assert "BitsAndBytesConfig(" in Path(quant.__file__).read_text(encoding="utf-8")


def test_load_4bit_flag_reaches_the_command_line_parser():
    with pytest.raises(SystemExit):
        evaluate.main(["--split", "val", "--load-4bit", "--decoding", "qwen_default"])      # parsed, then fails on the missing seed


def test_on4bit_names_commands_and_plan(tmp_path):
    c3, ec = _world(tmp_path)
    assert grid.candidate_name("qlora_4bit_lr2e-4", 2, True) == "qlora_4bit_lr2e-4_e2__on4bit"
    assert grid.candidate_name("qlora_4bit_lr2e-4", 2) == "qlora_4bit_lr2e-4_e2"
    plan = eval_adapters.plan(grid.grid_runs(c3), c3, ec, "val", on_4bit=True)
    assert len(plan) == 6 and all(p["run"]["method"] == "qlora_4bit" and p["load_4bit"] for p in plan)   # only QLoRA runs
    assert plan[0]["name"] == "qlora_4bit_lr1e-4_e1__on4bit" and plan[-1]["name"] == "qlora_4bit_lr2e-4_e3__on4bit"
    assert all(p["eval_file"].name == f"eval_{p['name']}_val.json" for p in plan)
    old = eval_adapters.plan(grid.grid_runs(c3), c3, ec, "val")
    assert len(old) == 12 and not any(p["load_4bit"] for p in old) and not any("on4bit" in p["name"] for p in old)
    assert {p["eval_file"] for p in plan}.isdisjoint({p["eval_file"] for p in old})            # the fp16 results are never overwritten
    cmd = eval_adapters.eval_command(plan[0]["adapter"], plan[0]["name"], "val", plan[0]["base_model_id"], True)
    assert "--load-4bit" in cmd and cmd[cmd.index("--split") + 1] == "val" and cmd[cmd.index("--name") + 1].endswith("__on4bit")
    assert "--load-4bit" not in eval_adapters.eval_command(old[0]["adapter"], old[0]["name"], "val", old[0]["base_model_id"])
    with pytest.raises(SystemExit):
        eval_adapters.main(["--group", "grid", "--split", "test", "--on-4bit-base"])           # still val only


def _fill_4bit(ec, script):
    for run, rows in script.items():
        for e, (fr, cr) in enumerate(rows, 1):
            _write_eval(ec, grid.candidate_name(run, e, True), "val", fr, cr)


def test_4bit_rows_are_extra_candidates_under_the_same_rule(tmp_path):
    c3, ec = _fill_world(tmp_path)
    before = select_best.collect(c3, ec)
    assert before["grid"]["winner"]["candidate"] == "qlora_4bit_lr2e-4_e2" and len(before["grid"]["candidates"]) == 12
    # the QLoRA adapters refuse better on their own 4-bit base: 10/11 for the lr 2e-4 epoch 3, 0 false refusals
    _fill_4bit(ec, {"qlora_4bit_lr1e-4": [(0, 0), (0, 0), (0, 0)], "qlora_4bit_lr2e-4": [(0, 6), (0, 9), (0, 10)]})
    data = select_best.collect(c3, ec)
    g = data["grid"]
    assert len(g["candidates"]) == 18 and g["missing_runs"] == []
    four = [c for c in g["candidates"] if c["eval_base"] == "4bit"]
    assert len(four) == 6 and all(c["candidate"].endswith("__on4bit") and c["method"] == "qlora_4bit" for c in four)
    assert all("__on4bit" not in c["candidate"] for c in g["candidates"] if c["eval_base"] == "fp16")
    assert g["winner"]["candidate"] == "qlora_4bit_lr2e-4_e3__on4bit" and g["winner"]["eval_base"] == "4bit"
    old_rows = {c["candidate"]: c["metrics"] for c in g["candidates"] if c["eval_base"] == "fp16"}
    assert old_rows == {c["candidate"]: c["metrics"] for c in before["grid"]["candidates"]}   # the old rows are exactly as before
    md = select_best.selection_md(data, c3)
    assert "evaluated on" in md and md.count("4-bit base |") >= 6 and "fp16 base |" in md and "EXTRA candidates" in md
    assert "correct refusals, 4-bit base" in md and "**The winner was evaluated on the 4-bit base.**" in md
    assert "**qlora_4bit_lr2e-4_e3__on4bit**" in md and "**WINNER**" in md
    cand_rows = [l for l in md.splitlines() if re.match(r"\| \*{0,2}(lora_fp16|qlora_4bit)_lr[0-9e-]+_e[123](__on4bit)?\*{0,2} \| (lora_fp16|qlora_4bit) \|", l)]
    assert len(cand_rows) == 18 and sum("| 4-bit base |" in l for l in cand_rows) == 6 and sum("| fp16 base |" in l for l in cand_rows) == 12


def test_selection_rule_is_unchanged_by_the_extra_rows(tmp_path):
    c3, ec = _fill_world(tmp_path)
    # 4-bit rows that are WORSE: the winner stays the old one, and the filter still applies to the new rows
    _fill_4bit(ec, {"qlora_4bit_lr1e-4": [(0, 0), (0, 0), (0, 0)], "qlora_4bit_lr2e-4": [(6, 11), (0, 3), (0, 2)]})
    g = select_best.collect(c3, ec)["grid"]
    assert g["winner"]["candidate"] == "qlora_4bit_lr2e-4_e2" and g["winner"]["eval_base"] == "fp16"
    excluded = {e["candidate"] for e in g["excluded"]}
    assert "qlora_4bit_lr2e-4_e1__on4bit" in excluded and "lora_fp16_lr2e-4_e3" in excluded    # 6 false refusals > 4
    assert "qlora_4bit_lr2e-4_e1__on4bit" not in g["ranked"]
    # a full tie between the same adapter on the two bases goes to the fp16 row (it comes first); nothing else is invented
    _fill_4bit(ec, {"qlora_4bit_lr2e-4": [(0, 7), (1, 8), (3, 8)]})                             # identical to the fp16 numbers
    g2 = select_best.collect(c3, ec)["grid"]
    assert g2["winner"]["candidate"] == "qlora_4bit_lr2e-4_e2"
    assert g2["ranked"].index("qlora_4bit_lr2e-4_e2") < g2["ranked"].index("qlora_4bit_lr2e-4_e2__on4bit")


def test_repeats_and_final_test_follow_the_winners_evaluation_base(tmp_path):
    c3, ec = _fill_world(tmp_path, with_repeats=False)
    _fill_4bit(ec, {"qlora_4bit_lr1e-4": [(0, 0), (0, 0), (0, 0)], "qlora_4bit_lr2e-4": [(0, 6), (0, 9), (0, 10)]})
    w = select_best.collect(c3, ec)["grid"]["winner"]
    assert w["eval_base"] == "4bit" and w["epoch"] == 3
    for seed in (43, 44):
        _write_eval(ec, f"qlora_4bit_lr2e-4_s{seed}_e3__on4bit", "val", 0, 8)
        _write_eval(ec, f"qlora_4bit_lr2e-4_s{seed}_e3", "val", 0, 0)                           # an fp16 evaluation exists too
    rp = select_best.collect(c3, ec)["repeats"]
    assert sorted(rp["seeds_found"]) == [42, 43, 44] and all(r["candidate"].endswith("__on4bit") for r in rp["rows"])
    cmd = final_test.test_command(c3, w, "0.5b")                                               # the final test uses the same base
    assert "--load-4bit" in cmd and cmd[cmd.index("--name") + 1] == "qlora_4bit_lr2e-4_e3__on4bit" and cmd[cmd.index("--split") + 1] == "test"
    assert cmd[cmd.index("--model") + 1].endswith("e3")
    assert "--load-4bit" not in final_test.test_command(c3, {**w, "eval_base": "fp16", "candidate": "x"}, "0.5b")


def test_4bit_metrics_go_to_their_own_mlflow_names(tmp_path, monkeypatch):
    c3, ec = _world(tmp_path)
    run = grid.grid_runs(c3)[2]
    _write_log(c3, run["name"], 3500.0)
    p = grid.run_log_path(c3, run["name"])
    log = json.loads(p.read_text(encoding="utf-8"))
    log["mlflow_run_id"] = "rid9"
    p.write_text(json.dumps(log), encoding="utf-8")
    item = eval_adapters.plan([run], c3, ec, "val", on_4bit=True)[1]
    _write_eval(ec, item["name"], "val", 0, 5)
    sent = {}
    monkeypatch.setattr(mlflow_log, "log_metrics_into_run",
                        lambda cfg3, rid, metrics, step: sent.update(rid=rid, metrics=metrics, step=step) or True)
    assert eval_adapters.log_to_mlflow(item, c3, ec, "val") is True
    assert sent["rid"] == "rid9" and sent["step"] == 2 and all(k.startswith("val_on4bit_") for k in sent["metrics"])


# ---- stale train_error.txt ------------------------------------------------------------------------------------------------------------------------
def _done_run(c3, name):
    """A finished run on disk: complete log and all epoch adapters."""
    _write_log(c3, name, 1000.0)
    for e in grid.epochs(c3):
        d = grid.adapter_dir(c3, name, e)
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")


def test_clear_stale_error_removes_only_the_error_file(tmp_path):
    c3, _ = _world(tmp_path)
    name = "lora_fp16_lr1e-4"
    assert grid.clear_stale_error(c3, name) is False                                   # nothing there: no error
    _write_log(c3, name, 1000.0)
    err = grid.error_file_path(c3, name)
    assert err.name == "train_error.txt" and err.parent == grid.run_log_path(c3, name).parent
    err.write_text("Traceback ... ImportError", encoding="utf-8")
    assert grid.clear_stale_error(c3, name) is True and not err.exists()
    assert grid.run_log_path(c3, name).exists()                                        # the log is kept
    other = "qlora_4bit_lr1e-4"
    _write_log(c3, other, 1000.0)
    grid.error_file_path(c3, other).write_text("another run's error", encoding="utf-8")
    grid.clear_stale_error(c3, name)
    assert grid.error_file_path(c3, other).exists()                                    # other runs are not touched


def test_training_success_path_clears_the_stale_error_and_failure_path_writes_it():
    src = Path(train.__file__).read_text(encoding="utf-8")
    body = src[src.index("def train_one"):src.index("# ---- command line")]
    done = body.index("complete=True")
    assert body.index("grid.clear_stale_error(", done) > done                            # after the run is marked complete
    assert body.index("grid.clear_stale_error(", done) < body.index("tracker.end()", done)
    assert body.count("clear_stale_error(") == 2                                        # success path + "already done" path
    main_src = src[src.index("def main"):]
    assert "error_file_path(" in main_src and "write_text(traceback.format_exc()" in main_src
    assert "clear_stale_error" not in main_src                                          # a failure never deletes the file


def test_group_skip_of_a_finished_run_also_clears_the_stale_error(tmp_path, capsys):
    c3, _ = _world(tmp_path)
    runs = grid.grid_runs(c3)
    _done_run(c3, runs[0]["name"])
    grid.error_file_path(c3, runs[0]["name"]).write_text("old failure", encoding="utf-8")
    assert train.run_group(runs[:1], c3) == []                                          # skipped, no subprocess started
    assert not grid.error_file_path(c3, runs[0]["name"]).exists()
    assert "already done, skipped" in capsys.readouterr().out


def test_unfinished_run_keeps_its_error_file(tmp_path):
    c3, _ = _world(tmp_path)
    run = grid.grid_runs(c3)[0]
    _write_log(c3, run["name"], 1000.0, complete=False)
    grid.error_file_path(c3, run["name"]).write_text("current failure", encoding="utf-8")
    assert not grid.run_is_done(c3, run)
    assert grid.error_file_path(c3, run["name"]).read_text(encoding="utf-8") == "current failure"
