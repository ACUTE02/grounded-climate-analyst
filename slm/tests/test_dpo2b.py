"""Tests for Phase 4b stage B (DPO v2): NLL loss term, grid, trainer structure, eval loop, test gate, v2 selection rule, collect sets,
notebook cells. CPU only. No torch, transformers, peft."""
import copy
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config, load_phase4b_config
from climate_slm.evaluation import collect
from climate_slm.evaluation import metrics as M
from climate_slm.training import (dpo2_eval, dpo2_final_test, dpo2_grid, dpo2_train, dpo_data, dpo_final_test, dpo_loss, grid,
                                  schedule, select_dpo, select_dpo2, selection_io)
from _release import needs_readme, needs_training_data

CFG = dpo2_grid.runtime_config()
CFG4B, CFG3, ECFG = load_phase4b_config(), load_phase3_config(), load_eval_config()
ROOT = REPO_ROOT


# ---- the loss ----------------------------------------------------------------------------------------------------------
def test_nll_per_token_and_alpha_zero_is_plain_dpo():
    assert dpo_loss.nll_per_token(-30.0, 10) == pytest.approx(3.0)
    args = (-4.0, -9.0, -5.0, -6.0, 0.1)
    assert dpo_loss.rpo_loss(*args, alpha=0.0, n_chosen_tokens=10) == pytest.approx(dpo_loss.dpo_loss(*args))


def test_rpo_loss_adds_alpha_times_the_chosen_nll():
    args = (-4.0, -9.0, -5.0, -6.0, 0.1)
    base, nll = dpo_loss.dpo_loss(*args), 4.0 / 8
    for alpha in (0.5, 1.0):
        assert dpo_loss.rpo_loss(*args, alpha=alpha, n_chosen_tokens=8) == pytest.approx(base + alpha * nll)
    # known value with policy == reference: DPO part = ln 2, NLL part = 12 / 6 = 2
    assert dpo_loss.rpo_loss(-12.0, -7.0, -12.0, -7.0, 0.1, alpha=0.5, n_chosen_tokens=6) == pytest.approx(math.log(2) + 0.5 * 2.0)


def test_nll_term_is_per_token_and_keeps_the_chosen_text_likely():
    # same sequence log-prob, longer text: smaller per-token NLL term (length-normalised)
    short = dpo_loss.rpo_loss(-12.0, -20.0, -12.0, -20.0, 0.1, 1.0, 3)
    long_ = dpo_loss.rpo_loss(-12.0, -20.0, -12.0, -20.0, 0.1, 1.0, 12)
    assert short - long_ == pytest.approx(12 / 3 - 12 / 12)
    # making the chosen text less likely (and not touching the rejected one) raises the loss
    assert dpo_loss.rpo_loss(-15.0, -20.0, -12.0, -20.0, 0.1, 1.0, 6) > dpo_loss.rpo_loss(-12.0, -20.0, -12.0, -20.0, 0.1, 1.0, 6)
    # the DPO part alone can be lowered by pushing the rejected text down; the NLL term does not allow the chosen to be pushed down for free
    only_dpo = dpo_loss.dpo_loss(-15.0, -40.0, -12.0, -20.0, 0.1)
    assert only_dpo < dpo_loss.dpo_loss(-12.0, -20.0, -12.0, -20.0, 0.1)
    assert dpo_loss.rpo_loss(-15.0, -40.0, -12.0, -20.0, 0.1, 1.0, 6) > only_dpo


def test_rpo_terms_and_meter():
    t = dpo_loss.rpo_terms(-4.0, -9.0, -5.0, -6.0, 0.1, 0.5, 8)
    assert t["loss"] == pytest.approx(t["dpo_loss"] + 0.5 * t["nll"]) and t["nll"] == pytest.approx(0.5)
    assert t["margin"] == pytest.approx(0.4) and t["correct"] is True
    m = dpo_loss.Meter()
    m.add(t)
    m.add(dpo_loss.rpo_terms(-6.0, -4.0, -5.0, -6.0, 0.1, 0.5, 4))
    s = m.summary()
    assert s["loss"] == pytest.approx(s["dpo_loss"] + 0.5 * s["nll"]) and s["reward_accuracy"] == 0.5
    # v1 terms (no NLL) still work in the same meter
    m1 = dpo_loss.Meter()
    m1.add(dpo_loss.dpo_terms(-4.0, -9.0, -5.0, -6.0, 0.1))
    assert m1.summary()["nll"] == 0.0 and m1.summary()["dpo_loss"] == m1.summary()["loss"]
    assert dpo_loss.Meter().summary()["nll"] is None


# ---- config and grid ----------------------------------------------------------------------------------------------------
def test_the_fixed_v2_design_and_the_batch_size_amendment():
    d = CFG4B["design"]
    assert d["effective_batch_size"] == 8 and d["micro_batch_size"] == 1 and d["beta"] == 0.1 and d["epochs"] == 3
    assert d["grid"] == {"pair_sets": ["v2-main", "v2-mix"], "alphas": [0.5, 1.0], "learning_rates": [1e-5, 2e-5]}
    assert d["repeat_seeds"] == [43, 44] and d["seed"] == 42 and d["grid_reduction"] == "none"
    assert dpo2_grid.validate(CFG) == []
    assert CFG["train"]["effective_batch_size"] == 8 and CFG["start"]["seeds"][42]["control"] == "lora_fp16_lr2e-4_e2"
    assert CFG["model"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct" and CFG["mlflow"]["experiment_name"] == "phase4b_dpo2"


@needs_training_data
def test_steps_per_epoch_follow_the_amendment():
    for ps, steps in (("v2-main", 10), ("v2-mix", 20)):
        lines = len(dpo2_train.read_set(CFG, ps))
        assert schedule.steps_per_epoch(lines, CFG["train"]["effective_batch_size"]) == steps
    assert schedule.steps_per_epoch(80, 8) * 3 == 30 and schedule.steps_per_epoch(160, 8) * 3 == 60


def test_validate_catches_broken_configs():
    def broken(path, value):
        c = copy.deepcopy(CFG)
        d = c
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        return dpo2_grid.validate(c)
    assert broken(["train", "effective_batch_size"], 16)
    assert broken(["design", "beta"], 0.3)
    assert broken(["design", "grid", "alphas"], [0.5])
    assert broken(["design", "grid", "learning_rates"], [1e-5, 5e-5])
    assert broken(["design", "grid", "pair_sets"], ["v2-main"])
    assert broken(["design", "repeat_seeds"], [42, 44])
    assert broken(["design", "repeat_seeds"], [43, 99])
    assert broken(["design", "grid_reduction"], "drop v2-mix")
    assert broken(["design", "selection", "order"], ["token_f1", "false_refusals"])


def test_grid_runs_names_and_start():
    runs = dpo2_grid.grid_runs(CFG)
    assert [r["name"] for r in runs] == [
        "dpo2_main_a0.5_lr1e-5", "dpo2_main_a0.5_lr2e-5", "dpo2_main_a1_lr1e-5", "dpo2_main_a1_lr2e-5",
        "dpo2_mix_a0.5_lr1e-5", "dpo2_mix_a0.5_lr2e-5", "dpo2_mix_a1_lr1e-5", "dpo2_mix_a1_lr2e-5"]
    assert all(r["seed"] == 42 and r["beta"] == 0.1 and r["start_adapter"].endswith("lora_fp16_lr2e-4/e2") for r in runs)
    assert all(r["control"] == "lora_fp16_lr2e-4_e2" for r in runs) and len(runs) * CFG["train"]["epochs"] == 24
    rep = dpo2_grid.repeat_runs(CFG, {"pair_set": "v2-mix", "alpha": 1.0, "lr": 2e-5})
    assert [r["name"] for r in rep] == ["dpo2_mix_a1_lr2e-5_s43", "dpo2_mix_a1_lr2e-5_s44"]
    assert [r["control"] for r in rep] == ["lora_fp16_lr2e-4_s43_e2", "lora_fp16_lr2e-4_s44_e2"]
    assert [r["start_adapter"] for r in rep] == ["outputs/adapters/lora_fp16_lr2e-4_s43/e2", "outputs/adapters/lora_fp16_lr2e-4_s44/e2"]
    assert grid.candidate_name("dpo2_main_a0.5_lr1e-5", 3) == "dpo2_main_a0.5_lr1e-5_e3"
    assert not {r["name"] for r in runs} & {"dpo_b0.1_lr5e-5", "dpo_b0.3_lr2e-4"}                  # no clash with v1 names


def test_winner_run_find_run_and_the_reference_cache_per_set():
    sel = {"winner": {"candidate": "dpo2_mix_a0.5_lr1e-5_e2", "pair_set": "v2-mix", "alpha": 0.5, "lr": 1e-5}}
    w = dpo2_grid.winner_run(CFG, sel)
    assert w["name"] == "dpo2_mix_a0.5_lr1e-5" and w["seed"] == 42
    assert dpo2_grid.find_run("dpo2_mix_a0.5_lr1e-5_s44", CFG, sel)["seed"] == 44
    with pytest.raises(SystemExit, match="unknown run"):
        dpo2_grid.find_run("dpo2_mix_a0.5_lr1e-5_s44", CFG, None)
    with pytest.raises(SystemExit, match="no winner"):
        dpo2_grid.winner_run(CFG, {"winner": None})
    a, b = dpo2_grid.grid_runs(CFG)[0], dpo2_grid.grid_runs(CFG)[1]
    c = dpo2_grid.grid_runs(CFG)[4]
    assert dpo2_grid.reference_cache_path(CFG, a) == dpo2_grid.reference_cache_path(CFG, b)                  # same set, same seed: shared
    assert dpo2_grid.reference_cache_path(CFG, a).name == "reference_logps_main_seed42.json"
    assert dpo2_grid.reference_cache_path(CFG, c).name == "reference_logps_mix_seed42.json"
    assert "runs_dpo2" in str(dpo2_grid.reference_cache_path(CFG, a))


def test_v2_files_do_not_overlap_with_v1_or_phase3():
    p3 = CFG3["paths"]
    for k in ("adapters_dir", "runs_dir", "selection_json", "selection_sha", "test_log", "final_md"):
        assert CFG["paths"][k] != p3.get(k)
    from climate_slm.config import load_phase4_config
    p4 = load_phase4_config()["paths"]
    for k in ("adapters_dir", "runs_dir", "selection_json", "selection_sha", "test_log", "final_md"):
        assert CFG["paths"][k] != p4[k], k


# ---- pair files and the trainer -----------------------------------------------------------------------------------------
@needs_training_data
def test_pair_sets_are_read_after_the_sha_check_and_have_the_stage_a_shape():
    main, mix = dpo2_train.read_set(CFG, "v2-main"), dpo2_train.read_set(CFG, "v2-mix")
    assert len(main) == 80 and all(p["pair_type"] == 3 for p in main)
    assert len(mix) == 160 and sum(1 for p in mix if p["pair_type"] == 1) == sum(1 for p in mix if p["pair_type"] == 3) == 80
    assert len({p["pair_id"] for p in mix}) == 160 and len(dpo_data.distinct_pairs(mix)) == 160
    assert dpo2_train.set_sha256(CFG, "v2-main") == "3713d8804280f51cf6044e486c07d3700cadd3b57ca056ac1d138552ed26e114"
    # the published copy has 12 source-document e-mail addresses and 1 local file link replaced by placeholders;
    # the file used in training had sha256 10913e7478f3bf6f74ccc26e67b7ceff06e86ea0a954cbf89e8785c91aef5924
    assert dpo2_train.set_sha256(CFG, "v2-mix") == "3d2cad31c10aac849e5c2f26841c41b9f01aa6fdc65c61d643c957a69a5857b6"
    assert {p["pair_id"] for p in main} <= {p["pair_id"] for p in mix}                                    # mix = main + type 1 lines


def test_trainer_structure_without_gpu():
    src = (ROOT / "climate_slm" / "training" / "dpo2_train.py").read_text(encoding="utf-8")
    for lib in ("torch", "transformers", "peft", "bitsandbytes"):
        assert f"\nimport {lib}" not in src.split("def train_one(")[0] and f"\nfrom {lib}" not in src.split("def train_one(")[0]
    assert "sft_val" not in src and "sft_test" not in src and "load_items" not in src and '"test"' not in src and "--split" not in src
    assert src.index("compute_references(model") < src.index('for epoch in range(1, t["epochs"] + 1)')       # reference BEFORE training
    assert "rpo_loss(pc, pr, rc, rr, beta, alpha" in src and "logsigmoid=F.logsigmoid" in src and "rpo_terms(" in src
    assert 'e["chosen"]["n_completion"]' in src                                                              # per-token NLL of the chosen text
    assert "dpo2_grid.reference_cache_path(cfg, run)" in src and "dpo_data.reference_for(refs, x)" in src
    assert "torch.amp.GradScaler" in src and "clip_grad_norm_" in src and "schedule.lr_at_step" in src and "check_start_adapter" in src
    assert "model.save_pretrained(str(out_dir))" in src and src.count("grid.clear_stale_error") == 3
    assert src.count("error_file_path") == 1 and "mlflow_log.start_tracker" in src and "read_set(cfg, run[\"pair_set\"])" in src
    v1 = (ROOT / "climate_slm" / "training" / "dpo_train.py").read_text(encoding="utf-8")
    assert "rpo_loss" not in v1 and "nll" not in v1                                                          # the v1 trainer is untouched


def test_check_inputs_real_and_with_a_missing_pair_file(capsys, monkeypatch, tmp_path):
    if (ROOT / CFG["start"]["seeds"][44]["adapter"] / dpo_grid_file()).exists():
        assert dpo2_train.check_inputs(CFG, CFG3) == []
        out = capsys.readouterr().out
        assert "v2-main: pair file ok" in out and "30 per run" in out and "60 per run" in out and "all checks passed" in out
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["main_sha256"] = str(tmp_path / "missing.sha256")
    problems = dpo2_train.check_inputs(cfg, CFG3)
    assert any(p.startswith("v2-main:") for p in problems)


def dpo_grid_file():
    from climate_slm.training import dpo_grid
    return dpo_grid.ADAPTER_FILE


def test_group_skip_clears_the_stale_error_and_a_failed_run_keeps_it(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["runs_dir"], cfg["paths"]["adapters_dir"] = str(tmp_path / "runs"), str(tmp_path / "ad")
    monkeypatch.setattr(grid, "REPO_ROOT", tmp_path)
    run = dpo2_grid.grid_runs(cfg)[0]
    log = grid.run_log_path(cfg, run["name"])
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"complete": True}), encoding="utf-8")
    for e in grid.epochs(cfg):
        d = grid.adapter_dir(cfg, run["name"], e)
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    err = grid.error_file_path(cfg, run["name"])
    err.write_text("old failure", encoding="utf-8")
    assert dpo2_train.run_group([run], cfg) == [] and not err.exists()
    log.write_text(json.dumps({"complete": False}), encoding="utf-8")
    err.write_text("still failing", encoding="utf-8")
    monkeypatch.setattr(dpo2_train.subprocess, "call", lambda cmd: 1)
    assert dpo2_train.run_group([run], cfg) == [run["name"]] and err.exists()


def test_resolve_runs():
    assert len(dpo2_train.resolve_runs("grid", CFG)) == 8
    with pytest.raises(SystemExit):
        dpo2_train.resolve_runs("repeats", {**CFG, "paths": {**CFG["paths"], "selection_json": "outputs/does_not_exist_phase4b.json"}})


# ---- eval loop and test gate ----------------------------------------------------------------------------------------------
def test_val_loop_is_val_only_and_plans_the_24_adapters():
    assert dpo2_eval.ALLOWED_SPLITS == ("val",)
    with pytest.raises(SystemExit):
        dpo2_eval.main(["--group", "grid", "--split", "test"])
    src = (ROOT / "climate_slm" / "training" / "dpo2_eval.py").read_text(encoding="utf-8")
    assert '"test"' not in src and "--split test" not in src
    items = dpo2_eval.plan(dpo2_grid.grid_runs(CFG), CFG, ECFG, "val")
    assert len(items) == 24 and items[0]["name"] == "dpo2_main_a0.5_lr1e-5_e1" and items[-1]["name"] == "dpo2_mix_a1_lr2e-5_e3"
    assert items[0]["eval_file"].name == "eval_dpo2_main_a0.5_lr1e-5_e1_val.json"
    assert all(i["eval_file"].name.endswith("_val.json") and "adapters_dpo2" in str(i["adapter"]) for i in items)
    assert all(i["base_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct" for i in items)


def _world(tmp_path, monkeypatch, winner=True, tested=False, frozen=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(CFG)
    for k in ("selection_json", "selection_sha", "test_log"):
        cfg["paths"][k] = str(tmp_path / Path(cfg["paths"][k]).name)
    cfg["paths"]["adapters_dir"] = str(tmp_path / "ad")
    for m in (grid, selection_io, dpo_final_test):
        monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    sel = {"winner": {"candidate": "dpo2_mix_a1_lr2e-5_e2", "run": "dpo2_mix_a1_lr2e-5", "epoch": 2, "pair_set": "v2-mix", "alpha": 1.0,
                      "beta": 0.1, "lr": 2e-5, "seed": 42} if winner else None}
    sp = tmp_path / Path(cfg["paths"]["selection_json"]).name
    sp.write_text(json.dumps(sel), encoding="utf-8")
    if frozen:
        selection_io.sha_path(cfg).write_text(f"{selection_io.file_sha256(sp)}  x\n", encoding="utf-8")
    d = grid.adapter_dir(cfg, "dpo2_mix_a1_lr2e-5", 2)
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dpo_final_test, "default_out", lambda c, n, s: tmp_path / f"eval_{n}_{s}.json")
    if tested:
        (tmp_path / "eval_dpo2_mix_a1_lr2e-5_e2_test.json").write_text("{}", encoding="utf-8")
    return cfg


def test_final_test_gate_for_v2(tmp_path, monkeypatch):
    cfg = _world(tmp_path, monkeypatch)
    w = dpo_final_test.check_can_run(cfg, ECFG, select_module="select_dpo2")
    cmd = dpo_final_test.test_command(cfg, w)
    assert cmd[cmd.index("--split") + 1] == "test" and cmd[cmd.index("--name") + 1] == "dpo2_mix_a1_lr2e-5_e2"
    assert cmd[cmd.index("--base-model-id") + 1] == "Qwen/Qwen2.5-0.5B-Instruct" and "--load-4bit" not in cmd
    cfg = _world(tmp_path / "unfrozen", monkeypatch, frozen=False)
    with pytest.raises(SystemExit, match="select_dpo2 --freeze"):
        dpo_final_test.check_can_run(cfg, ECFG, select_module="select_dpo2")
    cfg = _world(tmp_path / "nowinner", monkeypatch, winner=False)
    with pytest.raises(SystemExit, match="no winner"):
        dpo_final_test.check_can_run(cfg, ECFG, select_module="select_dpo2")
    cfg = _world(tmp_path / "tested", monkeypatch, tested=True)
    with pytest.raises(SystemExit, match="already evaluated on test"):
        dpo_final_test.check_can_run(cfg, ECFG, select_module="select_dpo2")


def test_final_test_main_records_once(tmp_path, monkeypatch):
    cfg = _world(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(dpo2_final_test.v1.subprocess, "call", lambda cmd, cwd=None: 0)
    monkeypatch.setattr(dpo2_final_test.subprocess, "call", lambda cmd, cwd=None: calls.append(cmd) or 0)
    monkeypatch.setattr(dpo2_final_test.dpo2_grid, "runtime_config", lambda: cfg)
    monkeypatch.setattr(dpo2_final_test, "REPO_ROOT", tmp_path)
    assert dpo2_final_test.main([]) == 0 and len(calls) == 1
    log = json.loads(Path(cfg["paths"]["test_log"]).read_text(encoding="utf-8"))
    assert [e["candidate"] for e in log] == ["dpo2_mix_a1_lr2e-5_e2"] and len(log[0]["selection_sha256"]) == 64
    with pytest.raises(SystemExit, match="already evaluated on test"):
        dpo2_final_test.main([])
    assert len(calls) == 1
    assert Path(cfg["paths"]["test_log"]).name == "phase4b_test_runs.json"


def test_only_the_final_test_module_names_the_test_split():
    for mod in ("dpo2_train", "dpo2_eval", "select_dpo2", "dpo2_grid"):
        src = (ROOT / "climate_slm" / "training" / f"{mod}.py").read_text(encoding="utf-8")
        assert '"--split", "test"' not in src and "--split test" not in src, mod
    assert '"--split", "test"' not in (ROOT / "climate_slm" / "training" / "dpo2_final_test.py").read_text(encoding="utf-8")   # uses the v1 command
    assert '"--split", "test"' in (ROOT / "climate_slm" / "training" / "dpo_final_test.py").read_text(encoding="utf-8")


# ---- the v2 selection rule --------------------------------------------------------------------------------------------------------
def _cand(name, epoch=1, correct=8, fr=2, f1=0.65, nf=1.0, peak=5000.0, secs=100.0, chars=100, hit=0, loops=0,
          pair_set="v2-main", alpha=0.5, lr=1e-5):
    def rate(k, n):
        return {"count": k, "n": n, "rate": k / n, "wilson95": list(M.wilson_ci(k, n))}
    m = {"n_answerable": 42, "n_refusal": 11, "token_f1": {"mean": f1, "ci95": [f1 - 0.05, f1 + 0.05], "n": 42},
         "key_recall": None, "key_support": None, "number_faithfulness": {"mean": nf, "ci95": [nf, nf], "n_applied": 14},
         "false_refusals": rate(fr, 42), "correct_strict": rate(correct, 11), "correct_loose": rate(correct, 11),
         "answered_when_should_refuse": rate(11 - correct, 11), "refusal_perplexity": 1.1, "hit_max_new_tokens": hit,
         "repetition_loops": loops, "output_chars": {"median": chars, "max": chars * 2}}
    return {"candidate": name, "run": name, "epoch": epoch, "pair_set": pair_set, "alpha": alpha, "beta": 0.1, "lr": lr, "seed": 42,
            "metrics": m, "cost": {"peak_gpu_mem_mb": peak, "train_seconds": secs}, "train_epoch": None}


CONTROL = _cand("sft", correct=8, fr=3, f1=0.615, chars=144)["metrics"]
SEL = CFG["selection"]


def test_every_filter_line_of_the_v2_rule():
    r = select_dpo2.drop_reasons
    assert r(_cand("ok"), CONTROL, SEL) == []
    assert r(_cand("a", correct=6), CONTROL, SEL)                                       # < 7/11
    assert r(_cand("b", correct=7), CONTROL, SEL) == []                                 # exactly 7 stays
    assert r(_cand("c", chars=216), CONTROL, SEL) == [] and r(_cand("d", chars=217), CONTROL, SEL)   # 1.5 x 144 = 216
    assert r(_cand("e", hit=2), CONTROL, SEL) and r(_cand("f", hit=1), CONTROL, SEL) == []
    assert r(_cand("g", loops=1), CONTROL, SEL) and r(_cand("h", loops=0), CONTROL, SEL) == []   # control has 0 loops
    ctl_loop = copy.deepcopy(CONTROL)
    ctl_loop["repetition_loops"] = 1
    assert r(_cand("i", loops=1), ctl_loop, SEL) == [] and r(_cand("j", loops=2), ctl_loop, SEL)       # "beyond the control"
    assert r(_cand("k", f1=0.574), CONTROL, SEL) and r(_cand("l", f1=0.575), CONTROL, SEL) == []      # F1 floor 0.575
    assert len(r(_cand("m", correct=5, chars=300, hit=3, loops=2, f1=0.5), CONTROL, SEL)) == 5


def test_winner_needs_fewer_false_refusals_than_the_control():
    cands = [_cand("same", fr=3), _cand("better", fr=2, correct=7), _cand("best_but_low_f1", fr=0, f1=0.5),
             _cand("best_but_few_refusals", fr=0, correct=5)]
    r = select_dpo2.select_winner(cands, CONTROL, CFG)
    assert {d["candidate"] for d in r["dropped"]} == {"best_but_low_f1", "best_but_few_refusals"}
    assert r["ranked"] == ["better", "same"] and r["winner"]["candidate"] == "better" and r["improves_on_control"] is True
    assert {"candidate", "run", "epoch", "pair_set", "alpha", "beta", "lr", "seed"} <= set(r["winner"])
    assert r["deciding_step"] == "false_refusals"


@pytest.mark.parametrize("better,worse,step", [
    (dict(fr=1), dict(fr=2), "false_refusals"),
    (dict(correct=9), dict(correct=8), "correct_refusals_strict"),
    (dict(f1=0.7), dict(f1=0.6), "token_f1"),
    (dict(nf=1.0), dict(nf=0.9), "number_faithfulness"),
    (dict(peak=4000.0), dict(peak=5000.0), "peak_gpu_mem"),
    (dict(secs=50.0), dict(secs=60.0), "train_seconds"),
    (dict(epoch=1), dict(epoch=2), "epoch")])
def test_each_ranking_step_decides_when_the_earlier_ones_tie(better, worse, step):
    base = dict(correct=8, fr=1)
    r = select_dpo2.select_winner([_cand("worse", **{**base, **worse}), _cand("better", **{**base, **better})], CONTROL, CFG)
    assert r["winner"]["candidate"] == "better" and r["deciding_step"] == step


def test_no_candidate_improves_and_other_empty_branches():
    r = select_dpo2.select_winner([_cand("a", fr=3), _cand("b", fr=5), _cand("c", fr=0, f1=0.4)], CONTROL, CFG)
    assert r["winner"] is None and r["ranked"] == ["a", "b"] and r["improves_on_control"] is False
    assert select_dpo2.NO_IMPROVEMENT in r["status"] and "has 3 false refusals" in r["status"]
    assert select_dpo2.NO_IMPROVEMENT == "no DPO v2 candidate improves on the SFT model on val"
    none_left = select_dpo2.select_winner([_cand("c", correct=3, fr=0)], CONTROL, CFG)
    assert none_left["winner"] is None and "passes the filter" in none_left["status"]
    nothing = select_dpo2.select_winner([], CONTROL, CFG)
    assert nothing["winner"] is None and nothing["status"].startswith("not yet measured") and select_dpo2.NO_IMPROVEMENT not in nothing["status"]
    assert "control val file is missing" in select_dpo2.select_winner([_cand("a")], None, CFG)["status"]


def test_the_real_control_matches_the_numbers_in_the_brief():
    p = ROOT / "outputs" / "eval_lora_fp16_lr2e-4_e2_val.json"
    if not p.exists():
        pytest.skip("control val file is not on this machine")
    m = select_dpo.sb.load_metrics(ECFG, CFG3, "lora_fp16_lr2e-4_e2", "val")
    assert (m["correct_strict"]["count"], m["false_refusals"]["count"], round(m["token_f1"]["mean"], 3)) == (8, 3, 0.615)
    assert m["false_refusals"]["count"] == CFG4B["design"]["selection"]["must_beat_control_false_refusals"]
    assert select_dpo.median_chars(m) == 144 and 1.5 * 144 == 216


def _write_eval(tmp_path, name, split, fr, correct, f1=0.6):
    def rate(k, n):
        return {"count": k, "n": n, "rate": k / n, "ci95": [0.0, 1.0], "wilson95": list(M.wilson_ci(k, n))}
    mean = {"n": 42, "mean": f1, "ci95": [f1 - 0.1, f1 + 0.1]}
    ev = {"run_name": name, "split": split, "summary": {
        "n_items": 53, "answerable": {"n": 42, "token_f1": mean, "key_recall": {**mean, "n_applied": 40, "n_answerable": 42},
                                      "key_support": {**mean, "n_applied": 40, "n_answerable": 42},
                                      "number_faithfulness": {"n": 14, "n_applied": 14, "mean": 1.0, "ci95": [1.0, 1.0]},
                                      "false_refusal_strict": rate(fr, 42)},
        "refusal": {"n": 11, "correct_refusal_strict": rate(correct, 11), "correct_refusal_loose": rate(correct, 11),
                    "answered_when_should_refuse": rate(11 - correct, 11)},
        "perplexity": {"refusal": {"perplexity": 1.1}}, "hit_max_new_tokens": {"count": 0}, "output_length_chars": {"median": 120, "max": 200}},
        "items": [{"output": "fine answer here", "key_recall": 0.5, "key_support": 0.5}]}
    (tmp_path / f"eval_{name}_{split}.json").write_text(json.dumps(ev), encoding="utf-8")


def _fake_world(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    monkeypatch.setattr(select_dpo2.sb, "default_out", lambda c, n, s: tmp_path / f"eval_{n}_{s}.json")
    monkeypatch.setattr(grid, "REPO_ROOT", tmp_path)
    for sd, fr, cr in ((42, 3, 8), (43, 0, 7), (44, 1, 8)):
        _write_eval(tmp_path, cfg["start"]["seeds"][sd]["control"], "val", fr, cr, 0.615)
    _write_eval(tmp_path, "base", "val", 0, 0, 0.575)
    return cfg


def test_collect_and_documents_from_saved_files(tmp_path, monkeypatch):
    cfg = _fake_world(tmp_path, monkeypatch)
    _write_eval(tmp_path, "dpo2_mix_a1_lr2e-5_e2", "val", 1, 8)
    _write_eval(tmp_path, "dpo2_mix_a1_lr2e-5_e1", "val", 3, 8)
    _write_eval(tmp_path, "dpo2_mix_a1_lr2e-5_s43_e2", "val", 0, 8)
    log = {"complete": True, "peak_gpu_mem_mb": 5000.0, "train_seconds": 170.0, "trainable_params": 8798208, "dropped_distinct_ids": [],
           "dropped_lines": 0, "pair_counts": {"lines": 160, "distinct": 160},
           "epochs": [{"epoch": e, "loss": 1.0 - 0.1 * e, "dpo_loss": 0.6 - 0.1 * e, "nll": 0.8 - 0.1 * e, "reward_accuracy": 0.5 + 0.1 * e,
                       "mean_margin": 0.1 * e, "adapter_mb": 33.6} for e in (1, 2, 3)]}
    (tmp_path / "runs" / "dpo2_mix_a1_lr2e-5").mkdir(parents=True)
    (tmp_path / "runs" / "dpo2_mix_a1_lr2e-5" / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    data = select_dpo2.collect(cfg, CFG3, ECFG)
    assert len(data["candidates"]) == 2 and data["control"]["val"]["false_refusals"]["count"] == 3
    w = data["winner"]
    assert w["candidate"] == "dpo2_mix_a1_lr2e-5_e2" and (w["pair_set"], w["alpha"], w["lr"]) == ("v2-mix", 1.0, 2e-5)
    assert data["candidates"][0]["train_epoch"] == {"dpo_loss": pytest.approx(0.5), "reward_accuracy": pytest.approx(0.6),
                                                    "mean_margin": pytest.approx(0.1), "loss": pytest.approx(0.9), "nll": pytest.approx(0.7)}
    p = data["paired"]
    assert p["seeds_paired"] == [42, 43] and p["counts"]["false_refusals"] == {"better": 1, "equal": 1, "worse": 0}
    assert data["val_candidates_tried"] == {"v1": 12, "v2": 24, "total": 36}
    md = select_dpo2.selection_md(data, cfg)
    assert "**WINNER**" in md and "SFT control, no DPO" in md and "v1 12 + v2 24 = 36" in md and "no longer a clean check" in md
    assert "NLL part" in md and "Paired seeds on val" in md and "not yet measured" in md and "pair set" in md
    pl = select_dpo2.payload(data)
    assert "test" not in pl["control"] and "test" not in pl["tuned"]
    assert dpo2_grid.winner_run(cfg, pl)["name"] == "dpo2_mix_a1_lr2e-5"
    assert [r["name"] for r in dpo2_grid.repeat_runs(cfg, dpo2_grid.winner_run(cfg, pl))] == ["dpo2_mix_a1_lr2e-5_s43", "dpo2_mix_a1_lr2e-5_s44"]


def test_no_winner_document_says_sft_stays(tmp_path, monkeypatch):
    cfg = _fake_world(tmp_path, monkeypatch)
    _write_eval(tmp_path, "dpo2_main_a0.5_lr1e-5_e1", "val", 4, 8)
    data = select_dpo2.collect(cfg, CFG3, ECFG)
    assert data["winner"] is None and select_dpo2.NO_IMPROVEMENT in data["status"] and data["paired"] is None
    md = select_dpo2.selection_md(data, cfg)
    assert "no DPO v2 candidate improves on the SFT model on val" in md and "SFT stays" in md and "no seed repeat" in md
    with pytest.raises(SystemExit, match="no winner"):
        dpo2_grid.winner_run(cfg, select_dpo2.payload(data))
    empty = select_dpo2.collect(_fake_world(tmp_path, monkeypatch), CFG3, ECFG)                    # nothing evaluated yet
    # (the fake world above already holds one eval file, so build the 'not yet measured' text from a copy without candidates)
    empty = {**empty, "candidates": [], "dropped": [], "ranked": [], "status": "not yet measured: no DPO v2 candidate has been evaluated on val"}
    md2 = select_dpo2.selection_md(empty, cfg)
    assert "Nothing can be decided yet" in md2 and "SFT stays" not in md2


def test_selection_main_freezes_and_does_not_rewrite(tmp_path, monkeypatch, capsys):
    cfg = _fake_world(tmp_path, monkeypatch)
    for k in ("selection_json", "selection_md", "selection_sha", "final_md", "test_log"):
        cfg["paths"][k] = str(tmp_path / Path(cfg["paths"][k]).name)
    monkeypatch.setattr(select_dpo2.dpo2_grid, "runtime_config", lambda: cfg)
    monkeypatch.setattr(select_dpo2, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(selection_io, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(select_dpo.selection_io, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="not frozen"):
        select_dpo2.main(["--final-tables"])
    assert select_dpo2.main(["--freeze"]) == 0
    sp = Path(cfg["paths"]["selection_json"])
    sha = selection_io.file_sha256(sp)
    assert Path(cfg["paths"]["selection_sha"]).read_text(encoding="utf-8").split()[0] == sha
    capsys.readouterr()
    assert select_dpo2.main([]) == 0 and "frozen" in capsys.readouterr().out and selection_io.file_sha256(sp) == sha
    assert select_dpo2.main(["--final-tables"]) == 0
    assert "Phase 4b (DPO v2) final tables" in Path(cfg["paths"]["final_md"]).read_text(encoding="utf-8")


# ---- collect sets, bundle, notebook -------------------------------------------------------------------------------------------------------
def test_phase4b_collect_sets_do_not_mix_with_earlier_phases(tmp_path):
    out = tmp_path / "outputs"
    rels = ("phase4b_selection.json", "phase4b_selection.md", "phase4b_selection.sha256", "eval_dpo2_mix_a1_lr2e-5_e1_val.json",
            "refusal_detection_review_dpo2_mix_a1_lr2e-5_e1_val.csv", "runs_dpo2/dpo2_mix_a1_lr2e-5/train_log.json",
            "runs_dpo2/reference_logps_mix_seed42.json", "mlruns/0/meta.yaml", "adapters_dpo2/dpo2_mix_a1_lr2e-5/e1/adapter_model.safetensors",
            "phase4_selection.json", "eval_dpo_b0.1_lr5e-5_e1_val.json", "runs_dpo/x/train_log.json", "adapters_dpo/x/e1/adapter_model.safetensors",
            "phase3_selection.json")
    for rel in rels:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text("x", encoding="utf-8")
    res = {p.relative_to(out).as_posix() for p in collect.phase4b_result_files(out, ECFG)}
    assert res == {"phase4b_selection.json", "phase4b_selection.md", "phase4b_selection.sha256", "eval_dpo2_mix_a1_lr2e-5_e1_val.json",
                   "refusal_detection_review_dpo2_mix_a1_lr2e-5_e1_val.csv", "runs_dpo2/dpo2_mix_a1_lr2e-5/train_log.json",
                   "runs_dpo2/reference_logps_mix_seed42.json", "mlruns/0/meta.yaml"}
    assert [p.relative_to(out).as_posix() for p in collect.phase4b_adapter_files(out, ECFG)] == [
        "adapters_dpo2/dpo2_mix_a1_lr2e-5/e1/adapter_model.safetensors"]
    v1 = {p.relative_to(out).as_posix() for p in collect.phase4_result_files(out, ECFG)}
    assert not any("dpo2" in x or "phase4b" in x for x in v1)                                  # v1 zip does not take v2 files
    assert not any("dpo2" in p.as_posix() for p in collect.phase4_adapter_files(out, ECFG))
    assert not any("dpo2" in p.name or "phase4b" in p.name for p in collect.phase3_result_files(out, ECFG))


def test_bundle_and_report_config_know_v2():
    b = ECFG["bundle"]["files"]
    for f in ("data/instruction/dpo2_main.jsonl", "data/instruction/dpo2_main.sha256", "data/instruction/dpo2_mix.jsonl",
              "data/instruction/dpo2_mix.sha256"):
        assert f in b, f
    assert "dpo2_*" in ECFG["report"]["hide_run_patterns"] and "dpo_*" in ECFG["report"]["hide_run_patterns"]
    z = ROOT / ECFG["paths"]["colab_bundle"]
    if z.exists():
        names = zipfile.ZipFile(z).namelist()
        for n in ("climate_slm/training/dpo2_train.py", "climate_slm/training/dpo2_eval.py", "climate_slm/training/select_dpo2.py",
                  "climate_slm/training/dpo2_final_test.py", "climate_slm/training/dpo2_grid.py", "configs/phase4b.yaml",
                  "data/instruction/dpo2_main.jsonl", "data/instruction/dpo2_mix.jsonl", "data/instruction/dpo2_main.sha256"):
            assert n in names, n


def _cells():
    nb = json.loads((ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_phase4b_cells():
    code = _cells()
    p4b = [c for c in code if c.lstrip().startswith("# P4b-")]
    ids = [c.split(".")[0].replace("# P4b-", "") for c in p4b]
    assert ids == [str(i) for i in range(11)]                                             # P4b-0 (stage A) + P4b-1 .. P4b-10 (stage B)
    by = dict(zip(ids, p4b))
    assert all(len(c.splitlines()) <= 15 for c in p4b)
    assert {i for i, c in by.items() if "test" in c.lower()} == {"9"}                     # only the one clearly labelled cell
    assert "FINAL TEST" in by["9"] and "dpo2_final_test" in by["9"]
    assert [c for c in code if "dpo2_final_test" in c] == [by["9"]]
    assert [c for c in code if "dpo2_generate" in c] == [by["0"]]
    for i in ("3", "4", "5", "6", "7", "8"):
        assert "dpo2_final_test" not in by[i] and "--split test" not in by[i]
    assert "--split val" in by["4"] and "--split val" in by["7"]
    assert "dpo2_train --group grid" in by["3"] and "dpo2_train --group repeats" in by["6"]
    assert "dpo2_eval --group grid" in by["4"] and "dpo2_eval --group repeats" in by["7"]
    assert "select_dpo2" in by["5"] and "--freeze" not in by["5"] and "select_dpo2 --freeze" in by["8"]
    assert "dpo2_train --check" in by["2"] and "phase4_start.zip" in by["2"] and "torchao" in by["1"]
    assert "phase4b-results" in by["10"] and "phase4b-adapters" in by["10"]
    order = [code.index(by[str(i)]) for i in range(11)]
    assert order == sorted(order) and code.index(by["9"]) > code.index(by["8"]) > code.index(by["6"]) > code.index(by["5"]) > code.index(by["3"])
    old = [c for c in code if not c.startswith("# P4b-")]
    assert not any("dpo2" in c or "select_dpo2" in c for c in old)                       # v1 and Phase 3 cells are unchanged in content


def test_v1_cells_are_still_the_eleven_old_ones():
    code = _cells()
    p4 = [c for c in code if c.lstrip().startswith("# P4-")]
    assert [c.split(".")[0].replace("# P4-", "") for c in p4] == [str(i) for i in range(11)]
    assert [c for c in code if "training.dpo_final_test" in c and c.startswith("# P4-")] == [p4[9]]


@needs_readme
def test_readme_colab_has_the_phase4b_cell_order():
    guide = ROOT / "notebooks" / "README_COLAB.md"                               # the notebook guide is optional in a copy
    if guide.exists():
        text = guide.read_text(encoding="utf-8")
        for i in range(11):
            assert f"| P4b-{i} |" in text, i
        assert "Phase 4b: DPO v2 (stage B)" in text and "time (estimate)" in text and "78 of the 80" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "78 of the 80 rejected texts came from the untuned model" in readme and "36 validation candidates" in readme
    banned = [bytes.fromhex(h).decode() for h in ("636c61756465", "616e7468726f706963")]   # the two assistant/vendor names, kept out of the text
    assert not any(w in readme.lower() for w in banned)
