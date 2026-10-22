"""Tests for Phase 4, stage B: DPO loss, grid, pair encoding, reference cache, trainer structure, selection, test gate,
collect sets, notebook cells. CPU only. No torch, transformers, peft."""
import copy
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config, load_phase4_config
from climate_slm.evaluation import collect
from climate_slm.evaluation import metrics as M
from climate_slm.training import (dpo_data, dpo_eval, dpo_final_test, dpo_grid, dpo_loss, dpo_pairs, dpo_train, grid,
                                  select_dpo, selection_io)
from conftest import needs
from _release import needs_training_data

CFG4, CFG3, ECFG = load_phase4_config(), load_phase3_config(), load_eval_config()
ROOT = REPO_ROOT


# ---- the loss ----------------------------------------------------------------------------------------------------------
def test_dpo_loss_known_values():
    assert dpo_loss.dpo_loss(-5.0, -6.0, -5.0, -6.0, 0.1) == pytest.approx(math.log(2))          # policy == reference
    z = 0.1 * ((-4.0 - -5.0) - (-9.0 - -6.0))                                                      # 0.1 * (1 + 3) = 0.4
    assert dpo_loss.dpo_loss(-4.0, -9.0, -5.0, -6.0, 0.1) == pytest.approx(-math.log(1 / (1 + math.exp(-z))))
    assert dpo_loss.dpo_loss(-4.0, -9.0, -5.0, -6.0, 0.1) < math.log(2)                           # moved the right way


def test_dpo_loss_symmetry_and_beta():
    a = dpo_loss.dpo_loss(-4.0, -9.0, -5.0, -6.0, 0.3)
    b = dpo_loss.dpo_loss(-9.0, -4.0, -6.0, -5.0, 0.3)                                            # chosen and rejected swapped
    z = 0.3 * 4.0
    assert b - a == pytest.approx(z)                                                               # loss(-z) - loss(z) = z
    # the same log-prob move is rewarded more at a larger beta
    assert dpo_loss.dpo_loss(-4.0, -9.0, -5.0, -6.0, 0.3) < dpo_loss.dpo_loss(-4.0, -9.0, -5.0, -6.0, 0.1)
    # and a wrong move is punished more
    assert dpo_loss.dpo_loss(-6.0, -4.0, -5.0, -6.0, 0.3) > dpo_loss.dpo_loss(-6.0, -4.0, -5.0, -6.0, 0.1) > math.log(2)


def test_logsigmoid_is_stable_and_the_loss_takes_a_custom_logsigmoid():
    assert dpo_loss.logsigmoid(1000.0) == pytest.approx(0.0, abs=1e-12)
    assert dpo_loss.logsigmoid(-1000.0) == pytest.approx(-1000.0)
    assert dpo_loss.logsigmoid(0.0) == pytest.approx(-math.log(2))
    assert dpo_loss.dpo_loss(1.0, 0.0, 0.0, 0.0, 1.0, logsigmoid=lambda x: -x) == -(-1.0)           # injected function is used


def test_terms_margin_accuracy_and_meter():
    t = dpo_loss.dpo_terms(-4.0, -9.0, -5.0, -6.0, 0.1)
    assert t["reward_chosen"] == pytest.approx(0.1) and t["reward_rejected"] == pytest.approx(-0.3)
    assert t["margin"] == pytest.approx(0.4) and t["correct"] is True
    assert dpo_loss.dpo_terms(-5.0, -6.0, -5.0, -6.0, 0.1)["correct"] is False                    # tie is not "correct"
    m = dpo_loss.Meter()
    assert m.summary()["loss"] is None
    for terms in (t, dpo_loss.dpo_terms(-6.0, -4.0, -5.0, -6.0, 0.1)):
        m.add(terms)
    s = m.summary()
    assert s["n_pairs"] == 2 and s["reward_accuracy"] == 0.5
    assert s["mean_margin"] == pytest.approx((0.4 + 0.1 * (-1 - 2)) / 2)
    assert s["loss"] == pytest.approx((t["loss"] + dpo_loss.dpo_loss(-6.0, -4.0, -5.0, -6.0, 0.1)) / 2)


# ---- config and grid ----------------------------------------------------------------------------------------------------
def test_grid_names_and_runs():
    runs = dpo_grid.grid_runs(CFG4)
    assert [r["name"] for r in runs] == ["dpo_b0.1_lr5e-5", "dpo_b0.1_lr2e-4", "dpo_b0.3_lr5e-5", "dpo_b0.3_lr2e-4"]
    assert all(r["seed"] == 42 and r["start_adapter"].endswith("lora_fp16_lr2e-4/e2") for r in runs)
    assert all(r["control"] == "lora_fp16_lr2e-4_e2" and r["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct" for r in runs)
    rep = dpo_grid.repeat_runs(CFG4, {"beta": 0.3, "lr": 2e-4})
    assert [r["name"] for r in rep] == ["dpo_b0.3_lr2e-4_s43", "dpo_b0.3_lr2e-4_s44"]
    assert [r["start_adapter"] for r in rep] == ["outputs/adapters/lora_fp16_lr2e-4_s43/e2", "outputs/adapters/lora_fp16_lr2e-4_s44/e2"]
    assert [r["control"] for r in rep] == ["lora_fp16_lr2e-4_s43_e2", "lora_fp16_lr2e-4_s44_e2"]
    assert grid.candidate_name("dpo_b0.1_lr5e-5", 2) == "dpo_b0.1_lr5e-5_e2"


def test_the_fixed_stage_b_settings():
    s, t, sel = CFG4["sweep"], CFG4["train"], CFG4["selection"]
    assert s["betas"] == [0.1, 0.3] and s["learning_rates"] == [5e-5, 2e-4] and s["seed"] == 42 and s["repeat_seeds"] == [43, 44]
    assert (t["epochs"], t["effective_batch_size"], t["warmup_ratio"], t["lr_schedule"], t["weight_decay"],
            t["max_grad_norm"], t["max_seq_len"]) == (3, 16, 0.1, "cosine", 0.0, 1.0, 2048)
    assert sel["min_correct_refusals_val"] == 7 and sel["max_median_length_ratio"] == 1.5 and sel["max_hit_max_new_tokens"] == 1
    assert sel["order"] == ["false_refusals", "correct_refusals_strict", "token_f1", "number_faithfulness", "peak_gpu_mem",
                            "train_seconds", "epoch"]
    assert CFG4["mlflow"]["experiment_name"] == "phase4_dpo" and CFG4["pairs"]["include_type2"] is False
    assert dpo_grid.validate(CFG4) == []
    assert CFG4["paths"]["adapters_dir"] != CFG3["paths"]["adapters_dir"] and CFG4["paths"]["runs_dir"] != CFG3["paths"]["runs_dir"]
    assert CFG4["paths"]["selection_json"] != CFG3["paths"]["selection_json"]


def test_validate_catches_broken_configs():
    def broken(path, value):
        c = copy.deepcopy(CFG4)
        d = c
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        return dpo_grid.validate(c)
    assert broken(["sweep", "betas"], [0.1])
    assert broken(["train", "micro_batch_size"], 3)
    assert broken(["sweep", "repeat_seeds"], [42, 44])
    assert broken(["sweep", "repeat_seeds"], [43, 99])                                       # no SFT start adapter for seed 99
    assert broken(["train", "epochs"], 0)
    assert broken(["selection", "order"], ["token_f1", "false_refusals"])
    assert broken(["pairs", "include_type2"], True)


def test_start_adapter_hashes_in_the_config_match_the_local_files_if_present():
    for sd, st in CFG4["start"]["seeds"].items():
        f = ROOT / st["adapter"] / dpo_grid.ADAPTER_FILE
        if f.exists():
            assert dpo_grid.file_sha256(f) == st["sha256"], sd


def test_check_start_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(dpo_grid, "REPO_ROOT", tmp_path)
    run = {"start_adapter": "ad", "start_sha256": ""}
    with pytest.raises(SystemExit, match="not found"):
        dpo_grid.check_start_adapter(run)
    (tmp_path / "ad").mkdir()
    (tmp_path / "ad" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ad" / dpo_grid.ADAPTER_FILE).write_bytes(b"weights")
    import hashlib
    good = hashlib.sha256(b"weights").hexdigest()
    assert dpo_grid.check_start_adapter({**run, "start_sha256": good}) == good
    with pytest.raises(SystemExit, match="not the frozen SFT adapter"):
        dpo_grid.check_start_adapter({**run, "start_sha256": "0" * 64})


def test_adapter_config_must_match_the_phase3_lora():
    lo = CFG3["lora"]
    ok = {"r": lo["r"], "lora_alpha": lo["alpha"], "lora_dropout": lo["dropout"], "target_modules": list(lo["target_modules"]),
          "base_model_name_or_path": CFG4["model"]["model_id"]}
    assert dpo_grid.adapter_config_problems(ok, CFG3, CFG4["model"]["model_id"]) == []
    assert len(dpo_grid.adapter_config_problems({**ok, "r": 8, "lora_dropout": 0.1}, CFG3, CFG4["model"]["model_id"])) == 2
    assert dpo_grid.adapter_config_problems({**ok, "target_modules": ["q_proj"]}, CFG3, CFG4["model"]["model_id"])
    assert dpo_grid.adapter_config_problems({**ok, "base_model_name_or_path": "other"}, CFG3, CFG4["model"]["model_id"])
    real = ROOT / CFG4["start"]["seeds"][42]["adapter"] / "adapter_config.json"
    if real.exists():
        assert dpo_grid.adapter_config_problems(json.loads(real.read_text(encoding="utf-8")), CFG3, CFG4["model"]["model_id"]) == []


# ---- pairs and encoding ------------------------------------------------------------------------------------------------------
def _template(messages, add_generation_prompt):
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _tokenize(text):
    return [ord(ch) for ch in text]


def _fake_pairs(repeat=3, long_answer=None):
    prompt = [{"role": "system", "content": "S"}, {"role": "user", "content": "Context:\nC\n\nQuestion: Q?"}]
    out = [{"pair_id": "a1:t1", "pair_type": 1, "prompt": prompt, "chosen": "ans one", "rejected": "I cannot answer.",
            "meta": {"distinct_pair_id": "a1:t1", "repeat_no": 1}},
           {"pair_id": "a2:t1", "pair_type": 1, "prompt": prompt, "chosen": long_answer or "ans two", "rejected": "I cannot answer.",
            "meta": {"distinct_pair_id": "a2:t1", "repeat_no": 1}}]
    for k in range(1, repeat + 1):
        out.append({"pair_id": f"u1:t3:r{k}", "pair_type": 3, "prompt": prompt, "chosen": "I cannot answer.", "rejected": "made up",
                    "meta": {"distinct_pair_id": "u1:t3", "repeat_no": k}})
    return out


@needs_training_data
def test_real_pairs_file_counts_and_integrity():
    pairs = dpo_data.read_pairs(CFG4)
    c = dpo_data.count_pairs(pairs)
    assert c == {"lines": 189, "distinct": 153, "type1_lines": 135, "type1_distinct": 135, "type3_lines": 54, "type3_distinct": 18}
    assert dpo_data.pairs_sha256(CFG4) == dpo_grid.file_sha256(ROOT / CFG4["paths"]["pairs"])
    assert len({p["pair_id"] for p in pairs}) == 189 and all(p["pair_type"] in (1, 3) for p in pairs)


def test_read_pairs_refuses_a_changed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dpo_data, "REPO_ROOT", tmp_path)
    (tmp_path / "data" / "instruction").mkdir(parents=True)
    p, h = tmp_path / CFG4["paths"]["pairs"], tmp_path / CFG4["paths"]["pairs_sha256"]
    p.write_text(dpo_pairs.pair_lines(_fake_pairs()), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing"):
        dpo_data.read_pairs(CFG4)
    h.write_text(f"{dpo_grid.file_sha256(p)}  x\n", encoding="utf-8")
    assert len(dpo_data.read_pairs(CFG4)) == 5
    p.write_text(dpo_pairs.pair_lines(_fake_pairs()) + "\n{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed"):
        dpo_data.read_pairs(CFG4)


def test_distinct_pairs_and_repeat_checks():
    d = dpo_data.distinct_pairs(_fake_pairs())
    assert list(d) == ["a1:t1", "a2:t1", "u1:t3"]
    bad = _fake_pairs()
    bad[-1]["rejected"] = "different"
    with pytest.raises(ValueError, match="repeats of u1:t3 differ"):
        dpo_data.distinct_pairs(bad)


def test_encode_pairs_masks_the_prompt_and_uses_the_sft_text():
    enc = dpo_data.encode_pairs(_fake_pairs(), _template, _tokenize, 100000, "<|im_end|>")
    assert len(enc["lines"]) == 5 and enc["dropped_distinct_ids"] == [] and enc["stats"]["n"] == 6   # 3 distinct pairs x 2 sides
    assert [x["distinct_id"] for x in enc["lines"]] == ["a1:t1", "a2:t1", "u1:t3", "u1:t3", "u1:t3"]
    ex = enc["examples"]["a1:t1"]
    for side, text in (("chosen", "ans one"), ("rejected", "I cannot answer.")):
        e = ex[side]
        n = e["n_prompt"]
        assert "".join(map(chr, e["input_ids"][:n])).endswith("<|im_start|>assistant\n")
        assert "".join(map(chr, e["input_ids"][n:])) == text + "<|im_end|>"                          # answer + end-of-turn token
        assert set(e["labels"][:n]) == {-100} and e["labels"][n:] == e["input_ids"][n:]


def test_a_pair_that_does_not_fit_is_dropped_whole_with_its_repeats():
    pairs = _fake_pairs(long_answer="x" * 400)
    enc = dpo_data.encode_pairs(pairs, _template, _tokenize, 300, "<|im_end|>")
    assert enc["dropped_distinct_ids"] == ["a2:t1"] and enc["dropped_lines"] == 1 and "a2:t1" not in enc["examples"]
    enc2 = dpo_data.encode_pairs(pairs, _template, _tokenize, 100, "<|im_end|>")               # now every pair is too long
    assert enc2["lines"] == [] and enc2["dropped_lines"] == 5 and len(enc2["dropped_distinct_ids"]) == 3
    assert enc["examples"]["a1:t1"]["chosen"]["input_ids"] == dpo_data.encode_pairs(pairs, _template, _tokenize, 100000)["examples"]["a1:t1"]["chosen"]["input_ids"]


def test_stack_pairs_puts_all_chosen_first_then_all_rejected():
    enc = dpo_data.encode_pairs(_fake_pairs(), _template, _tokenize, 100000, "")
    pe = [enc["examples"]["a1:t1"], enc["examples"]["u1:t3"]]
    batch, m = dpo_data.stack_pairs(pe, pad_id=0)
    assert m == 2 and len(batch["input_ids"]) == 4
    assert batch["input_ids"][0][:len(pe[0]["chosen"]["input_ids"])] == pe[0]["chosen"]["input_ids"]
    assert batch["input_ids"][1][:len(pe[1]["chosen"]["input_ids"])] == pe[1]["chosen"]["input_ids"]
    assert batch["input_ids"][2][:len(pe[0]["rejected"]["input_ids"])] == pe[0]["rejected"]["input_ids"]
    assert len({len(r) for r in batch["input_ids"]}) == 1 and set(batch["labels"][0][-1:]) <= {-100, ord(">")}


def test_reference_log_probs_are_per_distinct_pair_and_cached_by_key(tmp_path):
    key = dpo_data.reference_key("p" * 64, "a" * 64, "model", 2048)
    path = tmp_path / "runs" / "reference_logps_seed42.json"
    assert dpo_data.load_reference_cache(path, key, {"a1:t1"}) is None
    values = {"a1:t1": [-5.0, -6.0], "a2:t1": [-7.0, -8.0], "u1:t3": [-2.0, -9.0]}
    dpo_data.save_reference_cache(path, key, values)
    assert dpo_data.load_reference_cache(path, key, set(values)) == values
    assert dpo_data.load_reference_cache(path, {**key, "pairs_sha256": "q" * 64}, set(values)) is None          # other pairs
    assert dpo_data.load_reference_cache(path, {**key, "start_adapter_sha256": "b" * 64}, set(values)) is None  # other adapter
    assert dpo_data.load_reference_cache(path, key, set(values) | {"new:t1"}) is None                           # does not cover
    assert json.loads(path.read_text(encoding="utf-8"))["n_distinct_pairs"] == 3
    # the three repeats of the type 3 pair get the very same numbers
    lines = dpo_data.encode_pairs(_fake_pairs(), _template, _tokenize, 100000)["lines"]
    assert {dpo_data.reference_for(values, x) for x in lines if x["pair_type"] == 3} == {(-2.0, -9.0)}
    assert len(values) == 3 and len(lines) == 5                                                              # once per distinct pair


# ---- trainer structure (no GPU: source checks and imports) -----------------------------------------------------------
def test_modules_import_without_torch_and_trainer_never_touches_val_or_test():
    for mod in ("dpo_train", "dpo_eval", "dpo_data", "dpo_loss", "dpo_grid", "select_dpo", "dpo_final_test"):
        src = (ROOT / "climate_slm" / "training" / f"{mod}.py").read_text(encoding="utf-8")
        for lib in ("torch", "transformers", "peft", "bitsandbytes"):
            assert f"\nimport {lib}" not in src and f"\nfrom {lib}" not in src, (mod, lib)
    src = (ROOT / "climate_slm" / "training" / "dpo_train.py").read_text(encoding="utf-8")
    assert "sft_val" not in src and "sft_test" not in src and 'splits' not in src and "load_items" not in src
    assert '"test"' not in src and "--split" not in src


def test_trainer_follows_the_recipe():
    src = (ROOT / "climate_slm" / "training" / "dpo_train.py").read_text(encoding="utf-8")
    assert src.index("compute_references(model") < src.index("for epoch in range(1, t[\"epochs\"] + 1)")    # reference BEFORE training
    assert "is_trainable=True" in src and "PeftModel.from_pretrained" in src                                # policy = the SFT adapter
    assert "dpo_data.reference_for(refs, x)" in src and "logsigmoid=F.logsigmoid" in src
    assert "torch.amp.GradScaler" in src and "clip_grad_norm_" in src and "schedule.lr_at_step" in src
    assert "model.save_pretrained(str(out_dir))" in src and src.count("grid.clear_stale_error") == 3
    assert src.count("error_file_path") == 1 and "mlflow_log.start_tracker" in src and "check_start_adapter" in src
    assert "use_cache = False" in src and "gradient_checkpointing" not in src
    assert src.count("compute_references(") == 2                                                            # definition + one call


def test_trainer_helpers_read_the_right_run_list(tmp_path):
    assert [r["name"] for r in dpo_train.resolve_runs("grid", CFG4)] == [r["name"] for r in dpo_grid.grid_runs(CFG4)]
    with pytest.raises(SystemExit):
        dpo_train.resolve_runs("repeats", {**CFG4, "paths": {**CFG4["paths"], "selection_json": str(tmp_path / "none.json")}})


def test_check_inputs_on_the_real_files_if_the_adapters_are_here(capsys):
    if not (ROOT / CFG4["start"]["seeds"][44]["adapter"] / dpo_grid.ADAPTER_FILE).exists():
        pytest.skip("SFT adapters are not on this machine")
    assert dpo_train.check_inputs(CFG4, CFG3) == []
    out = capsys.readouterr().out
    assert "pairs file ok" in out and "all checks passed" in out and "seed 43" in out


def test_group_skip_clears_the_stale_error_and_a_failed_run_keeps_it(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG4)
    cfg["paths"]["runs_dir"], cfg["paths"]["adapters_dir"] = str(tmp_path / "runs"), str(tmp_path / "ad")
    monkeypatch.setattr(grid, "REPO_ROOT", tmp_path)
    run = dpo_grid.grid_runs(cfg)[0]
    log = grid.run_log_path(cfg, run["name"])
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"complete": True}), encoding="utf-8")
    for e in grid.epochs(cfg):
        d = grid.adapter_dir(cfg, run["name"], e)
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    err = grid.error_file_path(cfg, run["name"])
    err.write_text("old failure", encoding="utf-8")
    assert grid.run_is_done(cfg, run) and dpo_train.run_group([run], cfg) == [] and not err.exists()
    log.write_text(json.dumps({"complete": False}), encoding="utf-8")
    err.write_text("still failing", encoding="utf-8")
    monkeypatch.setattr(dpo_train.subprocess, "call", lambda cmd: 1)
    assert dpo_train.run_group([run], cfg) == [run["name"]] and err.exists()


# ---- evaluation loop and test gate --------------------------------------------------------------------------------------------------
def test_val_loop_is_val_only_and_plans_the_12_adapters():
    assert dpo_eval.ALLOWED_SPLITS == ("val",)
    with pytest.raises(SystemExit):
        dpo_eval.main(["--group", "grid", "--split", "test"])
    src = (ROOT / "climate_slm" / "training" / "dpo_eval.py").read_text(encoding="utf-8")
    assert '"test"' not in src.replace("`", "") or src.count('"test"') == 0
    items = dpo_eval.plan(dpo_grid.grid_runs(CFG4), CFG4, ECFG, "val")
    assert len(items) == 12 and items[0]["name"] == "dpo_b0.1_lr5e-5_e1" and items[-1]["name"] == "dpo_b0.3_lr2e-4_e3"
    assert items[0]["eval_file"].name == "eval_dpo_b0.1_lr5e-5_e1_val.json" and all(i["eval_file"].name.endswith("_val.json") for i in items)
    assert all(i["base_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct" for i in items)
    assert "adapters_dpo" in str(items[0]["adapter"])


def _world(tmp_path, monkeypatch, winner=True, tested=False, frozen=True):
    cfg = copy.deepcopy(CFG4)
    for k in ("selection_json", "selection_sha", "test_log"):
        cfg["paths"][k] = str(tmp_path / Path(cfg["paths"][k]).name)
    cfg["paths"]["adapters_dir"] = str(tmp_path / "ad")
    monkeypatch.setattr(grid, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(selection_io, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dpo_final_test, "REPO_ROOT", tmp_path)
    sel = {"winner": {"candidate": "dpo_b0.1_lr5e-5_e2", "run": "dpo_b0.1_lr5e-5", "epoch": 2, "beta": 0.1, "lr": 5e-5, "seed": 42}
           if winner else None}
    sp = tmp_path / Path(cfg["paths"]["selection_json"]).name
    sp.write_text(json.dumps(sel), encoding="utf-8")
    if frozen:
        selection_io.sha_path(cfg).write_text(f"{selection_io.file_sha256(sp)}  x\n", encoding="utf-8")
    d = grid.adapter_dir(cfg, "dpo_b0.1_lr5e-5", 2)
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    ecfg = copy.deepcopy(ECFG)
    ecfg["paths"]["outputs_dir"] = str(tmp_path)
    monkeypatch.setattr(dpo_final_test, "default_out", lambda c, n, s: tmp_path / f"eval_{n}_{s}.json")
    if tested:
        (tmp_path / "eval_dpo_b0.1_lr5e-5_e2_test.json").write_text("{}", encoding="utf-8")
    return cfg, ecfg, sp


def test_final_test_gate(tmp_path, monkeypatch):
    cfg, ecfg, sp = _world(tmp_path, monkeypatch)
    w = dpo_final_test.check_can_run(cfg, ecfg)
    assert w["candidate"] == "dpo_b0.1_lr5e-5_e2"
    cmd = dpo_final_test.test_command(cfg, w)
    assert cmd[cmd.index("--split") + 1] == "test" and cmd[cmd.index("--name") + 1] == "dpo_b0.1_lr5e-5_e2"
    assert cmd[cmd.index("--base-model-id") + 1] == "Qwen/Qwen2.5-0.5B-Instruct" and "--load-4bit" not in cmd
    assert cmd[cmd.index("--model") + 1].endswith("e2")
    sp.write_text(sp.read_text(encoding="utf-8") + " ", encoding="utf-8")                       # file changed after the freeze
    with pytest.raises(SystemExit, match="not frozen"):
        dpo_final_test.check_can_run(cfg, ecfg)


def test_final_test_refuses_without_freeze_without_winner_and_a_second_time(tmp_path, monkeypatch):
    cfg, ecfg, _ = _world(tmp_path, monkeypatch, frozen=False)
    with pytest.raises(SystemExit, match="not frozen"):
        dpo_final_test.check_can_run(cfg, ecfg)
    cfg, ecfg, _ = _world(tmp_path, monkeypatch, winner=False)
    with pytest.raises(SystemExit, match="no winner"):
        dpo_final_test.check_can_run(cfg, ecfg)
    cfg, ecfg, _ = _world(tmp_path, monkeypatch, tested=True)
    with pytest.raises(SystemExit, match="already evaluated on test"):
        dpo_final_test.check_can_run(cfg, ecfg)


def test_final_test_records_once_and_a_log_entry_blocks_a_second_run(tmp_path, monkeypatch):
    cfg, ecfg, _ = _world(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(dpo_final_test.subprocess, "call", lambda cmd, cwd: calls.append(cmd) or 0)
    monkeypatch.setattr(dpo_final_test, "load_phase4_config", lambda: cfg)
    monkeypatch.setattr(dpo_final_test, "load_eval_config", lambda: ecfg)
    assert dpo_final_test.main([]) == 0 and len(calls) == 1
    log = json.loads(Path(cfg["paths"]["test_log"]).read_text(encoding="utf-8"))
    assert [e["candidate"] for e in log] == ["dpo_b0.1_lr5e-5_e2"] and len(log[0]["selection_sha256"]) == 64
    with pytest.raises(SystemExit, match="already evaluated on test"):
        dpo_final_test.main([])
    assert len(calls) == 1


def test_only_the_final_test_module_names_the_test_split():
    for mod in ("dpo_train", "dpo_eval", "select_dpo", "dpo_data", "dpo_grid", "dpo_loss"):
        src = (ROOT / "climate_slm" / "training" / f"{mod}.py").read_text(encoding="utf-8")
        assert '"--split", "test"' not in src and "--split test" not in src, mod
    assert '"--split", "test"' in (ROOT / "climate_slm" / "training" / "dpo_final_test.py").read_text(encoding="utf-8")


# ---- selection ----------------------------------------------------------------------------------------------------------------------
def _cand(name, epoch=1, correct=8, fr=2, f1=0.65, nf=1.0, peak=5000.0, secs=100.0, chars=100, hit=0, beta=0.1, lr=5e-5, run=None):
    def rate(k, n):
        return {"count": k, "n": n, "rate": k / n, "wilson95": list(M.wilson_ci(k, n))}
    m = {"n_answerable": 42, "n_refusal": 11, "token_f1": {"mean": f1, "ci95": [f1 - 0.05, f1 + 0.05], "n": 42},
         "key_recall": None, "key_support": None,
         "number_faithfulness": {"mean": nf, "ci95": [nf, nf], "n_applied": 14},
         "false_refusals": rate(fr, 42), "correct_strict": rate(correct, 11), "correct_loose": rate(correct, 11),
         "answered_when_should_refuse": rate(11 - correct, 11), "refusal_perplexity": 1.1, "hit_max_new_tokens": hit,
         "repetition_loops": 0, "output_chars": {"median": chars, "max": chars * 2}}
    return {"candidate": name, "run": run or name, "epoch": epoch, "beta": beta, "lr": lr, "seed": 42, "metrics": m,
            "cost": {"peak_gpu_mem_mb": peak, "train_seconds": secs}, "train_epoch": None}


CONTROL = _cand("sft", correct=8, fr=3, f1=0.615, chars=100)["metrics"]


def test_selection_filter_and_rule_order():
    cands = [_cand("few_refusals", correct=6, fr=0),                  # dropped: only 6/11 correct refusals
             _cand("long", correct=9, fr=0, chars=151),               # dropped: median length 151 > 150
             _cand("loopy", correct=9, fr=0, hit=2),                  # dropped: 2 outputs at max_new_tokens
             _cand("ok_len", correct=8, fr=1, chars=150),             # exactly 1.5 x control: kept
             _cand("a", correct=7, fr=1), _cand("b", correct=9, fr=2)]
    r = select_dpo.select_winner(cands, CONTROL, CFG4)
    assert {d["candidate"] for d in r["dropped"]} == {"few_refusals", "long", "loopy"}
    assert r["ranked"] == ["ok_len", "a", "b"] and r["winner"]["candidate"] == "ok_len"                    # fewest false refusals first
    assert r["deciding_step"] == "correct_refusals_strict" and r["improves_on_control"] is True
    assert r["winner"]["beta"] == 0.1 and r["winner"]["run"] == "ok_len"


@pytest.mark.parametrize("better,worse,step", [
    (dict(fr=1), dict(fr=2), "false_refusals"),
    (dict(correct=9), dict(correct=8), "correct_refusals_strict"),
    (dict(f1=0.7), dict(f1=0.6), "token_f1"),
    (dict(nf=1.0), dict(nf=0.9), "number_faithfulness"),
    (dict(peak=4000.0), dict(peak=5000.0), "peak_gpu_mem"),
    (dict(secs=50.0), dict(secs=60.0), "train_seconds"),
    (dict(epoch=1), dict(epoch=2), "epoch")])
def test_each_rule_step_decides_when_the_earlier_ones_tie(better, worse, step):
    base = dict(correct=8, fr=1)
    r = select_dpo.select_winner([_cand("worse", **{**base, **worse}), _cand("better", **{**base, **better})], CONTROL, CFG4)
    assert r["winner"]["candidate"] == "better" and r["deciding_step"] == step


def test_ties_on_f1_are_compared_after_rounding():
    r = select_dpo.select_winner([_cand("x", f1=0.65004, fr=1), _cand("y", f1=0.65001, fr=1, peak=1.0)], CONTROL, CFG4)
    assert r["winner"]["candidate"] == "y" and r["deciding_step"] == "peak_gpu_mem"


def test_no_candidate_improves_branch():
    cands = [_cand("a", fr=3), _cand("b", fr=4), _cand("c", correct=5, fr=0)]       # c is dropped; best kept has 3 = control
    r = select_dpo.select_winner(cands, CONTROL, CFG4)
    assert r["winner"] is None and r["improves_on_control"] is False and r["ranked"] == ["a", "b"]
    assert select_dpo.NO_IMPROVEMENT in r["status"] and "has 3 false refusals" in r["status"]
    none_left = select_dpo.select_winner([_cand("c", correct=5, fr=0)], CONTROL, CFG4)
    assert none_left["winner"] is None and "passes the filter" in none_left["status"]
    assert select_dpo.select_winner([], None, CFG4)["winner"] is None
    nothing = select_dpo.select_winner([], CONTROL, CFG4)
    assert nothing["winner"] is None and nothing["status"].startswith("not yet measured") and select_dpo.NO_IMPROVEMENT not in nothing["status"]
    # one fewer false refusal than the control is enough
    assert select_dpo.select_winner([_cand("a", fr=2)], CONTROL, CFG4)["winner"]["candidate"] == "a"


def test_vs_control_lines_say_not_established_when_intervals_overlap():
    lines = select_dpo.vs_control_lines(_cand("w", fr=2, correct=8, f1=0.65)["metrics"], CONTROL, "DPO")
    assert len(lines) == 3 and "false refusals: DPO 2/42 against SFT control 3/42: lower" in lines[0]
    assert all("NOT established" in x for x in (lines[0], lines[2]))
    eq = select_dpo.vs_control_lines(_cand("w", fr=3, correct=8, f1=0.615)["metrics"], CONTROL, "DPO")
    assert "equal" in eq[0] and "NOT established" not in eq[0]


def _write_eval(tmp_path, name, split, fr, correct):
    """A minimal saved eval file with the given counts (the real evaluator format, as far as the tables read it)."""
    def rate(k, n):
        return {"count": k, "n": n, "rate": k / n, "ci95": [0.0, 1.0], "wilson95": list(M.wilson_ci(k, n))}
    mean = {"n": 42, "mean": 0.6, "ci95": [0.5, 0.7]}
    ev = {"run_name": name, "split": split, "summary": {
        "n_items": 53, "answerable": {"n": 42, "token_f1": mean, "key_recall": {**mean, "n_applied": 40, "n_answerable": 42},
                                      "key_support": {**mean, "n_applied": 40, "n_answerable": 42},
                                      "number_faithfulness": {"n": 14, "n_applied": 14, "mean": 1.0, "ci95": [1.0, 1.0]},
                                      "false_refusal_strict": rate(fr, 42)},
        "refusal": {"n": 11, "correct_refusal_strict": rate(correct, 11), "correct_refusal_loose": rate(correct, 11),
                    "answered_when_should_refuse": rate(11 - correct, 11)},
        "perplexity": {"refusal": {"perplexity": 1.1}}, "hit_max_new_tokens": {"count": 0}, "output_length_chars": {"median": 100, "max": 200}},
        "items": [{"output": "fine answer here", "key_recall": 0.5, "key_support": 0.5}]}
    (tmp_path / f"eval_{name}_{split}.json").write_text(json.dumps(ev), encoding="utf-8")


def test_collect_reads_saved_files_and_builds_the_documents(tmp_path, monkeypatch):
    ecfg = copy.deepcopy(ECFG)
    ecfg["paths"]["outputs_dir"] = str(tmp_path)
    monkeypatch.setattr(select_dpo.sb, "default_out", lambda c, n, s: tmp_path / f"eval_{n}_{s}.json")
    cfg = copy.deepcopy(CFG4)
    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    monkeypatch.setattr(grid, "REPO_ROOT", tmp_path)
    for sd, fr, cr in ((42, 3, 8), (43, 0, 7), (44, 1, 8)):
        _write_eval(tmp_path, dpo_grid.control_name(cfg, sd), "val", fr, cr)
    _write_eval(tmp_path, "base", "val", 0, 0)
    _write_eval(tmp_path, "dpo_b0.3_lr2e-4_e2", "val", 1, 8)
    _write_eval(tmp_path, "dpo_b0.3_lr2e-4_e1", "val", 2, 8)
    _write_eval(tmp_path, "dpo_b0.3_lr2e-4_s43_e2", "val", 0, 8)
    log = {"complete": True, "peak_gpu_mem_mb": 5000.0, "train_seconds": 200.0, "trainable_params": 8798208, "dropped_distinct_ids": [],
           "dropped_lines": 0, "pair_counts": {"lines": 189, "distinct": 153},
           "epochs": [{"epoch": e, "dpo_loss": 0.6 - 0.1 * e, "reward_accuracy": 0.5 + 0.1 * e, "mean_margin": 0.1 * e, "adapter_mb": 33.6}
                      for e in (1, 2, 3)]}
    (tmp_path / "runs" / "dpo_b0.3_lr2e-4").mkdir(parents=True)
    (tmp_path / "runs" / "dpo_b0.3_lr2e-4" / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    data = select_dpo.collect(cfg, CFG3, ecfg)
    assert data["control"]["val"]["false_refusals"]["count"] == 3 and len(data["candidates"]) == 2
    assert data["winner"]["candidate"] == "dpo_b0.3_lr2e-4_e2" and data["winner"]["beta"] == 0.3
    assert data["candidates"][0]["train_epoch"]["reward_accuracy"] == pytest.approx(0.6)
    p = data["paired"]
    assert p["seeds_paired"] == [42, 43] and p["counts"]["false_refusals"] == {"better": 1, "equal": 1, "worse": 0}
    md = select_dpo.selection_md(data, cfg)
    assert "**WINNER**" in md and "SFT control, no DPO" in md and "Paired seeds on val" in md and "not yet measured" in md
    assert "DPO training numbers per epoch" in md and "NOT established" in md
    pl = select_dpo.payload(data)
    assert pl["winner"]["candidate"] == "dpo_b0.3_lr2e-4_e2" and "test" not in pl["control"] and "test" not in pl["tuned"]
    assert dpo_grid.winner_run(cfg, pl)["name"] == "dpo_b0.3_lr2e-4"                      # what the repeat group reads
    assert [r["name"] for r in dpo_grid.repeat_runs(cfg, dpo_grid.winner_run(cfg, pl))] == ["dpo_b0.3_lr2e-4_s43", "dpo_b0.3_lr2e-4_s44"]


def test_no_winner_document_says_sft_stays(tmp_path):
    data = {"control_false_refusals": 3, "control": {"val": CONTROL, "candidate": "lora_fp16_lr2e-4_e2", "test": None}, "candidates": [],
            "dropped": [], "ranked": [], "winner": None, "runner_up": None, "deciding_step": None, "closeness": [], "missing_runs": [],
            "runs": [], "repeat_costs": {}, "paired": None, "status": select_dpo.NO_IMPROVEMENT + ": x", "tuned": None,
            "baselines": {"val": {}, "test": {}}}
    md = select_dpo.selection_md(data, CFG4)
    assert "no DPO candidate improves on the SFT model on val" in md and "SFT stays" in md and "no seed repeat" in md
    with pytest.raises(SystemExit, match="no winner"):
        dpo_grid.winner_run(CFG4, select_dpo.payload({**data, "control": {"val": CONTROL}}))


def test_paired_seed_counts_and_plain_words():
    def row(sd, dpo, sft):
        return {"seed": sd, "dpo": dpo, "sft": sft, "dpo_candidate": f"d{sd}", "sft_candidate": f"s{sd}"}
    rows = [row(42, _cand("a", fr=1, correct=8, f1=0.7)["metrics"], _cand("b", fr=3, correct=8, f1=0.6)["metrics"]),
            row(43, _cand("a", fr=0, correct=7, f1=0.6)["metrics"], _cand("b", fr=0, correct=8, f1=0.6)["metrics"]),
            row(44, None, None)]
    both = [r for r in rows if r["dpo"] and r["sft"]]
    assert len(both) == 2
    p = {"epoch": 2, "rows": rows, "seeds_expected": [42, 43, 44], "seeds_paired": [42, 43],
         "counts": {"false_refusals": {"better": 1, "equal": 1, "worse": 0}, "correct_refusals": {"better": 0, "equal": 1, "worse": 1},
                    "token_f1": {"better": 1, "equal": 1, "worse": 0}}}
    text = " ".join(select_dpo.consistency_lines(p))
    assert "2 paired seed(s)" in text and "fewer in 1" in text and "Only seeds [42, 43] of [42, 43, 44]" in text
    assert "cannot be established" in text
    table = "\n".join(select_dpo.paired_table(p))
    assert "| 42 | 3/42 | 1/42 | 8/11 | 8/11 |" in table and "not yet measured" in table
    assert select_dpo.consistency_lines({"seeds_paired": [], "seeds_expected": [42], "counts": {}}) == ["not yet measured"]


def test_final_tables_answer_the_plain_words_question(tmp_path):
    data = {"control_false_refusals": 3, "status": "winner found",
            "control": {"candidate": "sft", "val": CONTROL, "test": _cand("s", fr=4, correct=9, f1=0.617)["metrics"]},
            "tuned": {"winner": {"candidate": "dpo_x_e2"}, "val": _cand("w", fr=1)["metrics"],
                      "test": _cand("w", fr=3, correct=9, f1=0.62)["metrics"]},
            "baselines": {"val": {"base": _cand("b", fr=0, correct=0, f1=0.575)["metrics"]},
                          "test": {"base": _cand("b", fr=0, correct=0, f1=0.542)["metrics"]}},
            "paired": None, "runs": [], "repeat_costs": {}}
    q = "\n".join(select_dpo.question_lines(data, CFG3))
    assert "to <= 3/42? YES (3/42)" in q
    assert "Without losing strict correct refusals below 7/11? YES (9/11)" in q and "NOT claimed to be better" in q
    assert "NOT established" in q                                                           # 3/42 against 4/42: intervals overlap
    md = select_dpo.final_md(data, CFG4, CFG3, {"dpo_x_e2": 1}, "abc")
    assert "`abc`" in md and "criterion 7 (SFT checks before DPO)" in md and "not applicable to a DPO model" in md
    assert "DPO winner dpo_x_e2" in md and "test runs recorded: 1" in md and "answerable token F1 not below base (0.542)" in md
    failing = {**data, "tuned": {**data["tuned"], "test": _cand("w", fr=5, correct=6)["metrics"]}}
    q2 = "\n".join(select_dpo.question_lines(failing, CFG3))
    assert "? NO (5/42)" in q2 and "? NO (6/11)" in q2
    assert select_dpo.question_lines({**data, "tuned": None}, CFG3) == ["not yet measured (no test run)"]


def test_selection_main_writes_and_freezes_and_does_not_rewrite(tmp_path, monkeypatch, capsys):
    cfg = copy.deepcopy(CFG4)
    for k in ("selection_json", "selection_md", "selection_sha", "final_md", "test_log"):
        cfg["paths"][k] = str(tmp_path / Path(cfg["paths"][k]).name)
    monkeypatch.setattr(select_dpo, "load_phase4_config", lambda: cfg)
    monkeypatch.setattr(select_dpo, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(selection_io, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(select_dpo, "collect", lambda *a: {
        "control_false_refusals": 3, "control": {"val": CONTROL, "candidate": "c", "test": None}, "candidates": [], "dropped": [],
        "ranked": [], "winner": None, "runner_up": None, "deciding_step": None, "closeness": [], "missing_runs": [], "runs": [],
        "repeat_costs": {}, "paired": None, "status": select_dpo.NO_IMPROVEMENT, "tuned": None, "baselines": {"val": {}, "test": {}}})
    with pytest.raises(SystemExit, match="not frozen"):
        select_dpo.main(["--final-tables"])
    assert select_dpo.main(["--freeze"]) == 0
    sp = Path(cfg["paths"]["selection_json"])
    sha = selection_io.file_sha256(sp)
    assert Path(cfg["paths"]["selection_sha"]).read_text(encoding="utf-8").split()[0] == sha
    sp.write_text(sp.read_text(encoding="utf-8"), encoding="utf-8")
    capsys.readouterr()
    assert select_dpo.main([]) == 0 and "frozen" in capsys.readouterr().out and selection_io.file_sha256(sp) == sha   # not rewritten
    assert select_dpo.main(["--final-tables"]) == 0 and Path(cfg["paths"]["final_md"]).exists()


# ---- collect sets, bundle, notebook ---------------------------------------------------------------------------------------------------
def test_phase4_collect_sets(tmp_path):
    out = tmp_path / "outputs"
    for rel in ("phase4_selection.json", "phase4_selection.md", "phase4_selection.sha256", "eval_dpo_b0.1_lr5e-5_e1_val.json",
                "refusal_detection_review_dpo_b0.1_lr5e-5_e1_val.csv", "runs_dpo/dpo_b0.1_lr5e-5/train_log.json",
                "runs_dpo/reference_logps_seed42.json", "mlruns/0/meta.yaml", "adapters_dpo/dpo_b0.1_lr5e-5/e1/adapter_model.safetensors",
                "eval_lora_fp16_lr2e-4_e2_val.json", "phase3_selection.json", "adapters/lora_fp16_lr2e-4/e2/adapter_config.json"):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text("x", encoding="utf-8")
    res = {p.relative_to(out).as_posix() for p in collect.phase4_result_files(out, ECFG)}
    assert res == {"phase4_selection.json", "phase4_selection.md", "phase4_selection.sha256", "eval_dpo_b0.1_lr5e-5_e1_val.json",
                   "refusal_detection_review_dpo_b0.1_lr5e-5_e1_val.csv", "runs_dpo/dpo_b0.1_lr5e-5/train_log.json",
                   "runs_dpo/reference_logps_seed42.json", "mlruns/0/meta.yaml"}
    assert [p.relative_to(out).as_posix() for p in collect.phase4_adapter_files(out, ECFG)] == ["adapters_dpo/dpo_b0.1_lr5e-5/e1/adapter_model.safetensors"]
    assert "adapters_dpo" not in " ".join(p.as_posix() for p in collect.phase3_adapter_files(out, ECFG))   # Phase 3 zip is unchanged
    assert not any("dpo" in p.name for p in collect.phase3_result_files(out, ECFG))


def test_phase4_start_files_and_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "REPO_ROOT", tmp_path)
    cfg = copy.deepcopy(CFG4)
    out = tmp_path / "outputs"
    for st in cfg["start"]["seeds"].values():
        d = tmp_path / st["adapter"]
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        (d / "adapter_model.safetensors").write_bytes(b"w")
        (out / f"eval_{st['control']}_val.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing files"):
        collect.phase4_start_files(out, cfg)
    (out / "eval_base_val.json").write_text("{}", encoding="utf-8")
    files = collect.phase4_start_files(out, cfg)
    names = sorted(p.relative_to(out).as_posix() for p in files)
    assert len(names) == 3 * 2 + 3 + 1 and "adapters/lora_fp16_lr2e-4_s44/e2/adapter_model.safetensors" in names
    assert not any("e1" in n.split("/") or "e3" in n.split("/") for n in names) and not any("test" in n for n in names)
    z = collect.make_zip(files, tmp_path / "s.zip", out)
    assert sorted(zipfile.ZipFile(z).namelist()) == names


def test_bundle_has_the_pairs_and_the_dpo_code():
    bundle = ECFG["bundle"]
    assert "data/instruction/dpo_train.jsonl" in bundle["files"] and "data/instruction/dpo_train.sha256" in bundle["files"]
    assert "dpo_*" in ECFG["report"]["hide_run_patterns"]
    z = ROOT / ECFG["paths"]["colab_bundle"]
    if z.exists():
        names = zipfile.ZipFile(z).namelist()
        for n in ("climate_slm/training/dpo_train.py", "climate_slm/training/select_dpo.py", "climate_slm/training/dpo_final_test.py",
                  "configs/phase4.yaml", "data/instruction/dpo_train.jsonl"):
            assert n in names, n


def test_notebook_phase4_cells():
    nb = json.loads((ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    p4 = [c for c in code if c.lstrip().startswith("# P4-")]
    ids = [c.split(".")[0].replace("# P4-", "") for c in p4]
    assert ids == [str(i) for i in range(11)]
    by = dict(zip(ids, p4))
    assert all(len(c.splitlines()) <= 15 for c in p4)
    assert {i for i, c in by.items() if "test" in c.lower()} == {"9"}                         # only the one clearly labelled cell
    assert "FINAL TEST" in by["9"] and "dpo_final_test" in by["9"]
    assert [c for c in code if "dpo_final_test" in c] == [by["9"]]
    for i in ("3", "4", "6", "7"):
        assert "--split test" not in by[i] and "dpo_final_test" not in by[i]
    assert "--split val" in by["4"] and "--split val" in by["7"]
    assert "--group grid" in by["3"] and "--group repeats" in by["6"] and "--group grid" in by["4"] and "--group repeats" in by["7"]
    assert "select_dpo --freeze" in by["8"] and "select_dpo" in by["5"] and "--freeze" not in by["5"]
    assert "dpo_train --check" in by["2"] and "phase4_start.zip" in by["2"] and "torchao" in by["1"]
    assert "phase4-results" in by["10"] and "phase4-adapters" in by["10"]
    assert code.index(by["9"]) > code.index(by["8"]) > code.index(by["6"]) > code.index(by["5"]) > code.index(by["3"])
    assert "dpo_final_test" not in " ".join(c for c in code if not c.startswith("# P4-"))


@needs("notebooks/README_COLAB.md")
def test_readme_colab_has_the_phase4_cell_order_with_time_estimates():
    text = (ROOT / "notebooks" / "README_COLAB.md").read_text(encoding="utf-8")
    for cell in [f"| P4-{i} |" for i in range(11)]:
        assert cell in text, cell
    assert "phase4_start.zip" in text and "time (estimate)" in text
