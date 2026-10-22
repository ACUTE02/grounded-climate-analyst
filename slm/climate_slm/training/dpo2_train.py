"""Phase 4b: DPO v2 on top of the frozen SFT adapter (0.5B). Runs on Colab (GPU). torch, transformers and peft are imported
only inside the functions that need them, so the laptop (and the tests) never need them.

    python -m climate_slm.training.dpo2_train --check          # no GPU: config, pair files, start adapters (cell P4b-2)
    python -m climate_slm.training.dpo2_train --group grid     # the 8 grid runs, each in its own process
    python -m climate_slm.training.dpo2_train --group repeats  # the winning configuration with seeds 43 and 44
    python -m climate_slm.training.dpo2_train --run dpo2_main_a0.5_lr1e-5

Same method as v1 (climate_slm/training/dpo_train.py, whose model helpers are reused): the policy starts as the SFT adapter
(trainable LoRA, float32 weights) on the float16 base; the reference log-probabilities of the same SFT adapter are computed
ONCE per distinct pair before the first update and cached per pair file and seed; no second model in memory.
What is new in v2:
  * loss = sigmoid DPO + alpha * NLL of the CHOSEN text, per token (RPO style, dpo_loss.rpo_loss); alpha in {0.5, 1.0}
  * the pair file is one of two sets (dpo2_main.jsonl, dpo2_mix.jsonl), sha256-checked before every run
  * beta 0.1 fixed, learning rate 1e-5 or 2e-5, effective batch 8 pairs (micro batch 1), at most 3 epochs, adapter per epoch.
Reads pair files made from sft_train only; no val or test file is opened here. The v1 trainer was not changed."""
import argparse
import json
import math
import subprocess
import sys
import time
import traceback

from climate_slm.config import REPO_ROOT, load_phase3_config
from climate_slm.training import dpo2_grid, dpo_data, dpo_grid, grid, mlflow_log, schedule, selection_io
from climate_slm.training.dpo_loss import Meter, rpo_loss, rpo_terms
from climate_slm.training.dpo_train import compute_references, dir_mb, library_versions, load_policy, pair_logps, utc_now


def pair_file_paths(cfg: dict, pair_set: str) -> tuple:
    f = cfg["pair_files"][pair_set]
    return REPO_ROOT / cfg["paths"][f["pairs"]], REPO_ROOT / cfg["paths"][f["sha256"]]


def read_set(cfg: dict, pair_set: str) -> list[dict]:
    """The lines of one pair set, after the sha256 check."""
    return dpo_data.read_pair_file(*pair_file_paths(cfg, pair_set))


def set_sha256(cfg: dict, pair_set: str) -> str:
    return pair_file_paths(cfg, pair_set)[1].read_text(encoding="utf-8").split()[0]


def check_inputs(cfg: dict, cfg3: dict) -> list[str]:
    """Print and return problems: config, both pair files (sha256, counts, steps), start adapters (hash, LoRA settings)."""
    problems = dpo2_grid.validate(cfg)
    t = cfg["train"]
    for ps in cfg["design"]["grid"]["pair_sets"]:
        try:
            pairs = read_set(cfg, ps)
        except SystemExit as err:
            problems.append(f"{ps}: {err}")
            print(f"{ps}: {err}")
            continue
        c = dpo_data.count_pairs(pairs)
        steps = schedule.steps_per_epoch(c["lines"], t["effective_batch_size"])
        print(f"{ps}: pair file ok (sha256 {set_sha256(cfg, ps)[:16]}...): {c}; {steps} steps per epoch, {steps * t['epochs']} per run")
    for sd in [cfg["sweep"]["seed"]] + list(cfg["sweep"]["repeat_seeds"]):
        run = dpo2_grid.make_run(cfg, cfg["design"]["grid"]["pair_sets"][0], cfg["design"]["grid"]["alphas"][0],
                                 cfg["design"]["grid"]["learning_rates"][0], sd)
        d = dpo_grid.start_adapter_dir(run)
        try:
            sha = dpo_grid.check_start_adapter(run)
            bad = dpo_grid.adapter_config_problems(json.loads((d / "adapter_config.json").read_text(encoding="utf-8")), cfg3,
                                                   run["model_id"])
            print(f"seed {sd}: start adapter {d.relative_to(REPO_ROOT)} sha256 {sha[:16]}... " + ("ok" if not bad else f"PROBLEMS {bad}"))
            problems += [f"seed {sd}: {b}" for b in bad]
        except SystemExit as err:
            problems.append(f"seed {sd}: {err}")
            print(f"seed {sd}: {err}")
    print("problems:" if problems else "all checks passed", problems if problems else "")
    return problems


# ---- one run ---------------------------------------------------------------------------------------------------------------
def train_one(run: dict, cfg: dict, cfg3: dict) -> dict:
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    t, name, seed, beta, alpha = cfg["train"], run["name"], run["seed"], run["beta"], run["alpha"]
    if t["skip_if_done"] and grid.run_is_done(cfg, run):
        print(f"[{name}] already done, skipped")
        grid.clear_stale_error(cfg, name)                 # an old failed attempt must not look like a current error
        return json.loads(grid.run_log_path(cfg, name).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Colab: Runtime > Change runtime type > T4 GPU.")
    adapter_sha = dpo_grid.check_start_adapter(run)
    bad = dpo_grid.adapter_config_problems(
        json.loads((dpo_grid.start_adapter_dir(run) / "adapter_config.json").read_text(encoding="utf-8")), cfg3, run["model_id"])
    if bad:
        raise SystemExit(f"start adapter does not match the Phase 3 LoRA settings: {bad}")
    t0, started = time.perf_counter(), utc_now()
    torch.cuda.reset_peak_memory_stats()
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)

    pairs = read_set(cfg, run["pair_set"])
    tok = AutoTokenizer.from_pretrained(run["model_id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    template = lambda msgs, gen: tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)  # noqa: E731
    tokenize = lambda text: tok(text, add_special_tokens=False)["input_ids"]                              # noqa: E731
    enc = dpo_data.encode_pairs(pairs, template, tokenize, t["max_seq_len"], tok.eos_token or "")
    lines = enc["lines"]
    print(f"[{name}] token lengths (prompt + answer, both sides) {enc['stats']}")
    print(f"[{name}] dropped (too long): {len(enc['dropped_distinct_ids'])} distinct pairs = {enc['dropped_lines']} lines "
          f"{enc['dropped_distinct_ids']}")
    counts = dpo_data.count_pairs(pairs)
    n_per_epoch = schedule.steps_per_epoch(len(lines), t["effective_batch_size"])
    total_steps = n_per_epoch * t["epochs"]
    libs = library_versions()
    gpu = {"name": torch.cuda.get_device_name(0), "total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20}
    pairs_sha = set_sha256(cfg, run["pair_set"])
    params = {"run": name, "method": "dpo_sigmoid_plus_nll_chosen", "pair_set": run["pair_set"], "pairs_sha256": pairs_sha,
              "alpha": alpha, "beta": beta, "lr": run["lr"], "seed": seed, "model_id": run["model_id"],
              "start_adapter": run["start_adapter"], "start_adapter_sha256": adapter_sha, "epochs": t["epochs"],
              "effective_batch_size": t["effective_batch_size"], "micro_batch_size": t["micro_batch_size"],
              "warmup_ratio": t["warmup_ratio"], "lr_schedule": t["lr_schedule"], "weight_decay": t["weight_decay"],
              "max_grad_norm": t["max_grad_norm"], "max_seq_len": t["max_seq_len"], "dtype": t["dtype"],
              "n_lines": len(lines), "n_distinct_pairs_used": len(enc["examples"]),
              "dropped_distinct_pairs": len(enc["dropped_distinct_ids"]), **{f"pairs_{k}": v for k, v in counts.items()},
              "steps_per_epoch": n_per_epoch, "total_steps": total_steps, "gpu": gpu["name"],
              **{f"lib_{k}": v for k, v in libs.items()}}
    tracker = mlflow_log.start_tracker(cfg, name, params, {"phase": "4b", "method": "dpo2", "control": run["control"],
                                                           "pair_set": run["pair_set"]})
    log = {"run_name": name, "pair_set": run["pair_set"], "alpha": alpha, "beta": beta, "lr": run["lr"], "seed": seed,
           "model_id": run["model_id"], "mlflow_run_id": tracker.run_id, "started_utc": started, "complete": False,
           "params": params, "gpu": gpu, "library_versions": libs, "pair_counts": counts, "length_stats": enc["stats"],
           "dropped_distinct_ids": enc["dropped_distinct_ids"], "dropped_lines": enc["dropped_lines"],
           "start_adapter": run["start_adapter"], "start_adapter_sha256": adapter_sha, "pairs_sha256": pairs_sha,
           "steps": [], "epochs": []}
    log_path = grid.run_log_path(cfg, name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def save_log():
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")

    try:
        model = load_policy(run, cfg, cfg3, torch)
        trainable, total = model.get_nb_trainable_parameters()
        log.update(trainable_params=trainable, total_params=total, trainable_share=trainable / total)
        print(f"[{name}] trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
        tracker.log_metrics({"trainable_params": trainable, "trainable_share": trainable / total})

        # reference log-probs: ONCE per distinct pair, cached per pair file and seed (the runs of one set share the file)
        key = dpo_data.reference_key(pairs_sha, adapter_sha, run["model_id"], t["max_seq_len"])
        cache_path = dpo2_grid.reference_cache_path(cfg, run)
        refs = dpo_data.load_reference_cache(cache_path, key, set(enc["examples"])) if t["reference_cache"] else None
        if refs is None:
            t_ref = time.perf_counter()
            refs = compute_references(model, enc, max(t["micro_batch_size"], 2), pad_id, torch, F)
            dpo_data.save_reference_cache(cache_path, key, refs)
            print(f"[{name}] reference log-probs of {len(refs)} distinct pairs computed in {time.perf_counter() - t_ref:.0f} s")
        else:
            print(f"[{name}] reference log-probs of {len(refs)} distinct pairs read from {cache_path.name}")
        log["reference_logps_file"] = str(cache_path.relative_to(REPO_ROOT))

        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=run["lr"], weight_decay=t["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=t["grad_scaler_init_scale"])
        step, t_loop = 0, time.perf_counter()
        for epoch in range(1, t["epochs"] + 1):
            order = schedule.epoch_order(len(lines), seed, epoch)
            ep_meter = Meter()
            for group in schedule.optimizer_steps(order, t["effective_batch_size"], t["micro_batch_size"]):
                lr = schedule.lr_at_step(step, total_steps, run["lr"], t["warmup_ratio"])
                for g in opt.param_groups:
                    g["lr"] = lr
                n_pairs = sum(len(mb) for mb in group)
                st_meter = Meter()
                for mb in group:
                    ls = [lines[i] for i in mb]
                    exs = [enc["examples"][x["distinct_id"]] for x in ls]
                    pc, pr = pair_logps(model, exs, pad_id, torch, F)
                    rc = torch.tensor([dpo_data.reference_for(refs, x)[0] for x in ls], device=pc.device)
                    rr = torch.tensor([dpo_data.reference_for(refs, x)[1] for x in ls], device=pc.device)
                    n_tok = [e["chosen"]["n_completion"] for e in exs]            # answer tokens of the CHOSEN text
                    loss_sum = rpo_loss(pc, pr, rc, rr, beta, alpha, torch.tensor(n_tok, device=pc.device, dtype=pc.dtype),
                                        logsigmoid=F.logsigmoid).sum()
                    scaler.scale(loss_sum / n_pairs).backward()
                    for a_, b_, c_, d_, n_ in zip(pc.detach().tolist(), pr.detach().tolist(), rc.tolist(), rr.tolist(), n_tok):
                        terms = rpo_terms(a_, b_, c_, d_, beta, alpha, n_)
                        st_meter.add(terms)
                        ep_meter.add(terms)
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(trainable_params, t["max_grad_norm"]))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped = scaler.get_scale() < scale_before   # overflow: the optimiser step was skipped
                opt.zero_grad(set_to_none=True)
                step += 1
                s = st_meter.summary()
                log["steps"].append({"step": step, "epoch": epoch, "lr": lr, "n_pairs": n_pairs, "loss": s["loss"],
                                     "dpo_loss": s["dpo_loss"], "nll": s["nll"], "reward_accuracy": s["reward_accuracy"],
                                     "mean_margin": s["mean_margin"], "grad_norm": gn, "skipped": skipped})
                metrics = {"loss": s["loss"], "dpo_loss": s["dpo_loss"], "nll_chosen": s["nll"],
                           "reward_accuracy": s["reward_accuracy"], "mean_margin": s["mean_margin"], "learning_rate": lr}
                if math.isfinite(gn):
                    metrics["grad_norm"] = gn
                tracker.log_metrics(metrics, step=step)
                print(f"[{name}] step {step}/{total_steps} epoch {epoch} loss {s['loss']:.4f} (dpo {s['dpo_loss']:.4f} + "
                      f"{alpha:g} x nll {s['nll']:.4f}) acc {s['reward_accuracy']:.2f} margin {s['mean_margin']:.3f} "
                      f"lr {lr:.2e} grad_norm {gn:.2f}{' SKIPPED' if skipped else ''}")
            e = ep_meter.summary()
            out_dir = grid.adapter_dir(cfg, name, epoch)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            size = dir_mb(out_dir)
            log["epochs"].append({"epoch": epoch, "loss": e["loss"], "dpo_loss": e["dpo_loss"], "nll": e["nll"],
                                  "reward_accuracy": e["reward_accuracy"], "mean_margin": e["mean_margin"],
                                  "mean_reward_chosen": e["mean_reward_chosen"], "mean_reward_rejected": e["mean_reward_rejected"],
                                  "n_pairs": e["n_pairs"], "adapter_dir": str(out_dir.relative_to(REPO_ROOT)), "adapter_mb": size})
            tracker.log_metrics({"epoch_loss": e["loss"], "epoch_dpo_loss": e["dpo_loss"], "epoch_nll_chosen": e["nll"],
                                 "epoch_reward_accuracy": e["reward_accuracy"], "epoch_mean_margin": e["mean_margin"],
                                 "adapter_mb": size}, step=epoch)
            print(f"[{name}] epoch {epoch}: loss {e['loss']:.4f} (dpo {e['dpo_loss']:.4f}, nll {e['nll']:.4f}), reward accuracy "
                  f"{e['reward_accuracy']:.3f}, mean margin {e['mean_margin']:.3f}, adapter {size:.1f} MB -> {out_dir}")
            save_log()
        log.update(skipped_steps=sum(1 for x in log["steps"] if x["skipped"]),
                   train_seconds=time.perf_counter() - t_loop, total_seconds=time.perf_counter() - t0,
                   peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                   peak_gpu_reserved_mb=torch.cuda.max_memory_reserved() / 2 ** 20, finished_utc=utc_now(), complete=True)
        tracker.log_metrics({"peak_gpu_mem_mb": log["peak_gpu_mem_mb"], "train_seconds": log["train_seconds"],
                             "skipped_steps": log["skipped_steps"], "total_seconds": log["total_seconds"],
                             "adapter_mb_per_epoch": log["epochs"][-1]["adapter_mb"]})
        save_log()
        if grid.clear_stale_error(cfg, name):             # the run succeeded: an error file of an earlier attempt is stale
            print(f"[{name}] removed the stale {grid.ERROR_FILE} of an earlier failed attempt")
        tracker.end()
        print(f"[{name}] done in {log['total_seconds']:.0f} s, peak GPU memory {log['peak_gpu_mem_mb']:.0f} MB, "
              f"skipped optimiser steps {log['skipped_steps']}")
        return log
    except BaseException:
        tracker.end("FAILED")
        raise


# ---- command line -----------------------------------------------------------------------------------------------------------
def resolve_runs(group: str, cfg: dict) -> list[dict]:
    if group == "grid":
        return dpo2_grid.grid_runs(cfg)
    return dpo2_grid.repeat_runs(cfg, dpo2_grid.winner_run(cfg, selection_io.read_selection(cfg)))


def run_group(runs: list[dict], cfg: dict) -> list[str]:
    """Each run in its own process (clean GPU memory; one failing run cannot stop the others). Returns failed names."""
    failed = []
    for r in runs:
        if cfg["train"]["skip_if_done"] and grid.run_is_done(cfg, r):
            print(f"[{r['name']}] already done, skipped")
            grid.clear_stale_error(cfg, r["name"])
            continue
        print(f"=== DPO v2 training {r['name']} ===", flush=True)
        if subprocess.call([sys.executable, "-m", "climate_slm.training.dpo2_train", "--run", r["name"]]) != 0:
            failed.append(r["name"])
    if failed:
        print(f"FAILED RUNS: {failed}. See outputs/runs_dpo2/<run>/train_error.txt. The other runs are not affected.")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description="DPO v2 training on top of the SFT adapter (Colab GPU).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--group", choices=["grid", "repeats"])
    g.add_argument("--run", help="one run by name")
    g.add_argument("--check", action="store_true", help="check config, pair files and start adapters (no GPU)")
    a = ap.parse_args(argv)
    cfg, cfg3 = dpo2_grid.runtime_config(), load_phase3_config()
    if a.check:
        return 1 if check_inputs(cfg, cfg3) else 0
    problems = dpo2_grid.validate(cfg)
    if problems:
        raise SystemExit("config problems: " + "; ".join(problems))
    if a.group:
        run_group(resolve_runs(a.group, cfg), cfg)
        return 0
    sel = selection_io.read_selection(cfg) if selection_io.selection_path(cfg).exists() else None
    run = dpo2_grid.find_run(a.run, cfg, sel)
    try:
        train_one(run, cfg, cfg3)
    except BaseException:
        err = grid.error_file_path(cfg, run["name"])
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
