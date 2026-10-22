"""Phase 4: DPO on top of the frozen SFT adapter (0.5B). Runs on Colab (GPU). torch, transformers and peft are imported
only inside the functions that need them, so the laptop (and the tests) never need them.

    python -m climate_slm.training.dpo_train --check            # no GPU: pairs file, start adapters, config (cell P4-2)
    python -m climate_slm.training.dpo_train --group grid       # the 4 grid runs, each in its own process
    python -m climate_slm.training.dpo_train --group repeats    # the winning configuration with seeds 43 and 44
    python -m climate_slm.training.dpo_train --run dpo_b0.1_lr5e-5

The policy starts as the SFT adapter (trainable LoRA, float32 weights) on the float16 base. The reference model is the
SAME SFT adapter before the first update: its sequence log-probabilities are computed ONCE per distinct pair before
training (so only one model sits in memory) and saved per seed (outputs/runs_dpo/reference_logps_seed<k>.json). The type 3
repeats reuse them. Loss: sigmoid DPO (dpo_loss.py), no SFT term. Reads sft_train-derived pairs only; no val or test file.
Same optimiser recipe as Phase 3: AdamW, cosine schedule with warm-up, float16 autocast, GradScaler, clipping at 1.0."""
import argparse
import json
import math
import subprocess
import sys
import time
import traceback

from climate_slm.config import REPO_ROOT, load_phase3_config, load_phase4_config
from climate_slm.training import dpo_data, dpo_grid, grid, mlflow_log, schedule, selection_io
from climate_slm.training.dpo_loss import Meter, dpo_loss, dpo_terms
from climate_slm.training.masking import IGNORE_INDEX
from climate_slm.training.train import dir_mb, library_versions, utc_now


# ---- model pieces (torch) --------------------------------------------------------------------------------------------
def sequence_logps(logits, labels, torch, F):
    """Sum of the log-probabilities of the answer tokens of each row (prompt and padding positions have label -100).
    Computed from float32 log-softmax on the answer positions only."""
    shift = labels[:, 1:]
    mask = shift != IGNORE_INDEX
    selected = logits[:, :-1, :][mask].float()
    token_lp = -F.cross_entropy(selected, shift[mask], reduction="none")
    rows = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(mask)[mask]
    return torch.zeros(labels.shape[0], device=labels.device, dtype=token_lp.dtype).index_add(0, rows, token_lp)


def pair_logps(model, pair_examples: list[dict], pad_id: int, torch, F):
    """(chosen, rejected) sequence log-probs of a micro-batch of pairs, each a tensor of length m."""
    batch, m = dpo_data.stack_pairs(pair_examples, pad_id)
    dev = next(model.parameters()).device
    ids, attn = torch.tensor(batch["input_ids"], device=dev), torch.tensor(batch["attention_mask"], device=dev)
    labels = torch.tensor(batch["labels"], device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=attn).logits
    lp = sequence_logps(logits, labels, torch, F)
    return lp[:m], lp[m:]


def compute_references(model, encoded: dict, micro: int, pad_id: int, torch, F) -> dict[str, list[float]]:
    """Reference log-probs of every distinct pair, with the policy still equal to the SFT adapter (eval mode, no grad)."""
    model.eval()
    ids = list(encoded["examples"])
    out: dict[str, list[float]] = {}
    with torch.no_grad():
        for i in range(0, len(ids), micro):
            chunk = ids[i:i + micro]
            c, r = pair_logps(model, [encoded["examples"][d] for d in chunk], pad_id, torch, F)
            for d, cv, rv in zip(chunk, c.tolist(), r.tolist()):
                out[d] = [cv, rv]
    model.train()
    return out


def load_policy(run: dict, cfg4: dict, cfg3: dict, torch):
    """Base model in float16 on the GPU, the SFT adapter on top as trainable LoRA (weights in float32)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    dtype = getattr(torch, cfg4["train"]["dtype"])
    try:
        base = AutoModelForCausalLM.from_pretrained(run["model_id"], dtype=dtype)
    except TypeError:                                         # older transformers call the argument torch_dtype
        base = AutoModelForCausalLM.from_pretrained(run["model_id"], torch_dtype=dtype)
    base = base.to("cuda")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(dpo_grid.start_adapter_dir(run)), is_trainable=True)
    for p in model.parameters():                              # fp16 trainable weights would break the GradScaler
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


# ---- checks without a GPU ----------------------------------------------------------------------------------------------
def check_inputs(cfg4: dict, cfg3: dict, seeds: list[int] | None = None) -> list[str]:
    """Print and return problems: config, pairs file (sha256, counts), start adapters (present, hash, LoRA settings)."""
    problems = dpo_grid.validate(cfg4)
    pairs = dpo_data.read_pairs(cfg4)
    c = dpo_data.count_pairs(pairs)
    print(f"pairs file ok (sha256 {dpo_data.pairs_sha256(cfg4)[:16]}...): {c}")
    for sd in seeds or [cfg4["sweep"]["seed"]] + list(cfg4["sweep"]["repeat_seeds"]):
        run = dpo_grid.make_run(cfg4, cfg4["sweep"]["betas"][0], cfg4["sweep"]["learning_rates"][0], sd)
        d = dpo_grid.start_adapter_dir(run)
        try:
            sha = dpo_grid.check_start_adapter(run)
            acfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
            bad = dpo_grid.adapter_config_problems(acfg, cfg3, run["model_id"])
            print(f"seed {sd}: start adapter {d.relative_to(REPO_ROOT)} sha256 {sha[:16]}... " + ("ok" if not bad else f"PROBLEMS {bad}"))
            problems += [f"seed {sd}: {b}" for b in bad]
        except SystemExit as err:
            problems.append(f"seed {sd}: {err}")
            print(f"seed {sd}: {err}")
    print("config problems:" if problems else "all checks passed", problems if problems else "")
    return problems


# ---- one run ---------------------------------------------------------------------------------------------------------------
def train_one(run: dict, cfg4: dict, cfg3: dict) -> dict:
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    t, name, seed, beta = cfg4["train"], run["name"], run["seed"], run["beta"]
    if t["skip_if_done"] and grid.run_is_done(cfg4, run):
        print(f"[{name}] already done, skipped")
        grid.clear_stale_error(cfg4, name)                # an old failed attempt must not look like a current error
        return json.loads(grid.run_log_path(cfg4, name).read_text(encoding="utf-8"))
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

    pairs = dpo_data.read_pairs(cfg4)
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
    params = {"run": name, "method": "dpo_sigmoid", "beta": beta, "lr": run["lr"], "seed": seed, "model_id": run["model_id"],
              "start_adapter": run["start_adapter"], "start_adapter_sha256": adapter_sha,
              "pairs_sha256": dpo_data.pairs_sha256(cfg4), "epochs": t["epochs"],
              "effective_batch_size": t["effective_batch_size"], "micro_batch_size": t["micro_batch_size"],
              "warmup_ratio": t["warmup_ratio"], "lr_schedule": t["lr_schedule"], "weight_decay": t["weight_decay"],
              "max_grad_norm": t["max_grad_norm"], "max_seq_len": t["max_seq_len"], "dtype": t["dtype"],
              "n_lines": len(lines), "n_distinct_pairs_used": len(enc["examples"]),
              "dropped_distinct_pairs": len(enc["dropped_distinct_ids"]),
              **{f"pairs_{k}": v for k, v in counts.items()},
              "steps_per_epoch": n_per_epoch, "total_steps": total_steps, "gpu": gpu["name"],
              **{f"lib_{k}": v for k, v in libs.items()}}
    tracker = mlflow_log.start_tracker(cfg4, name, params, {"phase": "4", "method": "dpo", "control": run["control"]})
    log = {"run_name": name, "beta": beta, "lr": run["lr"], "seed": seed, "model_id": run["model_id"],
           "mlflow_run_id": tracker.run_id, "started_utc": started, "complete": False, "params": params, "gpu": gpu,
           "library_versions": libs, "pair_counts": counts, "length_stats": enc["stats"],
           "dropped_distinct_ids": enc["dropped_distinct_ids"], "dropped_lines": enc["dropped_lines"],
           "start_adapter": run["start_adapter"], "start_adapter_sha256": adapter_sha, "pairs_sha256": params["pairs_sha256"],
           "steps": [], "epochs": []}
    log_path = grid.run_log_path(cfg4, name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def save_log():
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")

    try:
        model = load_policy(run, cfg4, cfg3, torch)
        trainable, total = model.get_nb_trainable_parameters()
        log.update(trainable_params=trainable, total_params=total, trainable_share=trainable / total)
        print(f"[{name}] trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
        tracker.log_metrics({"trainable_params": trainable, "trainable_share": trainable / total})

        # reference log-probs: ONCE per distinct pair (and once per seed: a saved file is reused by the other runs)
        key = dpo_data.reference_key(params["pairs_sha256"], adapter_sha, run["model_id"], t["max_seq_len"])
        cache_path = dpo_grid.reference_cache_path(cfg4, seed)
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
                    pc, pr = pair_logps(model, [enc["examples"][x["distinct_id"]] for x in ls], pad_id, torch, F)
                    rc = torch.tensor([dpo_data.reference_for(refs, x)[0] for x in ls], device=pc.device)
                    rr = torch.tensor([dpo_data.reference_for(refs, x)[1] for x in ls], device=pc.device)
                    loss_sum = dpo_loss(pc, pr, rc, rr, beta, logsigmoid=F.logsigmoid).sum()
                    scaler.scale(loss_sum / n_pairs).backward()
                    for a, b, c_, d in zip(pc.detach().tolist(), pr.detach().tolist(), rc.tolist(), rr.tolist()):
                        terms = dpo_terms(a, b, c_, d, beta)
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
                                     "reward_accuracy": s["reward_accuracy"], "mean_margin": s["mean_margin"],
                                     "grad_norm": gn, "skipped": skipped})
                metrics = {"dpo_loss": s["loss"], "reward_accuracy": s["reward_accuracy"], "mean_margin": s["mean_margin"],
                           "learning_rate": lr}
                if math.isfinite(gn):
                    metrics["grad_norm"] = gn
                tracker.log_metrics(metrics, step=step)
                print(f"[{name}] step {step}/{total_steps} epoch {epoch} loss {s['loss']:.4f} acc {s['reward_accuracy']:.2f} "
                      f"margin {s['mean_margin']:.3f} lr {lr:.2e} grad_norm {gn:.2f}{' SKIPPED' if skipped else ''}")
            e = ep_meter.summary()
            out_dir = grid.adapter_dir(cfg4, name, epoch)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            size = dir_mb(out_dir)
            log["epochs"].append({"epoch": epoch, "dpo_loss": e["loss"], "reward_accuracy": e["reward_accuracy"],
                                  "mean_margin": e["mean_margin"], "mean_reward_chosen": e["mean_reward_chosen"],
                                  "mean_reward_rejected": e["mean_reward_rejected"], "n_pairs": e["n_pairs"],
                                  "adapter_dir": str(out_dir.relative_to(REPO_ROOT)), "adapter_mb": size})
            tracker.log_metrics({"epoch_dpo_loss": e["loss"], "epoch_reward_accuracy": e["reward_accuracy"],
                                 "epoch_mean_margin": e["mean_margin"], "adapter_mb": size}, step=epoch)
            print(f"[{name}] epoch {epoch}: dpo loss {e['loss']:.4f}, reward accuracy {e['reward_accuracy']:.3f}, "
                  f"mean margin {e['mean_margin']:.3f}, adapter {size:.1f} MB -> {out_dir}")
            save_log()
        log.update(skipped_steps=sum(1 for x in log["steps"] if x["skipped"]),
                   train_seconds=time.perf_counter() - t_loop, total_seconds=time.perf_counter() - t0,
                   peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                   peak_gpu_reserved_mb=torch.cuda.max_memory_reserved() / 2 ** 20, finished_utc=utc_now(), complete=True)
        tracker.log_metrics({"peak_gpu_mem_mb": log["peak_gpu_mem_mb"], "train_seconds": log["train_seconds"],
                             "skipped_steps": log["skipped_steps"], "total_seconds": log["total_seconds"],
                             "adapter_mb_per_epoch": log["epochs"][-1]["adapter_mb"]})
        save_log()
        if grid.clear_stale_error(cfg4, name):            # the run succeeded: an error file of an earlier attempt is stale
            print(f"[{name}] removed the stale {grid.ERROR_FILE} of an earlier failed attempt")
        tracker.end()
        print(f"[{name}] done in {log['total_seconds']:.0f} s, peak GPU memory {log['peak_gpu_mem_mb']:.0f} MB, "
              f"skipped optimiser steps {log['skipped_steps']}")
        return log
    except BaseException:
        tracker.end("FAILED")
        raise


# ---- command line --------------------------------------------------------------------------------------------------------------
def resolve_runs(group: str, cfg4: dict) -> list[dict]:
    if group == "grid":
        return dpo_grid.grid_runs(cfg4)
    sel = selection_io.read_selection(cfg4)
    return dpo_grid.repeat_runs(cfg4, dpo_grid.winner_run(cfg4, sel))


def run_group(runs: list[dict], cfg4: dict) -> list[str]:
    """Each run in its own process (clean GPU memory; one failing run cannot stop the others). Returns failed names."""
    failed = []
    for r in runs:
        if cfg4["train"]["skip_if_done"] and grid.run_is_done(cfg4, r):
            print(f"[{r['name']}] already done, skipped")
            grid.clear_stale_error(cfg4, r["name"])
            continue
        print(f"=== DPO training {r['name']} ===", flush=True)
        if subprocess.call([sys.executable, "-m", "climate_slm.training.dpo_train", "--run", r["name"]]) != 0:
            failed.append(r["name"])
    if failed:
        print(f"FAILED RUNS: {failed}. See outputs/runs_dpo/<run>/train_error.txt. The other runs are not affected.")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description="DPO training on top of the SFT adapter (Colab GPU).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--group", choices=["grid", "repeats"])
    g.add_argument("--run", help="one run by name")
    g.add_argument("--check", action="store_true", help="check config, pairs file and start adapters (no GPU)")
    a = ap.parse_args(argv)
    cfg4, cfg3 = load_phase4_config(), load_phase3_config()
    if a.check:
        return 1 if check_inputs(cfg4, cfg3) else 0
    problems = dpo_grid.validate(cfg4)
    if problems:
        raise SystemExit("config problems: " + "; ".join(problems))
    if a.group:
        run_group(resolve_runs(a.group, cfg4), cfg4)
        return 0
    sel = selection_io.read_selection(cfg4) if selection_io.selection_path(cfg4).exists() else None
    run = dpo_grid.find_run(a.run, cfg4, sel)
    try:
        train_one(run, cfg4, cfg3)
    except BaseException:
        err = grid.error_file_path(cfg4, run["name"])
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
