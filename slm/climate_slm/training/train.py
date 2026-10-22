"""Train LoRA / QLoRA runs on Colab (GPU). Never run on the laptop: torch, transformers, peft are imported only inside
the functions that need them.

    python -m climate_slm.training.train --group grid        # the 4 grid runs, each in its own process
    python -m climate_slm.training.train --group repeats     # the winning configuration with seeds 43 and 44
    python -m climate_slm.training.train --group q15         # one run on the 1.5B model (winning method and lr)
    python -m climate_slm.training.train --run lora_fp16_lr2e-4

Reads only the train and val split files. The test split is never opened here.
Loss: completion tokens only (prompt tokens have label -100), summed over the tokens of one optimiser step and
divided by their count. Precision: base in float16 (or 4-bit), LoRA weights in float32, float16 autocast and a GradScaler."""
import argparse
import datetime
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config
from climate_slm.evaluation.scoring import file_sha256, load_items
from climate_slm.quant import bnb_config
from climate_slm.training import grid, mlflow_log, schedule, selection_io
from climate_slm.training.masking import IGNORE_INDEX, loss_token_count, pad_batch
from climate_slm.training.prep import encode_rows

LIB_PACKAGES = ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "mlflow", "tokenizers",
                "huggingface_hub", "numpy")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def library_versions() -> dict:
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in LIB_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def resolve_runs(group: str, cfg3: dict) -> list[dict]:
    if group == "grid":
        return grid.grid_runs(cfg3)
    winner = selection_io.winner_run(cfg3)
    if group == "repeats":
        return grid.repeat_runs(cfg3, winner)
    if group == "q15":
        return [grid.q15_run(cfg3, winner)]
    raise ValueError(f"unknown group {group!r}")


def find_run(name: str, cfg3: dict) -> dict:
    """A run by its name: grid runs always, repeat and q15 runs once a winner is saved."""
    known = list(grid.grid_runs(cfg3))
    try:
        winner = selection_io.winner_run(cfg3)
        known += grid.repeat_runs(cfg3, winner) + [grid.q15_run(cfg3, winner)]
    except SystemExit:
        pass
    for r in known:
        if r["name"] == name:
            return r
    raise SystemExit(f"unknown run {name!r}. Known: {[r['name'] for r in known]}")


# ---- model -----------------------------------------------------------------------------------------------------------
def load_model(run: dict, cfg3: dict, torch):
    from transformers import AutoModelForCausalLM
    method = cfg3["methods"][run["method"]]
    dtype = getattr(torch, cfg3["train"]["dtype"])
    kw = {}
    if method["quantize"] == "nf4":
        kw["quantization_config"] = bnb_config(method, torch)       # the same helper the 4-bit evaluation uses
        kw["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(run["model_id"], dtype=dtype, **kw)
    except TypeError:                                         # older transformers call the argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(run["model_id"], torch_dtype=dtype, **kw)
    if method["quantize"] is None:
        model = model.to("cuda")
    model.config.use_cache = False
    return model


def add_lora(model, run: dict, cfg3: dict, torch):
    """LoRA on all linear layers of the block. LoRA weights in float32."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    method, lo = cfg3["methods"][run["method"]], cfg3["lora"]
    gc_kwargs = {"use_reentrant": False}
    if method.get("prepare_kbit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=run["gradient_checkpointing"],
                                                gradient_checkpointing_kwargs=gc_kwargs)
    elif run["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                                             bias=lo["bias"], target_modules=lo["target_modules"], task_type="CAUSAL_LM"))
    for p in model.parameters():                              # fp16 trainable weights would break the GradScaler
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


# ---- loss ------------------------------------------------------------------------------------------------------------
def micro_loss(model, batch: dict, torch, F):
    """(sum of token losses, number of loss tokens) for one padded micro-batch. Only completion tokens count."""
    dev = next(model.parameters()).device
    ids = torch.tensor(batch["input_ids"], device=dev)
    attn = torch.tensor(batch["attention_mask"], device=dev)
    labels = torch.tensor(batch["labels"], device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=attn).logits
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    selected = logits[:, :-1, :][mask]                        # only the completion positions, small
    loss_sum = F.cross_entropy(selected.float(), shift_labels[mask], reduction="sum")
    return loss_sum, int(mask.sum().item())


def val_loss(model, examples: list[dict], micro: int, pad_id: int, torch, F) -> float:
    """Mean loss per completion token on the val examples."""
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), micro):
            s, n = micro_loss(model, pad_batch(examples[i:i + micro], pad_id), torch, F)
            total, tokens = total + float(s), tokens + n
    model.train()
    return total / max(tokens, 1)


def dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2 ** 20


# ---- one run -----------------------------------------------------------------------------------------------------------
def train_one(run: dict, cfg3: dict, eval_cfg: dict) -> dict:
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    t, name = cfg3["train"], run["name"]
    if t["skip_if_done"] and grid.run_is_done(cfg3, run):
        print(f"[{name}] already done, skipped")
        grid.clear_stale_error(cfg3, name)                # an old failed attempt must not look like a current error
        return json.loads(grid.run_log_path(cfg3, name).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Colab: Runtime > Change runtime type > T4 GPU.")
    if cfg3["methods"][run["method"]]["quantize"]:
        import bitsandbytes  # noqa: F401  (fail now, not after the model download)
    t0 = time.perf_counter()
    started = utc_now()
    torch.cuda.reset_peak_memory_stats()
    seed = run["seed"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)

    # data: train and val only
    paths = eval_cfg["paths"]["splits"]
    train_path, val_path = REPO_ROOT / paths["train"], REPO_ROOT / paths["val"]
    tok = AutoTokenizer.from_pretrained(run["model_id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    template = lambda msgs, gen: tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)  # noqa: E731
    tokenize = lambda text: tok(text, add_special_tokens=False)["input_ids"]                              # noqa: E731
    train_rows = schedule.repeat_refusals(load_items(train_path), t["refusal_repeat"])
    tr = encode_rows(train_rows, template, tokenize, t["max_seq_len"], tok.eos_token or "")
    va = encode_rows(load_items(val_path), template, tokenize, t["max_seq_len"], tok.eos_token or "")
    print(f"[{name}] token lengths (prompt + completion) train: {tr['stats']}  val: {va['stats']}")
    print(f"[{name}] dropped: train {tr['dropped_ids']}  val {va['dropped_ids']}")
    train_ex, val_ex = tr["examples"], va["examples"]

    # MLflow is started here; if it fails the training goes on without logging (see mlflow_log)
    libs = library_versions()
    gpu = {"name": torch.cuda.get_device_name(0), "total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20}
    n_per_epoch = schedule.steps_per_epoch(len(train_ex), t["effective_batch_size"])
    total_steps = n_per_epoch * t["epochs"]
    params = {"run": name, "arm": run["arm"], "method": run["method"], "lr": run["lr"], "seed": seed,
              "model_id": run["model_id"], "epochs": t["epochs"], "effective_batch_size": t["effective_batch_size"],
              "micro_batch_size": t["micro_batch_size"], "warmup_ratio": t["warmup_ratio"], "lr_schedule": t["lr_schedule"],
              "weight_decay": t["weight_decay"], "max_grad_norm": t["max_grad_norm"], "max_seq_len": t["max_seq_len"],
              "refusal_repeat": t["refusal_repeat"], "gradient_checkpointing": run["gradient_checkpointing"],
              "dtype": t["dtype"], "lora_r": cfg3["lora"]["r"], "lora_alpha": cfg3["lora"]["alpha"],
              "lora_dropout": cfg3["lora"]["dropout"], "lora_targets": ",".join(cfg3["lora"]["target_modules"]),
              "n_train_examples": len(train_ex), "n_val_examples": len(val_ex), "steps_per_epoch": n_per_epoch,
              "total_steps": total_steps, "gpu": gpu["name"], "train_sha256": file_sha256(train_path),
              "val_sha256": file_sha256(val_path), **{f"lib_{k}": v for k, v in libs.items()}}
    tracker = mlflow_log.start_tracker(cfg3, name, params, {"phase": "3", "method": run["method"], "arm": run["arm"]})
    run_id = tracker.run_id

    log = {"run_name": name, "arm": run["arm"], "method": run["method"], "lr": run["lr"], "seed": seed,
           "model_id": run["model_id"], "mlflow_run_id": run_id, "started_utc": started, "complete": False,
           "params": params, "gpu": gpu, "library_versions": libs, "length_stats": {"train": tr["stats"], "val": va["stats"]},
           "dropped_train_ids": tr["dropped_ids"], "dropped_val_ids": va["dropped_ids"],
           "data_sha256": {"train": params["train_sha256"], "val": params["val_sha256"]},
           "steps": [], "epochs": []}
    log_path = grid.run_log_path(cfg3, name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def save_log():
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")

    try:
        model = add_lora(load_model(run, cfg3, torch), run, cfg3, torch)
        trainable, total = model.get_nb_trainable_parameters()
        log.update(trainable_params=trainable, total_params=total, trainable_share=trainable / total)
        print(f"[{name}] trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
        tracker.log_metrics({"trainable_params": trainable, "trainable_share": trainable / total})
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=run["lr"],
                                weight_decay=t["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=t["grad_scaler_init_scale"])
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        step = 0
        t_loop = time.perf_counter()
        for epoch in range(1, t["epochs"] + 1):
            order = schedule.epoch_order(len(train_ex), seed, epoch)
            epoch_losses = []
            for group in schedule.optimizer_steps(order, t["effective_batch_size"], t["micro_batch_size"]):
                lr = schedule.lr_at_step(step, total_steps, run["lr"], t["warmup_ratio"])
                for g in opt.param_groups:
                    g["lr"] = lr
                n_tokens = sum(loss_token_count(train_ex[i]["labels"]) for mb in group for i in mb)
                step_loss = 0.0
                for mb in group:
                    s, _ = micro_loss(model, pad_batch([train_ex[i] for i in mb], pad_id), torch, F)
                    scaler.scale(s / n_tokens).backward()
                    step_loss += float(s.detach())
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(trainable_params, t["max_grad_norm"]))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped = scaler.get_scale() < scale_before   # overflow: the optimiser step was skipped
                opt.zero_grad(set_to_none=True)
                step += 1
                step_loss /= n_tokens
                epoch_losses.append(step_loss)
                log["steps"].append({"step": step, "epoch": epoch, "lr": lr, "loss": step_loss, "grad_norm": gn,
                                     "skipped": skipped})
                metrics = {"train_loss": step_loss, "learning_rate": lr}
                if math.isfinite(gn):                         # an overflow step has no usable norm
                    metrics["grad_norm"] = gn
                tracker.log_metrics(metrics, step=step)
                print(f"[{name}] step {step}/{total_steps} epoch {epoch} loss {step_loss:.4f} lr {lr:.2e} "
                      f"grad_norm {gn:.2f}{' SKIPPED' if skipped else ''}")
            vl = val_loss(model, val_ex, t["micro_batch_size"], pad_id, torch, F)
            out_dir = grid.adapter_dir(cfg3, name, epoch)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            size = dir_mb(out_dir)
            log["epochs"].append({"epoch": epoch, "val_loss": vl, "train_loss_mean": sum(epoch_losses) / len(epoch_losses),
                                  "adapter_dir": str(out_dir.relative_to(REPO_ROOT)), "adapter_mb": size})
            tracker.log_metrics({"val_loss": vl, "epoch_train_loss": log["epochs"][-1]["train_loss_mean"],
                                "adapter_mb": size}, step=epoch)
            print(f"[{name}] epoch {epoch}: val loss {vl:.4f}, adapter {size:.1f} MB -> {out_dir}")
            save_log()
        log.update(skipped_steps=sum(1 for x in log["steps"] if x["skipped"]),
                   train_seconds=time.perf_counter() - t_loop, total_seconds=time.perf_counter() - t0,
                   peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                   peak_gpu_reserved_mb=torch.cuda.max_memory_reserved() / 2 ** 20,
                   finished_utc=utc_now(), complete=True)
        tracker.log_metrics({"peak_gpu_mem_mb": log["peak_gpu_mem_mb"], "train_seconds": log["train_seconds"],
                            "skipped_steps": log["skipped_steps"],
                            "total_seconds": log["total_seconds"], "adapter_mb_per_epoch": log["epochs"][-1]["adapter_mb"]})
        save_log()
        if grid.clear_stale_error(cfg3, name):            # the run succeeded: an error file of an earlier attempt is stale
            print(f"[{name}] removed the stale {grid.ERROR_FILE} of an earlier failed attempt")
        tracker.end()
        print(f"[{name}] done in {log['total_seconds']:.0f} s, peak GPU memory {log['peak_gpu_mem_mb']:.0f} MB, "
              f"skipped optimiser steps {log['skipped_steps']}")
        return log
    except BaseException:
        tracker.end("FAILED")
        raise


# ---- command line --------------------------------------------------------------------------------------------------------
def run_group(runs: list[dict], cfg3: dict) -> list[str]:
    """Each run in its own process (clean GPU memory, and one failing run cannot stop the others). Returns failed names."""
    failed = []
    for r in runs:
        if cfg3["train"]["skip_if_done"] and grid.run_is_done(cfg3, r):
            print(f"[{r['name']}] already done, skipped")
            grid.clear_stale_error(cfg3, r["name"])
            continue
        print(f"=== training {r['name']} ===", flush=True)
        code = subprocess.call([sys.executable, "-m", "climate_slm.training.train", "--run", r["name"]])
        if code != 0:
            failed.append(r["name"])
    if failed:
        print(f"FAILED RUNS: {failed}. See outputs/runs/<run>/train_error.txt. The other runs are not affected.")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description="LoRA / QLoRA training (Colab GPU).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--group", choices=["grid", "repeats", "q15"])
    g.add_argument("--run", help="one run by name")
    a = ap.parse_args(argv)
    cfg3, eval_cfg = load_phase3_config(), load_eval_config()
    problems = grid.validate(cfg3)
    if problems:
        raise SystemExit("config problems: " + "; ".join(problems))
    if a.group:
        run_group(resolve_runs(a.group, cfg3), cfg3)
        return 0
    run = find_run(a.run, cfg3)
    try:
        train_one(run, cfg3, eval_cfg)
    except BaseException:
        err = grid.error_file_path(cfg3, run["name"])
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
