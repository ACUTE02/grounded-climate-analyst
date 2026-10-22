"""Evaluate every epoch adapter of a run group with the existing evaluator, on VAL only, and log the numbers into
the MLflow run of that training.

    python -m climate_slm.training.eval_adapters --group grid --split val
    python -m climate_slm.training.eval_adapters --group grid --split val --on-4bit-base   # QLoRA adapters on their own base

The evaluator is called as a separate process (python -m climate_slm.evaluation.evaluate), so no evaluation code is
repeated here and the GPU is freed after each adapter. `--split` accepts val only: the test split cannot be
evaluated by accident from this loop. The one test run per final model is in climate_slm.training.final_test."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config
from climate_slm.evaluation.evaluate import default_out
from climate_slm.training import grid, mlflow_log, summary

ALLOWED_SPLITS = ("val",)


def eval_command(adapter: Path, name: str, split: str, base_model_id: str | None, load_4bit: bool = False) -> list[str]:
    """The evaluator command for one adapter. The base model id is always given, so an adapter is never
    evaluated on the wrong base."""
    cmd = [sys.executable, "-m", "climate_slm.evaluation.evaluate", "--model", str(adapter), "--split", split,
           "--name", name]
    if base_model_id:
        cmd += ["--base-model-id", base_model_id]
    if load_4bit:
        cmd.append("--load-4bit")
    return cmd


def plan(runs: list[dict], cfg3: dict, eval_cfg: dict, split: str, on_4bit: bool = False) -> list[dict]:
    """One entry per epoch adapter: run, epoch, adapter dir, evaluator name, expected eval file, base model.
    With on_4bit only the QLoRA runs are listed; they are evaluated on the 4-bit base and get the suffix __on4bit."""
    items = []
    for r in runs:
        if on_4bit and cfg3["methods"][r["method"]]["quantize"] is None:
            continue
        for e in grid.epochs(cfg3):
            name = grid.candidate_name(r["name"], e, on_4bit)
            items.append({"run": r, "epoch": e, "adapter": grid.adapter_dir(cfg3, r["name"], e), "name": name,
                          "eval_file": default_out(eval_cfg, name, split), "base_model_id": r["model_id"],
                          "load_4bit": on_4bit})
    return items


def runs_for_group(group: str, cfg3: dict) -> list[dict]:
    from climate_slm.training.train import resolve_runs
    return resolve_runs(group, cfg3)


def log_to_mlflow(item: dict, cfg3: dict, eval_cfg: dict, split: str) -> bool:
    """Put the val metrics of this epoch into the MLflow run of the training (step = epoch)."""
    log_path = grid.run_log_path(cfg3, item["run"]["name"])
    if not log_path.exists():
        return False
    run_id = json.loads(log_path.read_text(encoding="utf-8")).get("mlflow_run_id")
    ev = summary.load_eval(item["eval_file"], eval_cfg)
    if not run_id or ev is None:
        return False
    crit = cfg3["criteria"]
    m = summary.extract_metrics(ev, crit["loop_ngram"], crit["loop_min_repeats"])
    prefix = f"{split}_on4bit_" if item.get("load_4bit") else f"{split}_"       # separate names for the 4-bit base
    return mlflow_log.log_metrics_into_run(cfg3, run_id, summary.flat_metrics(m, prefix), step=item["epoch"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate epoch adapters on val and log to MLflow.")
    ap.add_argument("--group", choices=["grid", "repeats", "q15"], required=True)
    ap.add_argument("--split", choices=list(ALLOWED_SPLITS), default="val")
    ap.add_argument("--no-mlflow", action="store_true", help="skip the MLflow logging")
    ap.add_argument("--on-4bit-base", action="store_true",
                    help="evaluate the QLoRA adapters on the 4-bit base they were trained on (names end with __on4bit)")
    a = ap.parse_args(argv)
    cfg3, eval_cfg = load_phase3_config(), load_eval_config()
    runs = runs_for_group(a.group, cfg3)
    done, missing, failed = 0, [], []
    items = plan(runs, cfg3, eval_cfg, a.split, a.on_4bit_base)
    if a.on_4bit_base and not items:
        print("No QLoRA run in this group: nothing to evaluate on a 4-bit base.")
    for it in items:
        if not (it["adapter"] / "adapter_config.json").exists():
            missing.append(it["name"])
            continue
        if it["eval_file"].exists():
            print(f"[{it['name']}] eval file exists, skipped")
        else:
            print(f"=== evaluating {it['name']} on {a.split} ===", flush=True)
            if subprocess.call(eval_command(it["adapter"], it["name"], a.split, it["base_model_id"], it["load_4bit"]),
                               cwd=str(REPO_ROOT)) != 0:
                failed.append(it["name"])
                continue
        done += 1
        if not a.no_mlflow:
            try:
                ok = log_to_mlflow(it, cfg3, eval_cfg, a.split)
                print(f"[{it['name']}] logged to MLflow" if ok else f"[{it['name']}] not logged (no run id or no eval file)")
            except Exception as err:                          # logging must never lose an evaluation
                print(f"[{it['name']}] MLflow logging failed: {err}")
    print(f"evaluated: {done}; adapters missing (training failed or not run): {missing}; evaluator failed: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
