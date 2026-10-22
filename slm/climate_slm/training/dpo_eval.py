"""Evaluate every epoch adapter of the DPO runs with the existing evaluator, on VAL only, and log the numbers into the
MLflow run of that training.

    python -m climate_slm.training.dpo_eval --group grid --split val
    python -m climate_slm.training.dpo_eval --group repeats --split val

Same evaluator as every other model (separate process: python -m climate_slm.evaluation.evaluate; greedy, float16, same
chat template, max_new_tokens 256). `--split` accepts val only, so the test split cannot be evaluated by accident from
this loop. The one test run is in climate_slm.training.dpo_final_test (frozen selection only)."""
import argparse
import json
import subprocess
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config, load_phase4_config
from climate_slm.evaluation.evaluate import default_out
from climate_slm.training import grid, mlflow_log, summary
from climate_slm.training.dpo_train import resolve_runs
from climate_slm.training.eval_adapters import eval_command

ALLOWED_SPLITS = ("val",)


def plan(runs: list[dict], cfg4: dict, eval_cfg: dict, split: str) -> list[dict]:
    """One entry per epoch adapter: run, epoch, adapter dir, evaluator name, expected eval file, base model."""
    items = []
    for r in runs:
        for e in grid.epochs(cfg4):
            name = grid.candidate_name(r["name"], e)
            items.append({"run": r, "epoch": e, "adapter": grid.adapter_dir(cfg4, r["name"], e), "name": name,
                          "eval_file": default_out(eval_cfg, name, split), "base_model_id": r["model_id"]})
    return items


def log_to_mlflow(item: dict, cfg4: dict, cfg3: dict, eval_cfg: dict, split: str) -> bool:
    """Put the val metrics of this epoch into the MLflow run of the training (step = epoch)."""
    log_path = grid.run_log_path(cfg4, item["run"]["name"])
    if not log_path.exists():
        return False
    run_id = json.loads(log_path.read_text(encoding="utf-8")).get("mlflow_run_id")
    ev = summary.load_eval(item["eval_file"], eval_cfg)
    if not run_id or ev is None:
        return False
    crit = cfg3["criteria"]
    m = summary.extract_metrics(ev, crit["loop_ngram"], crit["loop_min_repeats"])
    return mlflow_log.log_metrics_into_run(cfg4, run_id, summary.flat_metrics(m, f"{split}_"), step=item["epoch"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate DPO epoch adapters on val and log to MLflow.")
    ap.add_argument("--group", choices=["grid", "repeats"], required=True)
    ap.add_argument("--split", choices=list(ALLOWED_SPLITS), default="val")
    ap.add_argument("--no-mlflow", action="store_true", help="skip the MLflow logging")
    a = ap.parse_args(argv)
    cfg4, cfg3, eval_cfg = load_phase4_config(), load_phase3_config(), load_eval_config()
    done, missing, failed = 0, [], []
    for it in plan(resolve_runs(a.group, cfg4), cfg4, eval_cfg, a.split):
        if not (it["adapter"] / "adapter_config.json").exists():
            missing.append(it["name"])
            continue
        if it["eval_file"].exists():
            print(f"[{it['name']}] eval file exists, skipped")
        else:
            print(f"=== evaluating {it['name']} on {a.split} ===", flush=True)
            if subprocess.call(eval_command(it["adapter"], it["name"], a.split, it["base_model_id"]), cwd=str(REPO_ROOT)) != 0:
                failed.append(it["name"])
                continue
        done += 1
        if not a.no_mlflow:
            try:
                ok = log_to_mlflow(it, cfg4, cfg3, eval_cfg, a.split)
                print(f"[{it['name']}] logged to MLflow" if ok else f"[{it['name']}] not logged (no run id or no eval file)")
            except Exception as err:                          # logging must never lose an evaluation
                print(f"[{it['name']}] MLflow logging failed: {err}")
    print(f"evaluated: {done}; adapters missing (training failed or not run): {missing}; evaluator failed: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
