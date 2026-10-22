"""Evaluate every epoch adapter of the DPO v2 runs with the existing evaluator, on VAL only, and log the numbers into the
MLflow run of that training.

    python -m climate_slm.training.dpo2_eval --group grid --split val       # 8 runs x 3 epochs = 24 adapters
    python -m climate_slm.training.dpo2_eval --group repeats --split val

Same evaluator as every other model (separate process: python -m climate_slm.evaluation.evaluate; greedy, float16, same chat
template, max_new_tokens 256). `--split` accepts val only, so the test split cannot be evaluated by accident from this loop.
The one test run is in climate_slm.training.dpo2_final_test (frozen selection only). Plan and MLflow logging are the v1 ones
(dpo_eval), which only read generic keys of the config."""
import argparse
import subprocess
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config
from climate_slm.training import dpo2_grid
from climate_slm.training.dpo2_train import resolve_runs
from climate_slm.training.dpo_eval import log_to_mlflow, plan
from climate_slm.training.eval_adapters import eval_command

ALLOWED_SPLITS = ("val",)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate DPO v2 epoch adapters on val and log to MLflow.")
    ap.add_argument("--group", choices=["grid", "repeats"], required=True)
    ap.add_argument("--split", choices=list(ALLOWED_SPLITS), default="val")
    ap.add_argument("--no-mlflow", action="store_true", help="skip the MLflow logging")
    a = ap.parse_args(argv)
    cfg, cfg3, eval_cfg = dpo2_grid.runtime_config(), load_phase3_config(), load_eval_config()
    done, missing, failed = 0, [], []
    for it in plan(resolve_runs(a.group, cfg), cfg, eval_cfg, a.split):
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
                ok = log_to_mlflow(it, cfg, cfg3, eval_cfg, a.split)
                print(f"[{it['name']}] logged to MLflow" if ok else f"[{it['name']}] not logged (no run id or no eval file)")
            except Exception as err:                          # logging must never lose an evaluation
                print(f"[{it['name']}] MLflow logging failed: {err}")
    print(f"evaluated: {done}; adapters missing (training failed or not run): {missing}; evaluator failed: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
