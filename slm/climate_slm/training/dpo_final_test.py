"""The ONE place where the test split is evaluated for a DPO model. Clearly separate from the val loop (dpo_eval).

    python -m climate_slm.training.dpo_final_test

Refuses to run unless (1) the DPO selection file is frozen (select_dpo --freeze wrote its sha256 and the file still
matches), (2) the frozen selection names a winner (it does not if "no DPO candidate improves on the SFT model on val"),
(3) the winner adapter exists, and (4) this candidate has not been evaluated on test before (no eval file, no entry in
the test log). The SFT control is NOT run again: its test result already exists (eval_lora_fp16_lr2e-4_e2_test.json).
Every test run is recorded (candidate, selection sha256, time) in outputs/phase4_test_runs.json."""
import argparse
import datetime
import json
import subprocess
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase4_config
from climate_slm.evaluation.evaluate import default_out
from climate_slm.training import grid, selection_io


def check_can_run(cfg4: dict, eval_cfg: dict, select_module: str = "select_dpo") -> dict:
    """The winner dict if a test run is allowed; otherwise raises SystemExit with the reason."""
    if not selection_io.is_frozen(cfg4):
        raise SystemExit(f"The DPO selection is not frozen. Run `python -m climate_slm.training.{select_module} --freeze` first.")
    w = selection_io.read_selection(cfg4).get("winner")
    if not w:
        raise SystemExit("The frozen DPO selection has no winner (no DPO candidate improves on the SFT model on val). "
                         "The test is not run: SFT stays.")
    if not (grid.adapter_dir(cfg4, w["run"], w["epoch"]) / "adapter_config.json").exists():
        raise SystemExit(f"The adapter of {w['candidate']} is missing in {cfg4['paths']['adapters_dir']}.")
    tested = default_out(eval_cfg, w["candidate"], "test").exists() or w["candidate"] in test_log_candidates(cfg4)
    if tested:
        raise SystemExit(f"{w['candidate']} was already evaluated on test. The test is run once per final candidate.")
    return w


def test_command(cfg4: dict, w: dict) -> list[str]:
    return [sys.executable, "-m", "climate_slm.evaluation.evaluate", "--model",
            str(grid.adapter_dir(cfg4, w["run"], w["epoch"])), "--split", "test", "--name", w["candidate"],
            "--base-model-id", cfg4["model"]["model_id"]]


def test_log_candidates(cfg4: dict) -> list[str]:
    p = REPO_ROOT / cfg4["paths"]["test_log"]
    return [e["candidate"] for e in json.loads(p.read_text(encoding="utf-8"))] if p.exists() else []


def record(cfg4: dict, w: dict) -> None:
    p = REPO_ROOT / cfg4["paths"]["test_log"]
    runs = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    runs.append({"candidate": w["candidate"], "selection_sha256": selection_io.file_sha256(selection_io.selection_path(cfg4)),
                 "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    p.write_text(json.dumps(runs, indent=1), encoding="utf-8")


def main(argv=None):
    argparse.ArgumentParser(description="Evaluate the frozen DPO winner on the test split, once.").parse_args(argv)
    cfg4, eval_cfg = load_phase4_config(), load_eval_config()
    w = check_can_run(cfg4, eval_cfg)
    print(f"FINAL TEST of {w['candidate']} (selection sha256 {selection_io.file_sha256(selection_io.selection_path(cfg4))})")
    code = subprocess.call(test_command(cfg4, w), cwd=str(REPO_ROOT))
    if code == 0:
        record(cfg4, w)
    return code


if __name__ == "__main__":
    sys.exit(main())
