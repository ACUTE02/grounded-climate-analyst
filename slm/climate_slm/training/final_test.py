"""The ONE place where the test split is evaluated for a tuned model. Clearly separate from the sweep loop.

    python -m climate_slm.training.final_test --size 0.5b
    python -m climate_slm.training.final_test --size 1.5b

Refuses to run unless (1) the selection file is frozen (select_best --freeze wrote its sha256 and the file still
matches), (2) the selection names a winner for that size, and (3) this candidate has not been evaluated on test before.
Every test run is recorded (candidate, selection sha256, time) in outputs/phase3_test_runs.json."""
import argparse
import datetime
import json
import subprocess
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config
from climate_slm.evaluation.evaluate import default_out
from climate_slm.training import grid, selection_io

SIZES = {"0.5b": ("grid", "main"), "1.5b": ("q15", "q15")}


def winner_for(sel: dict, size: str) -> dict | None:
    block = sel.get(SIZES[size][0]) or {}
    return block.get("winner")


def check_can_run(cfg3: dict, eval_cfg: dict, size: str) -> dict:
    """The winner dict if a test run is allowed; otherwise raises SystemExit with the reason."""
    if not selection_io.is_frozen(cfg3):
        raise SystemExit("The selection is not frozen. Run `python -m climate_slm.training.select_best --freeze` first.")
    w = winner_for(selection_io.read_selection(cfg3), size)
    if not w:
        raise SystemExit(f"The frozen selection has no winner for {size}.")
    if default_out(eval_cfg, w["candidate"], "test").exists():
        raise SystemExit(f"{w['candidate']} was already evaluated on test. The test is run once per final candidate.")
    return w


def test_command(cfg3: dict, w: dict, size: str) -> list[str]:
    arm = cfg3["arms"][SIZES[size][1]]
    cmd = [sys.executable, "-m", "climate_slm.evaluation.evaluate", "--model",
           str(grid.adapter_dir(cfg3, w["run"], w["epoch"])), "--split", "test", "--name", w["candidate"],
           "--base-model-id", arm["model_id"]]
    if w.get("eval_base") == "4bit":                      # the winner was chosen on the 4-bit base: test it there too
        cmd.append("--load-4bit")
    return cmd


def record(cfg3: dict, w: dict) -> None:
    p = REPO_ROOT / cfg3["paths"]["test_log"]
    runs = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    runs.append({"candidate": w["candidate"], "selection_sha256": selection_io.file_sha256(selection_io.selection_path(cfg3)),
                 "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    p.write_text(json.dumps(runs, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a frozen winner on the test split, once.")
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    a = ap.parse_args(argv)
    cfg3, eval_cfg = load_phase3_config(), load_eval_config()
    w = check_can_run(cfg3, eval_cfg, a.size)
    print(f"FINAL TEST of {w['candidate']} (selection sha256 {selection_io.file_sha256(selection_io.selection_path(cfg3))})")
    code = subprocess.call(test_command(cfg3, w, a.size), cwd=str(REPO_ROOT))
    if code == 0:
        record(cfg3, w)
    return code


if __name__ == "__main__":
    sys.exit(main())
