"""The ONE place where the test split is evaluated for a DPO v2 model. Clearly separate from the val loop (dpo2_eval).

    python -m climate_slm.training.dpo2_final_test

Refuses to run unless (1) the v2 selection file is frozen (select_dpo2 --freeze wrote its sha256 and the file still matches),
(2) the frozen selection names a winner (it does not if "no DPO v2 candidate improves on the SFT model on val"), (3) the winner
adapter exists, and (4) this candidate has not been evaluated on test before (no eval file, no entry in the test log).
The SFT control is NOT run again (its test result exists: eval_lora_fp16_lr2e-4_e2_test.json). The checks and the command are
the v1 ones (dpo_final_test), which read only generic keys of the config. Every test run is recorded in
outputs/phase4b_test_runs.json."""
import argparse
import subprocess
import sys

from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.training import dpo2_grid, dpo_final_test as v1, selection_io


def main(argv=None):
    argparse.ArgumentParser(description="Evaluate the frozen DPO v2 winner on the test split, once.").parse_args(argv)
    cfg, eval_cfg = dpo2_grid.runtime_config(), load_eval_config()
    w = v1.check_can_run(cfg, eval_cfg, select_module="select_dpo2")
    print(f"FINAL TEST of {w['candidate']} (selection sha256 {selection_io.file_sha256(selection_io.selection_path(cfg))})")
    code = subprocess.call(v1.test_command(cfg, w), cwd=str(REPO_ROOT))
    if code == 0:
        v1.record(cfg, w)
    return code


if __name__ == "__main__":
    sys.exit(main())
