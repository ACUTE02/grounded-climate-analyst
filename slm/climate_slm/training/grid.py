"""The run list of Phase 3, built from configs/phase3.yaml (pure Python)."""
import json
from pathlib import Path

from climate_slm.config import REPO_ROOT


def lr_tag(lr: float) -> str:
    """1e-4 -> '1e-4', 2e-4 -> '2e-4'."""
    return f"{lr:.0e}".replace("e-0", "e-")


def run_name(arm: dict, method: str, lr: float | None = None, seed: int | None = None, sweep_seed: int | None = None) -> str:
    """Grid run: lora_fp16_lr1e-4. Seed repeat: lora_fp16_lr1e-4_s43. Other-size run: q15_lora_fp16."""
    base = f"{method}_lr{lr_tag(lr)}" if lr is not None and not arm["run_prefix"] else method
    name = arm["run_prefix"] + base
    if seed is not None and seed != sweep_seed:
        name += f"_s{seed}"
    return name


def make_run(cfg3: dict, arm_key: str, method: str, lr: float, seed: int) -> dict:
    arm = cfg3["arms"][arm_key]
    return {"name": run_name(arm, method, lr, seed, cfg3["sweep"]["seed"]), "arm": arm_key, "method": method, "lr": lr,
            "seed": seed, "model_id": arm["model_id"], "gradient_checkpointing": arm["gradient_checkpointing"]}


def grid_runs(cfg3: dict) -> list[dict]:
    """The 4 grid runs, seed 42, LoRA runs first (so a problem with 4-bit loading cannot block them)."""
    s = cfg3["sweep"]
    return [make_run(cfg3, "main", m, lr, s["seed"]) for m in s["methods"] for lr in s["learning_rates"]]


def repeat_runs(cfg3: dict, winner_run: dict) -> list[dict]:
    """The winning configuration again with the repeat seeds."""
    return [make_run(cfg3, "main", winner_run["method"], winner_run["lr"], sd) for sd in cfg3["sweep"]["repeat_seeds"]]


def q15_run(cfg3: dict, winner_run: dict) -> dict:
    """One run on the larger model with the method and learning rate of the 0.5B winner."""
    return make_run(cfg3, "q15", winner_run["method"], winner_run["lr"], cfg3["sweep"]["seed"])


def epochs(cfg3: dict) -> list[int]:
    return list(range(1, cfg3["train"]["epochs"] + 1))


def adapter_dir(cfg3: dict, run_name_: str, epoch: int) -> Path:
    return REPO_ROOT / cfg3["paths"]["adapters_dir"] / run_name_ / f"e{epoch}"


def run_log_path(cfg3: dict, run_name_: str) -> Path:
    return REPO_ROOT / cfg3["paths"]["runs_dir"] / run_name_ / "train_log.json"


ON_4BIT_SUFFIX = "__on4bit"


ERROR_FILE = "train_error.txt"


def error_file_path(cfg3: dict, run_name_: str) -> Path:
    return run_log_path(cfg3, run_name_).parent / ERROR_FILE


def clear_stale_error(cfg3: dict, run_name_: str) -> bool:
    """Delete the train_error.txt left by an earlier failed attempt. Called when the run has completed.
    Returns True if a file was removed."""
    p = error_file_path(cfg3, run_name_)
    if p.exists():
        p.unlink()
        return True
    return False


def candidate_name(run_name_: str, epoch: int, on_4bit: bool = False) -> str:
    """Name used for the evaluator run of one epoch adapter. A QLoRA adapter evaluated on the 4-bit base it was
    trained on gets the suffix `__on4bit`, so the results on the fp16 base stay untouched."""
    return f"{run_name_}_e{epoch}" + (ON_4BIT_SUFFIX if on_4bit else "")


def run_is_done(cfg3: dict, run: dict) -> bool:
    """The log says complete and all epoch adapters exist."""
    log = run_log_path(cfg3, run["name"])
    if not log.exists() or not json.loads(log.read_text(encoding="utf-8")).get("complete"):
        return False
    return all((adapter_dir(cfg3, run["name"], e) / "adapter_config.json").exists() for e in epochs(cfg3))


def validate(cfg3: dict) -> list[str]:
    """Problems with the config (empty list = fine)."""
    problems = []
    s, t = cfg3["sweep"], cfg3["train"]
    if len(grid_runs(cfg3)) != 4:
        problems.append("the grid must have 4 runs (2 methods x 2 learning rates)")
    if len({r["name"] for r in grid_runs(cfg3)}) != len(grid_runs(cfg3)):
        problems.append("run names are not unique")
    if t["effective_batch_size"] % t["micro_batch_size"]:
        problems.append("effective_batch_size must be a multiple of micro_batch_size")
    if len(s["repeat_seeds"]) != 2 or s["seed"] in s["repeat_seeds"]:
        problems.append("repeat_seeds must be two seeds different from the sweep seed")
    if t["epochs"] < 1 or t["refusal_repeat"] < 1:
        problems.append("epochs and refusal_repeat must be at least 1")
    unknown = [m for m in s["methods"] if m not in cfg3["methods"]]
    if unknown:
        problems.append(f"methods without a definition: {unknown}")
    order = cfg3["selection"]["order"]
    if order[:3] != ["correct_refusals_strict", "token_f1", "number_faithfulness"]:
        problems.append("selection order must start with correct_refusals_strict, token_f1, number_faithfulness")
    return problems
