"""The run list of Phase 4 (DPO), built from configs/phase4.yaml (pure Python, no torch).

Run name: dpo_b0.1_lr5e-5 (seed 42), dpo_b0.1_lr5e-5_s43 (seed repeat). Candidate = run + epoch: dpo_b0.1_lr5e-5_e2.
The config of Phase 4 has the same keys as the Phase 3 one where the shared helpers of `grid` read them
(paths.adapters_dir, paths.runs_dir, train.epochs), so grid.adapter_dir / run_log_path / run_is_done / clear_stale_error
are reused as they are."""
import hashlib
from pathlib import Path

from climate_slm.config import REPO_ROOT
from climate_slm.training.grid import lr_tag

ADAPTER_FILE = "adapter_model.safetensors"


def beta_tag(beta: float) -> str:
    return f"{beta:g}"


def run_name(beta: float, lr: float, seed: int, sweep_seed: int) -> str:
    name = f"dpo_b{beta_tag(beta)}_lr{lr_tag(lr)}"
    return name if seed == sweep_seed else f"{name}_s{seed}"


def start_info(cfg4: dict, seed: int) -> dict:
    s = cfg4["start"]["seeds"].get(seed)
    if s is None:
        raise KeyError(f"no SFT start adapter for seed {seed} in configs/phase4.yaml")
    return s


def make_run(cfg4: dict, beta: float, lr: float, seed: int) -> dict:
    st = start_info(cfg4, seed)
    return {"name": run_name(beta, lr, seed, cfg4["sweep"]["seed"]), "beta": beta, "lr": lr, "seed": seed,
            "model_id": cfg4["model"]["model_id"], "start_adapter": st["adapter"], "start_sha256": st["sha256"],
            "control": st["control"]}


def grid_runs(cfg4: dict) -> list[dict]:
    """The 4 grid runs, seed 42: beta 0.1 first (both learning rates), then beta 0.3."""
    s = cfg4["sweep"]
    return [make_run(cfg4, b, lr, s["seed"]) for b in s["betas"] for lr in s["learning_rates"]]


def repeat_runs(cfg4: dict, winner_run_: dict) -> list[dict]:
    """The winning configuration again with the repeat seeds (each starts from the SFT adapter of its own seed)."""
    return [make_run(cfg4, winner_run_["beta"], winner_run_["lr"], sd) for sd in cfg4["sweep"]["repeat_seeds"]]


def winner_run(cfg4: dict, sel: dict) -> dict:
    """The run of the grid winner in a saved selection dict (outputs/phase4_selection.json)."""
    w = sel.get("winner")
    if not w:
        raise SystemExit("The saved DPO selection has no winner (no candidate improves on the SFT model on val). "
                         "Nothing to repeat and nothing to test. See outputs/phase4_selection.md.")
    return make_run(cfg4, w["beta"], w["lr"], cfg4["sweep"]["seed"])


def find_run(name: str, cfg4: dict, sel: dict | None = None) -> dict:
    known = list(grid_runs(cfg4))
    if sel and sel.get("winner"):
        known += repeat_runs(cfg4, winner_run(cfg4, sel))
    for r in known:
        if r["name"] == name:
            return r
    raise SystemExit(f"unknown run {name!r}. Known: {[r['name'] for r in known]}")


def control_name(cfg4: dict, seed: int) -> str:
    """Name of the saved eval files of the SFT control of this seed (eval_<name>_val.json)."""
    return start_info(cfg4, seed)["control"]


# ---- start adapter ----------------------------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def start_adapter_dir(run: dict) -> Path:
    return REPO_ROOT / run["start_adapter"]


def check_start_adapter(run: dict) -> str:
    """The sha256 of the SFT start adapter if the folder exists and matches the hash fixed in the config; raises
    SystemExit with the reason otherwise (a wrong or missing start adapter must stop the run, not change the result)."""
    d = start_adapter_dir(run)
    f = d / ADAPTER_FILE
    if not (d / "adapter_config.json").exists() or not f.exists():
        raise SystemExit(f"SFT start adapter not found: {d}. Upload phase4_start.zip (cell P4-2) and unzip it into outputs/.")
    sha = file_sha256(f)
    if sha != run["start_sha256"]:
        raise SystemExit(f"SFT start adapter {d} has sha256 {sha}, the config expects {run['start_sha256']}. "
                         "Not starting: this is not the frozen SFT adapter.")
    return sha


def adapter_config_problems(adapter_cfg: dict, cfg3: dict, model_id: str) -> list[str]:
    """The start adapter must be the Phase 3 LoRA (same r, alpha, dropout, target modules) on the same base model."""
    lo, problems = cfg3["lora"], []
    for key, want in (("r", lo["r"]), ("lora_alpha", lo["alpha"]), ("lora_dropout", lo["dropout"])):
        if adapter_cfg.get(key) != want:
            problems.append(f"{key} is {adapter_cfg.get(key)!r}, Phase 3 config has {want!r}")
    if sorted(adapter_cfg.get("target_modules") or []) != sorted(lo["target_modules"]):
        problems.append("target_modules differ from the Phase 3 config")
    if adapter_cfg.get("base_model_name_or_path") != model_id:
        problems.append(f"base model is {adapter_cfg.get('base_model_name_or_path')!r}, expected {model_id!r}")
    return problems


# ---- paths --------------------------------------------------------------------------------------------------------
def reference_cache_path(cfg4: dict, seed: int) -> Path:
    return REPO_ROOT / cfg4["paths"]["runs_dir"] / f"reference_logps_seed{seed}.json"


def validate(cfg4: dict) -> list[str]:
    """Problems with the config (empty list = fine)."""
    problems, s, t, sel = [], cfg4["sweep"], cfg4["train"], cfg4["selection"]
    runs = grid_runs(cfg4)
    if len(runs) != 4 or len(s["betas"]) != 2 or len(s["learning_rates"]) != 2:
        problems.append("the grid must have 4 runs (2 betas x 2 learning rates)")
    if len({r["name"] for r in runs}) != len(runs):
        problems.append("run names are not unique")
    if t["effective_batch_size"] % t["micro_batch_size"]:
        problems.append("effective_batch_size must be a multiple of micro_batch_size")
    if len(s["repeat_seeds"]) != 2 or s["seed"] in s["repeat_seeds"]:
        problems.append("repeat_seeds must be two seeds different from the sweep seed")
    for sd in [s["seed"]] + list(s["repeat_seeds"]):
        if sd not in cfg4["start"]["seeds"]:
            problems.append(f"no SFT start adapter for seed {sd}")
    if t["epochs"] < 1:
        problems.append("epochs must be at least 1")
    if sel["order"][:2] != ["false_refusals", "correct_refusals_strict"]:
        problems.append("selection order must start with false_refusals, correct_refusals_strict")
    if cfg4["pairs"]["include_type2"]:
        problems.append("pairs.include_type2 must stay false (reviewer amendment 2026-10-20)")
    return problems
