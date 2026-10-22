"""The run list of Phase 4b (DPO v2), built from configs/phase4b.yaml (pure Python, no torch).

Grid: pair set {v2-main, v2-mix} x alpha {0.5, 1.0} x lr {1e-5, 2e-5} = 8 runs x 3 epochs = 24 candidates (beta 0.1 fixed).
Run name: dpo2_main_a0.5_lr1e-5 (seed 42), dpo2_main_a0.5_lr1e-5_s43 (seed repeat). Candidate = run + epoch: ..._e2.

`runtime_config` turns the fixed design (`design` block) into the shape the shared Phase 3 / v1 helpers read
(paths.adapters_dir, paths.runs_dir, train.epochs, paths.selection_json, ...), so grid.adapter_dir / run_log_path /
run_is_done / clear_stale_error, selection_io and the v1 start-adapter checks are reused as they are. The SFT start
adapters (hash, control names) and the model id are read from configs/phase4.yaml: one copy only."""
import copy

from climate_slm.config import load_phase4_config, load_phase4b_config
from climate_slm.training import dpo_grid
from climate_slm.training.grid import lr_tag


def set_tag(cfg: dict, pair_set: str) -> str:
    return cfg["pair_files"][pair_set]["tag"]


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}"


def runtime_config(cfg4b: dict | None = None, cfg4: dict | None = None) -> dict:
    """cfg4b merged with the v1 `start` and `model` blocks, plus `train` and `sweep` views of the design."""
    cfg4b, cfg4 = cfg4b or load_phase4b_config(), cfg4 or load_phase4_config()
    d = cfg4b["design"]
    cfg = copy.deepcopy(cfg4b)
    cfg["start"], cfg["model"] = copy.deepcopy(cfg4["start"]), copy.deepcopy(cfg4["model"])
    cfg["train"] = {k: d[k] for k in ("epochs", "effective_batch_size", "micro_batch_size", "warmup_ratio", "lr_schedule",
                                      "weight_decay", "max_grad_norm", "max_seq_len", "dtype", "grad_scaler_init_scale",
                                      "skip_if_done", "reference_cache")}
    cfg["sweep"] = {"seed": d["seed"], "repeat_seeds": list(d["repeat_seeds"])}
    cfg["selection"] = d["selection"]
    return cfg


def run_name(cfg: dict, pair_set: str, alpha: float, lr: float, seed: int) -> str:
    name = f"dpo2_{set_tag(cfg, pair_set)}_a{alpha_tag(alpha)}_lr{lr_tag(lr)}"
    return name if seed == cfg["sweep"]["seed"] else f"{name}_s{seed}"


def make_run(cfg: dict, pair_set: str, alpha: float, lr: float, seed: int) -> dict:
    st = dpo_grid.start_info(cfg, seed)
    return {"name": run_name(cfg, pair_set, alpha, lr, seed), "pair_set": pair_set, "alpha": alpha, "beta": cfg["design"]["beta"],
            "lr": lr, "seed": seed, "model_id": cfg["model"]["model_id"], "start_adapter": st["adapter"],
            "start_sha256": st["sha256"], "control": st["control"]}


def grid_runs(cfg: dict) -> list[dict]:
    """The 8 grid runs, seed 42: v2-main first (both alphas, both learning rates), then v2-mix."""
    g, seed = cfg["design"]["grid"], cfg["design"]["seed"]
    return [make_run(cfg, ps, a, lr, seed) for ps in g["pair_sets"] for a in g["alphas"] for lr in g["learning_rates"]]


def repeat_runs(cfg: dict, winner_run_: dict) -> list[dict]:
    """The winning configuration again with the repeat seeds (each starts from the SFT adapter of its own seed)."""
    return [make_run(cfg, winner_run_["pair_set"], winner_run_["alpha"], winner_run_["lr"], sd) for sd in cfg["sweep"]["repeat_seeds"]]


def winner_run(cfg: dict, sel: dict) -> dict:
    """The run of the grid winner in a saved selection dict (outputs/phase4b_selection.json)."""
    w = sel.get("winner")
    if not w:
        raise SystemExit("The saved DPO v2 selection has no winner (no DPO v2 candidate improves on the SFT model on val). "
                         "Nothing to repeat and nothing to test. See outputs/phase4b_selection.md.")
    return make_run(cfg, w["pair_set"], w["alpha"], w["lr"], cfg["sweep"]["seed"])


def find_run(name: str, cfg: dict, sel: dict | None = None) -> dict:
    known = list(grid_runs(cfg))
    if sel and sel.get("winner"):
        known += repeat_runs(cfg, winner_run(cfg, sel))
    for r in known:
        if r["name"] == name:
            return r
    raise SystemExit(f"unknown run {name!r}. Known: {[r['name'] for r in known]}")


def reference_cache_path(cfg: dict, run: dict):
    """Reference log-probs are cached per pair file (set) and seed: the runs of one set share them."""
    from climate_slm.config import REPO_ROOT
    return REPO_ROOT / cfg["paths"]["runs_dir"] / f"reference_logps_{set_tag(cfg, run['pair_set'])}_seed{run['seed']}.json"


def validate(cfg: dict) -> list[str]:
    """Problems with the (runtime) config; empty list = fine."""
    problems, d, t = [], cfg["design"], cfg["train"]
    runs, g = grid_runs(cfg), d["grid"]
    if len(runs) != 8 or len(runs) * t["epochs"] != 24:
        problems.append("the grid must have 8 runs and 24 candidates (2 pair sets x 2 alphas x 2 learning rates x 3 epochs)")
    if len({r["name"] for r in runs}) != len(runs):
        problems.append("run names are not unique")
    if sorted(g["pair_sets"]) != sorted(cfg["pair_files"]):
        problems.append("grid pair sets differ from pair_files")
    if t["effective_batch_size"] != 8 or t["micro_batch_size"] != 1 or t["effective_batch_size"] % t["micro_batch_size"]:
        problems.append("effective batch size must be 8 (amendment 2026-10-20) with micro batch 1")
    if d["beta"] != 0.1 or g["alphas"] != [0.5, 1.0] or g["learning_rates"] != [1e-5, 2e-5]:
        problems.append("beta, alphas and learning rates must be the fixed v2 values")
    if len(d["repeat_seeds"]) != 2 or d["seed"] in d["repeat_seeds"]:
        problems.append("repeat_seeds must be two seeds different from the sweep seed")
    for sd in [d["seed"]] + list(d["repeat_seeds"]):
        if sd not in cfg["start"]["seeds"]:
            problems.append(f"no SFT start adapter for seed {sd}")
    if d["grid_reduction"] != "none":
        problems.append("the grid is reduced: it must be written down before any run (grid_reduction)")
    if d["selection"]["order"][:2] != ["false_refusals", "correct_refusals_strict"]:
        problems.append("selection order must start with false_refusals, correct_refusals_strict")
    return problems
