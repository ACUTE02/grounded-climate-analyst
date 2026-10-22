"""Read the saved selection file (outputs/phase3_selection.json) to find the 0.5B winner. Pure Python."""
import hashlib
import json
from pathlib import Path

from climate_slm.config import REPO_ROOT
from climate_slm.training import grid


def selection_path(cfg3: dict) -> Path:
    return REPO_ROOT / cfg3["paths"]["selection_json"]


def sha_path(cfg3: dict) -> Path:
    return REPO_ROOT / cfg3["paths"]["selection_sha"]


def read_selection(cfg3: dict) -> dict:
    p = selection_path(cfg3)
    if not p.exists():
        raise SystemExit(f"{p} not found. Run the grid, evaluate its adapters on val, then run "
                         "python -m climate_slm.training.select_best first.")
    return json.loads(p.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_frozen(cfg3: dict) -> bool:
    """True if the sha file exists and still matches the selection file."""
    sp, hp = selection_path(cfg3), sha_path(cfg3)
    return sp.exists() and hp.exists() and hp.read_text(encoding="utf-8").split()[0] == file_sha256(sp)


def winner_run(cfg3: dict, sel: dict | None = None) -> dict:
    """The run (method, lr, ...) of the 0.5B grid winner, as a run dict."""
    sel = sel or read_selection(cfg3)
    w = (sel.get("grid") or {}).get("winner")
    if not w:
        raise SystemExit("The saved selection has no winner (no candidate passed the false-refusal filter). "
                         "Nothing to repeat and no 1.5B run. See outputs/phase3_selection.md.")
    return grid.make_run(cfg3, "main", w["method"], w["lr"], cfg3["sweep"]["seed"])
