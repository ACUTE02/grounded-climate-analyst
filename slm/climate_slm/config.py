"""Load the Phase 1 config and resolve paths against the repo root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase1.yaml"


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


PHASE2_CONFIG = REPO_ROOT / "configs" / "phase2.yaml"


def load_eval_config(path=None) -> dict:
    """Phase 2 config, with the refusal text taken from the Phase 1 config (one copy only)."""
    cfg = load_config(path or PHASE2_CONFIG)
    cfg["refusal_text"] = load_config()["refusal_text"]
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    return REPO_ROOT / cfg["paths"][key]


def system_prompt(cfg: dict) -> str:
    return cfg["system_prompt"].format(refusal_text=cfg["refusal_text"]).strip()


PHASE3_CONFIG = REPO_ROOT / "configs" / "phase3.yaml"


def load_phase3_config(path=None) -> dict:
    """Phase 3 (training) config. Eval settings and the refusal text are read with load_eval_config()."""
    return load_config(path or PHASE3_CONFIG)


PHASE4_CONFIG = REPO_ROOT / "configs" / "phase4.yaml"


def load_phase4_config(path=None) -> dict:
    """Phase 4 (DPO) config. Eval settings and the refusal text are read with load_eval_config()."""
    return load_config(path or PHASE4_CONFIG)


PHASE4B_CONFIG = REPO_ROOT / "configs" / "phase4b.yaml"


def load_phase4b_config(path=None) -> dict:
    """Phase 4b (DPO v2) config. The v1 settings (start adapters, model, paths of v1) stay in phase4.yaml."""
    return load_config(path or PHASE4B_CONFIG)


PHASE5_CONFIG = REPO_ROOT / "configs" / "phase5.yaml"


def load_phase5_config(path=None) -> dict:
    """Phase 5 (serving) config. The prompt, refusal text and decoding come from load_config() / load_eval_config()."""
    return load_config(path or PHASE5_CONFIG)
