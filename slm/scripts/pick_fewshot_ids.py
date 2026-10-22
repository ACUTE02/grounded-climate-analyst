"""Print the few-shot example ids (seeded rule in climate_slm/evaluation/prompts.py) and the pool they came from.
Paste the ids into configs/phase2.yaml `eval.fewshot.ids`. A test checks that the config equals this pick."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.evaluation.prompts import pick_fewshot_ids
from climate_slm.evaluation.scoring import load_items

if __name__ == "__main__":
    cfg = load_eval_config()
    f = cfg["eval"]["fewshot"]
    pick = pick_fewshot_ids(load_items(REPO_ROOT / cfg["paths"]["splits"][f["split"]]), cfg)
    for kind, i in pick.items():
        print(f"      {kind}: {i}")
