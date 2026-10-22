"""Print the fixed ids for the qualitative file (seeded). Paste them into configs/phase2.yaml
`eval.qualitative_ids`. Run once, before any model output exists. A test checks config == this pick."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.evaluation.report import pick_qualitative_ids
from climate_slm.evaluation.scoring import load_items

if __name__ == "__main__":
    cfg = load_eval_config()
    e = cfg["eval"]
    items = load_items(REPO_ROOT / cfg["paths"]["splits"][e["qualitative_split"]])
    for i in pick_qualitative_ids(items, e["qualitative_seed"], e["qualitative_n_answerable"], e["qualitative_n_refusal"]):
        print("    -", i)
