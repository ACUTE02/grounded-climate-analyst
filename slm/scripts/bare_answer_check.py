"""How much of the token-F1 gap is style? For every eval file: answerable items with a short (bare) output
against longer outputs, mean token F1 / key_recall / key_support. Reads the saved outputs only.

    python scripts/bare_answer_check.py            # all outputs/eval_*.json (smoke files left out)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.evaluation import scoring

if __name__ == "__main__":
    cfg = load_eval_config()
    limit = cfg["report"]["bare_max_chars"]
    print(f"bare = output shorter than {limit} characters (answerable items only)")
    for p in sorted((REPO_ROOT / cfg["paths"]["outputs_dir"]).glob("eval_*.json")):
        ev = json.loads(p.read_text(encoding="utf-8"))
        if ev.get("limit"):
            continue
        scoring.add_key_scores(ev, cfg)
        t = scoring.bare_answer_table(ev["items"], limit)
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        for k, v in t.items():
            print(f"{p.name:34s} {k:6s} n={v['n']:2d}  F1 {fmt(v['token_f1'])}  key_recall {fmt(v['key_recall'])}"
                  f"  key_support {fmt(v['key_support'])}")
