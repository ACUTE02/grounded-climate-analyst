"""Write docs/REFUSAL_REVIEW.txt: the fill-in sheet for the 11 test + 11 val refusal items (context only).
Does not change any data file. Refuses to overwrite a sheet that has already been filled in."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.evaluation.review_sheet import build_sheet


def filled(path: Path) -> bool:
    """True if any CHOICE: line already has a letter after it."""
    return any(line.startswith("CHOICE:") and line[len("CHOICE:"):].strip()
               for line in path.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    cfg = load_eval_config()
    path = REPO_ROOT / cfg["paths"]["refusal_review"]
    if path.exists() and filled(path):
        raise SystemExit(f"{path} already has answers in it. Not overwritten.")
    text = build_sheet(cfg)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({text.count('ITEM ')} items, {len(text)} characters)")
