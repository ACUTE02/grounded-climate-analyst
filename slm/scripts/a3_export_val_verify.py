"""A3 G3: verifier input for the val items that were never verified (re-admitted + new)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from climate_slm.config import load_config, resolve
from climate_slm.data.textutil import read_jsonl
from export_verify_inputs import export


def main():
    cfg = load_config()
    rows = [r for r in read_jsonl(resolve(cfg, "candidates"))
            if r["dropped_by"] is None and "ai_verdict" not in r]
    assert all(r["meta"]["split"] == "val" for r in rows)
    for p in export(rows, resolve(cfg, "verify_dir"), "train_a3", 2):
        print(p.name)
    print(len(rows), "items")


if __name__ == "__main__":
    main()
