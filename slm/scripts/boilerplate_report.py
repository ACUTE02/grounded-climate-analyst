"""A2 F1: show the boilerplate heuristic at work: counts, 10 flagged and 10 not flagged chunks with reasons."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.a2 import boiler_for, load_chunks


def main():
    cfg = load_config()
    chunks = load_chunks(cfg)
    bm = boiler_for(chunks, cfg)
    rng = random.Random(cfg["split"]["seed"])
    flagged = sorted(c for c in bm if bm[c])
    clean = sorted(c for c in bm if not bm[c])
    by = {c["id"]: c for c in chunks}

    def show(ids):
        return [{"chunk_id": i, "reasons": bm[i], "chars": by[i]["chars"],
                 "start": " ".join(by[i]["text"].split())[:150]} for i in ids]

    out = {"chunks": len(chunks), "flagged": len(flagged), "not_flagged": len(clean),
           "flagged_by_reason": dict(Counter(r.split(":")[0] for c in flagged for r in bm[c])),
           "flagged_in_skip_log_only": sum(1 for c in flagged if bm[c] == ["skip_log"]),
           "flagged_by_heuristic_only": sum(1 for c in flagged if "skip_log" not in bm[c]),
           "sample_flagged": show(rng.sample(flagged, min(10, len(flagged)))),
           "sample_not_flagged": show(rng.sample(clean, min(10, len(clean))))}
    p = resolve(cfg, "processed_dir") / "boilerplate_report.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("sample")}, indent=1))
    for s in ("sample_flagged", "sample_not_flagged"):
        print(s)
        for r in out[s]:
            print(" ", r["chunk_id"][-28:], r["reasons"], "|", r["start"][:90])


if __name__ == "__main__":
    main()
