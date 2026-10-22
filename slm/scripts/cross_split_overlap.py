"""A2 F2: information for the reviewer. Chunk pairs in different splits or different sources with the most
shared 8-word sequences (buffer chunks excluded). The split itself is NOT changed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.a2 import load_chunks
from climate_slm.data.leakage import cross_split_overlap


def main():
    cfg = load_config()
    lk = cfg["leakage"]
    pairs = cross_split_overlap(load_chunks(cfg), lk["answer_ngram_n"], lk["overlap_report_top"])
    out = {"ngram_n": lk["answer_ngram_n"], "note": "pairs in different splits or different sources, buffer excluded",
           "pairs": pairs}
    resolve(cfg, "cross_split_overlap").write_text(json.dumps(out, indent=1), encoding="utf-8")
    cross_split = sum(1 for p in pairs if p["split_a"] != p["split_b"])
    print(len(pairs), "pairs written;", cross_split, "of them are between different splits")
    for p in pairs[:5]:
        print(p["shared_ngrams"], p["source_a"], p["split_a"], "<->", p["source_b"], p["split_b"])


if __name__ == "__main__":
    main()
