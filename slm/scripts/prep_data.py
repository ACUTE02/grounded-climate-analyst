"""A1: clean domain_reference chunks, split, write prep report and draft worklist."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.clean import clean_text
from climate_slm.data.split import make_splits
from climate_slm.data.textutil import read_jsonl, write_jsonl


def main():
    cfg = load_config()
    raw = read_jsonl(resolve(cfg, "raw_chunks"))
    assert len(raw) == cfg["data"]["expected_total_chunks"], len(raw)
    keep = [r for r in raw if r["source_type"] == cfg["data"]["source_type_keep"]]
    assert len(keep) == cfg["data"]["expected_kept_chunks"], len(keep)

    # how often do neighbouring chunks really overlap by overlap_chars? (raw text)
    ov = cfg["cleaning"]["overlap_chars"]
    pairs = overlapping = 0
    for a, b in zip(keep, keep[1:]):
        if a["source_id"] != b["source_id"]:
            continue
        pairs += 1
        tail = a["text"][-ov:].strip()[:60]
        if tail and tail in b["text"][: ov * 2]:
            overlapping += 1

    cleaned, total_stats, short = [], Counter(), []
    for r in keep:
        text, st = clean_text(r["text"], cfg)
        total_stats.update(st)
        if len(text) < cfg["cleaning"]["min_chunk_chars"]:
            short.append(r["id"])
        cleaned.append({
            "id": r["id"], "source_id": r["source_id"], "source": r["source"],
            "section": r.get("section"), "doc_type": r.get("doc_type"),
            "text": text, "chars_raw": len(r["text"]), "chars": len(text),
        })

    splits = make_splits(cleaned, cfg)
    for c in cleaned:
        c["split"] = splits[c["id"]]
    write_jsonl(resolve(cfg, "chunks_clean"), cleaned)
    with open(resolve(cfg, "chunk_splits"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)

    per_split = Counter(splits.values())
    per_split_source = {}
    for c in cleaned:
        per_split_source.setdefault(c["source_id"], Counter())[c["split"]] += 1

    # random 10 chunks are listed for a manual look; leftover split words are found with a
    # corpus-vocabulary rule: "a b" where "ab" is a word seen elsewhere and a or b is rare alone
    rng = random.Random(cfg["split"]["seed"])
    sample_ids = [c["id"] for c in rng.sample(cleaned, 10)]
    import re
    tok = re.compile(r"[A-Za-z]+")
    vocab = Counter(w.lower() for c in cleaned for w in tok.findall(c["text"]))
    split_hits = Counter()
    for c in cleaned:
        ws = [w.lower() for w in tok.findall(c["text"])]
        for a_, b_ in zip(ws, ws[1:]):
            if vocab[a_ + b_] >= 2 and min(vocab[a_], vocab[b_]) <= 2:
                split_hits[f"{a_} {b_}"] += 1
    chunks_with_split = sum(
        1 for c in cleaned
        if any(k in c["text"].lower() for k in split_hits))

    # draft worklist: train/val chunks only, per-source cap on questions
    g = cfg["generator"]
    qpc, cap = g["questions_per_chunk"], g["per_source_cap"]
    work = {}
    crng = random.Random(g["cap_seed"])
    for sid in sorted({c["source_id"] for c in cleaned}):
        cand = [c for c in cleaned if c["source_id"] == sid and c["split"] in ("train", "val")
                and c["id"] not in short]
        max_chunks = max(cap // qpc, 0)
        if len(cand) > max_chunks:
            cand = sorted(crng.sample(cand, max_chunks), key=lambda c: c["id"])
        work[sid] = [c["id"] for c in cand]

    # optional second draft round (config), sampled from chunks not yet in the worklist
    ext = g.get("extension_chunks_per_source", {})
    erng = random.Random(g["cap_seed"] + 1)
    for sid, n_extra in ext.items():
        rest = [c for c in cleaned if c["source_id"] == sid and c["split"] in ("train", "val")
                and c["id"] not in short and c["id"] not in work[sid]]
        pick = sorted(erng.sample(rest, min(n_extra, len(rest))), key=lambda c: c["id"])
        work[sid] = sorted(work[sid] + [c["id"] for c in pick])
    with open(resolve(cfg, "worklist"), "w", encoding="utf-8") as f:
        json.dump(work, f, indent=1)

    report = {
        "raw_chunks": len(raw), "kept_domain_reference": len(keep),
        "excluded_project_evidence": len(raw) - len(keep),
        "per_split_chunks": dict(per_split),
        "buffer_dropped": per_split.get("buffer", 0),
        "per_source_split": {k: dict(v) for k, v in per_split_source.items()},
        "cleaning_changes": dict(total_stats),
        "chunks_with_any_change": sum(1 for r, c in zip(keep, cleaned) if r["text"] != c["text"]),
        "too_short_chunks": short,
        "neighbour_pairs_checked": pairs, "neighbour_pairs_with_visible_overlap": overlapping,
        "leftover_split_word_total": sum(split_hits.values()),
        "leftover_split_word_distinct": len(split_hits),
        "leftover_split_word_chunks_affected": chunks_with_split,
        "leftover_split_word_top": split_hits.most_common(15),
        "random_10_chunk_ids_for_manual_look": sample_ids,
        "worklist_chunks_per_source": {k: len(v) for k, v in work.items()},
        "worklist_total_chunks": sum(len(v) for v in work.values()),
        "split_config": cfg["split"],
    }
    with open(resolve(cfg, "prep_report"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
