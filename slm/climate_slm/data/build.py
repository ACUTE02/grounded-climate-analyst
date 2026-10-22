"""Turn drafted Q&A into SFT candidates: unanswerable pairing, automatic checks, dedup."""
import json
import random
from pathlib import Path

from climate_slm.data.dedup import find_duplicates
from climate_slm.data.lexical import BM25
from climate_slm.data.sft import make_example
from climate_slm.data.validate import check_example

DRAFT_TYPES = ("definition", "numeric", "procedure", "criteria")


def resolve_chunk_id(short: str, chunk_ids: list[str]) -> str:
    hits = [c for c in chunk_ids if c.endswith("::" + short) or c == short]
    if len(hits) != 1:
        raise ValueError(f"chunk suffix {short!r} matched {len(hits)} chunks")
    return hits[0]


def read_parts(parts_dir: Path, chunk_ids: list[str]):
    """Read drafts_parts/*.jsonl (suffix ids). Returns (drafts, skips)."""
    drafts, skips = [], []
    for p in sorted(Path(parts_dir).glob("*.jsonl")):
        is_skip = ".skip." in p.name
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            cid = resolve_chunk_id(d["c"], chunk_ids)
            if is_skip:
                skips.append({"chunk_id": cid, "reason": d["reason"]})
            else:
                if d["t"] not in DRAFT_TYPES:
                    raise ValueError(f"bad type {d['t']} in {p.name}")
                drafts.append({"chunk_id": cid, "type": d["t"], "question": d["q"], "answer": d["a"]})
    return drafts, skips


def pick_context_chunk(question: str, home: dict, pool: list[dict], strategy: str, rng: random.Random) -> dict:
    """Choose a chunk from another source that should not answer the question."""
    others = [c for c in pool if c["source_id"] != home["source_id"] and c["split"] == home["split"]]
    if not others:
        others = [c for c in pool if c["source_id"] != home["source_id"]]
    if strategy == "random_other_source":
        return rng.choice(others)
    if strategy != "lexical_hard":
        raise ValueError(f"unknown unanswerable strategy {strategy}")
    bm = BM25([c["text"] for c in others])
    scores = [bm.score(question, i) for i in range(len(others))]
    best = max(range(len(others)), key=lambda i: (scores[i], -i))
    return others[best]


def build_candidates(drafts: list[dict], chunks: list[dict], cfg: dict, generator: str, id_prefix: str,
                     n_unanswerable: int | None = None) -> list[dict]:
    by_id = {c["id"]: c for c in chunks}
    u = cfg["unanswerable"]
    rng = random.Random(u["seed"])
    n_un = n_unanswerable if n_unanswerable is not None else round(u["ratio"] * len(drafts))
    un_idx = set(rng.sample(range(len(drafts)), n_un))

    examples = []
    for i, d in enumerate(drafts):
        home = by_id[d["chunk_id"]]
        if i in un_idx:
            ctx = pick_context_chunk(d["question"], home, chunks, u["strategy"], rng)
            meta = {"chunk_id": ctx["id"], "source_id": ctx["source_id"], "split": home["split"],
                    "type": "unanswerable", "generator": generator, "verification": "unverified",
                    "question_chunk_id": home["id"], "unanswerable_strategy": u["strategy"],
                    "needs_check": True}
            ex = make_example(cfg, ctx["text"], d["question"], cfg["refusal_text"], meta)
        else:
            meta = {"chunk_id": home["id"], "source_id": home["source_id"], "split": home["split"],
                    "type": d["type"], "generator": generator, "verification": "unverified"}
            ex = make_example(cfg, home["text"], d["question"], d["answer"], meta)
        ex["id"] = f"{id_prefix}{i + 1:04d}"
        examples.append(ex)
    run_checks(examples, cfg)
    return examples


def run_checks(examples: list[dict], cfg: dict) -> None:
    """Fill ex['auto_flags'] in place (checks, then dedup). Never deletes anything."""
    dups = find_duplicates(examples, cfg["checks"]["near_dup_jaccard"])
    for i, ex in enumerate(examples):
        flags = check_example(ex, cfg)
        if i in dups:
            flags.append(dups[i])
        ex["auto_flags"] = flags
