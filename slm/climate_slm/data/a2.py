"""Shared loading helpers for the A2 build (train/val and test)."""
import json
import re
from pathlib import Path

from climate_slm.config import REPO_ROOT, resolve
from climate_slm.data.boilerplate import boilerplate_map, skip_log_ids
from climate_slm.data.build import DRAFT_TYPES, read_parts, resolve_chunk_id
from climate_slm.data.textutil import norm_ws, read_jsonl

INSTR = REPO_ROOT / "data" / "instruction"
TEST_DIR = INSTR / "test_drafting"
NEARMISS_TRAIN = INSTR / "nearmiss_parts" / "nearmiss_train.jsonl"


def load_chunks(cfg: dict) -> list[dict]:
    return read_jsonl(resolve(cfg, "chunks_clean"))


def boiler_for(chunks: list[dict], cfg: dict) -> dict[str, list[str]]:
    skip = skip_log_ids([INSTR / "draft_skip_log.jsonl", TEST_DIR / "skipped_chunks.jsonl"])
    return boilerplate_map(chunks, skip, cfg)


def style_of(q: str) -> str:
    """Rough question style used only to alternate styles in the cap (the type label is unreliable)."""
    ql = q.lower()
    if re.search(r"\bhow (many|much|long|often)\b|\bwhat (share|proportion|percentage|year)\b|\bwhen\b|\bin which year\b", ql):
        return "number"
    if re.search(r"^(how|which steps|what should|what will|what measures)", ql):
        return "procedure"
    if re.search(r"^(what is|what are|who|why|which)", ql):
        return "definition"
    return "other"


def history_pass_lookup(cfg: dict) -> dict[tuple[str, str], str]:
    """(chunk_id, question) -> old Stage A verdict for answerable drafts."""
    p = resolve(cfg, "verdict_history")
    out = {}
    if p.exists():
        for r in read_jsonl(p):
            if r["type"] == "answerable":
                out[(r["chunk_id"], norm_ws(r["question"]))] = r["ai_verdict"]
    return out


def load_train_drafts(cfg: dict, chunks: list[dict]) -> list[dict]:
    """Answerable drafts (train + val chunks), stable order, with the old verdict attached."""
    ids = [c["id"] for c in chunks]
    drafts, _skips = read_parts(INSTR / "drafts_parts", ids)
    order = {cid: i for i, cid in enumerate(ids)}
    drafts.sort(key=lambda d: (order[d["chunk_id"]], d["question"]))
    hist = history_pass_lookup(cfg)
    split = {c["id"]: c["split"] for c in chunks}
    src = {c["id"]: c["source_id"] for c in chunks}
    for i, d in enumerate(drafts):
        d["idx"] = i
        d["split"] = split[d["chunk_id"]]
        d["source_id"] = src[d["chunk_id"]]
        d["prior_verdict"] = hist.get((d["chunk_id"], norm_ws(d["question"])))
        d["style"] = style_of(d["question"])
    return drafts


def load_nearmiss(path: Path, chunk_ids: list[str], full_ids: bool) -> list[dict]:
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cid = d["chunk_id"] if full_ids else resolve_chunk_id(d["c"], chunk_ids)
            out.append({"chunk_id": cid, "question": d.get("question") or d["q"], "missing": d.get("missing", "")})
    return out


def load_test_drafts(cfg: dict, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    test_ids = {c["id"] for c in chunks if c["split"] == "test"}
    order = {c["id"]: i for i, c in enumerate(chunks)}
    raw = read_jsonl(TEST_DIR / "drafts_test_a2_raw.jsonl")
    drafts = []
    for r in raw:
        assert r["chunk_id"] in test_ids, f"not a test chunk: {r['chunk_id']}"
        assert r["type"] in DRAFT_TYPES, r["type"]
        drafts.append({"chunk_id": r["chunk_id"], "type": r["type"], "question": r["question"], "answer": r["answer"]})
    drafts.sort(key=lambda d: (order[d["chunk_id"]], d["question"]))
    for i, d in enumerate(drafts):
        d["idx"] = i
        d["split"] = "test"
        d["source_id"] = next(c["source_id"] for c in chunks if c["id"] == d["chunk_id"])
        d["prior_verdict"] = None
        d["style"] = style_of(d["question"])
    near = load_nearmiss(TEST_DIR / "nearmiss_test_a2_raw.jsonl", list(test_ids), full_ids=True)
    for n in near:
        assert n["chunk_id"] in test_ids
    return drafts, near
