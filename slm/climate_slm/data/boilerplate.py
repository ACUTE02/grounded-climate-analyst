"""Heuristic for boilerplate chunks (menus, contents, abbreviation lists, forms, tables, name lists)."""
import json
import re
from pathlib import Path


def boilerplate_reasons(text: str, cfg: dict) -> list[str]:
    b = cfg["boilerplate"]
    reasons: list[str] = []
    low = text.lower()
    head = low[: b["head_chars"]]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    short_ratio = (sum(1 for ln in lines if len(ln.strip()) <= b["short_line_max_chars"]) / len(lines)) if lines else 0.0
    if re.search(b["dotted_leader_regex"], text):
        reasons.append("dotted_leaders")
    # a heading word only counts when the chunk is short or mostly short lines (else it is normal text)
    head_ok = len(text) < b["head_phrase_max_chars"] or short_ratio >= b["head_phrase_min_short_ratio"]
    for ph in (b["head_phrases"] if head_ok else []):
        # phrase as a line by itself or heading near the start of the chunk
        if re.search(r"(^|\n)\s*[\d.\s]*" + re.escape(ph) + r"\b[^\n]{0,40}(\n|$)", head):
            reasons.append(f"head_phrase:{ph}")
            break
    if len(lines) >= b["short_line_min_lines"] and short_ratio >= b["short_line_ratio"]:
        reasons.append("mostly_short_lines")
    if len(re.findall(b["name_list_regex"], text)) >= b["name_list_min_hits"]:
        reasons.append("name_list")
    alnum = [ch for ch in text if ch.isalnum()]
    if alnum and sum(ch.isdigit() for ch in alnum) / len(alnum) > b["digit_ratio_max"]:
        reasons.append("high_digit_ratio")
    if len(text.split()) < b["min_words"]:
        reasons.append("too_short")
    if len(re.findall(b["legacy_font_regex"], text)) >= b["legacy_font_min_hits"]:
        reasons.append("legacy_font_garbage")
    return reasons


def skip_log_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    ids.add(d.get("chunk_id") or d.get("c"))
    return ids


def boilerplate_map(chunks: list[dict], skip_ids: set[str], cfg: dict) -> dict[str, list[str]]:
    """chunk id -> list of reasons (empty list = usable). Skip-log chunks get reason 'skip_log'."""
    out = {}
    for c in chunks:
        r = boilerplate_reasons(c["text"], cfg)
        if c["id"] in skip_ids:
            r = ["skip_log"] + r
        out[c["id"]] = r
    return out
