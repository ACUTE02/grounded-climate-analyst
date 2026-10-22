"""Cleaning of chunk text. Never rewrites numbers or units."""
import re
from collections import Counter


def clean_text(text: str, cfg: dict) -> tuple[str, Counter]:
    c = cfg["cleaning"]
    stats: Counter = Counter()
    header = re.compile(c["page_header_regex"])
    hyphen = re.compile(c["hyphen_join_regex"])

    lines = []
    for line in text.split("\n"):
        if header.match(line):
            stats["page_header_removed"] += 1
            continue
        lines.append(line)
    out = "\n".join(lines)

    out, n = hyphen.subn(r"\1\2", out)
    stats["hyphen_split_joined"] += n

    out, n = re.subn(c["unicode_space_regex"], " ", out)
    stats["unicode_space_replaced"] += n

    before = out
    out = re.sub(r"[ \t]+", " ", out)
    if out != before:
        stats["space_runs_collapsed"] += 1
    before = out
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    if out != before:
        stats["newline_runs_collapsed"] += 1
    out = out.strip()
    return out, stats
