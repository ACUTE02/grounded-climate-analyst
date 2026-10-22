"""Small text helpers shared by clean, validate, verify."""
import json
import re

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[A-Za-z]+")


def norm_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def numbers(text: str) -> list[str]:
    """Numbers as written, with thousands commas removed."""
    return [n.replace(",", "") for n in _NUM.findall(text)]


def words(text: str, min_len: int = 3) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if len(w) >= min_len]


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
