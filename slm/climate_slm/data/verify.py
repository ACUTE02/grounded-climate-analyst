"""Evidence-quote matching and coverage checks for the independent AI verification pass."""
import re

from climate_slm.data.textutil import norm_ws, numbers, words

VALID_VERDICTS = ("pass", "fix", "fail")
_FOLD = str.maketrans({
    "‘": "'", "’": "'", "ʼ": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    "\u00ad": "", "\u200b": "",
})


def fold(text: str) -> str:
    """Fold curly quotes / unicode dashes, drop soft hyphens, tidy spaces around hyphens, collapse whitespace."""
    t = (text or "").translate(_FOLD)
    t = re.sub(r"\s*-\s*", "-", t)  # hyphenation artefacts: 'inter -agency', 'co- operative'
    return norm_ws(t)


def evidence_ok(quote: str, chunk_text: str, min_chars: int = 15) -> bool:
    """True if quote is an exact substring of the chunk after normalisation."""
    q = fold(quote)
    if len(q) < min_chars:
        return False
    return q in fold(chunk_text)


def _tokens(text: str) -> list[str]:
    return fold(text).split(" ")


def fuzzy_ok(quote: str, chunk_text: str, min_frac: float, span_factor: float) -> bool:
    """Words of the quote appear in order in the chunk (allowing small gaps), >= min_frac of them."""
    q, c = _tokens(quote), _tokens(chunk_text)
    if not q or not c:
        return False
    window = int(span_factor * len(q) + 5)
    first = q[0]
    for start in [i for i, t in enumerate(c) if t == first]:
        pos, matched = start, 1
        for tok in q[1:]:
            end = min(len(c), start + window)
            nxt = next((j for j in range(pos + 1, end) if c[j] == tok), None)
            if nxt is not None:
                pos, matched = nxt, matched + 1
        if matched / len(q) >= min_frac:
            return True
    return False


def quote_match(quote: str, chunk_text: str, cfg: dict) -> str | None:
    v = cfg["verification"]
    if evidence_ok(quote, chunk_text, v["evidence_min_chars"]):
        return "exact"
    if len(fold(quote)) >= v["evidence_min_chars"] and fuzzy_ok(quote, chunk_text, v["fuzzy_match_min"],
                                                                 v["fuzzy_span_factor"]):
        return "fuzzy"
    return None


def coverage_ok(answer: str, quotes: list[str], cfg: dict) -> tuple[bool, dict]:
    """Every number of the answer is in some quote, and enough content words are in the union of quotes."""
    v = cfg["verification"]
    joined = " ".join(quotes)
    qnums = set(numbers(joined))
    missing_nums = [n for n in numbers(answer) if n not in qnums]
    aw = words(answer, cfg["checks"]["min_word_len"])
    qw = set(words(fold(joined), cfg["checks"]["min_word_len"]))
    frac = (sum(1 for w in aw if w in qw) / len(aw)) if aw else 1.0
    ok = not missing_nums and frac >= v["evidence_min_coverage"]
    return ok, {"missing_numbers": missing_nums, "word_coverage": round(frac, 3)}
