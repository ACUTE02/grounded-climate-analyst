"""Evaluation metrics. Pure Python, CPU only, no torch.

Normalisation rule (one function, used everywhere): Unicode NFKC, lowercase, thousands commas removed
(1,000 -> 1000), apostrophes removed (can't -> cant), every other punctuation mark or symbol becomes a space
except a dot between two digits (4.5 stays 4.5), the articles a / an / the are removed, spaces are collapsed.

Number rule: numbers are compared by value (4.50 = 4.5, 1,000 = 1000). The source text sometimes lost the
degree sign, so "4.50C" means 4.5 C and "370C" means 37 C. On the context side a number written directly before
"0C" / "0 C" / "0F" therefore counts in both readings (370 and 37)."""
import math
import random
import re
import unicodedata
from collections import Counter
from decimal import Decimal

ARTICLES = ("a", "an", "the")
_APOSTROPHES = "'’‘ʼ`"
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUM = re.compile(r"(?<![A-Za-z\d])\d+(?:\.\d+)?")
_LOST_DEGREE = re.compile(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)\s?0\s?[CF]\b")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _THOUSANDS.sub("", t)
    out = []
    for i, ch in enumerate(t):
        if ch in _APOSTROPHES:
            continue
        if unicodedata.category(ch)[0] in "PS":
            keep = ch == "." and 0 < i < len(t) - 1 and t[i - 1].isdigit() and t[i + 1].isdigit()
            out.append(ch if keep else " ")
        else:
            out.append(ch)
    return " ".join(w for w in "".join(out).split() if w not in ARTICLES)


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p and not r:
        return 1.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


_scorer = None


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure from the rouge-score package, on normalised text, no stemming."""
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer
        _scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    p, r = normalize(pred), normalize(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    return _scorer.score(r, p)["rougeL"].fmeasure


def _canon(num: str) -> str:
    d = Decimal(num)
    return format(d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize(), "f")


def numbers(text: str) -> list[str]:
    """Numbers in the text by value, in order (with repeats)."""
    t = _THOUSANDS.sub("", unicodedata.normalize("NFKC", text or ""))
    return [_canon(n) for n in _NUM.findall(t)]


def context_numbers(context: str) -> set[str]:
    """Numbers of the context, plus the lost-degree-sign reading (370C -> 37, 4.50C -> 4.5, 40 0C -> 40)."""
    t = _THOUSANDS.sub("", unicodedata.normalize("NFKC", context or ""))
    return set(numbers(context)) | {_canon(n) for n in _LOST_DEGREE.findall(t)}


def number_faithfulness(answer: str, context: str) -> float | None:
    """Share of the numbers in the answer that also appear in the context. None if the answer has no numbers."""
    nums = numbers(answer)
    if not nums:
        return None
    ctx = context_numbers(context)
    return sum(n in ctx for n in nums) / len(nums)


def is_refusal_strict(text: str, refusal_text: str) -> bool:
    return normalize(text) == normalize(refusal_text)


def is_refusal_loose(text: str, refusal_text: str, phrases: list[str]) -> bool:
    """Strict match, or one of the phrases anywhere in the normalised text. Can be wrong (see the review CSV)."""
    t = normalize(text)
    if t == normalize(refusal_text):
        return True
    padded = f" {t} "
    return any(f" {normalize(p)} " in padded for p in phrases if normalize(p))


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def bootstrap_stat(items: list, stat, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile interval of stat(resample of items), resampling items with replacement. Seeded."""
    if not items:
        return None
    rng = random.Random(seed)
    n = len(items)
    vals = []
    for _ in range(n_boot):
        v = stat([items[rng.randrange(n)] for _ in range(n)])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    lo = vals[min(int(math.floor((1 - level) / 2 * len(vals))), len(vals) - 1)]
    hi = vals[max(int(math.ceil((1 + level) / 2 * len(vals))) - 1, 0)]
    return lo, hi


def bootstrap_ci(values, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile interval for the mean. For a rate pass 0/1 values."""
    return bootstrap_stat([float(v) for v in values], mean, n_boot, seed, level)


def wilson_ci(count: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Wilson score interval for a rate. Shown next to the bootstrap interval, because the bootstrap
    interval has zero width when every item has the same value (for example 0 of 11)."""
    if n == 0:
        return None
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---- key-answer metrics (Phase 2c) --------------------------------------------------------------
# These two scores are HEURISTICS (no model, no judge). They exist because a model that answers with a bare
# value ("24.8%", "cotton") is correct but short, so token F1 against a full-sentence reference is very low.
#
# Key tokens of a reference (answerable items only). Built from the NORMALISED text (see normalize()):
#   * numbers: every number of the reference (by value, same canonical form as number_faithfulness), except
#     numbers that also appear in the question. A number written with a lost degree sign ("370C", meaning 37 C)
#     counts as found when the output has 370 OR 37.
#   * words: every normalised word of the reference that is not a pure number, has at least KEY_MIN_WORD_LEN
#     letters, is not in STOP_WORDS, and does not appear in the question. A word matches a word of the other
#     text if they are equal or differ only by a final "s" (adopter / adopters). No stemming beyond that.
# key_recall  = share of the reference key tokens found in the output (None if the reference has no key token).
# key_support = share of the output's own key-type tokens (numbers and words by the same rules, taken from the
#               output; question words and stop words left out) that are found in the reference. None if the
#               output has none. It says whether what the model wrote is backed by the reference.
# A strict refusal output scores 0.0 on both (a refusal contains no answer).
# Known limit: a reference that also states other numbers (for example two yields and then the difference)
# lowers key_recall of a bare correct value, because the other numbers count as key tokens too.
# key_support is high for that bare value. Read the two together.
KEY_MIN_WORD_LEN = 3
STOP_WORDS = frozenset("""
a an the and or but nor if of to in on at by for with from as into onto over under about between among through
during before after above below up down out off than then so such too very also only just even still yet
is are was were be been being am do does did done doing has have had having it its this that these those there
here which who whom whose what when where why how i me my we us our you your he him his she her they them their
not no can could may might will would shall should must per via etc viz eg ie
more most less least some any each every all both either neither other another same own
one two three four five six seven eight nine ten
kg ha rs mm cm km sq tons ton lakh lakhs crore crores percent cent degree degrees
""".split())


def _is_number_token(tok: str) -> bool:
    return re.fullmatch(r"\d+(?:\.\d+)?", tok) is not None


def content_words(text: str) -> list[str]:
    """Normalised words that can carry an answer: not numbers, not short, not stop words. Order kept, no repeats."""
    seen, out = set(), []
    for t in tokens(text):
        if t in seen or _is_number_token(t) or len(t) < KEY_MIN_WORD_LEN or t in STOP_WORDS:
            continue
        seen.add(t)
        out.append(t)
    return out


def word_in(word: str, tokens_set: set[str]) -> bool:
    """Equal, or equal after adding / removing a final 's'."""
    return (word in tokens_set or word + "s" in tokens_set
            or (word.endswith("s") and len(word) > KEY_MIN_WORD_LEN and word[:-1] in tokens_set))


def _lost_degree_alternatives(text: str) -> dict[str, str]:
    """{'370': '37'} for '370C' in the text (the zero is the lost degree sign). '4.50C' maps 4.5 to 4.5."""
    t = _THOUSANDS.sub("", unicodedata.normalize("NFKC", text or ""))
    return {_canon(m.group(1) + "0"): _canon(m.group(1)) for m in _LOST_DEGREE.finditer(t)}


def key_tokens(reference: str, question: str) -> tuple[list[set[str]], list[str]]:
    """(number keys, word keys). A number key is a set of accepted values (the value and its lost-degree reading)."""
    alt = _lost_degree_alternatives(reference)
    q_nums = set(numbers(question))
    num_keys, seen = [], set()
    for n in numbers(reference):
        if n in seen or n in q_nums:
            continue
        seen.add(n)
        num_keys.append({n, alt[n]} if n in alt else {n})
    q_tok = set(tokens(question))
    return num_keys, [w for w in content_words(reference) if not word_in(w, q_tok)]


def key_recall(output: str, reference: str, question: str, refusal_text: str | None = None) -> float | None:
    num_keys, word_keys = key_tokens(reference, question)
    total = len(num_keys) + len(word_keys)
    if not total:
        return None
    if refusal_text is not None and is_refusal_strict(output, refusal_text):
        return 0.0
    out_nums, out_tok = set(numbers(output)), set(tokens(output))
    found = sum(1 for keys in num_keys if keys & out_nums) + sum(1 for w in word_keys if word_in(w, out_tok))
    return found / total


def key_support(output: str, reference: str, question: str, refusal_text: str | None = None) -> float | None:
    if refusal_text is not None and is_refusal_strict(output, refusal_text):
        return 0.0
    q_nums, q_tok = set(numbers(question)), set(tokens(question))
    out_nums = [n for n in dict.fromkeys(numbers(output)) if n not in q_nums]
    out_words = [w for w in content_words(output) if not word_in(w, q_tok)]
    total = len(out_nums) + len(out_words)
    if not total:
        return None
    ref_nums, ref_tok = context_numbers(reference), set(tokens(reference))
    found = sum(1 for n in out_nums if n in ref_nums) + sum(1 for w in out_words if word_in(w, ref_tok))
    return found / total
