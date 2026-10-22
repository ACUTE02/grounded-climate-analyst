"""Blind-answer check (A3 G2): scoring by code only, no LLM judgement."""
from collections import Counter

from climate_slm.data.leakage import norm_tokens
from climate_slm.data.textutil import norm_ws, numbers

FLAG_KINDS = ("answered_a_refusal_item", "refused_an_answerable_item", "low_overlap", "number_not_in_context")


def is_refusal_text(answer: str, refusal_text: str) -> bool:
    return norm_ws(answer).lower() == norm_ws(refusal_text).lower()


def token_f1(pred: str, ref: str) -> float:
    p, r = norm_tokens(pred), norm_tokens(ref)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def score_item(is_refusal_item: bool, reference: str, blind: str, context: str, cfg: dict) -> tuple[bool, list[str]]:
    """Return (blind_ok, flags)."""
    refusal = cfg["refusal_text"]
    if is_refusal_item:
        ok = is_refusal_text(blind, refusal)
        return ok, ([] if ok else ["answered_a_refusal_item"])
    if is_refusal_text(blind, refusal):
        return False, ["refused_an_answerable_item"]
    flags = []
    if token_f1(blind, reference) < cfg["a3"]["blind_f1_min"]:
        flags.append("low_overlap")
    ctx_nums = set(numbers(context))
    if any(n not in ctx_nums for n in numbers(blind)):
        flags.append("number_not_in_context")
    return not flags, flags
