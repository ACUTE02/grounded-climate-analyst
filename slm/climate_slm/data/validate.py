"""Automatic checks on one SFT example. Returns a list of flags (empty list = passed)."""
from climate_slm.data.sft import REQUIRED_KEYS, parse_example
from climate_slm.data.textutil import numbers, words


def check_example(ex: dict, cfg: dict) -> list[str]:
    c = cfg["checks"]
    flags: list[str] = []
    for k in REQUIRED_KEYS:
        if not ex.get(k):
            flags.append(f"missing_{k}")
    if flags:
        return flags
    try:
        chunk, question, answer = parse_example(ex)
    except (KeyError, IndexError, ValueError):
        return ["bad_format"]
    if not chunk.strip():
        flags.append("empty_chunk")
    if not question.strip():
        flags.append("empty_question")
    if not answer.strip():
        flags.append("empty_answer")
        return flags

    q = question.strip()
    if len(q) < c["question_min_chars"]:
        flags.append("question_too_short")
    if len(q) > c["question_max_chars"]:
        flags.append("question_too_long")
    if not q.endswith("?"):
        flags.append("question_no_question_mark")
    low = q.lower()
    if any(p in low for p in c["banned_question_phrases"]):
        flags.append("question_banned_phrase")

    if ex["meta"].get("type") == "unanswerable":
        if answer != cfg["refusal_text"]:
            flags.append("refusal_not_exact")
        return flags

    if len(answer) < c["answer_min_chars"]:
        flags.append("answer_too_short")
    if len(answer) > c["answer_max_chars"]:
        flags.append("answer_too_long")
    if answer == cfg["refusal_text"]:
        flags.append("answerable_has_refusal_text")
    chunk_nums = set(numbers(chunk))
    if any(n not in chunk_nums for n in numbers(answer)):
        flags.append("number_not_in_chunk")
    aw = words(answer, c["min_word_len"])
    if aw:
        cw = set(words(chunk, c["min_word_len"]))
        overlap = sum(1 for w in aw if w in cw) / len(aw)
        if overlap < c["min_support_overlap"]:
            flags.append("low_support_overlap")
    return flags
