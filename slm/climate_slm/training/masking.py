"""Completion-only loss masking and batching, on plain lists of token ids (no torch, testable on the laptop).

A training example is the prompt tokens followed by the completion tokens (the answer and the end-of-turn token).
Labels are -100 for every prompt token, so the loss is computed on the completion only."""
import math

IGNORE_INDEX = -100


def build_example(prompt_ids: list[int], completion_ids: list[int], max_len: int) -> dict | None:
    """One example, or None if it cannot be used: no completion tokens, or prompt + completion longer than max_len.
    Nothing is ever cut: a row that does not fit is dropped by the caller (and its id is logged)."""
    if not completion_ids:
        return None
    if len(prompt_ids) + len(completion_ids) > max_len:
        return None
    return {"input_ids": list(prompt_ids) + list(completion_ids),
            "labels": [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids),
            "n_prompt": len(prompt_ids), "n_completion": len(completion_ids)}


def loss_token_count(labels: list[int]) -> int:
    """Number of tokens the loss is computed on for one example. Position 0 can never be predicted, so it is
    not counted (it is a prompt token anyway)."""
    return sum(1 for x in labels[1:] if x != IGNORE_INDEX)


def pad_batch(examples: list[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right padding. attention_mask is 1 for real tokens, labels are -100 on padding."""
    width = max(len(e["input_ids"]) for e in examples)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for e in examples:
        pad = width - len(e["input_ids"])
        out["input_ids"].append(e["input_ids"] + [pad_id] * pad)
        out["labels"].append(e["labels"] + [IGNORE_INDEX] * pad)
        out["attention_mask"].append([1] * len(e["input_ids"]) + [0] * pad)
    return out


def length_stats(lengths: list[int], max_len: int) -> dict:
    """min, median, p95 (nearest rank), max and how many are longer than max_len."""
    if not lengths:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None, "over_limit": 0}
    v = sorted(lengths)
    mid = len(v) // 2
    median = v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2
    return {"n": len(v), "min": v[0], "median": median, "p95": v[max(math.ceil(0.95 * len(v)) - 1, 0)],
            "max": v[-1], "over_limit": sum(1 for x in v if x > max_len)}
