"""Turn SFT rows into tokenised training examples with the SAME chat template the evaluator uses.

`template(messages, add_generation_prompt) -> str` and `tokenize(text) -> list[int]` are passed in, so the
code runs with a fake tokenizer on the laptop and with the real Qwen tokenizer on Colab. Tokens are made
without special-token insertion, exactly like the evaluator's perplexity pass."""
from climate_slm.chat import completion_text, prompt_text
from climate_slm.training.masking import build_example, length_stats


def encode_row(row: dict, template, tokenize, max_len: int, eos_token: str = "") -> tuple[dict | None, int]:
    """(example or None, total length in tokens). The length is returned even if the row is dropped."""
    messages = row["prompt"]
    p_text = prompt_text(template, messages)
    c_text = completion_text(template, messages, row["completion"][0]["content"], p_text, eos_token)
    p_ids, c_ids = tokenize(p_text), tokenize(c_text)
    ex = build_example(p_ids, c_ids, max_len)
    if ex is not None:
        ex["id"] = row["meta"]["id"]
    return ex, len(p_ids) + len(c_ids)


def encode_rows(rows: list[dict], template, tokenize, max_len: int, eos_token: str = "") -> dict:
    """{'examples', 'dropped_ids', 'stats'}. Rows that do not fit are dropped, never cut."""
    examples, dropped, lengths = [], [], []
    for row in rows:
        ex, n = encode_row(row, template, tokenize, max_len, eos_token)
        lengths.append(n)
        if ex is None:
            dropped.append(row["meta"]["id"])
        else:
            examples.append(ex)
    return {"examples": examples, "dropped_ids": dropped, "stats": length_stats(lengths, max_len)}
