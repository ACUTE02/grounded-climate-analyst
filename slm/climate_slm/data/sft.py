"""SFT example format: prompt (system + user messages), completion (one assistant message), meta."""
from climate_slm.config import system_prompt

REQUIRED_KEYS = ("prompt", "completion", "meta")
_SEP = "\n\nQuestion: "


def user_message(chunk_text: str, question: str) -> str:
    return f"Context:\n{chunk_text}{_SEP}{question}"


def make_example(cfg: dict, chunk_text: str, question: str, answer: str, meta: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt(cfg)},
            {"role": "user", "content": user_message(chunk_text, question)},
        ],
        "completion": [{"role": "assistant", "content": answer}],
        "meta": meta,
    }


def parse_example(ex: dict) -> tuple[str, str, str]:
    """Return (chunk_text, question, answer) from an SFT example."""
    user = ex["prompt"][1]["content"]
    body = user[len("Context:\n"):]
    chunk, question = body.rsplit(_SEP, 1)
    return chunk, question, ex["completion"][0]["content"]


def set_question_answer(ex: dict, question=None, answer=None) -> None:
    chunk, q, a = parse_example(ex)
    ex["prompt"][1]["content"] = user_message(chunk, question if question is not None else q)
    ex["completion"][0]["content"] = answer if answer is not None else a
