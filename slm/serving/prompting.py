"""The ONE function that builds the chat messages from (context, question). Used by the API, by the parity check and by the
smoke script, and tested against the real rows of data/instruction/sft_val.jsonl: the messages are identical to the stored
prompt of the training and evaluation data (same system message, same "Context:\\n...\\n\\nQuestion: ..." user text). The chat
template is applied later by the backend, with the same helper the evaluator uses (climate_slm/chat.py)."""
from climate_slm.config import load_config, system_prompt
from climate_slm.data.sft import user_message


def build_messages(context: str, question: str, cfg1: dict | None = None) -> list[dict]:
    """[system, user] messages. The context and the question are used exactly as given (no stripping, no cutting)."""
    cfg1 = cfg1 or load_config()
    return [{"role": "system", "content": system_prompt(cfg1)},
            {"role": "user", "content": user_message(context, question)}]
