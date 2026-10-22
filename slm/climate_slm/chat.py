"""Chat-template helpers shared by the evaluator and the training code, so both build the same text."""
from typing import Callable

# apply_template(messages, add_generation_prompt) -> text (not tokenised)
Template = Callable[[list, bool], str]


def prompt_text(apply_template: Template, messages: list[dict]) -> str:
    """The text the model sees before it answers (ends with the start of the assistant turn)."""
    return apply_template(messages, True)


def completion_text(apply_template: Template, messages: list[dict], completion: str, prompt: str,
                    eos_token: str = "") -> str:
    """The assistant turn as the chat template writes it: the answer plus the end-of-turn token.
    Falls back to answer + eos_token if the template does not start with the prompt text."""
    full = apply_template(messages + [{"role": "assistant", "content": completion}], False)
    if full.startswith(prompt):
        return full[len(prompt):].rstrip("\n")
    return completion + eos_token
