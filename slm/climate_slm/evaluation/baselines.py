"""Two baselines that need no model. They write the same output format as the model evaluator.

    python -m climate_slm.evaluation.baselines --split test
"""
import argparse
import re

from climate_slm.config import load_eval_config
from climate_slm.evaluation import metrics as M
from climate_slm.evaluation.evaluate import default_out, run_eval

_CONTEXT, _QUESTION = "Context:\n", "\n\nQuestion: "


def split_user_message(messages: list[dict]) -> tuple[str, str]:
    user = next(m["content"] for m in messages if m["role"] == "user")
    context, question = user[len(_CONTEXT):].rsplit(_QUESTION, 1)
    return context, question


class _NoModel:
    name = ""

    def info(self) -> dict:
        return {"model_id": None, "revision": None, "adapter": None, "dtype": None, "device": "cpu",
                "uses_decoding": False}

    def completion_nll(self, prompts, completions):
        return None                                           # no model, no perplexity

    def generate(self, prompts: list[list[dict]]) -> list[dict]:
        return [{"text": self.answer(m), "n_new_tokens": None, "hit_max_new_tokens": None} for m in prompts]


class AlwaysRefuse(_NoModel):
    name = "always_refuse"

    def __init__(self, cfg: dict):
        self.text = cfg["refusal_text"]

    def answer(self, messages: list[dict]) -> str:
        return self.text


class BestSentence(_NoModel):
    """The context sentence that shares the most different words with the question. Never refuses."""
    name = "best_sentence"

    def __init__(self, cfg: dict):
        self.b = cfg["eval"]["best_sentence"]

    def answer(self, messages: list[dict]) -> str:
        context, question = split_user_message(messages)
        flat = " ".join(context.split())
        sents = [s.strip() for s in re.split(self.b["sentence_split_regex"], flat) if s.strip()]
        sents = [s for s in sents if len(s) >= self.b["min_sentence_chars"]] or [flat]
        q = {w for w in M.tokens(question) if len(w) >= self.b["min_word_len"]}
        return max(sents, key=lambda s: len(q & set(M.tokens(s))))   # ties: the first sentence wins


BASELINES = {c.name: c for c in (AlwaysRefuse, BestSentence)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the no-model baselines.")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--only", choices=sorted(BASELINES), default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_eval_config()
    for name, cls in BASELINES.items():
        if a.only and name != a.only:
            continue
        out = default_out(cfg, name, a.split)
        s = run_eval(cls(cfg), cfg, a.split, name, a.limit, out)["summary"]
        an, rf = s["answerable"], s["refusal"]
        print(f"{name} {a.split}: answerable n={an['n']} F1={an['token_f1']['mean']:.3f} "
              f"false refusals (strict)={an['false_refusal_strict']['count']}; refusal n={rf['n']} "
              f"correct (strict)={rf['correct_refusal_strict']['count']} -> {out}")


if __name__ == "__main__":
    main()
