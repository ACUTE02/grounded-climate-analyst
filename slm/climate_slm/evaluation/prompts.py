"""Prompt variants for the evaluation (Phase 2c).

  default          the stored prompt of the item, unchanged
  strict_sentence  same context and question; the system message gets an extra sentence (config)
  fewshot          strict_sentence system message + two worked examples as earlier chat turns

The two worked examples come only from the TRAIN split and are named in the config (`eval.fewshot.ids`).
`pick_fewshot_ids` is the seeded rule that chose them."""
import random

from climate_slm.config import REPO_ROOT
from climate_slm.data.sft import parse_example
from climate_slm.evaluation.scoring import kind_of, load_items


def variant_config(cfg: dict, name: str) -> dict:
    """The variant's settings with `extends` resolved (the child's own keys win)."""
    variants = cfg["eval"]["prompt_variants"]
    if name not in variants:
        raise ValueError(f"unknown prompt variant {name!r}; choose from {sorted(variants)}")
    v = dict(variants[name])
    parent = v.pop("extends", None)
    return {**variant_config(cfg, parent), **v} if parent else v


def system_message(base_system: str, variant: dict, refusal_text: str) -> str:
    suffix = variant.get("system_suffix")
    if not suffix:
        return base_system
    return base_system.rstrip() + " " + " ".join(suffix.format(refusal_text=refusal_text).split())


def load_fewshot(cfg: dict) -> list[dict]:
    """The worked examples (SFT items from the train split) in the configured order."""
    f = cfg["eval"]["fewshot"]
    by_id = {it["meta"]["id"]: it for it in load_items(REPO_ROOT / cfg["paths"]["splits"][f["split"]])}
    out = []
    for kind in f["order"]:
        i = f["ids"][kind]
        if i not in by_id:
            raise ValueError(f"few-shot id {i} is not in the {f['split']} split")
        if kind_of(by_id[i]) != kind:
            raise ValueError(f"few-shot id {i} is not a {kind} item")
        out.append(by_id[i])
    return out


def build_prompt(item: dict, variant: dict, cfg: dict, shots: list[dict] | None = None) -> list[dict]:
    """Messages for one item under a variant. `default` returns the stored messages (a copy)."""
    system, user = item["prompt"][0], item["prompt"][1]
    msgs = [{"role": "system", "content": system_message(system["content"], variant, cfg["refusal_text"])}]
    if variant.get("fewshot_examples"):
        for sh in shots or []:
            msgs.append({"role": "user", "content": sh["prompt"][1]["content"]})
            msgs.append({"role": "assistant", "content": sh["completion"][0]["content"]})
    msgs.append(dict(user))
    return msgs


def build_prompts(items: list[dict], variant_name: str, cfg: dict) -> tuple[list[list[dict]], dict]:
    """(messages per item, info for the output file)."""
    v = variant_config(cfg, variant_name)
    shots = load_fewshot(cfg) if v.get("fewshot_examples") else []
    prompts = [build_prompt(it, v, cfg, shots) for it in items]
    info = {"prompt_variant": variant_name,
            "fewshot_ids": [s["meta"]["id"] for s in shots] if shots else None,
            "system_prompt": prompts[0][0]["content"] if prompts else None}
    return prompts, info


def run_name(name: str, variant_name: str, decoding: str, seed: int | None, cfg: dict) -> str:
    """base -> base_p_sentence -> base_p_sentence_qwen_s1. The default variant and greedy add nothing."""
    e = cfg["eval"]
    parts = [name]
    suffix = variant_config(cfg, variant_name).get("run_suffix")
    if suffix:
        parts.append(suffix)
    if decoding != "greedy":
        parts.append(f"{e['decoding_run_suffix']}{seed}")
    return "_".join(parts)


def pick_fewshot_ids(train_items: list[dict], cfg: dict) -> dict[str, str]:
    """Seeded choice of one answerable and one refusal worked example from the train items.

    Pool: spot_check in `require_spot_check`, no blind flag, context <= max_context_chars, and for the
    answerable one a reference answer <= max_answer_chars that is a single sentence."""
    f = cfg["eval"]["fewshot"]
    rng = random.Random(f["seed"])
    pool: dict[str, list[str]] = {"answerable": [], "refusal": []}
    for it in sorted(train_items, key=lambda x: x["meta"]["id"]):
        m = it["meta"]
        context, _, answer = parse_example(it)
        if m["spot_check"] not in f["require_spot_check"] or m["blind_flag"] or len(context) > f["max_context_chars"]:
            continue
        if kind_of(it) == "answerable":
            one_sentence = answer.strip().rstrip(".").count(". ") == 0
            if len(answer) > f["max_answer_chars"] or not one_sentence:
                continue
        pool[kind_of(it)].append(m["id"])
    return {kind: rng.choice(pool[kind]) for kind in ("answerable", "refusal")}
