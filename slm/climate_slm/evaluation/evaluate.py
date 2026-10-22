"""The one evaluator. Same code path for the base model, a QLoRA adapter and a DPO adapter.

    python -m climate_slm.evaluation.evaluate --model base --split test --out outputs/eval_base_test.json
    python -m climate_slm.evaluation.evaluate --model <adapter folder or model id> --split val --name qlora
    python -m climate_slm.evaluation.evaluate --model base --split train      # Phase 4 pairs, untuned base only
    python -m climate_slm.evaluation.evaluate --model base --split test --prompt-variant fewshot      # base_p_fewshot
    python -m climate_slm.evaluation.evaluate --model base --split test --decoding qwen_default --seed 1   # base_qwen_s1
    python -m climate_slm.evaluation.evaluate --model base --base-model-id Qwen/Qwen2.5-1.5B-Instruct --name base15 --split val
    python -m climate_slm.evaluation.evaluate --model outputs/adapters/qlora_4bit_lr2e-4/e2 --load-4bit --split val --name qlora_4bit_lr2e-4_e2__on4bit
    python -m climate_slm.evaluation.evaluate --model outputs/adapters/q15_lora_fp16/e2 --base-model-id Qwen/Qwen2.5-1.5B-Instruct --split val --name q15_lora_fp16_e2

torch / transformers are imported only inside HFBackend, so the laptop and the tests work without them.
A backend is any object with: info() -> dict, generate(list of message lists) -> list of
{'text', 'n_new_tokens', 'hit_max_new_tokens'}, completion_nll(list of message lists, list of completions)
-> list of (NLL sum, token count) or None."""
import argparse
import datetime
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from climate_slm.chat import completion_text
from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.quant import bnb_config, quant_settings
from climate_slm.evaluation import prompts as prompt_lib
from climate_slm.evaluation import scoring


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "accelerate", "peft", "tokenizers", "huggingface_hub", "rouge-score"):
        out[pkg] = _version(pkg)
    return out


def decoding_params(cfg: dict, mode: str) -> dict:
    """Generation settings of a decoding mode. `greedy` is exactly the top-level settings of the config."""
    e = cfg["eval"]
    if mode not in e["decoding_modes"]:
        raise ValueError(f"unknown decoding mode {mode!r}; choose from {sorted(e['decoding_modes'])}")
    base = {k: e[k] for k in ("do_sample", "max_new_tokens", "repetition_penalty")}
    return {**base, **e["decoding_modes"][mode]}


def check_adapter_base(adapter_dir: str | Path, base_model_id: str) -> str | None:
    """The base model id stored in the adapter folder. Raises if it differs from the base that is about to be
    used (for example a 1.5B adapter evaluated on the 0.5B base), instead of giving silently wrong numbers."""
    cfg_file = Path(adapter_dir) / "adapter_config.json"
    stored = json.loads(cfg_file.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if stored and stored != base_model_id:
        raise ValueError(f"adapter {adapter_dir} was trained on {stored}, but the base model here is {base_model_id}. "
                         "Pass --base-model-id to name the right base.")
    return stored


def validate_model_args(model_arg: str, base_model_id: str | None) -> None:
    """--base-model-id is for the untuned base or for an adapter folder, never for a full model given by --model."""
    is_adapter = (Path(model_arg) / "adapter_config.json").exists()
    if base_model_id and model_arg != "base" and not is_adapter:
        raise ValueError("--base-model-id is for --model base or for an adapter folder, not for a full model")


def check_train_split(model_arg: str, base_model_id: str | None, prompt_variant: str, decoding: str,
                      load_4bit: bool) -> None:
    """The train split is only for the plain untuned 0.5B base (Phase 4 preference pairs): the rejected answers are the
    untuned model's greedy output on the train prompts. It must never be used for a tuned model."""
    if model_arg != "base":
        raise ValueError("--split train is only for the untuned base (--model base), never for a tuned model")
    if base_model_id or load_4bit or prompt_variant != "default" or decoding != "greedy":
        raise ValueError("--split train needs the plain base: default prompt, greedy, fp16, no --base-model-id, "
                         "no --load-4bit")


def _batches(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class HFBackend:
    """Real model through transformers. Needs a GPU machine (Colab T4)."""

    def __init__(self, cfg: dict, model_arg: str = "base", decoding: str = "greedy", seed: int | None = None,
                 base_model_id: str | None = None, load_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        e = cfg["eval"]
        self.e, self.torch = e, torch
        self.params, self.seed = decoding_params(cfg, decoding), seed
        self.adapter = None
        model_id = base_model_id or e["model_id"]            # the config default unless --base-model-id is given
        if model_arg != "base":
            if (Path(model_arg) / "adapter_config.json").exists():
                self.adapter = model_arg                     # adapter on top of the base model
                check_adapter_base(model_arg, model_id)
            elif base_model_id:
                raise ValueError("--base-model-id is for the untuned base or for an adapter folder, not for a full model")
            else:
                model_id = model_arg                         # another full model (hub id or folder)
        self.model_id = model_id
        self.load_4bit, self.quant = load_4bit, None
        qkw = {}
        if load_4bit:                                        # the base the QLoRA adapters were trained on
            from climate_slm.config import load_phase3_config
            method = load_phase3_config()["methods"]["qlora_4bit"]
            self.quant = quant_settings(method)
            qkw = {"quantization_config": bnb_config(method, torch), "device_map": {"": 0}}
        dtype = getattr(torch, e["dtype"])
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=e["model_revision"])
        self.tok.padding_side = e["padding_side"]
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=e["model_revision"], dtype=dtype, **qkw)
        except TypeError:                                    # older transformers use the old argument name
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=e["model_revision"], torch_dtype=dtype, **qkw)
        if not load_4bit and next(model.parameters()).dtype != dtype:   # make sure the dtype from the config is really used
            model = model.to(dtype)
        self.revision = getattr(model.config, "_commit_hash", None)
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
        self.model = (model if load_4bit else model.to(e["device"])).eval()     # a 4-bit model is already on the GPU
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        self.eos_ids = {i for i in list(eos) + [self.tok.eos_token_id] if i is not None}

    def info(self) -> dict:
        torch = self.torch
        return {"model_id": self.model_id, "revision": self.revision, "adapter": self.adapter,
                "dtype": self.e["dtype"], "load_4bit": self.load_4bit, "quantization": self.quant,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    def _template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)

    def _prompt_text(self, messages: list[dict]) -> str:
        return self._template(messages, True)

    def generate(self, prompts: list[list[dict]]) -> list[dict]:
        torch, e, p, out = self.torch, self.e, self.params, []
        if self.seed is not None:                            # sampling runs: same seed, same batches, same draws
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        for batch in _batches(prompts, e["batch_size"]):
            texts = [self._prompt_text(m) for m in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, do_sample=p["do_sample"], max_new_tokens=p["max_new_tokens"],
                                          repetition_penalty=p["repetition_penalty"], temperature=p.get("temperature"),
                                          top_p=p.get("top_p"), top_k=p.get("top_k"), pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]         # only the new tokens
            for row in new.tolist():
                stop = next((i for i, t in enumerate(row) if t in self.eos_ids), None)
                ids = row if stop is None else row[:stop]
                out.append({"text": self.tok.decode(ids, skip_special_tokens=True).strip(), "n_new_tokens": len(ids),
                            "hit_max_new_tokens": stop is None and len(row) >= p["max_new_tokens"]})
        return out

    def _completion_text(self, messages: list[dict], completion: str, prompt_text: str) -> str:
        """The assistant turn as the chat template writes it (answer + end-of-turn token). Same helper as training."""
        return completion_text(self._template, messages, completion, prompt_text, self.tok.eos_token or "")

    def completion_nll(self, prompts: list[list[dict]], completions: list[str]) -> list[tuple[float, int]]:
        """Sum of token NLL over the completion tokens only (the prompt is masked out)."""
        torch, e, out = self.torch, self.e, []
        pad = self.tok.pad_token_id
        pairs = []
        for m, c in zip(prompts, completions):
            p_text = self._prompt_text(m)
            p_ids = self.tok(p_text, add_special_tokens=False)["input_ids"]
            c_ids = self.tok(self._completion_text(m, c, p_text), add_special_tokens=False)["input_ids"]
            pairs.append((p_ids, c_ids))
        for batch in _batches(pairs, e["ppl_batch_size"]):
            width = max(len(p) + len(c) for p, c in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for i, (p, c) in enumerate(batch):                # right padding here, positions stay simple
                ids[i, :len(p) + len(c)] = torch.tensor(p + c)
                mask[i, :len(p) + len(c)] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=ids.to(self.model.device),
                                    attention_mask=mask.to(self.model.device)).logits
            for i, (p, c) in enumerate(batch):
                # the logit at position t predicts token t + 1
                lp = torch.log_softmax(logits[i, len(p) - 1:len(p) + len(c) - 1].float(), dim=-1)
                target = torch.tensor(c, device=lp.device).unsqueeze(1)
                out.append((float(-lp.gather(1, target).sum()), len(c)))
        return out


def run_eval(backend, cfg: dict, split: str, name: str, limit: int | None = None, out_path: Path | None = None,
             write_review: bool = True, prompt_variant: str = "default", decoding: str = "greedy",
             seed: int | None = None) -> dict:
    """`name` is the final run name (see prompts.run_name). The backend must have been built with the same
    decoding mode and seed; they are written into the output file."""
    e = cfg["eval"]
    if decoding != "greedy" and seed is None:
        raise ValueError(f"decoding mode {decoding!r} needs a seed")
    if split == "train" and backend.info().get("adapter"):
        raise ValueError("the train split must never be used for a tuned model (adapter loaded)")
    data_path = REPO_ROOT / cfg["paths"]["splits"][split]
    items = scoring.load_items(data_path, limit)
    prompts, pinfo = prompt_lib.build_prompts(items, prompt_variant, cfg)    # perplexity uses these too
    gens = backend.generate(prompts)
    nlls = backend.completion_nll(prompts, [it["completion"][0]["content"] for it in items])
    records = [scoring.score_record(it, g, nlls[i] if nlls else None, cfg, prompts[i])
               for i, (it, g) in enumerate(zip(items, gens))]
    info = backend.info()
    result = {
        "run_name": name, "split": split, "limit": limit, **pinfo, "decoding_mode": decoding, "seed": seed,
        "model_id": info.get("model_id"), "base_model_id": info.get("model_id"),
        "load_4bit": bool(info.get("load_4bit", False)), "quantization": info.get("quantization"),
        "model_revision": info.get("revision"), "adapter": info.get("adapter"),
        "dtype": info.get("dtype"), "device": info.get("device"),
        "decoding": ({**{k: e[k] for k in ("do_sample", "max_new_tokens", "repetition_penalty", "padding_side",
                                              "batch_size")}, **e["decoding_modes"][decoding]}
                     if info.get("uses_decoding", True) else None),
        "chat_template": "apply_chat_template(messages, add_generation_prompt=True)" if info.get("uses_decoding", True)
        else None,
        "data_file": cfg["paths"]["splits"][split], "data_sha256": scoring.file_sha256(data_path),
        "refusal_text": cfg["refusal_text"], "loose_refusal_phrases": e["loose_refusal_phrases"],
        "bootstrap": {"n_boot": e["n_boot"], "seed": e["boot_seed"], "level": e["ci_level"]},
        "library_versions": library_versions(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "summary": scoring.summarize(records, cfg), "items": records,
    }
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        if write_review and not limit:
            scoring.write_review_csv(out_path.parent / f"refusal_detection_review_{name}_{split}.csv", records)
    return result


def default_out(cfg: dict, name: str, split: str) -> Path:
    return REPO_ROOT / cfg["paths"]["outputs_dir"] / f"eval_{name}_{split}.json"


def print_short(result: dict) -> None:
    s = result["summary"]
    print(json.dumps({"run_name": result["run_name"], "split": result["split"], "n_items": s["n_items"],
                      "answerable": s["answerable"], "refusal": s["refusal"], "perplexity": s["perplexity"],
                      "output_length_chars": s["output_length_chars"], "hit_max_new_tokens": s["hit_max_new_tokens"]},
                     indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a model on val or test.")
    ap.add_argument("--model", default="base", help="'base', an adapter folder, or a model id / folder")
    ap.add_argument("--split", choices=["val", "test", "train"], required=True,
                    help="train is only for --model base (Phase 4 preference pairs)")
    ap.add_argument("--name", default=None, help="run name (default: 'base' or the folder name)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-model-id", default=None,
                    help="base model for 'base' or for an adapter folder (default: eval.model_id of the config)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the base in 4-bit (nf4, same config as QLoRA training) instead of float16")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N items")
    ap.add_argument("--prompt-variant", default="default", help="default | strict_sentence | fewshot (config)")
    ap.add_argument("--decoding", default="greedy", help="greedy (default) | qwen_default (sampling, needs --seed)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for a sampling decoding mode")
    a = ap.parse_args(argv)
    cfg = load_eval_config()
    if a.prompt_variant not in cfg["eval"]["prompt_variants"]:
        ap.error(f"--prompt-variant must be one of {sorted(cfg['eval']['prompt_variants'])}")
    if a.decoding not in cfg["eval"]["decoding_modes"]:
        ap.error(f"--decoding must be one of {sorted(cfg['eval']['decoding_modes'])}")
    if a.decoding != "greedy" and a.seed is None:
        ap.error(f"--decoding {a.decoding} needs --seed")
    try:
        validate_model_args(a.model, a.base_model_id)
        if a.split == "train":
            check_train_split(a.model, a.base_model_id, a.prompt_variant, a.decoding, a.load_4bit)
    except ValueError as err:
        ap.error(str(err))
    base_name = a.name or ("base" if a.model == "base" else Path(a.model).name)
    name = prompt_lib.run_name(base_name, a.prompt_variant, a.decoding, a.seed, cfg)
    out = Path(a.out) if a.out else default_out(cfg, name, a.split)
    result = run_eval(HFBackend(cfg, a.model, a.decoding, a.seed, a.base_model_id, a.load_4bit), cfg, a.split, name, a.limit, out,
                      prompt_variant=a.prompt_variant, decoding=a.decoding, seed=a.seed)
    print_short(result)
    if a.limit:
        for r in result["items"]:
            print("-" * 60, f"\n[{r['id']}] ({r['kind']}) Q: {r['question']}\nREFERENCE: {r['reference']}\nOUTPUT: {r['output']}")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
