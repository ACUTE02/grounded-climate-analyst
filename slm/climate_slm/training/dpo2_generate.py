"""Phase 4b, Stage A (Colab): rejected-text candidates for the DPO v2 pairs. Needs a GPU; torch is imported only inside
the functions that need it, so the laptop and the tests never need it.

    python -m climate_slm.training.dpo2_generate        # cell P4b-0

Prompts: the stored prompts of the 21 REFUSAL rows of sft_train.jsonl (default prompt, train split only; no val or test
file is opened). Two models: the untuned 0.5B and the SFT 0.5B (the seed-42 start adapter of configs/phase4.yaml, checked
against its sha256). Per model: the SFT model also greedy once; both models `n_samples` sampled outputs with the decoding
mode `qwen_default` and the seeds sample_seed + k. The untuned GREEDY output is not generated again (it is read from
eval_base_train.json, made by the same evaluator code). This is NOT an evaluation: no number is computed here and the
evaluator's rule "no train split for a tuned model" (it protects reported numbers) is not touched; the texts are only
raw material for preference pairs, which the reviewer reads before any training.
Output: outputs/dpo2_candidates.json."""
import argparse
import datetime
import json
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase4_config, load_phase4b_config
from climate_slm.evaluation.evaluate import HFBackend, decoding_params, library_versions
from climate_slm.evaluation.scoring import file_sha256, load_items, kind_of


def refusal_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if kind_of(r) == "refusal"]


def sample_seeds(cfg4b: dict) -> list[int]:
    g = cfg4b["generation"]
    return [g["sample_seed"] + k for k in range(g["n_samples"])]


def generation_plan(cfg4b: dict) -> list[dict]:
    """What is generated, in order: (model, source name, decoding mode, seed). Untuned greedy is not in the plan."""
    g = cfg4b["generation"]
    plan = []
    for model, greedy_too in (("untuned", False), ("sft", True)):
        if greedy_too:
            plan.append({"model": model, "source": f"{model}_greedy", "decoding": "greedy", "seed": None})
        plan += [{"model": model, "source": f"{model}_sampled", "decoding": g["sampling_mode"], "seed": s}
                 for s in sample_seeds(cfg4b)]
    return plan


def run(cfg4b: dict, cfg4: dict, eval_cfg: dict) -> dict:
    """Generate on the GPU. Returns the candidates dict (also written to disk by main)."""
    from climate_slm.training import dpo_grid
    train_path = REPO_ROOT / cfg4b["paths"]["sft_train"]
    rows = refusal_rows(load_items(train_path))
    prompts = [r["prompt"] for r in rows]
    st = dpo_grid.start_info(cfg4, cfg4b["generation"]["sft_seed"])
    run_ = {"start_adapter": st["adapter"], "start_sha256": st["sha256"]}
    adapter_sha = dpo_grid.check_start_adapter(run_)                       # not the frozen SFT adapter: stop
    adapter = str(REPO_ROOT / st["adapter"])
    cands: dict[str, dict[str, list[str]]] = {r["meta"]["id"]: {} for r in rows}
    log = []
    for model in ("untuned", "sft"):
        backend = HFBackend(eval_cfg, "base" if model == "untuned" else adapter, "greedy", None, None)
        for step in [p for p in generation_plan(cfg4b) if p["model"] == model]:
            backend.params = decoding_params(eval_cfg, step["decoding"])    # same backend, other decoding settings
            backend.seed = step["seed"]
            print(f"=== {step['source']} seed {step['seed']} ({len(prompts)} prompts) ===", flush=True)
            gens = backend.generate(prompts)
            for r, g in zip(rows, gens):
                cands[r["meta"]["id"]].setdefault(step["source"], []).append(g["text"])
            log.append({**step, "hit_max_new_tokens": sum(bool(g["hit_max_new_tokens"]) for g in gens)})
        del backend
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return {"meta": {"data_file": cfg4b["paths"]["sft_train"], "data_sha256": file_sha256(train_path),
                     "n_rows": len(rows), "sft_adapter": st["adapter"], "sft_adapter_sha256": adapter_sha,
                     "n_samples": cfg4b["generation"]["n_samples"], "sample_seeds": sample_seeds(cfg4b),
                     "sampling_mode": cfg4b["generation"]["sampling_mode"],
                     "sampling_params": decoding_params(eval_cfg, cfg4b["generation"]["sampling_mode"]),
                     "greedy_params": decoding_params(eval_cfg, "greedy"), "steps": log,
                     "library_versions": library_versions(),
                     "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
            "rows": cands}


def main(argv=None):
    argparse.ArgumentParser(description="Generate rejected-text candidates for the DPO v2 pairs (Colab GPU).").parse_args(argv)
    cfg4b, cfg4, eval_cfg = load_phase4b_config(), load_phase4_config(), load_eval_config()
    out = REPO_ROOT / cfg4b["paths"]["candidates"]
    result = run(cfg4b, cfg4, eval_cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    n = sum(len(t) for d in result["rows"].values() for t in d.values())
    print(f"wrote {out}: {len(result['rows'])} refusal rows, {n} texts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
