"""Cell P5-0: does vLLM install and run on the Colab T4, and does it serve the LoRA adapter? A few val prompts, nothing else.

    python scripts/vllm_smoke.py [--n 3] [--model sft_0.5b] [--out outputs/phase5_smoke.json]

Run it in a FRESH runtime (restart first): vLLM brings its own torch and may break the training environment. The script prints
each step as PASS or FAIL, the versions pip chose, and compares the outputs with the saved evaluator outputs of the same adapter
(outputs/eval_<parity_eval>_val.json): a first, small sign for the parity check of stage 2, not the parity check itself.
Prompts are built with serving/prompting.build_messages (the one shared function) and the chat template of the model's tokenizer
through climate_slm/chat.py, greedy decoding with repetition_penalty 1.0 and at most 256 new tokens. If a step fails, the report
says so and the hf backend is used instead; then every sentence about vLLM is removed from the results.
The laptop never runs this script (no vllm, no torch there); only compare_to_saved / first_val_prompts are tested there."""
import argparse
import datetime
import json
import platform
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm import chat
from climate_slm.config import REPO_ROOT, load_config, load_eval_config, load_phase5_config
from climate_slm.data.sft import parse_example
from climate_slm.evaluation import metrics as M
from climate_slm.evaluation.scoring import load_items
from serving.prompting import build_messages

PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "peft", "xformers", "flashinfer-python", "triton", "fastapi", "pydantic")


def versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def first_val_prompts(n: int, cfg_eval: dict | None = None, cfg1: dict | None = None) -> list[dict]:
    """The first n val items with the messages built by the shared function (checked equal to the stored prompt)."""
    cfg_eval = cfg_eval or load_eval_config()
    items = load_items(REPO_ROOT / cfg_eval["paths"]["splits"]["val"])[:n]
    out = []
    for it in items:
        context, question, reference = parse_example(it)
        messages = build_messages(context, question, cfg1)
        if messages != it["prompt"]:
            raise ValueError(f"build_messages differs from the stored prompt of {it['meta']['id']}")
        out.append({"id": it["meta"]["id"], "messages": messages, "reference": reference, "kind": "refusal" if it["meta"]["type"] == "unanswerable" else "answerable"})
    return out


def compare_to_saved(rows: list[dict], outputs: list[str], saved_items: list[dict], refusal_text: str) -> dict:
    """Per prompt: identical to the saved evaluator output, strict refusal flags of both. Summary: how many are identical."""
    by_id = {it["id"]: it for it in saved_items}
    per = []
    for r, out in zip(rows, outputs):
        saved = by_id.get(r["id"])
        per.append({"id": r["id"], "kind": r["kind"], "identical": saved is not None and out.strip() == saved["output"].strip(),
                    "refused_now": M.is_refusal_strict(out, refusal_text),
                    "refused_saved": None if saved is None else M.is_refusal_strict(saved["output"], refusal_text),
                    "saved_found": saved is not None})
    return {"n": len(per), "identical": sum(p["identical"] for p in per), "per_item": per}


def step(report: dict, name: str, fn):
    """Run one step; print PASS / FAIL; keep going only if it passed."""
    t0 = time.perf_counter()
    try:
        result = fn()
        report["steps"].append({"step": name, "status": "PASS", "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[PASS] {name} ({time.perf_counter() - t0:.1f} s)", flush=True)
        return result
    except BaseException as err:                                  # noqa: BLE001  (report every failure, then stop)
        tb = traceback.format_exc()
        report["steps"].append({"step": name, "status": "FAIL", "error": f"{type(err).__name__}: {err}", "traceback_tail": tb[-1500:]})
        print(f"[FAIL] {name}: {type(err).__name__}: {err}\n{tb[-1500:]}", flush=True)
        return None


def main(argv=None) -> int:
    cfg5, cfg1, cfg_eval = load_phase5_config(), load_config(), load_eval_config()
    ap = argparse.ArgumentParser(description="vLLM smoke check on a few val prompts (Colab).")
    ap.add_argument("--n", type=int, default=cfg5["smoke"]["n_prompts"])
    ap.add_argument("--model", default=cfg5["smoke"]["model"], choices=sorted(cfg5["models"]))
    ap.add_argument("--out", default="outputs/phase5_smoke.json")
    a = ap.parse_args(argv)
    entry, v = cfg5["models"][a.model], cfg5["vllm"]
    if entry.get("adapter_id"):                           # the published adapter: downloaded by its Hugging Face id
        from huggingface_hub import snapshot_download
        adapter = snapshot_download(repo_id=entry["adapter_id"], revision=entry.get("adapter_revision"))
    else:
        adapter = REPO_ROOT / entry["adapter"]
    report = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "model": a.model,
              "adapter": entry.get("adapter") or entry.get("adapter_id"), "base_model_id": entry["base_model_id"], "vllm_settings": v, "n_prompts": a.n,
              "versions_before": versions(), "steps": []}
    print("versions chosen by pip:", json.dumps(report["versions_before"]), flush=True)
    rows = step(report, "build val prompts with the shared function", lambda: first_val_prompts(a.n, cfg_eval, cfg1))
    if rows is None or not (adapter / "adapter_config.json").exists():
        if rows is not None:
            print(f"[FAIL] adapter not found: {adapter} (unzip phase4_start.zip into outputs/ first)")
            report["steps"].append({"step": "adapter present", "status": "FAIL", "error": str(adapter)})
        return finish(report, a.out, 1)
    mod = step(report, "import vllm", lambda: __import__("vllm"))
    if mod is None:
        return finish(report, a.out, 1)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = step(report, "load the base model in vLLM (dtype half, LoRA enabled)",
               lambda: LLM(model=entry["base_model_id"], dtype=v["dtype"], max_model_len=cfg5["service"]["max_model_len"],
                           gpu_memory_utilization=v["gpu_memory_utilization"], enable_lora=v["enable_lora"], max_lora_rank=v["max_lora_rank"]))
    if llm is None:
        return finish(report, a.out, 1)
    tok = llm.get_tokenizer()
    template = lambda msgs, gen: tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)     # noqa: E731
    prompts = [chat.prompt_text(template, r["messages"]) for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg5["service"]["max_new_tokens"], repetition_penalty=1.0)
    outs = step(report, "generate with the LoRA adapter (greedy)",
                lambda: llm.generate(prompts, sp, lora_request=LoRARequest(a.model, 1, str(adapter))))
    if outs is None:
        return finish(report, a.out, 1)
    texts = [o.outputs[0].text.strip() for o in outs]
    saved_path = REPO_ROOT / "outputs" / f"eval_{entry['parity_eval']}_val.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))["items"] if saved_path.exists() else []
    cmp_ = compare_to_saved(rows, texts, saved, cfg1["refusal_text"])
    report.update(outputs=[{"id": r["id"], "kind": r["kind"], "output": t, "n_prompt_tokens": len(o.prompt_token_ids),
                            "n_new_tokens": len(o.outputs[0].token_ids)} for r, t, o in zip(rows, texts, outs)],
                  compare_to_saved=cmp_, saved_file_found=saved_path.exists(), versions_after=versions())
    for r, t, p in zip(rows, texts, cmp_["per_item"]):
        print(f"--- {r['id']} ({r['kind']}) identical to the saved evaluator output: {p['identical']} | refused now: {p['refused_now']}"
              f" | refused in the saved output: {p['refused_saved']}\n    {t[:300]}")
    print(f"identical to the saved outputs: {cmp_['identical']} of {cmp_['n']}" + ("" if saved else " (saved eval file not found)"))
    return finish(report, a.out, 0)


def finish(report: dict, out: str, code: int) -> int:
    report["ok"] = code == 0 and all(s["status"] == "PASS" for s in report["steps"])
    p = REPO_ROOT / out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("RESULT:", "vLLM works on this machine for this adapter" if report["ok"] else
          "vLLM did NOT work: use the hf backend and remove every sentence about vLLM from the results", "| wrote", p, flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
