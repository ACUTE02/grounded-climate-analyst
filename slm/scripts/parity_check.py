"""Parity check (the honesty check of Phase 5): all val prompts through the LIVE service, scored with the SAME scoring code as the
evaluator, compared with the saved evaluator file of the same adapter (outputs/eval_<parity_eval>_val.json).

    python scripts/parity_check.py                      # backend(s), models from configs/phase5.yaml (parity block)
    python scripts/parity_check.py --backends vllm --models sft_0.5b [--limit 5]

For each (backend, model) a service process is started (serving.app), the val items are sent one at a time, the outputs are written in
the evaluator's JSON format (outputs/eval_svc_<backend>_<run>_val.json, so the existing report code reads them) and compared with the
saved file: how many outputs are identical, answerable token F1, strict correct refusals, false refusals.
RULE, fixed in configs/phase5.yaml before any run: if strict correct refusals or false refusals move by more than 1 item, or the F1 moves
by more than 0.03, the verdict is STOP: the script says so, exits with code 2, and the benchmark refuses to run for that model.
A small difference is expected (fp16 kernels differ) and is reported, not hidden. Val only; the test split is never read here."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_config, load_eval_config, load_phase5_config
from climate_slm.data.sft import parse_example
from climate_slm.evaluation import evaluate, scoring
from serving.client import ServerProcess, ServiceClient, nvidia_smi_query, wait_for_gpu_release
from serving.prompting import build_messages


class ServiceBackend:
    """Backend object for evaluate.run_eval that asks the live service. The prompt of each item is split into (context, question) and
    sent as such; the service builds the messages itself, and they must equal the item's stored prompt (checked here too)."""

    def __init__(self, client: ServiceClient, entry: dict, cfg1: dict, health: dict | None = None):
        self.client, self.entry, self.cfg1, self.health = client, entry, cfg1, health or {}
        self.server_latency_ms: list[float] = []

    def generate(self, prompts: list[list[dict]]) -> list[dict]:
        out = []
        for messages in prompts:
            context, question, _ = parse_example({"prompt": messages, "completion": [{"role": "assistant", "content": ""}]})
            if build_messages(context, question, self.cfg1) != messages:
                raise ValueError("the item prompt is not what the service builds from (context, question)")
            r = self.client.answer(context, question, max_new_tokens=256)
            self.server_latency_ms.append(r["latency_ms"])
            out.append({"text": r["answer"].strip(), "n_new_tokens": r["n_new_tokens"],
                        "hit_max_new_tokens": r["n_new_tokens"] >= 256,        # the response has no flag: 256 new tokens = the limit was reached
                        "n_prompt_tokens": r["n_prompt_tokens"]})
        return out

    def completion_nll(self, prompts, completions):
        return []                                          # the service does not compute perplexity: left empty in the file

    def info(self) -> dict:
        h = self.health
        return {"model_id": self.entry["base_model_id"], "revision": None, "adapter": self.entry.get("adapter") or self.entry.get("adapter_id"), "dtype": "float16",
                "device": "service:" + str(h.get("backend")), "load_4bit": False, "quantization": None}


# ---- comparison with the saved evaluator file ---------------------------------------------------------------------------------
def metrics_of(ev: dict) -> dict:
    s = ev["summary"]
    return {"token_f1": s["answerable"]["token_f1"]["mean"], "strict_correct_refusals": s["refusal"]["correct_refusal_strict"]["count"],
            "false_refusals": s["answerable"]["false_refusal_strict"]["count"], "n_items": s["n_items"],
            "n_refusal": s["refusal"]["n"], "n_answerable": s["answerable"]["n"]}


def compare_items(saved: dict, svc: dict) -> dict:
    """Identical outputs (compared after strip) and the list of differing items with both texts."""
    s_by = {it["id"]: it for it in saved["items"]}
    diff, identical = [], 0
    for it in svc["items"]:
        o = s_by.get(it["id"])
        if o is not None and o["output"].strip() == it["output"].strip():
            identical += 1
        else:
            diff.append({"id": it["id"], "kind": it["kind"], "saved_output": None if o is None else o["output"], "service_output": it["output"],
                         "saved_f1": None if o is None else o["token_f1"], "service_f1": it["token_f1"],
                         "saved_refused": None if o is None else o["refusal_strict"], "service_refused": it["refusal_strict"]})
    n = len(svc["items"])
    return {"n_items": n, "identical": identical, "identical_percent": (100.0 * identical / n) if n else None, "different": diff,
            "missing_in_saved": [d["id"] for d in diff if d["saved_output"] is None]}


def parity_verdict(saved_m: dict, svc_m: dict, rule: dict) -> dict:
    """OK or STOP under the rule fixed in advance. Deltas are service minus saved."""
    d = {"token_f1": svc_m["token_f1"] - saved_m["token_f1"],
         "strict_correct_refusals": svc_m["strict_correct_refusals"] - saved_m["strict_correct_refusals"],
         "false_refusals": svc_m["false_refusals"] - saved_m["false_refusals"]}
    reasons = []
    if abs(d["strict_correct_refusals"]) > rule["max_refusal_item_change"]:
        reasons.append(f"strict correct refusals moved by {d['strict_correct_refusals']:+d} items (limit {rule['max_refusal_item_change']})")
    if abs(d["false_refusals"]) > rule["max_refusal_item_change"]:
        reasons.append(f"false refusals moved by {d['false_refusals']:+d} items (limit {rule['max_refusal_item_change']})")
    if abs(d["token_f1"]) > rule["max_f1_change"] + 1e-12:
        reasons.append(f"token F1 moved by {d['token_f1']:+.4f} (limit {rule['max_f1_change']})")
    if svc_m["n_items"] != saved_m["n_items"]:
        reasons.append(f"{svc_m['n_items']} items scored, the saved file has {saved_m['n_items']}")
    return {"ok": not reasons, "verdict": "OK" if not reasons else "STOP", "reasons": reasons, "deltas": d}


def run_parity(client: ServiceClient, backend: str, model_key: str, cfg5: dict, cfg1: dict, cfg_eval: dict, limit: int | None = None,
               out_dir: Path | None = None, health: dict | None = None) -> dict:
    """One (backend, model): run the val items through the service, write the evaluator-format file, compare with the saved file."""
    entry, rule = cfg5["models"][model_key], cfg5["parity"]
    run_name = f"{rule['run_name_prefix']}_{backend}_{entry['parity_eval']}"
    out_dir = out_dir or REPO_ROOT / cfg5["paths"]["outputs_dir"]
    svc_path = out_dir / f"eval_{run_name}_val.json"
    sb = ServiceBackend(client, entry, cfg1, health)
    t0 = time.perf_counter()
    svc = evaluate.run_eval(sb, cfg_eval, "val", run_name, limit=limit, out_path=svc_path)
    seconds = time.perf_counter() - t0
    saved_path = out_dir / f"eval_{entry['parity_eval']}_val.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if limit:                                                # a partial run is compared with the same first items only
        keep = {it["id"] for it in svc["items"]}
        saved = {**saved, "items": [it for it in saved["items"] if it["id"] in keep]}
        saved["summary"] = scoring.summarize(saved["items"], cfg_eval)
    sm, vm = metrics_of(saved), metrics_of(svc)
    items = compare_items(saved, svc)
    verdict = parity_verdict(sm, vm, rule)
    return {"backend": backend, "model": model_key, "run_name": run_name, "adapter": entry.get("adapter") or entry.get("adapter_id"), "saved_file": saved_path.name,
            "service_file": svc_path.name, "limit": limit, "seconds": round(seconds, 1), "saved": sm, "service": vm, "items": items,
            **verdict, "rule": {k: rule[k] for k in ("max_refusal_item_change", "max_f1_change")},
            "server_latency_ms_mean": (sum(sb.server_latency_ms) / len(sb.server_latency_ms)) if sb.server_latency_ms else None}


def summary_of(results: dict[str, dict], rule: dict) -> dict:
    full = {k: r for k, r in results.items()}
    return {"rule": {k: rule[k] for k in ("max_refusal_item_change", "max_f1_change")},
            "all_ok": bool(full) and all(r["ok"] and not r["limit"] for r in full.values()),
            "results": {k: {"ok": r["ok"], "verdict": r["verdict"], "reasons": r["reasons"], "limit": r["limit"],
                            "identical": r["items"]["identical"], "n_items": r["items"]["n_items"], "deltas": r["deltas"]}
                        for k, r in full.items()}}


def format_result(r: dict) -> str:
    s, v, d = r["saved"], r["service"], r["deltas"]
    lines = [f"[{r['verdict']}] {r['backend']} / {r['model']}: identical outputs {r['items']['identical']} of {r['items']['n_items']} "
             f"({r['items']['identical_percent']:.1f}%)",
             f"    token F1 {s['token_f1']:.4f} -> {v['token_f1']:.4f} ({d['token_f1']:+.4f}); strict correct refusals "
             f"{s['strict_correct_refusals']}/{s['n_refusal']} -> {v['strict_correct_refusals']}/{v['n_refusal']} ({d['strict_correct_refusals']:+d}); "
             f"false refusals {s['false_refusals']}/{s['n_answerable']} -> {v['false_refusals']}/{v['n_answerable']} ({d['false_refusals']:+d})"]
    lines += [f"    STOP because: {x}" for x in r["reasons"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    import httpx
    cfg5, cfg1, cfg_eval = load_phase5_config(), load_config(), load_eval_config()
    pc = cfg5["parity"]
    ap = argparse.ArgumentParser(description="Parity check of the live service against the saved evaluator val files.")
    ap.add_argument("--backends", default=",".join(pc["backends"]))
    ap.add_argument("--models", default=",".join(pc["models"]))
    ap.add_argument("--limit", type=int, default=None, help="only the first N val items (a smoke run; it can never open the benchmark gate)")
    ap.add_argument("--port", type=int, default=cfg5["service"]["port"])
    a = ap.parse_args(argv)
    backends, models = a.backends.split(","), a.models.split(",")
    res_dir = REPO_ROOT / cfg5["paths"]["results_dir"]
    res_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    gpu0 = nvidia_smi_query()
    baseline = gpu0[0]["used_mib"] if gpu0 else None
    for backend in backends:
        for model in models:
            if model not in cfg5["models"]:
                raise SystemExit(f"unknown model {model!r}")
            log = REPO_ROOT / cfg5["server"]["log_dir"] / f"server_parity_{backend}_{model}.log"
            srv = ServerProcess(backend, model, cfg5, a.port, log)
            print(f"=== parity {backend} / {model}: starting the service ({srv.command_text()})", flush=True)
            srv.start()
            try:
                with httpx.Client(base_url=srv.base_url, timeout=cfg5["bench"]["request_timeout_s"]) as http:
                    client = ServiceClient(http)
                    health, wall = srv.wait_healthy(lambda: ServiceClient(http, 5.0), cfg5["server"]["health_timeout_s"], cfg5["server"]["health_poll_s"])
                    print(f"    healthy after {wall:.0f} s; versions {health['versions']}", flush=True)
                    r = run_parity(client, backend, model, cfg5, cfg1, cfg_eval, a.limit, health=health)
                    r.update(startup_wall_s=round(wall, 1), startup_service_s=health.get("startup_seconds"), versions=health["versions"],
                             service_command=srv.command_text())
            finally:
                srv.stop(cfg5["server"]["stop_timeout_s"])
                released, used_end = wait_for_gpu_release(baseline)
                if not released:
                    print(f"    WARNING: GPU memory is still {used_end} MiB after stopping the service (before: {baseline} MiB)", flush=True)
            results[f"{backend}/{model}"] = r
            (res_dir / f"parity_{backend}_{model}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
            print(format_result(r), flush=True)
    summary = summary_of(results, pc)
    (REPO_ROOT / cfg5["paths"]["parity_summary"]).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if any(not r["ok"] for r in results.values()):
        print("\nSTOP: the parity rule was broken (see above). Do NOT run the benchmark. Tell the reviewer and send "
              "outputs/phase5/parity_*.json.", flush=True)
        return 2
    print("\nparity within the rule for every model checked" + ("" if not a.limit else " (PARTIAL run: --limit, it does not open the benchmark)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
