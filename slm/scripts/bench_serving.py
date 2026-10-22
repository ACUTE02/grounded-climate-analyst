"""Benchmark of the service (a CLIENT; it is not part of the service). Val prompts only (the test split is never read), the same
prompts for every setting, warm-up requests excluded, max_new_tokens 256.

    python scripts/bench_serving.py                         # all backends, models, concurrency levels of configs/phase5.yaml (bench block)
    python scripts/bench_serving.py --backends vllm --models sft_0.5b [--repeats 1] [--n-prompts 8]   # a smaller run

For every (backend, model) a service process is started (python -m serving.app), its start-up time is measured and kept SEPARATE from
the request numbers, a few warm-up requests are sent, and then for concurrency 1, 4 and 16 the 53 val prompts are sent `repeats` times
(3): per repeat p50 / p95 request latency (client side), output tokens per second, requests per second; over the repeats min / median /
max are reported (never one run). The used GPU memory is sampled with nvidia-smi (whole GPU: for vLLM this is mostly the memory pool it
reserves on start, not what the model needs). Hardware, library versions and the exact commands are written into the result file.
GATE: the vLLM benchmark of a model only runs if the parity check for it passed on all 53 val items (outputs/phase5/parity_summary.json).
Results: outputs/phase5/bench_<backend>_<model>.json (written after every setting, so a disconnect loses at most one setting; a finished
setting is skipped unless --force). The tables come from scripts/phase5_report.py, never typed by hand."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config, load_phase5_config
from climate_slm.data.sft import parse_example
from climate_slm.evaluation.scoring import load_items
from serving.client import GpuSampler, ServerProcess, ServiceClient, nvidia_smi_query, wait_for_gpu_release

METRICS = ("latency_p50_ms", "latency_p95_ms", "tokens_per_s", "requests_per_s")


# ---- pure summary functions (tested on fake timings) ---------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float:
    """q-th percentile (0..100) with linear interpolation between the closest ranks."""
    if not values:
        raise ValueError("percentile of an empty list")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summarize_run(records: list[dict], wall_s: float) -> dict:
    """One repeat at one concurrency level. records: {latency_s, n_new_tokens, error}. Latency stats use the successful requests only."""
    ok = [r for r in records if not r.get("error")]
    lat = [r["latency_s"] * 1000.0 for r in ok]
    tokens = sum(r["n_new_tokens"] for r in ok)
    return {"n_requests": len(records), "n_errors": len(records) - len(ok), "wall_s": wall_s, "output_tokens": tokens,
            "latency_p50_ms": percentile(lat, 50) if lat else None, "latency_p95_ms": percentile(lat, 95) if lat else None,
            "latency_mean_ms": (sum(lat) / len(lat)) if lat else None,
            "tokens_per_s": tokens / wall_s if wall_s > 0 else None, "requests_per_s": len(ok) / wall_s if wall_s > 0 else None}


def aggregate_repeats(runs: list[dict], keys=METRICS) -> dict:
    """min / median / max over the repeats for every metric (None values are left out)."""
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[k] = ({"min": min(vals), "median": statistics.median(vals), "max": max(vals), "n": len(vals)} if vals
                  else {"min": None, "median": None, "max": None, "n": 0})
    return out


def ranges_overlap(a: dict, b: dict) -> bool:
    """Do the min..max ranges over the repeats of two settings overlap? Then the two numbers cannot be separated."""
    if None in (a["min"], a["max"], b["min"], b["max"]):
        return True
    return not (a["max"] < b["min"] or b["max"] < a["min"])


def compare_note(label_a: str, a: dict, label_b: str, b: dict, higher_is_better: bool, unit: str = "") -> str:
    """Plain words: is one setting clearly better, or are they too close to separate (overlapping ranges over the repeats)?"""
    rng = lambda s: f"{s['min']:.4g} to {s['max']:.4g}{unit}"                          # noqa: E731
    if ranges_overlap(a, b):
        return f"{label_a} ({rng(a)}) and {label_b} ({rng(b)}): too close to separate (the ranges over the repeats overlap)"
    better_a = (a["median"] > b["median"]) == higher_is_better
    return f"{label_a if better_a else label_b} is clearly better: {label_a} {rng(a)}, {label_b} {rng(b)} (no overlap over the repeats)"


# ---- load generation ---------------------------------------------------------------------------------------------------------
def run_load(post_fn, items: list[tuple[str, str]], concurrency: int) -> tuple[list[dict], float]:
    """Send every (context, question) once with `concurrency` requests in flight; returns (records, wall seconds).
    post_fn(context, question) -> response dict with n_new_tokens. A failed request is a record with an error, not an exception."""
    def one(item):
        t0 = time.perf_counter()
        try:
            r = post_fn(*item)
            return {"latency_s": time.perf_counter() - t0, "n_new_tokens": r["n_new_tokens"], "server_latency_ms": r.get("latency_ms"),
                    "error": None}
        except Exception as err:                                                      # noqa: BLE001  (counted, never hidden)
            return {"latency_s": time.perf_counter() - t0, "n_new_tokens": 0, "server_latency_ms": None, "error": f"{type(err).__name__}: {str(err)[:120]}"}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(one, items))
    return records, time.perf_counter() - t0


def run_setting(post_fn, items, concurrency_levels: list[int], repeats: int, warmup: int, gpu_query=nvidia_smi_query,
                gpu_interval_s: float = 0.5) -> dict:
    """Warm-up, then for each concurrency level `repeats` repeats; GPU memory sampled per level."""
    for it in items[:warmup]:
        post_fn(*it)                                                                      # excluded from every number
    per_level = {}
    for c in concurrency_levels:
        sampler = GpuSampler(gpu_interval_s, gpu_query).start()
        runs = []
        for _ in range(repeats):
            records, wall = run_load(post_fn, items, c)
            runs.append(summarize_run(records, wall))
        per_level[str(c)] = {"concurrency": c, "runs": runs, "stats": aggregate_repeats(runs), "gpu": sampler.stop(),
                             "n_errors": sum(r["n_errors"] for r in runs)}
    return per_level


def val_items(cfg_eval: dict, n: int | None = None) -> list[tuple[str, str]]:
    """(context, question) of the val items. Val only."""
    rows = load_items(REPO_ROOT / cfg_eval["paths"]["splits"]["val"])
    rows = rows[:n] if n else rows
    return [parse_example(r)[:2] for r in rows]


# ---- gate and preflight ---------------------------------------------------------------------------------------------------------
def parity_gate(summary: dict | None, backend: str, model: str) -> tuple[bool, str]:
    """The vLLM benchmark of a model needs a passed, complete parity check. The hf backend is the evaluator's own code: no gate."""
    if backend != "vllm":
        return True, "hf backend: the evaluator's own code, no parity gate"
    if summary is None:
        return False, "no parity check found (outputs/phase5/parity_summary.json): run scripts/parity_check.py first"
    r = summary.get("results", {}).get(f"{backend}/{model}")
    if r is None:
        return False, f"no parity result for {backend}/{model}"
    if r.get("limit"):
        return False, f"the parity result for {backend}/{model} is a partial run (--limit)"
    if not r["ok"]:
        return False, f"parity check for {backend}/{model} said STOP: {'; '.join(r['reasons'])}"
    return True, "parity passed"


def hf_available(python: str = sys.executable) -> tuple[bool, str]:
    """Can the hf backend run in this environment? (transformers and peft importable in a subprocess.)"""
    r = subprocess.run([python, "-c", "import transformers, peft, torch; print(transformers.__version__, peft.__version__, torch.__version__)"],
                       capture_output=True, text=True)
    return (r.returncode == 0, r.stdout.strip() if r.returncode == 0 else (r.stderr.strip().splitlines() or ["import failed"])[-1])


# ---- main ---------------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    import httpx
    cfg5, cfg_eval = load_phase5_config(), load_eval_config()
    bc = cfg5["bench"]
    ap = argparse.ArgumentParser(description="Benchmark the service (client).")
    ap.add_argument("--backends", default=",".join(bc["backends"]))
    ap.add_argument("--models", default=",".join(bc["models"]))
    ap.add_argument("--concurrency", default=",".join(str(c) for c in bc["concurrency"]))
    ap.add_argument("--repeats", type=int, default=bc["repeats"])
    ap.add_argument("--n-prompts", type=int, default=None, help="smoke: only the first N val prompts (flagged partial in the result)")
    ap.add_argument("--force", action="store_true", help="run a setting again even if its result file is complete")
    ap.add_argument("--port", type=int, default=cfg5["service"]["port"])
    a = ap.parse_args(argv)
    levels = [int(x) for x in a.concurrency.split(",")]
    items = val_items(cfg_eval, a.n_prompts)
    res_dir = REPO_ROOT / cfg5["paths"]["results_dir"]
    res_dir.mkdir(parents=True, exist_ok=True)
    ps = REPO_ROOT / cfg5["paths"]["parity_summary"]
    parity = json.loads(ps.read_text(encoding="utf-8")) if ps.exists() else None
    gpu = nvidia_smi_query()
    baseline = gpu[0]["used_mib"] if gpu else None                                       # GPU memory in use before any service
    print("GPU:", gpu[0]["name"] if gpu else "not found (nvidia-smi)", "| client command:", " ".join(sys.argv), flush=True)
    done, skipped = [], []
    for backend in a.backends.split(","):
        for model in a.models.split(","):
            out = res_dir / f"bench_{backend}_{model}.json"
            if out.exists() and not a.force and json.loads(out.read_text(encoding="utf-8")).get("status") == "ok":
                print(f"[{backend}/{model}] result exists, skipped (--force to repeat)")
                continue
            ok, why = parity_gate(parity, backend, model)
            result = {"backend": backend, "model": model, "split": "val", "n_prompts": len(items), "partial_prompts": bool(a.n_prompts),
                      "concurrency_levels": levels, "repeats": a.repeats, "warmup_requests": bc["warmup_requests"],
                      "max_new_tokens": bc["max_new_tokens"], "gpu": gpu, "client_command": " ".join(sys.argv),
                      "batching_config": cfg5["batching"], "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}
            if not ok:
                print(f"[{backend}/{model}] SKIPPED: {why}", flush=True)
                out.write_text(json.dumps({**result, "status": "skipped", "reason": why}, indent=1), encoding="utf-8")
                skipped.append((backend, model, why))
                continue
            if backend == "hf":
                avail, info = hf_available()
                if not avail:
                    why = f"the hf backend cannot run in this environment: {info}"
                    print(f"[{backend}/{model}] SKIPPED: {why}", flush=True)
                    out.write_text(json.dumps({**result, "status": "skipped", "reason": why}, indent=1), encoding="utf-8")
                    skipped.append((backend, model, why))
                    continue
            srv = ServerProcess(backend, model, cfg5, a.port, REPO_ROOT / cfg5["server"]["log_dir"] / f"server_bench_{backend}_{model}.log")
            print(f"=== {backend} / {model}: starting the service: {srv.command_text()}", flush=True)
            srv.start()
            try:
                with httpx.Client(base_url=srv.base_url, timeout=bc["request_timeout_s"]) as http:
                    client = ServiceClient(http, bc["request_timeout_s"])
                    health, wall = srv.wait_healthy(lambda: ServiceClient(http, 5.0), cfg5["server"]["health_timeout_s"],
                                                    cfg5["server"]["health_poll_s"])
                    idle = nvidia_smi_query()
                    print(f"    start-up {wall:.0f} s (service reports loading {health.get('startup_seconds')}); GPU in use after loading "
                          f"{idle[0]['used_mib'] if idle else 'n/a'} MiB", flush=True)
                    post = lambda c, q: client.answer(c, q, max_new_tokens=bc["max_new_tokens"])                   # noqa: E731
                    levels_out = run_setting(post, items, levels, a.repeats, bc["warmup_requests"], nvidia_smi_query, bc["gpu_sample_interval_s"])
                    health_after = client.health()
            finally:
                srv.stop(cfg5["server"]["stop_timeout_s"])
                released, used_end = wait_for_gpu_release(baseline)
                if not released:
                    print(f"    WARNING: GPU memory is still {used_end} MiB after stopping the service (before: {baseline} MiB); the next "
                          "setting may start with less free memory", flush=True)
            result.update(status="ok", gpu_released_after_stop=released, startup_wall_s=round(wall, 1), startup_service_s=health.get("startup_seconds"),
                          versions=health["versions"], gpu_used_after_load_mib=idle[0]["used_mib"] if idle else None,
                          service_command=srv.command_text(), batching_stats=health_after.get("batching"), levels=levels_out)
            out.write_text(json.dumps(result, indent=1), encoding="utf-8")
            done.append((backend, model))
            f = lambda x, spec: "n/a" if x is None else format(x, spec)                                              # noqa: E731
            for c, lv in levels_out.items():
                st = lv["stats"]
                print(f"    concurrency {c}: p50 {f(st['latency_p50_ms']['median'], '.0f')} ms, p95 {f(st['latency_p95_ms']['median'], '.0f')} ms, "
                      f"{f(st['tokens_per_s']['median'], '.1f')} tok/s, {f(st['requests_per_s']['median'], '.2f')} req/s (medians of {a.repeats}); "
                      f"errors {lv['n_errors']}; GPU peak {lv['gpu']['peak_used_mib']} MiB", flush=True)
    print(f"\nfinished settings: {done}; skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
