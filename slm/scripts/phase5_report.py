"""A short report with the Phase 5 tables, made ONLY from the saved result files (no number is typed by hand, no model is run):

    python scripts/phase5_report.py [--log-mlflow]

Reads outputs/phase5_smoke.json (cell P5-0), outputs/phase5/parity_<backend>_<model>.json and outputs/phase5/bench_<backend>_<model>.json.
Writes outputs/phase5_results.md and outputs/phase5_bench.csv. With --log-mlflow it also logs the saved results into the MLflow
experiment phase5_serving of outputs/mlruns (one run per benchmark setting and concurrency level, one per parity check); that needs
mlflow (it is NOT installed in the vLLM runtime on purpose: run this step where mlflow exists, for example on the laptop).
Where two numbers are too close to separate (the min..max ranges over the repeats overlap) the text says so."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_serving import METRICS, compare_note
from climate_slm.config import REPO_ROOT, load_phase5_config

BACKEND_ORDER = ("vllm", "hf")


def load_results(res_dir: Path) -> dict:
    out = {"parity": {}, "bench": {}}
    for kind in ("parity", "bench"):
        for p in sorted(Path(res_dir).glob(f"{kind}_*.json")):
            if p.name == "parity_summary.json":
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            out[kind][(d["backend"], d["model"])] = d
    return out


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def rng(stat: dict, spec: str) -> str:
    """median (min to max) of a metric over the repeats."""
    if stat["median"] is None:
        return "n/a"
    f = lambda x: format(x, spec)                                                       # noqa: E731
    return f"{f(stat['median'])} ({f(stat['min'])} to {f(stat['max'])})"


def parity_table(parity: dict) -> list[str]:
    rows = []
    for (b, m), r in sorted(parity.items()):
        s, v, d = r["saved"], r["service"], r["deltas"]
        rows.append([b, m, f"{r['items']['identical']}/{r['items']['n_items']} ({r['items']['identical_percent']:.1f}%)",
                     f"{s['token_f1']:.4f} to {v['token_f1']:.4f} ({d['token_f1']:+.4f})",
                     f"{s['strict_correct_refusals']}/{s['n_refusal']} to {v['strict_correct_refusals']}/{v['n_refusal']} ({d['strict_correct_refusals']:+d})",
                     f"{s['false_refusals']}/{s['n_answerable']} to {v['false_refusals']}/{v['n_answerable']} ({d['false_refusals']:+d})",
                     ("**OK**" if r["ok"] else "**STOP**") + ("" if not r.get("limit") else f" (partial, first {r['limit']} items)")])
    return table(["backend", "model", "identical outputs", "answerable token F1 (saved to service)",
                  "strict correct refusals (saved to service)", "false refusals (saved to service)", "rule"], rows)


def differing_lines(parity: dict, max_rows: int = 15) -> list[str]:
    lines = []
    for (b, m), r in sorted(parity.items()):
        diff = r["items"]["different"]
        if not diff:
            lines.append(f"- {b} / {m}: no differing output.")
            continue
        lines.append(f"- {b} / {m}: {len(diff)} differing outputs (full texts of both sides in `outputs/phase5/parity_{b}_{m}.json`):")
        rows = [[d["id"], d["kind"], f"{d['saved_f1']:.3f}" if d["saved_f1"] is not None else "n/a", f"{d['service_f1']:.3f}",
                 str(d["saved_refused"]), str(d["service_refused"])] for d in diff[:max_rows]]
        lines += [""] + table(["id", "kind", "F1 saved", "F1 service", "refused saved", "refused service"], rows) + [""]
        if len(diff) > max_rows:
            lines.append(f"  ... and {len(diff) - max_rows} more in the json.")
    return lines


def startup_table(bench: dict) -> list[str]:
    rows = []
    for (b, m), r in sorted(bench.items()):
        if r.get("status") != "ok":
            rows.append([b, m, "skipped", "", "", r.get("reason", "")])
            continue
        rows.append([b, m, f"{r['startup_wall_s']:.0f}", "n/a" if r.get("startup_service_s") is None else f"{r['startup_service_s']:.0f}",
                     "n/a" if r.get("gpu_used_after_load_mib") is None else str(r["gpu_used_after_load_mib"]), ""])
    return table(["backend", "model", "start-up, process start to healthy (s)", "model load reported by the service (s)",
                  "GPU memory in use after loading (MiB)", "note"], rows)


def bench_table(bench: dict, model: str) -> list[str]:
    rows = []
    for b in BACKEND_ORDER:
        r = bench.get((b, model))
        if not r or r.get("status") != "ok":
            continue
        for c, lv in sorted(r["levels"].items(), key=lambda kv: int(kv[0])):
            st = lv["stats"]
            rows.append([b, c, rng(st["latency_p50_ms"], ".0f"), rng(st["latency_p95_ms"], ".0f"), rng(st["tokens_per_s"], ".1f"),
                         rng(st["requests_per_s"], ".2f"), str(lv["gpu"]["peak_used_mib"]), str(lv["n_errors"])])
    return table(["backend", "concurrency", "p50 latency, ms: median (min to max)", "p95 latency, ms", "output tokens per second",
                  "requests per second", "GPU memory peak, MiB (whole GPU)", "failed requests"], rows) if rows else ["not measured"]


def comparison_notes(bench: dict, model: str) -> list[str]:
    a, b = bench.get(("vllm", model)), bench.get(("hf", model))
    if not (a and b and a.get("status") == "ok" and b.get("status") == "ok"):
        return ["- vLLM against hf: only one of the two backends was measured for this model, no comparison."]
    out = []
    for c in sorted(a["levels"], key=int):
        if c not in b["levels"]:
            continue
        for key, label, hib in (("latency_p50_ms", "p50 latency", False), ("tokens_per_s", "output tokens per second", True),
                                ("requests_per_s", "requests per second", True)):
            out.append(f"- concurrency {c}, {label}: " + compare_note("vLLM", a["levels"][c]["stats"][key], "hf", b["levels"][c]["stats"][key], hib))
    return out


def csv_rows(bench: dict) -> list[dict]:
    rows = []
    for (b, m), r in sorted(bench.items()):
        if r.get("status") != "ok":
            continue
        for c, lv in r["levels"].items():
            for i, run in enumerate(lv["runs"], 1):
                rows.append({"backend": b, "model": m, "concurrency": int(c), "repeat": i, **{k: run.get(k) for k in
                             ("n_requests", "n_errors", "wall_s", "output_tokens", *METRICS)}, "gpu_peak_mib": lv["gpu"]["peak_used_mib"]})
    return rows


def build_markdown(results: dict, smoke: dict | None, cfg5: dict) -> str:
    parity, bench = results["parity"], results["bench"]
    ok_runs = [r for r in bench.values() if r.get("status") == "ok"]
    first = ok_runs[0] if ok_runs else None
    L = ["# Phase 5: serving results (made by scripts/phase5_report.py from the saved files)", ""]
    L += ["## Setup", ""]
    if first:
        g = (first.get("gpu") or [{}])[0]
        L += [f"- GPU: {g.get('name', 'n/a')} ({g.get('total_mib', 'n/a')} MiB)",
              f"- library versions seen by the service: {json.dumps(first['versions'])}",
              f"- prompts: the {first['n_prompts']} val prompts{' (PARTIAL: smoke run)' if first.get('partial_prompts') else ''}; test split not used; "
              f"max_new_tokens {first['max_new_tokens']}; greedy decoding, repetition_penalty 1.0; {first['warmup_requests']} warm-up requests per start excluded; "
              f"concurrency {first['concurrency_levels']}; {first['repeats']} repeats of every setting",
              f"- micro-batching in both backends (requests that arrive together are one engine call; later arrivals wait for the next batch): "
              f"window {first['batching_config']['window_ms']} ms, max batch {first['batching_config']['max_batch']}",
              f"- client command: `{first['client_command']}`"]
        L += [f"- service command ({b}/{m}): `{r['service_command']}`" for (b, m), r in sorted(bench.items()) if r.get("status") == "ok"][:6]
    else:
        L += ["- no benchmark result file found."]
    if smoke:
        L += [f"- P5-0 install check: {'passed' if smoke.get('ok') else 'FAILED'} with versions {json.dumps(smoke.get('versions_before'))}"]
    L += ["", "## Parity: the live service against the saved evaluator val files", "",
          f"Rule fixed in advance: STOP if strict correct refusals or false refusals move by more than {cfg5['parity']['max_refusal_item_change']} item "
          f"or the answerable token F1 by more than {cfg5['parity']['max_f1_change']}. Scored with the evaluator's own scoring code.", ""]
    L += (parity_table(parity) if parity else ["not measured"]) + [""] + differing_lines(parity) + [""]
    L += ["## Start-up time (separate from the request numbers)", ""] + (startup_table(bench) if bench else ["not measured"]) + [""]
    for model in cfg5["bench"]["models"]:
        L += [f"## Benchmark: {cfg5['models'][model]['label']} (`{model}`)", "", *bench_table(bench, model), "", *comparison_notes(bench, model), ""]
    skipped = [(k, r) for k, r in sorted(bench.items()) if r.get("status") != "ok"]
    if skipped:
        L += ["## Settings that were not measured", ""] + [f"- {b} / {m}: {r.get('reason')}" for (b, m), r in skipped] + [""]
    L += ["## How to read the numbers", "",
          "- Latency is measured by the client for each request (queueing included). With concurrency 4 or 16 a request may wait for the batch that is "
          "running; both backends work this way, so the comparison is fair, but it is not what an engine with continuous batching could do.",
          "- Output tokens per second = all generated tokens of a repeat / its wall time. vLLM may count the end-of-turn token, the hf backend does not "
          "(a difference of one token per answer).",
          "- GPU memory is the used memory of the whole GPU seen by nvidia-smi. vLLM reserves a memory pool when it starts (`gpu_memory_utilization`), so its "
          "number is a reservation and not the memory the model needs; the hf number grows with the batch.",
          "- Min / median / max are over the repeats. Where the ranges of two settings overlap the text says they are too close to separate.", ""]
    return "\n".join(L)


# ---- MLflow (logged from the saved files; fail-safe) -------------------------------------------------------------------------------
def log_to_mlflow(cfg5: dict, results: dict, smoke: dict | None = None) -> int:
    """One run per benchmark setting and concurrency level, one per parity check. Returns the number of runs started.
    MLflow errors are warnings (mlflow_log never raises)."""
    from climate_slm.training import mlflow_log
    n = 0
    for (b, m), r in sorted(results["parity"].items()):
        params = {"kind": "parity", "backend": b, "model": m, "adapter": r["adapter"], "saved_file": r["saved_file"], "n_items": r["items"]["n_items"],
                  "rule_max_refusal_item_change": r["rule"]["max_refusal_item_change"], "rule_max_f1_change": r["rule"]["max_f1_change"],
                  **{f"lib_{k}": v for k, v in (r.get("versions") or {}).items()}}
        t = mlflow_log.start_tracker(cfg5, f"parity_{b}_{m}", params, {"phase": "5", "kind": "parity", "verdict": r["verdict"]})
        t.log_metrics({"identical_outputs": r["items"]["identical"], "identical_percent": r["items"]["identical_percent"],
                       "delta_token_f1": r["deltas"]["token_f1"], "delta_strict_correct_refusals": r["deltas"]["strict_correct_refusals"],
                       "delta_false_refusals": r["deltas"]["false_refusals"], "parity_ok": 1.0 if r["ok"] else 0.0})
        t.end()
        n += 1
    for (b, m), r in sorted(results["bench"].items()):
        if r.get("status") != "ok":
            continue
        for c, lv in sorted(r["levels"].items(), key=lambda kv: int(kv[0])):
            params = {"kind": "benchmark", "backend": b, "model": m, "concurrency": c, "repeats": r["repeats"], "n_prompts": r["n_prompts"],
                      "max_new_tokens": r["max_new_tokens"], "warmup_requests": r["warmup_requests"], "split": r["split"],
                      "gpu": (r.get("gpu") or [{}])[0].get("name"), "partial_prompts": r["partial_prompts"],
                      "service_command": r["service_command"], **{f"lib_{k}": v for k, v in r["versions"].items()}}
            t = mlflow_log.start_tracker(cfg5, f"bench_{b}_{m}_c{c}", params, {"phase": "5", "kind": "benchmark", "backend": b, "model": m})
            for i, run in enumerate(lv["runs"], 1):
                t.log_metrics({k: run[k] for k in METRICS if run.get(k) is not None}, step=i)
            summary = {}
            for k, st in lv["stats"].items():
                for q in ("min", "median", "max"):
                    if st[q] is not None:
                        summary[f"{k}_{q}"] = st[q]
            summary.update(gpu_peak_mib=lv["gpu"]["peak_used_mib"], failed_requests=lv["n_errors"], startup_wall_s=r["startup_wall_s"])
            t.log_metrics({k: v for k, v in summary.items() if v is not None})
            t.end()
            n += 1
    return n


def main(argv=None) -> int:
    cfg5 = load_phase5_config()
    ap = argparse.ArgumentParser(description="Phase 5 tables from the saved result files.")
    ap.add_argument("--log-mlflow", action="store_true", help="also log the saved results into the MLflow experiment phase5_serving")
    a = ap.parse_args(argv)
    results = load_results(REPO_ROOT / cfg5["paths"]["results_dir"])
    sp = REPO_ROOT / cfg5["paths"]["outputs_dir"] / "phase5_smoke.json"
    smoke = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None
    md = REPO_ROOT / cfg5["paths"]["results_md"]
    md.write_text(build_markdown(results, smoke, cfg5), encoding="utf-8")
    rows = csv_rows(results["bench"])
    cp = REPO_ROOT / cfg5["paths"]["results_csv"]
    if rows:
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("wrote", md, "and" if rows else "(no benchmark rows, no csv)", cp if rows else "")
    if a.log_mlflow:
        print(f"MLflow: {log_to_mlflow(cfg5, results, smoke)} runs started in experiment {cfg5['mlflow']['experiment_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
