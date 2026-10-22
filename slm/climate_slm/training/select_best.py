"""Phase 3 selection. Reads ONLY saved files (eval JSON of the evaluator, train_log.json of the runs).

    python -m climate_slm.training.select_best               # write outputs/phase3_selection.md and .json
    python -m climate_slm.training.select_best --freeze      # also store the sha256 of the selection file
    python -m climate_slm.training.select_best --final-tables  # after the test runs: outputs/phase3_final.md

The rule (fixed in configs/phase3.yaml before any run), applied on VAL only:
  1. keep candidates with false refusals <= selection.max_false_refusals_val (4 of 42)
  2. highest strict correct refusals   3. higher answerable token F1   4. higher number faithfulness
  5. cheaper (lower peak GPU memory, then fewer trainable parameters, then shorter training time)
  6. last resort: the earlier epoch.
Once the selection file is frozen it is not rewritten; the test split is evaluated only for frozen winners."""
import argparse
import json
import math
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config
from climate_slm.evaluation.evaluate import default_out
from climate_slm.training import grid, selection_io, summary

BASELINE_SPLITS = ("val", "test")


# ---- loading -----------------------------------------------------------------------------------------------------
def load_metrics(eval_cfg: dict, cfg3: dict, name: str, split: str) -> dict | None:
    ev = summary.load_eval(default_out(eval_cfg, name, split), eval_cfg)
    if ev is None:
        return None
    c = cfg3["criteria"]
    return summary.extract_metrics(ev, c["loop_ngram"], c["loop_min_repeats"])


def cost_of(cfg3: dict, run_name: str) -> dict | None:
    p = grid.run_log_path(cfg3, run_name)
    if not p.exists():
        return None
    log = json.loads(p.read_text(encoding="utf-8"))
    eps = log.get("epochs") or []
    return {"complete": bool(log.get("complete")), "peak_gpu_mem_mb": log.get("peak_gpu_mem_mb"),
            "train_seconds": log.get("train_seconds"), "total_seconds": log.get("total_seconds"),
            "trainable_params": log.get("trainable_params"), "total_params": log.get("total_params"),
            "trainable_share": log.get("trainable_share"), "adapter_mb": eps[-1]["adapter_mb"] if eps else None,
            "val_loss": {str(e["epoch"]): e["val_loss"] for e in eps}, "skipped_steps": log.get("skipped_steps"),
            "dropped_train_ids": log.get("dropped_train_ids"), "dropped_val_ids": log.get("dropped_val_ids"),
            "gradient_checkpointing": (log.get("params") or {}).get("gradient_checkpointing"),
            "model_id": log.get("model_id")}


def run_candidates(cfg3: dict, eval_cfg: dict, run: dict, split: str = "val") -> list[dict]:
    """Candidates of one run: every epoch evaluated on the fp16 base, and for a QLoRA run also every epoch evaluated
    on the 4-bit base it was trained on (extra candidates, name suffix __on4bit, same rule)."""
    cost = cost_of(cfg3, run["name"])
    quantized = cfg3["methods"][run["method"]]["quantize"] is not None
    out = []
    for on_4bit in ((False, True) if quantized else (False,)):
        for e in grid.epochs(cfg3):
            name = grid.candidate_name(run["name"], e, on_4bit)
            m = load_metrics(eval_cfg, cfg3, name, split)
            if m is not None:
                out.append({"candidate": name, "run": run["name"], "epoch": e, "method": run["method"], "lr": run["lr"],
                            "seed": run["seed"], "arm": run["arm"], "eval_base": "4bit" if on_4bit else "fp16",
                            "metrics": m, "cost": cost, "val_loss": (cost or {}).get("val_loss", {}).get(str(e))})
    return out


# ---- the rule ----------------------------------------------------------------------------------------------------
def rank_key(c: dict, sel_cfg: dict) -> tuple[tuple, dict]:
    """(sort key, readable parts). Smaller key = better."""
    d, m, cost = sel_cfg["round_decimals"], c["metrics"], c["cost"] or {}
    nf = m["number_faithfulness"]["mean"]
    parts = {"correct_refusals_strict": -m["correct_strict"]["count"],
             "token_f1": -round(m["token_f1"]["mean"], d),
             "number_faithfulness": -round(nf if nf is not None else -1.0, d),
             "peak_gpu_mem": cost.get("peak_gpu_mem_mb") if cost.get("peak_gpu_mem_mb") is not None else math.inf,
             "trainable_params": cost.get("trainable_params") if cost.get("trainable_params") is not None else math.inf,
             "train_seconds": cost.get("train_seconds") if cost.get("train_seconds") is not None else math.inf,
             "epoch": c["epoch"]}
    return tuple(parts[k] for k in sel_cfg["order"]), parts


def _ci_text(ci) -> str:
    return "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def closeness(a: dict, b: dict) -> list[str]:
    """Plain-words notes: is the winner's lead over another candidate inside the 95% intervals?"""
    lines = []
    ma, mb = a["metrics"], b["metrics"]
    ca, cb = ma["correct_strict"], mb["correct_strict"]
    cmp_ = summary.compare(ca["rate"], ca["wilson95"], cb["rate"], cb["wilson95"])
    if cmp_["direction"] and cmp_["direction"] != "equal":
        verdict = ("the intervals overlap, so this difference is NOT established" if cmp_["overlap"]
                   else "the intervals do not overlap")
        lines.append(f"strict correct refusals: {a['candidate']} {ca['count']}/{ca['n']} against {b['candidate']} "
                     f"{cb['count']}/{cb['n']} (95% Wilson {_ci_text(ca['wilson95'])} against {_ci_text(cb['wilson95'])}): {verdict}")
    elif cmp_["direction"] == "equal":
        lines.append(f"strict correct refusals: equal ({ca['count']}/{ca['n']}) for {a['candidate']} and {b['candidate']}")
    fa, fb = ma["token_f1"], mb["token_f1"]
    cmp_ = summary.compare(fa["mean"], fa["ci95"], fb["mean"], fb["ci95"])
    if cmp_["direction"] and cmp_["direction"] != "equal":
        verdict = ("the intervals overlap, so this difference is NOT established" if cmp_["overlap"]
                   else "the intervals do not overlap")
        lines.append(f"answerable token F1: {a['candidate']} {fa['mean']:.3f} {_ci_text(fa['ci95'])} against "
                     f"{b['candidate']} {fb['mean']:.3f} {_ci_text(fb['ci95'])}: {verdict}")
    return lines


def select_winner(cands: list[dict], cfg3: dict) -> dict:
    """Apply the rule to a list of candidates (all on val)."""
    sel = cfg3["selection"]
    eligible = [c for c in cands if c["metrics"]["false_refusals"]["count"] <= sel["max_false_refusals_val"]]
    excluded = [{"candidate": c["candidate"], "false_refusals": c["metrics"]["false_refusals"]["count"],
                 "n": c["metrics"]["false_refusals"]["n"]}
                for c in cands if c["metrics"]["false_refusals"]["count"] > sel["max_false_refusals_val"]]
    ranked = sorted(eligible, key=lambda c: rank_key(c, sel)[0])
    out = {"n_candidates": len(cands), "excluded": excluded, "ranked": [c["candidate"] for c in ranked],
           "winner": None, "runner_up": None, "deciding_step": None, "closeness": []}
    if not ranked:
        return out
    w = ranked[0]
    out["winner"] = {k: w[k] for k in ("candidate", "run", "epoch", "method", "lr", "seed", "arm")}
    out["winner"]["eval_base"] = w.get("eval_base", "fp16")
    if len(ranked) > 1:
        r = ranked[1]
        out["runner_up"] = r["candidate"]
        pw, pr = rank_key(w, sel)[1], rank_key(r, sel)[1]
        out["deciding_step"] = next((k for k in sel["order"] if pw[k] != pr[k]), None)
        for other in ranked[1:3]:
            out["closeness"] += closeness(w, other)
    return out


# ---- repeats, other size -------------------------------------------------------------------------------------------
STAT_KEYS = (("correct_refusals_strict", lambda m: m["correct_strict"]["count"]),
             ("false_refusals", lambda m: m["false_refusals"]["count"]),
             ("token_f1", lambda m: m["token_f1"]["mean"]),
             ("key_recall", lambda m: (m["key_recall"] or {}).get("mean")),
             ("key_support", lambda m: (m["key_support"] or {}).get("mean")),
             ("number_faithfulness", lambda m: m["number_faithfulness"]["mean"]),
             ("refusal_perplexity", lambda m: m["refusal_perplexity"]))


def repeat_summary(cfg3: dict, eval_cfg: dict, winner: dict) -> dict:
    """The winning configuration (same epoch) for every seed that has a val evaluation: per seed and mean / min / max."""
    runs = [grid.make_run(cfg3, "main", winner["method"], winner["lr"], winner["seed"])] + \
        grid.repeat_runs(cfg3, {"method": winner["method"], "lr": winner["lr"]})
    rows = []
    for r in runs:
        name = grid.candidate_name(r["name"], winner["epoch"], winner.get("eval_base") == "4bit")
        m = load_metrics(eval_cfg, cfg3, name, "val")
        if m is not None:
            rows.append({"seed": r["seed"], "candidate": name, "metrics": m})
    stats = {}
    for key, fn in STAT_KEYS:
        vals = [fn(r["metrics"]) for r in rows if fn(r["metrics"]) is not None]
        if vals:
            stats[key] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
    return {"epoch": winner["epoch"], "seeds_found": [r["seed"] for r in rows],
            "seeds_expected": [winner["seed"]] + cfg3["sweep"]["repeat_seeds"], "rows": rows, "stats": stats}


# ---- the 7 criteria ----------------------------------------------------------------------------------------------------
def criteria_rows(m: dict, split: str, base: dict | None, cfg3: dict, test_runs: int | None = None) -> list[dict]:
    c = cfg3["criteria"]
    rows = []

    def add(num, text, ok, value):
        rows.append({"id": num, "text": text, "ok": ok, "value": value})

    cs, fr = m["correct_strict"], m["false_refusals"]
    add(1, f"strict correct refusals >= {c['min_correct_refusals']}/{cs['n']}", cs["count"] >= c["min_correct_refusals"],
        f"{cs['count']}/{cs['n']}")
    add(2, f"false refusals on answerable <= {c['max_false_refusals'][split]}/{fr['n']}",
        fr["count"] <= c["max_false_refusals"][split], f"{fr['count']}/{fr['n']}")
    f1 = m["token_f1"]["mean"]
    if base is not None:
        b = base["token_f1"]["mean"]
        extra = ""
        if split == "test":
            extra = f"; target {c['f1_target_test']:.2f} " + ("reached" if f1 >= c["f1_target_test"] else "not reached")
        add(3, f"answerable token F1 not below base ({b:.3f})", f1 >= b, f"{f1:.3f}{extra}")
    else:
        add(3, "answerable token F1 not below base", None, f"{f1:.3f} (base file missing)")
    nf = m["number_faithfulness"]
    add(4, f"number faithfulness >= {c['min_number_faithfulness']}",
        None if nf["mean"] is None else nf["mean"] >= c["min_number_faithfulness"],
        "n/a (no numbers)" if nf["mean"] is None else f"{nf['mean']:.3f} (on {nf['n_applied']} items)")
    ok5 = m["hit_max_new_tokens"] <= c["max_hit_max_new_tokens"] and m["repetition_loops"] == 0
    add(5, f"at most {c['max_hit_max_new_tokens']} output hits max_new_tokens and no repetition loop", ok5,
        f"hit max {m['hit_max_new_tokens']}, loops {m['repetition_loops']}")
    if split == "val":
        add(6, "checkpoint chosen on val only; test run once", True, "selection uses val files only (by construction)")
    else:
        add(6, "checkpoint chosen on val only; test run once", test_runs == 1, f"test runs recorded: {test_runs}")
    add(7, "if criterion 1 fails: no DPO until the SFT data or settings are checked", None if cs["count"] >= c["min_correct_refusals"] else False,
        "criterion 1 met, no restriction" if cs["count"] >= c["min_correct_refusals"] else "criterion 1 NOT met: no DPO yet")
    return rows


def prompted_statements(m: dict, split: str, prompted: dict[str, dict], label: str) -> list[str]:
    """Does this model beat the best prompted baseline? Plain words, with interval overlap."""
    if not prompted:
        return []
    out = []
    specs = (("answerable token F1", lambda x: x["token_f1"]), ("key recall", lambda x: x["key_recall"]),
             ("key support", lambda x: x["key_support"]))
    for text, get in specs:
        best_name, best = max(((n, p) for n, p in prompted.items() if get(p)), key=lambda t: get(t[1])["mean"], default=(None, None))
        mine = get(m)
        if best is None or not mine:
            continue
        cmp_ = summary.compare(mine["mean"], mine["ci95"], get(best)["mean"], get(best)["ci95"])
        out.append(f"{label}, {split}: {text} {mine['mean']:.3f} {_ci_text(mine['ci95'])} is {cmp_['direction']} than the best "
                   f"prompted baseline {best_name} {get(best)['mean']:.3f} {_ci_text(get(best)['ci95'])}"
                   + (" (intervals overlap: no difference can be claimed)" if cmp_["overlap"] and cmp_["direction"] != "equal" else ""))
    bn, bp = max(prompted.items(), key=lambda t: t[1]["correct_strict"]["count"])
    a, b = m["correct_strict"], bp["correct_strict"]
    cmp_ = summary.compare(a["rate"], a["wilson95"], b["rate"], b["wilson95"])
    out.append(f"{label}, {split}: strict correct refusals {a['count']}/{a['n']} is {cmp_['direction']} than the best prompted "
               f"baseline {bn} {b['count']}/{b['n']}"
               + (" (intervals overlap: no difference can be claimed)" if cmp_["overlap"] and cmp_["direction"] != "equal" else ""))
    return out


# ---- collecting everything -------------------------------------------------------------------------------------------------
def collect(cfg3: dict, eval_cfg: dict) -> dict:
    """All numbers the tables need, from saved files. Tuned test results are included if the files exist."""
    data = {"grid": {"runs": [], "candidates": [], "missing_runs": []}, "baselines": {s: {} for s in BASELINE_SPLITS}}
    for r in grid.grid_runs(cfg3):
        cands = run_candidates(cfg3, eval_cfg, r)
        data["grid"]["runs"].append({"name": r["name"], "method": r["method"], "lr": r["lr"], "cost": cost_of(cfg3, r["name"])})
        data["grid"]["candidates"] += cands
        if sum(1 for c in cands if c["eval_base"] == "fp16") < len(grid.epochs(cfg3)):
            data["grid"]["missing_runs"].append({"run": r["name"], "evaluated_epochs": [c["epoch"] for c in cands]})
    data["grid"].update(select_winner(data["grid"]["candidates"], cfg3))
    names = [n for n, _ in cfg3["tables"]["baselines"]] + [a["base_eval_name"] for k, a in cfg3["arms"].items() if k != "main"]
    for split in BASELINE_SPLITS:
        for n in names:
            m = load_metrics(eval_cfg, cfg3, n, split)
            if m is not None:
                data["baselines"][split][n] = m
    w = data["grid"]["winner"]
    data["repeats"] = repeat_summary(cfg3, eval_cfg, w) if w else None
    data["q15"] = None
    if w:
        run = grid.q15_run(cfg3, w)
        cands = run_candidates(cfg3, eval_cfg, run)
        if cands:
            data["q15"] = {"run": run["name"], "cost": cost_of(cfg3, run["name"]), "candidates": cands,
                           **select_winner(cands, cfg3)}
    data["tuned"] = {}
    for key, block in (("main", data["grid"]), ("q15", data["q15"])):
        if block and block["winner"]:
            cand = next(c for c in block["candidates"] if c["candidate"] == block["winner"]["candidate"])
            data["tuned"][key] = {"winner": block["winner"], "val": cand["metrics"], "cost": cand["cost"],
                                  "test": load_metrics(eval_cfg, cfg3, cand["candidate"], "test")}
    data["other_costs"] = {r["name"]: cost_of(cfg3, r["name"]) for r in
                           (grid.repeat_runs(cfg3, w) if w else [])}
    return data


# ---- markdown -----------------------------------------------------------------------------------------------------------------
def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def mean_cell(b: dict | None, dec: int = 3) -> str:
    if not b or b.get("mean") is None:
        return "n/a"
    return f"{b['mean']:.{dec}f} {_ci_text(b['ci95'])}"


def rate_cell(b: dict) -> str:
    return f"{b['count']}/{b['n']}"


def metric_cells(m: dict | None, not_measured: str = "not yet measured") -> list[str]:
    if m is None:
        return [not_measured] * 9
    nf = m["number_faithfulness"]
    return [mean_cell(m["token_f1"]), mean_cell(m["key_recall"]), mean_cell(m["key_support"]),
            "n/a" if nf["mean"] is None else f"{nf['mean']:.3f} (n={nf['n_applied']})", rate_cell(m["false_refusals"]),
            rate_cell(m["correct_strict"]), rate_cell(m["correct_loose"]),
            f"{m['hit_max_new_tokens']} / {m['repetition_loops']}",
            "n/a" if m["refusal_perplexity"] is None else f"{m['refusal_perplexity']:.2f}"]


METRIC_HEADER = ["token F1", "key recall", "key support", "number faithfulness", "false refusals", "correct refusals, strict",
                 "correct refusals, loose", "hit max / loops", "refusal perplexity"]


def _fmt(x, spec: str, none: str = "n/a") -> str:
    return none if x is None else format(x, spec)


def grid_table(data: dict, cfg3: dict) -> list[str]:
    g = data["grid"]
    rows = []
    for c in g["candidates"]:
        if g["winner"] and c["candidate"] == g["winner"]["candidate"]:
            status = "**WINNER**"
        elif c["candidate"] in g["ranked"]:
            status = f"eligible, rank {g['ranked'].index(c['candidate']) + 1}"
        else:
            status = f"excluded (false refusals > {cfg3['selection']['max_false_refusals_val']})"
        name = f"**{c['candidate']}**" if status == "**WINNER**" else c["candidate"]
        base = "4-bit base" if c.get("eval_base") == "4bit" else "fp16 base"
        rows.append([name, c["method"], f"{c['lr']:g}", str(c["epoch"]), base, _fmt(c["val_loss"], ".3f")]
                    + metric_cells(c["metrics"]) + [status])
    return table(["candidate", "method", "lr", "epoch", "evaluated on", "val loss"] + METRIC_HEADER + ["status"], rows)


def paired_lines(cands: list[dict]) -> list[str]:
    """The same QLoRA adapter evaluated on the fp16 base and on the 4-bit base, side by side."""
    by = {(c["run"], c["epoch"], c.get("eval_base", "fp16")): c["metrics"] for c in cands}
    rows = []
    for (run, epoch, base), m4 in sorted(by.items()):
        if base != "4bit" or (run, epoch, "fp16") not in by:
            continue
        m16 = by[(run, epoch, "fp16")]
        rows.append([f"{run}_e{epoch}", rate_cell(m16["correct_strict"]), rate_cell(m4["correct_strict"]),
                     rate_cell(m16["false_refusals"]), rate_cell(m4["false_refusals"]),
                     f"{m16['token_f1']['mean']:.3f}", f"{m4['token_f1']['mean']:.3f}"])
    if not rows:
        return []
    return table(["adapter", "correct refusals, fp16 base", "correct refusals, 4-bit base", "false refusals, fp16 base",
                  "false refusals, 4-bit base", "token F1, fp16 base", "token F1, 4-bit base"], rows) + [""]


def cost_table(runs: list[tuple[str, dict | None]]) -> list[str]:
    rows = []
    for name, c in runs:
        if not c:
            rows.append([name, "not run"] + [""] * 7)
            continue
        rows.append([name, "complete" if c["complete"] else "NOT complete", _fmt(c["peak_gpu_mem_mb"], ".0f") + " MB",
                     _fmt(c["train_seconds"], ".0f") + " s",
                     f"{c['trainable_params']:,}" if c["trainable_params"] else "n/a",
                     _fmt(None if c["trainable_share"] is None else 100 * c["trainable_share"], ".2f") + " %",
                     _fmt(c["adapter_mb"], ".1f") + " MB",
                     str(len(c.get("dropped_train_ids") or [])) + " / " + str(len(c.get("dropped_val_ids") or [])),
                     "n/a" if c.get("skipped_steps") is None else str(c["skipped_steps"])])
    return table(["run", "state", "peak GPU memory", "training time", "trainable parameters", "share", "adapter size (one epoch)",
                  "rows dropped (train / val)", "skipped optimiser steps"], rows)


def size_rows(data: dict, cfg3: dict) -> list[dict]:
    """Rows of the model-size table: label, val metrics, test metrics (None if not measured)."""
    b = data["baselines"]
    rows = [{"label": lab, "name": n, "val": b["val"].get(n), "test": b["test"].get(n), "tuned": None}
            for n, lab in cfg3["tables"]["baselines"]]
    t = data["tuned"].get("main")
    if t:
        rows.append({"label": f"tuned 0.5B ({t['winner']['candidate']})", "name": t["winner"]["candidate"], "val": t["val"],
                     "test": t["test"], "tuned": "main"})
    b15 = cfg3["arms"]["q15"]["base_eval_name"]
    rows.append({"label": "base 1.5B (default prompt)", "name": b15, "val": b["val"].get(b15), "test": b["test"].get(b15), "tuned": None})
    t = data["tuned"].get("q15")
    if t:
        rows.append({"label": f"tuned 1.5B ({t['winner']['candidate']})", "name": t["winner"]["candidate"], "val": t["val"],
                     "test": t["test"], "tuned": "q15"})
    return rows


def size_table_lines(data: dict, cfg3: dict) -> list[str]:
    lines = []
    rows = size_rows(data, cfg3)
    for split in BASELINE_SPLITS:
        lines += [f"#### {split}", ""] + table(["model"] + METRIC_HEADER, [[r["label"]] + metric_cells(r[split]) for r in rows]) + [""]
    return lines


def criteria_lines(data: dict, cfg3: dict, test_runs: dict[str, int] | None = None) -> list[str]:
    lines = []
    ref = cfg3["tables"]["base_reference"]
    for r in size_rows(data, cfg3):
        if not r["tuned"]:
            continue
        for split in BASELINE_SPLITS:
            m = r[split]
            if m is None:
                lines += [f"**{r['label']}, {split}:** not yet measured", ""]
                continue
            tr = None if test_runs is None else test_runs.get(r["name"], 0)
            # criterion 3 uses the approved base numbers (the untuned 0.5B model) for every tuned row
            rows = criteria_rows(m, split, data["baselines"][split].get(ref), cfg3, tr)
            ok = {True: "yes", False: "NO", None: "n/a"}
            lines += [f"**{r['label']}, {split}**", ""] + table(
                ["#", "criterion", "met", "value"], [[str(x["id"]), x["text"], ok[x["ok"]], x["value"]] for x in rows]) + [""]
    return lines


def statement_lines(data: dict, cfg3: dict) -> list[str]:
    lines = []
    prompted_names = cfg3["tables"]["prompted_baselines"]
    for r in size_rows(data, cfg3):
        if not r["tuned"]:
            continue
        for split in BASELINE_SPLITS:
            if r[split] is None:
                continue
            prompted = {n: data["baselines"][split][n] for n in prompted_names if n in data["baselines"][split]}
            lines += [f"- {s}" for s in prompted_statements(r[split], split, prompted, r["label"])]
    return lines or ["- (no tuned model measured yet)"]


def selection_md(data: dict, cfg3: dict) -> str:
    g, sel = data["grid"], cfg3["selection"]
    L = ["# Phase 3 selection (validation set only)", "",
         "Built by `python -m climate_slm.training.select_best` from saved files only. The rule was fixed in "
         "`configs/phase3.yaml` before any run:", "",
         f"1. keep candidates with false refusals <= {sel['max_false_refusals_val']} of 42 on val",
         "2. highest strict correct refusals on val", "3. higher val answerable token F1", "4. higher number faithfulness",
         "5. the cheaper one: lower peak GPU memory, then fewer trainable parameters, then shorter training time",
         "6. last resort: the earlier epoch", "",
         "## Grid: all candidates (4 runs x 3 epochs)", ""]
    L += grid_table(data, cfg3) + [""]
    pairs = paired_lines(g["candidates"])
    if pairs:
        L += ["Rows with `__on4bit` in the name are EXTRA candidates: the same QLoRA adapter, evaluated on the 4-bit base it "
              "was trained on (the other rows use the fp16 base). They are ranked under the same fixed rule as every other row. "
              "Same adapter, both bases:", ""] + pairs
    if g["missing_runs"]:
        L += ["Runs with fewer than 3 evaluated epochs: " + "; ".join(f"{m['run']} (epochs found {m['evaluated_epochs']})"
                                                                     for m in g["missing_runs"]), ""]
    L += ["## Winner", ""]
    if g["winner"]:
        w = g["winner"]
        L += [f"**{w['candidate']}** (method {w['method']}, learning rate {w['lr']:g}, epoch {w['epoch']}).", ""]
        if w.get("eval_base") == "4bit":
            L += ["**The winner was evaluated on the 4-bit base.** The seed repeats must be evaluated the same way "
                  "(`eval_adapters --group repeats --on-4bit-base`), and the final test uses `--load-4bit`.", ""]
        if g["runner_up"]:
            L += [f"Runner-up: {g['runner_up']}. The rule step that decided: **{g['deciding_step']}**.", ""]
        else:
            L += ["Only one candidate passed the false-refusal filter.", ""]
        if g["closeness"]:
            wm = next(c for c in g["candidates"] if c["candidate"] == w["candidate"])["metrics"]
            L += [f"How close is it (95% intervals; val has {wm['n_answerable']} answerable and {wm['n_refusal']} refusal items):", ""]
            L += [f"- {x}" for x in g["closeness"]] + [""]
            if any("NOT established" in x for x in g["closeness"]):
                L += ["In plain words: where a line says the intervals overlap, the data cannot show that the winner is better "
                      "than that candidate. The choice between them rests on the fixed rule (and, at the end, on cost), not on a "
                      "proven quality difference.", ""]
            if not any("do not overlap" in x for x in g["closeness"]):
                L += [f"**In plain words: the top candidates cannot be separated with this much data. The rule picked {w['candidate']}; "
                      "another top candidate could just as well be the better one.**", ""]
    else:
        L += ["**No candidate passed the false-refusal filter.** Nothing is selected. Criterion 1 cannot be met by "
              "these runs; per criterion 7 no DPO until the SFT data or settings are checked. The optional run "
              "`train.refusal_repeat: 2` is the planned next step, only if told.", ""]
    if g["excluded"]:
        L += ["Excluded by the false-refusal filter: " + ", ".join(f"{e['candidate']} ({e['false_refusals']}/{e['n']})" for e in g["excluded"]), ""]
    L += ["## Training cost per run", ""]
    runs = [(r["name"], r["cost"]) for r in g["runs"]] + sorted(data["other_costs"].items())
    if data["q15"]:
        runs.append((data["q15"]["run"], data["q15"]["cost"]))
    L += cost_table(runs) + [""]
    rp = data["repeats"]
    L += ["## Seed repeat of the winner (same epoch)", ""]
    if rp and len(rp["rows"]) > 1:
        L += table(["seed", "candidate"] + METRIC_HEADER, [[str(r["seed"]), r["candidate"]] + metric_cells(r["metrics"]) for r in rp["rows"]]) + [""]
        names = {"correct_refusals_strict": "strict correct refusals (of 11)", "false_refusals": "false refusals (of 42)",
                 "token_f1": "answerable token F1", "key_recall": "key recall", "key_support": "key support",
                 "number_faithfulness": "number faithfulness", "refusal_perplexity": "refusal perplexity"}
        L += table(["metric", "mean", "min", "max", "seeds"],
                   [[names[k], f"{v['mean']:.3f}", f"{v['min']:.3f}", f"{v['max']:.3f}", str(v["n"])] for k, v in rp["stats"].items()]) + [""]
        if sorted(rp["seeds_found"]) != sorted(rp["seeds_expected"]):
            L += [f"Seeds found {rp['seeds_found']}, expected {rp['seeds_expected']}.", ""]
    else:
        L += ["not yet measured (run the seed repeats and evaluate them on val)", ""]
    L += ["## Other model size (1.5B)", ""]
    q = data["q15"]
    if q:
        L += grid_table({"grid": {"candidates": q["candidates"], "winner": q["winner"], "ranked": q["ranked"]}}, cfg3) + [""]
        L += [f"Chosen epoch by the same rule: **{q['winner']['candidate']}**." if q["winner"] else "No epoch passed the false-refusal filter.", ""]
    else:
        L += ["not yet measured", ""]
    L += ["## Model size comparison (val, then test once)", "",
          "Tuned rows show test only after the single test run of the frozen winner. Prompted-baseline rows are the saved "
          "Phase 2c files (nothing is run for them).", ""] + size_table_lines(data, cfg3)
    L += ["## The 7 pass criteria", ""] + criteria_lines(data, cfg3)
    L += ["## Against the prompted baselines (base_p_sentence, base_p_fewshot)", ""] + statement_lines(data, cfg3) + [""]
    return "\n".join(L)


def payload(data: dict) -> dict:
    """What is frozen: the validation selection. Tuned test results are NOT part of it."""
    out = json.loads(json.dumps(data))
    for k in out["tuned"].values():
        k.pop("test", None)
    return out


def test_command(winner: dict, size: str) -> str:
    return f"python -m climate_slm.training.final_test --size {size}"


def final_md(data: dict, cfg3: dict, test_runs: dict[str, int], sha: str | None) -> str:
    L = ["# Phase 3 final tables (validation and test)", "",
         f"Selection file sha256 (frozen before any test run): `{sha}`" if sha else "Selection file is NOT frozen.", "",
         "## Model size comparison (val, then test once)", ""] + size_table_lines(data, cfg3)
    L += ["## The 7 pass criteria", ""] + criteria_lines(data, cfg3, test_runs)
    L += ["## Against the prompted baselines", ""] + statement_lines(data, cfg3) + [""]
    L += ["## Training cost per run", ""]
    runs = [(r["name"], r["cost"]) for r in data["grid"]["runs"]] + sorted(data["other_costs"].items())
    if data["q15"]:
        runs.append((data["q15"]["run"], data["q15"]["cost"]))
    L += cost_table(runs) + [""]
    return "\n".join(L)


# ---- command line -----------------------------------------------------------------------------------------------------------------
def read_test_runs(cfg3: dict) -> dict[str, int]:
    p = REPO_ROOT / cfg3["paths"]["test_log"]
    if not p.exists():
        return {}
    counts: dict[str, int] = {}
    for e in json.loads(p.read_text(encoding="utf-8")):
        counts[e["candidate"]] = counts.get(e["candidate"], 0) + 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply the Phase 3 selection rule to the saved val evaluations.")
    ap.add_argument("--freeze", action="store_true", help="store the sha256 of the selection file")
    ap.add_argument("--final-tables", action="store_true", help="after the test runs: write phase3_final.md")
    a = ap.parse_args(argv)
    cfg3, eval_cfg = load_phase3_config(), load_eval_config()
    sp, mp = selection_io.selection_path(cfg3), REPO_ROOT / cfg3["paths"]["selection_md"]
    data = collect(cfg3, eval_cfg)
    if a.final_tables:
        if not selection_io.is_frozen(cfg3):
            raise SystemExit("The selection is not frozen. Run select_best --freeze first.")
        sha = selection_io.sha_path(cfg3).read_text(encoding="utf-8").split()[0]
        out = REPO_ROOT / cfg3["paths"]["final_md"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(final_md(data, cfg3, read_test_runs(cfg3), sha), encoding="utf-8")
        print("wrote", out)
        return 0
    if selection_io.is_frozen(cfg3):
        print(f"The selection is frozen ({sp.name}, sha256 {selection_io.file_sha256(sp)}). Not rewritten.")
    else:
        sp.parent.mkdir(parents=True, exist_ok=True)
        mp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(payload(data), indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        mp.write_text(selection_md(data, cfg3), encoding="utf-8")
        print("wrote", sp, "and", mp)
    if a.freeze:
        if not selection_io.is_frozen(cfg3):
            sha = selection_io.file_sha256(sp)
            selection_io.sha_path(cfg3).write_text(f"{sha}  {sp.name}\n", encoding="utf-8")
        print("frozen: sha256", selection_io.file_sha256(sp))
    g = data["grid"]["winner"]
    print("0.5B winner:", g["candidate"] if g else "none")
    if g:
        print("test command for it (run ONCE, after --freeze):", test_command(g, "0.5b"))
    q = data["q15"]["winner"] if data["q15"] else None
    print("1.5B winner:", q["candidate"] if q else "none (or not measured yet)")
    if q:
        print("test command for it (run ONCE, after --freeze):", test_command(q, "1.5b"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
