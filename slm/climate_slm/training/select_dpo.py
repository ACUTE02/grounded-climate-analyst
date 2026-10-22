"""Phase 4 selection (DPO). Reads ONLY saved files (eval JSON of the evaluator, train_log.json of the runs).

    python -m climate_slm.training.select_dpo                 # write outputs/phase4_selection.md and .json
    python -m climate_slm.training.select_dpo --freeze        # also store the sha256 of the selection file
    python -m climate_slm.training.select_dpo --final-tables  # after the one test run: outputs/phase4_final.md

(A separate module, not a flag of select_best: the Phase 3 rule, its 4-bit rows and its 1.5B arm do not apply here.)

The rule (fixed in configs/phase4.yaml before any run), applied on VAL only, to the 12 candidates (4 runs x 3 epochs):
  filter: strict correct refusals >= 7/11; median output length <= 1.5 x that of the SFT control; at most 1 output at
          max_new_tokens
  1. fewest false refusals   2. more strict correct refusals   3. higher answerable token F1
  4. higher number faithfulness   5. cheaper (lower peak GPU memory, then shorter training time)   6. earlier epoch.
If no candidate that passes the filter has FEWER false refusals than the SFT control seed 42 on val, the result is
"no DPO candidate improves on the SFT model on val": there is no winner and the test step cannot run.
Once the selection file is frozen it is not rewritten."""
import argparse
import json
import math
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config, load_phase4_config
from climate_slm.training import dpo_grid, grid, selection_io, summary
from climate_slm.training import select_best as sb

NO_IMPROVEMENT = "no DPO candidate improves on the SFT model on val"
SPLITS = ("val", "test")


# ---- loading ---------------------------------------------------------------------------------------------------------
def cost_of(cfg4: dict, run_name: str) -> dict | None:
    p = grid.run_log_path(cfg4, run_name)
    if not p.exists():
        return None
    log = json.loads(p.read_text(encoding="utf-8"))
    eps = log.get("epochs") or []
    return {"complete": bool(log.get("complete")), "peak_gpu_mem_mb": log.get("peak_gpu_mem_mb"),
            "train_seconds": log.get("train_seconds"), "total_seconds": log.get("total_seconds"),
            "trainable_params": log.get("trainable_params"), "trainable_share": log.get("trainable_share"),
            "adapter_mb": eps[-1]["adapter_mb"] if eps else None, "skipped_steps": log.get("skipped_steps"),
            "dropped_distinct_ids": log.get("dropped_distinct_ids"), "dropped_lines": log.get("dropped_lines"),
            "pair_counts": log.get("pair_counts"), "start_adapter_sha256": log.get("start_adapter_sha256"),
            "pairs_sha256": log.get("pairs_sha256"),
            "epochs": {str(e["epoch"]): {k: e.get(k) for k in ("dpo_loss", "reward_accuracy", "mean_margin")} for e in eps}}


def run_candidates(cfg4: dict, cfg3: dict, eval_cfg: dict, run: dict, split: str = "val") -> list[dict]:
    cost, out = cost_of(cfg4, run["name"]), []
    for e in grid.epochs(cfg4):
        name = grid.candidate_name(run["name"], e)
        m = sb.load_metrics(eval_cfg, cfg3, name, split)
        if m is not None:
            out.append({"candidate": name, "run": run["name"], "epoch": e, "beta": run["beta"], "lr": run["lr"],
                        "seed": run["seed"], "metrics": m, "cost": cost,
                        "train_epoch": (cost or {}).get("epochs", {}).get(str(e))})
    return out


def control_metrics(cfg4: dict, cfg3: dict, eval_cfg: dict, seed: int, split: str = "val") -> dict | None:
    return sb.load_metrics(eval_cfg, cfg3, dpo_grid.control_name(cfg4, seed), split)


# ---- the rule --------------------------------------------------------------------------------------------------------------
def median_chars(m: dict) -> float | None:
    return (m.get("output_chars") or {}).get("median")


def drop_reasons(c: dict, control: dict, sel: dict) -> list[str]:
    """Why a candidate is removed before ranking ([] = it stays)."""
    m, reasons = c["metrics"], []
    cs = m["correct_strict"]
    if cs["count"] < sel["min_correct_refusals_val"]:
        reasons.append(f"strict correct refusals {cs['count']}/{cs['n']} < {sel['min_correct_refusals_val']}")
    mc, cc = median_chars(m), median_chars(control)
    if mc is not None and cc is not None and mc > sel["max_median_length_ratio"] * cc:
        reasons.append(f"median output length {mc:g} chars > {sel['max_median_length_ratio']} x control {cc:g}")
    if m["hit_max_new_tokens"] > sel["max_hit_max_new_tokens"]:
        reasons.append(f"{m['hit_max_new_tokens']} outputs at max_new_tokens (> {sel['max_hit_max_new_tokens']})")
    return reasons


def rank_key(c: dict, sel: dict) -> tuple[tuple, dict]:
    """(sort key, readable parts). Smaller key = better."""
    d, m, cost = sel["round_decimals"], c["metrics"], c["cost"] or {}
    nf = m["number_faithfulness"]["mean"]
    parts = {"false_refusals": m["false_refusals"]["count"],
             "correct_refusals_strict": -m["correct_strict"]["count"],
             "token_f1": -round(m["token_f1"]["mean"], d),
             "number_faithfulness": -round(nf if nf is not None else -1.0, d),
             "peak_gpu_mem": cost.get("peak_gpu_mem_mb") if cost.get("peak_gpu_mem_mb") is not None else math.inf,
             "train_seconds": cost.get("train_seconds") if cost.get("train_seconds") is not None else math.inf,
             "epoch": c["epoch"]}
    return tuple(parts[k] for k in sel["order"]), parts


def vs_control_lines(c: dict, control: dict, label_c: str, label_s: str = "SFT control") -> list[str]:
    """Plain-words comparison of a DPO candidate with its SFT control: false refusals, strict correct refusals, F1."""
    m, lines = c, []
    for text, key in (("false refusals", "false_refusals"), ("strict correct refusals", "correct_strict")):
        a, b = m[key], control[key]
        cmp_ = summary.compare(a["rate"], a["wilson95"], b["rate"], b["wilson95"])
        verdict = {"equal": "equal", None: "n/a"}.get(cmp_["direction"], cmp_["direction"])
        tail = (" (the 95% intervals overlap: NOT established)" if cmp_["overlap"] and cmp_["direction"] not in ("equal", None)
                else "" if cmp_["direction"] in ("equal", None) else " (intervals do not overlap)")
        lines.append(f"{text}: {label_c} {a['count']}/{a['n']} against {label_s} {b['count']}/{b['n']}: {verdict}{tail}")
    fa, fb = m["token_f1"], control["token_f1"]
    cmp_ = summary.compare(fa["mean"], fa["ci95"], fb["mean"], fb["ci95"])
    tail = " (the 95% intervals overlap: NOT established)" if cmp_["overlap"] and cmp_["direction"] not in ("equal", None) else ""
    lines.append(f"answerable token F1: {label_c} {fa['mean']:.3f} against {label_s} {fb['mean']:.3f}: "
                 f"{cmp_['direction'] or 'n/a'}{tail}")
    return lines


def select_winner(cands: list[dict], control: dict | None, cfg4: dict) -> dict:
    """Apply the rule to the val candidates. `control` = val metrics of the SFT control (seed 42)."""
    sel = cfg4["selection"]
    out = {"n_candidates": len(cands), "dropped": [], "ranked": [], "winner": None, "runner_up": None,
           "deciding_step": None, "closeness": [], "improves_on_control": False, "status": None,
           "control_false_refusals": None}
    if control is None:
        out["status"] = "the SFT control val file is missing: nothing can be decided"
        return out
    fr0 = control["false_refusals"]["count"]
    out["control_false_refusals"] = fr0
    if not cands:
        out["status"] = "not yet measured: no DPO candidate has been evaluated on val"
        return out
    kept = []
    for c in cands:
        why = drop_reasons(c, control, sel)
        if why:
            out["dropped"].append({"candidate": c["candidate"], "reasons": why})
        else:
            kept.append(c)
    ranked = sorted(kept, key=lambda c: rank_key(c, sel)[0])
    out["ranked"] = [c["candidate"] for c in ranked]
    if not ranked:
        out["status"] = f"{NO_IMPROVEMENT} (no candidate passes the filter)"
        return out
    best = ranked[0]
    if best["metrics"]["false_refusals"]["count"] >= fr0:
        out["status"] = (f"{NO_IMPROVEMENT}: the best candidate that passes the filter has "
                         f"{best['metrics']['false_refusals']['count']} false refusals on val, the SFT control has {fr0}")
        return out
    out["improves_on_control"] = True
    out["winner"] = {k: best[k] for k in ("candidate", "run", "epoch", "beta", "lr", "seed")}
    out["status"] = "winner found"
    if len(ranked) > 1:
        r = ranked[1]
        out["runner_up"] = r["candidate"]
        pw, pr = rank_key(best, sel)[1], rank_key(r, sel)[1]
        out["deciding_step"] = next((k for k in sel["order"] if pw[k] != pr[k]), None)
        for other in ranked[1:3]:
            out["closeness"] += sb.closeness(best, other)
    return out


# ---- seeds: SFT then DPO, paired -----------------------------------------------------------------------------------
def paired_seeds(cfg4: dict, cfg3: dict, eval_cfg: dict, winner: dict, run_fn=None) -> dict:
    """The winning configuration (same epoch) for every seed with a val evaluation, next to the SFT control of that seed.
    run_fn(seed) -> run dict; the default is the v1 configuration (beta, lr). DPO v2 passes its own."""
    rows = []
    seeds = [cfg4["sweep"]["seed"]] + list(cfg4["sweep"]["repeat_seeds"])
    for sd in seeds:
        run = run_fn(sd) if run_fn else dpo_grid.make_run(cfg4, winner["beta"], winner["lr"], sd)
        name = grid.candidate_name(run["name"], winner["epoch"])
        dpo, sft = sb.load_metrics(eval_cfg, cfg3, name, "val"), control_metrics(cfg4, cfg3, eval_cfg, sd)
        rows.append({"seed": sd, "dpo_candidate": name, "sft_candidate": dpo_grid.control_name(cfg4, sd), "dpo": dpo, "sft": sft})
    both = [r for r in rows if r["dpo"] and r["sft"]]
    counts = {}
    for key, label, better in (("false_refusals", "false_refusals", "lower"), ("correct_strict", "correct_refusals", "higher")):
        d = {"better": 0, "equal": 0, "worse": 0}
        for r in both:
            a, b = r["dpo"][key]["count"], r["sft"][key]["count"]
            win = a < b if better == "lower" else a > b
            d["better" if win else "equal" if a == b else "worse"] += 1
        counts[label] = d
    f1 = {"better": 0, "equal": 0, "worse": 0}
    for r in both:
        a, b = round(r["dpo"]["token_f1"]["mean"], 4), round(r["sft"]["token_f1"]["mean"], 4)
        f1["better" if a > b else "equal" if a == b else "worse"] += 1
    counts["token_f1"] = f1
    return {"epoch": winner["epoch"], "rows": rows, "seeds_expected": seeds, "seeds_paired": [r["seed"] for r in both],
            "counts": counts}


def consistency_lines(p: dict) -> list[str]:
    n = len(p["seeds_paired"])
    if n == 0:
        return ["not yet measured"]
    c = p["counts"]
    lines = [f"Across the {n} paired seed(s) on val (DPO against its own SFT seed): false refusals fewer in "
             f"{c['false_refusals']['better']}, equal in {c['false_refusals']['equal']}, more in {c['false_refusals']['worse']}; "
             f"strict correct refusals more in {c['correct_refusals']['better']}, equal in {c['correct_refusals']['equal']}, "
             f"fewer in {c['correct_refusals']['worse']}; token F1 higher in {c['token_f1']['better']}, equal in "
             f"{c['token_f1']['equal']}, lower in {c['token_f1']['worse']}."]
    if n < len(p["seeds_expected"]):
        lines.append(f"Only seeds {p['seeds_paired']} of {p['seeds_expected']} are measured so far.")
    lines.append("With 42 answerable and 11 refusal items on val, a difference of one or two items cannot be established "
                 "(see the intervals in the tables); this block counts seeds, it does not prove an effect.")
    return lines


# ---- collecting everything ---------------------------------------------------------------------------------------------
def collect(cfg4: dict, cfg3: dict, eval_cfg: dict) -> dict:
    data = {"runs": [], "candidates": [], "missing_runs": []}
    for r in dpo_grid.grid_runs(cfg4):
        cands = run_candidates(cfg4, cfg3, eval_cfg, r)
        data["runs"].append({"name": r["name"], "beta": r["beta"], "lr": r["lr"], "cost": cost_of(cfg4, r["name"])})
        data["candidates"] += cands
        if len(cands) < len(grid.epochs(cfg4)):
            data["missing_runs"].append({"run": r["name"], "evaluated_epochs": [c["epoch"] for c in cands]})
    seed0 = cfg4["selection"]["control_seed"]
    data["control"] = {"candidate": dpo_grid.control_name(cfg4, seed0), "val": control_metrics(cfg4, cfg3, eval_cfg, seed0),
                       "test": control_metrics(cfg4, cfg3, eval_cfg, seed0, "test")}
    data.update(select_winner(data["candidates"], data["control"]["val"], cfg4))
    w = data["winner"]
    data["paired"] = paired_seeds(cfg4, cfg3, eval_cfg, w) if w else None
    data["repeat_costs"] = {r["name"]: cost_of(cfg4, r["name"]) for r in dpo_grid.repeat_runs(cfg4, w)} if w else {}
    data["baselines"] = {s: {} for s in SPLITS}
    for split in SPLITS:
        for n, _ in cfg3["tables"]["baselines"]:
            m = sb.load_metrics(eval_cfg, cfg3, n, split)
            if m is not None:
                data["baselines"][split][n] = m
    data["tuned"] = None
    if w:
        cand = next(c for c in data["candidates"] if c["candidate"] == w["candidate"])
        data["tuned"] = {"winner": w, "val": cand["metrics"], "cost": cand["cost"],
                         "test": sb.load_metrics(eval_cfg, cfg3, w["candidate"], "test")}
    return data


# ---- markdown ---------------------------------------------------------------------------------------------------------------
def candidate_table(data: dict, cfg4: dict) -> list[str]:
    rows = []
    cm = data["control"]["val"]
    if cm is not None:
        rows.append([f"{data['control']['candidate']} (SFT control, no DPO)", "-", "-", "2 (SFT)", "-"] + sb.metric_cells(cm)
                    + [str(median_chars(cm)), "control"])
    dropped = {d["candidate"]: d["reasons"] for d in data["dropped"]}
    for c in data["candidates"]:
        if data["winner"] and c["candidate"] == data["winner"]["candidate"]:
            status, name = "**WINNER**", f"**{c['candidate']}**"
        elif c["candidate"] in dropped:
            status, name = "dropped: " + "; ".join(dropped[c["candidate"]]), c["candidate"]
        else:
            status, name = f"eligible, rank {data['ranked'].index(c['candidate']) + 1}", c["candidate"]
        te = c["train_epoch"] or {}
        rows.append([name, f"{c['beta']:g}", f"{c['lr']:g}", str(c["epoch"]),
                     "n/a" if te.get("dpo_loss") is None else f"{te['dpo_loss']:.3f}"] + sb.metric_cells(c["metrics"])
                    + [str(median_chars(c["metrics"])), status])
    return sb.table(["candidate", "beta", "lr", "epoch", "train DPO loss"] + sb.METRIC_HEADER + ["median output chars", "status"], rows)


def training_stats_table(data: dict) -> list[str]:
    rows = []
    for r in data["runs"]:
        c = r["cost"]
        for e, s in sorted(((c or {}).get("epochs") or {}).items()):
            rows.append([r["name"], e, "n/a" if s["dpo_loss"] is None else f"{s['dpo_loss']:.4f}",
                         "n/a" if s["reward_accuracy"] is None else f"{s['reward_accuracy']:.3f}",
                         "n/a" if s["mean_margin"] is None else f"{s['mean_margin']:.3f}"])
    return sb.table(["run", "epoch", "DPO loss (train, mean over the epoch)", "reward accuracy", "mean margin"], rows) if rows else ["not run"]


def cost_table(runs: list[tuple[str, dict | None]]) -> list[str]:
    rows = []
    for name, c in runs:
        if not c:
            rows.append([name, "not run"] + [""] * 7)
            continue
        pc = c.get("pair_counts") or {}
        rows.append([name, "complete" if c["complete"] else "NOT complete", sb._fmt(c["peak_gpu_mem_mb"], ".0f") + " MB",
                     sb._fmt(c["train_seconds"], ".0f") + " s",
                     f"{c['trainable_params']:,}" if c["trainable_params"] else "n/a", sb._fmt(c["adapter_mb"], ".1f") + " MB",
                     f"{len(c.get('dropped_distinct_ids') or [])} distinct / {c.get('dropped_lines')} lines",
                     "n/a" if c.get("skipped_steps") is None else str(c["skipped_steps"]),
                     f"{pc.get('distinct', 'n/a')} / {pc.get('lines', 'n/a')}"])
    return sb.table(["run", "state", "peak GPU memory", "training time", "trainable parameters", "adapter size (one epoch)",
                     "pairs dropped (too long)", "skipped optimiser steps", "pairs used: distinct / lines"], rows)


def paired_table(p: dict) -> list[str]:
    rows = []
    for r in p["rows"]:
        d, s = r["dpo"], r["sft"]
        rows.append([str(r["seed"]), f"{s['false_refusals']['count']}/{s['false_refusals']['n']}" if s else "n/a",
                     f"{d['false_refusals']['count']}/{d['false_refusals']['n']}" if d else "not yet measured",
                     f"{s['correct_strict']['count']}/{s['correct_strict']['n']}" if s else "n/a",
                     f"{d['correct_strict']['count']}/{d['correct_strict']['n']}" if d else "not yet measured",
                     f"{s['token_f1']['mean']:.3f}" if s else "n/a", f"{d['token_f1']['mean']:.3f}" if d else "not yet measured"])
    return sb.table(["seed", "false refusals, SFT", "false refusals, DPO", "strict correct refusals, SFT",
                     "strict correct refusals, DPO", "token F1, SFT", "token F1, DPO"], rows)


def selection_md(data: dict, cfg4: dict) -> str:
    sel = cfg4["selection"]
    L = ["# Phase 4 selection: DPO on top of the SFT model (validation set only)", "",
         "Built by `python -m climate_slm.training.select_dpo` from saved files only. Rule fixed in `configs/phase4.yaml` "
         "before any run:", "",
         f"- filter: strict correct refusals >= {sel['min_correct_refusals_val']}/11; median output length <= "
         f"{sel['max_median_length_ratio']} x the SFT control; at most {sel['max_hit_max_new_tokens']} output at max_new_tokens",
         "- then: 1. fewest false refusals, 2. more strict correct refusals, 3. higher answerable token F1, 4. higher number "
         "faithfulness, 5. cheaper (lower peak GPU memory, then shorter time), 6. earlier epoch",
         f"- if no candidate has fewer false refusals than the SFT control seed {sel['control_seed']} on val "
         f"({data['control_false_refusals']}/42), the result is \"{NO_IMPROVEMENT}\" and the test step is NOT run", "",
         "## All candidates (4 runs x 3 epochs) and the SFT control", ""]
    L += candidate_table(data, cfg4) + [""]
    if data["missing_runs"]:
        L += ["Runs with fewer than 3 evaluated epochs: " + "; ".join(
            f"{m['run']} (epochs found {m['evaluated_epochs']})" for m in data["missing_runs"]), ""]
    L += ["## Result", ""]
    if data["winner"]:
        w = data["winner"]
        L += [f"**{w['candidate']}** (beta {w['beta']:g}, learning rate {w['lr']:g}, epoch {w['epoch']}) has fewer false refusals "
              f"on val than the SFT control ({data['control_false_refusals']}/42).", ""]
        if data["runner_up"]:
            L += [f"Runner-up: {data['runner_up']}. The rule step that decided: **{data['deciding_step']}**.", ""]
        wm = next(c for c in data["candidates"] if c["candidate"] == w["candidate"])["metrics"]
        L += ["Against the SFT control (95% intervals; val has 42 answerable and 11 refusal items):", ""]
        L += [f"- {x}" for x in vs_control_lines(wm, data["control"]["val"], w["candidate"])] + [""]
        if data["closeness"]:
            L += ["Against the next candidates:", ""] + [f"- {x}" for x in data["closeness"]] + [""]
        L += ["In plain words: where a line says the intervals overlap, the data cannot show a difference. A one- or two-item "
              "change in 11 refusal or 42 answerable items is not established.", ""]
    else:
        L += [f"**{data['status']}.** There is no winner, no seed repeat and no test run. The report must say that SFT stays.", ""]
    if data["dropped"]:
        L += ["Dropped by the filter: " + "; ".join(f"{d['candidate']} ({', '.join(d['reasons'])})" for d in data["dropped"]), ""]
    L += ["## DPO training numbers per epoch (from the training logs)", ""] + training_stats_table(data) + [""]
    L += ["## Training cost per run", ""]
    runs = [(r["name"], r["cost"]) for r in data["runs"]] + sorted(data["repeat_costs"].items())
    L += cost_table(runs) + [""]
    L += ["## Paired seeds on val: SFT then DPO (winning configuration, same epoch)", ""]
    if data["paired"]:
        L += paired_table(data["paired"]) + [""] + [f"- {x}" for x in consistency_lines(data["paired"])] + [""]
    else:
        L += ["no winner, so no seed repeat", ""]
    return "\n".join(L)


def payload(data: dict) -> dict:
    """What is frozen: the validation selection. Test results are NOT part of it."""
    out = json.loads(json.dumps(data))
    out["control"].pop("test", None)
    if out.get("tuned"):
        out["tuned"].pop("test", None)
    out["baselines"] = {"val": out["baselines"]["val"]}
    return out


def test_command() -> str:
    return "python -m climate_slm.training.dpo_final_test"


def criteria_lines(m: dict | None, split: str, base: dict | None, cfg3: dict, label: str, test_runs: int | None) -> list[str]:
    if m is None:
        return [f"**{label}, {split}:** not yet measured", ""]
    rows = sb.criteria_rows(m, split, base, cfg3, test_runs)
    for r in rows:
        if r["id"] == 7:
            r.update(text="criterion 7 (SFT checks before DPO)", ok=None, value="not applicable to a DPO model")
    ok = {True: "yes", False: "NO", None: "n/a"}
    return [f"**{label}, {split}**", ""] + sb.table(["#", "criterion", "met", "value"],
                                                    [[str(r["id"]), r["text"], ok[r["ok"]], r["value"]] for r in rows]) + [""]


def question_lines(data: dict, cfg3: dict) -> list[str]:
    """The plain-words question of the brief, answered from the numbers (test false refusals / correct refusals, seeds)."""
    t = data["tuned"]
    if not t or t["test"] is None:
        return ["not yet measured (no test run)"]
    c = cfg3["criteria"]
    fr, cr = t["test"]["false_refusals"], t["test"]["correct_strict"]
    fr_ok, cr_ok = fr["count"] <= c["max_false_refusals"]["test"], cr["count"] >= c["min_correct_refusals"]
    lines = [f"- Did DPO bring false refusals on test to <= {c['max_false_refusals']['test']}/{fr['n']}? "
             f"{'YES' if fr_ok else 'NO'} ({fr['count']}/{fr['n']}).",
             f"- Without losing strict correct refusals below {c['min_correct_refusals']}/{cr['n']}? "
             f"{'YES' if cr_ok else 'NO'} ({cr['count']}/{cr['n']})."]
    s = data["control"]["test"]
    if s:
        lines += [f"- Against the SFT control on test: {x}" for x in vs_control_lines(t["test"], s, "DPO")]
    if data["paired"]:
        lines += [f"- Paired seeds on val: {x}" for x in consistency_lines(data["paired"])[:1]]
    lines.append("- DPO is NOT claimed to be better than SFT unless the intervals above do not overlap.")
    return lines


def final_md(data: dict, cfg4: dict, cfg3: dict, test_runs: dict[str, int], sha: str | None,
             title: str = "Phase 4 final tables (validation and test)") -> str:
    L = [f"# {title}", "",
         f"Selection file sha256 (frozen before the test run): `{sha}`" if sha else "Selection file is NOT frozen.", ""]
    t, ctl, b = data["tuned"], data["control"], data["baselines"]
    L += ["## SFT control, DPO winner and the base (val, then test once)", ""]
    for split in SPLITS:
        rows = [[lab] + sb.metric_cells(b[split].get(n)) for n, lab in cfg3["tables"]["baselines"]]
        rows.append([f"{ctl['candidate']} (SFT control, no DPO)"] + sb.metric_cells(ctl[split]))
        if t:
            rows.append([f"DPO winner {t['winner']['candidate']}"] + sb.metric_cells(t[split]))
        L += [f"#### {split}", ""] + sb.table(["model"] + sb.METRIC_HEADER, rows) + [""]
    L += ["## The pass criteria (1 to 6; 7 not applicable)", ""]
    if t:
        for split in SPLITS:
            L += criteria_lines(t[split], split, b[split].get(cfg3["tables"]["base_reference"]), cfg3,
                                f"DPO winner {t['winner']['candidate']}", test_runs.get(t["winner"]["candidate"], 0))
    else:
        L += ["No DPO winner: " + data["status"], ""]
    L += ["## The question that matters", ""] + question_lines(data, cfg3) + [""]
    if t:
        prompted = {n: b["test"][n] for n in cfg3["tables"]["prompted_baselines"] if n in b["test"]}
        if t["test"]:
            L += ["## Against the prompted baselines (test)", ""] + [
                f"- {x}" for x in sb.prompted_statements(t["test"], "test", prompted, "DPO winner")] + [""]
    if data["paired"]:
        L += ["## Paired seeds on val: SFT then DPO", ""] + paired_table(data["paired"]) + [""]
    L += ["## Training cost per run", ""] + cost_table([(r["name"], r["cost"]) for r in data["runs"]]
                                                        + sorted(data["repeat_costs"].items())) + [""]
    return "\n".join(L)


# ---- command line --------------------------------------------------------------------------------------------------------------
def read_test_runs(cfg4: dict) -> dict[str, int]:
    p = REPO_ROOT / cfg4["paths"]["test_log"]
    if not p.exists():
        return {}
    counts: dict[str, int] = {}
    for e in json.loads(p.read_text(encoding="utf-8")):
        counts[e["candidate"]] = counts.get(e["candidate"], 0) + 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply the Phase 4 selection rule to the saved val evaluations.")
    ap.add_argument("--freeze", action="store_true", help="store the sha256 of the selection file")
    ap.add_argument("--final-tables", action="store_true", help="after the test run: write phase4_final.md")
    a = ap.parse_args(argv)
    cfg4, cfg3, eval_cfg = load_phase4_config(), load_phase3_config(), load_eval_config()
    sp, mp = selection_io.selection_path(cfg4), REPO_ROOT / cfg4["paths"]["selection_md"]
    data = collect(cfg4, cfg3, eval_cfg)
    if a.final_tables:
        if not selection_io.is_frozen(cfg4):
            raise SystemExit("The selection is not frozen. Run select_dpo --freeze first.")
        sha = selection_io.sha_path(cfg4).read_text(encoding="utf-8").split()[0]
        out = REPO_ROOT / cfg4["paths"]["final_md"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(final_md(data, cfg4, cfg3, read_test_runs(cfg4), sha), encoding="utf-8")
        print("wrote", out)
        return 0
    if selection_io.is_frozen(cfg4):
        print(f"The selection is frozen ({sp.name}, sha256 {selection_io.file_sha256(sp)}). Not rewritten.")
    else:
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(payload(data), indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        mp.write_text(selection_md(data, cfg4), encoding="utf-8")
        print("wrote", sp, "and", mp)
    if a.freeze:
        if not selection_io.is_frozen(cfg4):
            sha = selection_io.file_sha256(sp)
            selection_io.sha_path(cfg4).write_text(f"{sha}  {sp.name}\n", encoding="utf-8")
        print("frozen: sha256", selection_io.file_sha256(sp))
    w = data["winner"]
    print("DPO winner:", w["candidate"] if w else f"none ({data['status']})")
    if w:
        print("test command (run ONCE, after --freeze):", test_command())
    return 0


if __name__ == "__main__":
    sys.exit(main())
