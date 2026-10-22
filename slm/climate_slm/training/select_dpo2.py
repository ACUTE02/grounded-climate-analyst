"""Phase 4b selection (DPO v2). Reads ONLY saved files (eval JSON of the evaluator, train_log.json of the runs).

    python -m climate_slm.training.select_dpo2                 # write outputs/phase4b_selection.md and .json
    python -m climate_slm.training.select_dpo2 --freeze        # also store the sha256 of the selection file
    python -m climate_slm.training.select_dpo2 --final-tables  # after the one test run: outputs/phase4b_final.md

The v1 rule (select_dpo.py) plus the v2 lines of docs/PHASE_4b_DPO_v2.md, applied on VAL only to the 24 candidates
(8 runs x 3 epochs), all fixed in configs/phase4b.yaml before any run. A candidate is removed before ranking if
  * strict correct refusals < 7/11, or
  * median output length > 1.5 x the SFT control, or more than 1 output at max_new_tokens, or more repetition-loop
    outputs than the SFT control, or
  * answerable token F1 < 0.575.
The rest is ranked: 1. fewest false refusals, 2. more strict correct refusals, 3. higher token F1, 4. higher number
faithfulness, 5. cheaper (lower peak GPU memory, then shorter time), 6. earlier epoch. The first one only wins if it has FEWER
false refusals than the SFT control seed 42 on val (3/42). Otherwise: "no DPO v2 candidate improves on the SFT model on val",
no winner, no test, SFT stays. Total candidates tried on val across v1 and v2: 12 + 24 = 36: val is no longer a clean check.
Once the selection file is frozen it is not rewritten."""
import argparse
import json
import sys

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase3_config
from climate_slm.training import dpo2_grid, dpo_grid, grid, selection_io
from climate_slm.training import select_best as sb
from climate_slm.training import select_dpo as v1

NO_IMPROVEMENT = "no DPO v2 candidate improves on the SFT model on val"
SPLITS = v1.SPLITS


# ---- loading ---------------------------------------------------------------------------------------------------------
def cost_of(cfg: dict, run_name: str) -> dict | None:
    """v1 cost numbers plus the v2 loss parts per epoch (total loss, DPO part, NLL part)."""
    c = v1.cost_of(cfg, run_name)
    if c is None:
        return None
    log = json.loads(grid.run_log_path(cfg, run_name).read_text(encoding="utf-8"))
    for e in log.get("epochs") or []:
        c["epochs"][str(e["epoch"])].update({"loss": e.get("loss"), "nll": e.get("nll")})
    return c


def run_candidates(cfg: dict, cfg3: dict, eval_cfg: dict, run: dict, split: str = "val") -> list[dict]:
    cost, out = cost_of(cfg, run["name"]), []
    for e in grid.epochs(cfg):
        name = grid.candidate_name(run["name"], e)
        m = sb.load_metrics(eval_cfg, cfg3, name, split)
        if m is not None:
            out.append({"candidate": name, "run": run["name"], "epoch": e, "pair_set": run["pair_set"], "alpha": run["alpha"],
                        "beta": run["beta"], "lr": run["lr"], "seed": run["seed"], "metrics": m, "cost": cost,
                        "train_epoch": (cost or {}).get("epochs", {}).get(str(e))})
    return out


# ---- the rule --------------------------------------------------------------------------------------------------------------
def drop_reasons(c: dict, control: dict, sel: dict) -> list[str]:
    """Why a candidate is removed before ranking ([] = it stays)."""
    m, reasons = c["metrics"], []
    cs = m["correct_strict"]
    if cs["count"] < sel["min_correct_refusals_val"]:
        reasons.append(f"strict correct refusals {cs['count']}/{cs['n']} < {sel['min_correct_refusals_val']}")
    mc, cc = v1.median_chars(m), v1.median_chars(control)
    if mc is not None and cc is not None and mc > sel["max_median_length_ratio"] * cc:
        reasons.append(f"median output length {mc:g} chars > {sel['max_median_length_ratio']} x control {cc:g}")
    if m["hit_max_new_tokens"] > sel["max_hit_max_new_tokens"]:
        reasons.append(f"{m['hit_max_new_tokens']} outputs at max_new_tokens (> {sel['max_hit_max_new_tokens']})")
    if sel["no_repetition_loop_beyond_control"] and m["repetition_loops"] > control["repetition_loops"]:
        reasons.append(f"{m['repetition_loops']} repetition-loop outputs (control {control['repetition_loops']})")
    f1 = m["token_f1"]["mean"]
    if f1 < sel["min_token_f1_val"]:
        reasons.append(f"answerable token F1 {f1:.3f} < {sel['min_token_f1_val']}")
    return reasons


def select_winner(cands: list[dict], control: dict | None, cfg: dict) -> dict:
    """Apply the v2 rule to the val candidates. `control` = val metrics of the SFT control (seed 42)."""
    sel = cfg["selection"]
    out = {"n_candidates": len(cands), "dropped": [], "ranked": [], "winner": None, "runner_up": None,
           "deciding_step": None, "closeness": [], "improves_on_control": False, "status": None,
           "control_false_refusals": None}
    if control is None:
        out["status"] = "the SFT control val file is missing: nothing can be decided"
        return out
    fr0 = control["false_refusals"]["count"]
    out["control_false_refusals"] = fr0
    if not cands:
        out["status"] = "not yet measured: no DPO v2 candidate has been evaluated on val"
        return out
    kept = []
    for c in cands:
        why = drop_reasons(c, control, sel)
        if why:
            out["dropped"].append({"candidate": c["candidate"], "reasons": why})
        else:
            kept.append(c)
    ranked = sorted(kept, key=lambda c: v1.rank_key(c, sel)[0])
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
    out["winner"] = {k: best[k] for k in ("candidate", "run", "epoch", "pair_set", "alpha", "beta", "lr", "seed")}
    out["status"] = "winner found"
    if len(ranked) > 1:
        r = ranked[1]
        out["runner_up"] = r["candidate"]
        pw, pr = v1.rank_key(best, sel)[1], v1.rank_key(r, sel)[1]
        out["deciding_step"] = next((k for k in sel["order"] if pw[k] != pr[k]), None)
        for other in ranked[1:3]:
            out["closeness"] += sb.closeness(best, other)
    return out


# ---- collecting everything ---------------------------------------------------------------------------------------------
def collect(cfg: dict, cfg3: dict, eval_cfg: dict) -> dict:
    data = {"runs": [], "candidates": [], "missing_runs": []}
    for r in dpo2_grid.grid_runs(cfg):
        cands = run_candidates(cfg, cfg3, eval_cfg, r)
        data["runs"].append({"name": r["name"], "pair_set": r["pair_set"], "alpha": r["alpha"], "lr": r["lr"], "cost": cost_of(cfg, r["name"])})
        data["candidates"] += cands
        if len(cands) < len(grid.epochs(cfg)):
            data["missing_runs"].append({"run": r["name"], "evaluated_epochs": [c["epoch"] for c in cands]})
    seed0 = cfg["selection"]["control_seed"]
    data["control"] = {"candidate": dpo_grid.control_name(cfg, seed0), "val": v1.control_metrics(cfg, cfg3, eval_cfg, seed0),
                       "test": v1.control_metrics(cfg, cfg3, eval_cfg, seed0, "test")}
    data.update(select_winner(data["candidates"], data["control"]["val"], cfg))
    data["val_candidates_tried"] = cfg["design"]["val_candidates_tried"]
    w = data["winner"]
    data["paired"] = (v1.paired_seeds(cfg, cfg3, eval_cfg, w, lambda sd: dpo2_grid.make_run(cfg, w["pair_set"], w["alpha"], w["lr"], sd))
                      if w else None)
    data["repeat_costs"] = {r["name"]: cost_of(cfg, r["name"]) for r in dpo2_grid.repeat_runs(cfg, w)} if w else {}
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
def candidate_table(data: dict) -> list[str]:
    rows = []
    cm = data["control"]["val"]
    if cm is not None:
        rows.append([f"{data['control']['candidate']} (SFT control, no DPO)", "-", "-", "-", "2 (SFT)"] + sb.metric_cells(cm)
                    + [str(v1.median_chars(cm)), "control"])
    dropped = {d["candidate"]: d["reasons"] for d in data["dropped"]}
    for c in data["candidates"]:
        if data["winner"] and c["candidate"] == data["winner"]["candidate"]:
            status, name = "**WINNER**", f"**{c['candidate']}**"
        elif c["candidate"] in dropped:
            status, name = "dropped: " + "; ".join(dropped[c["candidate"]]), c["candidate"]
        else:
            status, name = f"eligible, rank {data['ranked'].index(c['candidate']) + 1}", c["candidate"]
        rows.append([name, c["pair_set"], f"{c['alpha']:g}", f"{c['lr']:g}", str(c["epoch"])] + sb.metric_cells(c["metrics"])
                    + [str(v1.median_chars(c["metrics"])), status])
    return sb.table(["candidate", "pair set", "alpha", "lr", "epoch"] + sb.METRIC_HEADER + ["median output chars", "status"], rows)


def training_stats_table(data: dict) -> list[str]:
    rows = []
    for r in data["runs"]:
        for e, s in sorted(((r["cost"] or {}).get("epochs") or {}).items()):
            cells = [sb._fmt(s.get(k), ".4f") for k in ("loss", "dpo_loss", "nll", "reward_accuracy", "mean_margin")]
            rows.append([r["name"], e] + cells)
    return sb.table(["run", "epoch", "loss (train, epoch mean)", "DPO part", "NLL part (chosen, per token)", "reward accuracy",
                     "mean margin"], rows) if rows else ["not run"]


def selection_md(data: dict, cfg: dict) -> str:
    sel, tried = cfg["selection"], data["val_candidates_tried"]
    L = ["# Phase 4b selection: DPO v2 on top of the SFT model (validation set only)", "",
         "Built by `python -m climate_slm.training.select_dpo2` from saved files only. Rule fixed in `configs/phase4b.yaml` before "
         "any run:", "",
         f"- filter: strict correct refusals >= {sel['min_correct_refusals_val']}/11; median output length <= "
         f"{sel['max_median_length_ratio']} x the SFT control; at most {sel['max_hit_max_new_tokens']} output at max_new_tokens; "
         f"no more repetition-loop outputs than the SFT control; answerable token F1 >= {sel['min_token_f1_val']}",
         "- then: 1. fewest false refusals, 2. more strict correct refusals, 3. higher answerable token F1, 4. higher number "
         "faithfulness, 5. cheaper (lower peak GPU memory, then shorter time), 6. earlier epoch",
         f"- a candidate must have FEWER false refusals than the SFT control seed {sel['control_seed']} on val "
         f"({data['control_false_refusals']}/42); otherwise the result is \"{NO_IMPROVEMENT}\" and the test step is NOT run", "",
         f"**Candidates tried on val so far: v1 {tried['v1']} + v2 {tried['v2']} = {tried['total']}. Val is no longer a clean "
         "check: the more candidates are looked at on the same 42 + 11 items, the more likely one of them looks good by chance.**", "",
         "## All candidates (8 runs x 3 epochs) and the SFT control", ""]
    L += candidate_table(data) + [""]
    if data["missing_runs"]:
        L += ["Runs with fewer than 3 evaluated epochs: " + "; ".join(
            f"{m['run']} (epochs found {m['evaluated_epochs']})" for m in data["missing_runs"]), ""]
    L += ["## Result", ""]
    if data["winner"]:
        w = data["winner"]
        L += [f"**{w['candidate']}** (pair set {w['pair_set']}, alpha {w['alpha']:g}, learning rate {w['lr']:g}, epoch {w['epoch']}) "
              f"has fewer false refusals on val than the SFT control ({data['control_false_refusals']}/42).", ""]
        if data["runner_up"]:
            L += [f"Runner-up: {data['runner_up']}. The rule step that decided: **{data['deciding_step']}**.", ""]
        wm = next(c for c in data["candidates"] if c["candidate"] == w["candidate"])["metrics"]
        L += ["Against the SFT control (95% intervals; val has 42 answerable and 11 refusal items):", ""]
        L += [f"- {x}" for x in v1.vs_control_lines(wm, data["control"]["val"], w["candidate"])] + [""]
        if data["closeness"]:
            L += ["Against the next candidates:", ""] + [f"- {x}" for x in data["closeness"]] + [""]
        L += ["In plain words: where a line says the intervals overlap, the data cannot show a difference. A one- or two-item "
              "change in 11 refusal or 42 answerable items is not established.", ""]
    else:
        if data["status"].startswith("not yet measured"):
            L += [f"**{data['status']}.** Nothing can be decided yet.", ""]
        else:
            L += [f"**{data['status']}.** There is no winner, no seed repeat and no test run. The report must say that SFT stays.", ""]
    if data["dropped"]:
        L += ["Dropped by the filter: " + "; ".join(f"{d['candidate']} ({', '.join(d['reasons'])})" for d in data["dropped"]), ""]
    L += ["## DPO v2 training numbers per epoch (from the training logs)", ""] + training_stats_table(data) + [""]
    L += ["## Training cost per run", ""]
    runs = [(r["name"], r["cost"]) for r in data["runs"]] + sorted(data["repeat_costs"].items())
    L += v1.cost_table(runs) + [""]
    L += ["## Paired seeds on val: SFT then DPO v2 (winning configuration, same epoch)", ""]
    if data["paired"]:
        L += v1.paired_table(data["paired"]) + [""] + [f"- {x}" for x in v1.consistency_lines(data["paired"])] + [""]
    else:
        L += ["no winner, so no seed repeat", ""]
    return "\n".join(L)


def payload(data: dict) -> dict:
    return v1.payload(data)


def test_command() -> str:
    return "python -m climate_slm.training.dpo2_final_test"


# ---- command line --------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply the Phase 4b (DPO v2) selection rule to the saved val evaluations.")
    ap.add_argument("--freeze", action="store_true", help="store the sha256 of the selection file")
    ap.add_argument("--final-tables", action="store_true", help="after the test run: write phase4b_final.md")
    a = ap.parse_args(argv)
    cfg, cfg3, eval_cfg = dpo2_grid.runtime_config(), load_phase3_config(), load_eval_config()
    sp, mp = selection_io.selection_path(cfg), REPO_ROOT / cfg["paths"]["selection_md"]
    data = collect(cfg, cfg3, eval_cfg)
    if a.final_tables:
        if not selection_io.is_frozen(cfg):
            raise SystemExit("The selection is not frozen. Run select_dpo2 --freeze first.")
        sha = selection_io.sha_path(cfg).read_text(encoding="utf-8").split()[0]
        out = REPO_ROOT / cfg["paths"]["final_md"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(v1.final_md(data, cfg, cfg3, v1.read_test_runs(cfg), sha, "Phase 4b (DPO v2) final tables (validation and test)"),
                       encoding="utf-8")
        print("wrote", out)
        return 0
    if selection_io.is_frozen(cfg):
        print(f"The selection is frozen ({sp.name}, sha256 {selection_io.file_sha256(sp)}). Not rewritten.")
    else:
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(payload(data), indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        mp.write_text(selection_md(data, cfg), encoding="utf-8")
        print("wrote", sp, "and", mp)
    if a.freeze:
        if not selection_io.is_frozen(cfg):
            sha = selection_io.file_sha256(sp)
            selection_io.sha_path(cfg).write_text(f"{sha}  {sp.name}\n", encoding="utf-8")
        print("frozen: sha256", selection_io.file_sha256(sp))
    w = data["winner"]
    print("DPO v2 winner:", w["candidate"] if w else f"none ({data['status']})")
    if w:
        print("test command (run ONCE, after --freeze):", test_command())
    return 0


if __name__ == "__main__":
    sys.exit(main())
