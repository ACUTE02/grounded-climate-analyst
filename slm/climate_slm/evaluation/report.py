"""Build the result table and the reading files from every outputs/eval_*.json. No number is typed by hand.

    python -m climate_slm.evaluation.report

Writes: outputs/baseline_table.md, outputs/qualitative_<name>.md (test split, greedy runs),
outputs/refusal_detection_review_<name>_<split>.csv. Smoke files made with --limit are left out.
key_recall / key_support are computed here for eval files written before Phase 2c (the files are not changed).
Runs with a sampling decoding mode (base_qwen_s1 ...) never appear in the main rows; they have their own section."""
import argparse
import fnmatch
import json
import random
from pathlib import Path

from climate_slm.config import REPO_ROOT, load_eval_config
from climate_slm.evaluation import scoring


def pick_qualitative_ids(items: list[dict], seed: int, n_answerable: int, n_refusal: int) -> list[str]:
    """Seeded pick of fixed ids (done once, before any model output). Returns sorted ids."""
    rng = random.Random(seed)
    ans = sorted(i["meta"]["id"] for i in items if scoring.kind_of(i) == "answerable")
    ref = sorted(i["meta"]["id"] for i in items if scoring.kind_of(i) == "refusal")
    return sorted(rng.sample(ans, n_answerable) + rng.sample(ref, n_refusal))


def is_sampled(ev: dict) -> bool:
    """True for a run made with a sampling decoding mode (old files have no decoding_mode = greedy)."""
    return ev.get("decoding_mode", "greedy") != "greedy"


def is_hidden(ev: dict, cfg: dict) -> bool:
    """Phase 3 sweep runs have their own tables (outputs/phase3_selection.md)."""
    return any(fnmatch.fnmatch(ev["run_name"], p) for p in cfg["report"].get("hide_run_patterns", []))


def load_evals(out_dir: Path, cfg: dict | None = None) -> tuple[list[dict], list[str]]:
    evals, skipped = [], []
    for p in sorted(Path(out_dir).glob("eval_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("split") == "train":                    # untuned base on the train split: input of the Phase 4 pairs
            skipped.append(f"{p.name} (train split, used only to build the Phase 4 preference pairs)")
            continue
        if d.get("limit"):
            skipped.append(f"{p.name} (smoke run, limit={d['limit']})")
            continue
        d["_file"] = p.name
        if cfg is not None:
            scoring.add_key_scores(d, cfg)
        evals.append(d)
    return evals, skipped


def _num(x, dec: int) -> str:
    return "n/a" if x is None else f"{x:.{dec}f}"


def _range(ci, dec: int) -> str:
    return "n/a" if not ci else f"[{_num(ci[0], dec)}, {_num(ci[1], dec)}]"


def mean_cell(b: dict, dec: int) -> str:
    if not b or not b["n"]:
        return "n/a (n=0)"
    return f"{_num(b['mean'], dec)} {_range(b['ci95'], dec)} (n={b['n']})"


def rate_cell(b: dict, dec: int) -> str:
    if not b or not b["n"]:
        return "n/a (n=0)"
    return f"{b['count']}/{b['n']} = {_num(b['rate'], dec)} {_range(b['ci95'], dec)}; Wilson {_range(b['wilson95'], dec)}"


def key_cell(b: dict, dec: int) -> str:
    if not b or not b["n"]:
        return "n/a (n=0)"
    return f"{_num(b['mean'], dec)} {_range(b['ci95'], dec)} (n={b['n']} of {b['n_answerable']})"


def ppl_cell(b: dict | None, dec: int) -> str:
    if not b:
        return "n/a (no model)"
    return f"{_num(b['perplexity'], dec)} {_range(b['ci95'], dec)} (n={b['n']}, {b['n_tokens']} tokens)"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def build_table(evals: list[dict], skipped: list[str], cfg: dict) -> str:
    rp = cfg["report"]
    dec, missing = rp["decimals"], rp["not_measured_text"]
    labels = {r["name"]: r["label"] for r in rp["expected_runs"]}
    lines = ["# Evaluation table", "",
             "Built by `python -m climate_slm.evaluation.report` from the `outputs/eval_*.json` files. "
             "No number is typed by hand.", "",
             "- Number cells: value [95% bootstrap interval over items] (n = items).",
             "- Rate cells: count/n = rate [95% bootstrap interval]; Wilson [95% Wilson interval]. The bootstrap "
             "interval has zero width when all items agree (for example 0 of 11), so the Wilson interval is shown too.",
             "- The sets are small. Intervals that overlap mean: no difference can be claimed.",
             "- Answerable and refusal items are always shown separately.", ""]
    splits = list(rp["splits"]) + sorted({e["split"] for e in evals} - set(rp["splits"]))
    for split in splits:
        by_name = {e["run_name"]: e for e in evals if e["split"] == split and not is_sampled(e) and not is_hidden(e, cfg)}
        names = list(labels) + sorted(n for n in by_name if n not in labels)
        some = next(iter(by_name.values()), None)
        n_ans = some["summary"]["answerable"]["n"] if some else "?"
        n_ref = some["summary"]["refusal"]["n"] if some else "?"
        lines += [f"## Split: {split}", ""]

        def rows(cells):
            out = []
            for n in names:
                e = by_name.get(n)
                label = labels.get(n, n)
                out.append([label] + (cells(e["summary"]) if e else [missing] * cells(None)))
            return out

        def block(title, header, fn):
            width = len(header) - 1
            return [f"### {title}", ""] + _table(header, rows(lambda s: fn(s) if s else width)) + [""]

        lines += block(
            f"Answerable items ({n_ans})",
            ["model", "token F1", "key recall", "key support", "ROUGE-L", "number faithfulness (items with numbers)",
             "false refusals, strict", "false refusals, loose"],
            lambda s: [mean_cell(s["answerable"]["token_f1"], dec), key_cell(s["answerable"]["key_recall"], dec),
                       key_cell(s["answerable"]["key_support"], dec), mean_cell(s["answerable"]["rouge_l"], dec),
                       mean_cell(s["answerable"]["number_faithfulness"], dec)
                       + f" of {s['answerable']['number_faithfulness']['n_answerable']}",
                       rate_cell(s["answerable"]["false_refusal_strict"], dec),
                       rate_cell(s["answerable"]["false_refusal_loose"], dec)])
        lines += block(
            f"Refusal items ({n_ref})",
            ["model", "correct refusals, strict", "correct refusals, loose", "answered when it should refuse"],
            lambda s: [rate_cell(s["refusal"]["correct_refusal_strict"], dec),
                       rate_cell(s["refusal"]["correct_refusal_loose"], dec),
                       rate_cell(s["refusal"]["answered_when_should_refuse"], dec)])
        lines += block(
            "Perplexity on the reference answer (completion tokens only) and output length",
            ["model", "perplexity, all", "perplexity, answerable", "perplexity, refusal",
             "output chars min / median / max", "outputs that hit max_new_tokens"],
            lambda s: [ppl_cell(s["perplexity"]["overall"], dec), ppl_cell(s["perplexity"]["answerable"], dec),
                       ppl_cell(s["perplexity"]["refusal"], dec),
                       (f"{s['output_length_chars']['min']} / {s['output_length_chars']['median']:g} / "
                        f"{s['output_length_chars']['max']}" if s["output_length_chars"] else "n/a"),
                       (f"{s['hit_max_new_tokens']['count']}/{s['hit_max_new_tokens']['n']}"
                        if s["hit_max_new_tokens"] else "n/a (no model)")])
        src = [[labels.get(n, n), e["_file"], str(e.get("model_id")), str(e.get("model_revision")),
                str(e.get("dtype")), e.get("prompt_variant", "default"), e.get("decoding_mode", "greedy"),
                e["data_sha256"][:12], e["timestamp_utc"]]
               for n in names if (e := by_name.get(n))]
        lines += ["### Source files", ""] + _table(
            ["model", "file", "model id", "revision", "dtype", "prompt variant", "decoding", "data sha256 (first 12)",
             "time (UTC)"], src) + [""]
    lines += build_decoding_section(evals, cfg, splits)
    if skipped:
        lines += ["Left out: " + "; ".join(skipped), ""]
    return "\n".join(lines)


def _spread(values: list[float]) -> tuple[float, float, float]:
    return sum(values) / len(values), min(values), max(values)


def _decoding_row(ev: dict, dec: int) -> list[float | int | None]:
    a, r = ev["summary"]["answerable"], ev["summary"]["refusal"]
    return [a["token_f1"]["mean"], a["key_recall"]["mean"], a["key_support"]["mean"],
            r["correct_refusal_strict"]["count"], r["correct_refusal_loose"]["count"], r["n"]]


def build_decoding_section(evals: list[dict], cfg: dict, splits: list[str]) -> list[str]:
    """Greedy row of the group model next to the sampled runs (qwen_default, several seeds): mean and min-max.
    Kept apart from the main comparison on purpose."""
    rp = cfg["report"]
    dec, missing, group = rp["decimals"], rp["not_measured_text"], rp["decoding_group"]
    lines = [f"## {rp['decoding_title']}", "",
             "Same model, same prompt, sampling decoding (`qwen_default`: temperature 0.7, top_p 0.8, top_k 20, "
             "repetition penalty 1.1) with several seeds, next to the greedy run that the main rows use. "
             "These runs are never mixed into the main comparison. With so few seeds no interval is given: "
             "min-max shows how far the random draws move the numbers.", ""]
    for split in splits:
        greedy = next((e for e in evals if e["split"] == split and e["run_name"] == group and not is_sampled(e)), None)
        seeds = sorted((e for e in evals if e["split"] == split and is_sampled(e)
                        and e["run_name"].startswith(group + "_")), key=lambda e: e.get("seed") or 0)
        lines += [f"### Split: {split}", ""]
        header = ["run", "answerable token F1", "key recall", "key support", "refusals correct, strict",
                  "refusals correct, loose"]

        def row(label, vals, n_ref, fmt):
            return [label, fmt(vals[0]), fmt(vals[1]), fmt(vals[2]), f"{fmt(vals[3], True)} of {n_ref}",
                    f"{fmt(vals[4], True)} of {n_ref}"]

        def f(v, count=False):
            if count and v is not None:                      # counts: whole numbers stay whole, a mean gets one decimal
                return str(int(v)) if float(v).is_integer() else f"{v:.1f}"
            return _num(v, dec)

        rows = []
        if greedy:
            g = _decoding_row(greedy, dec)
            rows.append(row(f"greedy ({group})", g, g[5], f))
        else:
            rows.append([f"greedy ({group})"] + [missing] * 5)
        if seeds:
            per = [_decoding_row(e, dec) for e in seeds]
            n_ref = per[0][5]
            for label, k in (("mean", 0), ("min", 1), ("max", 2)):
                vals = [_spread([p[i] for p in per if p[i] is not None])[k] if any(p[i] is not None for p in per)
                        else None for i in range(5)]
                rows.append(row(f"qwen_default, {label} over {len(seeds)} seeds", vals, n_ref, f))
        else:
            rows.append(["qwen_default, mean / min / max over seeds"] + [missing] * 5)
        lines += _table(header, rows) + [""]
        if seeds:
            det = [row(f"seed {e.get('seed')} ({e['_file']})", _decoding_row(e, dec), e["summary"]["refusal"]["n"], f)
                   for e in seeds]
            lines += ["Each seed:", ""] + _table(header, det) + [""]
    return lines


def build_qualitative(ev: dict, ids: list[str]) -> str | None:
    by_id = {r["id"]: r for r in ev["items"]}
    if not all(i in by_id for i in ids):
        return None
    lines = [f"# Qualitative examples: {ev['run_name']} ({ev['split']})", "",
             f"Source: `{ev['_file']}`. The {len(ids)} ids are fixed in `configs/phase2.yaml` "
             "(`eval.qualitative_ids`) and were picked with a seed before any model output existed.", ""]
    for i in ids:
        r = by_id[i]
        nf = "no numbers" if r["number_faithfulness"] is None else f"{r['number_faithfulness']:.2f}"
        lines += [f"## {i} ({r['kind']})", "", f"**Question:** {r['question']}", "",
                  f"**Reference:** {r['reference']}", "", f"**Model output:** {r['output']}", "",
                  f"token F1 {r['token_f1']:.3f}, ROUGE-L {r['rouge_l']:.3f}, number faithfulness {nf}, "
                  f"key recall {_num(r.get('key_recall'), 3)}, key support {_num(r.get('key_support'), 3)}, "
                  f"refusal strict {r['refusal_strict']}, refusal loose {r['refusal_loose']}", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the table and reading files from outputs/eval_*.json")
    ap.add_argument("--outputs", default=None)
    a = ap.parse_args(argv)
    cfg = load_eval_config()
    out_dir = Path(a.outputs) if a.outputs else REPO_ROOT / cfg["paths"]["outputs_dir"]
    evals, skipped = load_evals(out_dir, cfg)
    table = out_dir / Path(cfg["paths"]["baseline_table"]).name
    table.write_text(build_table(evals, skipped, cfg), encoding="utf-8")
    print("wrote", table, f"({len(evals)} eval files)")
    for ev in evals:
        if is_hidden(ev, cfg):
            continue
        n = scoring.write_review_csv(out_dir / f"refusal_detection_review_{ev['run_name']}_{ev['split']}.csv", ev["items"])
        print(f"review rows for {ev['run_name']} {ev['split']}: {n}")
        if ev["split"] == cfg["eval"]["qualitative_split"] and not is_sampled(ev):
            text = build_qualitative(ev, cfg["eval"]["qualitative_ids"])
            if text:
                (out_dir / f"qualitative_{ev['run_name']}.md").write_text(text, encoding="utf-8")
                print("wrote", f"qualitative_{ev['run_name']}.md")
    for s in skipped:
        print("left out:", s)


if __name__ == "__main__":
    main()
