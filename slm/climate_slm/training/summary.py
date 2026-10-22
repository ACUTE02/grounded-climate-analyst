"""Numbers out of saved eval JSON files. Pure Python: no model, no torch. Used by the selection step and the
MLflow logging, so both read the same values."""
import json
from collections import Counter
from pathlib import Path

from climate_slm.evaluation import metrics as M
from climate_slm.evaluation import scoring


def load_eval(path: Path, eval_cfg: dict | None = None) -> dict | None:
    """The eval dict, or None if the file does not exist. Old files get key_recall / key_support in memory."""
    path = Path(path)
    if not path.exists():
        return None
    ev = json.loads(path.read_text(encoding="utf-8"))
    if eval_cfg is not None:
        scoring.add_key_scores(ev, eval_cfg)
    return ev


def has_repetition_loop(text: str, ngram: int, min_repeats: int) -> bool:
    """True if one sequence of `ngram` words occurs at least `min_repeats` times in the output."""
    toks = M.tokens(text)
    if len(toks) < ngram:
        return False
    counts = Counter(tuple(toks[i:i + ngram]) for i in range(len(toks) - ngram + 1))
    return max(counts.values()) >= min_repeats


def _mean(block: dict | None) -> dict | None:
    if not block:
        return None
    return {"mean": block["mean"], "ci95": block["ci95"], "n": block["n"]}


def _rate(block: dict) -> dict:
    return {"count": block["count"], "n": block["n"], "rate": block["rate"], "wilson95": block["wilson95"]}


def extract_metrics(ev: dict, loop_ngram: int, loop_min_repeats: int) -> dict:
    """The values every table and rule uses, from one eval dict."""
    s = ev["summary"]
    a, r = s["answerable"], s["refusal"]
    nf = a["number_faithfulness"]
    ppl = (s.get("perplexity") or {}).get("refusal")
    hit = s.get("hit_max_new_tokens")
    return {
        "n_answerable": a["n"], "n_refusal": r["n"],
        "token_f1": _mean(a["token_f1"]),
        "key_recall": _mean(a.get("key_recall")), "key_support": _mean(a.get("key_support")),
        "number_faithfulness": {"mean": nf["mean"], "ci95": nf["ci95"], "n_applied": nf["n_applied"]},
        "false_refusals": _rate(a["false_refusal_strict"]),
        "correct_strict": _rate(r["correct_refusal_strict"]), "correct_loose": _rate(r["correct_refusal_loose"]),
        "answered_when_should_refuse": _rate(r["answered_when_should_refuse"]),
        "refusal_perplexity": ppl["perplexity"] if ppl else None,
        "hit_max_new_tokens": hit["count"] if hit else 0,
        "repetition_loops": sum(has_repetition_loop(it["output"], loop_ngram, loop_min_repeats) for it in ev["items"]),
        "output_chars": s.get("output_length_chars"),
    }


def compare(a_mean: float | None, a_ci, b_mean: float | None, b_ci, tol: float = 1e-9) -> dict:
    """a against b: direction and whether the 95% intervals overlap. Overlap means: no difference can be claimed."""
    if a_mean is None or b_mean is None:
        return {"direction": None, "overlap": None}
    if abs(a_mean - b_mean) <= tol:
        return {"direction": "equal", "overlap": True}
    overlap = None
    if a_ci and b_ci:
        overlap = not (a_ci[1] < b_ci[0] or b_ci[1] < a_ci[0])
    return {"direction": "higher" if a_mean > b_mean else "lower", "overlap": overlap}


def flat_metrics(m: dict, prefix: str = "val_") -> dict[str, float]:
    """Flat name -> number dict for MLflow (None values are left out)."""
    out = {
        f"{prefix}token_f1": (m["token_f1"] or {}).get("mean"),
        f"{prefix}key_recall": (m["key_recall"] or {}).get("mean") if m["key_recall"] else None,
        f"{prefix}key_support": (m["key_support"] or {}).get("mean") if m["key_support"] else None,
        f"{prefix}number_faithfulness": m["number_faithfulness"]["mean"],
        f"{prefix}correct_refusals_strict": m["correct_strict"]["count"],
        f"{prefix}correct_refusals_loose": m["correct_loose"]["count"],
        f"{prefix}false_refusals_strict": m["false_refusals"]["count"],
        f"{prefix}answered_when_should_refuse": m["answered_when_should_refuse"]["count"],
        f"{prefix}refusal_perplexity": m["refusal_perplexity"],
        f"{prefix}hit_max_new_tokens": m["hit_max_new_tokens"],
        f"{prefix}repetition_loops": m["repetition_loops"],
    }
    if m["output_chars"]:
        out[f"{prefix}output_chars_median"] = m["output_chars"]["median"]
        out[f"{prefix}output_chars_max"] = m["output_chars"]["max"]
    return {k: float(v) for k, v in out.items() if v is not None}
