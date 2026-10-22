"""Preference pairs for the DPO trainer, on plain lists (no torch, testable on the laptop).

dpo_train.jsonl has one line per WEIGHTED pair: the type 3 pairs appear 3 times (ids <row>:t3:r1..r3) with the same
`meta.distinct_pair_id`. The two answers of a pair are tokenised with the SAME chat template and the same completion-only
masking as the SFT training (prep.encode_row). The reference log-probabilities exist once per DISTINCT pair, and every
repeat reuses them."""
import json
from pathlib import Path

from climate_slm.config import REPO_ROOT
from climate_slm.training import prep
from climate_slm.training.masking import length_stats, pad_batch

SIDES = ("chosen", "rejected")


def read_pairs(cfg4: dict) -> list[dict]:
    """The lines of dpo_train.jsonl, after checking the file against its saved sha256."""
    return read_pair_file(REPO_ROOT / cfg4["paths"]["pairs"], REPO_ROOT / cfg4["paths"]["pairs_sha256"])


def read_pair_file(path: Path, sha_file: Path) -> list[dict]:
    """The lines of any pair file (v1 dpo_train.jsonl, v2 dpo2_main.jsonl / dpo2_mix.jsonl) after the sha256 check."""
    from climate_slm.training.dpo_grid import file_sha256
    if not path.exists() or not sha_file.exists():
        raise SystemExit(f"{path} or its sha256 file is missing. Build the pairs first (dpo_pairs) and rebuild the bundle.")
    if sha_file.read_text(encoding="utf-8").split()[0] != file_sha256(path):
        raise SystemExit(f"{path} does not match {sha_file.name}: the pair file was changed.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pairs_sha256(cfg4: dict) -> str:
    return (REPO_ROOT / cfg4["paths"]["pairs_sha256"]).read_text(encoding="utf-8").split()[0]


def distinct_pairs(pairs: list[dict]) -> dict[str, dict]:
    """distinct_pair_id -> the first line of that pair. Repeats must carry exactly the same text."""
    out: dict[str, dict] = {}
    for p in pairs:
        d = p["meta"]["distinct_pair_id"]
        if d not in out:
            out[d] = p
        elif any(out[d][k] != p[k] for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"repeats of {d} differ")
    return out


def count_pairs(pairs: list[dict]) -> dict:
    """Lines and distinct pairs, in total and per type (for the logs and MLflow)."""
    dist = distinct_pairs(pairs)
    out = {"lines": len(pairs), "distinct": len(dist)}
    for t in sorted({p["pair_type"] for p in pairs}):
        out[f"type{t}_lines"] = sum(1 for p in pairs if p["pair_type"] == t)
        out[f"type{t}_distinct"] = sum(1 for p in dist.values() if p["pair_type"] == t)
    return out


def side_row(pair: dict, side: str) -> dict:
    """One answer of a pair as an SFT-style row, so prep.encode_row builds the identical prompt and completion text."""
    return {"prompt": pair["prompt"], "completion": [{"role": "assistant", "content": pair[side]}],
            "meta": {"id": f"{pair['meta']['distinct_pair_id']}:{side}"}}


def encode_pairs(pairs: list[dict], template, tokenize, max_len: int, eos_token: str = "") -> dict:
    """Tokenise every distinct pair once. A pair of which either answer does not fit (prompt + answer > max_len) is
    dropped completely, never cut; all its repeats go with it.
    Returns {'examples': {distinct_id: {'chosen': ex, 'rejected': ex}}, 'lines': [kept lines, file order],
             'dropped_distinct_ids': [...], 'dropped_lines': n, 'stats': length stats of all sides}."""
    examples, dropped, lengths = {}, [], []
    for d, p in distinct_pairs(pairs).items():
        enc = {}
        for side in SIDES:
            ex, n = prep.encode_row(side_row(p, side), template, tokenize, max_len, eos_token)
            enc[side] = ex
            lengths.append(n)
        if enc["chosen"] is None or enc["rejected"] is None:
            dropped.append(d)
        else:
            examples[d] = enc
    lines = [{"pair_id": p["pair_id"], "distinct_id": p["meta"]["distinct_pair_id"], "pair_type": p["pair_type"]}
             for p in pairs if p["meta"]["distinct_pair_id"] in examples]
    return {"examples": examples, "lines": lines, "dropped_distinct_ids": dropped,
            "dropped_lines": len(pairs) - len(lines), "stats": length_stats(lengths, max_len)}


def stack_pairs(pair_examples: list[dict], pad_id: int) -> tuple[dict, int]:
    """One padded batch with all chosen sequences first, then all rejected ones; returns (batch, number of pairs).
    Rows 0..m-1 are the chosen answers, rows m..2m-1 the rejected answers of the same pairs."""
    chosen = [e["chosen"] for e in pair_examples]
    rejected = [e["rejected"] for e in pair_examples]
    return pad_batch(chosen + rejected, pad_id), len(pair_examples)


# ---- reference log-probabilities ------------------------------------------------------------------------------------------
def reference_key(pairs_sha: str, adapter_sha: str, model_id: str, max_len: int) -> dict:
    """Everything the reference numbers depend on. A cache with a different key is never used."""
    return {"pairs_sha256": pairs_sha, "start_adapter_sha256": adapter_sha, "model_id": model_id, "max_seq_len": max_len}


def load_reference_cache(path: Path, key: dict, wanted_ids: set[str]) -> dict[str, list[float]] | None:
    """{distinct_id: [chosen, rejected]} if the cache exists, has this key and covers every wanted id; else None."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("key") != key or not wanted_ids <= set(data.get("values", {})):
        return None
    return data["values"]


def save_reference_cache(path: Path, key: dict, values: dict[str, list[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "n_distinct_pairs": len(values), "values": values}, indent=1), encoding="utf-8")


def reference_for(refs: dict[str, list[float]], line: dict) -> tuple[float, float]:
    """(chosen, rejected) reference log-probs of one line of the file: looked up by the DISTINCT id, so the repeats of a
    type 3 pair use the numbers that were computed once."""
    c, r = refs[line["distinct_id"]]
    return c, r
