"""A3 G2: blind-answer check. `export` writes shards (system prompt, context, question only);
`score` reads the blind answers, scores them by code and writes blind_answer / blind_flag into the candidates.
Nothing is dropped."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_config, resolve, system_prompt
from climate_slm.data.blind import score_item
from climate_slm.data.sft import parse_example
from climate_slm.data.textutil import read_jsonl, write_jsonl


def final_rows(cfg):
    out = []
    for key in ("candidates", "test_candidates"):
        for r in read_jsonl(resolve(cfg, key)):
            if r["dropped_by"] is None and r["meta"]["verification"] == "llm_verified":
                out.append((key, r))
    return out


def export(cfg):
    vdir = resolve(cfg, "verify_dir")
    items = []
    for _, r in final_rows(cfg):
        ctx, q, _a = parse_example(r)
        items.append({"id": r["id"], "context": ctx, "question": q})
    items.sort(key=lambda i: i["id"])
    n = cfg["a3"]["blind_shards"]
    for k in range(n):
        shard = items[k::n]
        (vdir / f"input_blind_{k + 1}.json").write_text(json.dumps(
            {"system_prompt": system_prompt(cfg), "items": shard}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"input_blind_{k + 1}.json", len(shard), "items")


def score(cfg):
    vdir = resolve(cfg, "verify_dir")
    answers = {}
    for p in sorted(vdir.glob("blind_out_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                answers[d["id"]] = d["answer"]
    files = {"candidates": read_jsonl(resolve(cfg, "candidates")), "test_candidates": read_jsonl(resolve(cfg, "test_candidates"))}
    stats = defaultdict(Counter)
    flagged, missing = [], 0
    for key, rows in files.items():
        for r in rows:
            r.pop("blind_answer", None)
            r.pop("blind_flag", None)
            r.pop("blind_ok", None)
            if r["dropped_by"] is not None or r["meta"]["verification"] != "llm_verified":
                continue
            if r["id"] not in answers:
                missing += 1
                continue
            ctx, q, ref = parse_example(r)
            is_ref = r["meta"]["type"] == "unanswerable"
            ok, flags = score_item(is_ref, ref, answers[r["id"]], ctx, cfg)
            r["blind_answer"], r["blind_ok"], r["blind_flag"] = answers[r["id"]], ok, ";".join(flags)
            s = stats[f"{r['meta']['split']}/{'refusal' if is_ref else 'answerable'}"]
            s["items"] += 1
            s["blind_ok"] += ok
            for f in flags:
                s[f] += 1
            if flags:
                flagged.append({"id": r["id"], "split": r["meta"]["split"], "source": r["meta"]["source_id"],
                                "kind": "refusal" if is_ref else "answerable", "flags": ";".join(flags),
                                "question": q, "reference_answer": ref, "blind_answer": answers[r["id"]],
                                "chunk_id": r["meta"]["chunk_id"]})
        write_jsonl(resolve(cfg, key), rows)
    rp = REPO_ROOT / cfg["a3"]["blind_report"]
    cols = ["id", "split", "source", "kind", "flags", "question", "reference_answer", "blind_answer", "chunk_id"]
    with open(rp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(flagged)
    out = {k: dict(v) for k, v in sorted(stats.items())}
    tot = sum(v["items"] for v in stats.values())
    ok = sum(v["blind_ok"] for v in stats.values())
    out["_total"] = {"items": tot, "blind_ok": ok, "answers_missing": missing}
    for k, v in out.items():
        if k != "_total":
            v["blind_ok_rate"] = round(v["blind_ok"] / v["items"], 3)
    sp = resolve(cfg, "stage_a_summary")
    prev = json.loads(sp.read_text(encoding="utf-8"))
    prev["blind_check_a3"] = out
    sp.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "score"])
    a = ap.parse_args()
    cfg = load_config()
    export(cfg) if a.step == "export" else score(cfg)
