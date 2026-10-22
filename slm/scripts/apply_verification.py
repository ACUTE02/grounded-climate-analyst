"""A2 F4: merge the independent AI verdicts (new evidence rules), send incomplete quotes back once,
apply fixes, trim the test set, write the review log and print counts."""
import csv
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.build import run_checks
from climate_slm.data.sft import parse_example, set_question_answer
from climate_slm.data.textutil import read_jsonl, write_jsonl
from climate_slm.data.verify import VALID_VERDICTS, coverage_ok, quote_match

NONE = "NONE"


def load_verdicts(verify_dir: Path, pattern: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(verify_dir / pattern))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = d
    return out


def quotes_of(v: dict) -> list[str]:
    q = v.get("evidence_quotes")
    if q is None:
        q = [v["evidence_quote"]] if v.get("evidence_quote") else []
    if isinstance(q, str):
        q = [q]
    return [x.strip() for x in q if x and x.strip() and x.strip().upper() != NONE]


def evaluate(ex: dict, v: dict, cfg: dict, final_answer: str, chunk: str) -> dict:
    """Return the status of one verdict: verdict, reason, quotes, match, fail_reason, needs_recheck."""
    is_ref = ex["meta"]["type"] == "unanswerable"
    verdict = v.get("ai_verdict")
    st = {"verdict": verdict if verdict in VALID_VERDICTS else "fail", "reason": (v.get("ai_reason") or "").strip(),
          "quotes": [], "match": None, "fail_reason": None, "needs_recheck": False, "fake": 0,
          "topic": (v.get("chunk_topic") or "").strip()}
    if verdict not in VALID_VERDICTS:
        st["fail_reason"] = "invalid_verdict_value"
    if is_ref:
        return st
    quotes = quotes_of(v)
    st["quotes"] = quotes
    matches = [quote_match(q, chunk, cfg) for q in quotes]
    st["fake"] = sum(1 for m in matches if m is None)
    if matches and all(matches):
        st["match"] = "fuzzy" if "fuzzy" in matches else "exact"
    if st["verdict"] in ("pass", "fix"):
        if st["fake"]:
            st["verdict"], st["fail_reason"] = "fail", "fake_quote"
            st["reason"] = "FORCED FAIL (fake_quote): a quote is not in the chunk. " + st["reason"]
        elif not quotes or len(quotes) > cfg["verification"]["evidence_max_quotes"]:
            st["needs_recheck"] = True
        else:
            ok, detail = coverage_ok(final_answer, quotes, cfg)
            if not ok:
                st["needs_recheck"] = True
                st["coverage"] = detail
    return st


def apply_set(rows: list[dict], verdicts: dict, rechecks: dict, cfg: dict, name: str, recheck_out: Path) -> dict:
    stats = defaultdict(Counter)
    need = []
    for ex in rows:
        if ex.get("dropped_by") not in (None, "over_target"):
            continue  # dropped rows (cap, leakage, reviewer) keep whatever they have
        ex["verdict_history"] = [h for h in ex.get("verdict_history", []) if h.get("stage") == "A"]
        if ex.get("dropped_by") == "over_target":
            ex["dropped_by"] = None
        for k in ("ai_verdict", "ai_reason", "evidence_quotes", "evidence_match", "fail_reason", "chunk_topic"):
            ex.pop(k, None)
        if ex.get("ai_fix"):
            fx = ex.pop("ai_fix")
            set_question_answer(ex, fx["orig_question"], fx["orig_answer"])
        ex["meta"]["verification"] = "unverified"
    act = [r for r in rows if r["dropped_by"] is None]
    for ex in act:
        chunk, q, a = parse_example(ex)
        is_ref = ex["meta"]["type"] == "unanswerable"
        v = verdicts.get(ex["id"])
        if v is None:
            stats["_all"]["missing_verdict"] += 1
            ex.update(ai_verdict="fail", ai_reason="no verdict returned", evidence_quotes=[], fail_reason="no_verdict")
            continue
        fa = (v.get("fix_answer") or None) if not is_ref else None
        st = evaluate(ex, v, cfg, fa or a, chunk)
        ex["verdict_history"].append({"stage": "A2_round1", "ai_verdict": v.get("ai_verdict"),
                                      "ai_reason": st["reason"], "evidence_quotes": st["quotes"]})
        if st["needs_recheck"]:
            rv = rechecks.get(ex["id"])
            if rv is None:
                need.append((ex, v))
                ex.update(ai_verdict="pending", ai_reason="quote incomplete, sent back once", evidence_quotes=st["quotes"],
                          fail_reason="pending_recheck")
                continue
            fa = (rv.get("fix_answer") or None)
            st2 = evaluate(ex, rv, cfg, fa or a, chunk)
            ex["verdict_history"].append({"stage": "A2_recheck", "ai_verdict": rv.get("ai_verdict"),
                                          "ai_reason": st2["reason"], "evidence_quotes": st2["quotes"]})
            if st2["needs_recheck"] and st2["verdict"] in ("pass", "fix"):
                st2["verdict"], st2["fail_reason"] = "fail", "quote_incomplete"
                st2["reason"] = "FORCED FAIL (quote_incomplete) after second look. " + st2["reason"]
            st, v = st2, rv
        ex["ai_verdict"], ex["ai_reason"] = st["verdict"], st["reason"]
        ex["evidence_quotes"], ex["evidence_match"], ex["fail_reason"] = st["quotes"], st["match"], st["fail_reason"]
        ex["chunk_topic"] = st["topic"]
        if st["verdict"] == "fix":
            fq, fa = v.get("fix_question"), (None if is_ref else v.get("fix_answer"))
            if not fq and not fa:
                ex["ai_verdict"], ex["fail_reason"] = "fail", "fix_without_content"
            else:
                ex["ai_fix"] = {"orig_question": q, "orig_answer": a, "fix_question": fq or None, "fix_answer": fa or None}
                set_question_answer(ex, question=fq or None, answer=fa or None)

    if need:
        items = []
        chunks = {}
        for ex, v in need:
            chunk, q, a = parse_example(ex)
            chunks[ex["meta"]["chunk_id"]] = chunk
            items.append({"id": ex["id"], "chunk_id": ex["meta"]["chunk_id"], "question": q, "answer": a,
                          "note": "quote incomplete: the quotes you gave must together cover EVERY factual claim and number "
                                  "in the answer (1 to 3 exact quotes). Re-read the chunk and answer again."})
        recheck_out.write_text(json.dumps({"chunks": chunks, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")

    run_checks(rows, cfg)
    for ex in act:
        split = ex["meta"]["split"]
        k = "refusal" if ex["meta"]["type"] == "unanswerable" else "answerable"
        key = f"{split}/{k}"
        v = ex.get("ai_verdict")
        s = stats[key]
        s["items"] += 1
        s[v] += 1
        if ex.get("fail_reason") == "fake_quote":
            s["fake_quote"] += 1
        if ex.get("evidence_match") == "fuzzy":
            s["fuzzy_match"] += 1
        if ex.get("fail_reason") == "quote_incomplete":
            s["quote_incomplete"] += 1
        ok = v == "pass" or (v == "fix" and ex.get("ai_fix") and not ex["auto_flags"])
        if v == "fix" and ex.get("ai_fix") and ex["auto_flags"]:
            s["fix_failed_recheck"] += 1
        if v == "pass" and ex["auto_flags"]:
            s["pass_but_auto_flagged"] += 1
        ex["meta"]["verification"] = "llm_verified" if ok else "unverified"
        if ok:
            s["llm_verified"] += 1
    out = {k: dict(v) for k, v in stats.items()}
    out["_pending_recheck"] = len(need)
    return out


def trim_test(rows: list[dict], cfg: dict) -> dict:
    """Keep all valid refusals (up to test_max_refusals) and fill up to test_target_items with valid answerable
    items spread over chunks. The rest of the valid items get dropped_by='over_target'."""
    g = cfg["generator"]
    valid = [r for r in rows if r["dropped_by"] is None and r["meta"]["verification"] == "llm_verified"]
    ref = sorted((r for r in valid if r["meta"]["type"] == "unanswerable"), key=lambda r: r["id"])
    ans = sorted((r for r in valid if r["meta"]["type"] != "unanswerable"), key=lambda r: r["id"])
    rng = random.Random(g["cap_seed_final"])
    by_s = defaultdict(list)
    for r in ref:
        by_s[r["meta"]["unanswerable_strategy"]].append(r)
    for lst in by_s.values():
        rng.shuffle(lst)
    keep_ref = []
    while len(keep_ref) < g["test_max_refusals"] and any(by_s.values()):
        for s in sorted(by_s):
            if by_s[s] and len(keep_ref) < g["test_max_refusals"]:
                keep_ref.append(by_s[s].pop())
    n_ans = max(g["test_target_items"] - len(keep_ref), 0)
    by_c = defaultdict(list)
    for r in ans:
        by_c[r["meta"]["chunk_id"]].append(r)
    order = sorted(by_c)
    rng.shuffle(order)
    keep_ans = []
    for rnd in range(3):
        for c in order:
            if len(by_c[c]) > rnd and len(keep_ans) < n_ans:
                keep_ans.append(by_c[c][rnd])
    keep = {r["id"] for r in keep_ref + keep_ans}
    for r in valid:
        if r["id"] not in keep:
            r["dropped_by"] = "over_target"
    return {"valid_refusals": len(ref), "kept_refusals": len(keep_ref), "valid_answerable": len(ans),
            "kept_answerable": len(keep_ans), "final_items": len(keep)}


def check_min_refusals(valid_refusals: int, cfg: dict) -> None:
    """Fail loudly if the test set has too few valid (AI-verified) refusals."""
    need = cfg["generator"]["test_min_refusals"]
    if valid_refusals < need:
        raise SystemExit(f"FAIL: only {valid_refusals} valid test refusals, need test_min_refusals={need}. "
                         "Redraft test refusals.")


def write_review_log(path: Path, sets: dict[str, list[dict]]) -> None:
    cols = ["set", "id", "split", "source_id", "chunk_id", "kind", "strategy", "partner_chunk_id", "question", "answer",
            "dropped_by", "auto_flags", "ai_verdict", "ai_reason", "evidence_quotes", "evidence_match", "fail_reason",
            "chunk_topic", "ai_fix_orig_question", "ai_fix_orig_answer", "verification", "verdict_history"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for name, rows in sets.items():
            for ex in rows:
                _, q, a = parse_example(ex)
                fx = ex.get("ai_fix") or {}
                m = ex["meta"]
                w.writerow({
                    "set": name, "id": ex["id"], "split": m["split"], "source_id": m["source_id"],
                    "chunk_id": m["chunk_id"], "kind": "refusal" if m["type"] == "unanswerable" else "answerable",
                    "strategy": m.get("unanswerable_strategy", ""), "partner_chunk_id": m.get("partner_chunk_id", ""),
                    "question": q, "answer": a, "dropped_by": ex.get("dropped_by") or "",
                    "auto_flags": ";".join(ex["auto_flags"]), "ai_verdict": ex.get("ai_verdict", ""),
                    "ai_reason": ex.get("ai_reason", ""), "evidence_quotes": " || ".join(ex.get("evidence_quotes") or []),
                    "evidence_match": ex.get("evidence_match") or "", "fail_reason": ex.get("fail_reason") or "",
                    "chunk_topic": ex.get("chunk_topic", ""), "ai_fix_orig_question": fx.get("orig_question", ""),
                    "ai_fix_orig_answer": fx.get("orig_answer", ""), "verification": m["verification"],
                    "verdict_history": json.dumps(ex.get("verdict_history", []), ensure_ascii=False)})


def check_val_minimums(rows: list[dict], cfg: dict) -> dict:
    """Validation minimums (A3 G3). Loud failure if not met."""
    a3 = cfg["a3"]
    fin = [r for r in rows if r["meta"]["split"] == "val" and r["dropped_by"] is None
           and r["meta"]["verification"] == "llm_verified"]
    ref = sum(1 for r in fin if r["meta"]["type"] == "unanswerable")
    return {"val_items": len(fin), "val_refusals": ref, "val_answerable": len(fin) - ref,
            "ok": len(fin) >= a3["val_min_items"] and ref >= a3["val_min_refusals"]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="train,test")
    ap.add_argument("--tag", default="1")
    a = ap.parse_args()
    sets_wanted = a.sets.split(",")
    cfg = load_config()
    vdir = resolve(cfg, "verify_dir")
    result, sets = {}, {}
    for name, key in (("train", "candidates"), ("test", "test_candidates")):
        rows = read_jsonl(resolve(cfg, key))
        sets[name] = rows
        if name not in sets_wanted:
            continue
        verdicts = load_verdicts(vdir, f"verdicts_a2_{name}_*.jsonl")
        rechecks = load_verdicts(vdir, f"verdicts_a2_recheck_{name}_*.jsonl")
        result[name] = apply_set(rows, verdicts, rechecks, cfg, name, vdir / f"input_a2_recheck_{name}_{a.tag}.json")
        result[name]["verdict_lines_read"] = len(verdicts)
        result[name]["recheck_lines_read"] = len(rechecks)
        if name == "test" and not result[name]["_pending_recheck"]:
            result["test_trim"] = trim_test(rows, cfg)
        write_jsonl(resolve(cfg, key), rows)
    write_review_log(resolve(cfg, "review_log"), sets)
    prev = {}
    sp = resolve(cfg, "stage_a_summary")
    if sp.exists():
        prev = json.loads(sp.read_text(encoding="utf-8"))
    key = "verification_a2" if a.tag == "1" else f"verification_a3_{a.tag}"
    prev[key] = result
    sp.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    pending = sum(result[n]["_pending_recheck"] for n in result if n in ("train", "test"))
    if pending:
        print(f"RECHECK NEEDED for {pending} items: see input_a2_recheck_*_{a.tag}.json in {vdir}")
        return
    if "test_trim" in result:
        check_min_refusals(result["test_trim"]["valid_refusals"], cfg)
    if "train" in result:
        vm = check_val_minimums(sets["train"], cfg)
        print("val minimums:", vm)
        if not vm["ok"]:
            print("WARNING: validation minimums not met (val_min_items / val_min_refusals)")


if __name__ == "__main__":
    main()
