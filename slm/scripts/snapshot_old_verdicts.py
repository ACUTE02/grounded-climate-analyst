"""A2: save the Stage A verdicts before the candidate files are overwritten (used for verdict_history
and for the 'AI-passed first' rule of the per-source cap). Run once, before build_candidates.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_config, resolve
from climate_slm.data.sft import parse_example
from climate_slm.data.textutil import read_jsonl, write_jsonl

OUT = REPO_ROOT / "data" / "instruction" / "verdict_history_stageA.jsonl"


def main():
    if OUT.exists():
        print("snapshot already exists, not overwriting:", OUT)
        return
    cfg = load_config()
    rows = []
    for name, key in (("train", "candidates"), ("test", "test_candidates")):
        for ex in read_jsonl(resolve(cfg, key)):
            _, q, a = parse_example(ex)
            fx = ex.get("ai_fix") or {}
            rows.append({
                "stage": "A", "set": name, "old_id": ex["id"], "chunk_id": ex["meta"]["chunk_id"],
                "question_chunk_id": ex["meta"].get("question_chunk_id"),
                "type": "refusal" if ex["meta"]["type"] == "unanswerable" else "answerable",
                "question": fx.get("orig_question", q), "answer": fx.get("orig_answer", a),
                "final_question": q, "final_answer": a,
                "ai_verdict": ex["ai_verdict"], "ai_reason": ex["ai_reason"],
                "evidence_quote": ex["evidence_quote"]})
    write_jsonl(OUT, rows)
    print("saved", len(rows), "old verdicts to", OUT)


if __name__ == "__main__":
    main()
