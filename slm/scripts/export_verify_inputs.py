"""A2 F4: write verifier input shards for every ACTIVE item (train+val and test). The verifier only sees
chunk text, question and answer. Old verdicts are never shown."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import load_config, resolve
from climate_slm.data.sft import parse_example
from climate_slm.data.textutil import read_jsonl


def export(rows, out_dir: Path, name: str, n_shards: int) -> list[Path]:
    rows = sorted(rows, key=lambda r: (r["meta"]["chunk_id"], r["id"]))
    paths = []
    for k, shard in enumerate([rows[i::n_shards] for i in range(n_shards)]):
        chunks, items = {}, []
        for r in shard:
            chunk, q, a = parse_example(r)
            chunks[r["meta"]["chunk_id"]] = chunk
            items.append({"id": r["id"], "chunk_id": r["meta"]["chunk_id"], "question": q, "answer": a})
        p = out_dir / f"input_a2_{name}_{k + 1}.json"
        p.write_text(json.dumps({"chunks": chunks, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
        paths.append(p)
    return paths


def main():
    cfg = load_config()
    out_dir = resolve(cfg, "verify_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [r for r in read_jsonl(resolve(cfg, "candidates")) if r.get("dropped_by") is None]
    test = [r for r in read_jsonl(resolve(cfg, "test_candidates")) if r.get("dropped_by") is None]
    paths = export(train, out_dir, "train", cfg["verification"]["shards_train"]) + export(test, out_dir, "test", 1)
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        print(p.name, len(d["items"]), "items", len(d["chunks"]), "chunks")


if __name__ == "__main__":
    main()
