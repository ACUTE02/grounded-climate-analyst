"""Zip result files of the Colab runs for download.

    python -m climate_slm.evaluation.collect --out /content/eval_results_2c.zip                    # Phase 2c (default)
    python -m climate_slm.evaluation.collect --set phase3-results --out /content/phase3_results.zip
    python -m climate_slm.evaluation.collect --set phase3-adapters --out /content/phase3_adapters.zip
    python -m climate_slm.evaluation.collect --set phase4-results --out /content/phase4_results.zip
    python -m climate_slm.evaluation.collect --set phase4-adapters --out /content/phase4_adapters.zip
    python -m climate_slm.evaluation.collect --set phase4b-results --out /content/phase4b_results.zip
    python -m climate_slm.evaluation.collect --set phase4b-adapters --out /content/phase4b_adapters.zip
    python -m climate_slm.evaluation.collect --set phase5-results --out /content/phase5_results.zip
    python -m climate_slm.evaluation.collect --set phase5-start --out outputs/phase5_start.zip    # on the laptop, to upload
    python -m climate_slm.evaluation.collect --set phase4-start --out outputs/phase4_start.zip    # on the laptop, to upload

Phase 2c: only the files that match `collect.patterns` in configs/phase2.yaml go in (flat, no folders). The default
base run files are NOT in it, so unzipping can never overwrite the first measurement.
Phase 3: tables, eval files of the sweep runs, run logs and the MLflow store go in one zip (small); the adapters go
in a second zip (large). Paths inside the zip are relative to outputs/, so unzip both into outputs/ on the laptop."""
import argparse
import zipfile
from pathlib import Path

from climate_slm.config import REPO_ROOT, load_eval_config, load_phase4_config, load_phase5_config


def result_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    found: set[Path] = set()
    for pattern in cfg["collect"]["patterns"]:
        found.update(p for p in Path(outputs_dir).glob(pattern) if p.is_file())
    return sorted(found)


def phase3_result_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    """Tables, selection files, sweep eval files and review CSVs, run logs, MLflow store. Not the adapters."""
    out = Path(outputs_dir)
    found: set[Path] = set()
    for pattern in cfg["collect"]["phase3_patterns"]:
        found.update(p for p in out.glob(pattern) if p.is_file())
    for d in cfg["collect"]["phase3_dirs"]:
        found.update(p for p in (out / d).rglob("*") if p.is_file())
    return sorted(found)


def phase3_adapter_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    return sorted(p for p in (Path(outputs_dir) / cfg["collect"]["phase3_adapter_dir"]).rglob("*") if p.is_file())


def phase4_result_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    """DPO tables, selection files, DPO eval files and review CSVs, run logs (with the reference log-probs), MLflow store."""
    out = Path(outputs_dir)
    found: set[Path] = set()
    for pattern in cfg["collect"]["phase4_patterns"]:
        found.update(p for p in out.glob(pattern) if p.is_file())
    for d in cfg["collect"]["phase4_dirs"]:
        found.update(p for p in (out / d).rglob("*") if p.is_file())
    return sorted(found)


def phase4_adapter_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    return sorted(p for p in (Path(outputs_dir) / cfg["collect"]["phase4_adapter_dir"]).rglob("*") if p.is_file())


def phase4b_result_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    """DPO v2 tables, selection files, eval files and review CSVs, run logs (with the reference log-probs), MLflow store."""
    out = Path(outputs_dir)
    found: set[Path] = set()
    for pattern in cfg["collect"]["phase4b_patterns"]:
        found.update(p for p in out.glob(pattern) if p.is_file())
    for d in cfg["collect"]["phase4b_dirs"]:
        found.update(p for p in (out / d).rglob("*") if p.is_file())
    return sorted(found)


def phase4b_adapter_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    return sorted(p for p in (Path(outputs_dir) / cfg["collect"]["phase4b_adapter_dir"]).rglob("*") if p.is_file())


def phase5_result_files(outputs_dir: Path, cfg: dict) -> list[Path]:
    """Phase 5 tables, raw parity / benchmark results and server logs, evaluator-format files of the live service, MLflow store."""
    out = Path(outputs_dir)
    found: set[Path] = set()
    for pattern in cfg["collect"]["phase5_patterns"]:
        found.update(p for p in out.glob(pattern) if p.is_file())
    for d in cfg["collect"]["phase5_dirs"]:
        found.update(p for p in (out / d).rglob("*") if p.is_file())
    return sorted(found)


def phase5_start_files(outputs_dir: Path, cfg5: dict) -> list[Path]:
    """What Colab needs for the parity check and the benchmark: the adapters of the three served models and the saved val evaluator
    files they are compared with. Made on the laptop, uploaded to Colab, unzipped into outputs/."""
    out, files = Path(outputs_dir), []
    for m in cfg5["models"].values():
        if m.get("adapter"):                    # a Hugging Face adapter_id has no local files to bundle
            files += [p for p in (REPO_ROOT / m["adapter"]).rglob("*") if p.is_file()]
        files.append(out / f"eval_{m['parity_eval']}_val.json")
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"missing files for the phase 5 start zip: {missing}")
    return sorted(set(files))


def phase4_start_files(outputs_dir: Path, cfg4: dict) -> list[Path]:
    """What Colab needs to start Phase 4: the three SFT start adapters (epoch 2 folders) and the saved val evaluations of
    their SFT controls (+ the base val file). Made on the laptop, uploaded to Colab, unzipped into outputs/."""
    out, files = Path(outputs_dir), []
    for st in cfg4["start"]["seeds"].values():
        files += [p for p in (REPO_ROOT / st["adapter"]).rglob("*") if p.is_file()]
        files.append(out / f"eval_{st['control']}_val.json")
    files += [out / n for n in cfg4["start"]["extra_eval_files"]]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"missing files for the start zip: {missing}")
    return sorted(set(files))


def make_zip(files: list[Path], out: Path, root: Path | None = None) -> Path:
    """Zip the files. With `root`, the paths inside the zip are relative to it; otherwise only the file names."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(root).as_posix() if root else p.name)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zip result files for download.")
    ap.add_argument("--outputs", default=None, help="folder with the eval files (default: outputs/)")
    ap.add_argument("--set", choices=["phase2c", "phase3-results", "phase3-adapters", "phase4-results", "phase4-adapters", "phase4b-results", "phase4b-adapters",
                             "phase4-start", "phase5-results", "phase5-start"], default="phase2c")
    ap.add_argument("--out", required=True, help="zip file to write")
    a = ap.parse_args(argv)
    cfg = load_eval_config()
    out_dir = Path(a.outputs) if a.outputs else REPO_ROOT / cfg["paths"]["outputs_dir"]
    if a.set == "phase2c":
        files, root = result_files(out_dir, cfg), None
    elif a.set == "phase3-results":
        files, root = phase3_result_files(out_dir, cfg), out_dir
    elif a.set == "phase3-adapters":
        files, root = phase3_adapter_files(out_dir, cfg), out_dir
    elif a.set == "phase4-results":
        files, root = phase4_result_files(out_dir, cfg), out_dir
    elif a.set == "phase4-adapters":
        files, root = phase4_adapter_files(out_dir, cfg), out_dir
    elif a.set == "phase4b-results":
        files, root = phase4b_result_files(out_dir, cfg), out_dir
    elif a.set == "phase4b-adapters":
        files, root = phase4b_adapter_files(out_dir, cfg), out_dir
    elif a.set == "phase5-results":
        files, root = phase5_result_files(out_dir, cfg), out_dir
    elif a.set == "phase5-start":
        files, root = phase5_start_files(out_dir, load_phase5_config()), out_dir
    else:
        files, root = phase4_start_files(out_dir, load_phase4_config()), out_dir
    if not files:
        raise SystemExit(f"No {a.set} result files in {out_dir}. Run the cells before this one first.")
    make_zip(files, Path(a.out), root)
    total = sum(p.stat().st_size for p in files) / 2 ** 20
    for p in files[:40]:
        print(" ", p.relative_to(out_dir).as_posix())
    if len(files) > 40:
        print(f"  ... and {len(files) - 40} more")
    print(f"{len(files)} files ({total:.1f} MB before compression) -> {a.out}")


if __name__ == "__main__":
    main()
