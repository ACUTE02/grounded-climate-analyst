"""Zip what Colab needs for evaluation into outputs/colab_bundle.zip (the repo is not on GitHub yet).
What goes in is listed in configs/phase2.yaml (`bundle`). Prints the size and the file list."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_slm.config import REPO_ROOT, load_eval_config


def bundle_files(cfg: dict) -> list[Path]:
    files = []
    for folder in cfg["bundle"]["folders"]:
        for p in sorted((REPO_ROOT / folder["path"]).rglob("*")):
            if p.is_file() and p.suffix in folder["endings"] and "__pycache__" not in p.parts:
                files.append(p)
    for f in cfg["bundle"]["files"]:
        p = REPO_ROOT / f
        if not p.is_file():
            raise SystemExit(f"missing file for the bundle: {f}")
        files.append(p)
    return files


def make_bundle(cfg: dict, out: Path | None = None) -> tuple[Path, list[str]]:
    out = Path(out) if out else REPO_ROOT / cfg["paths"]["colab_bundle"]
    out.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in bundle_files(cfg):
            name = p.relative_to(REPO_ROOT).as_posix()
            z.write(p, name)
            names.append(name)
    return out, names


if __name__ == "__main__":
    path, names = make_bundle(load_eval_config())
    for n in names:
        print(" ", n)
    size = path.stat().st_size
    print(f"{len(names)} files -> {path}  ({size} bytes, {size / 1024:.1f} KB)")
