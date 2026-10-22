"""Cell P5-1: are the files and packages for the parity check and the benchmark here? Nothing is run on the GPU.

    python scripts/phase5_check.py

Checks, for the three models of configs/phase5.yaml: the adapter folder (adapter_config.json and the weights), the saved val file of the
evaluator run it is compared with, the val data file; prints the GPU (nvidia-smi), the versions of vllm / torch / transformers / peft /
httpx / fastapi / uvicorn / rouge_score / torchao, and whether the hf backend can run here (transformers and peft importable). If
transformers or peft is missing in the vLLM runtime the hf settings are reported as 'not measured' by the benchmark; they are never made up.
Two runtime problems seen on Colab are checked: rouge_score missing (the evaluator's scoring code imports it) and torchao older than
TORCHAO_MIN (peft refuses it, so the hf backend does not start); the P5-1 cell installs the first and removes the second."""
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_serving import hf_available
from climate_slm.config import REPO_ROOT, load_eval_config, load_phase5_config
from serving.client import nvidia_smi_query

PACKAGES = ("vllm", "torch", "transformers", "peft", "httpx", "fastapi", "uvicorn", "pydantic", "rouge_score", "torchao", "mlflow")
TORCHAO_MIN = (0, 16)                    # peft 0.21.2 needs torchao above 0.16 if torchao is installed at all


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def version_tuple(v: str) -> tuple:
    """'0.10.0' -> (0, 10, 0); a part that is not a number ('0.16.0.dev1') ends the tuple."""
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits or digits != part:
            break
        out.append(int(digits))
    return tuple(out)


def package_problems(v: dict) -> list[str]:
    """Problems with the installed packages (v = package_versions())."""
    problems = []
    if not v.get("vllm"):
        problems.append("vllm is not installed in this runtime (cell P5-0 installs it)")
    if not v.get("httpx"):
        problems.append("httpx is missing (pip install httpx)")
    if not v.get("rouge_score"):
        problems.append("rouge_score is missing: the evaluator's scoring code needs it (pip install rouge_score)")
    ta = v.get("torchao")
    if ta and version_tuple(ta) < TORCHAO_MIN:
        problems.append(f"torchao {ta} is older than {'.'.join(map(str, TORCHAO_MIN))}: peft will not start, so the hf backend fails "
                        "(pip uninstall -y torchao)")
    return problems


def check_inputs(cfg5: dict, cfg_eval: dict, root: Path = REPO_ROOT) -> list[str]:
    """Problems with the files (empty list = ready)."""
    problems = []
    for key, m in cfg5["models"].items():
        if m.get("adapter"):                    # a local adapter directory; a Hugging Face adapter_id is downloaded at load time
            d = root / m["adapter"]
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                if not (d / f).exists():
                    problems.append(f"{key}: {d / f} is missing (unzip phase5_start.zip into outputs/)")
        ev = root / cfg5["paths"]["outputs_dir"] / f"eval_{m['parity_eval']}_val.json"
        if not ev.exists():
            problems.append(f"{key}: the saved val file {ev.name} is missing (it is in phase5_start.zip)")
    if not (root / cfg_eval["paths"]["splits"]["val"]).exists():
        problems.append("data/instruction/sft_val.jsonl is missing (the bundle)")
    return problems


def main(argv=None) -> int:
    cfg5, cfg_eval = load_phase5_config(), load_eval_config()
    problems = check_inputs(cfg5, cfg_eval)
    gpu = nvidia_smi_query()
    v = package_versions()
    print("GPU:", f"{gpu[0]['name']} ({gpu[0]['total_mib']} MiB, {gpu[0]['used_mib']} MiB in use)" if gpu else "none found (nvidia-smi)")
    print("versions:", v)
    avail, info = hf_available()
    print("hf backend can run here:", "yes (" + info + ")" if avail else "NO: " + info + " -> the hf settings will be reported as not measured")
    problems += package_problems(v)
    for p in problems:
        print("PROBLEM:", p)
    print("READY" if not problems else f"NOT READY ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
