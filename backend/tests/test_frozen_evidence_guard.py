"""No code may write the frozen evidence in models/.

The tracked .json, .md and .png files in models/ are saved evidence. Anything that
computes new results writes under models/generated/ instead. This is a static
check: it parses every non-test Python file under backend/ and fails if the target
of a write (write_text, write_bytes, savefig, to_json, open(..., "w")) names a
frozen file, either through a config constant or path function that points into
models/ or through a literal file name.

Limit: a write through a variable that was assigned elsewhere is not followed.
"""

import ast
from pathlib import Path

from forecasting import config

BACKEND = Path(__file__).resolve().parents[1]
EVIDENCE_SUFFIXES = {".json", ".md", ".png"}
SKIP_DIRS = {".venv", "__pycache__", "tests", "chroma_store", "cache", "models", "data"}


def _frozen_names() -> tuple[set[str], set[str]]:
    """(identifiers in config that point at frozen files, frozen file names)."""
    idents: set[str] = set()
    files: set[str] = {p.name for p in config.MODELS_DIR.iterdir()
                       if p.is_file() and p.suffix in EVIDENCE_SUFFIXES}
    for name in dir(config):
        value = getattr(config, name)
        if (isinstance(value, Path) and value.parent == config.MODELS_DIR
                and value.suffix in EVIDENCE_SUFFIXES):
            idents.add(name)
            files.add(value.name)
        elif callable(value) and getattr(value, "__module__", None) == config.__name__:
            try:
                got = value("rajasthan")
            except Exception:
                continue
            if (isinstance(got, Path) and got.parent == config.MODELS_DIR
                    and got.suffix in EVIDENCE_SUFFIXES):
                idents.add(name)
                files.add(got.name.replace("rajasthan", "barmer"))
                files.add(got.name)
    # the three files the horizon work produces, named explicitly
    files |= {"metrics_t1_ridge.json", "horizon_comparison.json",
              "horizon_manifest.json", "iod_comparison.json",
              "rolling_window_check.json"}
    return idents, files


def _write_targets(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in {"write_text", "write_bytes"}:
                yield node, func.value
            elif func.attr in {"savefig", "to_json", "to_csv"} and node.args:
                yield node, node.args[0]
            elif func.attr == "open" and _is_write_mode(node):
                yield node, func.value
        elif isinstance(func, ast.Name) and func.id == "open" and node.args \
                and _is_write_mode(node, offset=1):
            yield node, node.args[0]


def _is_write_mode(call: ast.Call, offset: int = 0) -> bool:
    mode = None
    if len(call.args) > offset and isinstance(call.args[offset], ast.Constant):
        mode = call.args[offset].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    return isinstance(mode, str) and any(c in mode for c in "wax")


def frozen_writes(source: str, idents: set[str], files: set[str]) -> list[str]:
    hits = []
    for call, target in _write_targets(ast.parse(source)):
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name) and sub.id in idents:
                hits.append(f"line {call.lineno}: {sub.id}")
            elif isinstance(sub, ast.Attribute) and sub.attr in idents:
                hits.append(f"line {call.lineno}: {sub.attr}")
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and "generated" not in sub.value \
                    and Path(sub.value.replace("\\", "/")).name in files:
                hits.append(f"line {call.lineno}: {sub.value}")
    return hits


def _source_files():
    for path in sorted(BACKEND.rglob("*.py")):
        if not (set(path.relative_to(BACKEND).parts) & SKIP_DIRS):
            yield path


def test_no_code_writes_the_frozen_evidence():
    idents, files = _frozen_names()
    assert {"T1_METRICS_PATH", "metrics_path", "COMPARISON_PATH"} <= idents
    assert "iod_comparison.json" in files
    problems = []
    scanned = 0
    for path in _source_files():
        scanned += 1
        for hit in frozen_writes(path.read_text(encoding="utf-8"), idents, files):
            problems.append(f"{path.relative_to(BACKEND)} {hit}")
    assert scanned > 30, "the guard is not looking at the source tree"
    assert not problems, "frozen evidence is a write target:\n" + "\n".join(problems)


def test_the_guard_catches_a_write_to_frozen_evidence():
    """Positive control: the checker is not vacuous."""
    idents, files = _frozen_names()
    bad = [
        "config.T1_METRICS_PATH.write_text('x')",
        "config.metrics_path(region).write_text('x')",
        "fig.savefig(config.plot_path(region))",
        "open('models/horizon_manifest.json', 'w')",
        "(MODELS / 'rolling_window_check.json').write_text('x')",
    ]
    good = [
        "config.T1_METRICS_GENERATED_PATH.write_text('x')",
        "config.metrics_generated_path(region).write_text('x')",
        "open('models/generated/horizon_manifest.json', 'w')",
        "config.T1_METRICS_PATH.read_text()",
    ]
    for src in bad:
        assert frozen_writes(src, idents, files), src
    for src in good:
        assert not frozen_writes(src, idents, files), src
