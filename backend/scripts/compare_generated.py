"""Compare what scripts.setup generated with the frozen evidence. Read-only.

    python -m scripts.compare_generated

For every JSON file that exists in both models/ (frozen, tracked) and
models/generated/, walk the two documents leaf by leaf and print each difference.
Nothing is written. Exit code 0 when every compared leaf is equal, 1 when any differs
or when a file is missing on one side.
"""

from __future__ import annotations

import json
import sys

from forecasting import config

COMPARED = ("metrics_t1_ridge.json", "horizon_comparison.json", "horizon_manifest.json")


def diff(frozen, generated, path: str = "$") -> list[str]:
    if isinstance(frozen, dict) and isinstance(generated, dict):
        out = []
        for key in sorted(set(frozen) | set(generated)):
            if key not in frozen:
                out.append(f"{path}.{key}: only in generated")
            elif key not in generated:
                out.append(f"{path}.{key}: only in frozen")
            else:
                out += diff(frozen[key], generated[key], f"{path}.{key}")
        return out
    if isinstance(frozen, list) and isinstance(generated, list):
        if len(frozen) != len(generated):
            return [f"{path}: list length {len(frozen)} (frozen) vs {len(generated)} (generated)"]
        out = []
        for i, (a, b) in enumerate(zip(frozen, generated)):
            out += diff(a, b, f"{path}[{i}]")
        return out
    if frozen != generated:
        return [f"{path}: frozen {frozen!r} vs generated {generated!r}"]
    return []


def compare(names=COMPARED) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name in names:
        frozen_path = config.MODELS_DIR / name
        generated_path = config.GENERATED_DIR / name
        if not frozen_path.exists() or not generated_path.exists():
            result[name] = [f"missing: frozen {frozen_path.exists()}, "
                            f"generated {generated_path.exists()}"]
            continue
        result[name] = diff(json.loads(frozen_path.read_text(encoding="utf-8")),
                            json.loads(generated_path.read_text(encoding="utf-8")))
    return result


def main() -> int:
    result = compare()
    for name, lines in result.items():
        print(f"{name}: {'identical' if not lines else f'{len(lines)} difference(s)'}")
        for line in lines:
            print(f"  {line}")
    return 0 if not any(result.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
