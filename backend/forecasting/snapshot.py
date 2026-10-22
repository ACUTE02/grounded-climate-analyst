"""The data snapshot: the exact input files the published numbers were built from.

backend/data_snapshot/ holds the four weather parquet files, the ONI series and the
IOD (DMI) series, with a manifest.json that records, for each file, where it came from,
when it was fetched, its SHA256 and size, and the attribution it requires. `python -m
scripts.setup` restores these files (offline) by default and only fetches fresh data
from the network when asked with `--refetch`. Each restored or fetched file is noted in
backend/data/source.json, so setup can say which kind of data a model was built from.

Why a snapshot: the weather archive and NOAA's ONI are revised at the source, so a fresh
fetch today does not give the numbers behind the saved evidence (see the Phase 3A report).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from forecasting import config

SNAPSHOT_DIR = config.REPO_ROOT / "data_snapshot"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"
SOURCE_RECORD_PATH = config.DATA_RAW.parent / "source.json"

WEATHER = "weather"
ENSO = "enso"          # the ONI series and the IOD (DMI) series


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def destinations() -> dict[str, tuple[str, Path]]:
    """snapshot file name -> (group, where setup puts it)."""
    out: dict[str, tuple[str, Path]] = {}
    for region in config.REGIONS:
        for path in (config.raw_path(region), config.daily_path(region)):
            out[path.name] = (WEATHER, path)
    out[config.ONI_PATH.name] = (ENSO, config.ONI_PATH)
    out[config.IOD_PATH.name] = (ENSO, config.IOD_PATH)
    return out


def load_manifest(path: Path | None = None) -> dict:
    path = path or MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing, so there is no data snapshot to restore. Run "
            "`python -m scripts.setup --refetch` to fetch the data from the network.")
    return json.loads(path.read_text(encoding="utf-8"))


def verify(manifest: dict | None = None, directory: Path | None = None,
           names: list[str] | None = None) -> list[str]:
    """Problems with the snapshot files (missing, wrong size, wrong SHA256). Empty = ok."""
    manifest = manifest or load_manifest()
    directory = directory or SNAPSHOT_DIR
    problems = []
    for entry in manifest["files"]:
        if names is not None and entry["file"] not in names:
            continue
        path = directory / entry["file"]
        if not path.exists():
            problems.append(f"{entry['file']}: missing")
        elif path.stat().st_size != entry["size_bytes"]:
            problems.append(f"{entry['file']}: size {path.stat().st_size} != "
                            f"{entry['size_bytes']}")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"{entry['file']}: SHA256 does not match the manifest")
    return problems


def _read_record() -> dict:
    if SOURCE_RECORD_PATH.exists():
        return json.loads(SOURCE_RECORD_PATH.read_text(encoding="utf-8"))
    return {"files": {}}


def record(name: str, source: str, path: Path, note: str = "") -> None:
    """Note that `path` now holds data from `source` ("snapshot" or "network")."""
    record_ = _read_record()
    record_["files"][name] = {
        "source": source,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256(path), "note": note}
    SOURCE_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_RECORD_PATH.write_text(json.dumps(record_, indent=2) + "\n", encoding="utf-8")


def restore(group: str, directory: Path | None = None,
            manifest: dict | None = None, names: list[str] | None = None) -> list[str]:
    """Copy the verified snapshot files of one group to where the code reads them.

    `names` limits the copy to some files of the group (the loaders restore only the
    file they are missing, never over another one).

    Refuses (raises) if any file of the group is missing or does not match the manifest;
    it never falls back to the network by itself.
    """
    manifest = manifest or load_manifest()
    directory = directory or SNAPSHOT_DIR
    wanted = [n for n, (g, _) in destinations().items()
              if g == group and (names is None or n in names)]
    in_manifest = {e["file"] for e in manifest["files"]}
    absent = [n for n in wanted if n not in in_manifest]
    if absent:
        raise FileNotFoundError(f"the snapshot manifest has no entry for {absent}")
    problems = verify(manifest, directory, wanted)
    if problems:
        raise RuntimeError("the data snapshot is not intact: " + "; ".join(problems)
                           + ". Re-checkout backend/data_snapshot/, or run "
                           "`python -m scripts.setup --refetch`.")
    done = []
    for name in wanted:
        target = destinations()[name][1]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / name, target)
        record(name, "snapshot", target, note=f"manifest sha256 verified; fetched "
               f"{next(e for e in manifest['files'] if e['file'] == name)['fetch_date']}")
        done.append(name)
    return done


def describe_snapshot(manifest: dict | None = None) -> str:
    manifest = manifest or load_manifest()
    dates = sorted({e["fetch_date"] for e in manifest["files"]})
    return (f"snapshot backend/data_snapshot/ ({len(manifest['files'])} files, SHA256 "
            f"checked; fetched {', '.join(dates)})")


def data_source() -> str:
    """"snapshot" if every data file on disk is the recorded snapshot copy, "refetched" if any
    came from the network, "unknown" if a file is missing or has no matching record."""
    record_ = _read_record()["files"]
    kinds = set()
    for name, (_, path) in destinations().items():
        entry = record_.get(name)
        if not path.exists() or entry is None or entry["sha256"] != sha256(path):
            return "unknown" if "network" not in kinds else "refetched"
        kinds.add(entry["source"])
    if "network" in kinds:
        return "refetched"
    return "snapshot"


def source_summary() -> list[str]:
    """One line per data file: where the copy now on disk came from."""
    record_ = _read_record()["files"]
    lines = []
    for name, (_, path) in destinations().items():
        entry = record_.get(name)
        if not path.exists():
            lines.append(f"{name}: missing")
        elif entry is None:
            lines.append(f"{name}: present, source not recorded (put there before this "
                         "setup wrote a source record)")
        elif entry["sha256"] != sha256(path):
            lines.append(f"{name}: changed after its source was recorded "
                         f"({entry['source']} at {entry['recorded_at']})")
        else:
            lines.append(f"{name}: {entry['source']} ({entry['recorded_at']})")
    return lines
