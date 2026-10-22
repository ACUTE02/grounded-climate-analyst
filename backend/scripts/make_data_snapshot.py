"""Build backend/data_snapshot/ (the files and manifest.json) from existing data files.

    python -m scripts.make_data_snapshot --weather-dir DIR --models-dir DIR

DIR for the weather files holds the four *_raw / *_daily parquet files; DIR for the
models holds oni_series.parquet and iod_series.parquet. The files are copied as they
are (byte for byte) and described in manifest.json. The fetch date is the file's
modification time in the folder it is copied from (the files are written by the fetch);
it is not a date recorded by the fetch itself, and the manifest says so.

One-off script: the snapshot is meant to be built once from the data behind the saved
evidence, then left alone. `python -m scripts.setup` restores it; it does not rebuild it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forecasting import config, fetch_data, snapshot
from forecasting import enso, iod

OPEN_METEO_LICENCE = {
    "name": "CC BY 4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "terms_url": "https://open-meteo.com/en/terms",
    "note": "The free API may only be used for non-commercial purposes. Attribution, the licence link and a "
            "statement of changes are required by CC BY 4.0. See NOTICE.",
}
OPEN_METEO_ATTRIBUTION = "Weather data by Open-Meteo.com (https://open-meteo.com/)"
NOAA_ATTRIBUTION = ("Data provided by the NOAA/OAR/PSL, Boulder, Colorado, USA, from their Web site at "
                    "https://psl.noaa.gov/")
NOAA_LICENCE = {
    "name": "NOAA PSL: public domain unless annotated otherwise",
    "url": "https://psl.noaa.gov/disclaimer/",
    "note": "Terms read on 2026-10-22. The DMI is made from HadISST1.1: see NOTICE.",
}

FETCH_DATE_BASIS = ("UTC date of the modification time of the file in the folder it was copied from; the "
                    "file is written when it is fetched, but this is not a date recorded "
                    "by the fetch itself")


def _frame_facts(path: Path) -> dict:
    frame = pd.read_parquet(path)
    return {"rows": int(len(frame)), "columns": list(frame.columns),
            "first": str(frame.index.min().date()), "last": str(frame.index.max().date())}


def _mtime_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()


def _weather_entry(path: Path, region: str, monthly: bool) -> dict:
    meta = config.REGIONS[region]
    request = {
        "latitude": meta["lat"], "longitude": meta["lon"],
        "start_date": config.FETCH_START, "end_date": config.FETCH_END,
        "daily": fetch_data.DAILY_VARS, "hourly": fetch_data.HOURLY_VARS,
        "timezone": "GMT", "chunk_years": fetch_data.CHUNK_YEARS}
    if monthly:
        change = ("Modified: daily values aggregated to monthly. temp_c, tmax_c and tmin_c "
                  "are the monthly mean of the daily temperature_2m_mean, _max and _min; "
                  "rainfall_mm is the monthly sum of the daily precipitation_sum; "
                  "soil_moisture is the monthly mean of the hourly soil_moisture_0_to_7cm.")
    else:
        change = ("Modified: columns renamed (temperature_2m_mean -> temp_c, _max -> tmax_c, "
                  "_min -> tmin_c, precipitation_sum -> rainfall_mm); daily values are not "
                  "aggregated; soil moisture is not included.")
    return {
        "kind": "weather_monthly" if monthly else "weather_daily",
        "region": region,
        "source": {"name": "Open-Meteo Historical Weather API",
                   "url": fetch_data.ARCHIVE_URL, "request": request},
        "transformation": change,
        "attribution": OPEN_METEO_ATTRIBUTION,
        "licence": OPEN_METEO_LICENCE,
    }


def _series_entry(name: str) -> dict:
    if name == config.ONI_PATH.name:
        return {
            "kind": "oni",
            "source": {"name": "Oceanic Nino Index (ONI), NOAA",
                       "url": enso.PSL_URL, "fallback_url": enso.CPC_URL,
                       "which_url_was_used": "unknown (the code tries the first and falls "
                                             "back to the second; the choice was not recorded)"},
            "transformation": ("Parsed into a monthly series from NOAA's year-per-row ASCII "
                               "grid (or the CPC seasonal table). Values are NOAA's 3-month "
                               "running mean; the series is revised at the source."),
            "attribution": NOAA_ATTRIBUTION, "licence": NOAA_LICENCE}
    return {
        "kind": "iod",
        "source": {"name": "Dipole Mode Index (DMI, HadISST1.1), NOAA PSL",
                   "url": iod.DMI_URL, "fallback_url": iod.DMI_CSV_URL,
                   "which_url_was_used": "unknown (the code tries the first and falls back "
                                         "to the second; the choice was not recorded)"},
        "transformation": "Parsed into a monthly series from NOAA PSL's year-per-row grid.",
        "attribution": NOAA_ATTRIBUTION, "licence": NOAA_LICENCE}


def build(weather_dir: Path, models_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    plan = []
    for region in config.REGIONS:
        plan.append((weather_dir / config.raw_path(region).name,
                     _weather_entry, (region, True)))
        plan.append((weather_dir / config.daily_path(region).name,
                     _weather_entry, (region, False)))
    for target in (config.ONI_PATH, config.IOD_PATH):
        plan.append((models_dir / target.name, None, (target.name,)))

    for source, make, args in plan:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, out_dir / source.name)
        entry = (make(source, *args) if make else _series_entry(*args))
        entry.update({
            "file": source.name,
            "restores_to": str((config.REPO_ROOT / "data" / "raw" / source.name
                                if entry["kind"].startswith("weather")
                                else config.MODELS_DIR / source.name)
                               .relative_to(config.REPO_ROOT.parent)).replace("\\", "/"),
            "sha256": snapshot.sha256(out_dir / source.name),
            "size_bytes": (out_dir / source.name).stat().st_size,
            "fetch_date": _mtime_date(source),
            "fetch_date_basis": FETCH_DATE_BASIS,
            **_frame_facts(out_dir / source.name)})
        entries.append(entry)

    manifest = {
        "about": ("The input data behind the saved evidence, copied byte for byte. "
                  "`python -m scripts.setup` restores these files; `--refetch` fetches "
                  "fresh data from the network instead."),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_bytes": sum(e["size_bytes"] for e in entries),
        "attribution": [OPEN_METEO_ATTRIBUTION, NOAA_ATTRIBUTION],
        "files": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weather-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=snapshot.SNAPSHOT_DIR)
    args = parser.parse_args(argv)
    manifest = build(args.weather_dir, args.models_dir, args.out)
    print(f"wrote {len(manifest['files'])} files and manifest.json to {args.out} "
          f"({manifest['total_bytes']} bytes of data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
