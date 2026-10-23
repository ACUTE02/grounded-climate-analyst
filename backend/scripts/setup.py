"""Clone -> working local API, in one command.

    python -m scripts.setup

**Why this exists.** Every artifact the running system loads at request time is
regenerable, and therefore git-ignored: the fetched weather data, the fitted
scaler and SPI-3 gamma parameters, the per-horizon Ridge models, and the ChromaDB
index. A fresh clone legitimately has none of them. That is correct — they are
derived data, not source — but it means "clone and run" was previously a
multi-step ritual documented nowhere. This script is that ritual, automated and
verified.

**The non-obvious part, found by actually auditing it.** `forecast_drought_risk()`
loads `scaler_<region>.joblib` and `spi_params_<region>.joblib` at runtime, but
**neither `t1_model` nor `recursive` writes them** — both call
`prepare_dataset(..., save=False)`. The only modules that persist them are
`forecasting.train` and `forecasting.evaluate`, i.e. the *LSTM* path. So the
obvious-looking sequence "fetch, t1_model, recursive" produces a manifest
pointing at models that load, and a tool that then dies on a missing scaler. On
this machine those files existed only as a leftover side effect of Phase-1 LSTM
training — a model that has no skill and no longer serves any forecast.

Step 3 below fixes that directly by persisting exactly those artifacts, so
setting up does not require training a model nobody uses.

Steps are idempotent: anything already present is skipped unless --force.

**Where the data comes from.** By default setup restores the input data from the
checked-in snapshot (backend/data_snapshot/: the weather files, ONI and IOD, SHA256
verified against its manifest) and fetches nothing, so it runs offline and gives the
numbers behind the saved evidence. `--refetch` fetches fresh data from the network
instead (Open-Meteo, NOAA) and rebuilds every model step from it; the result can differ
from the saved evidence because those sources are revised. Setup prints which one was
used and keeps a record in backend/data/source.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from forecasting import config as fconfig
from forecasting import snapshot
from retrieval import config as rconfig


@dataclass
class Step:
    name: str
    why: str
    outputs: Callable[[], list[Path]]
    run: Callable[[], None]
    needs_api_key: bool = False
    costs_quota: str = ""
    refetch: Callable[[], None] | None = None   # data steps: the network version


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def _restore_weather() -> None:
    """Default: the four weather files from the checked-in snapshot (offline)."""
    snapshot.restore(snapshot.WEATHER)


def _restore_enso() -> None:
    """Default: the ONI and IOD series from the checked-in snapshot (offline)."""
    snapshot.restore(snapshot.ENSO)


def _refetch_weather() -> None:
    from forecasting.fetch_data import load_or_fetch_daily, load_or_fetch_region

    for region in fconfig.REGIONS:
        load_or_fetch_region(region, force=True)     # one pull writes monthly and daily
        load_or_fetch_daily(region)
        for path in (fconfig.raw_path(region), fconfig.daily_path(region)):
            snapshot.record(path.name, "network", path,
                            note="fresh fetch from Open-Meteo (--refetch)")


def _refetch_enso() -> None:
    from forecasting.enso import fetch_oni
    from forecasting.iod import fetch_iod

    fetch_oni(force=True)
    fetch_iod(force=True)
    for path in (fconfig.ONI_PATH, fconfig.IOD_PATH):
        snapshot.record(path.name, "network", path,
                        note="fresh fetch from NOAA PSL (--refetch)")


def _persist_runtime_artifacts() -> None:
    """Write the scaler / month_stats / SPI-3 gamma params the tool loads.

    This is the step the obvious sequence misses — see the module docstring.
    It calls the project's own `prepare_dataset` rather than reimplementing the
    fit, so the artifacts are identical to the ones every phase measured with.
    """
    from forecasting.split import prepare_dataset

    for region in fconfig.REGIONS:
        prepare_dataset(region, save=True)


def _train_t1() -> None:
    from forecasting.t1_model import run as t1_run

    t1_run()


def _train_horizons() -> None:
    from forecasting.recursive import run as recursive_run

    recursive_run()


def _build_index() -> None:
    from retrieval.build import main as build_main

    build_main()


STEPS: list[Step] = [
    Step(
        name="weather data",
        why="Open-Meteo monthly + daily files for every region, in data/raw/ "
            "(from the snapshot, or a fresh pull with --refetch).",
        outputs=lambda: [fconfig.raw_path(r) for r in fconfig.REGIONS],
        run=_restore_weather,
        refetch=_refetch_weather,
    ),
    Step(
        name="ENSO series",
        why="NOAA Oceanic Nino Index, the one exogenous feature that survived testing "
            "(and the IOD series, kept for the IOD check). From the snapshot, or a "
            "fresh pull with --refetch.",
        outputs=lambda: [fconfig.ONI_PATH],
        run=_restore_enso,
        refetch=_refetch_enso,
    ),
    Step(
        name="runtime scaler + SPI-3 parameters",
        why=("The artifacts forecast_drought_risk() loads at request time. "
             "Neither t1_model nor recursive writes these — see the module "
             "docstring. Without this step the API starts and then fails on the "
             "first drought question."),
        outputs=lambda: [p for r in fconfig.REGIONS
                         for p in (fconfig.scaler_path(r),
                                   fconfig.spi_params_path(r))],
        run=_persist_runtime_artifacts,
    ),
    Step(
        name="t+1 Ridge models",
        why="The one validated forecast in the project (skill ~+0.21 to +0.26).",
        outputs=lambda: [fconfig.horizon_model_path(r, 1) for r in fconfig.REGIONS],
        run=_train_t1,
    ),
    Step(
        name="t+2 / t+3 models and horizon manifest",
        why="The remaining horizons, plus the manifest the tool reads to decide "
            "which model answers which horizon and with what confidence label.",
        outputs=lambda: [fconfig.manifest_path()]
        + [fconfig.horizon_model_path(r, h)
           for r in fconfig.REGIONS for h in (2, 3)],
        run=_train_horizons,
    ),
    Step(
        name="ChromaDB retrieval index",
        why="Embeds and indexes the source corpus so reports can cite sources.",
        outputs=lambda: [rconfig.CHUNKS_PATH,
                         rconfig.CHROMA_DIR / "chroma.sqlite3"],
        needs_api_key=True,
        costs_quota=("embedding quota (a separate budget from the 20/day "
                     "generate_content cap); the build backs off on 429 and "
                     "resumes from disk if interrupted"),
        run=_build_index,
    ),
]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _has_api_key() -> bool:
    """Defer to the one resolver that knows what a usable key looks like.

    Same fix as /health: a substring search for "API_KEY" in the raw .env text
    counted a commented-out line, an empty value, and OPENAI_API_KEY as a key.
    """
    from retrieval.embed import MissingAPIKey, get_api_key

    try:
        return bool(get_api_key())
    except MissingAPIKey:
        return False


def step_is_done(step: Step) -> bool:
    return all(p.exists() for p in step.outputs())


def verify() -> tuple[bool, list[str]]:
    """Exactly what /health checks, so setup and health cannot disagree."""
    missing: list[str] = []
    for step in STEPS:
        missing += [p.name for p in step.outputs() if not p.exists()]
    return not missing, sorted(set(missing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate everything the local API needs to run.")
    parser.add_argument("--force", action="store_true",
                        help="re-run steps even if their outputs already exist")
    parser.add_argument("--refetch", action="store_true",
                        help="fetch fresh weather, ONI and IOD data from the network "
                             "instead of restoring the checked-in snapshot, and rebuild "
                             "the models from it (results can differ from the saved "
                             "evidence)")
    parser.add_argument("--skip-index", action="store_true",
                        help="skip the ChromaDB build (needs an API key and "
                             "spends embedding quota)")
    parser.add_argument("--check", action="store_true",
                        help="report what is missing and exit, changing nothing")
    args = parser.parse_args(argv)

    if args.check:
        ok, missing = verify()
        print("ready" if ok else f"missing {len(missing)} artifact(s):")
        for name in missing:
            print(f"  - {name}")
        print("data now on disk:")
        for line in snapshot.source_summary():
            print(f"  {line}")
        return 0 if ok else 1

    print(f"Setting up {fconfig.REPO_ROOT}")
    print("Data: " + ("fresh network fetch (--refetch)" if args.refetch
                      else snapshot.describe_snapshot()) + "\n")
    for number, step in enumerate(STEPS, 1):
        if step.name == "ChromaDB retrieval index" and args.skip_index:
            print(f"[{number}/{len(STEPS)}] {step.name}: SKIPPED (--skip-index)")
            continue
        run_it, forced = step.run, args.force
        if args.refetch:
            if step.refetch is not None:
                run_it, forced = step.refetch, True
            elif not step.needs_api_key:
                forced = True        # models must be rebuilt from the new data
        if step_is_done(step) and not forced:
            print(f"[{number}/{len(STEPS)}] {step.name}: already present, skipping")
            continue
        if step.needs_api_key and not _has_api_key():
            print(f"[{number}/{len(STEPS)}] {step.name}: SKIPPED — no API key. "
                  f"Set GEMINI_API_KEY (or put it in .env) and re-run.")
            continue

        print(f"[{number}/{len(STEPS)}] {step.name}")
        print(f"      {step.why}")
        if step.refetch is not None:
            print("      data source: " + ("fresh network fetch" if args.refetch
                                          else snapshot.describe_snapshot()))
        if step.costs_quota:
            print(f"      cost: {step.costs_quota}")
        started = time.time()
        try:
            run_it()
        except Exception as exc:
            print(f"      FAILED after {time.time() - started:.1f}s: "
                  f"{type(exc).__name__}: {exc}")
            return 1
        print(f"      done in {time.time() - started:.1f}s")

    ok, missing = verify()
    print("\nData used:")
    for line in snapshot.source_summary():
        print(f"    {line}")
    print()
    if ok:
        # Setup builds the *fixed* 1980-2024 archive, which is deliberately
        # frozen. Live inputs are a separate, additive step — so tell the user
        # how current their forecasts would be right now rather than letting
        # them discover it from a report anchored two years back.
        try:
            from forecasting.fetch_data import data_currency

            print("Forecast inputs are currently anchored to:")
            for region in fconfig.REGIONS:
                info = data_currency(region)
                print(f"    {region:12s} {info['data_current_through']} "
                      f"({info['months_behind_today']} month(s) behind today)")
            print("\n    Run `python -m scripts.refresh` to bring them up to "
                  "date (free, no API key).")
        except Exception as exc:                 # never block a good setup
            print(f"    (could not read data currency: {type(exc).__name__})")

        print("\nSetup complete. Start the API with:")
        print("    uvicorn api.app:app --reload --port 8000")
        print("Then, in frontend/: npm install && npm run dev, and open http://localhost:5173")
        return 0
    print(f"Setup incomplete — still missing: {', '.join(missing)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
