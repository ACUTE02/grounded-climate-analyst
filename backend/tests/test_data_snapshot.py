"""The data snapshot and how scripts.setup uses it.

Default: restore the checked-in snapshot (offline, SHA256 verified). `--refetch`: fetch
fresh data from the network and rebuild the models. Setup says which one it used.
"""

import json
import shutil
import socket

import pytest

from forecasting import config, snapshot
from scripts import setup

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def offline(monkeypatch):
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def connect(sock, address, *a, **k):
        if isinstance(address, tuple) and address[0] in LOOPBACK:
            return real_connect(sock, address, *a, **k)
        raise AssertionError(f"tried to connect to {address!r}")

    def getaddrinfo(host, *a, **k):
        if host in LOOPBACK or host is None:
            return real_getaddrinfo(host, *a, **k)
        raise AssertionError(f"tried to resolve {host!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Point every data destination and the source record into tmp_path."""
    monkeypatch.setattr(config, "DATA_RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "ONI_PATH", tmp_path / "models" / "oni_series.parquet")
    monkeypatch.setattr(config, "IOD_PATH", tmp_path / "models" / "iod_series.parquet")
    monkeypatch.setattr(snapshot, "SOURCE_RECORD_PATH", tmp_path / "data" / "source.json")
    return tmp_path


# --------------------------------------------------------------------------- #
# The checked-in snapshot
# --------------------------------------------------------------------------- #
def test_the_snapshot_is_intact():
    assert snapshot.verify() == []


def test_the_manifest_describes_every_file_the_code_needs():
    manifest = snapshot.load_manifest()
    assert {e["file"] for e in manifest["files"]} == set(snapshot.destinations())
    assert manifest["total_bytes"] == sum(e["size_bytes"] for e in manifest["files"])
    for entry in manifest["files"]:
        for key in ("sha256", "size_bytes", "fetch_date", "fetch_date_basis", "source",
                    "attribution", "licence", "transformation", "restores_to"):
            assert entry[key], (entry["file"], key)
        assert entry["source"]["url"].startswith("https://")
        assert (snapshot.SNAPSHOT_DIR / entry["file"]).stat().st_size == entry["size_bytes"]


def test_weather_files_carry_the_open_meteo_attribution_and_licence():
    weather = [e for e in snapshot.load_manifest()["files"]
               if e["kind"].startswith("weather")]
    assert len(weather) == 2 * len(config.REGIONS)
    for entry in weather:
        assert "Weather data by Open-Meteo.com" in entry["attribution"]
        assert entry["licence"]["name"] == "CC BY 4.0"
        assert entry["licence"]["url"] == "https://creativecommons.org/licenses/by/4.0/"
        assert entry["source"]["url"] == "https://archive-api.open-meteo.com/v1/archive"
    monthly = [e for e in weather if e["kind"] == "weather_monthly"]
    assert all("aggregated to monthly" in e["transformation"] for e in monthly)


def test_an_unknown_fetch_source_is_written_as_unknown_not_guessed():
    for entry in snapshot.load_manifest()["files"]:
        if entry["kind"] in ("oni", "iod"):
            assert entry["source"]["which_url_was_used"].startswith("unknown")


# --------------------------------------------------------------------------- #
# Restoring
# --------------------------------------------------------------------------- #
def test_restore_copies_verified_files_offline_and_records_the_source(sandbox, offline):
    done = snapshot.restore(snapshot.WEATHER)
    assert sorted(done) == sorted(n for n, (g, _) in snapshot.destinations().items()
                                  if g == snapshot.WEATHER)
    for name in done:
        target = snapshot.destinations()[name][1]
        assert target.exists() and snapshot.sha256(target) == snapshot.sha256(
            snapshot.SNAPSHOT_DIR / name)
    record = json.loads(snapshot.SOURCE_RECORD_PATH.read_text(encoding="utf-8"))
    assert {v["source"] for v in record["files"].values()} == {"snapshot"}
    assert snapshot.restore(snapshot.ENSO)


def test_a_damaged_snapshot_is_refused_not_used(sandbox, tmp_path):
    copy = tmp_path / "snap"
    shutil.copytree(snapshot.SNAPSHOT_DIR, copy)
    victim = copy / "barmer_raw.parquet"
    victim.write_bytes(victim.read_bytes()[:-5] + b"xxxxx")
    assert snapshot.verify(directory=copy) == ["barmer_raw.parquet: SHA256 does not match "
                                               "the manifest"]
    with pytest.raises(RuntimeError, match="not intact"):
        snapshot.restore(snapshot.WEATHER, directory=copy)
    assert not config.raw_path("barmer").exists(), "nothing is copied from a bad snapshot"


def test_a_missing_snapshot_never_falls_back_to_the_network(sandbox, tmp_path):
    with pytest.raises(FileNotFoundError, match="--refetch"):
        snapshot.load_manifest(tmp_path / "nothing" / "manifest.json")


def test_source_summary_says_where_each_file_came_from(sandbox):
    assert all("missing" in line for line in snapshot.source_summary())
    snapshot.restore(snapshot.WEATHER)
    lines = " ".join(snapshot.source_summary())
    assert "rajasthan_raw.parquet: snapshot" in lines
    assert "oni_series.parquet: missing" in lines


# --------------------------------------------------------------------------- #
# scripts.setup: default and --refetch
# --------------------------------------------------------------------------- #
def _fake_steps(tmp_path, calls, existing=True):
    def make(name, data):
        out = tmp_path / f"{name.replace(' ', '_')}.out"
        if existing:
            out.write_text("x")
        return setup.Step(
            name=name, why="test", outputs=lambda out=out: [out],
            run=lambda name=name: calls.append((name, "default")),
            refetch=(lambda name=name: calls.append((name, "refetch"))) if data else None)

    return [make("weather data", True), make("ENSO series", True),
            make("t+1 Ridge models", False)]


def test_setup_registers_the_snapshot_as_default_and_the_network_as_refetch():
    by_name = {s.name: s for s in setup.STEPS}
    assert by_name["weather data"].run is setup._restore_weather
    assert by_name["weather data"].refetch is setup._refetch_weather
    assert by_name["ENSO series"].run is setup._restore_enso
    assert by_name["ENSO series"].refetch is setup._refetch_enso
    assert by_name["t+1 Ridge models"].refetch is None


def test_setup_default_run_restores_from_the_snapshot_and_says_so(
        monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(setup, "STEPS", _fake_steps(tmp_path, calls, existing=False))
    setup.main(["--skip-index"])
    out = capsys.readouterr().out
    assert [c for c in calls] == [("weather data", "default"), ("ENSO series", "default"),
                                  ("t+1 Ridge models", "default")]
    assert "Data: snapshot backend/data_snapshot/" in out
    assert "data source: snapshot" in out
    assert "fresh network fetch" not in out


def test_setup_leaves_present_steps_alone_by_default(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(setup, "STEPS", _fake_steps(tmp_path, calls, existing=True))
    setup.main(["--skip-index"])
    assert calls == []
    assert capsys.readouterr().out.count("already present, skipping") == 3


def test_setup_refetch_fetches_data_and_rebuilds_the_models(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(setup, "STEPS", _fake_steps(tmp_path, calls, existing=True))
    setup.main(["--refetch", "--skip-index"])
    out = capsys.readouterr().out
    assert calls == [("weather data", "refetch"), ("ENSO series", "refetch"),
                     ("t+1 Ridge models", "default")]
    assert "Data: fresh network fetch (--refetch)" in out
    assert "data source: fresh network fetch" in out


def test_setup_force_without_refetch_still_does_not_touch_the_network(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(setup, "STEPS", _fake_steps(tmp_path, calls, existing=True))
    setup.main(["--force", "--skip-index"])
    assert all(kind == "default" for _, kind in calls) and len(calls) == 3


def test_setup_stops_with_a_plain_message_if_the_snapshot_is_broken(
        monkeypatch, tmp_path, capsys):
    def broken():
        raise RuntimeError("the data snapshot is not intact: x")

    steps = _fake_steps(tmp_path, [], existing=False)
    steps[0].run = broken
    monkeypatch.setattr(setup, "STEPS", steps)
    assert setup.main(["--skip-index"]) == 1
    assert "not intact" in capsys.readouterr().out


def test_the_real_data_steps_do_not_import_the_network_code_by_default():
    import inspect

    for fn in (setup._restore_weather, setup._restore_enso):
        source = inspect.getsource(fn)
        assert "fetch" not in source.replace("snapshot.restore", "")


# --------------------------------------------------------------------------- #
# data_source and the loaders
# --------------------------------------------------------------------------- #
def test_data_source_is_snapshot_after_a_restore_and_refetched_after_a_fetch(sandbox):
    assert snapshot.data_source() == "unknown"
    snapshot.restore(snapshot.WEATHER)
    snapshot.restore(snapshot.ENSO)
    assert snapshot.data_source() == "snapshot"
    victim = config.raw_path("barmer")
    snapshot.record(victim.name, "network", victim)
    assert snapshot.data_source() == "refetched"


def test_a_file_changed_after_it_was_recorded_is_not_called_snapshot(sandbox):
    snapshot.restore(snapshot.WEATHER)
    snapshot.restore(snapshot.ENSO)
    config.ONI_PATH.write_bytes(config.ONI_PATH.read_bytes() + b"x")
    assert snapshot.data_source() == "unknown"


def test_the_loaders_restore_a_missing_file_from_the_snapshot_not_the_network(
        sandbox, offline, monkeypatch):
    import requests

    from forecasting import enso, fetch_data, iod

    def no_network(*a, **k):
        raise AssertionError("a loader went to the network")

    monkeypatch.setattr(requests, "get", no_network)
    assert not config.raw_path("barmer").exists()
    monthly = fetch_data.load_or_fetch_region("barmer")
    daily = fetch_data.load_or_fetch_daily("barmer")
    assert len(monthly) == 540 and len(daily) == 16437
    assert not config.raw_path("rajasthan").exists(), "only the file asked for is restored"
    assert len(enso.fetch_oni()) == 918
    assert len(iod.fetch_iod()) == 1877
    assert snapshot.source_summary()[0].startswith("rajasthan_raw.parquet: missing")


def test_force_is_the_only_way_to_the_network(sandbox, monkeypatch):
    import requests

    from forecasting import enso

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        raise OSError("blocked in test")

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(RuntimeError):
        enso.fetch_oni(force=True)
    assert calls, "force=True should have tried the network"
